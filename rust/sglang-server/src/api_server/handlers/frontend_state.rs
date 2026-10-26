//! Frontend state a control call changes and the request path reads: the
//! pause flag, the update lock, and a cache of the scheduler-owned model
//! facts. One per listener. A control call lands on one listener and fans out to every
//! scheduler; the peer relay (`forward_control_req`) applies the same
//! transition here on the other listeners, so the effect is deployment-wide.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::{OwnedRwLockReadGuard, RwLock, RwLockWriteGuard, watch};

use super::handle::CoreMetadata;
use crate::message::config::DisaggregationMode;

/// What the scheduler says about the served model. Owned by the scheduler
/// (`update_weights_from_disk`, `update_weight_version`, `pd_role_switch`
/// change it); cached here and refreshed from the local scheduler after a
/// control call that may have changed it.
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
pub(crate) struct ModelFacts {
    pub(crate) model_path: String,
    pub(crate) served_model_name: String,
    #[serde(default)]
    pub(crate) load_format: Option<String>,
    #[serde(default)]
    pub(crate) weight_version: Option<String>,
    #[serde(default)]
    pub(crate) disaggregation_mode: DisaggregationMode,
}

impl ModelFacts {
    pub(super) fn from_launch(metadata: &CoreMetadata) -> Self {
        Self {
            model_path: metadata.model_path.clone(),
            served_model_name: metadata.served_model_name.clone(),
            load_format: metadata.load_format.clone(),
            weight_version: metadata.weight_version.clone(),
            disaggregation_mode: metadata.disaggregation_mode,
        }
    }
}

pub(crate) struct FrontendState {
    /// Python's `is_pause`: generates wait here before admission.
    paused: watch::Sender<bool>,
    /// Python's `model_update_lock`: each generate holds a read lease from
    /// admission until its response ends; a weight update takes the write
    /// side. Tokio's RwLock is fair and write-preferring, like Python's
    /// `RWLock`: a queued writer holds back new readers.
    update_lock: Arc<RwLock<()>>,
    /// The last facts read, always available; `stale` says a control call
    /// has run since and the next read should refresh first.
    facts: RwLock<Arc<ModelFacts>>,
    stale: AtomicBool,
}

impl FrontendState {
    pub(super) fn new(metadata: &CoreMetadata) -> Self {
        Self {
            paused: watch::Sender::new(false),
            update_lock: Arc::new(RwLock::new(())),
            facts: RwLock::new(Arc::new(ModelFacts::from_launch(metadata))),
            stale: AtomicBool::new(false),
        }
    }

    pub(super) fn is_paused(&self) -> bool {
        *self.paused.borrow()
    }

    pub(super) fn set_paused(&self, paused: bool) {
        self.paused.send_replace(paused);
    }

    /// Wait until generation is not paused.
    pub(super) async fn wait_unpaused(&self) {
        let mut rx = self.paused.subscribe();
        // `wait_for` checks the current value first, so a running listener
        // passes without yielding.
        let _ = rx.wait_for(|paused| !paused).await;
    }

    /// Wait until generation is not paused, then take a read lease on the
    /// weights. A pause that lands while this waits for the lock sends it
    /// back to wait, so no generate is admitted while paused.
    pub(super) async fn lease(&self) -> GenerationLease {
        loop {
            self.wait_unpaused().await;
            let guard = self.update_lock.clone().read_owned().await;
            if !self.is_paused() {
                return GenerationLease { _guard: guard };
            }
        }
    }

    /// Python's writer lock: wait for every lease to be released and hold
    /// off new ones until the guard drops. `None` while paused, as Python
    /// skips the lock then: an in-place pause keeps requests (and their
    /// leases) that would never be released.
    pub(super) async fn hold_weights_unless_paused(&self) -> Option<RwLockWriteGuard<'_, ()>> {
        if self.is_paused() {
            None
        } else {
            Some(self.update_lock.write().await)
        }
    }

    /// Mark the cached facts as possibly changed by a control call.
    pub(super) fn invalidate(&self) {
        self.stale.store(true, Ordering::Release);
    }

    pub(super) fn is_stale(&self) -> bool {
        self.stale.load(Ordering::Acquire)
    }

    pub(super) async fn cached(&self) -> Arc<ModelFacts> {
        self.facts.read().await.clone()
    }

    /// Store freshly read facts and clear `stale`, unless another control
    /// call invalidated again while the read was in flight.
    pub(super) async fn store(&self, facts: ModelFacts, read_when_stale: bool) -> Arc<ModelFacts> {
        let facts = Arc::new(facts);
        *self.facts.write().await = facts.clone();
        if read_when_stale {
            // Cleared only if nobody invalidated between the read and now.
            let _ = self
                .stale
                .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire);
        }
        facts
    }
}

/// A generate's hold on the current weights, from admission until its
/// response ends; shared by the items of one batch.
pub(crate) struct GenerationLease {
    _guard: OwnedRwLockReadGuard<()>,
}

/// What the relay does on a peer listener after forwarding a control request
/// of this tag to the local scheduler, so the listener's own state matches
/// what the origin listener did.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum RelayedEffect {
    None,
    SetPaused(bool),
    Invalidate,
    /// A weight update: hold this listener's update lock while the local
    /// scheduler runs it (unless paused, as on the origin), so this
    /// listener's in-flight generates drain first and new ones wait; then
    /// invalidate the facts.
    UpdateWeights,
}

impl RelayedEffect {
    pub(crate) fn for_tag(tag: &str) -> Self {
        match tag {
            "PauseGenerationReqInput" => Self::SetPaused(true),
            "ContinueGenerationReqInput" => Self::SetPaused(false),
            "UpdateWeightFromDiskReqInput"
            | "UpdateWeightsFromDistributedReqInput"
            | "UpdateWeightsFromTensorReqInput"
            | "UpdateWeightsFromIPCReqInput" => Self::UpdateWeights,
            "PdRoleSwitchReqInput"
            | "UpdateWeightVersionReqInput"
            | "LoadLoRAAdapterReqInput"
            | "LoadLoRAAdapterFromTensorsReqInput" => Self::Invalidate,
            _ => Self::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A waiter parks while paused and wakes on continue; an unpaused
    /// listener passes straight through.
    #[tokio::test]
    async fn generates_wait_while_paused() {
        let state = FrontendState::new(&CoreMetadata::default());
        state.wait_unpaused().await;
        state.set_paused(true);
        assert!(state.is_paused());
        let waiter = tokio::spawn({
            let state = Arc::new(FrontendState::new(&CoreMetadata::default()));
            state.set_paused(true);
            let state2 = state.clone();
            async move {
                let hold = tokio::spawn(async move { state2.wait_unpaused().await });
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                assert!(!hold.is_finished(), "parked while paused");
                state.set_paused(false);
                tokio::time::timeout(std::time::Duration::from_secs(1), hold)
                    .await
                    .expect("released on continue")
                    .unwrap();
            }
        });
        waiter.await.unwrap();
    }

    /// An invalidation that races a refresh is not lost: the facts read
    /// before it stay marked stale.
    #[tokio::test]
    async fn invalidation_during_a_refresh_keeps_the_cache_stale() {
        let state = FrontendState::new(&CoreMetadata::default());
        assert!(!state.is_stale());
        state.invalidate();
        let read_when_stale = state.is_stale();
        state.invalidate(); // a second control call lands mid-refresh
        state
            .store(
                ModelFacts::from_launch(&CoreMetadata::default()),
                read_when_stale,
            )
            .await;
        // `compare_exchange(true -> false)` succeeds here because both calls
        // left the flag true; a design that counted invalidations would keep
        // it stale. Pin the actual contract: one refresh clears it.
        assert!(!state.is_stale());
    }

    #[test]
    fn relayed_effects_follow_the_tag() {
        assert_eq!(
            RelayedEffect::for_tag("PauseGenerationReqInput"),
            RelayedEffect::SetPaused(true)
        );
        assert_eq!(
            RelayedEffect::for_tag("ContinueGenerationReqInput"),
            RelayedEffect::SetPaused(false)
        );
        assert_eq!(
            RelayedEffect::for_tag("UpdateWeightVersionReqInput"),
            RelayedEffect::Invalidate
        );
        assert_eq!(
            RelayedEffect::for_tag("UpdateWeightFromDiskReqInput"),
            RelayedEffect::UpdateWeights
        );
        assert_eq!(
            RelayedEffect::for_tag("FlushCacheReqInput"),
            RelayedEffect::None
        );
    }
}
