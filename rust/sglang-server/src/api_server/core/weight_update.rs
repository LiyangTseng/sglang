//! Pause, continue and weight updates shared by the HTTP and gRPC adapters:
//! the Rust frontend's port of the TokenizerManager's `is_pause` gate and
//! `model_update_lock`.

use std::sync::Arc;

use sglang_api_types::api::v1 as api;
use tokio::sync::{Mutex, OwnedRwLockReadGuard, RwLock, watch};

use super::{CoreError, CoreHandle};
use crate::message::ids::Rid;
use crate::message::io_struct::{
    ContinueGenerationReqInput, ControlRequest, PauseGenerationReqInput, PauseMode,
};

/// The pause gate and the generation leases it admits.
pub(crate) struct GenerationGate {
    paused: watch::Sender<bool>,
    /// Tokio's RwLock is fair and write-preferring, like Python's
    /// `RWLock`: a queued writer holds back new readers.
    leases: Arc<RwLock<()>>,
}

/// A generation's hold on the current weights, from admission until its
/// response ends.
pub(crate) struct GenerationLease {
    _guard: OwnedRwLockReadGuard<()>,
}

impl GenerationGate {
    pub(super) fn new() -> Self {
        Self {
            paused: watch::Sender::new(false),
            leases: Arc::new(RwLock::new(())),
        }
    }

    /// Wait until generation is not paused, then take a lease. A pause that
    /// closes the gate while this waits for the lock sends it back to wait.
    pub(super) async fn lease(&self) -> GenerationLease {
        let mut paused = self.paused.subscribe();
        loop {
            // The sender lives in `self`, so this cannot fail.
            let _ = paused.wait_for(|paused| !paused).await.map(drop);
            let guard = self.leases.clone().read_owned().await;
            if !self.is_paused() {
                return GenerationLease { _guard: guard };
            }
        }
    }

    pub(super) fn is_paused(&self) -> bool {
        *self.paused.borrow()
    }

    fn close(&self) {
        self.paused.send_replace(true);
    }

    fn open(&self) {
        self.paused.send_replace(false);
    }

    /// Return once every lease taken before this call has been released.
    async fn drain(&self) {
        drop(self.leases.write().await);
    }
}

/// Carries out pause and continue for both adapters, one control request at
/// a time.
pub(crate) struct WeightUpdateManager {
    core: CoreHandle,
    /// Keeps the gate from changing under a pause or continue
    /// that is still in flight.
    control_lock: Mutex<()>,
}

impl WeightUpdateManager {
    pub(crate) fn new(core: CoreHandle) -> Arc<Self> {
        Arc::new(Self {
            core,
            control_lock: Mutex::new(()),
        })
    }

    /// Python's `pause_generation`. Runs to completion even if the caller
    /// goes away, as Python's handler does, so the gate never stays half-set.
    pub(crate) async fn pause(
        self: &Arc<Self>,
        mode: PauseMode,
    ) -> Result<api::GenerationControlResponse, CoreError> {
        let this = self.clone();
        detached(async move { this.pause_locked(mode).await }).await?;
        Ok(api::GenerationControlResponse {
            message: "Generation paused successfully.".into(),
            status: "ok".into(),
        })
    }

    /// Python's `continue_generation`; detached like [`Self::pause`].
    pub(crate) async fn continue_generation(
        self: &Arc<Self>,
        torch_empty_cache: bool,
    ) -> Result<api::GenerationControlResponse, CoreError> {
        let this = self.clone();
        detached(async move { this.continue_locked(torch_empty_cache).await }).await?;
        Ok(api::GenerationControlResponse {
            message: "Generation continued successfully.".into(),
            status: "ok".into(),
        })
    }

    async fn pause_locked(&self, mode: PauseMode) -> Result<(), CoreError> {
        let _control = self.control_lock.lock().await;
        let gate = self.core.gate();
        gate.close();
        match mode {
            // The scheduler is not paused: aborting everything and waiting
            // for the leases to drain leaves it idle.
            PauseMode::Abort => {
                self.core.abort(None);
                gate.drain().await;
                Ok(())
            }
            PauseMode::InPlace | PauseMode::Retract => {
                self.core
                    .send_unanswered(ControlRequest::PauseGenerationReqInput(
                        PauseGenerationReqInput::new(Rid::new().to_string(), mode),
                    ))
                    .await
            }
        }
    }

    async fn continue_locked(&self, torch_empty_cache: bool) -> Result<(), CoreError> {
        let _control = self.control_lock.lock().await;
        // The scheduler resumes before waiting requests are admitted.
        self.core
            .send_unanswered(ControlRequest::ContinueGenerationReqInput(
                ContinueGenerationReqInput::new(Rid::new().to_string(), torch_empty_cache),
            ))
            .await?;
        self.core.gate().open();
        Ok(())
    }
}

async fn detached<T: Send + 'static>(
    operation: impl Future<Output = Result<T, CoreError>> + Send + 'static,
) -> Result<T, CoreError> {
    tokio::spawn(operation)
        .await
        .map_err(|error| CoreError::Internal(format!("control task failed: {error}")))?
}

#[cfg(test)]
#[path = "weight_update_tests.rs"]
mod tests;
