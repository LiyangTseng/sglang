//! Pause, continue and weight updates shared by the HTTP and gRPC adapters:
//! the Rust frontend's port of the TokenizerManager's `is_pause` gate and
//! `model_update_lock`.

use std::sync::Arc;

use sglang_api_types::api::v1 as api;
use tokio::sync::{Mutex, OwnedRwLockReadGuard, RwLock, RwLockWriteGuard, watch};

use super::{CoreError, CoreHandle};
use crate::message::ids::Rid;
use crate::message::io_struct::{
    ContinueGenerationReqInput, ControlRequest, PauseGenerationReqInput, PauseMode,
    UpdateWeightFromDiskReqInput, UpdateWeightVersionReqInput,
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
        drop(self.hold_weights().await);
    }

    /// Python's writer lock: wait for every lease to be released and hold
    /// off new ones until the guard drops.
    async fn hold_weights(&self) -> RwLockWriteGuard<'_, ()> {
        self.leases.write().await
    }
}

/// An `update_weights_from_disk` body that passed Python's request validation.
pub(crate) struct DiskUpdate {
    body: api::UpdateWeightsFromDiskRequest,
    model_path: String,
    manifest: Option<serde_json::Map<String, serde_json::Value>>,
}

impl TryFrom<api::UpdateWeightsFromDiskRequest> for DiskUpdate {
    type Error = String;

    fn try_from(mut body: api::UpdateWeightsFromDiskRequest) -> Result<Self, Self::Error> {
        let model_path = body.model_path.take().ok_or("missing field `model_path`")?;
        let manifest = match body.manifest.take() {
            None => None,
            Some(text) => match serde_json::from_str(&text) {
                Ok(serde_json::Value::Object(manifest)) => Some(manifest),
                Ok(serde_json::Value::Null) => None,
                _ => return Err("`manifest` must be a JSON object".into()),
            },
        };
        Ok(Self {
            body,
            model_path,
            manifest,
        })
    }
}

/// An `update_weight_version` body that passed Python's request validation.
pub(crate) struct VersionUpdate {
    new_version: String,
    abort_all_requests: bool,
}

impl TryFrom<api::UpdateWeightVersionRequest> for VersionUpdate {
    type Error = String;

    fn try_from(body: api::UpdateWeightVersionRequest) -> Result<Self, Self::Error> {
        Ok(Self {
            new_version: body.new_version.ok_or("missing field `new_version`")?,
            abort_all_requests: body.abort_all_requests,
        })
    }
}

/// Carries out pause, continue and weight updates for both adapters, one
/// control request at a time.
pub(crate) struct WeightUpdateManager {
    core: CoreHandle,
    /// Keeps the gate and the weight version from changing under a control
    /// request that is still in flight.
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

    /// Python's `update_weights_from_disk`; detached like [`Self::pause`]. A
    /// failure the scheduler reports is `success: false`, not an error.
    pub(crate) async fn update_weights_from_disk(
        self: &Arc<Self>,
        update: DiskUpdate,
    ) -> Result<api::UpdateWeightsFromDiskResponse, CoreError> {
        let this = self.clone();
        detached(async move { this.update_from_disk_locked(update).await }).await
    }

    /// Python's `/update_weight_version`; detached like [`Self::pause`]. It
    /// takes no weight writer, as in Python: the weights do not change.
    pub(crate) async fn update_weight_version(
        self: &Arc<Self>,
        update: VersionUpdate,
    ) -> Result<api::UpdateWeightVersionResponse, CoreError> {
        let this = self.clone();
        detached(async move { Ok(this.update_version_locked(update).await) }).await
    }

    /// A failure the scheduler reports is `success: false`, as in Python.
    async fn update_version_locked(
        &self,
        update: VersionUpdate,
    ) -> api::UpdateWeightVersionResponse {
        // Without the lock, a disk update carrying its own weight_version
        // could land after this relabel and overwrite it.
        let _control = self.control_lock.lock().await;
        if update.abort_all_requests {
            self.core.abort(None);
        }
        let reply = self
            .core
            .control(ControlRequest::UpdateWeightVersionReqInput(
                UpdateWeightVersionReqInput::new(
                    Rid::new().to_string(),
                    update.new_version.clone(),
                    update.abort_all_requests,
                ),
            ))
            .await;
        match reply {
            Ok(_) => {
                self.core.update_served_model(|served| {
                    served.weight_version = Some(update.new_version.clone());
                });
                api::UpdateWeightVersionResponse {
                    success: true,
                    message: format!("Weight version updated to {}", update.new_version),
                    new_version: Some(update.new_version),
                }
            }
            Err(error) => api::UpdateWeightVersionResponse {
                success: false,
                message: format!("Failed to update weight version: {error}"),
                new_version: None,
            },
        }
    }

    async fn update_from_disk_locked(
        &self,
        update: DiskUpdate,
    ) -> Result<api::UpdateWeightsFromDiskResponse, CoreError> {
        let _control = self.control_lock.lock().await;
        let load_format = update
            .body
            .load_format
            .clone()
            .or_else(|| self.core.served_model().load_format);
        if update.body.abort_all_requests {
            self.core.abort(None);
        }
        let gate = self.core.gate();
        // While paused, nothing holds a lease the scheduler would wait on,
        // and an in-place pause keeps requests that never release theirs.
        let _writer = if gate.is_paused() {
            None
        } else {
            Some(gate.hold_weights().await)
        };
        let reply = self
            .core
            .control(ControlRequest::UpdateWeightFromDiskReqInput(
                UpdateWeightFromDiskReqInput::new(
                    Rid::new().to_string(),
                    update.model_path.clone(),
                    load_format.clone(),
                    update.manifest,
                    &update.body,
                ),
            ))
            .await?;
        let output: UpdateWeightFromDiskOutput =
            rmp_serde::from_slice(&reply).map_err(|error| {
                CoreError::InvalidResponse(format!("invalid weight update response: {error}"))
            })?;
        let mut message = output.message;
        // TODO(LiyangTseng): Python also clears the multimodal preprocess cache
        // here when flush_cache is set; find whether Rust mode has one to clear.
        if output.success {
            let weight_version = update.body.weight_version;
            self.core.update_served_model(|served| {
                served.served_model_name = update.model_path.clone();
                served.model_path = update.model_path;
                served.load_format = load_format;
                if let Some(version) = &weight_version {
                    served.weight_version = Some(version.clone());
                }
            });
            if let Some(version) = weight_version {
                message.push_str(&format!(" Weight version updated to {version}."));
            }
        }
        Ok(api::UpdateWeightsFromDiskResponse {
            success: output.success,
            message,
            num_paused_requests: output.num_paused_requests,
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

/// The scheduler's `UpdateWeightFromDiskReqOutput`, as a named map.
#[derive(serde::Deserialize)]
struct UpdateWeightFromDiskOutput {
    success: bool,
    #[serde(default)]
    message: String,
    #[serde(default)]
    num_paused_requests: i64,
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
