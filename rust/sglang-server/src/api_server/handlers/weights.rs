//! Weight management: process-group setup, remote-instance sends, parameter
//! reads, and the weight updates, which hold the frontend's update lock.

use sglang_api_types::api::v1 as api;

use crate::message::ids::Rid;
use crate::message::io_struct::{
    ControlRequest, DestroyWeightsUpdateGroupReqInput, DiskWeightUpdate, GetWeightsByNameReqInput,
    InitWeightsSendGroupForRemoteInstanceReqInput, InitWeightsUpdateGroupReqInput,
    SendWeightsToRemoteInstanceReqInput, UpdateWeightFromDiskReqInput, WeightsSendGroup,
    WeightsUpdateGroup,
};

use super::handle::CoreHandle;
use crate::api_server::core::CoreError;
use crate::api_server::core::fanout::{self, RequestBuilder};

impl CoreHandle {
    /// Read `name` from every DP rank, each dimension truncated to
    /// `truncate_size` elements: one JSON value per rank in rank order, `null`
    /// where a rank found no such parameter.
    pub(crate) async fn get_weights_by_name(
        &self,
        name: String,
        truncate_size: i64,
    ) -> Result<Vec<serde_json::Value>, CoreError> {
        if name.is_empty() {
            return Err(CoreError::InvalidArgument("missing field `name`".into()));
        }
        let outputs = self
            .control(&|| {
                ControlRequest::GetWeightsByNameReqInput(GetWeightsByNameReqInput::new(
                    Rid::new().to_string(),
                    name.clone(),
                    truncate_size,
                ))
            })
            .await
            .decode::<WeightsByNameOutput>("get-weights-by-name")?;
        Ok(outputs
            .into_iter()
            .map(|output| output.parameter.unwrap_or(serde_json::Value::Null))
            .collect())
    }

    pub(crate) async fn init_weights_update_group(
        &self,
        group: WeightsUpdateGroup,
    ) -> Result<WeightGroupOutcome, CoreError> {
        self.single_rank_group("init_weights_update_group", &|| {
            ControlRequest::InitWeightsUpdateGroupReqInput(InitWeightsUpdateGroupReqInput::new(
                Rid::new().to_string(),
                group.clone(),
            ))
        })
        .await
    }

    pub(crate) async fn destroy_weights_update_group(
        &self,
        group_name: String,
    ) -> Result<WeightGroupOutcome, CoreError> {
        self.single_rank_group("destroy_weights_update_group", &|| {
            ControlRequest::DestroyWeightsUpdateGroupReqInput(
                DestroyWeightsUpdateGroupReqInput::new(Rid::new().to_string(), group_name.clone()),
            )
        })
        .await
    }

    pub(crate) async fn init_weights_send_group_for_remote_instance(
        &self,
        group: WeightsSendGroup,
    ) -> Result<WeightGroupOutcome, CoreError> {
        self.single_rank_group("init_weights_send_group_for_remote_instance", &|| {
            ControlRequest::InitWeightsSendGroupForRemoteInstanceReqInput(
                InitWeightsSendGroupForRemoteInstanceReqInput::new(
                    Rid::new().to_string(),
                    group.clone(),
                ),
            )
        })
        .await
    }

    pub(crate) async fn send_weights_to_remote_instance(
        &self,
        master_address: String,
        ports: String,
        group_name: String,
    ) -> Result<WeightGroupOutcome, CoreError> {
        self.single_rank_group("send_weights_to_remote_instance", &|| {
            ControlRequest::SendWeightsToRemoteInstanceReqInput(
                SendWeightsToRemoteInstanceReqInput::new(
                    Rid::new().to_string(),
                    master_address.clone(),
                    ports.clone(),
                    group_name.clone(),
                ),
            )
        })
        .await
    }

    /// Python's `update_weights_from_disk`: reload on every DP rank, holding
    /// the update lock (unless paused) so this listener's in-flight generates
    /// finish on the old weights and new ones wait for the new. Each peer
    /// listener holds its own lock through the relay. A failure a scheduler
    /// reports is `success: false`, not an error. Detached: a client that
    /// goes away does not leave the lock or the schedulers half-updated.
    pub(crate) async fn update_weights_from_disk(
        &self,
        update: DiskWeightUpdate,
    ) -> Result<DiskUpdateOutcome, CoreError> {
        if update.model_path.is_empty() {
            return Err(CoreError::InvalidArgument(
                "missing field `model_path`".into(),
            ));
        }
        let this = self.clone();
        tokio::spawn(async move { this.update_weights_from_disk_detached(update).await })
            .await
            .map_err(|error| CoreError::Internal(format!("weight update task failed: {error}")))?
    }

    async fn update_weights_from_disk_detached(
        &self,
        mut update: DiskWeightUpdate,
    ) -> Result<DiskUpdateOutcome, CoreError> {
        if update.load_format.is_none() {
            update.load_format = self.model_facts().await.load_format.clone();
        }
        if update.abort_all_requests {
            self.abort(None);
        }
        let outcomes = {
            let _writer = self.inner.frontend.hold_weights_unless_paused().await;
            self.control(&|| {
                ControlRequest::UpdateWeightFromDiskReqInput(UpdateWeightFromDiskReqInput::new(
                    Rid::new().to_string(),
                    update.clone(),
                ))
            })
            .await
            .decode::<DiskUpdateOutcome>("update-weights-from-disk")
        };
        // Any rank may have reloaded, even when another failed.
        self.inner.frontend.invalidate();
        let outcomes = outcomes?;
        let (success, mut message) = fanout::merge_status(
            outcomes
                .iter()
                .map(|outcome| (outcome.success, outcome.message.as_str())),
        );
        // TODO: Python's TokenizerManager also moves `model_path` and
        // `served_model_name` to `update.model_path` here, and clears the MM
        // preprocess cache when `flush_cache` is set; the scheduler-owned
        // facts do not carry the new path yet.
        if success && let Some(version) = &update.weight_version {
            message.push_str(&format!(" Weight version updated to {version}."));
        }
        Ok(DiskUpdateOutcome {
            success,
            message,
            num_paused_requests: outcomes
                .iter()
                .map(|outcome| outcome.num_paused_requests)
                .sum(),
        })
    }

    /// The group operations Python restricts to one DP rank (its
    /// `assert dp_size == 1`); kept as a 400 here rather than lifted, so the
    /// first pass behaves as the Python server does on the deployments it
    /// supports. Merged as `merge_results`.
    async fn single_rank_group(
        &self,
        what: &str,
        build: &RequestBuilder<'_>,
    ) -> Result<WeightGroupOutcome, CoreError> {
        let dp_size = self.inner.metadata.deployment_info.dp_size.max(1);
        if dp_size > 1 {
            return Err(CoreError::InvalidArgument(format!(
                "data-parallel replicas are not supported for {what} (dp_size={dp_size})"
            )));
        }
        let outcomes = self
            .control(build)
            .await
            .decode::<WeightGroupOutcome>(what)?;
        let (success, message) = fanout::merge_status(
            outcomes
                .iter()
                .map(|outcome| (outcome.success, outcome.message.as_str())),
        );
        Ok(WeightGroupOutcome { success, message })
    }
}

/// Python answers both a missing bootstrap reply and a rejected one with
/// this message and a 400.
fn transfer_info_unavailable(rank: i64) -> CoreError {
    CoreError::InvalidArgument(format!(
        "Failed to get transfer engine info for rank {rank}"
    ))
}

impl CoreHandle {
    /// The transfer-engine info `rank` registered with this node's engine-info
    /// bootstrap server, as the bootstrap server's JSON text. A proxy, not a
    /// scheduler control: Python's handler does the same outbound GET with a
    /// five-second timeout.
    pub(crate) async fn remote_instance_transfer_engine_info(
        &self,
        rank: i64,
    ) -> Result<String, CoreError> {
        if rank < 0 {
            return Err(CoreError::InvalidArgument(
                "Missing or invalid rank parameter".into(),
            ));
        }
        let Some(base_url) = &self
            .inner
            .metadata
            .deployment_info
            .engine_info_bootstrap_url
        else {
            return Err(transfer_info_unavailable(rank));
        };
        let url = format!(
            "{}/get_transfer_engine_info",
            base_url.trim_end_matches('/')
        );
        let response = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .map_err(|error| CoreError::Internal(error.to_string()))?
            .get(url)
            .query(&[("rank", rank)])
            .send()
            .await;
        let response = match response {
            Ok(response) if response.status().is_success() => response,
            Ok(response) => {
                tracing::warn!(rank, status = %response.status(), "transfer engine info lookup refused");
                return Err(transfer_info_unavailable(rank));
            }
            Err(error) => {
                tracing::warn!(rank, %error, "transfer engine info lookup failed");
                return Err(transfer_info_unavailable(rank));
            }
        };
        let body = response
            .text()
            .await
            .map_err(|_| transfer_info_unavailable(rank))?;
        // Python returns `resp.json()`, so a body that is not JSON is the
        // same failure as no body.
        serde_json::from_str::<serde::de::IgnoredAny>(&body)
            .map_err(|_| transfer_info_unavailable(rank))?;
        Ok(body)
    }
}

/// The scheduler's `GetWeightsByNameReqOutput`: the parameter as the
/// scheduler shaped it (`List[float | List[float]]`), left untyped.
#[derive(Debug, serde::Deserialize)]
struct WeightsByNameOutput {
    #[serde(default)]
    parameter: Option<serde_json::Value>,
}

/// An `/update_weights_from_disk` body as Python validates it: `model_path`
/// required, `manifest` a JSON object (or null).
pub(crate) fn disk_weight_update(
    request: api::UpdateWeightsFromDiskRequest,
) -> Result<DiskWeightUpdate, CoreError> {
    if request.model_path.is_empty() {
        return Err(CoreError::InvalidArgument(
            "missing field `model_path`".into(),
        ));
    }
    let manifest = match request.manifest.as_deref().map(serde_json::from_str) {
        None | Some(Ok(serde_json::Value::Null)) => None,
        Some(Ok(serde_json::Value::Object(manifest))) => Some(manifest),
        Some(_) => {
            return Err(CoreError::InvalidArgument(
                "`manifest` must be a JSON object".into(),
            ));
        }
    };
    Ok(DiskWeightUpdate {
        model_path: request.model_path,
        load_format: request.load_format,
        abort_all_requests: request.abort_all_requests,
        weight_version: request.weight_version,
        is_async: request.is_async,
        torch_empty_cache: request.torch_empty_cache,
        keep_pause: request.keep_pause,
        recapture_cuda_graph: request.recapture_cuda_graph,
        token_step: request.token_step,
        flush_cache: request.flush_cache.unwrap_or(true),
        manifest,
    })
}

/// The scheduler's `UpdateWeightFromDiskReqOutput`, merged across ranks.
#[derive(Debug, serde::Deserialize, PartialEq)]
pub(crate) struct DiskUpdateOutcome {
    pub(crate) success: bool,
    #[serde(default)]
    pub(crate) message: String,
    #[serde(default)]
    pub(crate) num_paused_requests: i64,
}

/// The `{success, message}` output every group operation returns.
#[derive(Debug, serde::Deserialize, PartialEq)]
pub(crate) struct WeightGroupOutcome {
    pub(crate) success: bool,
    #[serde(default)]
    pub(crate) message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::core::CoreErrorKind;
    use crate::api_server::handlers::test_utils::*;
    use crate::api_server::handlers::{CoreConfig, CoreMetadata};
    use crate::message::config::DeploymentInfo;
    use crate::message::request::RequestKind;
    use crate::message::response::ResponseItem;
    use bytes::Bytes;

    /// A parameter read keeps the scheduler's numbers as it shaped them, one
    /// entry per rank, with a missing parameter as `null`.
    #[tokio::test]
    async fn get_weights_by_name_relays_each_ranks_parameter_untyped() {
        use crate::api_server::core::fanout::testing::DummyTarget;
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, _abort_rx) = flume::unbounded();
        let handle = CoreHandle::new(
            intake_tx,
            abort_tx,
            CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Vec::new(),
                metadata: CoreMetadata::default(),
                peer_endpoints: vec![DummyTarget::replying(&[1], |_| {
                    Ok(Bytes::from(
                        rmp_serde::to_vec_named(&serde_json::json!({"parameter": null})).unwrap(),
                    ))
                })],
            },
        );
        let (values, ()) = tokio::join!(handle.get_weights_by_name("w".into(), 2), async {
            let request = accept_intake(intake_rx.recv_async().await.unwrap());
            let RequestKind::Control(control) = &request.kind else {
                panic!("expected a control request");
            };
            let rmpv::Value::Array(items) =
                rmpv::decode::read_value(&mut &control.encode().unwrap()[..]).unwrap()
            else {
                panic!("positional wire");
            };
            assert_eq!(items[3].as_str(), Some("w"));
            assert_eq!(items[4].as_i64(), Some(2));
            request
                .sink
                .try_send(ResponseItem::Control(Bytes::from(
                    rmp_serde::to_vec_named(&serde_json::json!({
                        "parameter": [[0.5, 1], [2.25, -3]]
                    }))
                    .unwrap(),
                )))
                .unwrap();
        },);
        assert_eq!(
            values.unwrap(),
            vec![
                serde_json::json!([[0.5, 1], [2.25, -3]]),
                serde_json::Value::Null
            ]
        );
        assert_eq!(
            handle
                .get_weights_by_name(String::new(), 100)
                .await
                .unwrap_err()
                .kind(),
            CoreErrorKind::InvalidArgument
        );
    }

    /// The proxy relays the bootstrap server's JSON as is, and maps a missing
    /// rank, a refused lookup, and an unconfigured bootstrap server onto
    /// Python's one failure message.
    #[tokio::test]
    async fn transfer_engine_info_is_proxied_from_the_bootstrap_server() {
        use axum::{Router, routing::get};
        use std::future::IntoFuture;
        let app = Router::new().route(
            "/get_transfer_engine_info",
            get(
                |axum::extract::Query(q): axum::extract::Query<
                    std::collections::HashMap<String, i64>,
                >| async move {
                    match q.get("rank") {
                        Some(0) => (
                            axum::http::StatusCode::OK,
                            r#"{"rank": 0, "remote_instance_transfer_engine_info": ["h", 1]}"#,
                        ),
                        _ => (axum::http::StatusCode::NOT_FOUND, r#"{"detail": "no"}"#),
                    }
                },
            ),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(axum::serve(listener, app).into_future());

        let (intake_tx, _intake_rx) = flume::unbounded();
        let (abort_tx, _abort_rx) = flume::unbounded();
        let handle = CoreHandle::new(
            intake_tx,
            abort_tx,
            CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Vec::new(),
                metadata: CoreMetadata {
                    deployment_info: DeploymentInfo {
                        engine_info_bootstrap_url: Some(url),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                peer_endpoints: Vec::new(),
            },
        );
        let info = handle
            .remote_instance_transfer_engine_info(0)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&info).unwrap(),
            serde_json::json!({"rank": 0, "remote_instance_transfer_engine_info": ["h", 1]})
        );
        for rank in [-1, 7] {
            let error = handle
                .remote_instance_transfer_engine_info(rank)
                .await
                .unwrap_err();
            assert_eq!(error.kind(), CoreErrorKind::InvalidArgument, "{rank}");
        }
        assert!(
            Harness::unbounded(8)
                .handle
                .remote_instance_transfer_engine_info(0)
                .await
                .unwrap_err()
                .to_string()
                .contains("rank 0"),
            "no bootstrap server configured"
        );
    }

    /// The group operations fan out on one rank and merge the outcome, and
    /// refuse a multi-rank deployment before reaching any scheduler.
    #[tokio::test]
    async fn group_operations_merge_outcome_and_require_one_dp_rank() {
        let harness = Harness::unbounded(8);
        let handle = harness.handle.clone();
        let (outcome, ()) = tokio::join!(
            handle.destroy_weights_update_group("g".into()),
            answer_local_control(
                &harness.intake_rx,
                Bytes::from(
                    rmp_serde::to_vec_named(
                        &serde_json::json!({"success": false, "message": "no such group"}),
                    )
                    .unwrap(),
                ),
            ),
        );
        assert_eq!(
            outcome.unwrap(),
            WeightGroupOutcome {
                success: false,
                message: "no such group".into()
            }
        );

        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, _abort_rx) = flume::unbounded();
        let two_ranks = CoreHandle::new(
            intake_tx,
            abort_tx,
            CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Vec::new(),
                metadata: CoreMetadata {
                    deployment_info: DeploymentInfo {
                        dp_size: 2,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                peer_endpoints: Vec::new(),
            },
        );
        let error = two_ranks
            .send_weights_to_remote_instance("h".into(), "1".into(), "g".into())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), CoreErrorKind::InvalidArgument);
        assert!(intake_rx.try_recv().is_err(), "refused before any rank");
    }

    fn disk_update(model_path: &str) -> DiskWeightUpdate {
        disk_weight_update(api::UpdateWeightsFromDiskRequest {
            model_path: model_path.into(),
            // Set, so the update does not refresh the facts first.
            load_format: Some("auto".into()),
            ..Default::default()
        })
        .unwrap()
    }

    fn generate_request() -> crate::message::request::GenerateRequest {
        crate::message::request::GenerateRequest {
            rid: Rid::new(),
            input_ids: Some(vec![1]),
            ..Default::default()
        }
    }

    fn disk_output(success: bool, message: &str) -> Bytes {
        Bytes::from(
            rmp_serde::to_vec_named(&serde_json::json!({
                "success": success,
                "message": message,
                "num_paused_requests": 0,
            }))
            .unwrap(),
        )
    }

    fn control_tag(request: &crate::message::request::Request) -> String {
        let RequestKind::Control(control) = &request.kind else {
            panic!("expected a control request");
        };
        let rmpv::Value::Array(items) =
            rmpv::decode::read_value(&mut &control.encode().unwrap()[..]).unwrap()
        else {
            panic!("positional wire");
        };
        items[0].as_str().unwrap().to_owned()
    }

    async fn settle() {
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    }

    /// Python's writer lock: the update reaches the scheduler only after the
    /// in-flight generate's response ends, and a generate arriving while the
    /// update waits is held back until the update is done.
    #[tokio::test]
    async fn disk_update_drains_in_flight_generates_and_holds_new_ones() {
        let harness = Harness::unbounded(8);
        let in_flight = harness.handle.generate(generate_request()).await.unwrap();
        let _ = accept_intake(harness.intake_rx.recv_async().await.unwrap());

        let handle = harness.handle.clone();
        let update = tokio::spawn(async move {
            handle
                .update_weights_from_disk(DiskWeightUpdate {
                    weight_version: Some("v2".into()),
                    ..disk_update("/ckpt/step-10")
                })
                .await
        });
        settle().await;
        assert!(harness.intake_rx.try_recv().is_err(), "waits for the lease");

        let handle = harness.handle.clone();
        let late = tokio::spawn(async move { handle.generate(generate_request()).await });
        settle().await;
        assert!(
            harness.intake_rx.try_recv().is_err(),
            "a queued writer holds back new generates"
        );

        drop(in_flight);
        let request = accept_intake(harness.intake_rx.recv_async().await.unwrap());
        assert_eq!(control_tag(&request), "UpdateWeightFromDiskReqInput");
        settle().await;
        assert!(
            harness.intake_rx.try_recv().is_err(),
            "no generate while the scheduler updates"
        );
        request
            .sink
            .try_send(ResponseItem::Control(disk_output(true, "Succeeded.")))
            .unwrap();
        let outcome = update.await.unwrap().unwrap();
        assert!(outcome.success);
        assert_eq!(outcome.message, "Succeeded. Weight version updated to v2.");
        assert!(
            harness.handle.inner.frontend.is_stale(),
            "facts refresh next read"
        );

        let released = accept_intake(harness.intake_rx.recv_async().await.unwrap());
        assert!(matches!(released.kind, RequestKind::Generate(_)));
        drop(late);
    }

    /// While paused, the update skips the lock, as Python does: an in-place
    /// pause keeps requests whose leases would never be released.
    #[tokio::test]
    async fn disk_update_skips_the_lock_while_paused() {
        let harness = Harness::unbounded(8);
        let _in_flight = harness.handle.generate(generate_request()).await.unwrap();
        let _ = accept_intake(harness.intake_rx.recv_async().await.unwrap());
        harness.handle.inner.frontend.set_paused(true);

        let handle = harness.handle.clone();
        let (outcome, ()) =
            tokio::join!(handle.update_weights_from_disk(disk_update("m")), async {
                let request = accept_intake(harness.intake_rx.recv_async().await.unwrap());
                request
                    .sink
                    .try_send(ResponseItem::Control(disk_output(false, "bad path")))
                    .unwrap();
            });
        let outcome = outcome.unwrap();
        assert!(!outcome.success, "a scheduler failure is in the body");
        assert_eq!(outcome.message, "bad path");
    }

    /// A peer listener serving a relayed update holds its own lock, so its
    /// generates drain before its scheduler reloads.
    #[tokio::test]
    async fn relayed_disk_update_holds_this_listeners_lock() {
        let harness = Harness::unbounded(8);
        let in_flight = harness.handle.generate(generate_request()).await.unwrap();
        let _ = accept_intake(harness.intake_rx.recv_async().await.unwrap());

        let header = ControlRequest::UpdateWeightFromDiskReqInput(
            UpdateWeightFromDiskReqInput::new("peer-rid".into(), disk_update("m")),
        )
        .encode()
        .unwrap();
        let handle = harness.handle.clone();
        let relay = tokio::spawn(async move { handle.forward_control_req(header).await });
        settle().await;
        assert!(harness.intake_rx.try_recv().is_err(), "waits for the lease");

        drop(in_flight);
        let request = accept_intake(harness.intake_rx.recv_async().await.unwrap());
        request
            .sink
            .try_send(ResponseItem::Control(disk_output(true, "")))
            .unwrap();
        relay.await.unwrap().unwrap();
        assert!(harness.handle.inner.frontend.is_stale());
    }

    #[test]
    fn disk_update_body_is_validated_as_python_does() {
        let error = disk_weight_update(api::UpdateWeightsFromDiskRequest::default()).unwrap_err();
        assert_eq!(error.kind(), CoreErrorKind::InvalidArgument);
        let error = disk_weight_update(api::UpdateWeightsFromDiskRequest {
            model_path: "m".into(),
            manifest: Some("[1]".into()),
            ..Default::default()
        })
        .unwrap_err();
        assert_eq!(error.kind(), CoreErrorKind::InvalidArgument);
        let update = disk_weight_update(api::UpdateWeightsFromDiskRequest {
            model_path: "m".into(),
            manifest: Some(r#"{"a": 1}"#.into()),
            ..Default::default()
        })
        .unwrap();
        assert!(update.flush_cache, "Python's default");
        assert_eq!(update.manifest.unwrap()["a"], 1);
    }
}
