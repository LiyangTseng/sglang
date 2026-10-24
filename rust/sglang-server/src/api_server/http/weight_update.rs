//! Pause, continue and weight update routes, sharing the weight update
//! manager with the gRPC adapter.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use sglang_api_types::api::v1 as api;

use crate::api_server::core::{CoreError, DiskUpdate, VersionUpdate, WeightUpdateManager};
use crate::message::io_struct::PauseMode;

pub(super) fn routes(manager: Arc<WeightUpdateManager>) -> Router {
    Router::new()
        .route("/pause_generation", post(pause))
        .route("/continue_generation", post(continue_generation))
        .route("/update_weights_from_disk", post(update_weights_from_disk))
        .route("/update_weight_version", post(update_weight_version))
        .with_state(manager)
}

async fn pause(
    State(manager): State<Arc<WeightUpdateManager>>,
    body: Result<Json<api::PauseGenerationRequest>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(error) => return (StatusCode::UNPROCESSABLE_ENTITY, error.body_text()).into_response(),
    };
    let mode = match body.mode.parse::<PauseMode>() {
        Ok(mode) => mode,
        Err(error) => return (StatusCode::UNPROCESSABLE_ENTITY, error.to_string()).into_response(),
    };
    control_response(manager.pause(mode).await)
}

async fn continue_generation(
    State(manager): State<Arc<WeightUpdateManager>>,
    body: Result<Json<api::ContinueGenerationRequest>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(error) => return (StatusCode::UNPROCESSABLE_ENTITY, error.body_text()).into_response(),
    };
    control_response(manager.continue_generation(body.torch_empty_cache).await)
}

async fn update_weights_from_disk(
    State(manager): State<Arc<WeightUpdateManager>>,
    body: Result<Json<api::UpdateWeightsFromDiskRequest>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(error) => return (StatusCode::UNPROCESSABLE_ENTITY, error.body_text()).into_response(),
    };
    let update = match DiskUpdate::try_from(body) {
        Ok(update) => update,
        Err(error) => return (StatusCode::UNPROCESSABLE_ENTITY, error).into_response(),
    };
    match manager.update_weights_from_disk(update).await {
        Ok(output) => (success_status(output.success), Json(output)).into_response(),
        Err(error) => (error.http_status(), error.to_string()).into_response(),
    }
}

async fn update_weight_version(
    State(manager): State<Arc<WeightUpdateManager>>,
    body: Result<Json<api::UpdateWeightVersionRequest>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(error) => return (StatusCode::UNPROCESSABLE_ENTITY, error.body_text()).into_response(),
    };
    let update = match VersionUpdate::try_from(body) {
        Ok(update) => update,
        Err(error) => return (StatusCode::UNPROCESSABLE_ENTITY, error).into_response(),
    };
    match manager.update_weight_version(update).await {
        Ok(output) => (success_status(output.success), Json(output)).into_response(),
        Err(error) => (error.http_status(), error.to_string()).into_response(),
    }
}

fn success_status(success: bool) -> StatusCode {
    if success {
        StatusCode::OK
    } else {
        StatusCode::BAD_REQUEST
    }
}

fn control_response(result: Result<api::GenerationControlResponse, CoreError>) -> Response {
    match result {
        Ok(output) => Json(output).into_response(),
        Err(error) => (error.http_status(), error.to_string()).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::Request,
    };
    use bytes::Bytes;
    use serde_json::{Value, json};
    use tower::ServiceExt;

    use crate::api_server::core::{CoreConfig, CoreHandle};
    use crate::message::{config::ServerArgs, request::RequestKind, response::ResponseItem};
    use crate::tokenizer_manager::wiring::TmEvent;

    fn app() -> (Router, flume::Receiver<TmEvent>) {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, _) = flume::unbounded();
        let core = CoreHandle::new(
            intake_tx,
            abort_tx,
            CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Default::default(),
                metadata: (&ServerArgs::default()).into(),
            },
        );
        (routes(WeightUpdateManager::new(core)), intake_rx)
    }

    fn post(uri: &str, body: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    async fn into_parts(response: Response) -> (StatusCode, String) {
        let status = response.status();
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    /// Send one request and play intake for the unanswered control it sends,
    /// checking its wire form.
    async fn exchange(uri: &str, body: &str, wire: Value) -> (StatusCode, String) {
        exchange_replying(uri, body, wire, ResponseItem::Control(Bytes::new())).await
    }

    /// [`exchange`] for an answered control, replying with `reply`.
    async fn exchange_replying(
        uri: &str,
        body: &str,
        wire: Value,
        reply: ResponseItem,
    ) -> (StatusCode, String) {
        let (router, intake) = app();
        let response = tokio::spawn(router.oneshot(post(uri, body)));
        let TmEvent::Intake { request, admission } =
            tokio::time::timeout(std::time::Duration::from_secs(5), intake.recv_async())
                .await
                .expect("handler did not send a control request")
                .unwrap()
        else {
            panic!("expected control intake");
        };
        assert!(admission.try_accept());
        let RequestKind::Control(control) = &request.kind else {
            panic!("expected control request");
        };
        let mut header: Value = rmp_serde::from_slice(&control.encode().unwrap()).unwrap();
        header[1] = Value::Null; // a fresh rid per request
        assert_eq!(header, wire);
        request.sink.try_send(reply).unwrap();
        let (status, body) = into_parts(response.await.unwrap().unwrap()).await;
        // Normalize JSON key order, preserving plain-text errors.
        let body = serde_json::from_str::<Value>(&body).map_or(body, |value| value.to_string());
        (status, body)
    }

    fn named(value: Value) -> ResponseItem {
        ResponseItem::Control(Bytes::from(rmp_serde::to_vec_named(&value).unwrap()))
    }

    /// Python's statuses: 200 on success, 400 with the same body on a
    /// scheduler failure.
    #[tokio::test]
    async fn update_weights_from_disk_relays_success_and_failure() {
        for (success, status) in [(true, StatusCode::OK), (false, StatusCode::BAD_REQUEST)] {
            assert_eq!(
                exchange_replying(
                    "/update_weights_from_disk",
                    r#"{"model_path":"/ckpt/2","manifest":{"w":[2,3]}}"#,
                    json!([
                        "UpdateWeightFromDiskReqInput", null, null, "/ckpt/2", null, false,
                        null, false, false, false, false, 0, true, {"w": [2, 3]}
                    ]),
                    named(json!({"success": success, "message": "m", "num_paused_requests": 0})),
                )
                .await,
                (
                    status,
                    json!({"success": success, "message": "m", "num_paused_requests": 0})
                        .to_string()
                )
            );
        }
    }

    /// An explicit null on a field with a non-zero default reads as that
    /// default; Python returns 422, a deviation the generated types impose.
    #[tokio::test]
    async fn null_on_a_defaulted_field_reads_as_the_default() {
        let (status, _) = exchange_replying(
            "/update_weights_from_disk",
            r#"{"model_path":"m","flush_cache":null}"#,
            json!([
                "UpdateWeightFromDiskReqInput",
                null,
                null,
                "m",
                null,
                false,
                null,
                false,
                false,
                false,
                false,
                0,
                true,
                null
            ]),
            named(json!({"success": true, "message": "m"})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn update_weight_version_relays_success_and_failure() {
        assert_eq!(
            exchange_replying(
                "/update_weight_version",
                r#"{"new_version":"v5"}"#,
                json!(["UpdateWeightVersionReqInput", null, null, "v5", true]),
                named(json!({})),
            )
            .await,
            (
                StatusCode::OK,
                json!({"success": true, "message": "Weight version updated to v5", "new_version": "v5"})
                    .to_string()
            )
        );
        let (status, body) = exchange_replying(
            "/update_weight_version",
            r#"{"new_version":"v6","abort_all_requests":false}"#,
            json!(["UpdateWeightVersionReqInput", null, null, "v6", false]),
            ResponseItem::Error(crate::utils::error::Error::Internal("boom".into())),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let body: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(body["success"], false);
        assert!(body.get("new_version").is_none());
    }

    /// Python's fixed acknowledgement, sent once the scheduler has the request.
    #[tokio::test]
    async fn pause_and_continue_return_pythons_acknowledgement() {
        assert_eq!(
            exchange(
                "/pause_generation",
                r#"{"mode":"retract"}"#,
                json!(["PauseGenerationReqInput", null, null, "retract"]),
            )
            .await,
            (
                StatusCode::OK,
                json!({"message": "Generation paused successfully.", "status": "ok"}).to_string()
            )
        );
        assert_eq!(
            exchange(
                "/continue_generation",
                "{}",
                json!(["ContinueGenerationReqInput", null, null, true]),
            )
            .await,
            (
                StatusCode::OK,
                json!({"message": "Generation continued successfully.", "status": "ok"})
                    .to_string()
            )
        );
    }

    /// An abort-mode pause with nothing running sends nothing to the scheduler.
    #[tokio::test]
    async fn abort_pause_with_no_generations_returns_at_once() {
        let (router, intake) = app();
        let (status, body) = into_parts(
            router
                .oneshot(post("/pause_generation", "{}"))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body,
            json!({"message": "Generation paused successfully.", "status": "ok"}).to_string()
        );
        assert!(intake.try_recv().is_err());
    }

    /// Python's Literal and body validation reject before anything is sent.
    #[tokio::test]
    async fn invalid_bodies_are_rejected_with_422() {
        for (uri, body) in [
            ("/pause_generation", r#"{"mode":"inplace"}"#),
            ("/pause_generation", r#"{"mode":1}"#),
            ("/pause_generation", "not json"),
            ("/continue_generation", r#"{"torch_empty_cache":"yes"}"#),
            ("/update_weights_from_disk", "{}"),
            ("/update_weights_from_disk", r#"{"model_path":null}"#),
            (
                "/update_weights_from_disk",
                r#"{"model_path":"m","manifest":[1]}"#,
            ),
            ("/update_weight_version", "{}"),
            ("/update_weight_version", r#"{"new_version":null}"#),
        ] {
            let (router, intake) = app();
            let (status, _) = into_parts(router.oneshot(post(uri, body)).await.unwrap()).await;
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{uri} {body}");
            assert!(intake.try_recv().is_err(), "{uri} {body} must send nothing");
        }
    }
}
