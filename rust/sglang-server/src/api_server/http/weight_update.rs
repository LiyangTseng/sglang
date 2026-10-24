//! Pause and continue routes, sharing the weight update manager with the gRPC
//! adapter.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use sglang_api_types::api::v1 as api;

use crate::api_server::core::{CoreError, WeightUpdateManager};
use crate::message::io_struct::PauseMode;

pub(super) fn routes(manager: Arc<WeightUpdateManager>) -> Router {
    Router::new()
        .route("/pause_generation", post(pause))
        .route("/continue_generation", post(continue_generation))
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
        request
            .sink
            .try_send(ResponseItem::Control(Bytes::new()))
            .unwrap();
        into_parts(response.await.unwrap().unwrap()).await
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
        ] {
            let (router, intake) = app();
            let (status, _) = into_parts(router.oneshot(post(uri, body)).await.unwrap()).await;
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{uri} {body}");
            assert!(intake.try_recv().is_err(), "{uri} {body} must send nothing");
        }
    }
}
