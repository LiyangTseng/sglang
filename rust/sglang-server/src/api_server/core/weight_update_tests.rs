//! Tests for the pause gate, generation leases and the control lock. Each test
//! plays the scheduler by reading `TmEvent::Intake` from the core's inbox.

use std::time::Duration;

use bytes::Bytes;

use super::*;
use crate::api_server::core::{CoreCall, CoreConfig, CoreMetadata, HealthStatus};
use crate::message::request::{GenerateRequest, Request, RequestKind};
use crate::message::response::ResponseItem;
use crate::tokenizer_manager::wiring::{AbortSource, TmEvent};

const QUIET: Duration = Duration::from_millis(100);
const DEADLINE: Duration = Duration::from_secs(5);

struct Harness {
    core: CoreHandle,
    manager: Arc<WeightUpdateManager>,
    intake_rx: flume::Receiver<TmEvent>,
    abort_rx: flume::Receiver<AbortSource>,
}

impl Harness {
    fn new() -> Self {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::unbounded();
        let core = CoreHandle::new(
            intake_tx,
            abort_tx,
            CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Vec::new(),
                metadata: CoreMetadata::default(),
            },
        );
        Self {
            manager: WeightUpdateManager::new(core.clone()),
            core,
            intake_rx,
            abort_rx,
        }
    }

    /// The next request intake accepts, as the TokenizerManager would.
    async fn next_intake(&self) -> Request {
        let TmEvent::Intake { request, admission } =
            tokio::time::timeout(DEADLINE, self.intake_rx.recv_async())
                .await
                .expect("no request reached intake")
                .unwrap()
        else {
            panic!("expected an intake event");
        };
        assert!(admission.try_accept());
        request
    }

    async fn assert_no_intake(&self) {
        assert!(
            tokio::time::timeout(QUIET, self.intake_rx.recv_async())
                .await
                .is_err(),
            "nothing may reach intake yet"
        );
    }

    /// Play intake for one unanswered control: check its wire tag and fields,
    /// then acknowledge the push.
    async fn ack_control(&self, tag: &str, fields: serde_json::Value) {
        let request = self.next_intake().await;
        let RequestKind::Control(control) = &request.kind else {
            panic!("expected a control request");
        };
        let wire: serde_json::Value = rmp_serde::from_slice(&control.encode().unwrap()).unwrap();
        let mut expected = vec![
            tag.into(),
            request.rid.as_str().into(),
            serde_json::Value::Null,
        ];
        expected.extend(fields.as_array().unwrap().iter().cloned());
        assert_eq!(wire, serde_json::Value::Array(expected));
        request
            .sink
            .try_send(ResponseItem::Control(Bytes::new()))
            .unwrap();
    }

    async fn pause(&self, mode: PauseMode) {
        let manager = self.manager.clone();
        let pause = tokio::spawn(async move { manager.pause(mode).await });
        if mode != PauseMode::Abort {
            let mode = serde_json::to_value(mode).unwrap();
            self.ack_control("PauseGenerationReqInput", serde_json::json!([mode]))
                .await;
        }
        finished(pause).await.unwrap();
    }

    fn generate(&self) -> tokio::task::JoinHandle<Result<CoreCall, CoreError>> {
        let core = self.core.clone();
        tokio::spawn(async move { core.generate(generate_request()).await })
    }
}

fn generate_request() -> GenerateRequest {
    GenerateRequest {
        rid: Rid::new(),
        input_ids: Some(vec![1]),
        ..Default::default()
    }
}

async fn finished<T>(task: tokio::task::JoinHandle<T>) -> T {
    tokio::time::timeout(DEADLINE, task)
        .await
        .expect("task did not finish")
        .unwrap()
}

async fn assert_pending<T>(task: &mut tokio::task::JoinHandle<T>) {
    assert!(
        tokio::time::timeout(QUIET, task).await.is_err(),
        "task must still be waiting"
    );
}

/// In-place and retract pauses send one frame, wait for no scheduler reply,
/// and close the gate; nothing aborts the frame's rid.
#[tokio::test]
async fn in_place_pause_sends_one_unanswered_frame() {
    for mode in [PauseMode::InPlace, PauseMode::Retract] {
        let h = Harness::new();
        h.pause(mode).await;
        assert!(h.core.gate().is_paused());
        h.assert_no_intake().await;
        assert!(h.abort_rx.try_recv().is_err(), "no AbortReq for the pause");
    }
}

/// A generation waits at the gate while paused. Continue tells the scheduler
/// to resume before it admits the waiting generation.
#[tokio::test]
async fn generate_waits_while_paused_and_continue_resumes_the_scheduler_first() {
    let h = Harness::new();
    h.pause(PauseMode::InPlace).await;
    let mut waiting = h.generate();
    h.assert_no_intake().await;
    assert_pending(&mut waiting).await;

    let manager = h.manager.clone();
    let resume = tokio::spawn(async move { manager.continue_generation(false).await });
    h.ack_control("ContinueGenerationReqInput", serde_json::json!([false]))
        .await;
    let response = finished(resume).await.unwrap();
    assert_eq!(response.message, "Generation continued successfully.");
    assert_eq!(response.status, "ok");

    let request = h.next_intake().await;
    assert!(matches!(request.kind, RequestKind::Generate(_)));
    finished(waiting).await.unwrap();
    assert!(!h.core.gate().is_paused());
}

/// An abort-mode pause aborts every generation and returns only once their
/// leases are released; the scheduler itself is not paused.
#[tokio::test]
async fn abort_pause_aborts_generations_and_waits_for_their_leases() {
    let h = Harness::new();
    let call = finished(h.generate()).await.unwrap();
    let request = h.next_intake().await;

    let manager = h.manager.clone();
    let mut pause = tokio::spawn(async move { manager.pause(PauseMode::Abort).await });
    let AbortSource::Guard(aborted) = tokio::time::timeout(DEADLINE, h.abort_rx.recv_async())
        .await
        .unwrap()
        .unwrap()
    else {
        panic!("expected a scheduler abort");
    };
    assert_eq!(aborted, request.rid);
    assert_pending(&mut pause).await;

    drop(call); // the client disconnects: its lease goes with it
    let response = finished(pause).await.unwrap();
    assert_eq!(response.message, "Generation paused successfully.");
    h.assert_no_intake().await; // no PauseGenerationReqInput
}

/// A request holds a lease but is not yet registered when an abort-mode
/// pause closes the gate and aborts everything.
/// Its registration must be undone so it waits at the gate instead of running.
#[tokio::test]
async fn a_registration_after_the_gate_closes_is_undone() {
    let h = Harness::new();
    let rid = Rid::new();
    let lease = h.core.gate().lease().await;
    let manager = h.manager.clone();
    let mut pause = tokio::spawn(async move { manager.pause(PauseMode::Abort).await });
    while !h.core.gate().is_paused() {
        tokio::task::yield_now().await;
    }
    assert!(
        h.core
            .register_unless_paused(std::slice::from_ref(&rid))
            .is_none(),
        "registered after the abort-all, so the pause would never abort it"
    );
    assert!(
        !h.core.abort(Some(rid.client_facing())),
        "the undone registration leaves nothing to abort"
    );
    assert_pending(&mut pause).await;
    drop(lease);
    finished(pause).await.unwrap();
}

/// Control requests never interleave: a continue sent during an abort-mode
/// pause's drain waits for the pause to finish.
#[tokio::test]
async fn control_requests_run_one_at_a_time() {
    let h = Harness::new();
    let call = finished(h.generate()).await.unwrap();
    let _request = h.next_intake().await;

    let manager = h.manager.clone();
    let mut pause = tokio::spawn(async move { manager.pause(PauseMode::Abort).await });
    assert_pending(&mut pause).await;
    let manager = h.manager.clone();
    let mut resume = tokio::spawn(async move { manager.continue_generation(true).await });
    h.assert_no_intake().await;
    assert_pending(&mut resume).await;

    drop(call);
    finished(pause).await.unwrap();
    h.ack_control("ContinueGenerationReqInput", serde_json::json!([true]))
        .await;
    finished(resume).await.unwrap();
    assert!(!h.core.gate().is_paused());
}

/// A pause whose caller goes away still runs to completion and releases the
/// control lock, and its frame is not aborted.
#[tokio::test]
async fn pause_completes_after_its_caller_goes_away() {
    let h = Harness::new();
    let caller = tokio::time::timeout(
        Duration::from_millis(20),
        h.manager.pause(PauseMode::InPlace),
    );
    assert!(caller.await.is_err(), "the caller gives up before the ack");
    h.ack_control("PauseGenerationReqInput", serde_json::json!(["in_place"]))
        .await;

    let manager = h.manager.clone();
    let resume = tokio::spawn(async move { manager.continue_generation(true).await });
    h.ack_control("ContinueGenerationReqInput", serde_json::json!([true]))
        .await;
    finished(resume).await.unwrap();
    assert!(
        h.abort_rx.try_recv().is_err(),
        "no AbortReq for either frame"
    );
}

/// Items of one batch share a lease that outlives every item but the last.
#[tokio::test]
async fn a_batch_holds_its_lease_until_its_last_item_ends() {
    let h = Harness::new();
    let core = h.core.clone();
    let mut calls = core
        .generate_batch(vec![generate_request(), generate_request()])
        .await
        .unwrap();
    h.next_intake().await;
    h.next_intake().await;

    let manager = h.manager.clone();
    let mut pause = tokio::spawn(async move { manager.pause(PauseMode::Abort).await });
    drop(calls.pop());
    assert_pending(&mut pause).await;
    drop(calls);
    finished(pause).await.unwrap();
}

/// The health probe waits at the gate like any generation, so a paused
/// server reports a stall instead of submitting the probe.
#[tokio::test]
async fn health_probe_stalls_while_paused() {
    let h = Harness::new();
    h.pause(PauseMode::InPlace).await;
    let status = h
        .core
        .probe_health(Duration::from_millis(100))
        .await
        .unwrap();
    assert!(matches!(status, HealthStatus::Stalled));
    h.assert_no_intake().await;
}
