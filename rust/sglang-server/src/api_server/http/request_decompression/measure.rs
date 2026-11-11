//! Measurement harness for sgl-project/sglang#49934 (item 9 of #49055); not
//! for merge: its `#[global_allocator]` replaces the allocator of the whole
//! `sglang-server` test binary.
//!
//! All measurements for one method: `measure_arms/run.sh <method>`, from `rust/`.
//! One measurement: `cargo test --release -p sglang-server --lib
//! request_decompression::measure::<name> -- --ignored --nocapture`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::convert::Infallible;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::time::{Duration, Instant};

use axum::routing::post;
use bytes::Bytes;
use futures::StreamExt;
use tower::ServiceExt;

use super::*;

// ---------------------------------------------------------------------------
// Counting allocator: wraps the system allocator (glibc malloc on Linux, the
// allocator the server ships with) and counts every thread's calls.
// ---------------------------------------------------------------------------

struct Counting;

#[global_allocator]
static GLOBAL: Counting = Counting;

static ON: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static REALLOCS: AtomicU64 = AtomicU64::new(0);
static REQUESTED: AtomicU64 = AtomicU64::new(0);
static MOVED: AtomicU64 = AtomicU64::new(0);
static MOVED_BYTES: AtomicU64 = AtomicU64::new(0);
// Live and peak are tracked even outside a window, so a window's peak is exact.
static LIVE: AtomicI64 = AtomicI64::new(0);
static PEAK: AtomicI64 = AtomicI64::new(0);

/// Reallocs touching at least this many bytes are logged one by one.
const BIG: usize = 1 << 20;
const EVENTS: usize = 64;
static EV_N: AtomicUsize = AtomicUsize::new(0);
/// [old size, new size, nanoseconds inside realloc, pointer moved]
static EV: [[AtomicU64; 4]; EVENTS] = [const { [const { AtomicU64::new(0) }; 4] }; EVENTS];

fn grow(delta: i64) {
    let now = LIVE.fetch_add(delta, Relaxed) + delta;
    PEAK.fetch_max(now, Relaxed);
}

fn count_alloc(size: usize) {
    grow(size as i64);
    if ON.load(Relaxed) {
        ALLOCS.fetch_add(1, Relaxed);
        REQUESTED.fetch_add(size as u64, Relaxed);
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            count_alloc(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            count_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        grow(-(layout.size() as i64));
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let start = Instant::now();
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        let nanos = start.elapsed().as_nanos() as u64;
        if new_ptr.is_null() {
            return new_ptr;
        }
        let old_size = layout.size();
        grow(new_size as i64 - old_size as i64);
        if ON.load(Relaxed) {
            REALLOCS.fetch_add(1, Relaxed);
            REQUESTED.fetch_add(new_size as u64, Relaxed);
            let moved = new_ptr != ptr;
            if moved {
                MOVED.fetch_add(1, Relaxed);
                MOVED_BYTES.fetch_add(old_size.min(new_size) as u64, Relaxed);
            }
            if old_size.max(new_size) >= BIG {
                let i = EV_N.fetch_add(1, Relaxed);
                if i < EVENTS {
                    for (slot, value) in
                        EV[i]
                            .iter()
                            .zip([old_size as u64, new_size as u64, nanos, moved as u64])
                    {
                        slot.store(value, Relaxed);
                    }
                }
            }
        }
        new_ptr
    }
}

#[derive(Debug)]
struct Stats {
    allocs: u64,
    reallocs: u64,
    requested: u64,
    moved: u64,
    moved_bytes: u64,
    peak: i64,
    events: Vec<[u64; 4]>,
}

/// Opens a counting window; returns the live bytes it is measured against.
fn start() -> i64 {
    for counter in [&ALLOCS, &REALLOCS, &REQUESTED, &MOVED, &MOVED_BYTES] {
        counter.store(0, Relaxed);
    }
    EV_N.store(0, Relaxed);
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    ON.store(true, Relaxed);
    base
}

fn stop(base: i64) -> Stats {
    ON.store(false, Relaxed);
    let n = EV_N.load(Relaxed).min(EVENTS);
    Stats {
        allocs: ALLOCS.load(Relaxed),
        reallocs: REALLOCS.load(Relaxed),
        requested: REQUESTED.load(Relaxed),
        moved: MOVED.load(Relaxed),
        moved_bytes: MOVED_BYTES.load(Relaxed),
        peak: PEAK.load(Relaxed) - base,
        events: EV[..n]
            .iter()
            .map(|e| std::array::from_fn(|k| e[k].load(Relaxed)))
            .collect(),
    }
}

fn live_since(base: i64) -> i64 {
    LIVE.load(Relaxed) - base
}

fn vm_rss_kib() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .and_then(|rest| rest.trim().trim_end_matches(" kB").trim().parse().ok())
        .unwrap_or(0)
}

fn mib(bytes: impl Into<f64>) -> String {
    format!("{:.2} MiB", bytes.into() / f64::from(1 << 20))
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

/// `{"input_ids":[...]}` with pseudo-random token ids below Qwen's vocab size:
/// a realistic, deterministic body that compresses about 2-3x.
fn json_payload(target: usize) -> Vec<u8> {
    let mut body = Vec::with_capacity(target + 32);
    body.extend_from_slice(br#"{"input_ids":["#);
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    while body.len() < target {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        write!(body, "{},", (x >> 33) % 151_936).unwrap();
    }
    body.pop();
    body.extend_from_slice(b"]}");
    body
}

/// A zstd frame that declares its content size (`known`), like Python's
/// one-shot `ZstdCompressor().compress()`, or not, like a streaming writer.
fn frame(payload: &[u8], known: bool) -> Vec<u8> {
    let mut encoder = zstd::stream::write::Encoder::new(Vec::new(), 3).unwrap();
    encoder.include_contentsize(known).unwrap();
    if known {
        encoder
            .set_pledged_src_size(Some(payload.len() as u64))
            .unwrap();
    }
    encoder.write_all(payload).unwrap();
    let frame = encoder.finish().unwrap();
    let declared = zstd::zstd_safe::get_frame_content_size(&frame)
        .ok()
        .flatten();
    assert_eq!(declared.is_some(), known);
    frame
}

/// A body read lazily, like a socket: each chunk is a view of `frame` (no
/// copy), handed out only when the server polls, and counted in `polled`.
fn lazy_body(frame: &Bytes, chunk: usize, polled: Arc<AtomicUsize>) -> Body {
    let chunks: Vec<Bytes> = (0..frame.len())
        .step_by(chunk)
        .map(|i| frame.slice(i..(i + chunk).min(frame.len())))
        .collect();
    with_length(
        Body::from_stream(futures::stream::iter(chunks).map(move |c| {
            polled.fetch_add(c.len(), Relaxed);
            Ok::<_, Infallible>(c)
        })),
        frame.len(),
    )
}

/// Declares `length` like hyper does for a request with `Content-Length`.
struct Sized {
    inner: Body,
    remaining: u64,
}

impl http_body::Body for Sized {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<http_body::Frame<Bytes>, axum::Error>>> {
        let poll = std::pin::Pin::new(&mut self.inner).poll_frame(cx);
        if let std::task::Poll::Ready(Some(Ok(frame))) = &poll
            && let Some(data) = frame.data_ref()
        {
            self.remaining -= data.len() as u64;
        }
        poll
    }

    fn size_hint(&self) -> http_body::SizeHint {
        http_body::SizeHint::with_exact(self.remaining)
    }
}

fn with_length(inner: Body, length: usize) -> Body {
    Body::new(Sized {
        inner,
        remaining: length as u64,
    })
}

fn budget_kib() -> u32 {
    std::env::var("ITEM9_BUDGET_MIB")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .map_or(BUFFERED_BUDGET_KIB, |mib| mib * 1024)
}

fn length_app(limits: Limits) -> Router {
    with_decompression(
        // Like app.rs:181: the `Bytes` extractor otherwise rejects bodies over 2 MiB.
        Router::new()
            .route(
                "/len",
                post(|body: Bytes| async move { body.len().to_string() }),
            )
            .layer(axum::extract::DefaultBodyLimit::disable()),
        limits,
    )
}

fn zstd_request(body: Body) -> Request {
    Request::post("/len")
        .header(COMPRESSION_HEADER, "zstd")
        .body(body)
        .unwrap()
}

// ---------------------------------------------------------------------------
// Measurements
// ---------------------------------------------------------------------------

/// Allocations of one `decode_zstd`, per decompressed size, with and without a
/// declared content size. Three runs each: glibc moves its mmap threshold
/// after the first large free, so the first run can differ.
#[test]
#[ignore = "measurement"]
fn measure_decode_allocations() {
    println!(
        "\n| decompressed | content size | compressed | run | allocs | reallocs | requested | moved reallocs | moved bytes | peak live | decode time |"
    );
    println!("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|");
    let mut event_log = Vec::new();
    for size_mib in [1usize, 16, 64, 128] {
        let payload = json_payload(size_mib << 20);
        for known in [true, false] {
            let compressed = frame(&payload, known);
            for run in 1..=3 {
                let base = start();
                let started = Instant::now();
                let body = decode_zstd(&compressed).unwrap();
                let elapsed = started.elapsed();
                let stats = stop(base);
                assert_eq!(body.len(), payload.len());
                assert_eq!(body.capacity(), body.len());
                drop(body);
                println!(
                    "| {} | {} | {} | {run} | {} | {} | {} | {} | {} | {} | {:.2?} |",
                    mib(payload.len() as f64),
                    if known { "known" } else { "unknown" },
                    mib(compressed.len() as f64),
                    stats.allocs,
                    stats.reallocs,
                    mib(stats.requested as f64),
                    stats.moved,
                    mib(stats.moved_bytes as f64),
                    mib(stats.peak as f64),
                    elapsed,
                );
                event_log.push((size_mib, known, run, stats.events));
            }
        }
    }
    println!("\nReallocs of at least 1 MiB (old -> new, time inside realloc, pointer moved):");
    for (size_mib, known, run, events) in event_log {
        let known = if known { "known" } else { "unknown" };
        let events: Vec<String> = events
            .iter()
            .map(|[old, new, nanos, moved]| {
                format!(
                    "{}->{} {:.1?}{}",
                    mib(*old as f64),
                    mib(*new as f64),
                    Duration::from_nanos(*nanos),
                    if *moved == 1 { " moved" } else { "" }
                )
            })
            .collect();
        println!("{size_mib} MiB {known} run {run}: [{}]", events.join(", "));
    }
}

/// While every permit is held, a queued request's body is still read to the
/// end: the queue holds the compressed bytes, not just the headers.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "measurement"]
async fn measure_queued_body_is_read() {
    let payload = json_payload(4 << 20);
    let compressed = Bytes::from(frame(&payload, true));
    let limits = Limits::new(1, budget_kib());
    let app = length_app(limits.clone());
    let held = limits.decompression().await.unwrap();

    let polled = Arc::new(AtomicUsize::new(0));
    let body = lazy_body(&compressed, 64 << 10, polled.clone());
    let started = Instant::now();
    let response = tokio::spawn(app.oneshot(zstd_request(body)));
    while polled.load(Relaxed) < compressed.len() && started.elapsed() < Duration::from_secs(1) {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    let read_after = started.elapsed();
    tokio::time::sleep(Duration::from_millis(200)).await;
    println!(
        "\npermit held: {} of {} compressed bytes polled ({:.1?} after the request started); request finished: {}",
        polled.load(Relaxed),
        compressed.len(),
        read_after,
        response.is_finished(),
    );

    drop(held);
    let response = response.await.unwrap().unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(body, payload.len().to_string());
}

/// N concurrent compressed bodies against P permits. Phase 1 holds every
/// permit, so whatever the heap gained is queued-body memory; phase 2 lets
/// them run and records the peak and each request's wait.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "measurement"]
async fn measure_concurrent_queued_memory() {
    let env = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let requests = env("ITEM9_N", 64);
    let permits = env("ITEM9_P", 4);
    let size_mib = env("ITEM9_MIB", 32);
    let known = env("ITEM9_KNOWN", 1) == 1;

    let payload = json_payload(size_mib << 20);
    let compressed = Bytes::from(frame(&payload, known));
    let limits = Limits::new(permits, budget_kib());
    let app = length_app(limits.clone());
    let held = limits
        .decompressions
        .clone()
        .acquire_many_owned(permits as u32)
        .await
        .unwrap();

    let rss_before = vm_rss_kib();
    let base = start();
    let polled = Arc::new(AtomicUsize::new(0));
    let mut tasks = Vec::with_capacity(requests);
    for _ in 0..requests {
        let request = zstd_request(lazy_body(&compressed, 64 << 10, polled.clone()));
        let app = app.clone();
        tasks.push(tokio::spawn(async move {
            let response = app.oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            Instant::now()
        }));
    }
    let total = requests * compressed.len();
    let started = Instant::now();
    while polled.load(Relaxed) < total && started.elapsed() < Duration::from_secs(5) {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    tokio::time::sleep(Duration::from_millis(100)).await;
    let queued_live = live_since(base);
    let queued_rss = vm_rss_kib().saturating_sub(rss_before);
    let queued_polled = polled.load(Relaxed);

    let released = Instant::now();
    drop(held);
    let mut waits = Vec::with_capacity(requests);
    for task in tasks {
        waits.push(task.await.unwrap().duration_since(released));
    }
    let wall = released.elapsed();
    let stats = stop(base);
    waits.sort();

    println!(
        "\nbudget {} MiB, N = {requests} requests, P = {permits} permits, {} decompressed, {} compressed ({} content size)",
        budget_kib() / 1024,
        mib(payload.len() as f64),
        mib(compressed.len() as f64),
        if known { "known" } else { "unknown" },
    );
    println!(
        "phase 1, all permits held: polled {} of {} ({}), heap +{}, RSS +{}",
        mib(queued_polled as f64),
        mib(total as f64),
        if queued_polled == total {
            "every body read"
        } else {
            "not every body read"
        },
        mib(queued_live as f64),
        mib(queued_rss as f64 * 1024.0),
    );
    println!(
        "phase 2, permits released: all done in {wall:.2?}; completion after release p50 {:.2?}, p90 {:.2?}, max {:.2?}; peak heap over both phases +{}; {} allocs, {} reallocs, {} requested",
        waits[waits.len() / 2],
        waits[waits.len() * 9 / 10],
        waits[waits.len() - 1],
        mib(stats.peak as f64),
        stats.allocs,
        stats.reallocs,
        mib(stats.requested as f64),
    );
}

/// K slow uploads arrive first, then M fast requests, against P permits. The
/// fast requests' latency shows what a slow uploader costs everyone else.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "measurement"]
async fn measure_slow_uploads() {
    let env = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let slow = env("ITEM9_SLOW", 4);
    let fast = env("ITEM9_FAST", 16);
    let permits = env("ITEM9_P", 4);
    let upload_ms = env("ITEM9_UPLOAD_MS", 1000) as u64;

    let slow_frame = Bytes::from(frame(&json_payload(4 << 20), true));
    let fast_frame = Bytes::from(frame(&json_payload(1 << 20), true));
    let app = length_app(Limits::new(permits, budget_kib()));

    // 20 chunks, one every upload_ms / 20: a client on a slow link.
    let chunk = slow_frame.len().div_ceil(20);
    let mut tasks = Vec::new();
    for _ in 0..slow {
        let chunks: Vec<Bytes> = (0..slow_frame.len())
            .step_by(chunk)
            .map(|i| slow_frame.slice(i..(i + chunk).min(slow_frame.len())))
            .collect();
        let body = with_length(
            Body::from_stream(futures::stream::iter(chunks).then(move |c| async move {
                tokio::time::sleep(Duration::from_millis(upload_ms / 20)).await;
                Ok::<_, Infallible>(c)
            })),
            slow_frame.len(),
        );
        let app = app.clone();
        let started = Instant::now();
        tasks.push((
            true,
            tokio::spawn(async move {
                assert_eq!(
                    app.oneshot(zstd_request(body)).await.unwrap().status(),
                    StatusCode::OK
                );
                started.elapsed()
            }),
        ));
    }
    tokio::time::sleep(Duration::from_millis(20)).await;
    for _ in 0..fast {
        let body = lazy_body(&fast_frame, 64 << 10, Arc::new(AtomicUsize::new(0)));
        let app = app.clone();
        let started = Instant::now();
        tasks.push((
            false,
            tokio::spawn(async move {
                assert_eq!(
                    app.oneshot(zstd_request(body)).await.unwrap().status(),
                    StatusCode::OK
                );
                started.elapsed()
            }),
        ));
    }
    let (mut slow_lat, mut fast_lat) = (Vec::new(), Vec::new());
    for (is_slow, task) in tasks {
        let elapsed = task.await.unwrap();
        if is_slow {
            slow_lat.push(elapsed)
        } else {
            fast_lat.push(elapsed)
        }
    }
    fast_lat.sort();
    slow_lat.sort();
    println!(
        "\nK = {slow} slow uploads ({} over {upload_ms} ms), M = {fast} fast requests ({}), P = {permits}",
        mib(slow_frame.len() as f64),
        mib(fast_frame.len() as f64),
    );
    println!(
        "fast requests: p50 {:.1?}, max {:.1?}; slow requests: max {:.1?}",
        fast_lat[fast_lat.len() / 2],
        fast_lat[fast_lat.len() - 1],
        slow_lat.last().copied().unwrap_or_default(),
    );
}

/// Heap held by decoding jobs, with no queue: `jobs` requests at once against
/// as many permits, each body read lazily. The window opens after the
/// compressed frame exists, so the peak counts only what a request allocates:
/// its contiguous compressed copy and its decoder output.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "measurement"]
async fn measure_active_job_memory() {
    let permits: usize = std::env::var("ITEM9_P")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let size_mib: usize = std::env::var("ITEM9_MIB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let payload = json_payload(size_mib << 20);
    println!(
        "\n| content size | jobs at once | run | compressed per job | decompressed per job | peak heap | peak per job | compressed + decompressed |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|");
    for known in [true, false] {
        let compressed = Bytes::from(frame(&payload, known));
        for jobs in [1, permits] {
            for run in 1..=3 {
                let app = length_app(Limits::new(jobs, BUFFERED_BUDGET_KIB));
                let base = start();
                let polled = Arc::new(AtomicUsize::new(0));
                let tasks: Vec<_> = (0..jobs)
                    .map(|_| {
                        let request =
                            zstd_request(lazy_body(&compressed, 64 << 10, polled.clone()));
                        let app = app.clone();
                        tokio::spawn(async move {
                            let response = app.oneshot(request).await.unwrap();
                            assert_eq!(response.status(), StatusCode::OK);
                            axum::body::to_bytes(response.into_body(), usize::MAX)
                                .await
                                .unwrap()
                        })
                    })
                    .collect();
                for task in tasks {
                    let length = task.await.unwrap();
                    assert_eq!(length, payload.len().to_string().as_bytes());
                }
                let stats = stop(base);
                println!(
                    "| {} | {jobs} | {run} | {} | {} | {} | {} | {} |",
                    if known { "known" } else { "unknown" },
                    mib(compressed.len() as f64),
                    mib(payload.len() as f64),
                    mib(stats.peak as f64),
                    mib(stats.peak as f64 / jobs as f64),
                    mib((compressed.len() + payload.len()) as f64),
                );
            }
        }
    }
}
