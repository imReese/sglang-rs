use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};

use serde_json::Value;
use tokio::sync::oneshot;

use sglang_srt::cli::ServerArgs;
use sglang_srt::http::serve_http_router_with_shutdown;
use sglang_srt::pd_bootstrap::PrefillBootstrapService;
use sglang_srt::server::{
    build_bootstrap_http_router_service, build_bootstrap_mooncake_prefill_http_router_service,
    build_bootstrap_pd_http_router_service, build_bootstrap_prefill_http_router_service,
    launch_http_server_with_shutdown,
};
use sglang_srt::transfer::{
    DecodeBootstrapRegistry, MooncakeBatchId, MooncakeBatchReleaser, MooncakeError,
    MooncakeKvCacheLayout, MooncakeKvCacheTransferExecutor, MooncakeTransferRequest,
    MooncakeTransferStatus, MooncakeTransferStatusCode, MooncakeTransferStatusReader,
    MooncakeTransferSubmitter, MooncakeTransferTarget,
};
use sglang_srt::types::BootstrapRoom;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_server_accepts_model_and_generate_requests() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-http",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_bootstrap_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let models = get_json_with_retry(addr, "/v1/models").await;
    let generated = post_json_with_retry(
        addr,
        "/generate",
        r#"{"text":"hello","sampling_params":{"max_new_tokens":1}}"#,
    )
    .await;

    assert_eq!(models["data"][0]["id"], "glm-http");
    assert_eq!(generated["text"], " ");
    assert_eq!(generated["usage"]["prompt_tokens"], 5);
    assert_eq!(generated["usage"]["completion_tokens"], 1);

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_server_rejects_disaggregated_generate_without_transfer_runtime() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_bootstrap_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let response = request_raw_with_retry(
        addr,
        "POST",
        "/generate",
        Some(r#"{"text":"hello","bootstrap_host":"10.0.0.8","bootstrap_port":8200,"bootstrap_room":77}"#),
    )
    .await;

    assert!(response.starts_with("HTTP/1.1 501"));
    assert!(response.contains("PD transfer-enabled runtime"));

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_prefill_server_accepts_disaggregated_generate_requests() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-prefill-http",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "fake",
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_bootstrap_prefill_http_router_service(&args);
    let inspected_service = service.clone();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let generated = post_json_with_retry(
        addr,
        "/generate",
        r#"{"request_id":"http-pd-prefill","text":"hello","sampling_params":{"max_new_tokens":1},"bootstrap_host":"10.0.0.8","bootstrap_port":8200,"bootstrap_room":77}"#,
    )
    .await;

    assert_eq!(generated["request_id"], "http-pd-prefill");
    assert_eq!(generated["text"], " ");

    let runtime = inspected_service
        .runtime()
        .lock()
        .expect("runtime lock should be held");
    let worker = runtime.engine().scheduler().worker();
    let summary = worker
        .last_transfer_summary()
        .expect("PD prefill request should record transfer summary");
    assert_eq!(summary.submitted_spans(), 1);
    assert_eq!(worker.transfer_executor().transferred_rooms(), &[77]);

    drop(runtime);
    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_pd_server_polls_async_transfer_before_decode() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-pd-http",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-decode-polling-interval",
        "1",
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let transfer_executor = MooncakeKvCacheTransferExecutor::new(
        RecordingMooncakeBackend::completed(),
        MooncakeKvCacheLayout {
            source_base_addr: 0x3000,
            page_size_bytes: 64,
            target_base_offset: 0,
        },
        MooncakeTransferTarget { target_id: 17 },
    );
    let service = build_bootstrap_pd_http_router_service(
        &args,
        DecodeBootstrapRegistry::default(),
        transfer_executor,
    );
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let generated = post_json_with_retry(
        addr,
        "/generate",
        r#"{"request_id":"http-pd-mooncake","text":"hi","sampling_params":{"max_new_tokens":2},"bootstrap_host":"10.0.0.8","bootstrap_port":8200,"bootstrap_room":41}"#,
    )
    .await;

    assert_eq!(generated["request_id"], "http-pd-mooncake");
    assert_eq!(generated["usage"]["completion_tokens"], 2);

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mooncake_prefill_http_uses_bootstrap_kv_layout_for_transfer() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-mooncake-prefill-http",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-decode-polling-interval",
        "1",
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let bootstrap_service = PrefillBootstrapService::default();
    {
        let mut state = bootstrap_service
            .state()
            .lock()
            .expect("bootstrap state lock should be held");
        state
            .ingest_mooncake_bootstrap_frame(&kv_args_frame("session-a", &[0x9000], 128))
            .expect("KVArgs frame should parse");
        state
            .ingest_mooncake_bootstrap_frame(&transfer_metadata_frame(34, "session-a", &[4, 5]))
            .expect("transfer metadata frame should parse");
    }
    let transfer_executor = MooncakeKvCacheTransferExecutor::new(
        RecordingMooncakeBackend::completed(),
        MooncakeKvCacheLayout {
            source_base_addr: 0x2000,
            page_size_bytes: 128,
            target_base_offset: 0xdead_0000,
        },
        MooncakeTransferTarget { target_id: 7 },
    );
    let service = build_bootstrap_mooncake_prefill_http_router_service(
        &args,
        bootstrap_service,
        transfer_executor,
    );
    let inspected_service = service.clone();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let generated = post_json_with_retry(
        addr,
        "/generate",
        r#"{"request_id":"http-pd-mooncake-prefill","text":"hi","sampling_params":{"max_new_tokens":2},"bootstrap_host":"10.0.0.8","bootstrap_port":8200,"bootstrap_room":34}"#,
    )
    .await;

    assert_eq!(generated["request_id"], "http-pd-mooncake-prefill");
    assert_eq!(generated["usage"]["completion_tokens"], 2);

    let runtime = inspected_service
        .runtime()
        .lock()
        .expect("runtime lock should be held");
    let worker = runtime.engine().scheduler().worker();
    let submitted_requests = &worker
        .transfer_executor()
        .inner()
        .submitter()
        .submitted_requests;
    assert_eq!(submitted_requests.len(), 1);
    assert_eq!(submitted_requests[0].len(), 2);
    assert_eq!(submitted_requests[0][0].target_id, 7);
    assert_eq!(submitted_requests[0][0].target_offset, 0x9000 + 4 * 128);
    assert_eq!(submitted_requests[0][1].target_id, 7);
    assert_eq!(submitted_requests[0][1].target_offset, 0x9000 + 5 * 128);

    drop(runtime);
    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prefill_http_launch_starts_main_and_bootstrap_listeners() {
    let http_addr = unused_local_addr();
    let bootstrap_addr = unused_local_addr();
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--host",
        "127.0.0.1",
        "--port",
        &http_addr.port().to_string(),
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-bootstrap-port",
        &bootstrap_addr.port().to_string(),
    ])
    .expect("args should parse");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        launch_http_server_with_shutdown(args, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let health = get_json_with_retry(http_addr, "/health").await;
    let bootstrap_health = request_raw_with_retry(bootstrap_addr, "GET", "/health", None).await;
    let bootstrap_route = request_raw_with_retry(
        bootstrap_addr,
        "GET",
        "/route?prefill_dp_rank=-1&prefill_cp_rank=-1&target_tp_rank=-1&target_pp_rank=-1",
        None,
    )
    .await;

    assert_eq!(health["healthy"], true);
    assert!(bootstrap_health.ends_with("\r\n\r\nOK"));
    assert!(bootstrap_route.starts_with("HTTP/1.1 503"));

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("servers should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prefill_http_launch_registers_mooncake_zmq_routes() {
    let http_addr = unused_local_addr();
    let bootstrap_addr = unused_local_addr();
    let zmq_ports = unused_contiguous_local_ports(2);
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--host",
        "127.0.0.1",
        "--port",
        &http_addr.port().to_string(),
        "--tp-size",
        "2",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-bootstrap-port",
        &bootstrap_addr.port().to_string(),
        "--disaggregation-zmq-ports",
        &format!("{}-{}", zmq_ports[0], zmq_ports[1]),
    ])
    .expect("args should parse");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        launch_http_server_with_shutdown(args, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let topology = get_json_with_retry(
        bootstrap_addr,
        "/route?prefill_dp_rank=-1&prefill_cp_rank=-1&target_tp_rank=-1&target_pp_rank=-1",
    )
    .await;
    let tp0 = get_json_with_retry(
        bootstrap_addr,
        "/route?prefill_dp_rank=0&prefill_cp_rank=0&target_tp_rank=0&target_pp_rank=0",
    )
    .await;
    let tp1 = get_json_with_retry(
        bootstrap_addr,
        "/route?prefill_dp_rank=0&prefill_cp_rank=0&target_tp_rank=1&target_pp_rank=0",
    )
    .await;

    assert_eq!(topology["attn_tp_size"], 2);
    assert_eq!(topology["dp_size"], 1);
    assert_eq!(tp0["rank_ip"], "127.0.0.1");
    assert_eq!(tp0["rank_port"], zmq_ports[0]);
    assert_eq!(tp1["rank_ip"], "127.0.0.1");
    assert_eq!(tp1["rank_port"], zmq_ports[1]);

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("servers should stop cleanly");
}

fn kv_args_frame(session_id: &str, dst_kv_ptrs: &[u64], dst_kv_item_len: usize) -> Vec<Vec<u8>> {
    vec![
        b"None".to_vec(),
        b"10.0.0.9".to_vec(),
        b"41001".to_vec(),
        session_id.as_bytes().to_vec(),
        pack_u64s(dst_kv_ptrs),
        pack_u64s(&[]),
        pack_list_of_buffers(&[]),
        b"1".to_vec(),
        b"8".to_vec(),
        dst_kv_item_len.to_string().into_bytes(),
    ]
}

fn transfer_metadata_frame(
    room: BootstrapRoom,
    session_id: &str,
    dst_kv_indices: &[i32],
) -> Vec<Vec<u8>> {
    vec![
        room.to_string().into_bytes(),
        b"10.0.0.9".to_vec(),
        b"41001".to_vec(),
        session_id.as_bytes().to_vec(),
        pack_i32s(dst_kv_indices),
        b"11".to_vec(),
        pack_list_of_buffers(&[]),
        b"1".to_vec(),
        b"64".to_vec(),
    ]
}

fn pack_u64s(values: &[u64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn pack_i32s(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn pack_list_of_buffers(buffers: &[Vec<u8>]) -> Vec<u8> {
    let mut packed = Vec::new();
    packed.extend_from_slice(&(buffers.len() as u64).to_le_bytes());
    for buffer in buffers {
        packed.extend_from_slice(&(buffer.len() as u64).to_le_bytes());
        packed.extend_from_slice(buffer);
    }
    packed
}

fn unused_local_addr() -> SocketAddr {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("ephemeral port should bind");
    listener
        .local_addr()
        .expect("ephemeral listener should have local addr")
}

fn unused_contiguous_local_ports(count: u16) -> Vec<u16> {
    for _ in 0..100 {
        let first = unused_local_addr().port();
        let Some(last) = first.checked_add(count - 1) else {
            continue;
        };
        let listeners = (first..=last)
            .map(|port| TcpListener::bind(("127.0.0.1", port)))
            .collect::<Result<Vec<_>, _>>();
        if let Ok(listeners) = listeners {
            drop(listeners);
            return (first..=last).collect();
        }
    }
    panic!("contiguous local ports should be available");
}

struct RecordingMooncakeBackend {
    status: MooncakeTransferStatusCode,
    submitted_requests: Vec<Vec<MooncakeTransferRequest>>,
    freed_batches: Vec<MooncakeBatchId>,
}

impl RecordingMooncakeBackend {
    fn completed() -> Self {
        Self {
            status: MooncakeTransferStatusCode::Completed,
            submitted_requests: Vec::new(),
            freed_batches: Vec::new(),
        }
    }
}

impl MooncakeTransferSubmitter for RecordingMooncakeBackend {
    fn submit_transfer(
        &mut self,
        requests: &mut [MooncakeTransferRequest],
    ) -> Result<MooncakeBatchId, MooncakeError> {
        self.submitted_requests.push(requests.to_vec());
        Ok(500 + self.submitted_requests.len() as MooncakeBatchId - 1)
    }
}

impl MooncakeTransferStatusReader for RecordingMooncakeBackend {
    fn transfer_status(
        &mut self,
        _batch_id: MooncakeBatchId,
        _task_id: usize,
    ) -> Result<MooncakeTransferStatus, MooncakeError> {
        Ok(MooncakeTransferStatus {
            status: self.status as i32,
            transferred_bytes: 0,
        })
    }
}

impl MooncakeBatchReleaser for RecordingMooncakeBackend {
    fn free_batch(&mut self, batch_id: MooncakeBatchId) -> Result<(), MooncakeError> {
        self.freed_batches.push(batch_id);
        Ok(())
    }
}

async fn get_json_with_retry(addr: SocketAddr, path: &str) -> Value {
    request_json_with_retry(addr, "GET", path, None).await
}

async fn post_json_with_retry(addr: SocketAddr, path: &str, body: &'static str) -> Value {
    request_json_with_retry(addr, "POST", path, Some(body)).await
}

async fn request_json_with_retry(
    addr: SocketAddr,
    method: &'static str,
    path: &str,
    body: Option<&'static str>,
) -> Value {
    let mut last_error = None;

    for _ in 0..20 {
        match request_json(addr, method, path, body).await {
            Ok(value) => return value,
            Err(error) => {
                last_error = Some(error);
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        }
    }

    panic!(
        "HTTP client should connect to test server: {}",
        last_error.expect("at least one connection attempt should run")
    );
}

async fn request_raw_with_retry(
    addr: SocketAddr,
    method: &'static str,
    path: &str,
    body: Option<&'static str>,
) -> String {
    let mut last_error = None;

    for _ in 0..20 {
        match request_raw(addr, method, path, body).await {
            Ok(value) => return value,
            Err(error) => {
                last_error = Some(error);
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        }
    }

    panic!(
        "HTTP client should connect to test server: {}",
        last_error.expect("at least one connection attempt should run")
    );
}

async fn request_json(
    addr: SocketAddr,
    method: &'static str,
    path: &str,
    body: Option<&'static str>,
) -> Result<Value, std::io::Error> {
    let path = path.to_string();
    let response = request_raw(addr, method, &path, body).await?;
    let (_, body) = response
        .split_once("\r\n\r\n")
        .expect("HTTP response should include headers");
    serde_json::from_str(body).map_err(std::io::Error::other)
}

async fn request_raw(
    addr: SocketAddr,
    method: &'static str,
    path: &str,
    body: Option<&'static str>,
) -> Result<String, std::io::Error> {
    let path = path.to_string();
    tokio::task::spawn_blocking(move || {
        let mut stream = TcpStream::connect(addr)?;
        let body = body.unwrap_or_default();
        let request = format!(
            "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(request.as_bytes())?;
        let mut response = String::new();
        stream.read_to_string(&mut response)?;
        Ok(response)
    })
    .await
    .expect("blocking HTTP request should join")
}
