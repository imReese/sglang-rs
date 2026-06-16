//! End-to-end PD router dispatch against real Rust SRT HTTP workers.
//!
//! The mock-worker tests pin request shaping. This file goes one step
//! further: it starts actual `sglang-srt` prefill/decode HTTP workers
//! and drives the router's PD fan-out through reqwest into those workers.

#[cfg(feature = "mooncake-link")]
use std::io::{Read, Write};
#[cfg(feature = "mooncake-link")]
use std::net::TcpStream;
use std::net::{SocketAddr, TcpListener};
#[cfg(feature = "mooncake-link")]
use std::sync::Arc;
#[cfg(feature = "mooncake-link")]
use std::time::Duration;

#[cfg(feature = "mooncake-link")]
use axum::body::Body;
#[cfg(feature = "mooncake-link")]
use axum::http::{Request, StatusCode};
#[cfg(feature = "mooncake-link")]
use http_body_util::BodyExt;
#[cfg(feature = "mooncake-link")]
use serde_json::json;
#[cfg(feature = "mooncake-link")]
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, DiscoveryConfig, ModelConfig, ObservabilityConfig,
    PolicyKind, ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
#[cfg(feature = "mooncake-link")]
use sgl_router::discovery::{DiscoveryEvent, ModelId, WorkerId, WorkerMode, WorkerSpec};
#[cfg(feature = "mooncake-link")]
use sgl_router::policies::factory::build_registry_with_defaults;
#[cfg(feature = "mooncake-link")]
use sgl_router::proxy::Proxy;
#[cfg(feature = "mooncake-link")]
use sgl_router::server::app::build_router;
#[cfg(feature = "mooncake-link")]
use sgl_router::server::app_context::AppContext;
#[cfg(feature = "mooncake-link")]
use sgl_router::tokenizer::TokenizerRegistry;
#[cfg(feature = "mooncake-link")]
use sgl_router::workers::{manager, WorkerRegistry};
use sglang_srt::cli::ServerArgs;
use sglang_srt::server::launch_http_server_with_shutdown;
#[cfg(feature = "mooncake-link")]
use tokio::sync::mpsc;
use tokio::sync::oneshot;
#[cfg(feature = "mooncake-link")]
use tower::ServiceExt;

#[cfg(feature = "mooncake-link")]
fn config() -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        models: vec![ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: PolicyKind::RoundRobin,
            circuit_breaker: None,
            cache_aware: None,
        }],
        discovery: DiscoveryConfig {
            backend: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
                urls: vec!["http://placeholder:0".into()],
            }),
        },
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
    }
}

#[cfg(feature = "mooncake-link")]
async fn build_ctx(prefill_addr: SocketAddr, decode_addr: SocketAddr) -> Arc<AppContext> {
    let cfg = config();
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    register_real_srt_workers_with_manager(prefill_addr, decode_addr, Arc::clone(&registry)).await;

    let model_id = ModelId("tiny".into());
    assert_eq!(
        registry
            .workers_for_mode(&model_id, WorkerMode::Prefill)
            .len(),
        1,
        "manager should classify the real SRT prefill worker from /server_info"
    );
    assert_eq!(
        registry
            .workers_for_mode(&model_id, WorkerMode::Decode)
            .len(),
        1,
        "manager should classify the real SRT decode worker from /server_info"
    );

    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

#[cfg(feature = "mooncake-link")]
async fn register_real_srt_workers_with_manager(
    prefill_addr: SocketAddr,
    decode_addr: SocketAddr,
    registry: Arc<WorkerRegistry>,
) {
    let (tx, rx) = mpsc::channel(8);
    let manager = tokio::spawn(manager::run(rx, Arc::clone(&registry)));

    for (id, addr) in [("srt-prefill", prefill_addr), ("srt-decode", decode_addr)] {
        tx.send(DiscoveryEvent::Added(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{addr}"),
            // Static discovery seeds a plain/empty worker; the manager
            // must resolve the real mode and model from /server_info.
            mode: WorkerMode::Plain,
            model_ids: Vec::new(),
            bootstrap_port: None,
        }))
        .await
        .expect("worker discovery event should send");
    }
    drop(tx);
    manager
        .await
        .expect("worker manager task should join after discovery sender closes");
}

#[cfg(not(feature = "mooncake-link"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn real_rust_srt_mooncake_workers_reject_dummy_runtime_without_transferable_kv_memory() {
    let prefill_addr = unused_local_addr();
    let bootstrap_addr = unused_local_addr();
    let prefill_zmq_addr = unused_local_addr();
    let decode_addr = unused_local_addr();

    let prefill_args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny",
        "--host",
        "127.0.0.1",
        "--port",
        &prefill_addr.port().to_string(),
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-bootstrap-port",
        &bootstrap_addr.port().to_string(),
        "--disaggregation-zmq-ports",
        &prefill_zmq_addr.port().to_string(),
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("prefill args should parse");
    let decode_args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny",
        "--host",
        "127.0.0.1",
        "--port",
        &decode_addr.port().to_string(),
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--kv-cache-dtype",
        "bfloat16",
        "--kv-cache-num-layers",
        "2",
        "--kv-cache-kv-heads",
        "1",
        "--kv-cache-head-dim",
        "8",
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("decode args should parse");

    let (prefill_shutdown_tx, prefill_shutdown_rx) = oneshot::channel::<()>();
    let prefill_server = tokio::spawn(async move {
        launch_http_server_with_shutdown(prefill_args, async move {
            let _ = prefill_shutdown_rx.await;
        })
        .await
    });
    let (decode_shutdown_tx, decode_shutdown_rx) = oneshot::channel::<()>();
    let decode_server = tokio::spawn(async move {
        launch_http_server_with_shutdown(decode_args, async move {
            let _ = decode_shutdown_rx.await;
        })
        .await
    });

    let prefill_error = prefill_server
        .await
        .expect("prefill server task should join")
        .expect_err("dummy prefill worker should reject Mooncake PD startup");
    let decode_error = decode_server
        .await
        .expect("decode server task should join")
        .expect_err("dummy decode worker should reject Mooncake PD startup");

    assert!(
        prefill_error
            .to_string()
            .contains("does not expose transferable Mooncake KV memory"),
        "{prefill_error}"
    );
    assert!(
        decode_error
            .to_string()
            .contains("does not expose transferable Mooncake KV memory"),
        "{decode_error}"
    );
    drop(prefill_shutdown_tx);
    drop(decode_shutdown_tx);
}

#[cfg(feature = "mooncake-link")]
async fn wait_for_health(addr: SocketAddr) {
    let mut last_error = None;
    for _ in 0..100 {
        match request_raw(addr, "GET", "/health", None).await {
            Ok(response) if response.starts_with("HTTP/1.1 200") => return,
            Ok(response) => last_error = Some(response),
            Err(error) => last_error = Some(error.to_string()),
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!(
        "SRT worker at {addr} should become healthy: {}",
        last_error.unwrap_or_else(|| "no attempts made".to_string())
    );
}

fn unused_local_addr() -> SocketAddr {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("ephemeral port should bind");
    listener
        .local_addr()
        .expect("ephemeral listener should have local addr")
}

#[cfg(feature = "mooncake-link")]
async fn request_raw(
    addr: SocketAddr,
    method: &'static str,
    path: &'static str,
    body: Option<&'static str>,
) -> Result<String, std::io::Error> {
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
