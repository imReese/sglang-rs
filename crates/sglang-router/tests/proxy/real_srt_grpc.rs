//! End-to-end router dispatch against a real Rust SRT gRPC worker.
//!
//! This pins the next transport step after gRPC worker introspection:
//! the router must be able to forward OpenAI-compatible HTTP traffic to
//! a `grpc://` worker, not only register it.

use std::net::{SocketAddr, TcpListener};
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::json;
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, DiscoveryConfig, ModelConfig, ObservabilityConfig,
    PolicyKind, ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{DiscoveryEvent, ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::{manager, WorkerRegistry};
use sglang_srt::cli::ServerArgs;
use sglang_srt::proto::sglang::runtime::v1::sglang_service_client::SglangServiceClient;
use sglang_srt::proto::sglang::runtime::v1::HealthCheckRequest;
use sglang_srt::server::launch_grpc_server_with_shutdown;
use tokio::sync::{mpsc, oneshot};
use tower::ServiceExt;

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
                urls: vec!["grpc://placeholder:0".into()],
            }),
        },
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
    }
}

fn build_ctx_with_grpc_worker(addr: SocketAddr) -> Arc<AppContext> {
    let cfg = config();
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    registry
        .add(WorkerSpec {
            id: WorkerId("grpc-worker".into()),
            url: format!("grpc://{addr}"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        })
        .expect("grpc worker should register");
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

async fn build_ctx_with_grpc_pd_workers(
    prefill_addr: SocketAddr,
    decode_addr: SocketAddr,
) -> Arc<AppContext> {
    let cfg = config();
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    register_real_srt_grpc_workers_with_manager(prefill_addr, decode_addr, Arc::clone(&registry))
        .await;

    let model_id = ModelId("tiny".into());
    assert_eq!(
        registry
            .workers_for_mode(&model_id, WorkerMode::Prefill)
            .len(),
        1,
        "manager should classify the real SRT gRPC prefill worker from GetServerInfo"
    );
    assert_eq!(
        registry
            .workers_for_mode(&model_id, WorkerMode::Decode)
            .len(),
        1,
        "manager should classify the real SRT gRPC decode worker from GetServerInfo"
    );

    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

async fn register_real_srt_grpc_workers_with_manager(
    prefill_addr: SocketAddr,
    decode_addr: SocketAddr,
    registry: Arc<WorkerRegistry>,
) {
    let (tx, rx) = mpsc::channel(8);
    let manager = tokio::spawn(manager::run(rx, Arc::clone(&registry)));

    for (id, addr) in [("grpc-prefill", prefill_addr), ("grpc-decode", decode_addr)] {
        tx.send(DiscoveryEvent::Added(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("grpc://{addr}"),
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_chat_completions_reaches_real_rust_srt_grpc_worker() {
    let addr = unused_local_addr();
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny",
        "--host",
        &addr.ip().to_string(),
        "--port",
        &addr.port().to_string(),
        "--grpc-mode",
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("gRPC SRT args should parse");

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(launch_grpc_server_with_shutdown(args, async move {
        let _ = shutdown_rx.await;
    }));
    wait_for_grpc_health(addr).await;

    let app = build_router(build_ctx_with_grpc_worker(addr));
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            }))
            .unwrap(),
        ))
        .unwrap();

    let response = app.oneshot(request).await.expect("router should respond");
    assert_eq!(response.status(), StatusCode::OK);
    let body = response
        .into_body()
        .collect()
        .await
        .expect("router response body should collect")
        .to_bytes();
    let body: serde_json::Value =
        serde_json::from_slice(&body).expect("response should be OpenAI chat JSON");
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "tiny");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");

    shutdown_tx
        .send(())
        .expect("gRPC worker should still be running");
    server
        .await
        .expect("gRPC server task should join")
        .expect("gRPC server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_streaming_chat_completions_reaches_real_rust_srt_grpc_worker() {
    let addr = unused_local_addr();
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny",
        "--host",
        &addr.ip().to_string(),
        "--port",
        &addr.port().to_string(),
        "--grpc-mode",
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("gRPC SRT args should parse");

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(launch_grpc_server_with_shutdown(args, async move {
        let _ = shutdown_rx.await;
    }));
    wait_for_grpc_health(addr).await;

    let app = build_router(build_ctx_with_grpc_worker(addr));
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 2,
                "stream": true,
            }))
            .unwrap(),
        ))
        .unwrap();

    let response = app.oneshot(request).await.expect("router should respond");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    let body = response
        .into_body()
        .collect()
        .await
        .expect("router SSE body should collect")
        .to_bytes();
    let events = crate::common::streaming::parse_sse_data(&body);
    assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
    let chunks = events
        .iter()
        .filter(|event| event.as_str() != "[DONE]")
        .map(|event| serde_json::from_str::<serde_json::Value>(event))
        .collect::<Result<Vec<_>, _>>()
        .expect("SSE data chunks should be JSON");
    assert!(
        chunks
            .iter()
            .any(|chunk| chunk["object"] == "chat.completion.chunk"),
        "expected at least one OpenAI chat completion chunk, got {chunks:?}"
    );

    shutdown_tx
        .send(())
        .expect("gRPC worker should still be running");
    server
        .await
        .expect("gRPC server task should join")
        .expect("gRPC server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_completions_reaches_real_rust_srt_grpc_worker() {
    let addr = unused_local_addr();
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny",
        "--host",
        &addr.ip().to_string(),
        "--port",
        &addr.port().to_string(),
        "--grpc-mode",
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("gRPC SRT args should parse");

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(launch_grpc_server_with_shutdown(args, async move {
        let _ = shutdown_rx.await;
    }));
    wait_for_grpc_health(addr).await;

    let app = build_router(build_ctx_with_grpc_worker(addr));
    let request = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny",
                "prompt": "hi",
                "max_tokens": 2,
            }))
            .unwrap(),
        ))
        .unwrap();

    let response = app.oneshot(request).await.expect("router should respond");
    assert_eq!(response.status(), StatusCode::OK);
    let body = response
        .into_body()
        .collect()
        .await
        .expect("router response body should collect")
        .to_bytes();
    let body: serde_json::Value =
        serde_json::from_slice(&body).expect("response should be OpenAI completion JSON");
    assert_eq!(body["object"], "text_completion");
    assert_eq!(body["model"], "tiny");
    assert_eq!(body["choices"][0]["text"], "  ");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");

    shutdown_tx
        .send(())
        .expect("gRPC worker should still be running");
    server
        .await
        .expect("gRPC server task should join")
        .expect("gRPC server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_streaming_completions_reaches_real_rust_srt_grpc_worker() {
    let addr = unused_local_addr();
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny",
        "--host",
        &addr.ip().to_string(),
        "--port",
        &addr.port().to_string(),
        "--grpc-mode",
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("gRPC SRT args should parse");

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(launch_grpc_server_with_shutdown(args, async move {
        let _ = shutdown_rx.await;
    }));
    wait_for_grpc_health(addr).await;

    let app = build_router(build_ctx_with_grpc_worker(addr));
    let request = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny",
                "prompt": "hi",
                "max_tokens": 2,
                "stream": true,
            }))
            .unwrap(),
        ))
        .unwrap();

    let response = app.oneshot(request).await.expect("router should respond");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    let body = response
        .into_body()
        .collect()
        .await
        .expect("router SSE body should collect")
        .to_bytes();
    let events = crate::common::streaming::parse_sse_data(&body);
    assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
    let chunks = events
        .iter()
        .filter(|event| event.as_str() != "[DONE]")
        .map(|event| serde_json::from_str::<serde_json::Value>(event))
        .collect::<Result<Vec<_>, _>>()
        .expect("SSE data chunks should be JSON");
    assert!(
        chunks
            .iter()
            .any(|chunk| chunk["object"] == "text_completion"),
        "expected at least one OpenAI completion chunk, got {chunks:?}"
    );

    shutdown_tx
        .send(())
        .expect("gRPC worker should still be running");
    server
        .await
        .expect("gRPC server task should join")
        .expect("gRPC server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn router_pd_chat_reaches_real_rust_srt_grpc_mooncake_workers() {
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
        &prefill_addr.ip().to_string(),
        "--port",
        &prefill_addr.port().to_string(),
        "--grpc-mode",
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
        &decode_addr.ip().to_string(),
        "--port",
        &decode_addr.port().to_string(),
        "--grpc-mode",
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

    let (prefill_shutdown_tx, prefill_shutdown_rx) = oneshot::channel();
    let prefill_server = tokio::spawn(launch_grpc_server_with_shutdown(prefill_args, async move {
        let _ = prefill_shutdown_rx.await;
    }));
    let (decode_shutdown_tx, decode_shutdown_rx) = oneshot::channel();
    let decode_server = tokio::spawn(launch_grpc_server_with_shutdown(decode_args, async move {
        let _ = decode_shutdown_rx.await;
    }));

    wait_for_grpc_health(prefill_addr).await;
    wait_for_grpc_health(decode_addr).await;

    let app = build_router(build_ctx_with_grpc_pd_workers(prefill_addr, decode_addr).await);
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            }))
            .unwrap(),
        ))
        .unwrap();

    let response = app.oneshot(request).await.expect("router should respond");
    assert_eq!(
        response.status(),
        StatusCode::INTERNAL_SERVER_ERROR,
        "the default build should reach the unlinked Mooncake gRPC runtime, not fail in router dispatch"
    );
    let body = response
        .into_body()
        .collect()
        .await
        .expect("router response body should collect")
        .to_bytes();
    let body = std::str::from_utf8(&body).expect("router body should be UTF-8");
    assert!(
        body.contains(
            "mooncake transfer engine requires building sglang-srt with the mooncake-link feature"
        ),
        "router must have reached the real Rust SRT gRPC decode transfer runtime; body={body}"
    );

    prefill_shutdown_tx
        .send(())
        .expect("prefill server should still be running");
    decode_shutdown_tx
        .send(())
        .expect("decode server should still be running");
    prefill_server
        .await
        .expect("prefill server task should join")
        .expect("prefill server should stop cleanly");
    decode_server
        .await
        .expect("decode server task should join")
        .expect("decode server should stop cleanly");
}

async fn wait_for_grpc_health(addr: SocketAddr) {
    let endpoint = format!("http://{addr}");
    let mut last_error = None;
    for _ in 0..50 {
        match SglangServiceClient::connect(endpoint.clone()).await {
            Ok(mut client) => match client.health_check(HealthCheckRequest {}).await {
                Ok(response) => {
                    if response.into_inner().healthy {
                        return;
                    }
                    last_error = Some("health check returned unhealthy".to_string());
                }
                Err(error) => last_error = Some(error.to_string()),
            },
            Err(error) => last_error = Some(error.to_string()),
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!(
        "SRT gRPC worker at {addr} should become healthy: {}",
        last_error.unwrap_or_else(|| "no attempts made".to_string())
    );
}

fn unused_local_addr() -> SocketAddr {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("ephemeral port should bind");
    listener
        .local_addr()
        .expect("ephemeral listener should have local addr")
}
