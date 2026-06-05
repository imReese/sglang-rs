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
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use sglang_srt::cli::ServerArgs;
use sglang_srt::proto::sglang::runtime::v1::sglang_service_client::SglangServiceClient;
use sglang_srt::proto::sglang::runtime::v1::HealthCheckRequest;
use sglang_srt::server::launch_grpc_server_with_shutdown;
use tokio::sync::oneshot;
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
