//! End-to-end router dispatch against a real Rust SRT HTTP worker.

use std::net::{SocketAddr, TcpListener};
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::json;
use sglang_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, DiscoveryConfig, ModelConfig, ObservabilityConfig,
    PolicyKind, ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sglang_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sglang_router::policies::factory::build_registry_with_defaults;
use sglang_router::proxy::Proxy;
use sglang_router::server::app::build_router;
use sglang_router::server::app_context::AppContext;
use sglang_router::tokenizer::TokenizerRegistry;
use sglang_router::workers::WorkerRegistry;
use sglang_srt::cli::ServerArgs;
use sglang_srt::engine_info_bootstrap::{
    EngineInfoBootstrapService, TransferEngineInfo, TransferEngineInfoRegistration,
};
use sglang_srt::http::serve_http_router_with_shutdown;
use sglang_srt::server::{build_bootstrap_http_router_service, launch_http_server_with_shutdown};
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
            id: "tiny-reranker".into(),
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

fn build_ctx_with_http_worker(addr: SocketAddr) -> Arc<AppContext> {
    let cfg = config();
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    registry
        .add(WorkerSpec {
            id: WorkerId("http-worker".into()),
            url: format!("http://{addr}"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny-reranker".into())],
            bootstrap_port: None,
        })
        .expect("http worker should register");
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_rerank_reaches_real_rust_srt_http_worker() {
    let addr = unused_local_addr();
    let engine_info_addr = unused_local_addr();
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny-reranker",
        "--host",
        &addr.ip().to_string(),
        "--port",
        &addr.port().to_string(),
        "--engine-info-bootstrap-port",
        &engine_info_addr.port().to_string(),
    ])
    .expect("HTTP SRT args should parse");

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(launch_http_server_with_shutdown(args, async move {
        let _ = shutdown_rx.await;
    }));
    wait_for_http_health(addr).await;

    let app = build_router(build_ctx_with_http_worker(addr));
    let request = Request::builder()
        .method("POST")
        .uri("/v1/rerank")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny-reranker",
                "query": "rust pd router",
                "documents": [
                    "python gateway only",
                    "rust pd router transfers kv cache",
                    "router",
                ],
                "top_k": 2,
                "return_documents": false,
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
        serde_json::from_slice(&body).expect("response should be rerank JSON");
    assert_eq!(body["object"], "rerank");
    assert_eq!(body["model"], "tiny-reranker");
    assert_eq!(body["results"].as_array().unwrap().len(), 2);
    assert_eq!(body["results"][0]["index"], 1);
    assert!(body["results"][0].get("document").is_none());
    assert_eq!(body["results"][1]["index"], 2);

    shutdown_tx
        .send(())
        .expect("HTTP worker should still be running");
    server
        .await
        .expect("HTTP server task should join")
        .expect("HTTP server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_score_reaches_real_rust_srt_http_worker() {
    let addr = unused_local_addr();
    let engine_info_addr = unused_local_addr();
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny-reranker",
        "--host",
        &addr.ip().to_string(),
        "--port",
        &addr.port().to_string(),
        "--engine-info-bootstrap-port",
        &engine_info_addr.port().to_string(),
    ])
    .expect("HTTP SRT args should parse");

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(launch_http_server_with_shutdown(args, async move {
        let _ = shutdown_rx.await;
    }));
    wait_for_http_health(addr).await;

    let app = build_router(build_ctx_with_http_worker(addr));
    let request = Request::builder()
        .method("POST")
        .uri("/v1/score")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny-reranker",
                "query": "rust pd router",
                "items": [
                    "rust pd router transfers kv cache",
                    "python gateway",
                ],
                "label_token_ids": [1, 2, 3],
                "apply_softmax": true,
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
        serde_json::from_slice(&body).expect("response should be score JSON");
    assert_eq!(body["object"], "scoring");
    assert_eq!(body["model"], "tiny-reranker");
    assert_eq!(body["scores"].as_array().unwrap().len(), 2);
    assert_eq!(body["scores"][0].as_array().unwrap().len(), 3);
    assert!(body["usage"]["prompt_tokens"].as_i64().unwrap() > 0);

    shutdown_tx
        .send(())
        .expect("HTTP worker should still be running");
    server
        .await
        .expect("HTTP server task should join")
        .expect("HTTP server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_embeddings_reaches_real_rust_srt_http_worker() {
    let addr = unused_local_addr();
    let engine_info_addr = unused_local_addr();
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny-reranker",
        "--host",
        &addr.ip().to_string(),
        "--port",
        &addr.port().to_string(),
        "--engine-info-bootstrap-port",
        &engine_info_addr.port().to_string(),
    ])
    .expect("HTTP SRT args should parse");

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(launch_http_server_with_shutdown(args, async move {
        let _ = shutdown_rx.await;
    }));
    wait_for_http_health(addr).await;

    let app = build_router(build_ctx_with_http_worker(addr));
    let request = Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny-reranker",
                "input": ["rust pd router", "python gateway"],
                "dimensions": 4,
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
        serde_json::from_slice(&body).expect("response should be embeddings JSON");
    assert_eq!(body["object"], "list");
    assert_eq!(body["model"], "tiny-reranker");
    assert_eq!(body["data"].as_array().unwrap().len(), 2);
    assert_eq!(body["data"][0]["embedding"].as_array().unwrap().len(), 4);
    assert!(body["usage"]["prompt_tokens"].as_i64().unwrap() > 0);

    shutdown_tx
        .send(())
        .expect("HTTP worker should still be running");
    server
        .await
        .expect("HTTP server task should join")
        .expect("HTTP server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_classify_reaches_real_rust_srt_http_worker() {
    let addr = unused_local_addr();
    let engine_info_addr = unused_local_addr();
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny-reranker",
        "--host",
        &addr.ip().to_string(),
        "--port",
        &addr.port().to_string(),
        "--engine-info-bootstrap-port",
        &engine_info_addr.port().to_string(),
    ])
    .expect("HTTP SRT args should parse");

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(launch_http_server_with_shutdown(args, async move {
        let _ = shutdown_rx.await;
    }));
    wait_for_http_health(addr).await;

    let app = build_router(build_ctx_with_http_worker(addr));
    let request = Request::builder()
        .method("POST")
        .uri("/v1/classify")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny-reranker",
                "input": ["rust pd router", "python gateway"],
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
        serde_json::from_slice(&body).expect("response should be classify JSON");
    assert_eq!(body["object"], "list");
    assert_eq!(body["model"], "tiny-reranker");
    assert_eq!(body["data"].as_array().unwrap().len(), 2);
    assert_eq!(body["data"][0]["num_classes"], 3);
    assert!(body["usage"]["prompt_tokens"].as_i64().unwrap() > 0);

    shutdown_tx
        .send(())
        .expect("HTTP worker should still be running");
    server
        .await
        .expect("HTTP server task should join")
        .expect("HTTP server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_remote_instance_transfer_engine_info_reaches_real_rust_srt_http_worker() {
    let addr = unused_local_addr();
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny-reranker",
        "--host",
        &addr.ip().to_string(),
        "--port",
        &addr.port().to_string(),
    ])
    .expect("HTTP SRT args should parse");
    let engine_info = EngineInfoBootstrapService::default();
    engine_info
        .state()
        .lock()
        .expect("engine info state should lock")
        .register_transfer_engine_info(TransferEngineInfoRegistration {
            tp_rank: 0,
            transfer_engine_info: TransferEngineInfo {
                session_id: "session-a".to_string(),
                weights_info_dict: json!({
                    "layer.0": {
                        "addr": 4096,
                        "length": 8192,
                    }
                }),
            },
        });
    let service =
        build_bootstrap_http_router_service(&args).with_engine_info_bootstrap_service(engine_info);

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(serve_http_router_with_shutdown(addr, service, async move {
        let _ = shutdown_rx.await;
    }));
    wait_for_http_health(addr).await;

    let app = build_router(build_ctx_with_http_worker(addr));
    let request = Request::builder()
        .method("GET")
        .uri("/remote_instance_transfer_engine_info?rank=0")
        .body(Body::empty())
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
        serde_json::from_slice(&body).expect("response should be transfer engine info JSON");
    assert_eq!(body["rank"], 0);
    assert_eq!(body["remote_instance_transfer_engine_info"][0], "session-a");
    assert_eq!(
        body["remote_instance_transfer_engine_info"][1]["layer.0"]["length"],
        8192
    );

    shutdown_tx
        .send(())
        .expect("HTTP worker should still be running");
    server
        .await
        .expect("HTTP server task should join")
        .expect("HTTP server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_poll_transfers_reaches_real_rust_srt_http_worker() {
    let addr = unused_local_addr();
    let engine_info_addr = unused_local_addr();
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny-reranker",
        "--host",
        &addr.ip().to_string(),
        "--port",
        &addr.port().to_string(),
        "--engine-info-bootstrap-port",
        &engine_info_addr.port().to_string(),
    ])
    .expect("HTTP SRT args should parse");

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(launch_http_server_with_shutdown(args, async move {
        let _ = shutdown_rx.await;
    }));
    wait_for_http_health(addr).await;

    let app = build_router(build_ctx_with_http_worker(addr));
    let request = Request::builder()
        .method("POST")
        .uri("/poll_transfers")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({"model": "tiny-reranker"})).unwrap(),
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
        serde_json::from_slice(&body).expect("response should be transfer poll JSON");
    assert_eq!(body["completed_batches"], 0);
    assert_eq!(body["pending_batches"], 0);
    assert_eq!(body["polled_workers"], 1);
    assert_eq!(body["model"], "tiny-reranker");
    assert!(body["worker_type"].is_null());

    shutdown_tx
        .send(())
        .expect("HTTP worker should still be running");
    server
        .await
        .expect("HTTP server task should join")
        .expect("HTTP server should stop cleanly");
}

async fn wait_for_http_health(addr: SocketAddr) {
    let client = reqwest::Client::new();
    let url = format!("http://{addr}/health");
    for _ in 0..100 {
        if let Ok(response) = client.get(&url).send().await {
            if response.status().is_success() {
                return;
            }
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("HTTP worker did not become healthy at {addr}");
}

fn unused_local_addr() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("read local addr")
}
