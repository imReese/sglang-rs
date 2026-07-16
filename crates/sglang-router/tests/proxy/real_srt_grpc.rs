//! End-to-end router dispatch against a real Rust SRT gRPC worker.
//!
//! This pins the next transport step after gRPC worker introspection:
//! the router must be able to forward OpenAI-compatible HTTP traffic to
//! a `grpc://` worker, not only register it.

use std::fs;
use std::mem::size_of_val;
use std::net::{SocketAddr, TcpListener};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
use sglang_srt::proto::sglang::runtime::v1::sglang_service_client::SglangServiceClient;
use sglang_srt::proto::sglang::runtime::v1::{
    GetModelInfoRequest, GetWeightsByNameRequest, HealthCheckRequest,
};
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

fn unique_profile_dir() -> std::path::PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "sglang-rs-router-grpc-profile-{}-{suffix}",
        std::process::id()
    ))
}

fn unique_model_dir(name: &str) -> std::path::PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "sglang-rs-router-grpc-{name}-{}-{suffix}",
        std::process::id()
    ))
}

fn write_embedding_lm_artifacts_with_weight_values(model_dir: &std::path::Path, values: &[f32]) {
    assert_eq!(
        values.len(),
        3,
        "embedding LM fixture uses a 3-token vocabulary"
    );
    fs::create_dir_all(model_dir).expect("model directory should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "sglang_embedding_lm",
  "vocab_size": 3,
  "hidden_size": 1,
  "eos_token_id": [2, 3]
}"#,
    )
    .expect("config should be written");
    let byte_len = size_of_val(values);
    let header = format!(
        r#"{{"model.embed_tokens.weight":{{"dtype":"F32","shape":[3,1],"data_offsets":[0,{byte_len}]}},"lm_head.weight":{{"dtype":"F32","shape":[3,1],"data_offsets":[{byte_len},{}]}}}}"#,
        byte_len * 2
    );
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for _ in values {
        bytes.extend_from_slice(&0.0_f32.to_le_bytes());
    }
    fs::write(model_dir.join("model.safetensors"), bytes)
        .expect("safetensors shard should be written");
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
        .header("x-request-id", "chat-header-id")
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
    assert_eq!(body["id"], "chatcmpl-chat-header-id");
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
async fn router_update_weights_from_disk_reaches_real_rust_srt_grpc_worker() {
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
        .uri("/update_weights_from_disk")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny",
                "model_path": "/path/that/does/not/exist",
                "load_format": "safetensors",
            }))
            .unwrap(),
        ))
        .unwrap();

    let response = app.oneshot(request).await.expect("router should respond");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = response
        .into_body()
        .collect()
        .await
        .expect("router response body should collect")
        .to_bytes();
    let body: serde_json::Value =
        serde_json::from_slice(&body).expect("response should be gRPC status JSON");
    let message = body["error"]["message"]
        .as_str()
        .expect("gRPC status response should include an error message");
    assert!(
        message.contains("model_path") || message.contains("model path"),
        "expected worker validation error, got {body}"
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
async fn router_update_weight_version_reaches_real_rust_srt_grpc_worker() {
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
        .uri("/update_weight_version")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny",
                "new_version": "router-grpc-checkpoint",
                "abort_all_requests": false,
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
        serde_json::from_slice(&body).expect("response should be control JSON");
    assert_eq!(body["success"], true);
    assert_eq!(body["new_version"], "router-grpc-checkpoint");
    assert_eq!(body["updated_workers"], 1);

    let mut client = SglangServiceClient::connect(format!("http://{addr}"))
        .await
        .expect("gRPC client should connect");
    let model_info = client
        .get_model_info(GetModelInfoRequest {})
        .await
        .expect("model info should be readable after router update")
        .into_inner();
    assert_eq!(model_info.weight_version, "router-grpc-checkpoint");

    shutdown_tx
        .send(())
        .expect("gRPC worker should still be running");
    server
        .await
        .expect("gRPC server task should join")
        .expect("gRPC server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_get_weights_by_name_reaches_real_rust_srt_grpc_worker() {
    let model_dir = unique_model_dir("get-weights");
    write_embedding_lm_artifacts_with_weight_values(&model_dir, &[1.5, 2.5, 3.5]);
    let addr = unused_local_addr();
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("model dir should be utf-8"),
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
        .uri("/get_weights_by_name")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny",
                "name": "model.embed_tokens.weight",
                "truncate_size": 2,
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
        serde_json::from_slice(&body).expect("response should be parameter JSON");
    assert_eq!(body["parameter"], serde_json::json!([1.5, 2.5]));
    assert_eq!(body["queried_workers"], 1);

    let mut client = SglangServiceClient::connect(format!("http://{addr}"))
        .await
        .expect("gRPC client should connect");
    let direct = client
        .get_weights_by_name(GetWeightsByNameRequest {
            name: "model.embed_tokens.weight".to_string(),
            truncate_size: Some(2),
        })
        .await
        .expect("direct gRPC parameter read should work")
        .into_inner();
    assert_eq!(direct.parameter, vec![1.5, 2.5]);

    shutdown_tx
        .send(())
        .expect("gRPC worker should still be running");
    server
        .await
        .expect("gRPC server task should join")
        .expect("gRPC server should stop cleanly");
    fs::remove_dir_all(model_dir).expect("model temp directory should clean up");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_flush_cache_reaches_real_rust_srt_grpc_worker() {
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
        .uri("/flush_cache")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({"model": "tiny"})).unwrap(),
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
        serde_json::from_slice(&body).expect("response should be flush JSON");
    assert_eq!(body["success"], true);
    assert_eq!(body["flushed_workers"], 1);

    shutdown_tx
        .send(())
        .expect("gRPC worker should still be running");
    server
        .await
        .expect("gRPC server task should join")
        .expect("gRPC server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_start_and_stop_profile_reach_real_rust_srt_grpc_worker() {
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
    let output_dir = unique_profile_dir();

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(launch_grpc_server_with_shutdown(args, async move {
        let _ = shutdown_rx.await;
    }));
    wait_for_grpc_health(addr).await;

    let app = build_router(build_ctx_with_grpc_worker(addr));
    let start = Request::builder()
        .method("POST")
        .uri("/start_profile")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(
                &json!({"model": "tiny", "output_dir": output_dir.to_string_lossy()}),
            )
            .unwrap(),
        ))
        .unwrap();
    let start_response = app
        .clone()
        .oneshot(start)
        .await
        .expect("router should respond");
    assert_eq!(start_response.status(), StatusCode::OK);
    let body = start_response
        .into_body()
        .collect()
        .await
        .expect("router response body should collect")
        .to_bytes();
    let body: serde_json::Value =
        serde_json::from_slice(&body).expect("response should be profile JSON");
    assert_eq!(body["success"], true);
    assert_eq!(body["affected_workers"], 1);

    let stop = Request::builder()
        .method("POST")
        .uri("/stop_profile")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({"model": "tiny"})).unwrap(),
        ))
        .unwrap();
    let stop_response = app.oneshot(stop).await.expect("router should respond");
    assert_eq!(stop_response.status(), StatusCode::OK);
    let body = stop_response
        .into_body()
        .collect()
        .await
        .expect("router response body should collect")
        .to_bytes();
    let body: serde_json::Value =
        serde_json::from_slice(&body).expect("response should be profile JSON");
    assert_eq!(body["success"], true);
    assert_eq!(body["affected_workers"], 1);

    let entries = fs::read_dir(&output_dir)
        .expect("profile output directory should exist")
        .collect::<Result<Vec<_>, _>>()
        .expect("profile directory should be readable");
    assert_eq!(entries.len(), 1);
    let profile: serde_json::Value = serde_json::from_slice(
        &fs::read(entries[0].path()).expect("profile file should be readable"),
    )
    .expect("profile file should contain JSON");
    assert_eq!(profile["profile"]["transport"], "tonic-grpc");
    assert!(profile["profile"]["duration_ms"].as_u64().is_some());

    shutdown_tx
        .send(())
        .expect("gRPC worker should still be running");
    server
        .await
        .expect("gRPC server task should join")
        .expect("gRPC server should stop cleanly");
    fs::remove_dir_all(output_dir).expect("profile temp directory should clean up");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_pause_and_continue_generation_reach_real_rust_srt_grpc_worker() {
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
    let pause = Request::builder()
        .method("POST")
        .uri("/pause_generation")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({"model": "tiny"})).unwrap(),
        ))
        .unwrap();
    let pause_response = app
        .clone()
        .oneshot(pause)
        .await
        .expect("router should respond");
    assert_eq!(pause_response.status(), StatusCode::OK);
    let body = pause_response
        .into_body()
        .collect()
        .await
        .expect("router response body should collect")
        .to_bytes();
    let body: serde_json::Value =
        serde_json::from_slice(&body).expect("response should be pause JSON");
    assert_eq!(body["success"], true);
    assert_eq!(body["message"], "generation paused");
    assert_eq!(body["affected_workers"], 1);

    let paused_generate = Request::builder()
        .method("POST")
        .uri("/generate")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny",
                "text": "hi",
                "sampling_params": {"max_new_tokens": 1},
            }))
            .unwrap(),
        ))
        .unwrap();
    let paused_response = app
        .clone()
        .oneshot(paused_generate)
        .await
        .expect("router should respond");
    assert_eq!(paused_response.status(), StatusCode::BAD_REQUEST);

    let cont = Request::builder()
        .method("POST")
        .uri("/continue_generation")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({"model": "tiny"})).unwrap(),
        ))
        .unwrap();
    let continue_response = app.oneshot(cont).await.expect("router should respond");
    assert_eq!(continue_response.status(), StatusCode::OK);
    let body = continue_response
        .into_body()
        .collect()
        .await
        .expect("router response body should collect")
        .to_bytes();
    let body: serde_json::Value =
        serde_json::from_slice(&body).expect("response should be continue JSON");
    assert_eq!(body["success"], true);
    assert_eq!(body["message"], "generation continued");
    assert_eq!(body["affected_workers"], 1);

    shutdown_tx
        .send(())
        .expect("gRPC worker should still be running");
    server
        .await
        .expect("gRPC server task should join")
        .expect("gRPC server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_abort_request_reaches_real_rust_srt_grpc_worker() {
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
        .uri("/abort_request")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({"model": "tiny", "rid": "missing"})).unwrap(),
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
        serde_json::from_slice(&body).expect("response should be abort JSON");
    assert_eq!(body["success"], false);
    assert_eq!(body["message"], "request not found");
    assert_eq!(body["affected_workers"], 1);

    shutdown_tx
        .send(())
        .expect("gRPC worker should still be running");
    server
        .await
        .expect("gRPC server task should join")
        .expect("gRPC server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_abort_all_reaches_real_rust_srt_grpc_worker() {
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
        .uri("/abort_request")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({"model": "tiny", "abort_all": true})).unwrap(),
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
        serde_json::from_slice(&body).expect("response should be abort_all JSON");
    assert_eq!(body["success"], true);
    assert_eq!(body["message"], "request aborted");
    assert_eq!(body["affected_workers"], 1);
    assert_eq!(body["aborted_workers"], 1);
    assert_eq!(body["abort_all"], true);

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
        .header("x-request-id", "completion-header-id")
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
    assert_eq!(body["id"], "cmpl-completion-header-id");
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_responses_reaches_real_rust_srt_grpc_worker() {
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
        .uri("/v1/responses")
        .header("content-type", "application/json")
        .header("x-request-id", "responses-header-id")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny",
                "input": "hi",
                "max_output_tokens": 2,
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
        serde_json::from_slice(&body).expect("response should be OpenAI responses JSON");
    assert_eq!(body["id"], "resp-responses-header-id");
    assert_eq!(body["object"], "response");
    assert_eq!(body["model"], "tiny");
    assert_eq!(body["status"], "completed");
    assert_eq!(body["output"][0]["type"], "message");
    assert_eq!(body["output"][0]["content"][0]["type"], "output_text");
    assert_eq!(body["output"][0]["content"][0]["text"], "  ");
    assert_eq!(body["output_text"], "  ");

    shutdown_tx
        .send(())
        .expect("gRPC worker should still be running");
    server
        .await
        .expect("gRPC server task should join")
        .expect("gRPC server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_streaming_responses_reaches_real_rust_srt_grpc_worker() {
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
        .uri("/v1/responses")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny",
                "input": "hi",
                "max_output_tokens": 2,
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
            .any(|chunk| chunk["type"] == "response.output_text.delta"
                && chunk["delta"]
                    .as_str()
                    .is_some_and(|delta| !delta.is_empty())),
        "expected output text deltas, got {chunks:?}"
    );
    assert!(
        chunks
            .iter()
            .any(|chunk| chunk["type"] == "response.completed"
                && chunk["response"]["status"] == "completed"
                && chunk["response"]["output_text"] == "  "),
        "expected completed response event, got {chunks:?}"
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
async fn router_v1_loads_reaches_real_rust_srt_grpc_worker() {
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
        .method("GET")
        .uri("/v1/loads")
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
        serde_json::from_slice(&body).expect("response should be load JSON");

    assert_eq!(body["total_workers"], 1);
    assert_eq!(body["successful"], 1);
    assert_eq!(body["failed"], 0);
    assert_eq!(body["loads"][0]["worker"], format!("grpc://{addr}"));
    assert!(body["loads"][0]["worker_type"].is_null());
    assert_eq!(body["loads"][0]["load"], 0);

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_poll_transfers_reaches_real_rust_srt_grpc_worker() {
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
        .uri("/poll_transfers")
        .header("content-type", "application/json")
        .body(Body::from(json!({ "model": "tiny" }).to_string()))
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
    assert_eq!(
        body["completed_descriptor_checksums"],
        serde_json::json!([])
    );
    assert_eq!(body["pending_descriptor_checksums"], serde_json::json!([]));
    assert_eq!(body["polled_workers"], 1);
    assert_eq!(body["model"], "tiny");
    assert!(body["worker_type"].is_null());

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_generate_reaches_real_rust_srt_grpc_worker() {
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
        .uri("/generate")
        .header("content-type", "application/json")
        .header("x-request-id", "native-generate-header-id")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "text": "hi",
                "sampling_params": {
                    "max_new_tokens": 2,
                },
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
        serde_json::from_slice(&body).expect("response should be SGLang generate JSON");
    assert_eq!(body["request_id"], "native-generate-header-id");
    assert_eq!(body["text"], "  ");
    assert_eq!(body["output_ids"], json!([32, 32]));
    assert_eq!(body["finish_reason"], "stop");
    assert_eq!(body["usage"]["prompt_tokens"], 2);
    assert_eq!(body["usage"]["completion_tokens"], 2);

    shutdown_tx
        .send(())
        .expect("gRPC worker should still be running");
    server
        .await
        .expect("gRPC server task should join")
        .expect("gRPC server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_rerank_reaches_real_rust_srt_grpc_worker() {
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
        .uri("/v1/rerank")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny",
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
    assert_eq!(body["model"], "tiny");
    assert_eq!(body["results"].as_array().unwrap().len(), 2);
    assert_eq!(body["results"][0]["index"], 1);
    assert!(body["results"][0].get("document").is_none());
    assert_eq!(body["results"][1]["index"], 2);

    shutdown_tx
        .send(())
        .expect("gRPC worker should still be running");
    server
        .await
        .expect("gRPC server task should join")
        .expect("gRPC server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_score_reaches_real_rust_srt_grpc_worker() {
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
        .uri("/v1/score")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny",
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
    assert_eq!(body["model"], "tiny");
    assert_eq!(body["scores"].as_array().unwrap().len(), 2);
    assert_eq!(body["scores"][0].as_array().unwrap().len(), 3);
    assert!(body["usage"]["prompt_tokens"].as_i64().unwrap() > 0);

    shutdown_tx
        .send(())
        .expect("gRPC worker should still be running");
    server
        .await
        .expect("gRPC server task should join")
        .expect("gRPC server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_embeddings_reaches_real_rust_srt_grpc_worker() {
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
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny",
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
    assert_eq!(body["model"], "tiny");
    assert_eq!(body["data"].as_array().unwrap().len(), 2);
    assert_eq!(body["data"][0]["embedding"].as_array().unwrap().len(), 4);
    assert!(body["usage"]["prompt_tokens"].as_i64().unwrap() > 0);

    shutdown_tx
        .send(())
        .expect("gRPC worker should still be running");
    server
        .await
        .expect("gRPC server task should join")
        .expect("gRPC server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_classify_reaches_real_rust_srt_grpc_worker() {
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
        .uri("/v1/classify")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny",
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
    assert_eq!(body["model"], "tiny");
    assert_eq!(body["data"].as_array().unwrap().len(), 2);
    assert_eq!(body["data"][0]["num_classes"], 3);
    assert!(body["usage"]["prompt_tokens"].as_i64().unwrap() > 0);

    shutdown_tx
        .send(())
        .expect("gRPC worker should still be running");
    server
        .await
        .expect("gRPC server task should join")
        .expect("gRPC server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_streaming_generate_reaches_real_rust_srt_grpc_worker() {
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
        .uri("/generate")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "text": "hi",
                "sampling_params": {
                    "max_new_tokens": 2,
                },
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
            .any(|chunk| chunk["request_id"].is_string() && chunk["text"].is_string()),
        "expected native generate stream chunks, got {chunks:?}"
    );
    assert!(
        chunks.iter().any(|chunk| chunk["finish_reason"] == "stop"),
        "expected final stop chunk, got {chunks:?}"
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
async fn router_generate_accepts_tokenized_input_ids_for_real_rust_srt_grpc_worker() {
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
        .uri("/generate")
        .header("content-type", "application/json")
        .header("x-request-id", "native-tokenized-header-id")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "input_ids": [71, 72],
                "original_text": "hi",
                "sampling_params": {
                    "max_new_tokens": 2,
                },
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
        serde_json::from_slice(&body).expect("response should be SGLang generate JSON");
    assert_eq!(body["request_id"], "native-tokenized-header-id");
    assert_eq!(body["output_ids"], json!([32, 32]));
    assert_eq!(body["finish_reason"], "stop");
    assert_eq!(body["usage"]["prompt_tokens"], 2);
    assert_eq!(body["usage"]["completion_tokens"], 2);

    shutdown_tx
        .send(())
        .expect("gRPC worker should still be running");
    server
        .await
        .expect("gRPC server task should join")
        .expect("gRPC server should stop cleanly");
}

#[cfg(not(feature = "mooncake-link"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn real_rust_srt_grpc_mooncake_workers_reject_unlinked_transfer_backend() {
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

    let (prefill_shutdown_tx, prefill_shutdown_rx) = oneshot::channel::<()>();
    let prefill_server = tokio::spawn(launch_grpc_server_with_shutdown(prefill_args, async move {
        let _ = prefill_shutdown_rx.await;
    }));
    let (decode_shutdown_tx, decode_shutdown_rx) = oneshot::channel::<()>();
    let decode_server = tokio::spawn(launch_grpc_server_with_shutdown(decode_args, async move {
        let _ = decode_shutdown_rx.await;
    }));

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
            .contains("requires building sglang-srt with the mooncake-link feature"),
        "{prefill_error}"
    );
    assert!(
        decode_error
            .to_string()
            .contains("requires building sglang-srt with the mooncake-link feature"),
        "{decode_error}"
    );
    drop(prefill_shutdown_tx);
    drop(decode_shutdown_tx);
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
