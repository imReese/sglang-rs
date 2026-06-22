//! End-to-end PD router dispatch against real Rust SRT HTTP workers.
//!
//! The mock-worker tests pin request shaping. This file goes one step
//! further: it starts actual `sglang-srt` prefill/decode HTTP workers
//! and drives the router's PD fan-out through reqwest into those workers.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::net::{SocketAddr, TcpListener};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::{fs, io};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::json;
use sglang_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, DiscoveryConfig, ModelConfig, ObservabilityConfig,
    PolicyKind, ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sglang_router::discovery::{DiscoveryEvent, ModelId, WorkerId, WorkerMode, WorkerSpec};
use sglang_router::policies::factory::build_registry_with_defaults;
use sglang_router::proxy::Proxy;
use sglang_router::server::app::build_router;
use sglang_router::server::app_context::AppContext;
use sglang_router::tokenizer::TokenizerRegistry;
use sglang_router::workers::{manager, WorkerRegistry};
use sglang_srt::cli::ServerArgs;
use sglang_srt::server::launch_http_server_with_shutdown;
use sglang_srt::server::ServerLaunchError;
use tempfile::TempDir;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
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
                urls: vec!["http://placeholder:0".into()],
            }),
        },
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
    }
}

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
        "mooncake_tcp",
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
        "mooncake_tcp",
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn router_pd_chat_completes_with_real_cpu_embedding_lm_http_workers() {
    let model_dir = write_cpu_embedding_lm_fixture_model("router-fake-pd-chat");
    let prefill_addr = unused_local_addr();
    let bootstrap_addr = unused_local_addr();
    let decode_addr = unused_local_addr();
    let (mut prefill_server, prefill_shutdown_tx) =
        spawn_fake_prefill_worker(prefill_addr, bootstrap_addr, model_dir.path()).await;
    let (mut decode_server, decode_shutdown_tx) =
        spawn_fake_decode_worker(decode_addr, model_dir.path()).await;
    wait_for_worker_health(prefill_addr, "prefill", &mut prefill_server).await;
    wait_for_worker_health(decode_addr, "decode", &mut decode_server).await;

    let app = build_router(build_ctx(prefill_addr, decode_addr).await);
    let response = app
        .oneshot(chat_request("tiny", "hi", 1))
        .await
        .expect("router should respond");
    let status = response.status();
    let body = response
        .into_body()
        .collect()
        .await
        .expect("router response body should collect")
        .to_bytes();
    assert_eq!(
        status,
        StatusCode::OK,
        "router response body: {}",
        String::from_utf8_lossy(&body)
    );
    let body: serde_json::Value =
        serde_json::from_slice(&body).expect("response should be OpenAI chat JSON");
    assert_eq!(body["model"], "tiny");
    assert_eq!(body["choices"][0]["message"]["content"], "world");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["prompt_tokens"], 1);
    assert_eq!(body["usage"]["completion_tokens"], 1);

    prefill_shutdown_tx
        .send(())
        .expect("prefill should still run");
    decode_shutdown_tx
        .send(())
        .expect("decode should still run");
    prefill_server
        .await
        .expect("prefill server task should join")
        .expect("prefill server should stop cleanly");
    decode_server
        .await
        .expect("decode server task should join")
        .expect("decode server should stop cleanly");
}

#[cfg(feature = "mooncake-link")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires linked Mooncake runtime and a TCP-capable local environment"]
async fn router_pd_chat_completes_with_real_rust_srt_mooncake_workers() {
    let model_dir = write_linked_glm_fixture_model("router-linked-mooncake-pd-chat");
    let (prefill_addr, bootstrap_addr, prefill_zmq_addr, decode_addr) = test_addrs();
    let (mut prefill_server, prefill_shutdown_tx) = spawn_prefill_worker(
        prefill_addr,
        bootstrap_addr,
        prefill_zmq_addr,
        model_dir.path(),
    )
    .await;
    let (mut decode_server, decode_shutdown_tx) =
        spawn_decode_worker(decode_addr, model_dir.path()).await;
    wait_for_worker_health(prefill_addr, "prefill", &mut prefill_server).await;
    wait_for_worker_health(decode_addr, "decode", &mut decode_server).await;

    let app = build_router(build_ctx(prefill_addr, decode_addr).await);
    let response = app
        .oneshot(chat_request("tiny", "hi", 1))
        .await
        .expect("router should respond");
    let status = response.status();
    let body = response
        .into_body()
        .collect()
        .await
        .expect("router response body should collect")
        .to_bytes();
    assert_eq!(
        status,
        StatusCode::OK,
        "router response body: {}",
        String::from_utf8_lossy(&body)
    );
    let body: serde_json::Value =
        serde_json::from_slice(&body).expect("response should be OpenAI chat JSON");
    assert_eq!(body["model"], "tiny");
    assert!(body["choices"][0]["message"]["content"].is_string());

    prefill_shutdown_tx
        .send(())
        .expect("prefill should still run");
    decode_shutdown_tx
        .send(())
        .expect("decode should still run");
    prefill_server
        .await
        .expect("prefill server task should join")
        .expect("prefill server should stop cleanly");
    decode_server
        .await
        .expect("decode server task should join")
        .expect("decode server should stop cleanly");
}

async fn wait_for_worker_health(
    addr: SocketAddr,
    worker_name: &'static str,
    server: &mut JoinHandle<Result<(), ServerLaunchError>>,
) {
    let mut last_error = None;
    for _ in 0..100 {
        if server.is_finished() {
            let outcome = server.await;
            panic!("{worker_name} SRT worker exited before health at {addr}: {outcome:?}");
        }
        match request_raw(addr, "GET", "/health", None).await {
            Ok(response) if response.starts_with("HTTP/1.1 200") => return,
            Ok(response) => last_error = Some(response),
            Err(error) => last_error = Some(error.to_string()),
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!(
        "{worker_name} SRT worker at {addr} should become healthy: {}",
        last_error.unwrap_or_else(|| "no attempts made".to_string())
    );
}

#[cfg(feature = "mooncake-link")]
fn test_addrs() -> (SocketAddr, SocketAddr, SocketAddr, SocketAddr) {
    (
        unused_local_addr(),
        unused_local_addr(),
        unused_local_addr(),
        unused_local_addr(),
    )
}

#[cfg(feature = "mooncake-link")]
async fn spawn_prefill_worker(
    prefill_addr: SocketAddr,
    bootstrap_addr: SocketAddr,
    prefill_zmq_addr: SocketAddr,
    model_path: &Path,
) -> (
    JoinHandle<Result<(), ServerLaunchError>>,
    oneshot::Sender<()>,
) {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_path.to_str().expect("model path should be UTF-8"),
        "--served-model-name",
        "tiny",
        "--host",
        "127.0.0.1",
        "--port",
        &prefill_addr.port().to_string(),
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "mooncake_tcp",
        "--disaggregation-bootstrap-port",
        &bootstrap_addr.port().to_string(),
        "--disaggregation-zmq-ports",
        &prefill_zmq_addr.port().to_string(),
        "--kv-cache-dtype",
        "bfloat16",
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("prefill args should parse");
    spawn_worker(args)
}

#[cfg(feature = "mooncake-link")]
async fn spawn_decode_worker(
    decode_addr: SocketAddr,
    model_path: &Path,
) -> (
    JoinHandle<Result<(), ServerLaunchError>>,
    oneshot::Sender<()>,
) {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_path.to_str().expect("model path should be UTF-8"),
        "--served-model-name",
        "tiny",
        "--host",
        "127.0.0.1",
        "--port",
        &decode_addr.port().to_string(),
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "mooncake_tcp",
        "--kv-cache-dtype",
        "bfloat16",
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("decode args should parse");
    spawn_worker(args)
}

async fn spawn_fake_prefill_worker(
    prefill_addr: SocketAddr,
    bootstrap_addr: SocketAddr,
    model_path: &Path,
) -> (
    JoinHandle<Result<(), ServerLaunchError>>,
    oneshot::Sender<()>,
) {
    let engine_info_addr = unused_local_addr();
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_path.to_str().expect("model path should be UTF-8"),
        "--served-model-name",
        "tiny",
        "--host",
        "127.0.0.1",
        "--port",
        &prefill_addr.port().to_string(),
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "fake",
        "--disaggregation-bootstrap-port",
        &bootstrap_addr.port().to_string(),
        "--engine-info-bootstrap-port",
        &engine_info_addr.port().to_string(),
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("prefill args should parse");
    spawn_worker(args)
}

async fn spawn_fake_decode_worker(
    decode_addr: SocketAddr,
    model_path: &Path,
) -> (
    JoinHandle<Result<(), ServerLaunchError>>,
    oneshot::Sender<()>,
) {
    let engine_info_addr = unused_local_addr();
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_path.to_str().expect("model path should be UTF-8"),
        "--served-model-name",
        "tiny",
        "--host",
        "127.0.0.1",
        "--port",
        &decode_addr.port().to_string(),
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "fake",
        "--engine-info-bootstrap-port",
        &engine_info_addr.port().to_string(),
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("decode args should parse");
    spawn_worker(args)
}

fn spawn_worker(
    args: ServerArgs,
) -> (
    JoinHandle<Result<(), ServerLaunchError>>,
    oneshot::Sender<()>,
) {
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        launch_http_server_with_shutdown(args, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });
    (server, shutdown_tx)
}

fn chat_request(model: &str, content: &str, max_tokens: u32) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "stream": false
            })
            .to_string(),
        ))
        .expect("chat request should build")
}

fn write_cpu_embedding_lm_fixture_model(name: &str) -> TempDir {
    let model_dir = tempfile::Builder::new()
        .prefix(&format!("sglang-rs-{name}-"))
        .tempdir()
        .expect("temp model dir should be created");
    write_cpu_embedding_lm_fixture(model_dir.path());
    fs::write(
        model_dir.path().join("tokenizer.json"),
        word_level_tokenizer_json(),
    )
    .expect("tokenizer.json should be written");
    model_dir
}

fn write_cpu_embedding_lm_fixture(model_dir: &Path) {
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "sglang_embedding_lm",
  "vocab_size": 3,
  "hidden_size": 2,
  "eos_token_id": 2
}"#,
    )
    .expect("config should be written");

    let tensors = [
        (
            "model.embed_tokens.weight",
            vec![3, 2],
            vec![
                0.0, 0.0, // [UNK]
                1.0, 0.0, // hi
                0.0, 1.0, // world
            ],
        ),
        (
            "lm_head.weight",
            vec![3, 2],
            vec![
                0.0, 0.0, // [UNK]
                0.25, 0.0, // hi
                1.0, 0.0, // world
            ],
        ),
    ];
    let mut cursor = 0_usize;
    let mut metadata = Vec::new();
    let mut payload = Vec::new();
    for (name, shape, values) in tensors {
        let start = cursor;
        for value in values.into_iter().map(|value| value as f32) {
            payload.extend_from_slice(&value.to_le_bytes());
            cursor += 4;
        }
        metadata.push((name, "F32", shape, [start, cursor]));
    }

    write_safetensors_file(&model_dir.join("model.safetensors"), &metadata, &payload)
        .expect("safetensors shard should be written");
}

#[cfg(feature = "mooncake-link")]
fn write_linked_glm_fixture_model(name: &str) -> TempDir {
    let model_dir = tempfile::Builder::new()
        .prefix(&format!("sglang-rs-{name}-"))
        .tempdir()
        .expect("temp model dir should be created");
    write_glm_moe_dsa_attention_output_fixture(model_dir.path());
    fs::write(
        model_dir.path().join("tokenizer.json"),
        word_level_tokenizer_json(),
    )
    .expect("tokenizer.json should be written");
    model_dir
}

#[cfg(feature = "mooncake-link")]
fn write_glm_moe_dsa_attention_output_fixture(model_dir: &Path) {
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "glm_moe_dsa",
  "vocab_size": 2,
  "num_hidden_layers": 1,
  "hidden_size": 2,
  "intermediate_size": 2,
  "num_attention_heads": 2,
  "num_key_value_heads": 2,
  "head_dim": 1,
  "qk_nope_head_dim": 1,
  "qk_rope_head_dim": 0,
  "v_head_dim": 1,
  "rms_norm_eps": 0.0,
  "n_routed_experts": 1,
  "first_k_dense_replace": 1,
  "moe_layer_freq": 1
}"#,
    )
    .expect("config should be written");

    let tensors = [
        (
            "model.embed_tokens.weight",
            vec![2, 2],
            vec![1.0, 1.0, 1.0, -1.0],
        ),
        ("model.norm.weight", vec![2], vec![1.0, 1.0]),
        ("lm_head.weight", vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        (
            "model.layers.0.self_attn.q_a_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "model.layers.0.self_attn.q_a_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.layers.0.self_attn.q_b_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "model.layers.0.self_attn.kv_a_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.layers.0.self_attn.kv_b_proj.weight",
            vec![4, 2],
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            vec![2, 2],
            vec![2.0, 3.0, 5.0, 7.0],
        ),
        (
            "model.layers.0.input_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            vec![2, 2],
            vec![2.0, 0.0, 0.0, 3.0],
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            vec![2, 2],
            vec![5.0, 7.0, 11.0, 13.0],
        ),
    ];
    let mut cursor = 0_usize;
    let mut metadata = Vec::new();
    let mut payload = Vec::new();
    for (name, shape, values) in tensors {
        let start = cursor;
        for value in values.into_iter().map(|value| value as f32) {
            payload.extend_from_slice(&value.to_le_bytes());
            cursor += 4;
        }
        metadata.push((name, "F32", shape, [start, cursor]));
    }

    write_safetensors_file(&model_dir.join("model.safetensors"), &metadata, &payload)
        .expect("safetensors shard should be written");
}

fn write_safetensors_file(
    path: &Path,
    tensors: &[(&str, &str, Vec<usize>, [usize; 2])],
    payload: &[u8],
) -> io::Result<()> {
    let mut fields = Vec::new();
    for (name, dtype, shape, data_offsets) in tensors {
        let shape = shape
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        fields.push(format!(
            r#""{name}":{{"dtype":"{dtype}","shape":[{shape}],"data_offsets":[{},{}]}}"#,
            data_offsets[0], data_offsets[1]
        ));
    }
    let header = format!("{{{}}}", fields.join(","));
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(payload);
    fs::write(path, bytes)
}

fn word_level_tokenizer_json() -> &'static str {
    r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "Whitespace"
  },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "[UNK]": 0,
      "hello": 1,
      "world": 2,
      "hi": 1
    },
    "unk_token": "[UNK]"
  }
}"#
}

fn unused_local_addr() -> SocketAddr {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("ephemeral port should bind");
    listener
        .local_addr()
        .expect("ephemeral listener should have local addr")
}

async fn request_raw(
    addr: SocketAddr,
    method: &'static str,
    path: &'static str,
    body: Option<&'static str>,
) -> Result<String, io::Error> {
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
