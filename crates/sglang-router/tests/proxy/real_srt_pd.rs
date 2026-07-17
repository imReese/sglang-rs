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
use std::path::Path;
#[cfg(feature = "mooncake-link")]
use std::sync::Arc;
#[cfg(feature = "mooncake-link")]
use std::time::Duration;
#[cfg(feature = "mooncake-link")]
use std::{fs, io};

#[cfg(feature = "mooncake-link")]
use axum::body::Body;
#[cfg(feature = "mooncake-link")]
use axum::http::{Request, StatusCode};
#[cfg(feature = "mooncake-link")]
use http_body_util::BodyExt;
#[cfg(feature = "mooncake-link")]
use serde_json::json;
#[cfg(feature = "mooncake-link")]
use sglang_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, DiscoveryConfig, ModelConfig, ObservabilityConfig,
    PolicyKind, ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
#[cfg(feature = "mooncake-link")]
use sglang_router::discovery::ModelId;
#[cfg(feature = "mooncake-link")]
use sglang_router::discovery::{DiscoveryEvent, WorkerId, WorkerMode, WorkerSpec};
#[cfg(feature = "mooncake-link")]
use sglang_router::policies::factory::build_registry_with_defaults;
#[cfg(feature = "mooncake-link")]
use sglang_router::proxy::Proxy;
#[cfg(feature = "mooncake-link")]
use sglang_router::server::app::build_router;
#[cfg(feature = "mooncake-link")]
use sglang_router::server::app_context::AppContext;
#[cfg(feature = "mooncake-link")]
use sglang_router::tokenizer::TokenizerRegistry;
#[cfg(feature = "mooncake-link")]
use sglang_router::workers::{manager, WorkerRegistry};
use sglang_srt::cli::ServerArgs;
use sglang_srt::server::launch_http_server_with_shutdown;
#[cfg(feature = "mooncake-link")]
use sglang_srt::server::ServerLaunchError;
#[cfg(feature = "mooncake-link")]
use tempfile::TempDir;
#[cfg(feature = "mooncake-link")]
use tokio::sync::mpsc;
use tokio::sync::oneshot;
#[cfg(feature = "mooncake-link")]
use tokio::task::JoinHandle;
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
async fn real_rust_srt_mooncake_workers_reject_unlinked_transfer_backend() {
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

#[cfg(feature = "mooncake-link")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires linked Mooncake, CUDA, and a TCP-capable local environment"]
async fn router_pd_chat_completes_with_real_cuda_qwen_mooncake_workers() {
    let model_dir = write_linked_qwen3_fixture_model("router-linked-cuda-qwen-pd-chat");
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

#[cfg(feature = "mooncake-link")]
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
fn cuda_test_device_ordinal() -> usize {
    let ordinal = std::env::var("SGLANG_CUDA_TEST_DEVICE")
        .expect("SGLANG_CUDA_TEST_DEVICE must select a nonzero CUDA ordinal")
        .parse()
        .expect("SGLANG_CUDA_TEST_DEVICE must be a CUDA device ordinal");
    assert_ne!(
        ordinal, 0,
        "linked P/D acceptance must exercise a nonzero CUDA ordinal"
    );
    ordinal
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
    let device_ordinal = cuda_test_device_ordinal().to_string();
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_path.to_str().expect("model path should be UTF-8"),
        "--device",
        "cuda",
        "--base-gpu-id",
        &device_ordinal,
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
    let device_ordinal = cuda_test_device_ordinal().to_string();
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_path.to_str().expect("model path should be UTF-8"),
        "--device",
        "cuda",
        "--base-gpu-id",
        &device_ordinal,
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

#[cfg(feature = "mooncake-link")]
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

#[cfg(feature = "mooncake-link")]
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

#[cfg(feature = "mooncake-link")]
fn write_linked_qwen3_fixture_model(name: &str) -> TempDir {
    let model_dir = tempfile::Builder::new()
        .prefix(&format!("sglang-rs-{name}-"))
        .tempdir()
        .expect("temp model dir should be created");
    write_qwen3_dense_fixture(model_dir.path());
    fs::write(
        model_dir.path().join("tokenizer.json"),
        word_level_tokenizer_json(),
    )
    .expect("tokenizer.json should be written");
    fs::write(
        model_dir.path().join("tokenizer_config.json"),
        r#"{"chat_template":"{% for message in messages %}{{ message.content }}{% endfor %}"}"#,
    )
    .expect("tokenizer_config.json should be written");
    model_dir
}

#[cfg(feature = "mooncake-link")]
fn write_qwen3_dense_fixture(model_dir: &Path) {
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "architectures": ["Qwen3ForCausalLM"],
  "model_type": "qwen3",
  "vocab_size": 3,
  "num_hidden_layers": 1,
  "hidden_size": 2,
  "intermediate_size": 2,
  "num_attention_heads": 1,
  "num_key_value_heads": 1,
  "head_dim": 2,
  "hidden_act": "silu",
  "attention_bias": false,
  "rms_norm_eps": 0.000001,
  "rope_theta": 1000000.0,
  "max_position_embeddings": 32,
  "tie_word_embeddings": false
}"#,
    )
    .expect("Qwen3 config should be written");

    let tensors = [
        (
            "model.embed_tokens.weight",
            vec![3, 2],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ),
        ("model.norm.weight", vec![2], vec![1.0, 1.0]),
        (
            "lm_head.weight",
            vec![3, 2],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.self_attn.q_norm.weight",
            vec![2],
            vec![1.0; 2],
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.self_attn.k_norm.weight",
            vec![2],
            vec![1.0; 2],
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.input_layernorm.weight",
            vec![2],
            vec![1.0; 2],
        ),
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![2],
            vec![1.0; 2],
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
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

#[cfg(feature = "mooncake-link")]
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

#[cfg(feature = "mooncake-link")]
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
