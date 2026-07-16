use std::fs;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;
use sglang_srt::backend::{ComputeCapability, CudaBackend};
use sglang_srt::cli::ServerArgs;
use sglang_srt::http::serve_http_router_with_shutdown;
use sglang_srt::server::build_bootstrap_http_router_service;
use tokio::sync::oneshot;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a CUDA device, NVIDIA driver, and cuBLAS"]
async fn cuda_auto_selects_cublas_for_weight_backed_http_inference() {
    let backend = CudaBackend::initialize(0).expect("CUDA backend should initialize");
    let ComputeCapability::Cuda(_) = backend.capabilities().compute_capability else {
        panic!("CUDA backend must report CUDA compute capability");
    };
    drop(backend);

    let model_dir = temp_model_dir("cuda-cublas-http");
    write_embedding_lm_artifacts(&model_dir);
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model path should be utf-8"),
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("server args should parse");
    let service = build_bootstrap_http_router_service(&args);
    let addr = unused_local_addr();
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
        r#"{"text":"hello","sampling_params":{"max_new_tokens":1}}"#,
    )
    .await;

    assert_eq!(generated["output_ids"], serde_json::json!([2]));
    assert_eq!(generated["text"], "world");
    assert_eq!(generated["usage"]["prompt_tokens"], 1);
    assert_eq!(generated["usage"]["completion_tokens"], 1);

    shutdown_tx
        .send(())
        .expect("CUDA HTTP server should still be running");
    server
        .await
        .expect("CUDA HTTP server task should join")
        .expect("CUDA HTTP server should stop cleanly");
    fs::remove_dir_all(model_dir).expect("temp model directory should be removed");
}

fn write_embedding_lm_artifacts(model_dir: &Path) {
    fs::create_dir_all(model_dir).expect("temp model directory should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "sglang_embedding_lm",
  "vocab_size": 3,
  "hidden_size": 2
}"#,
    )
    .expect("model config should be written");
    fs::write(
        model_dir.join("tokenizer.json"),
        word_level_tokenizer_json(),
    )
    .expect("tokenizer should be written");

    let token_embeddings = [0.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0];
    let lm_head = [0.0_f32, 0.0, 1.0, 0.0, 2.0, 0.0];
    let payload = token_embeddings
        .into_iter()
        .chain(lm_head)
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    write_safetensors_file(
        &model_dir.join("model.safetensors"),
        &[
            ("model.embed_tokens.weight", "F32", &[3, 2], [0, 24]),
            ("lm_head.weight", "F32", &[3, 2], [24, 48]),
        ],
        &payload,
    )
    .expect("model weights should be written");
}

fn write_safetensors_file(
    path: &Path,
    tensors: &[(&str, &str, &[usize], [usize; 2])],
    payload: &[u8],
) -> std::io::Result<()> {
    let mut fields = Vec::new();
    for (name, dtype, shape, offsets) in tensors {
        let shape = shape
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        fields.push(format!(
            r#""{name}":{{"dtype":"{dtype}","shape":[{shape}],"data_offsets":[{},{}]}}"#,
            offsets[0], offsets[1]
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
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {"[UNK]": 0, "hello": 1, "world": 2},
    "unk_token": "[UNK]"
  }
}"#
}

fn temp_model_dir(name: &str) -> PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("sglang-rs-{name}-{}-{suffix}", std::process::id()))
}

fn unused_local_addr() -> SocketAddr {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("ephemeral port should bind");
    listener
        .local_addr()
        .expect("ephemeral listener should have local address")
}

async fn post_json_with_retry(addr: SocketAddr, path: &str, body: &'static str) -> Value {
    let mut last_error = None;
    for _ in 0..100 {
        match post_json(addr, path, body).await {
            Ok(value) => return value,
            Err(error) => {
                last_error = Some(error);
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        }
    }
    panic!(
        "HTTP client should connect to CUDA test server: {}",
        last_error.expect("at least one request should run")
    );
}

async fn post_json(
    addr: SocketAddr,
    path: &str,
    body: &'static str,
) -> Result<Value, std::io::Error> {
    let path = path.to_string();
    let response = tokio::task::spawn_blocking(move || {
        let mut stream = TcpStream::connect(addr)?;
        let request = format!(
            "POST {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(request.as_bytes())?;
        let mut response = String::new();
        stream.read_to_string(&mut response)?;
        Ok::<_, std::io::Error>(response)
    })
    .await
    .expect("blocking HTTP request should join")?;
    let (_, body) = response
        .split_once("\r\n\r\n")
        .ok_or_else(|| std::io::Error::other("HTTP response is missing headers"))?;
    serde_json::from_str(body).map_err(std::io::Error::other)
}
