use std::fs;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;
use sglang_srt::backend::CudaBackend;
use sglang_srt::cli::ServerArgs;
use sglang_srt::http::serve_http_router_with_shutdown;
use sglang_srt::server::build_bootstrap_http_router_service;
use tokio::sync::oneshot;

fn cuda_test_device_ordinal() -> usize {
    std::env::var("SGLANG_CUDA_TEST_DEVICE")
        .unwrap_or_else(|_| "0".to_string())
        .parse()
        .expect("SGLANG_CUDA_TEST_DEVICE must be a CUDA device ordinal")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a CUDA device, NVIDIA driver, NVRTC, and cuBLAS with BF16 support"]
async fn cuda_qwen3_uses_the_shared_dense_decoder_and_runtime_kv_pool() {
    let device_ordinal = cuda_test_device_ordinal();
    let backend = CudaBackend::initialize(device_ordinal).expect("CUDA backend should initialize");
    assert!(
        backend
            .capabilities()
            .supported_dtypes
            .contains(&sglang_srt::backend::RuntimeDtype::Bf16),
        "CUDA acceptance device must support BF16"
    );
    drop(backend);

    let model_dir = temp_model_dir("cuda-qwen3-dense-http");
    write_qwen3_dense_artifacts(&model_dir);
    let mut args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model path should be utf-8"),
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--page-size",
        "4",
        "--num-reserved-decode-tokens",
        "32",
    ])
    .expect("server args should parse");
    args.base_gpu_id = device_ordinal;
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
        r#"{"text":"hello","sampling_params":{"max_new_tokens":2}}"#,
    )
    .await;

    assert_eq!(generated["output_ids"], serde_json::json!([2, 1]));
    assert_eq!(generated["usage"]["prompt_tokens"], 1);
    assert_eq!(generated["usage"]["completion_tokens"], 2);

    shutdown_tx
        .send(())
        .expect("CUDA Qwen3 HTTP server should still be running");
    server
        .await
        .expect("CUDA Qwen3 HTTP server task should join")
        .expect("CUDA Qwen3 HTTP server should stop cleanly");
    fs::remove_dir_all(model_dir).expect("temp model directory should be removed");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a CUDA device, NVIDIA driver, NVRTC, and cuBLAS with BF16 support"]
async fn cuda_kimi_linear_uses_shared_kda_mla_moe_and_runtime_kv_pool() {
    let device_ordinal = cuda_test_device_ordinal();
    let backend = CudaBackend::initialize(device_ordinal).expect("CUDA backend should initialize");
    assert!(
        backend
            .capabilities()
            .supported_dtypes
            .contains(&sglang_srt::backend::RuntimeDtype::Bf16),
        "CUDA acceptance device must support BF16"
    );
    drop(backend);

    let model_dir = temp_model_dir("cuda-hybrid-kda-mla-moe-http");
    write_kimi_linear_artifacts(&model_dir);
    let mut args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model path should be utf-8"),
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--page-size",
        "4",
        "--num-reserved-decode-tokens",
        "32",
    ])
    .expect("server args should parse");
    args.base_gpu_id = device_ordinal;
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
        r#"{"text":"hello","sampling_params":{"max_new_tokens":2}}"#,
    )
    .await;

    assert_eq!(generated["output_ids"], serde_json::json!([2, 1]));
    assert_eq!(generated["usage"]["prompt_tokens"], 1);
    assert_eq!(generated["usage"]["completion_tokens"], 2);

    shutdown_tx
        .send(())
        .expect("CUDA hybrid HTTP server should still be running");
    server
        .await
        .expect("CUDA hybrid HTTP server task should join")
        .expect("CUDA hybrid HTTP server should stop cleanly");
    fs::remove_dir_all(model_dir).expect("temp model directory should be removed");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires a CUDA device, NVIDIA driver, NVRTC, and cuBLAS with BF16 support"]
async fn cuda_deepseek_v3_uses_shared_mla_moe_and_runtime_kv_pool() {
    let device_ordinal = cuda_test_device_ordinal();
    let backend = CudaBackend::initialize(device_ordinal).expect("CUDA backend should initialize");
    assert!(
        backend
            .capabilities()
            .supported_dtypes
            .contains(&sglang_srt::backend::RuntimeDtype::Bf16),
        "CUDA acceptance device must support BF16"
    );
    drop(backend);

    let model_dir = temp_model_dir("cuda-deepseek-v3-mla-moe-http");
    write_deepseek_v3_artifacts(&model_dir);
    let mut args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model path should be utf-8"),
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--page-size",
        "4",
        "--num-reserved-decode-tokens",
        "32",
    ])
    .expect("server args should parse");
    args.base_gpu_id = device_ordinal;
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
        r#"{"text":"hello","sampling_params":{"max_new_tokens":2}}"#,
    )
    .await;

    assert_eq!(generated["output_ids"], serde_json::json!([2, 1]));
    assert_eq!(generated["usage"]["prompt_tokens"], 1);
    assert_eq!(generated["usage"]["completion_tokens"], 2);

    shutdown_tx
        .send(())
        .expect("CUDA DeepSeek V3 HTTP server should still be running");
    server
        .await
        .expect("CUDA DeepSeek V3 HTTP server task should join")
        .expect("CUDA DeepSeek V3 HTTP server should stop cleanly");
    fs::remove_dir_all(model_dir).expect("temp model directory should be removed");
}

fn write_qwen3_dense_artifacts(model_dir: &Path) {
    fs::create_dir_all(model_dir).expect("temp model directory should be created");
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
    fs::write(
        model_dir.join("tokenizer.json"),
        word_level_tokenizer_json(),
    )
    .expect("Qwen3 tokenizer should be written");

    let descriptors: Vec<(&str, &[usize], Vec<f32>)> = vec![
        (
            "model.embed_tokens.weight",
            &[3, 2],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ),
        ("model.norm.weight", &[2], vec![1.0, 1.0]),
        (
            "lm_head.weight",
            &[3, 2],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            &[2, 2],
            vec![0.0; 4],
        ),
        ("model.layers.0.self_attn.q_norm.weight", &[2], vec![1.0; 2]),
        (
            "model.layers.0.self_attn.k_proj.weight",
            &[2, 2],
            vec![0.0; 4],
        ),
        ("model.layers.0.self_attn.k_norm.weight", &[2], vec![1.0; 2]),
        (
            "model.layers.0.self_attn.v_proj.weight",
            &[2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            &[2, 2],
            vec![0.0; 4],
        ),
        ("model.layers.0.input_layernorm.weight", &[2], vec![1.0; 2]),
        (
            "model.layers.0.post_attention_layernorm.weight",
            &[2],
            vec![1.0; 2],
        ),
        ("model.layers.0.mlp.gate_proj.weight", &[2, 2], vec![0.0; 4]),
        ("model.layers.0.mlp.up_proj.weight", &[2, 2], vec![0.0; 4]),
        ("model.layers.0.mlp.down_proj.weight", &[2, 2], vec![0.0; 4]),
    ];
    let mut payload = Vec::new();
    let mut tensors = Vec::new();
    for (name, shape, values) in descriptors {
        let start = payload.len();
        payload.extend(values.into_iter().flat_map(f32::to_le_bytes));
        tensors.push((name, "F32", shape, [start, payload.len()]));
    }
    write_safetensors_file(&model_dir.join("model.safetensors"), &tensors, &payload)
        .expect("Qwen3 weights should be written");
}

fn write_kimi_linear_artifacts(model_dir: &Path) {
    fs::create_dir_all(model_dir).expect("temp model directory should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "architectures": ["KimiLinearForCausalLM"],
  "model_type": "kimi_linear",
  "vocab_size": 3,
  "model_max_length": 32,
  "num_hidden_layers": 2,
  "hidden_size": 2,
  "intermediate_size": 2,
  "num_attention_heads": 1,
  "num_key_value_heads": 1,
  "hidden_act": "silu",
  "rms_norm_eps": 0.00001,
  "rope_theta": 10000.0,
  "rope_scaling": null,
  "tie_word_embeddings": false,
  "q_lora_rank": 2,
  "kv_lora_rank": 2,
  "qk_nope_head_dim": 2,
  "qk_rope_head_dim": 2,
  "v_head_dim": 2,
  "mla_use_nope": true,
  "moe_intermediate_size": 2,
  "moe_renormalize": true,
  "moe_router_activation_func": "sigmoid",
  "num_experts": 1,
  "num_experts_per_token": 1,
  "num_shared_experts": 1,
  "routed_scaling_factor": 1.0,
  "first_k_dense_replace": 1,
  "moe_layer_freq": 1,
  "use_grouped_topk": true,
  "num_expert_group": 1,
  "topk_group": 1,
  "num_nextn_predict_layers": 0,
  "linear_attn_config": {
    "head_dim": 2,
    "num_heads": 1,
    "short_conv_kernel_size": 2,
    "kda_layers": [1],
    "full_attn_layers": [2]
  }
}"#,
    )
    .expect("hybrid config should be written");
    fs::write(
        model_dir.join("tokenizer.json"),
        word_level_tokenizer_json(),
    )
    .expect("hybrid tokenizer should be written");

    let mut descriptors: Vec<(String, Vec<usize>, Vec<f32>)> = vec![
        (
            "model.embed_tokens.weight".to_string(),
            vec![3, 2],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ),
        ("model.norm.weight".to_string(), vec![2], vec![1.0; 2]),
        (
            "lm_head.weight".to_string(),
            vec![3, 2],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        ),
    ];
    let mut add_tensor = |name: String, shape: Vec<usize>, value: f32| {
        descriptors.push((name, shape.clone(), vec![value; shape.iter().product()]));
    };
    for layer_id in 0..2 {
        let prefix = format!("model.layers.{layer_id}");
        add_tensor(format!("{prefix}.input_layernorm.weight"), vec![2], 1.0);
        add_tensor(
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![2],
            1.0,
        );
    }
    for (suffix, shape) in [
        ("self_attn.A_log", vec![1, 1, 1, 1]),
        ("self_attn.dt_bias", vec![2]),
        ("self_attn.q_proj.weight", vec![2, 2]),
        ("self_attn.k_proj.weight", vec![2, 2]),
        ("self_attn.v_proj.weight", vec![2, 2]),
        ("self_attn.b_proj.weight", vec![1, 2]),
        ("self_attn.f_a_proj.weight", vec![2, 2]),
        ("self_attn.f_b_proj.weight", vec![2, 2]),
        ("self_attn.g_a_proj.weight", vec![2, 2]),
        ("self_attn.g_b_proj.weight", vec![2, 2]),
        ("self_attn.q_conv1d.weight", vec![2, 2]),
        ("self_attn.k_conv1d.weight", vec![2, 2]),
        ("self_attn.v_conv1d.weight", vec![2, 2]),
        ("self_attn.o_proj.weight", vec![2, 2]),
        ("mlp.gate_proj.weight", vec![2, 2]),
        ("mlp.up_proj.weight", vec![2, 2]),
        ("mlp.down_proj.weight", vec![2, 2]),
    ] {
        add_tensor(format!("model.layers.0.{suffix}"), shape, 0.0);
    }
    add_tensor(
        "model.layers.0.self_attn.o_norm.weight".to_string(),
        vec![2],
        1.0,
    );
    for (suffix, shape) in [
        ("self_attn.q_a_proj.weight", vec![2, 2]),
        ("self_attn.q_b_proj.weight", vec![4, 2]),
        ("self_attn.kv_a_proj_with_mqa.weight", vec![4, 2]),
        ("self_attn.kv_b_proj.weight", vec![4, 2]),
        ("self_attn.o_proj.weight", vec![2, 2]),
        ("mlp.gate.weight", vec![1, 2]),
        ("mlp.gate.e_score_correction_bias", vec![1]),
        ("mlp.experts.0.w1.weight", vec![2, 2]),
        ("mlp.experts.0.w2.weight", vec![2, 2]),
        ("mlp.experts.0.w3.weight", vec![2, 2]),
        ("mlp.shared_experts.gate_proj.weight", vec![2, 2]),
        ("mlp.shared_experts.up_proj.weight", vec![2, 2]),
        ("mlp.shared_experts.down_proj.weight", vec![2, 2]),
    ] {
        add_tensor(format!("model.layers.1.{suffix}"), shape, 0.0);
    }
    add_tensor(
        "model.layers.1.self_attn.q_a_layernorm.weight".to_string(),
        vec![2],
        1.0,
    );
    add_tensor(
        "model.layers.1.self_attn.kv_a_layernorm.weight".to_string(),
        vec![2],
        1.0,
    );

    let mut payload = Vec::new();
    let mut tensors = Vec::new();
    for (name, shape, values) in &descriptors {
        let start = payload.len();
        payload.extend(values.iter().flat_map(|value| value.to_le_bytes()));
        tensors.push((
            name.as_str(),
            "F32",
            shape.as_slice(),
            [start, payload.len()],
        ));
    }
    write_safetensors_file(&model_dir.join("model.safetensors"), &tensors, &payload)
        .expect("hybrid weights should be written");
}

fn write_deepseek_v3_artifacts(model_dir: &Path) {
    fs::create_dir_all(model_dir).expect("temp model directory should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "architectures": ["DeepseekV3ForCausalLM"],
  "model_type": "kimi_k2",
  "vocab_size": 3,
  "max_position_embeddings": 32,
  "num_hidden_layers": 2,
  "hidden_size": 2,
  "intermediate_size": 2,
  "num_attention_heads": 1,
  "hidden_act": "silu",
  "rms_norm_eps": 0.00001,
  "rope_theta": 10000.0,
  "rope_scaling": null,
  "attention_bias": false,
  "tie_word_embeddings": false,
  "q_lora_rank": 2,
  "kv_lora_rank": 2,
  "qk_nope_head_dim": 1,
  "qk_rope_head_dim": 2,
  "v_head_dim": 2,
  "moe_intermediate_size": 2,
  "n_routed_experts": 1,
  "n_shared_experts": 1,
  "num_experts_per_tok": 1,
  "routed_scaling_factor": 1.0,
  "first_k_dense_replace": 1,
  "moe_layer_freq": 1,
  "n_group": 1,
  "topk_group": 1,
  "norm_topk_prob": true,
  "scoring_func": "sigmoid",
  "topk_method": "noaux_tc",
  "num_nextn_predict_layers": 0,
  "quantization_config": null
}"#,
    )
    .expect("DeepSeek V3 config should be written");
    fs::write(
        model_dir.join("tokenizer.json"),
        word_level_tokenizer_json(),
    )
    .expect("DeepSeek V3 tokenizer should be written");

    let mut descriptors: Vec<(String, Vec<usize>, Vec<f32>)> = vec![
        (
            "model.embed_tokens.weight".to_string(),
            vec![3, 2],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ),
        ("model.norm.weight".to_string(), vec![2], vec![1.0; 2]),
        (
            "lm_head.weight".to_string(),
            vec![3, 2],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        ),
    ];
    let mut add_tensor = |name: String, shape: Vec<usize>, value: f32| {
        descriptors.push((name, shape.clone(), vec![value; shape.iter().product()]));
    };
    for layer_id in 0..2 {
        let prefix = format!("model.layers.{layer_id}");
        add_tensor(format!("{prefix}.input_layernorm.weight"), vec![2], 1.0);
        add_tensor(
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![2],
            1.0,
        );
        for (suffix, shape) in [
            ("self_attn.q_a_proj.weight", vec![2, 2]),
            ("self_attn.q_a_layernorm.weight", vec![2]),
            ("self_attn.q_b_proj.weight", vec![3, 2]),
            ("self_attn.kv_a_proj_with_mqa.weight", vec![4, 2]),
            ("self_attn.kv_a_layernorm.weight", vec![2]),
            ("self_attn.kv_b_proj.weight", vec![3, 2]),
            ("self_attn.o_proj.weight", vec![2, 2]),
        ] {
            add_tensor(format!("{prefix}.{suffix}"), shape, 0.0);
        }
    }
    for suffix in ["gate_proj", "up_proj", "down_proj"] {
        add_tensor(
            format!("model.layers.0.mlp.{suffix}.weight"),
            vec![2, 2],
            0.0,
        );
    }
    for (suffix, shape) in [
        ("gate.weight", vec![1, 2]),
        ("gate.e_score_correction_bias", vec![1]),
        ("experts.0.w1.weight", vec![2, 2]),
        ("experts.0.w2.weight", vec![2, 2]),
        ("experts.0.w3.weight", vec![2, 2]),
        ("shared_experts.gate_proj.weight", vec![2, 2]),
        ("shared_experts.up_proj.weight", vec![2, 2]),
        ("shared_experts.down_proj.weight", vec![2, 2]),
    ] {
        add_tensor(format!("model.layers.1.mlp.{suffix}"), shape, 0.0);
    }

    let mut payload = Vec::new();
    let mut tensors = Vec::new();
    for (name, shape, values) in &descriptors {
        let start = payload.len();
        payload.extend(values.iter().flat_map(|value| value.to_le_bytes()));
        tensors.push((
            name.as_str(),
            "F32",
            shape.as_slice(),
            [start, payload.len()],
        ));
    }
    write_safetensors_file(&model_dir.join("model.safetensors"), &tensors, &payload)
        .expect("DeepSeek V3 weights should be written");
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
