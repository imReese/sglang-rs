#![cfg(feature = "test-support")]

use std::ffi::c_void;
use std::fs;
use std::net::TcpListener;
use std::path::PathBuf;

use tokio::io::{AsyncReadExt as TokioAsyncReadExt, AsyncWriteExt as TokioAsyncWriteExt};
use tokio::sync::oneshot;
use tonic::Request;
use tonic::transport::Channel;

use sglang_srt::cli::ServerArgs;
use sglang_srt::model_artifacts::ModelArtifactError;
use sglang_srt::model_registry::ModelRegistryError;
use sglang_srt::pd_bootstrap::PrefillBootstrapService;
use sglang_srt::proto::sglang::runtime::v1::generate_response::Body;
use sglang_srt::proto::sglang::runtime::v1::sglang_service_client::SglangServiceClient;
use sglang_srt::proto::sglang::runtime::v1::sglang_service_server::SglangService;
use sglang_srt::proto::sglang::runtime::v1::{
    GetModelInfoRequest, GetServerInfoRequest, HealthCheckRequest, RequestOptions, SamplingParams,
    TextGenerateRequest, TokenizeRequest,
};
use sglang_srt::router::RouterGetModelInfoResponse;
use sglang_srt::server::test_support::{
    build_reference_fake_pd_grpc_router_service, build_reference_grpc_router_service,
    build_reference_pd_grpc_router_service, try_build_reference_grpc_router_service,
    try_build_reference_prefill_http_router_service,
};
use sglang_srt::server::{
    ServerLaunchError, grpc_http_sidecar_listen_addr, grpc_listen_addr, launch_grpc_server,
    launch_grpc_server_with_shutdown, prefill_mooncake_zmq_endpoints,
    register_prefill_mooncake_routes_from_args, try_build_bootstrap_grpc_router_service,
    try_build_bootstrap_prefill_http_router_service,
};
use sglang_srt::tokenizer::TokenizerError;
use sglang_srt::transfer::{
    DecodeBootstrapRegistry, DisaggregationMode, KvCacheMemoryLocation, KvTransferBackend,
    MooncakeBatchId, MooncakeBatchReleaser, MooncakeBufferEntry, MooncakeError,
    MooncakeKvCacheLayout, MooncakeKvCacheTransferExecutor, MooncakeMemoryRegistrar,
    MooncakeTransferRequest, MooncakeTransferStatus, MooncakeTransferStatusCode,
    MooncakeTransferStatusReader, MooncakeTransferSubmitter, MooncakeTransferTarget,
    TransferBackend, TransferableKvCacheMemory, TransferableKvCacheRegion,
};

#[test]
fn grpc_listen_addr_uses_server_host_and_port() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--host",
        "127.0.0.1",
        "--port",
        "30001",
        "--grpc-mode",
    ])
    .expect("args should parse");

    let addr = grpc_listen_addr(&args).expect("listen address should resolve");

    assert_eq!(addr.ip().to_string(), "127.0.0.1");
    assert_eq!(addr.port(), 30001);
}

#[test]
fn grpc_http_sidecar_address_matches_community_defaults_and_override() {
    let default_args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--host",
        "127.0.0.1",
        "--port",
        "30001",
        "--smg-grpc-mode",
    ])
    .expect("args should parse");
    let default_addr = grpc_http_sidecar_listen_addr(&default_args)
        .expect("default sidecar address should resolve");
    assert_eq!(default_addr.port(), 30002);

    let explicit_args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--port",
        "65535",
        "--smg-http-sidecar-port",
        "30100",
    ])
    .expect("args should parse");
    let explicit_addr = grpc_http_sidecar_listen_addr(&explicit_args)
        .expect("explicit sidecar address should resolve");
    assert_eq!(explicit_addr.port(), 30100);

    let overflow_args =
        ServerArgs::parse_from(["serve", "--model-path", "dummy", "--port", "65535"])
            .expect("args should parse");
    assert_eq!(
        grpc_http_sidecar_listen_addr(&overflow_args),
        Err(ServerLaunchError::InvalidGrpcSidecarPort { grpc_port: 65535 })
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn production_grpc_launch_starts_and_stops_community_http_sidecar() {
    let model_dir = temp_model_dir("grpc-launch-sidecar");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_complete_qwen2_checkpoint(&model_dir);
    let grpc_addr = unused_local_addr();
    let sidecar_addr = unused_local_addr();
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("model dir should be UTF-8"),
        "--device",
        "cpu",
        "--host",
        "127.0.0.1",
        "--port",
        &grpc_addr.port().to_string(),
        "--smg-grpc-mode",
        "--smg-http-sidecar-port",
        &sidecar_addr.port().to_string(),
        "--enable-metrics",
    ])
    .expect("args should parse");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(launch_grpc_server_with_shutdown(args, async move {
        let _ = shutdown_rx.await;
    }));

    let mut client = connect_grpc_with_retry(grpc_addr).await;
    let health = client
        .health_check(HealthCheckRequest {})
        .await
        .expect("gRPC health check should execute")
        .into_inner();
    assert!(health.healthy);
    let metrics = request_sidecar_with_retry(sidecar_addr, "/metrics").await;
    assert!(metrics.starts_with("HTTP/1.1 200"), "{metrics}");
    assert!(metrics.contains("sglang_requests_total 0\n"), "{metrics}");

    shutdown_tx
        .send(())
        .expect("production gRPC server should still run");
    server
        .await
        .expect("production gRPC task should join")
        .expect("production gRPC and sidecar should stop cleanly");
    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn production_grpc_launch_fails_when_http_sidecar_port_is_occupied() {
    let model_dir = temp_model_dir("grpc-launch-sidecar-conflict");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_complete_qwen2_checkpoint(&model_dir);
    let grpc_addr = unused_local_addr();
    let occupied_sidecar =
        TcpListener::bind(("127.0.0.1", 0)).expect("sidecar conflict listener should bind");
    let sidecar_addr = occupied_sidecar
        .local_addr()
        .expect("sidecar conflict address should resolve");
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("model dir should be UTF-8"),
        "--device",
        "cpu",
        "--host",
        "127.0.0.1",
        "--port",
        &grpc_addr.port().to_string(),
        "--smg-grpc-mode",
        "--smg-http-sidecar-port",
        &sidecar_addr.port().to_string(),
        "--enable-metrics",
    ])
    .expect("args should parse");

    let error = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        launch_grpc_server_with_shutdown(args, std::future::pending::<()>()),
    )
    .await
    .expect("sidecar bind failure should return promptly")
    .expect_err("occupied sidecar port must fail production launch");
    assert!(
        matches!(
            error,
            ServerLaunchError::Http(sglang_srt::http::HttpServeError::Io(ref io_error))
                if io_error.kind() == std::io::ErrorKind::AddrInUse
        ),
        "unexpected error: {error:?}"
    );

    drop(occupied_sidecar);
    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn bootstrap_pd_service_rejects_unaligned_kv_slot_capacity_before_serving() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--page-size",
        "4",
        "--num-reserved-decode-tokens",
        "10",
    ])
    .expect("args should parse");

    let error = match try_build_reference_prefill_http_router_service(&args) {
        Ok(_) => panic!("unaligned KV slot capacity should fail during service construction"),
        Err(error) => error,
    };

    assert!(
        matches!(
            &error,
            ServerLaunchError::KvCacheTransfer(message)
                if message.contains("slot capacity 10 must be divisible by page size 4")
        ),
        "unexpected error: {error:?}"
    );
}

#[test]
fn bootstrap_cuda_device_rejects_missing_model_without_fallback() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--device",
        "cuda",
        "--grpc-mode",
    ])
    .expect("args should parse");

    let error = match try_build_bootstrap_grpc_router_service(&args) {
        Ok(_) => panic!("cuda device must not fall back when model artifacts are missing"),
        Err(error) => error,
    };

    assert!(
        matches!(
            error,
            ServerLaunchError::ModelRegistry(ModelRegistryError::ModelArtifact(
                ModelArtifactError::ModelPathNotLocalDirectory { .. }
            ))
        ),
        "unexpected error: {error:?}"
    );
}

#[test]
fn bootstrap_rejects_tensor_parallel_before_loading_model_artifacts() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "missing-model",
        "--device",
        "cuda",
        "--tp-size",
        "2",
        "--grpc-mode",
    ])
    .expect("args should parse");

    let error = match try_build_bootstrap_grpc_router_service(&args) {
        Ok(_) => panic!("tensor parallel must fail before model artifacts are loaded"),
        Err(error) => error,
    };

    assert!(
        matches!(
            error,
            ServerLaunchError::ModelRegistry(ModelRegistryError::BackendInitialization {
                requested: sglang_srt::backend::RuntimeBackend::Cuda,
                ref message,
            }) if message.contains("WorkerGroup")
                && message.contains("rank lifecycle")
                && message.contains("collective backend")
                && message.contains("tp_size=1")
                && message.contains("requested 2")
        ),
        "unexpected error: {error:?}"
    );
}

#[test]
fn unsupported_accelerator_fails_before_cpu_weight_materialization() {
    let model_dir = temp_model_dir("unsupported-accelerator-fast-fail");
    fs::create_dir_all(&model_dir).expect("temp model directory should be created");
    write_complete_qwen3_checkpoint(&model_dir);
    write_safetensors_file(
        &model_dir.join("model.safetensors"),
        &[("unrelated.weight", "U8", &[1], [0, 1])],
        &[0],
    )
    .expect("minimal safetensors file should be written");

    for device in ["metal", "rocm", "musa", "xpu", "npu", "hpu"] {
        let args = ServerArgs::parse_from([
            "serve",
            "--model-path",
            model_dir.to_str().expect("temp model path should be utf-8"),
            "--device",
            device,
            "--grpc-mode",
        ])
        .expect("server args should parse");
        let error = match try_build_bootstrap_grpc_router_service(&args) {
            Ok(_) => panic!("unsupported accelerator {device} must fail before serving"),
            Err(error) => error,
        };

        assert!(
            matches!(
                error,
                ServerLaunchError::ModelRegistry(ModelRegistryError::BackendInitialization {
                    requested,
                    ref message,
                })
                    if requested.as_str() == device
                        && message.contains("backend provider is not registered")
            ),
            "unexpected {device} error: {error:?}"
        );
    }

    fs::remove_dir_all(model_dir).expect("temp model directory should be removed");
}

#[tokio::test]
async fn launch_rejects_missing_model_without_reference_fallback() {
    let args = ServerArgs::parse_from(["serve", "--model-path", "dummy", "--grpc-mode"])
        .expect("args should parse");

    let error = launch_grpc_server(args)
        .await
        .expect_err("production launch must reject missing model artifacts");

    assert!(
        matches!(
            error,
            ServerLaunchError::ModelRegistry(ModelRegistryError::ModelArtifact(
                ModelArtifactError::ModelPathNotLocalDirectory { .. }
            ))
        ),
        "unexpected error: {error:?}"
    );
    assert!(error.to_string().contains("not a local directory"));
}

#[tokio::test]
async fn bootstrap_grpc_router_service_carries_model_metadata() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "meta-llama/Llama-3.1-8B-Instruct",
        "--served-model-name",
        "llama3",
        "--tokenizer-path",
        "hf-tokenizer",
        "--grpc-mode",
    ])
    .expect("args should parse");
    let service = build_reference_grpc_router_service(&args);

    let response = service
        .get_model_info(Request::new(GetModelInfoRequest {}))
        .await
        .expect("model info should execute")
        .into_inner();

    assert_eq!(response.model_path, "meta-llama/Llama-3.1-8B-Instruct");
    assert_eq!(response.tokenizer_path, "hf-tokenizer");
    assert_eq!(response.served_model_name, "llama3");
}

#[tokio::test]
async fn bootstrap_grpc_router_service_reports_local_model_config_metadata() {
    let model_dir = temp_model_dir("server-model-config");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        deepseek_v4_model_config_json(),
    )
    .expect("config should be written");
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model dir should be utf-8"),
        "--grpc-mode",
    ])
    .expect("args should parse");
    let service = build_reference_grpc_router_service(&args);

    let response = service
        .get_model_info(Request::new(GetModelInfoRequest {}))
        .await
        .expect("model info should execute")
        .into_inner();

    assert_eq!(response.model_type, "deepseek_v4");
    assert_eq!(response.vocab_size, 129_280);
    assert_eq!(response.max_context_length, 163_840);
    assert_eq!(response.max_request_input_length, 163_840);

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn bootstrap_grpc_router_service_reports_local_moe_checkpoint_coverage() {
    let model_dir = temp_model_dir("server-moe-checkpoint-coverage");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "deepseek_v4",
  "architectures": ["DeepseekV4ForCausalLM"],
  "vocab_size": 1,
  "max_position_embeddings": 32,
  "num_hidden_layers": 1,
  "hidden_size": 1,
  "num_attention_heads": 1,
  "hc_mult": 1,
  "n_routed_experts": 1,
  "num_experts_per_tok": 1,
  "moe_intermediate_size": 1,
  "first_k_dense_replace": 0,
  "moe_layer_freq": 1,
  "num_key_value_heads": 1,
  "qk_nope_head_dim": 64,
  "qk_rope_head_dim": 32,
  "v_head_dim": 64
}"#,
    )
    .expect("config should be written");
    fs::write(
        model_dir.join("model.safetensors.index.json"),
        r#"{
  "weight_map": {
    "model.embed_tokens.weight": "model.safetensors",
    "model.norm.weight": "model.safetensors",
    "lm_head.weight": "model.safetensors",
    "model.hc_head_fn": "model.safetensors",
    "model.hc_head_base": "model.safetensors",
    "model.hc_head_scale": "model.safetensors",
    "model.layers.0.self_attn.wq_a.weight": "model.safetensors",
    "model.layers.0.self_attn.wq_b.weight": "model.safetensors",
    "model.layers.0.self_attn.wkv.weight": "model.safetensors",
    "model.layers.0.self_attn.q_norm.weight": "model.safetensors",
    "model.layers.0.self_attn.kv_norm.weight": "model.safetensors",
    "model.layers.0.self_attn.wo_a.weight": "model.safetensors",
    "model.layers.0.self_attn.wo_b.weight": "model.safetensors",
    "model.layers.0.input_layernorm.weight": "model.safetensors",
    "model.layers.0.post_attention_layernorm.weight": "model.safetensors",
    "model.layers.0.hc_attn_fn": "model.safetensors",
    "model.layers.0.hc_attn_base": "model.safetensors",
    "model.layers.0.hc_attn_scale": "model.safetensors",
    "model.layers.0.hc_ffn_fn": "model.safetensors",
    "model.layers.0.hc_ffn_base": "model.safetensors",
    "model.layers.0.hc_ffn_scale": "model.safetensors",
    "model.layers.0.mlp.gate.weight": "model.safetensors",
    "model.layers.0.ffn.experts.0.w1.weight": "model.safetensors",
    "model.layers.0.ffn.experts.0.w2.weight": "model.safetensors",
    "model.layers.0.ffn.experts.0.w3.weight": "model.safetensors"
  }
}"#,
    )
    .expect("index should be written");
    write_safetensors_file(
        &model_dir.join("model.safetensors"),
        &[
            ("model.embed_tokens.weight", "U8", &[1], [0, 1]),
            ("model.norm.weight", "U8", &[1], [1, 2]),
            ("lm_head.weight", "U8", &[1], [2, 3]),
            ("model.hc_head_fn", "U8", &[1, 1], [3, 4]),
            ("model.hc_head_base", "U8", &[1], [4, 5]),
            ("model.hc_head_scale", "U8", &[1], [5, 6]),
            ("model.layers.0.self_attn.wq_a.weight", "U8", &[1], [6, 7]),
            ("model.layers.0.self_attn.wq_b.weight", "U8", &[1], [7, 8]),
            ("model.layers.0.self_attn.wkv.weight", "U8", &[1], [8, 9]),
            (
                "model.layers.0.self_attn.q_norm.weight",
                "U8",
                &[1],
                [9, 10],
            ),
            (
                "model.layers.0.self_attn.kv_norm.weight",
                "U8",
                &[1],
                [10, 11],
            ),
            ("model.layers.0.self_attn.wo_a.weight", "U8", &[1], [11, 12]),
            ("model.layers.0.self_attn.wo_b.weight", "U8", &[1], [12, 13]),
            (
                "model.layers.0.input_layernorm.weight",
                "U8",
                &[1],
                [13, 14],
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "U8",
                &[1],
                [14, 15],
            ),
            ("model.layers.0.hc_attn_fn", "U8", &[1], [15, 16]),
            ("model.layers.0.hc_attn_base", "U8", &[1], [16, 17]),
            ("model.layers.0.hc_attn_scale", "U8", &[1], [17, 18]),
            ("model.layers.0.hc_ffn_fn", "U8", &[1], [18, 19]),
            ("model.layers.0.hc_ffn_base", "U8", &[1], [19, 20]),
            ("model.layers.0.hc_ffn_scale", "U8", &[1], [20, 21]),
            ("model.layers.0.mlp.gate.weight", "U8", &[1], [21, 22]),
            (
                "model.layers.0.ffn.experts.0.w1.weight",
                "U8",
                &[1],
                [22, 23],
            ),
            (
                "model.layers.0.ffn.experts.0.w2.weight",
                "U8",
                &[1],
                [23, 24],
            ),
            (
                "model.layers.0.ffn.experts.0.w3.weight",
                "U8",
                &[1],
                [24, 25],
            ),
        ],
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25,
        ],
    )
    .expect("shard should be written");
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model dir should be utf-8"),
        "--grpc-mode",
    ])
    .expect("args should parse");
    let response = RouterGetModelInfoResponse::from_server_args(&args);

    assert_eq!(response.routed_expert_expected_group_count, 1);
    assert_eq!(response.routed_expert_actual_group_count, 1);
    assert_eq!(response.routed_expert_expected_weight_count, 3);
    assert_eq!(response.routed_expert_actual_weight_count, 3);

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn bootstrap_grpc_router_service_rejects_generation_for_unsupported_local_model_runtime() {
    let model_dir = temp_model_dir("server-unsupported-local-forward");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_complete_deepseek_v4_checkpoint(&model_dir);
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model dir should be utf-8"),
        "--device",
        "cpu",
        "--grpc-mode",
    ])
    .expect("args should parse");
    let error = match try_build_bootstrap_grpc_router_service(&args) {
        Ok(_) => panic!("DeepSeek without a forward executor must fail during startup"),
        Err(error) => error,
    };

    assert!(
        matches!(
            error,
            ServerLaunchError::ModelRegistry(ModelRegistryError::MissingCapabilities {
                architecture: "DeepseekV4ForCausalLM",
                backend: sglang_srt::backend::RuntimeBackend::Cpu,
                ref missing,
                ..
            }) if missing.iter().any(|capability| capability.contains("bfloat16") && capability.contains("float32"))
                && missing.iter().any(|capability| capability.contains("multi-latent attention") && capability.contains("mixture-of-experts"))
                && missing.iter().any(|capability| capability == "runtime-owned KV cache allocation")
        ),
        "unexpected error: {error:?}"
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn bootstrap_rejects_glm_through_the_shared_mla_moe_runtime_preflight() {
    let model_dir = temp_model_dir("server-glm-runtime-forward");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_complete_glm_moe_dsa_forward_checkpoint(&model_dir);
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model dir should be utf-8"),
        "--device",
        "cpu",
        "--grpc-mode",
    ])
    .expect("args should parse");
    let error = match try_build_bootstrap_grpc_router_service(&args) {
        Ok(_) => panic!("GLM must not bypass the shared MLA/MoE runtime capability preflight"),
        Err(error) => error,
    };

    assert!(
        matches!(
            error,
            ServerLaunchError::ModelRegistry(ModelRegistryError::MissingCapabilities {
                architecture: "GlmMoeDsaForCausalLM",
                backend: sglang_srt::backend::RuntimeBackend::Cpu,
                ref missing,
                ..
            }) if missing.iter().any(|capability| capability.contains("bfloat16") && capability.contains("float32"))
                && missing.iter().any(|capability| capability.contains("multi-latent attention") && capability.contains("mixture-of-experts"))
                && missing.iter().any(|capability| capability == "runtime-owned KV cache allocation")
        ),
        "unexpected error: {error:?}"
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[tokio::test]
async fn bootstrap_runs_qwen_centralized_prefill_and_decode_without_transfer() {
    let model_dir = temp_model_dir("server-qwen-runtime-forward");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_complete_qwen2_checkpoint(&model_dir);
    assert_centralized_qwen_generation(&model_dir, "centralized-qwen").await;

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[tokio::test]
async fn bootstrap_runs_qwen3_centralized_prefill_and_decode_without_transfer() {
    let model_dir = temp_model_dir("server-qwen3-runtime-forward");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_complete_qwen3_checkpoint(&model_dir);
    assert_centralized_qwen_generation(&model_dir, "centralized-qwen3").await;

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[tokio::test]
async fn bootstrap_runs_qwen3_5_hybrid_centralized_prefill_and_decode_without_transfer() {
    let model_dir = temp_model_dir("server-qwen35-runtime-forward");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_complete_qwen3_5_checkpoint(&model_dir);
    assert_centralized_qwen_generation(&model_dir, "centralized-qwen35").await;

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[tokio::test]
async fn bootstrap_runs_kimi_linear_hybrid_centralized_prefill_and_decode_without_transfer() {
    let model_dir = temp_model_dir("server-kimi-linear-runtime-forward");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_complete_kimi_linear_checkpoint(&model_dir);
    assert_centralized_generation(&model_dir, "centralized-kimi-linear").await;

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[tokio::test]
async fn bootstrap_runs_kimi_k2_compatible_deepseek_v3_mla_moe_without_transfer() {
    let model_dir = temp_model_dir("server-kimi-k2-deepseek-v3-forward");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_complete_deepseek_v3_checkpoint(&model_dir);
    assert_centralized_generation_with_kv_layers(&model_dir, "centralized-kimi-k2-deepseek-v3", 2)
        .await;

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn qwen3_5_pd_startup_rejects_kv_only_transfer_before_serving() {
    let model_dir = temp_model_dir("server-qwen35-pd-fail-fast");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_complete_qwen3_5_checkpoint(&model_dir);
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model dir should be utf-8"),
        "--device",
        "cpu",
        "--num-reserved-decode-tokens",
        "16",
    ])
    .expect("args should parse");

    let error = match try_build_bootstrap_prefill_http_router_service(&args) {
        Ok(_) => panic!("hybrid state must not start with KV-only transfer"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        ServerLaunchError::MissingRuntimeCapabilities { ref missing, .. }
            if missing.iter().any(|item| item.contains("hybrid recurrent state") && item.contains("KV-only"))
    ));
    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

async fn assert_centralized_qwen_generation(model_dir: &std::path::Path, request_id: &str) {
    assert_centralized_generation(model_dir, request_id).await;
}

async fn assert_centralized_generation(model_dir: &std::path::Path, request_id: &str) {
    assert_centralized_generation_with_kv_layers(model_dir, request_id, 1).await;
}

async fn assert_centralized_generation_with_kv_layers(
    model_dir: &std::path::Path,
    request_id: &str,
    expected_kv_layers: usize,
) {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model dir should be utf-8"),
        "--device",
        "cpu",
        "--grpc-mode",
        "--num-reserved-decode-tokens",
        "16",
    ])
    .expect("args should parse");
    let service = try_build_bootstrap_grpc_router_service(&args)
        .expect("model should start on the CPU reference backend");
    let server_info = service
        .get_server_info(Request::new(GetServerInfoRequest {}))
        .await
        .expect("active model KV metadata should be observable")
        .into_inner();
    assert_eq!(
        server_info.attributes.get("kv_cache.dtype"),
        Some(&"float32".to_string())
    );
    assert_eq!(
        server_info.attributes.get("kv_cache.num_layers"),
        Some(&expected_kv_layers.to_string())
    );
    assert_eq!(
        server_info.attributes.get("kv_cache.page_size"),
        Some(&"1".to_string())
    );
    assert_eq!(
        server_info.attributes.get("kv_cache.page_size_bytes"),
        server_info.attributes.get("kv_cache.bytes_per_token")
    );
    let mut stream = service
        .text_generate(Request::new(TextGenerateRequest {
            text: "hello".to_string(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(2),
                top_k: Some(1),
                ..Default::default()
            }),
            options: Some(RequestOptions {
                request_id: Some(request_id.to_string()),
                stream: true,
                data_parallel_rank: 0,
                trace_headers: Default::default(),
            }),
            disaggregated_params: None,
        }))
        .await
        .expect("centralized model generation should execute")
        .into_inner();

    let first = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("prefill response")
        .expect("prefill response should succeed");
    let second = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("decode response")
        .expect("decode response should succeed");
    let third = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("completion response")
        .expect("completion response should succeed");

    assert_eq!(first.request_id, request_id);
    assert!(matches!(first.body, Some(Body::Chunk(_))));
    assert_eq!(second.request_id, request_id);
    assert!(matches!(second.body, Some(Body::Chunk(_))));
    assert_eq!(third.request_id, request_id);
    assert!(
        matches!(
            third.body,
            Some(Body::Complete(ref complete))
                if complete.output_ids == [2, 1]
                    && complete.prompt_tokens == 1
                    && complete.completion_tokens == 2
                    && complete.cached_tokens == 0
                    && complete.finish_reason == "stop"
        ),
        "unexpected completion response: {third:?}"
    );
    assert!(
        tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
            .await
            .is_none()
    );
}

#[tokio::test]
async fn bootstrap_grpc_router_service_generates_through_model_runner() {
    let args = ServerArgs::parse_from(["serve", "--model-path", "dummy", "--grpc-mode"])
        .expect("args should parse");
    let service = build_reference_grpc_router_service(&args);

    let mut stream = service
        .text_generate(Request::new(TextGenerateRequest {
            text: "hello".to_string(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(4),
                stop_token_ids: vec![b' ' as u32],
                ..Default::default()
            }),
            options: Some(RequestOptions {
                request_id: Some("bootstrap-generate".to_string()),
                stream: true,
                data_parallel_rank: 0,
                trace_headers: Default::default(),
            }),
            disaggregated_params: None,
        }))
        .await
        .expect("text generate should execute")
        .into_inner();

    let chunk = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("token response")
        .expect("token response should be ok");
    let response = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("completion response")
        .expect("completion response should be ok");

    assert!(matches!(chunk.body, Some(Body::Chunk(_))));
    assert_eq!(response.request_id, "bootstrap-generate");
    assert_eq!(
        response.body,
        Some(Body::Complete(
            sglang_srt::proto::sglang::runtime::v1::GenerateComplete {
                output_ids: vec![b' ' as u32],
                text: " ".to_string(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 5,
                completion_tokens: 1,
                cached_tokens: 0,
                index: 0,
            }
        ))
    );
    assert!(
        tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
            .await
            .is_none()
    );
}

#[tokio::test]
async fn bootstrap_grpc_router_service_uses_config_eos_token_as_default_stop() {
    let model_dir = temp_model_dir("server-config-eos-stop");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "llama",
  "eos_token_id": 32
}"#,
    )
    .expect("config should be written");
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model dir should be utf-8"),
        "--grpc-mode",
    ])
    .expect("args should parse");
    let service = build_reference_grpc_router_service(&args);

    let mut stream = service
        .text_generate(Request::new(TextGenerateRequest {
            text: "hello".to_string(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(4),
                ..Default::default()
            }),
            options: Some(RequestOptions {
                request_id: Some("bootstrap-config-eos-stop".to_string()),
                stream: true,
                data_parallel_rank: 0,
                trace_headers: Default::default(),
            }),
            disaggregated_params: None,
        }))
        .await
        .expect("text generate should execute")
        .into_inner();

    let chunk = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("token response")
        .expect("token response should be ok");
    let response = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("completion response")
        .expect("completion response should be ok");

    assert!(matches!(chunk.body, Some(Body::Chunk(_))));
    assert_eq!(response.request_id, "bootstrap-config-eos-stop");
    assert_eq!(
        response.body,
        Some(Body::Complete(
            sglang_srt::proto::sglang::runtime::v1::GenerateComplete {
                output_ids: vec![b' ' as u32],
                text: " ".to_string(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 5,
                completion_tokens: 1,
                cached_tokens: 0,
                index: 0,
            }
        ))
    );
    assert!(
        tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
            .await
            .is_none()
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[tokio::test]
async fn bootstrap_grpc_router_service_ignores_config_eos_when_requested() {
    let model_dir = temp_model_dir("server-ignore-config-eos");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "llama",
  "eos_token_id": 32
}"#,
    )
    .expect("config should be written");
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model dir should be utf-8"),
        "--grpc-mode",
    ])
    .expect("args should parse");
    let service = build_reference_grpc_router_service(&args);

    let mut stream = service
        .text_generate(Request::new(TextGenerateRequest {
            text: "hello".to_string(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(2),
                ignore_eos: Some(true),
                ..Default::default()
            }),
            options: Some(RequestOptions {
                request_id: Some("bootstrap-ignore-config-eos".to_string()),
                stream: true,
                data_parallel_rank: 0,
                trace_headers: Default::default(),
            }),
            disaggregated_params: None,
        }))
        .await
        .expect("text generate should execute")
        .into_inner();

    let first = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("first response")
        .expect("response should be ok");
    let second = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("second response")
        .expect("response should be ok");
    let third = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("completion response")
        .expect("completion response should be ok");

    assert_eq!(first.request_id, "bootstrap-ignore-config-eos");
    assert_eq!(
        first.body,
        Some(Body::Chunk(
            sglang_srt::proto::sglang::runtime::v1::GenerateStreamChunk {
                token_ids: vec![b' ' as u32],
                text: " ".to_string(),
                prompt_tokens: 5,
                completion_tokens: 1,
                cached_tokens: 0,
                index: 0,
            }
        ))
    );
    assert_eq!(
        second.body,
        Some(Body::Chunk(
            sglang_srt::proto::sglang::runtime::v1::GenerateStreamChunk {
                token_ids: vec![b' ' as u32],
                text: " ".to_string(),
                prompt_tokens: 5,
                completion_tokens: 2,
                cached_tokens: 0,
                index: 0,
            }
        ))
    );
    assert_eq!(
        third.body,
        Some(Body::Complete(
            sglang_srt::proto::sglang::runtime::v1::GenerateComplete {
                output_ids: vec![b' ' as u32, b' ' as u32],
                text: "  ".to_string(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 5,
                completion_tokens: 2,
                cached_tokens: 0,
                index: 0,
            }
        ))
    );
    assert!(
        tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
            .await
            .is_none()
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[tokio::test]
async fn bootstrap_grpc_router_service_uses_local_hf_tokenizer_when_available() {
    let model_dir = temp_model_dir("server-hf-tokenizer");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("tokenizer.json"),
        word_level_tokenizer_json(),
    )
    .expect("tokenizer.json should be written");
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model dir should be utf-8"),
        "--grpc-mode",
    ])
    .expect("args should parse");
    let service = build_reference_grpc_router_service(&args);

    let response = service
        .tokenize(Request::new(TokenizeRequest {
            text: "hello world".to_string(),
            add_special_tokens: true,
        }))
        .await
        .expect("tokenize should execute")
        .into_inner();

    assert_eq!(response.count, 2);
    assert_eq!(response.token_ids, vec![1, 2]);

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn bootstrap_grpc_router_service_rejects_missing_explicit_tokenizer_path() {
    let model_dir = temp_model_dir("server-missing-tokenizer");
    let tokenizer_dir = model_dir.join("missing-tokenizer");
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model dir should be utf-8"),
        "--tokenizer-path",
        tokenizer_dir
            .to_str()
            .expect("temp tokenizer dir should be utf-8"),
        "--grpc-mode",
    ])
    .expect("args should parse");

    let error = match try_build_reference_grpc_router_service(&args) {
        Ok(_) => panic!("explicit missing tokenizer path should fail"),
        Err(error) => error,
    };

    assert_eq!(
        error,
        ServerLaunchError::Tokenizer(TokenizerError::TokenizerFileNotFound {
            path: tokenizer_dir
        })
    );
}

#[tokio::test]
async fn bootstrap_pd_grpc_router_service_polls_transfer_before_decode() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--grpc-mode",
        "--disaggregation-mode",
        "decode",
        "--disaggregation-decode-polling-interval",
        "1",
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("args should parse");
    let mut transfer_executor = MooncakeKvCacheTransferExecutor::new(
        RecordingMooncakeBackend::completed(),
        MooncakeKvCacheLayout {
            source_base_addr: 0x3000,
            page_size_bytes: 64,
            target_base_offset: 0,
        },
        MooncakeTransferTarget { target_id: 17 },
    )
    .with_memory_registrar(RecordingMooncakeBackend::default());
    transfer_executor
        .register(
            TransferableKvCacheMemory::new(
                vec![TransferableKvCacheRegion {
                    base_addr: 0x3000,
                    byte_len: 64 * 8,
                    page_size_bytes: 64,
                }],
                64,
                KvCacheMemoryLocation::Cpu { numa_node: 0 },
            )
            .expect("test NexusKV descriptor should be valid"),
        )
        .expect("test Mooncake executor should register active KV memory");
    let service = build_reference_pd_grpc_router_service(
        &args,
        DecodeBootstrapRegistry::default(),
        transfer_executor,
    );

    let mut stream = service
        .text_generate(Request::new(TextGenerateRequest {
            text: "hi".to_string(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            options: Some(RequestOptions {
                request_id: Some("bootstrap-pd".to_string()),
                stream: true,
                data_parallel_rank: 0,
                trace_headers: Default::default(),
            }),
            disaggregated_params: Some(
                sglang_srt::proto::sglang::runtime::v1::DisaggregatedParams {
                    bootstrap_host: "10.0.0.9".to_string(),
                    bootstrap_port: 8998,
                    bootstrap_room: 41,
                },
            ),
        }))
        .await
        .expect("PD bootstrap service should poll transfer and generate")
        .into_inner();

    let first = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("first response")
        .expect("first response should be ok");
    let second = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("second response")
        .expect("second response should be ok");
    let third = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("final response")
        .expect("final response should be ok");

    assert_eq!(first.request_id, "bootstrap-pd");
    assert!(matches!(first.body, Some(Body::Chunk(_))));
    assert_eq!(second.request_id, "bootstrap-pd");
    assert!(matches!(second.body, Some(Body::Chunk(_))));
    assert_eq!(third.request_id, "bootstrap-pd");
    assert!(matches!(third.body, Some(Body::Complete(_))));
    assert!(
        tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
            .await
            .is_none()
    );
}

#[tokio::test]
async fn bootstrap_fake_pd_grpc_router_service_uses_decode_transfer_path() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--grpc-mode",
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "fake",
        "--disaggregation-decode-polling-interval",
        "1",
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("args should parse");
    let service = build_reference_fake_pd_grpc_router_service(&args);

    let mut stream = service
        .text_generate(Request::new(TextGenerateRequest {
            text: "hi".to_string(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            options: Some(RequestOptions {
                request_id: Some("bootstrap-fake-pd".to_string()),
                stream: true,
                data_parallel_rank: 1,
                trace_headers: Default::default(),
            }),
            disaggregated_params: Some(
                sglang_srt::proto::sglang::runtime::v1::DisaggregatedParams {
                    bootstrap_host: "10.0.0.9".to_string(),
                    bootstrap_port: 8998,
                    bootstrap_room: 42,
                },
            ),
        }))
        .await
        .expect("fake PD bootstrap service should generate")
        .into_inner();

    let first = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("first response")
        .expect("first response should be ok");
    let second = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("second response")
        .expect("second response should be ok");
    let third = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("final response")
        .expect("final response should be ok");

    assert_eq!(first.request_id, "bootstrap-fake-pd");
    assert!(matches!(first.body, Some(Body::Chunk(_))));
    assert_eq!(second.request_id, "bootstrap-fake-pd");
    assert!(matches!(second.body, Some(Body::Chunk(_))));
    assert_eq!(third.request_id, "bootstrap-fake-pd");
    assert!(matches!(third.body, Some(Body::Complete(_))));
}

#[tokio::test]
async fn bootstrap_pd_grpc_router_service_applies_max_running_requests() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--grpc-mode",
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "fake",
        "--max-running-requests",
        "1",
    ])
    .expect("args should parse");
    let service = build_reference_fake_pd_grpc_router_service(&args);
    let runtime = service
        .runtime()
        .lock()
        .expect("runtime lock should be held");

    assert_eq!(runtime.engine().scheduler().max_running_requests(), Some(1));
}

#[tokio::test]
async fn launch_grpc_server_rejects_unsupported_nixl_pd_backend() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--grpc-mode",
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "nixl",
    ])
    .expect("args should parse");

    let error = launch_grpc_server(args)
        .await
        .expect_err("unsupported PD backend should fail before serving");

    assert_eq!(
        error,
        ServerLaunchError::UnsupportedBootstrapPdRuntime {
            mode: DisaggregationMode::Decode,
            transfer_backend: TransferBackend::Nixl,
        }
    );
}

#[cfg(not(feature = "mooncake-link"))]
#[tokio::test]
async fn launch_grpc_server_rejects_unlinked_mooncake_before_serving() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        ".",
        "--grpc-mode",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "mooncake",
    ])
    .expect("args should parse");

    let error = launch_grpc_server(args)
        .await
        .expect_err("unlinked Mooncake must fail before serving");

    assert!(
        matches!(
            error,
            ServerLaunchError::MooncakeTransfer(MooncakeError::UnavailableWithoutLink)
        ),
        "unexpected error: {error:?}"
    );
}

#[test]
fn prefill_mooncake_zmq_endpoints_follow_launch_host_and_port_range() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--host",
        "0.0.0.0",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-zmq-ports",
        "7000-7002",
    ])
    .expect("args should parse");

    assert_eq!(
        prefill_mooncake_zmq_endpoints(&args),
        vec![
            "tcp://0.0.0.0:7000".to_string(),
            "tcp://0.0.0.0:7001".to_string(),
            "tcp://0.0.0.0:7002".to_string(),
        ]
    );
}

#[test]
fn prefill_mooncake_route_registration_maps_tp_ranks_to_zmq_ports() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--host",
        "10.0.0.9",
        "--tp-size",
        "2",
        "--dp-size",
        "1",
        "--page-size",
        "64",
        "--kv-cache-dtype",
        "bfloat16",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-zmq-ports",
        "7000-7001",
    ])
    .expect("args should parse");
    let service = PrefillBootstrapService::default();

    register_prefill_mooncake_routes_from_args(&service, &args)
        .expect("prefill ZMQ routes should register");

    let state = service.state().lock().expect("state lock should be held");
    let topology = state
        .server_info()
        .expect("registered prefill routes should make topology ready");
    assert_eq!(topology.attn_tp_size, 2);
    assert_eq!(topology.dp_size, 1);
    assert_eq!(topology.page_size, Some(64));
    assert_eq!(topology.kv_cache_dtype.as_deref(), Some("bfloat16"));
    assert_eq!(
        state
            .rank_info(0, 0, 0, 0)
            .expect("TP0 endpoint should be registered")
            .rank_port,
        7000
    );
    assert_eq!(
        state
            .rank_info(0, 0, 1, 0)
            .expect("TP1 endpoint should be registered")
            .rank_port,
        7001
    );
}

#[test]
fn prefill_mooncake_route_registration_uses_dist_init_host_for_wildcard_bind_host() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--host",
        "0.0.0.0",
        "--dist-init-addr",
        "192.0.2.21:6676",
        "--tp-size",
        "1",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-zmq-ports",
        "7000-7000",
    ])
    .expect("args should parse");
    let service = PrefillBootstrapService::default();

    register_prefill_mooncake_routes_from_args(&service, &args)
        .expect("prefill ZMQ routes should register");

    let state = service.state().lock().expect("state lock should be held");
    assert_eq!(
        state
            .rank_info(0, 0, 0, 0)
            .expect("rank endpoint should be registered")
            .rank_ip,
        "192.0.2.21"
    );
}

#[test]
fn prefill_mooncake_route_registration_rejects_incomplete_zmq_port_range() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--tp-size",
        "2",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-zmq-ports",
        "7000-7000",
    ])
    .expect("args should parse");
    let service = PrefillBootstrapService::default();

    let error = register_prefill_mooncake_routes_from_args(&service, &args)
        .expect_err("incomplete ZMQ port range should fail route registration");

    assert_eq!(
        error,
        ServerLaunchError::ZmqRoutePortCountMismatch {
            expected: 2,
            actual: 1,
        }
    );
}

#[derive(Default)]
struct RecordingMooncakeBackend {
    submitted_batches: usize,
    statuses: Vec<MooncakeTransferStatusCode>,
    freed_batches: Vec<MooncakeBatchId>,
}

impl RecordingMooncakeBackend {
    fn completed() -> Self {
        Self {
            submitted_batches: 0,
            statuses: vec![MooncakeTransferStatusCode::Completed],
            freed_batches: Vec::new(),
        }
    }
}

impl MooncakeTransferSubmitter for RecordingMooncakeBackend {
    fn submit_transfer(
        &mut self,
        requests: &mut [MooncakeTransferRequest],
    ) -> Result<MooncakeBatchId, MooncakeError> {
        assert!(!requests.is_empty());
        self.submitted_batches += 1;
        Ok(700 + self.submitted_batches as MooncakeBatchId - 1)
    }
}

impl MooncakeTransferStatusReader for RecordingMooncakeBackend {
    fn transfer_status(
        &mut self,
        _batch_id: MooncakeBatchId,
        task_id: usize,
    ) -> Result<MooncakeTransferStatus, MooncakeError> {
        let status = self
            .statuses
            .get(task_id)
            .or_else(|| self.statuses.last())
            .copied()
            .expect("recording Mooncake backend needs at least one status");
        Ok(MooncakeTransferStatus {
            status: status as i32,
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

impl MooncakeMemoryRegistrar for RecordingMooncakeBackend {
    fn register_memory_batch(
        &mut self,
        _buffers: &mut [MooncakeBufferEntry],
        _location: &str,
    ) -> Result<(), MooncakeError> {
        Ok(())
    }

    fn unregister_memory_batch(&mut self, _addrs: &mut [*mut c_void]) -> Result<(), MooncakeError> {
        Ok(())
    }
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
      "world": 2
    },
    "unk_token": "[UNK]"
  }
}"#
}

fn deepseek_v4_model_config_json() -> &'static str {
    r#"{
  "model_type": "deepseek_v4",
  "architectures": ["DeepseekV4ForCausalLM"],
  "vocab_size": 129280,
  "max_position_embeddings": 163840,
  "num_hidden_layers": 43,
  "hidden_size": 1024,
  "num_attention_heads": 16,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "v_head_dim": 128,
  "n_routed_experts": 32,
  "num_experts_per_tok": 4,
  "moe_intermediate_size": 256,
  "hc_mult": 1
}"#
}

fn write_complete_deepseek_v4_checkpoint(model_dir: &std::path::Path) {
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "architectures": ["DeepseekV4ForCausalLM"],
  "model_type": "deepseek_v4",
  "vocab_size": 1,
  "max_position_embeddings": 32,
  "num_hidden_layers": 1,
  "hidden_size": 1,
  "num_attention_heads": 1,
  "hc_mult": 1,
  "n_routed_experts": 1,
  "num_experts_per_tok": 1,
  "moe_intermediate_size": 1,
  "first_k_dense_replace": 0,
  "moe_layer_freq": 1,
  "num_key_value_heads": 1,
  "qk_nope_head_dim": 64,
  "qk_rope_head_dim": 32,
  "v_head_dim": 64
}"#,
    )
    .expect("config should be written");
    fs::write(
        model_dir.join("model.safetensors.index.json"),
        r#"{
  "weight_map": {
    "model.embed_tokens.weight": "model.safetensors",
    "model.norm.weight": "model.safetensors",
    "lm_head.weight": "model.safetensors",
    "model.hc_head_fn": "model.safetensors",
    "model.hc_head_base": "model.safetensors",
    "model.hc_head_scale": "model.safetensors",
    "model.layers.0.self_attn.wq_a.weight": "model.safetensors",
    "model.layers.0.self_attn.wq_b.weight": "model.safetensors",
    "model.layers.0.self_attn.wkv.weight": "model.safetensors",
    "model.layers.0.self_attn.q_norm.weight": "model.safetensors",
    "model.layers.0.self_attn.kv_norm.weight": "model.safetensors",
    "model.layers.0.self_attn.wo_a.weight": "model.safetensors",
    "model.layers.0.self_attn.wo_b.weight": "model.safetensors",
    "model.layers.0.input_layernorm.weight": "model.safetensors",
    "model.layers.0.post_attention_layernorm.weight": "model.safetensors",
    "model.layers.0.hc_attn_fn": "model.safetensors",
    "model.layers.0.hc_attn_base": "model.safetensors",
    "model.layers.0.hc_attn_scale": "model.safetensors",
    "model.layers.0.hc_ffn_fn": "model.safetensors",
    "model.layers.0.hc_ffn_base": "model.safetensors",
    "model.layers.0.hc_ffn_scale": "model.safetensors",
    "model.layers.0.mlp.gate.weight": "model.safetensors",
    "model.layers.0.ffn.experts.0.w1.weight": "model.safetensors",
    "model.layers.0.ffn.experts.0.w2.weight": "model.safetensors",
    "model.layers.0.ffn.experts.0.w3.weight": "model.safetensors"
  }
}"#,
    )
    .expect("index should be written");
    write_safetensors_file(
        &model_dir.join("model.safetensors"),
        &[
            ("model.embed_tokens.weight", "U8", &[1], [0, 1]),
            ("model.norm.weight", "U8", &[1], [1, 2]),
            ("lm_head.weight", "U8", &[1], [2, 3]),
            ("model.hc_head_fn", "U8", &[1, 1], [3, 4]),
            ("model.hc_head_base", "U8", &[1], [4, 5]),
            ("model.hc_head_scale", "U8", &[1], [5, 6]),
            ("model.layers.0.self_attn.wq_a.weight", "U8", &[1], [6, 7]),
            ("model.layers.0.self_attn.wq_b.weight", "U8", &[1], [7, 8]),
            ("model.layers.0.self_attn.wkv.weight", "U8", &[1], [8, 9]),
            (
                "model.layers.0.self_attn.q_norm.weight",
                "U8",
                &[1],
                [9, 10],
            ),
            (
                "model.layers.0.self_attn.kv_norm.weight",
                "U8",
                &[1],
                [10, 11],
            ),
            ("model.layers.0.self_attn.wo_a.weight", "U8", &[1], [11, 12]),
            ("model.layers.0.self_attn.wo_b.weight", "U8", &[1], [12, 13]),
            (
                "model.layers.0.input_layernorm.weight",
                "U8",
                &[1],
                [13, 14],
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "U8",
                &[1],
                [14, 15],
            ),
            ("model.layers.0.hc_attn_fn", "U8", &[1], [15, 16]),
            ("model.layers.0.hc_attn_base", "U8", &[1], [16, 17]),
            ("model.layers.0.hc_attn_scale", "U8", &[1], [17, 18]),
            ("model.layers.0.hc_ffn_fn", "U8", &[1], [18, 19]),
            ("model.layers.0.hc_ffn_base", "U8", &[1], [19, 20]),
            ("model.layers.0.hc_ffn_scale", "U8", &[1], [20, 21]),
            ("model.layers.0.mlp.gate.weight", "U8", &[1], [21, 22]),
            (
                "model.layers.0.ffn.experts.0.w1.weight",
                "U8",
                &[1],
                [22, 23],
            ),
            (
                "model.layers.0.ffn.experts.0.w2.weight",
                "U8",
                &[1],
                [23, 24],
            ),
            (
                "model.layers.0.ffn.experts.0.w3.weight",
                "U8",
                &[1],
                [24, 25],
            ),
        ],
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25,
        ],
    )
    .expect("shard should be written");
}

fn write_complete_qwen2_checkpoint(model_dir: &std::path::Path) {
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "architectures": ["Qwen2ForCausalLM"],
  "model_type": "qwen2",
  "vocab_size": 3,
  "num_hidden_layers": 1,
  "hidden_size": 2,
  "intermediate_size": 2,
  "num_attention_heads": 1,
  "num_key_value_heads": 1,
  "hidden_act": "silu",
  "rms_norm_eps": 0.000001,
  "rope_theta": 1000000.0,
  "max_position_embeddings": 32,
  "tie_word_embeddings": false
}"#,
    )
    .expect("config should be written");

    fs::write(
        model_dir.join("tokenizer.json"),
        word_level_tokenizer_json(),
    )
    .expect("Qwen tokenizer should be written");

    let descriptors: Vec<(&str, Vec<usize>, Vec<f32>)> = vec![
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
            "model.layers.0.self_attn.q_proj.bias",
            vec![2],
            vec![0.0; 2],
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.self_attn.k_proj.bias",
            vec![2],
            vec![0.0; 2],
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.self_attn.v_proj.bias",
            vec![2],
            vec![0.0; 2],
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
    let mut payload = Vec::new();
    let mut tensors = Vec::new();
    for (name, shape, values) in &descriptors {
        let start = payload.len();
        payload.extend(values.iter().flat_map(|value| value.to_le_bytes()));
        tensors.push((*name, "F32", shape.as_slice(), [start, payload.len()]));
    }
    write_safetensors_file(&model_dir.join("model.safetensors"), &tensors, &payload)
        .expect("Qwen checkpoint should be written");
}

fn write_complete_qwen3_checkpoint(model_dir: &std::path::Path) {
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
    .expect("config should be written");

    fs::write(
        model_dir.join("tokenizer.json"),
        word_level_tokenizer_json(),
    )
    .expect("Qwen3 tokenizer should be written");

    let descriptors: Vec<(&str, Vec<usize>, Vec<f32>)> = vec![
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
    let mut payload = Vec::new();
    let mut tensors = Vec::new();
    for (name, shape, values) in &descriptors {
        let start = payload.len();
        payload.extend(values.iter().flat_map(|value| value.to_le_bytes()));
        tensors.push((*name, "F32", shape.as_slice(), [start, payload.len()]));
    }
    write_safetensors_file(&model_dir.join("model.safetensors"), &tensors, &payload)
        .expect("Qwen3 checkpoint should be written");
}

fn write_complete_qwen3_5_checkpoint(model_dir: &std::path::Path) {
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "text_config": {
    "model_type": "qwen3_5_text",
    "vocab_size": 3,
    "num_hidden_layers": 2,
    "hidden_size": 2,
    "intermediate_size": 2,
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": 2,
    "hidden_act": "silu",
    "attention_bias": false,
    "attn_output_gate": true,
    "rms_norm_eps": 0.000001,
    "max_position_embeddings": 32,
    "tie_word_embeddings": false,
    "layer_types": ["linear_attention", "full_attention"],
    "linear_conv_kernel_dim": 2,
    "linear_key_head_dim": 2,
    "linear_value_head_dim": 2,
    "linear_num_key_heads": 1,
    "linear_num_value_heads": 1,
    "mamba_ssm_dtype": "float32",
    "rope_parameters": {
      "rope_type": "default",
      "rope_theta": 1000000.0,
      "partial_rotary_factor": 1.0
    }
  },
  "tie_word_embeddings": false
}"#,
    )
    .expect("config should be written");
    fs::write(
        model_dir.join("tokenizer.json"),
        word_level_tokenizer_json(),
    )
    .expect("Qwen3.5 tokenizer should be written");

    let mut descriptors: Vec<(String, Vec<usize>, Vec<f32>)> = vec![
        (
            "model.language_model.embed_tokens.weight".to_string(),
            vec![3, 2],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ),
        (
            "model.language_model.norm.weight".to_string(),
            vec![2],
            vec![0.0; 2],
        ),
        (
            "lm_head.weight".to_string(),
            vec![3, 2],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        ),
    ];
    for layer_id in 0..2 {
        let prefix = format!("model.language_model.layers.{layer_id}");
        descriptors.extend([
            (
                format!("{prefix}.input_layernorm.weight"),
                vec![2],
                vec![0.0; 2],
            ),
            (
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![2],
                vec![0.0; 2],
            ),
            (
                format!("{prefix}.mlp.gate_proj.weight"),
                vec![2, 2],
                vec![0.0; 4],
            ),
            (
                format!("{prefix}.mlp.up_proj.weight"),
                vec![2, 2],
                vec![0.0; 4],
            ),
            (
                format!("{prefix}.mlp.down_proj.weight"),
                vec![2, 2],
                vec![0.0; 4],
            ),
        ]);
    }
    descriptors.extend([
        (
            "model.language_model.layers.0.linear_attn.A_log".to_string(),
            vec![1],
            vec![0.0],
        ),
        (
            "model.language_model.layers.0.linear_attn.conv1d.weight".to_string(),
            vec![6, 1, 2],
            vec![0.0; 12],
        ),
        (
            "model.language_model.layers.0.linear_attn.dt_bias".to_string(),
            vec![1],
            vec![0.0],
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_a.weight".to_string(),
            vec![1, 2],
            vec![0.0; 2],
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_b.weight".to_string(),
            vec![1, 2],
            vec![0.0; 2],
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight".to_string(),
            vec![6, 2],
            vec![0.0; 12],
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_z.weight".to_string(),
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.language_model.layers.0.linear_attn.norm.weight".to_string(),
            vec![2],
            vec![1.0; 2],
        ),
        (
            "model.language_model.layers.0.linear_attn.out_proj.weight".to_string(),
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.language_model.layers.1.self_attn.q_proj.weight".to_string(),
            vec![4, 2],
            vec![0.0; 8],
        ),
        (
            "model.language_model.layers.1.self_attn.q_norm.weight".to_string(),
            vec![2],
            vec![0.0; 2],
        ),
        (
            "model.language_model.layers.1.self_attn.k_proj.weight".to_string(),
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.language_model.layers.1.self_attn.k_norm.weight".to_string(),
            vec![2],
            vec![0.0; 2],
        ),
        (
            "model.language_model.layers.1.self_attn.v_proj.weight".to_string(),
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.language_model.layers.1.self_attn.o_proj.weight".to_string(),
            vec![2, 2],
            vec![0.0; 4],
        ),
    ]);

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
        .expect("Qwen3.5 checkpoint should be written");
}

fn write_complete_kimi_linear_checkpoint(model_dir: &std::path::Path) {
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
    .expect("config should be written");
    fs::write(
        model_dir.join("tokenizer.json"),
        word_level_tokenizer_json(),
    )
    .expect("Kimi Linear tokenizer should be written");

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
    {
        let mut add_tensor = |name: String, shape: Vec<usize>, value: f32| {
            let element_count = shape.iter().product();
            descriptors.push((name, shape, vec![value; element_count]));
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
        .expect("Kimi Linear checkpoint should be written");
}

fn write_complete_deepseek_v3_checkpoint(model_dir: &std::path::Path) {
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
        .expect("DeepSeek V3 checkpoint should be written");
}

fn write_complete_glm_moe_dsa_forward_checkpoint(model_dir: &std::path::Path) {
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "architectures": ["GlmMoeDsaForCausalLM"],
  "model_type": "glm_moe_dsa",
  "vocab_size": 3,
  "num_hidden_layers": 1,
  "hidden_size": 2,
  "intermediate_size": 2,
  "num_attention_heads": 2,
  "num_key_value_heads": 2,
  "head_dim": 1,
  "qk_nope_head_dim": 64,
  "qk_rope_head_dim": 32,
  "v_head_dim": 64,
  "rms_norm_eps": 0.0,
  "n_routed_experts": 1,
  "num_experts_per_tok": 1,
  "moe_intermediate_size": 2,
  "first_k_dense_replace": 1,
  "moe_layer_freq": 1
}"#,
    )
    .expect("config should be written");

    let values = [
        0.0, 0.0, // [UNK] embedding
        1.0, 1.0, // hello embedding
        1.0, -1.0, // world embedding
        1.0, 1.0, // final norm
        0.0, 0.0, // [UNK] lm_head
        1.5, 0.0, // hello lm_head
        0.0, 1.0, // world lm_head
        1.0, 0.0, 0.0, 1.0, // q_a_proj
        1.0, 1.0, // q_a_layernorm
        1.0, 0.0, 0.0, 1.0, // q_b_proj
        1.0, 0.0, 0.0, 1.0, // kv_a_proj_with_mqa
        1.0, 1.0, // kv_a_layernorm
        1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, // kv_b_proj
        2.0, 3.0, 5.0, 7.0, // o_proj
        1.0, 1.0, // input_layernorm
        1.0, 1.0, // post_attention_layernorm
        1.0, 0.0, 0.0, 1.0, // mlp.gate_proj
        2.0, 0.0, 0.0, 3.0, // mlp.up_proj
        5.0, 7.0, 11.0, 13.0, // mlp.down_proj
    ];
    let payload = values
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    write_safetensors_file(
        &model_dir.join("model.safetensors"),
        &[
            ("model.embed_tokens.weight", "F32", &[3, 2], [0, 24]),
            ("model.norm.weight", "F32", &[2], [24, 32]),
            ("lm_head.weight", "F32", &[3, 2], [32, 56]),
            (
                "model.layers.0.self_attn.q_a_proj.weight",
                "F32",
                &[2, 2],
                [56, 72],
            ),
            (
                "model.layers.0.self_attn.q_a_layernorm.weight",
                "F32",
                &[2],
                [72, 80],
            ),
            (
                "model.layers.0.self_attn.q_b_proj.weight",
                "F32",
                &[2, 2],
                [80, 96],
            ),
            (
                "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                "F32",
                &[2, 2],
                [96, 112],
            ),
            (
                "model.layers.0.self_attn.kv_a_layernorm.weight",
                "F32",
                &[2],
                [112, 120],
            ),
            (
                "model.layers.0.self_attn.kv_b_proj.weight",
                "F32",
                &[4, 2],
                [120, 152],
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                "F32",
                &[2, 2],
                [152, 168],
            ),
            (
                "model.layers.0.input_layernorm.weight",
                "F32",
                &[2],
                [168, 176],
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "F32",
                &[2],
                [176, 184],
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                "F32",
                &[2, 2],
                [184, 200],
            ),
            (
                "model.layers.0.mlp.up_proj.weight",
                "F32",
                &[2, 2],
                [200, 216],
            ),
            (
                "model.layers.0.mlp.down_proj.weight",
                "F32",
                &[2, 2],
                [216, 232],
            ),
        ],
        &payload,
    )
    .expect("shard should be written");
}

fn write_safetensors_file(
    path: &std::path::Path,
    tensors: &[(&str, &str, &[usize], [usize; 2])],
    payload: &[u8],
) -> std::io::Result<()> {
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

async fn connect_grpc_with_retry(addr: std::net::SocketAddr) -> SglangServiceClient<Channel> {
    let endpoint = format!("http://{addr}");
    let mut last_error = None;
    for _ in 0..100 {
        match SglangServiceClient::connect(endpoint.clone()).await {
            Ok(client) => return client,
            Err(error) => last_error = Some(error),
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    panic!(
        "gRPC client should connect to production server: {}",
        last_error.expect("at least one connection attempt should run")
    );
}

async fn request_sidecar_with_retry(addr: std::net::SocketAddr, path: &str) -> String {
    let mut last_error = None;
    for _ in 0..100 {
        match tokio::net::TcpStream::connect(addr).await {
            Ok(mut stream) => {
                let request =
                    format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
                if let Err(error) = stream.write_all(request.as_bytes()).await {
                    last_error = Some(error);
                } else {
                    let mut response = Vec::new();
                    match stream.read_to_end(&mut response).await {
                        Ok(_) => {
                            return String::from_utf8(response)
                                .expect("sidecar response should be UTF-8");
                        }
                        Err(error) => last_error = Some(error),
                    }
                }
            }
            Err(error) => last_error = Some(error),
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    panic!(
        "HTTP client should connect to production gRPC sidecar: {}",
        last_error.expect("at least one connection attempt should run")
    );
}

fn unused_local_addr() -> std::net::SocketAddr {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("ephemeral port should bind");
    listener
        .local_addr()
        .expect("ephemeral listener should have local addr")
}

fn temp_model_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("sglang-rs-{name}-{}", std::process::id()))
}
