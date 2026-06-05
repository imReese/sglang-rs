use sglang_srt::cli::{CliCommand, ServerArgs};

#[test]
fn parse_sglang_serve_style_worker_args() {
    let parsed = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "meta-llama/Llama-3.1-8B-Instruct",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "--tp-size",
        "1",
        "--dp-size",
        "8",
        "--kv-cache-dtype",
        "bfloat16",
        "--page-size",
        "64",
        "--base-gpu-id",
        "2",
        "--gpu-id-step",
        "3",
        "--grpc-mode",
    ])
    .expect("args should parse");

    assert_eq!(parsed.command, CliCommand::Serve);
    assert_eq!(parsed.model_path, "meta-llama/Llama-3.1-8B-Instruct");
    assert_eq!(parsed.host, "0.0.0.0");
    assert_eq!(parsed.port, 8080);
    assert_eq!(parsed.tp_size, 1);
    assert_eq!(parsed.dp_size, 8);
    assert_eq!(parsed.kv_cache_dtype, "bfloat16");
    assert_eq!(parsed.page_size, 64);
    assert_eq!(parsed.base_gpu_id, 2);
    assert_eq!(parsed.gpu_id_step, 3);
    assert!(parsed.grpc_mode);
}

#[test]
fn parse_model_alias_and_default_network_args() {
    let parsed = ServerArgs::parse_from(["--model", "Qwen/Qwen3-4B"]).expect("args should parse");

    assert_eq!(parsed.command, CliCommand::Serve);
    assert_eq!(parsed.model_path, "Qwen/Qwen3-4B");
    assert_eq!(parsed.host, "127.0.0.1");
    assert_eq!(parsed.port, 30000);
    assert_eq!(parsed.tp_size, 1);
    assert_eq!(parsed.dp_size, 1);
    assert_eq!(parsed.kv_cache_dtype, "auto");
    assert_eq!(parsed.page_size, 1);
    assert_eq!(parsed.base_gpu_id, 0);
    assert_eq!(parsed.gpu_id_step, 1);
    assert!(!parsed.grpc_mode);
}

#[test]
fn parse_accepts_equals_style_sglang_args() {
    let parsed = ServerArgs::parse_from([
        "serve",
        "--model-path=zai-org/GLM-5-FP8",
        "--port=8000",
        "--tp-size=8",
        "--dp-size=1",
        "--grpc-mode",
    ])
    .expect("equals style args should parse");

    assert_eq!(parsed.model_path, "zai-org/GLM-5-FP8");
    assert_eq!(parsed.port, 8000);
    assert_eq!(parsed.tp_size, 8);
    assert_eq!(parsed.dp_size, 1);
    assert!(parsed.grpc_mode);
    assert!(parsed.extra_args.is_empty());
}

#[test]
fn parse_preserves_unknown_server_args_for_future_compatibility() {
    let parsed = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--attention-backend",
        "flashinfer",
    ])
    .expect("args should parse");

    assert_eq!(
        parsed.extra_args,
        vec!["--attention-backend".to_string(), "flashinfer".to_string()]
    );
    assert!(!parsed.trust_remote_code);
}

#[test]
fn parse_model_metadata_args_used_by_router_registration() {
    let parsed = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "meta-llama/Llama-3.1-8B-Instruct",
        "--served-model-name",
        "llama3",
        "--tokenizer-path",
        "hf-tokenizer",
    ])
    .expect("args should parse");

    assert_eq!(parsed.served_model_name.as_deref(), Some("llama3"));
    assert_eq!(parsed.tokenizer_path.as_deref(), Some("hf-tokenizer"));
    assert!(parsed.extra_args.is_empty());
}

#[test]
fn parse_pd_disaggregation_args_matches_sglang_server_args() {
    let parsed = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-bootstrap-port",
        "8999",
        "--disaggregation-ib-device",
        "mlx5_0",
        "--disaggregation-decode-enable-radix-cache",
        "--disaggregation-decode-enable-offload-kvcache",
        "--num-reserved-decode-tokens",
        "1024",
        "--disaggregation-decode-polling-interval",
        "2",
    ])
    .expect("pd args should parse");

    assert_eq!(parsed.disaggregation_mode, "prefill");
    assert_eq!(parsed.disaggregation_transfer_backend, "mooncake");
    assert_eq!(parsed.disaggregation_bootstrap_port, 8999);
    assert_eq!(parsed.disaggregation_ib_device.as_deref(), Some("mlx5_0"));
    assert!(parsed.disaggregation_decode_enable_radix_cache);
    assert!(parsed.disaggregation_decode_enable_offload_kvcache);
    assert_eq!(parsed.num_reserved_decode_tokens, 1024);
    assert_eq!(parsed.disaggregation_decode_polling_interval, 2);
    assert!(parsed.extra_args.is_empty());
}

#[test]
fn parse_deepseek_pd_multinode_launch_args_as_structured_runtime_config() {
    let parsed = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "deepseek-ai/DeepSeek-V3-0324",
        "--disaggregation-mode",
        "decode",
        "--host",
        "10.0.0.8",
        "--port",
        "30001",
        "--trust-remote-code",
        "--dist-init-addr",
        "10.0.0.1:5000",
        "--nnodes",
        "2",
        "--node-rank",
        "1",
        "--tp-size",
        "16",
        "--dp-size",
        "8",
        "--kv-cache-dtype",
        "fp8_e5m2",
        "--kv-cache-num-layers",
        "61",
        "--kv-cache-kv-heads",
        "1",
        "--kv-cache-head-dim",
        "512",
        "--enable-dp-attention",
        "--moe-a2a-backend",
        "deepep",
        "--mem-fraction-static",
        "0.8",
        "--max-running-requests",
        "128",
    ])
    .expect("DeepSeek PD launch args should parse");

    assert_eq!(parsed.model_path, "deepseek-ai/DeepSeek-V3-0324");
    assert_eq!(parsed.disaggregation_mode, "decode");
    assert!(parsed.trust_remote_code);
    assert_eq!(parsed.dist_init_addr.as_deref(), Some("10.0.0.1:5000"));
    assert_eq!(parsed.nnodes, 2);
    assert_eq!(parsed.node_rank, 1);
    assert_eq!(parsed.kv_cache_dtype, "fp8_e5m2");
    assert_eq!(parsed.kv_cache_num_layers, Some(61));
    assert_eq!(parsed.kv_cache_kv_heads, Some(1));
    assert_eq!(parsed.kv_cache_head_dim, Some(512));
    assert!(parsed.enable_dp_attention);
    assert_eq!(parsed.moe_a2a_backend.as_deref(), Some("deepep"));
    assert_eq!(parsed.mem_fraction_static, Some(0.8));
    assert_eq!(parsed.max_running_requests, Some(128));
    assert!(parsed.extra_args.is_empty());
}
