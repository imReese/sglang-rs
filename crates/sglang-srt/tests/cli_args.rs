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
    assert_eq!(parsed.load_balance_method, "round_robin");
    assert_eq!(parsed.device, "auto");
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
    assert_eq!(parsed.device, "auto");
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
        "--future-scheduler-policy",
        "latency",
    ])
    .expect("args should parse");

    assert_eq!(
        parsed.extra_args,
        vec![
            "--future-scheduler-policy".to_string(),
            "latency".to_string()
        ]
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
        "--engine-info-bootstrap-port",
        "6790",
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
    assert_eq!(parsed.load_balance_method, "follow_bootstrap_room");
    assert_eq!(parsed.disaggregation_transfer_backend, "mooncake");
    assert_eq!(parsed.disaggregation_bootstrap_port, 8999);
    assert_eq!(parsed.engine_info_bootstrap_port, 6790);
    assert_eq!(parsed.disaggregation_ib_device.as_deref(), Some("mlx5_0"));
    assert!(parsed.disaggregation_decode_enable_radix_cache);
    assert!(parsed.disaggregation_decode_enable_offload_kvcache);
    assert_eq!(parsed.num_reserved_decode_tokens, 1024);
    assert_eq!(parsed.disaggregation_decode_polling_interval, 2);
    assert!(parsed.extra_args.is_empty());
}

#[test]
fn parse_device_accepts_explicit_production_target() {
    let parsed = ServerArgs::parse_from(["serve", "--model-path", "dummy", "--device", "cuda"])
        .expect("device should parse");

    assert_eq!(parsed.device, "cuda");
    assert!(parsed.extra_args.is_empty());
}

#[test]
fn parse_device_accepts_community_accelerator_targets() {
    for device in ["musa", "xpu", "npu", "hpu"] {
        let parsed = ServerArgs::parse_from(["serve", "--model-path", "dummy", "--device", device])
            .expect("community device target should parse");

        assert_eq!(parsed.device, device);
        assert!(parsed.extra_args.is_empty());
    }
}

#[test]
fn parse_device_rejects_unknown_target() {
    let error = ServerArgs::parse_from(["serve", "--model-path", "dummy", "--device", "vulkan"])
        .expect_err("unknown device should fail");

    assert_eq!(error.to_string(), "invalid --device: vulkan");
}

#[test]
fn parse_runtime_backend_flag_points_to_community_device_arg() {
    let error = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--runtime-backend",
        "cuda",
    ])
    .expect_err("non-community runtime backend flag should fail");

    assert_eq!(
        error.to_string(),
        "--runtime-backend is not a community SGLang CLI flag; use --device instead"
    );
}

#[test]
fn parse_load_balance_method_accepts_explicit_sglang_value() {
    let parsed = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--load-balance-method",
        "total_tokens",
    ])
    .expect("load balance method should parse");

    assert_eq!(parsed.load_balance_method, "total_tokens");
    assert!(parsed.extra_args.is_empty());
}

#[test]
fn parse_engine_info_bootstrap_port_defaults_to_sglang_value() {
    let parsed =
        ServerArgs::parse_from(["serve", "--model-path", "dummy"]).expect("args should parse");

    assert_eq!(parsed.engine_info_bootstrap_port, 6789);
}

#[test]
fn parse_multinode_worker_args() {
    let parsed = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--trust-remote-code",
        "--dist-init-addr",
        "192.0.2.1:5000",
        "--nnodes",
        "2",
        "--node-rank",
        "1",
    ])
    .expect("multinode args should parse");

    assert!(parsed.trust_remote_code);
    assert_eq!(parsed.dist_init_addr.as_deref(), Some("192.0.2.1:5000"));
    assert_eq!(parsed.nnodes, 2);
    assert_eq!(parsed.node_rank, 1);
    assert!(parsed.extra_args.is_empty());
}

#[test]
fn parse_glm5_cookbook_args() {
    let parsed = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "zai-org/GLM-5-FP8",
        "--tp",
        "8",
        "--tool-call-parser",
        "glm47",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
    ])
    .expect("GLM-5 cookbook args should parse");

    assert_eq!(parsed.model_path, "zai-org/GLM-5-FP8");
    assert_eq!(parsed.tp_size, 8);
    assert_eq!(parsed.tool_call_parser.as_deref(), Some("glm47"));
    assert_eq!(parsed.speculative_algorithm.as_deref(), Some("EAGLE"));
    assert_eq!(parsed.speculative_num_steps, Some(3));
    assert_eq!(parsed.speculative_eagle_topk, Some(1));
    assert_eq!(parsed.speculative_num_draft_tokens, Some(4));
    assert!(parsed.extra_args.is_empty());
}
