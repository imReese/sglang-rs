use serde_json::json;
use sglang_srt::cli::{CliCommand, ServerArgs, ZmqPortRange};

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

#[test]
fn parse_glm5_prefill_extended_launch_args_as_structured_runtime_config() {
    let deepep_config = r#"{"normal_dispatch":{"num_sms":8},"normal_combine":{"num_sms":8}}"#;
    let loader_config = r#"{"enable_multithread_load":true,"num_threads":4}"#;
    let parsed = ServerArgs::parse_from([
        "serve",
        "--host",
        "0.0.0.0",
        "--log-level",
        "info",
        "--model-path",
        "/models/glm-5-fp8",
        "--device",
        "cuda",
        "--port",
        "8000",
        "--served-model-name",
        "glm-5",
        "--trust-remote-code",
        "--disaggregation-bootstrap-port",
        "8200",
        "--disaggregation-ib-device",
        "rdma-test0",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-zmq-ports",
        "7100-7103",
        "--dist-init-addr",
        "192.0.2.10:6676",
        "--dp-size",
        "2",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--nnodes",
        "2",
        "--node-rank",
        "1",
        "--tp-size",
        "4",
        "--disable-cuda-graph",
        "--max-prefill-tokens",
        "4096",
        "--max-running-requests",
        "64",
        "--max-total-tokens",
        "16384",
        "--mem-fraction-static",
        "0.75",
        "--page-size",
        "64",
        "--deepep-config",
        deepep_config,
        "--deepep-mode",
        "normal",
        "--moe-a2a-backend",
        "deepep",
        "--moe-dense-tp-size",
        "1",
        "--attention-backend",
        "nsa",
        "--enable-nsa-prefill-context-parallel",
        "--nsa-prefill-backend",
        "flashmla_sparse",
        "--nsa-prefill-cp-mode",
        "round-robin-split",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-eagle-topk",
        "2",
        "--speculative-num-draft-tokens",
        "4",
        "--speculative-num-steps",
        "2",
        "--chunked-prefill-size",
        "2048",
        "--decode-log-interval",
        "1",
        "--disable-overlap-schedule",
        "--model-loader-extra-config",
        loader_config,
        "--tokenizer-worker-num",
        "4",
        "--allow-auto-truncate",
        "--collect-tokens-histogram",
        "--enable-cache-report",
        "--enable-metrics",
        "--disable-radix-cache",
        "--tool-call-parser",
        "glm47",
    ])
    .expect("GLM-5 prefill launch args should parse");

    assert_eq!(parsed.host, "0.0.0.0");
    assert_eq!(parsed.log_level.as_deref(), Some("info"));
    assert_eq!(parsed.model_path, "/models/glm-5-fp8");
    assert_eq!(parsed.device, "cuda");
    assert_eq!(parsed.port, 8000);
    assert_eq!(parsed.served_model_name.as_deref(), Some("glm-5"));
    assert_eq!(parsed.disaggregation_bootstrap_port, 8200);
    assert_eq!(
        parsed.disaggregation_ib_device.as_deref(),
        Some("rdma-test0")
    );
    assert_eq!(parsed.disaggregation_mode, "prefill");
    assert_eq!(
        parsed.disaggregation_zmq_ports,
        Some(ZmqPortRange {
            start: 7100,
            end: 7103
        })
    );
    assert_eq!(parsed.dist_init_addr.as_deref(), Some("192.0.2.10:6676"));
    assert_eq!(parsed.tp_size, 4);
    assert_eq!(parsed.dp_size, 2);
    assert_eq!(parsed.nnodes, 2);
    assert_eq!(parsed.node_rank, 1);
    assert!(parsed.enable_dp_attention);
    assert!(parsed.enable_dp_lm_head);
    assert!(parsed.disable_cuda_graph);
    assert_eq!(parsed.max_prefill_tokens, Some(4096));
    assert_eq!(parsed.max_running_requests, Some(64));
    assert_eq!(parsed.max_total_tokens, Some(16384));
    assert_eq!(parsed.mem_fraction_static, Some(0.75));
    assert_eq!(parsed.page_size, 64);
    assert_eq!(
        parsed.deepep_config.as_ref().unwrap()["normal_dispatch"]["num_sms"],
        8
    );
    assert_eq!(parsed.deepep_mode.as_deref(), Some("normal"));
    assert_eq!(parsed.moe_a2a_backend.as_deref(), Some("deepep"));
    assert_eq!(parsed.moe_dense_tp_size, Some(1));
    assert_eq!(parsed.attention_backend.as_deref(), Some("nsa"));
    assert!(parsed.enable_nsa_prefill_context_parallel);
    assert_eq!(
        parsed.nsa_prefill_backend.as_deref(),
        Some("flashmla_sparse")
    );
    assert_eq!(
        parsed.nsa_prefill_cp_mode.as_deref(),
        Some("round-robin-split")
    );
    assert_eq!(parsed.speculative_algorithm.as_deref(), Some("EAGLE"));
    assert_eq!(parsed.speculative_eagle_topk, Some(2));
    assert_eq!(parsed.speculative_num_draft_tokens, Some(4));
    assert_eq!(parsed.speculative_num_steps, Some(2));
    assert_eq!(parsed.chunked_prefill_size, Some(2048));
    assert_eq!(parsed.decode_log_interval, Some(1));
    assert!(parsed.disable_overlap_schedule);
    assert_eq!(
        parsed.model_loader_extra_config,
        Some(json!({"enable_multithread_load": true, "num_threads": 4}))
    );
    assert_eq!(parsed.tokenizer_worker_num, Some(4));
    assert!(parsed.allow_auto_truncate);
    assert!(parsed.collect_tokens_histogram);
    assert!(parsed.enable_cache_report);
    assert!(parsed.enable_metrics);
    assert!(parsed.disable_radix_cache);
    assert_eq!(parsed.tool_call_parser.as_deref(), Some("glm47"));
    assert!(parsed.extra_args.is_empty());
}
