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
    assert_eq!(parsed.runtime_backend, "auto");
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
    assert_eq!(parsed.runtime_backend, "auto");
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
fn parse_runtime_backend_accepts_explicit_production_target() {
    let parsed = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--runtime-backend",
        "cuda",
    ])
    .expect("runtime backend should parse");

    assert_eq!(parsed.runtime_backend, "cuda");
    assert!(parsed.extra_args.is_empty());
}

#[test]
fn parse_runtime_backend_rejects_unknown_target() {
    let error = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--runtime-backend",
        "vulkan",
    ])
    .expect_err("unknown runtime backend should fail");

    assert_eq!(error.to_string(), "invalid --runtime-backend: vulkan");
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
fn parse_glm5_prefill_production_launch_args_as_structured_runtime_config() {
    let deepep_config = r#"{"normal_dispatch":{"num_sms":24,"num_max_nvl_chunked_send_tokens":36,"num_max_nvl_chunked_recv_tokens":256,"num_max_rdma_chunked_send_tokens":8,"num_max_rdma_chunked_recv_tokens":128},"normal_combine":{"num_sms":24,"num_max_nvl_chunked_send_tokens":36,"num_max_nvl_chunked_recv_tokens":256,"num_max_rdma_chunked_send_tokens":8,"num_max_rdma_chunked_recv_tokens":128}}"#;
    let loader_config = r#"{"enable_multithread_load":true,"num_threads":8}"#;
    let parsed = ServerArgs::parse_from([
        "serve",
        "--host",
        "0.0.0.0",
        "--log-level",
        "info",
        "--model-path",
        "/GLM-5-0212-FP8",
        "--runtime-backend",
        "cuda",
        "--port",
        "8000",
        "--served-model-name",
        "aiak_bzz2_glm_5_community_rd1",
        "--trust-remote-code",
        "--disaggregation-bootstrap-port",
        "8200",
        "--disaggregation-ib-device",
        "mlx5_bond_0",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-zmq-ports",
        "7000-7007",
        "--dist-init-addr",
        "10.95.250.21:6676",
        "--dp-size",
        "1",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--tp-size",
        "8",
        "--disable-cuda-graph",
        "--max-prefill-tokens",
        "196608",
        "--max-running-requests",
        "256",
        "--max-total-tokens",
        "512000",
        "--mem-fraction-static",
        "0.835",
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
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--speculative-num-steps",
        "3",
        "--chunked-prefill-size",
        "65536",
        "--decode-log-interval",
        "1",
        "--disable-overlap-schedule",
        "--model-loader-extra-config",
        loader_config,
        "--tokenizer-worker-num",
        "32",
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
    assert_eq!(parsed.model_path, "/GLM-5-0212-FP8");
    assert_eq!(parsed.runtime_backend, "cuda");
    assert_eq!(parsed.port, 8000);
    assert_eq!(
        parsed.served_model_name.as_deref(),
        Some("aiak_bzz2_glm_5_community_rd1")
    );
    assert_eq!(parsed.disaggregation_bootstrap_port, 8200);
    assert_eq!(
        parsed.disaggregation_ib_device.as_deref(),
        Some("mlx5_bond_0")
    );
    assert_eq!(parsed.disaggregation_mode, "prefill");
    assert_eq!(
        parsed.disaggregation_zmq_ports,
        Some(ZmqPortRange {
            start: 7000,
            end: 7007
        })
    );
    assert_eq!(parsed.dist_init_addr.as_deref(), Some("10.95.250.21:6676"));
    assert_eq!(parsed.tp_size, 8);
    assert_eq!(parsed.dp_size, 1);
    assert!(parsed.enable_dp_attention);
    assert!(parsed.enable_dp_lm_head);
    assert!(parsed.disable_cuda_graph);
    assert_eq!(parsed.max_prefill_tokens, Some(196608));
    assert_eq!(parsed.max_running_requests, Some(256));
    assert_eq!(parsed.max_total_tokens, Some(512000));
    assert_eq!(parsed.mem_fraction_static, Some(0.835));
    assert_eq!(parsed.page_size, 64);
    assert_eq!(
        parsed.deepep_config.as_ref().unwrap()["normal_dispatch"]["num_sms"],
        24
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
    assert_eq!(parsed.speculative_eagle_topk, Some(1));
    assert_eq!(parsed.speculative_num_draft_tokens, Some(4));
    assert_eq!(parsed.speculative_num_steps, Some(3));
    assert_eq!(parsed.chunked_prefill_size, Some(65536));
    assert_eq!(parsed.decode_log_interval, Some(1));
    assert!(parsed.disable_overlap_schedule);
    assert_eq!(
        parsed.model_loader_extra_config,
        Some(json!({"enable_multithread_load": true, "num_threads": 8}))
    );
    assert_eq!(parsed.tokenizer_worker_num, Some(32));
    assert!(parsed.allow_auto_truncate);
    assert!(parsed.collect_tokens_histogram);
    assert!(parsed.enable_cache_report);
    assert!(parsed.enable_metrics);
    assert!(parsed.disable_radix_cache);
    assert_eq!(parsed.tool_call_parser.as_deref(), Some("glm47"));
    assert!(parsed.extra_args.is_empty());
}
