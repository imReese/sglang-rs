use std::fs;
use std::path::PathBuf;

use serde_json::json;
use sglang_srt::cli::ServerArgs;
use sglang_srt::cli::ZmqPortRange;
use sglang_srt::transfer::{
    DisaggregationMode, KvCacheDtype, KvCacheModelLayout, KvPoll, MooncakeKvCacheLayout,
    MooncakeOpcode, MooncakeTransferEngineConfig, MooncakeTransferStatusCode, PdConfig,
    PdConfigError, TransferBackend,
};

#[cfg(feature = "mooncake-link")]
use sglang_srt::transfer::{LinkedMooncakeTransferEngine, MooncakeError};

#[test]
fn pd_config_defaults_to_unified_mooncake_backend() {
    let args =
        ServerArgs::parse_from(["serve", "--model-path", "dummy"]).expect("args should parse");

    let config = PdConfig::from_server_args(&args).expect("pd config should normalize");

    assert_eq!(config.mode, DisaggregationMode::Null);
    assert_eq!(config.transfer_backend, TransferBackend::Mooncake);
    assert!(!config.force_tcp_transport);
    assert_eq!(config.bootstrap_port, 8998);
    assert!(config.ib_device.is_none());
}

#[test]
fn pd_config_normalizes_mooncake_tcp_to_mooncake_with_forced_tcp() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "mooncake_tcp",
        "--disaggregation-ib-device",
        "mlx5_0",
    ])
    .expect("args should parse");

    let config = PdConfig::from_server_args(&args).expect("pd config should normalize");

    assert_eq!(config.mode, DisaggregationMode::Decode);
    assert_eq!(config.transfer_backend, TransferBackend::Mooncake);
    assert!(config.force_tcp_transport);
    assert!(config.ib_device.is_none());
}

#[test]
fn pd_config_accepts_fake_backend_for_prefill_smoke_tests() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "fake",
    ])
    .expect("args should parse");

    let config = PdConfig::from_server_args(&args).expect("prefill fake should normalize");

    assert_eq!(config.mode, DisaggregationMode::Prefill);
    assert_eq!(config.transfer_backend, TransferBackend::Fake);
}

#[test]
fn pd_config_rejects_unknown_disaggregation_mode() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--disaggregation-mode",
        "split",
    ])
    .expect("args should parse");

    let error = PdConfig::from_server_args(&args).expect_err("unknown mode should fail");

    assert_eq!(
        error,
        PdConfigError::InvalidDisaggregationMode("split".to_string())
    );
}

#[test]
fn mooncake_engine_config_uses_tcp_when_mooncake_tcp_is_normalized() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--disaggregation-transfer-backend",
        "mooncake_tcp",
        "--disaggregation-ib-device",
        "mlx5_0",
    ])
    .expect("args should parse");
    let pd_config = PdConfig::from_server_args(&args).expect("pd config should normalize");

    let engine_config = MooncakeTransferEngineConfig::from_pd_config("127.0.0.1", 0, &pd_config);

    assert_eq!(engine_config.hostname, "127.0.0.1");
    assert_eq!(engine_config.gpu_id, 0);
    assert_eq!(engine_config.protocol, "tcp");
    assert_eq!(engine_config.metadata_server, "P2PHANDSHAKE");
    assert!(engine_config.device_name.is_empty());
}

#[test]
fn mooncake_engine_config_uses_upstream_gpu_placement_args() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--base-gpu-id",
        "2",
        "--gpu-id-step",
        "3",
        "--disaggregation-mode",
        "decode",
    ])
    .expect("args should parse");
    let pd_config = PdConfig::from_server_args(&args).expect("pd config should normalize");

    let engine_config =
        MooncakeTransferEngineConfig::from_pd_config_for_rank("127.0.0.1", 1, &pd_config);

    assert_eq!(pd_config.base_gpu_id, 2);
    assert_eq!(pd_config.gpu_id_step, 3);
    assert_eq!(engine_config.gpu_id, 5);
}

#[test]
fn mooncake_engine_config_builds_session_id_from_hostname_and_rpc_port() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--host",
        "127.0.0.1",
        "--port",
        "30002",
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-mooncake-rpc-port",
        "41002",
        "--kv-cache-dtype",
        "bfloat16",
        "--kv-cache-num-layers",
        "1",
        "--kv-cache-kv-heads",
        "1",
        "--kv-cache-head-dim",
        "8",
    ])
    .expect("args should parse");
    let pd_config = PdConfig::from_server_args(&args).expect("PD config should parse");

    let engine_config =
        MooncakeTransferEngineConfig::from_pd_config_for_rank("127.0.0.1", 0, &pd_config);

    assert_eq!(pd_config.mooncake_rpc_port, Some(41002));
    assert_eq!(engine_config.rpc_port, 41002);
    assert_eq!(engine_config.session_id, "127.0.0.1:41002");
}

#[test]
fn pd_config_carries_page_size_for_mooncake_kv_layout() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--disaggregation-mode",
        "decode",
        "--page-size",
        "64",
    ])
    .expect("args should parse");

    let config = PdConfig::from_server_args(&args).expect("pd config should normalize");
    let layout = MooncakeKvCacheLayout::from_pd_config(0x1000, 16, 0x200, &config);

    assert_eq!(config.page_size, 64);
    assert_eq!(layout.source_base_addr, 0x1000);
    assert_eq!(layout.page_size_bytes, 1024);
    assert_eq!(layout.target_base_offset, 0x200);
}

#[test]
fn pd_config_normalizes_kv_cache_dtype_for_mooncake_layout_bytes() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--disaggregation-mode",
        "decode",
        "--kv-cache-dtype",
        "bf16",
        "--page-size",
        "64",
    ])
    .expect("args should parse");

    let config = PdConfig::from_server_args(&args).expect("pd config should normalize");
    let layout =
        MooncakeKvCacheLayout::from_pd_config_kv_elements(0x1000, 512, 0x200, &config).unwrap();

    assert_eq!(config.kv_cache_dtype, KvCacheDtype::Bfloat16);
    assert_eq!(config.kv_cache_dtype.bytes_per_element(), Some(2));
    assert_eq!(layout.page_size_bytes, 64 * 512 * 2);
}

#[test]
fn pd_config_builds_mooncake_layout_from_model_kv_geometry() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "deepseek-ai/DeepSeek-V3-0324",
        "--disaggregation-mode",
        "decode",
        "--kv-cache-dtype",
        "bfloat16",
        "--page-size",
        "64",
    ])
    .expect("args should parse");

    let config = PdConfig::from_server_args(&args).expect("pd config should normalize");
    let model_layout = KvCacheModelLayout::multi_tensor(61, 1, 512, 2);

    let layout =
        MooncakeKvCacheLayout::from_pd_config_model_layout(0x1000, 0x200, &config, &model_layout)
            .unwrap();

    assert_eq!(model_layout.elements_per_token(), Some(61 * 2 * 512));
    assert_eq!(
        model_layout.token_size_bytes(config.kv_cache_dtype),
        Ok(61 * 2 * 512 * 2)
    );
    assert_eq!(layout.page_size_bytes, 64 * 61 * 2 * 512 * 2);
}

#[test]
fn pd_config_carries_cli_kv_model_geometry_for_mooncake_layout() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "deepseek-ai/DeepSeek-V3-0324",
        "--disaggregation-mode",
        "decode",
        "--kv-cache-dtype",
        "bfloat16",
        "--kv-cache-num-layers",
        "61",
        "--kv-cache-kv-heads",
        "1",
        "--kv-cache-head-dim",
        "512",
        "--page-size",
        "64",
    ])
    .expect("args should parse");

    let config = PdConfig::from_server_args(&args).expect("pd config should normalize");
    let model_layout = config
        .kv_cache_model_layout
        .expect("kv cache model layout should be present");
    let layout =
        MooncakeKvCacheLayout::from_pd_config_model_layout(0x1000, 0x200, &config, &model_layout)
            .unwrap();

    assert_eq!(
        model_layout,
        KvCacheModelLayout::multi_tensor(61, 1, 512, 2)
    );
    assert_eq!(layout.page_size_bytes, 64 * 61 * 2 * 512 * 2);
}

#[test]
fn pd_config_exposes_runtime_kv_layout_for_control_plane_metadata() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "zai-org/GLM-5-FP8",
        "--disaggregation-mode",
        "prefill",
        "--kv-cache-dtype",
        "bfloat16",
        "--kv-cache-num-layers",
        "78",
        "--kv-cache-kv-heads",
        "64",
        "--kv-cache-head-dim",
        "64",
        "--page-size",
        "64",
    ])
    .expect("args should parse");

    let config = PdConfig::from_server_args(&args).expect("pd config should normalize");
    let layout = config
        .kv_cache_runtime_layout()
        .expect("runtime KV layout should calculate")
        .expect("explicit KV model geometry should produce runtime layout");

    assert_eq!(layout.dtype, KvCacheDtype::Bfloat16);
    assert_eq!(layout.page_size, 64);
    assert_eq!(layout.num_layers, 78);
    assert_eq!(layout.kv_heads, 64);
    assert_eq!(layout.head_dim, 64);
    assert_eq!(layout.kv_tensors_per_token, 2);
    assert_eq!(layout.bytes_per_token, 78 * 2 * 64 * 64 * 2);
    assert_eq!(layout.page_size_bytes, 64 * 78 * 2 * 64 * 64 * 2);
}

#[test]
fn pd_config_derives_deepseek_v4_kv_layout_from_model_config() {
    let model_dir = temp_model_dir("deepseek-v4-kv-layout");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "deepseek_v4",
            "num_hidden_layers": 43,
            "qk_nope_head_dim": 448,
            "qk_rope_head_dim": 64,
            "num_key_value_heads": 1
        }"#,
    )
    .expect("config should be written");

    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp dir should be utf8"),
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--page-size",
        "256",
    ])
    .expect("args should parse");

    let config = PdConfig::from_server_args(&args).expect("pd config should normalize");
    let model_layout = config
        .kv_cache_model_layout
        .expect("DeepSeek V4 config should produce a KV model layout");
    let layout =
        MooncakeKvCacheLayout::from_pd_config_model_layout(0x1000, 0x200, &config, &model_layout)
            .unwrap();

    assert_eq!(
        model_layout,
        KvCacheModelLayout::packed_bytes_per_layer(43, 584)
    );
    assert_eq!(
        model_layout.token_size_bytes(config.kv_cache_dtype),
        Ok(43 * 584)
    );
    assert_eq!(layout.page_size_bytes, 256 * 43 * 584);

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn kv_cache_model_layout_loads_repo_id_from_huggingface_cache_snapshot() {
    let hub_dir = temp_model_dir("pd-hf-cache-hub");
    let snapshot_dir = hub_dir
        .join("models--zai-org--GLM-5-FP8")
        .join("snapshots")
        .join("abc123");
    fs::create_dir_all(&snapshot_dir).expect("snapshot dir should be created");
    fs::create_dir_all(hub_dir.join("models--zai-org--GLM-5-FP8").join("refs"))
        .expect("refs dir should be created");
    fs::write(
        hub_dir
            .join("models--zai-org--GLM-5-FP8")
            .join("refs")
            .join("main"),
        "abc123\n",
    )
    .expect("main ref should be written");
    fs::write(
        snapshot_dir.join("config.json"),
        r#"{
  "model_type": "glm",
  "num_hidden_layers": 48,
  "num_attention_heads": 64,
  "num_key_value_heads": 8,
  "head_dim": 128
}"#,
    )
    .expect("config should be written");

    let layout = KvCacheModelLayout::from_model_path_with_hf_cache("zai-org/GLM-5-FP8", &hub_dir)
        .expect("cached repo config should parse")
        .expect("cached repo config should provide KV layout");

    assert_eq!(layout, KvCacheModelLayout::multi_tensor(48, 8, 128, 2));

    fs::remove_dir_all(hub_dir).expect("temp hub dir should be removed");
}

#[test]
fn pd_config_derives_glm_moe_dsa_kv_layout_from_hf_cache_repo_id() {
    let hub_dir = temp_model_dir("pd-glm-moe-dsa-hf-cache-hub");
    let snapshot_dir = hub_dir
        .join("models--zai-org--GLM-5-FP8")
        .join("snapshots")
        .join("abc123");
    fs::create_dir_all(&snapshot_dir).expect("snapshot dir should be created");
    fs::create_dir_all(hub_dir.join("models--zai-org--GLM-5-FP8").join("refs"))
        .expect("refs dir should be created");
    fs::write(
        hub_dir
            .join("models--zai-org--GLM-5-FP8")
            .join("refs")
            .join("main"),
        "abc123\n",
    )
    .expect("main ref should be written");
    fs::write(
        snapshot_dir.join("config.json"),
        r#"{
  "model_type": "glm_moe_dsa",
  "num_hidden_layers": 78,
  "num_attention_heads": 64,
  "num_key_value_heads": 64,
  "head_dim": 64
}"#,
    )
    .expect("config should be written");
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "zai-org/GLM-5-FP8",
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--kv-cache-dtype",
        "bfloat16",
    ])
    .expect("args should parse");

    let config = PdConfig::from_server_args_with_hf_cache(&args, &hub_dir)
        .expect("pd config should normalize cached GLM repo id");

    assert_eq!(
        config.kv_cache_model_layout,
        Some(KvCacheModelLayout::multi_tensor(78, 64, 64, 2))
    );

    fs::remove_dir_all(hub_dir).expect("temp hub dir should be removed");
}

#[test]
fn pd_config_rejects_partial_cli_kv_model_geometry() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--kv-cache-num-layers",
        "61",
        "--kv-cache-head-dim",
        "512",
    ])
    .expect("args should parse");

    let error = PdConfig::from_server_args(&args).expect_err("partial geometry should fail");

    assert_eq!(error, PdConfigError::IncompleteKvCacheModelLayout);
}

#[test]
fn pd_config_rejects_kv_layout_byte_overflow() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--kv-cache-dtype",
        "bfloat16",
        "--page-size",
        &usize::MAX.to_string(),
    ])
    .expect("args should parse");
    let config = PdConfig::from_server_args(&args).expect("pd config should normalize");

    let error = MooncakeKvCacheLayout::from_pd_config_kv_elements(0x1000, 2, 0x200, &config)
        .expect_err("layout byte overflow should fail");

    assert_eq!(error, PdConfigError::KvCacheLayoutOverflow);
}

#[test]
fn pd_config_tracks_fp8_kv_cache_dtype_byte_width() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--kv-cache-dtype",
        "fp8_e5m2",
    ])
    .expect("args should parse");

    let config = PdConfig::from_server_args(&args).expect("pd config should normalize");

    assert_eq!(config.kv_cache_dtype, KvCacheDtype::Fp8E5M2);
    assert_eq!(config.kv_cache_dtype.bytes_per_element(), Some(1));
}

#[test]
fn pd_config_rejects_unknown_kv_cache_dtype() {
    let args =
        ServerArgs::parse_from(["serve", "--model-path", "dummy", "--kv-cache-dtype", "int8"])
            .expect("args should parse");

    let error = PdConfig::from_server_args(&args).expect_err("unknown dtype should fail");

    assert_eq!(
        error,
        PdConfigError::InvalidKvCacheDtype("int8".to_string())
    );
}

#[test]
fn pd_config_carries_deepseek_distributed_runtime_args() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "deepseek-ai/DeepSeek-V3-0324",
        "--disaggregation-mode",
        "decode",
        "--trust-remote-code",
        "--dist-init-addr",
        "10.0.0.1:5000",
        "--nnodes",
        "2",
        "--node-rank",
        "1",
        "--enable-dp-attention",
        "--moe-a2a-backend",
        "deepep",
        "--mem-fraction-static",
        "0.8",
        "--max-running-requests",
        "128",
    ])
    .expect("DeepSeek PD launch args should parse");

    let config = PdConfig::from_server_args(&args).expect("pd config should normalize");

    assert!(config.trust_remote_code);
    assert_eq!(config.dist_init_addr.as_deref(), Some("10.0.0.1:5000"));
    assert_eq!(config.nnodes, 2);
    assert_eq!(config.node_rank, 1);
    assert!(config.enable_dp_attention);
    assert_eq!(config.moe_a2a_backend.as_deref(), Some("deepep"));
    assert_eq!(config.mem_fraction_static, Some(0.8));
    assert_eq!(config.max_running_requests, Some(128));
}

#[test]
fn pd_config_carries_glm5_prefill_production_runtime_args() {
    let deepep_config = r#"{"normal_dispatch":{"num_sms":24},"normal_combine":{"num_sms":24}}"#;
    let loader_config = r#"{"enable_multithread_load":true,"num_threads":8}"#;
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "/GLM-5-0212-FP8",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-bootstrap-port",
        "8200",
        "--disaggregation-ib-device",
        "mlx5_bond_0",
        "--disaggregation-zmq-ports",
        "7000-7007",
        "--dist-init-addr",
        "10.95.250.21:6676",
        "--tp-size",
        "8",
        "--dp-size",
        "1",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
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

    let config = PdConfig::from_server_args(&args).expect("pd config should normalize");

    assert_eq!(config.mode, DisaggregationMode::Prefill);
    assert_eq!(config.bootstrap_port, 8200);
    assert_eq!(config.ib_device.as_deref(), Some("mlx5_bond_0"));
    assert_eq!(
        config.disaggregation_zmq_ports,
        Some(ZmqPortRange {
            start: 7000,
            end: 7007
        })
    );
    assert_eq!(config.dist_init_addr.as_deref(), Some("10.95.250.21:6676"));
    assert_eq!(config.tp_size, 8);
    assert_eq!(config.dp_size, 1);
    assert!(config.enable_dp_attention);
    assert!(config.enable_dp_lm_head);
    assert!(config.disable_cuda_graph);
    assert_eq!(config.max_prefill_tokens, Some(196608));
    assert_eq!(config.max_running_requests, Some(256));
    assert_eq!(config.max_total_tokens, Some(512000));
    assert_eq!(config.mem_fraction_static, Some(0.835));
    assert_eq!(config.page_size, 64);
    assert_eq!(
        config.deepep_config,
        Some(json!({"normal_dispatch": {"num_sms": 24}, "normal_combine": {"num_sms": 24}}))
    );
    assert_eq!(config.deepep_mode.as_deref(), Some("normal"));
    assert_eq!(config.moe_a2a_backend.as_deref(), Some("deepep"));
    assert_eq!(config.moe_dense_tp_size, Some(1));
    assert_eq!(config.attention_backend.as_deref(), Some("nsa"));
    assert!(config.enable_nsa_prefill_context_parallel);
    assert_eq!(
        config.nsa_prefill_backend.as_deref(),
        Some("flashmla_sparse")
    );
    assert_eq!(
        config.nsa_prefill_cp_mode.as_deref(),
        Some("round-robin-split")
    );
    assert_eq!(config.speculative_algorithm.as_deref(), Some("EAGLE"));
    assert_eq!(config.speculative_eagle_topk, Some(1));
    assert_eq!(config.speculative_num_draft_tokens, Some(4));
    assert_eq!(config.speculative_num_steps, Some(3));
    assert_eq!(config.chunked_prefill_size, Some(65536));
    assert_eq!(config.decode_log_interval, Some(1));
    assert!(config.disable_overlap_schedule);
    assert_eq!(
        config.model_loader_extra_config,
        Some(json!({"enable_multithread_load": true, "num_threads": 8}))
    );
    assert_eq!(config.tokenizer_worker_num, Some(32));
    assert!(config.allow_auto_truncate);
    assert!(config.collect_tokens_histogram);
    assert!(config.enable_cache_report);
    assert!(config.enable_metrics);
    assert!(config.disable_radix_cache);
    assert_eq!(config.tool_call_parser.as_deref(), Some("glm47"));
}

#[test]
fn mooncake_ffi_enums_match_upstream_c_and_sglang_poll_values() {
    assert_eq!(MooncakeOpcode::Read as i32, 0);
    assert_eq!(MooncakeOpcode::Write as i32, 1);

    assert_eq!(MooncakeTransferStatusCode::Waiting as i32, 0);
    assert_eq!(MooncakeTransferStatusCode::Completed as i32, 4);
    assert_eq!(MooncakeTransferStatusCode::Failed as i32, 6);

    assert_eq!(KvPoll::Failed as u8, 0);
    assert_eq!(KvPoll::Bootstrapping as u8, 1);
    assert_eq!(KvPoll::WaitingForInput as u8, 2);
    assert_eq!(KvPoll::Transferring as u8, 3);
    assert_eq!(KvPoll::Success as u8, 4);
}

#[cfg(feature = "mooncake-link")]
#[test]
fn linked_mooncake_engine_constructor_is_available_under_feature() {
    let constructor: fn(
        &MooncakeTransferEngineConfig,
    ) -> Result<LinkedMooncakeTransferEngine, MooncakeError> = LinkedMooncakeTransferEngine::new;
    let endpoint_query: fn(&LinkedMooncakeTransferEngine) -> Result<String, MooncakeError> =
        LinkedMooncakeTransferEngine::local_endpoint;

    let _ = constructor;
    let _ = endpoint_query;
}

#[cfg(feature = "mooncake-link")]
#[test]
#[ignore = "requires local Mooncake libraries and TCP-capable runtime"]
fn linked_mooncake_engine_transfers_registered_host_buffers() {
    use std::ffi::c_void;
    use std::time::{Duration, Instant};

    use sglang_srt::transfer::{
        LinkedMooncakeTransferEngine, MooncakeBufferEntry, MooncakeTransferRequest,
    };

    let config = MooncakeTransferEngineConfig {
        hostname: "127.0.0.1".to_string(),
        gpu_id: 0,
        rpc_port: 0,
        session_id: "127.0.0.1:0".to_string(),
        metadata_server: "P2PHANDSHAKE".to_string(),
        protocol: "tcp".to_string(),
        device_name: String::new(),
    };
    let engine = LinkedMooncakeTransferEngine::new(&config).expect("engine should initialize");
    let mut source = vec![1_u8, 2, 3, 4, 5, 6, 7, 8];
    let mut target = vec![0_u8; source.len()];
    let mut buffers = vec![
        MooncakeBufferEntry {
            addr: source.as_mut_ptr().cast::<c_void>(),
            length: source.len(),
        },
        MooncakeBufferEntry {
            addr: target.as_mut_ptr().cast::<c_void>(),
            length: target.len(),
        },
    ];
    engine
        .register_memory_batch(&mut buffers, "cpu:0")
        .expect("host buffers should register");

    let local_endpoint = engine
        .local_endpoint()
        .expect("actual Mooncake endpoint should be available");
    let target_id = engine
        .open_segment(&local_endpoint)
        .expect("local segment should open");
    let mut requests = vec![MooncakeTransferRequest {
        opcode: MooncakeOpcode::Write as i32,
        source: source.as_mut_ptr().cast::<c_void>(),
        target_id,
        target_offset: target.as_mut_ptr() as u64,
        length: source.len() as u64,
    }];
    let batch_id = engine
        .submit_transfer(&mut requests)
        .expect("transfer should submit");
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let status = engine
            .transfer_status(batch_id, 0)
            .expect("status should query");
        if status.status == MooncakeTransferStatusCode::Completed as i32 {
            break;
        }
        assert!(Instant::now() < deadline, "Mooncake transfer timed out");
        std::thread::sleep(Duration::from_millis(10));
    }
    engine.free_batch(batch_id).expect("batch should free");
    let mut addrs = vec![
        source.as_mut_ptr().cast::<c_void>(),
        target.as_mut_ptr().cast::<c_void>(),
    ];
    engine
        .unregister_memory_batch(&mut addrs)
        .expect("buffers should unregister");
    assert_eq!(target, source);
}

fn temp_model_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("sglang-rs-{name}-{}", std::process::id()))
}
