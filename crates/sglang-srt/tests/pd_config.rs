use std::fs;
use std::path::PathBuf;

use sglang_srt::cli::ServerArgs;
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
fn pd_config_rejects_fake_backend_for_prefill() {
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

    let error = PdConfig::from_server_args(&args).expect_err("prefill fake should fail");

    assert_eq!(error, PdConfigError::FakePrefillUnsupported);
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

    let _ = constructor;
}

fn temp_model_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("sglang-rs-{name}-{}", std::process::id()))
}
