use sglang_srt::cli::ServerArgs;
use sglang_srt::transfer::{
    DisaggregationMode, KvPoll, MooncakeOpcode, MooncakeTransferEngineConfig,
    MooncakeTransferStatusCode, PdConfig, PdConfigError, TransferBackend,
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
