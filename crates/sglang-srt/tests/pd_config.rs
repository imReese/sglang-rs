use sglang_srt::cli::ServerArgs;
use sglang_srt::transfer::{
    DisaggregationMode, KvPoll, MooncakeOpcode, MooncakeTransferEngineConfig,
    MooncakeTransferStatusCode, PdConfig, PdConfigError, TransferBackend,
};

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
