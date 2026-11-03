use sglang_srt::backend::{TransferBackendCapability, TransferBackendClass};
use sglang_srt::cli::{CliParseError, ServerArgs};
use sglang_srt::transfer::{
    DisaggregationMode, PdConfig, PdConfigError, PdTransferEngineConfig, TransferBackend,
};

#[test]
fn pd_config_defaults_to_tcp_backend() {
    let args =
        ServerArgs::parse_from(["serve", "--model-path", "dummy"]).expect("args should parse");
    let config = PdConfig::from_server_args(&args).expect("PD config should parse");

    assert_eq!(config.mode, DisaggregationMode::Null);
    assert_eq!(config.transfer_backend, TransferBackend::Tcp);
    assert!(config.force_tcp_transport);
}

#[test]
fn production_config_rejects_fake_transfer_backend() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "fake",
    ])
    .expect("CLI syntax should parse before backend validation");

    assert_eq!(
        PdConfig::from_server_args(&args),
        Err(PdConfigError::InvalidTransferBackend("fake".to_string()))
    );
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
    .expect("CLI syntax should parse before mode validation");

    assert_eq!(
        PdConfig::from_server_args(&args),
        Err(PdConfigError::InvalidDisaggregationMode(
            "split".to_string()
        ))
    );
}

#[test]
fn noncommunity_kv_geometry_flags_are_rejected_by_cli() {
    for flag in [
        "--kv-cache-num-layers",
        "--kv-cache-kv-heads",
        "--kv-cache-head-dim",
    ] {
        let error = ServerArgs::parse_from(["serve", "--model-path", "dummy", flag, "1"])
            .expect_err("retired KV geometry flag must fail");
        assert_eq!(
            error,
            CliParseError::RemovedKvCacheGeometryFlag(flag.to_string())
        );
    }
}

#[test]
fn pd_transfer_engine_config_consumes_runtime_device_ordinal() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--base-gpu-id",
        "2",
        "--gpu-id-step",
        "3",
    ])
    .expect("args should parse");
    let config = PdConfig::from_server_args(&args).expect("PD config should parse");

    let engine = PdTransferEngineConfig::from_pd_config("127.0.0.1", 5, &config);
    assert_eq!(engine.gpu_id, 5);
}

#[test]
fn transfer_capabilities_distinguish_production_and_planned_backends() {
    let tcp = TransferBackendCapability::from_backend(TransferBackend::Tcp);
    let nixl = TransferBackendCapability::from_backend(TransferBackend::Nixl);

    assert_eq!(tcp.class, TransferBackendClass::Production);
    assert!(tcp.is_production());
    assert!(tcp.linked_transport);
    assert_eq!(nixl.class, TransferBackendClass::Planned);
    assert!(!nixl.is_production());
}

#[test]
fn pd_config_supports_native_tcp_backend() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "tcp",
    ])
    .expect("args should parse");
    let config = PdConfig::from_server_args(&args).expect("PD config should parse");

    assert_eq!(config.mode, DisaggregationMode::Prefill);
    assert_eq!(config.transfer_backend, TransferBackend::Tcp);
}
