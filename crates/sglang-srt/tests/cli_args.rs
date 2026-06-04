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
    assert_eq!(parsed.base_gpu_id, 0);
    assert_eq!(parsed.gpu_id_step, 1);
    assert!(!parsed.grpc_mode);
}

#[test]
fn parse_preserves_unknown_server_args_for_future_compatibility() {
    let parsed = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--trust-remote-code",
        "--attention-backend",
        "flashinfer",
    ])
    .expect("args should parse");

    assert_eq!(
        parsed.extra_args,
        vec![
            "--trust-remote-code".to_string(),
            "--attention-backend".to_string(),
            "flashinfer".to_string()
        ]
    );
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
