use std::fs;
use std::path::PathBuf;

use tonic::Request;

use sglang_srt::cli::ServerArgs;
use sglang_srt::model_artifacts::ModelArtifactError;
use sglang_srt::proto::sglang::runtime::v1::generate_response::Body;
use sglang_srt::proto::sglang::runtime::v1::sglang_service_server::SglangService;
use sglang_srt::proto::sglang::runtime::v1::{
    GetModelInfoRequest, RequestOptions, SamplingParams, TextGenerateRequest, TokenizeRequest,
};
use sglang_srt::server::{
    ServerLaunchError, build_bootstrap_fake_pd_grpc_router_service,
    build_bootstrap_grpc_router_service, build_bootstrap_pd_grpc_router_service, grpc_listen_addr,
    launch_grpc_server, try_build_bootstrap_grpc_router_service,
};
use sglang_srt::tokenizer::TokenizerError;
use sglang_srt::transfer::{
    DecodeBootstrapRegistry, DisaggregationMode, MooncakeBatchId, MooncakeBatchReleaser,
    MooncakeError, MooncakeKvCacheLayout, MooncakeKvCacheTransferExecutor, MooncakeTransferRequest,
    MooncakeTransferStatus, MooncakeTransferStatusCode, MooncakeTransferStatusReader,
    MooncakeTransferSubmitter, MooncakeTransferTarget, PdConfigError, TransferBackend,
};

#[test]
fn grpc_listen_addr_uses_server_host_and_port() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--host",
        "127.0.0.1",
        "--port",
        "30001",
        "--grpc-mode",
    ])
    .expect("args should parse");

    let addr = grpc_listen_addr(&args).expect("listen address should resolve");

    assert_eq!(addr.ip().to_string(), "127.0.0.1");
    assert_eq!(addr.port(), 30001);
}

#[tokio::test]
async fn bootstrap_grpc_router_service_carries_model_metadata() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "meta-llama/Llama-3.1-8B-Instruct",
        "--served-model-name",
        "llama3",
        "--tokenizer-path",
        "hf-tokenizer",
        "--grpc-mode",
    ])
    .expect("args should parse");
    let service = build_bootstrap_grpc_router_service(&args);

    let response = service
        .get_model_info(Request::new(GetModelInfoRequest {}))
        .await
        .expect("model info should execute")
        .into_inner();

    assert_eq!(response.model_path, "meta-llama/Llama-3.1-8B-Instruct");
    assert_eq!(response.tokenizer_path, "hf-tokenizer");
    assert_eq!(response.served_model_name, "llama3");
}

#[tokio::test]
async fn bootstrap_grpc_router_service_reports_local_model_config_metadata() {
    let model_dir = temp_model_dir("server-model-config");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        deepseek_v4_model_config_json(),
    )
    .expect("config should be written");
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model dir should be utf-8"),
        "--grpc-mode",
    ])
    .expect("args should parse");
    let service = build_bootstrap_grpc_router_service(&args);

    let response = service
        .get_model_info(Request::new(GetModelInfoRequest {}))
        .await
        .expect("model info should execute")
        .into_inner();

    assert_eq!(response.model_type, "deepseek_v4");
    assert_eq!(response.vocab_size, 129_280);
    assert_eq!(response.max_context_length, 163_840);
    assert_eq!(response.max_request_input_length, 163_840);

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[tokio::test]
async fn bootstrap_grpc_router_service_reports_local_moe_checkpoint_coverage() {
    let model_dir = temp_model_dir("server-moe-checkpoint-coverage");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "deepseek_v4",
  "num_hidden_layers": 1,
  "hidden_size": 1,
  "hc_mult": 1,
  "n_routed_experts": 1,
  "first_k_dense_replace": 0,
  "moe_layer_freq": 1
}"#,
    )
    .expect("config should be written");
    fs::write(
        model_dir.join("model.safetensors.index.json"),
        r#"{
  "weight_map": {
    "model.embed_tokens.weight": "model.safetensors",
    "model.norm.weight": "model.safetensors",
    "lm_head.weight": "model.safetensors",
    "model.hc_head_fn": "model.safetensors",
    "model.hc_head_base": "model.safetensors",
    "model.hc_head_scale": "model.safetensors",
    "model.layers.0.self_attn.wq_a.weight": "model.safetensors",
    "model.layers.0.self_attn.wq_b.weight": "model.safetensors",
    "model.layers.0.self_attn.wkv.weight": "model.safetensors",
    "model.layers.0.self_attn.q_norm.weight": "model.safetensors",
    "model.layers.0.self_attn.kv_norm.weight": "model.safetensors",
    "model.layers.0.self_attn.wo_a.weight": "model.safetensors",
    "model.layers.0.self_attn.wo_b.weight": "model.safetensors",
    "model.layers.0.input_layernorm.weight": "model.safetensors",
    "model.layers.0.post_attention_layernorm.weight": "model.safetensors",
    "model.layers.0.hc_attn_fn": "model.safetensors",
    "model.layers.0.hc_attn_base": "model.safetensors",
    "model.layers.0.hc_attn_scale": "model.safetensors",
    "model.layers.0.hc_ffn_fn": "model.safetensors",
    "model.layers.0.hc_ffn_base": "model.safetensors",
    "model.layers.0.hc_ffn_scale": "model.safetensors",
    "model.layers.0.mlp.gate.weight": "model.safetensors",
    "model.layers.0.ffn.experts.0.w1.weight": "model.safetensors",
    "model.layers.0.ffn.experts.0.w2.weight": "model.safetensors",
    "model.layers.0.ffn.experts.0.w3.weight": "model.safetensors"
  }
}"#,
    )
    .expect("index should be written");
    write_safetensors_file(
        &model_dir.join("model.safetensors"),
        &[
            ("model.embed_tokens.weight", "U8", &[1], [0, 1]),
            ("model.norm.weight", "U8", &[1], [1, 2]),
            ("lm_head.weight", "U8", &[1], [2, 3]),
            ("model.hc_head_fn", "U8", &[1, 1], [3, 4]),
            ("model.hc_head_base", "U8", &[1], [4, 5]),
            ("model.hc_head_scale", "U8", &[1], [5, 6]),
            ("model.layers.0.self_attn.wq_a.weight", "U8", &[1], [6, 7]),
            ("model.layers.0.self_attn.wq_b.weight", "U8", &[1], [7, 8]),
            ("model.layers.0.self_attn.wkv.weight", "U8", &[1], [8, 9]),
            (
                "model.layers.0.self_attn.q_norm.weight",
                "U8",
                &[1],
                [9, 10],
            ),
            (
                "model.layers.0.self_attn.kv_norm.weight",
                "U8",
                &[1],
                [10, 11],
            ),
            ("model.layers.0.self_attn.wo_a.weight", "U8", &[1], [11, 12]),
            ("model.layers.0.self_attn.wo_b.weight", "U8", &[1], [12, 13]),
            (
                "model.layers.0.input_layernorm.weight",
                "U8",
                &[1],
                [13, 14],
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "U8",
                &[1],
                [14, 15],
            ),
            ("model.layers.0.hc_attn_fn", "U8", &[1], [15, 16]),
            ("model.layers.0.hc_attn_base", "U8", &[1], [16, 17]),
            ("model.layers.0.hc_attn_scale", "U8", &[1], [17, 18]),
            ("model.layers.0.hc_ffn_fn", "U8", &[1], [18, 19]),
            ("model.layers.0.hc_ffn_base", "U8", &[1], [19, 20]),
            ("model.layers.0.hc_ffn_scale", "U8", &[1], [20, 21]),
            ("model.layers.0.mlp.gate.weight", "U8", &[1], [21, 22]),
            (
                "model.layers.0.ffn.experts.0.w1.weight",
                "U8",
                &[1],
                [22, 23],
            ),
            (
                "model.layers.0.ffn.experts.0.w2.weight",
                "U8",
                &[1],
                [23, 24],
            ),
            (
                "model.layers.0.ffn.experts.0.w3.weight",
                "U8",
                &[1],
                [24, 25],
            ),
        ],
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25,
        ],
    )
    .expect("shard should be written");
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model dir should be utf-8"),
        "--grpc-mode",
    ])
    .expect("args should parse");
    let service = build_bootstrap_grpc_router_service(&args);

    let response = service
        .get_model_info(Request::new(GetModelInfoRequest {}))
        .await
        .expect("model info should execute")
        .into_inner();

    assert_eq!(response.routed_expert_expected_group_count, 1);
    assert_eq!(response.routed_expert_actual_group_count, 1);
    assert_eq!(response.routed_expert_expected_weight_count, 3);
    assert_eq!(response.routed_expert_actual_weight_count, 3);

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn bootstrap_grpc_router_service_rejects_missing_deepseek_v4_model_root_tensor() {
    let model_dir = temp_model_dir("server-missing-deepseek-root");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "deepseek_v4",
  "num_hidden_layers": 0
}"#,
    )
    .expect("config should be written");
    write_safetensors_file(
        &model_dir.join("model.safetensors"),
        &[
            ("model.embed_tokens.weight", "U8", &[1], [0, 1]),
            ("model.norm.weight", "U8", &[1], [1, 2]),
        ],
        &[1, 2],
    )
    .expect("shard should be written");
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model dir should be utf-8"),
        "--grpc-mode",
    ])
    .expect("args should parse");

    let error = match try_build_bootstrap_grpc_router_service(&args) {
        Ok(_) => panic!("missing DeepSeek V4 model root tensor should fail bootstrap"),
        Err(error) => error,
    };

    assert!(
        matches!(
            error,
            ServerLaunchError::ModelArtifact(ModelArtifactError::InvalidSafetensorsData {
                ref path,
                ref message,
            }) if path == &model_dir
                && message.contains("missing DeepSeek model tensor")
                && message.contains("lm_head.weight")
        ),
        "unexpected error: {error:?}"
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[tokio::test]
async fn bootstrap_grpc_router_service_generates_through_model_runner() {
    let args = ServerArgs::parse_from(["serve", "--model-path", "dummy", "--grpc-mode"])
        .expect("args should parse");
    let service = build_bootstrap_grpc_router_service(&args);

    let mut stream = service
        .text_generate(Request::new(TextGenerateRequest {
            text: "hello".to_string(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(4),
                stop_token_ids: vec![b' ' as u32],
                ..Default::default()
            }),
            options: Some(RequestOptions {
                request_id: Some("bootstrap-generate".to_string()),
                stream: true,
                data_parallel_rank: 0,
                trace_headers: Default::default(),
            }),
            disaggregated_params: None,
        }))
        .await
        .expect("text generate should execute")
        .into_inner();

    let response = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("one response")
        .expect("response should be ok");

    assert_eq!(response.request_id, "bootstrap-generate");
    assert_eq!(
        response.body,
        Some(Body::Complete(
            sglang_srt::proto::sglang::runtime::v1::GenerateComplete {
                output_ids: vec![b' ' as u32],
                text: " ".to_string(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 5,
                completion_tokens: 1,
                cached_tokens: 0,
                index: 0,
            }
        ))
    );
    assert!(
        tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
            .await
            .is_none()
    );
}

#[tokio::test]
async fn bootstrap_grpc_router_service_uses_config_eos_token_as_default_stop() {
    let model_dir = temp_model_dir("server-config-eos-stop");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "llama",
  "eos_token_id": 32
}"#,
    )
    .expect("config should be written");
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model dir should be utf-8"),
        "--grpc-mode",
    ])
    .expect("args should parse");
    let service = build_bootstrap_grpc_router_service(&args);

    let mut stream = service
        .text_generate(Request::new(TextGenerateRequest {
            text: "hello".to_string(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(4),
                ..Default::default()
            }),
            options: Some(RequestOptions {
                request_id: Some("bootstrap-config-eos-stop".to_string()),
                stream: true,
                data_parallel_rank: 0,
                trace_headers: Default::default(),
            }),
            disaggregated_params: None,
        }))
        .await
        .expect("text generate should execute")
        .into_inner();

    let response = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("one response")
        .expect("response should be ok");

    assert_eq!(response.request_id, "bootstrap-config-eos-stop");
    assert_eq!(
        response.body,
        Some(Body::Complete(
            sglang_srt::proto::sglang::runtime::v1::GenerateComplete {
                output_ids: vec![b' ' as u32],
                text: " ".to_string(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 5,
                completion_tokens: 1,
                cached_tokens: 0,
                index: 0,
            }
        ))
    );
    assert!(
        tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
            .await
            .is_none()
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[tokio::test]
async fn bootstrap_grpc_router_service_ignores_config_eos_when_requested() {
    let model_dir = temp_model_dir("server-ignore-config-eos");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "llama",
  "eos_token_id": 32
}"#,
    )
    .expect("config should be written");
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model dir should be utf-8"),
        "--grpc-mode",
    ])
    .expect("args should parse");
    let service = build_bootstrap_grpc_router_service(&args);

    let mut stream = service
        .text_generate(Request::new(TextGenerateRequest {
            text: "hello".to_string(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(2),
                ignore_eos: Some(true),
                ..Default::default()
            }),
            options: Some(RequestOptions {
                request_id: Some("bootstrap-ignore-config-eos".to_string()),
                stream: true,
                data_parallel_rank: 0,
                trace_headers: Default::default(),
            }),
            disaggregated_params: None,
        }))
        .await
        .expect("text generate should execute")
        .into_inner();

    let first = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("first response")
        .expect("response should be ok");
    let second = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("second response")
        .expect("response should be ok");

    assert_eq!(first.request_id, "bootstrap-ignore-config-eos");
    assert_eq!(
        first.body,
        Some(Body::Chunk(
            sglang_srt::proto::sglang::runtime::v1::GenerateStreamChunk {
                token_ids: vec![b' ' as u32],
                text: " ".to_string(),
                prompt_tokens: 5,
                completion_tokens: 1,
                cached_tokens: 0,
                index: 0,
            }
        ))
    );
    assert_eq!(
        second.body,
        Some(Body::Complete(
            sglang_srt::proto::sglang::runtime::v1::GenerateComplete {
                output_ids: vec![b' ' as u32, b' ' as u32],
                text: "  ".to_string(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 5,
                completion_tokens: 2,
                cached_tokens: 0,
                index: 0,
            }
        ))
    );
    assert!(
        tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
            .await
            .is_none()
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[tokio::test]
async fn bootstrap_grpc_router_service_uses_local_hf_tokenizer_when_available() {
    let model_dir = temp_model_dir("server-hf-tokenizer");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("tokenizer.json"),
        word_level_tokenizer_json(),
    )
    .expect("tokenizer.json should be written");
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model dir should be utf-8"),
        "--grpc-mode",
    ])
    .expect("args should parse");
    let service = build_bootstrap_grpc_router_service(&args);

    let response = service
        .tokenize(Request::new(TokenizeRequest {
            text: "hello world".to_string(),
            add_special_tokens: true,
        }))
        .await
        .expect("tokenize should execute")
        .into_inner();

    assert_eq!(response.count, 2);
    assert_eq!(response.token_ids, vec![1, 2]);

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn bootstrap_grpc_router_service_rejects_missing_explicit_tokenizer_path() {
    let model_dir = temp_model_dir("server-missing-tokenizer");
    let tokenizer_dir = model_dir.join("missing-tokenizer");
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model dir should be utf-8"),
        "--tokenizer-path",
        tokenizer_dir
            .to_str()
            .expect("temp tokenizer dir should be utf-8"),
        "--grpc-mode",
    ])
    .expect("args should parse");

    let error = match try_build_bootstrap_grpc_router_service(&args) {
        Ok(_) => panic!("explicit missing tokenizer path should fail"),
        Err(error) => error,
    };

    assert_eq!(
        error,
        ServerLaunchError::Tokenizer(TokenizerError::TokenizerFileNotFound {
            path: tokenizer_dir
        })
    );
}

#[test]
fn bootstrap_grpc_router_service_rejects_incomplete_local_moe_checkpoint() {
    let model_dir = temp_model_dir("server-incomplete-moe-checkpoint");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "deepseek_v4",
  "num_hidden_layers": 2,
  "n_routed_experts": 2,
  "first_k_dense_replace": 0,
  "moe_layer_freq": 1
}"#,
    )
    .expect("config should be written");
    fs::write(
        model_dir.join("model.safetensors.index.json"),
        r#"{
  "weight_map": {
    "model.layers.0.ffn.experts.0.w1.weight": "model.safetensors",
    "model.layers.0.ffn.experts.0.w2.weight": "model.safetensors",
    "model.layers.0.ffn.experts.0.w3.weight": "model.safetensors"
  }
}"#,
    )
    .expect("index should be written");
    write_safetensors_file(
        &model_dir.join("model.safetensors"),
        &[
            ("model.layers.0.ffn.experts.0.w1.weight", "U8", &[1], [0, 1]),
            ("model.layers.0.ffn.experts.0.w2.weight", "U8", &[1], [1, 2]),
            ("model.layers.0.ffn.experts.0.w3.weight", "U8", &[1], [2, 3]),
        ],
        &[1, 2, 3],
    )
    .expect("shard should be written");
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model dir should be utf-8"),
        "--grpc-mode",
    ])
    .expect("args should parse");

    let error = match try_build_bootstrap_grpc_router_service(&args) {
        Ok(_) => panic!("incomplete local MoE checkpoint should fail bootstrap"),
        Err(error) => error,
    };

    assert!(
        matches!(
            error,
            ServerLaunchError::ModelArtifact(ModelArtifactError::InvalidSafetensorsData {
                ref path,
                ref message,
            }) if path == &model_dir
                && message.contains("expected 4 routed expert groups")
                && message.contains("found 1")
        ),
        "unexpected error: {error:?}"
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn bootstrap_grpc_router_service_rejects_duplicate_local_layer_tensors() {
    let model_dir = temp_model_dir("server-duplicate-layer-tensors");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        deepseek_v4_model_config_json(),
    )
    .expect("config should be written");
    let first_shard = model_dir.join("model-00001.safetensors");
    write_safetensors_file(
        &first_shard,
        &[(
            "model.layers.0.self_attn.q_a_proj.weight",
            "U8",
            &[1],
            [0, 1],
        )],
        &[1],
    )
    .expect("first shard should be written");
    let second_shard = model_dir.join("model-00002.safetensors");
    write_safetensors_file(
        &second_shard,
        &[(
            "model.layers.0.self_attn.q_a_proj.weight",
            "U8",
            &[1],
            [0, 1],
        )],
        &[2],
    )
    .expect("second shard should be written");
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model dir should be utf-8"),
        "--grpc-mode",
    ])
    .expect("args should parse");

    let error = match try_build_bootstrap_grpc_router_service(&args) {
        Ok(_) => panic!("duplicate local layer tensor should fail bootstrap"),
        Err(error) => error,
    };

    assert!(
        matches!(
            error,
            ServerLaunchError::ModelArtifact(ModelArtifactError::InvalidSafetensorsData {
                ref path,
                ref message,
            }) if path == &second_shard
                && message.contains("duplicate layer tensor suffix")
                && message.contains("layer 0")
                && message.contains("self_attn.q_a_proj.weight")
        ),
        "unexpected error: {error:?}"
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[tokio::test]
async fn bootstrap_pd_grpc_router_service_polls_transfer_before_decode() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--grpc-mode",
        "--disaggregation-mode",
        "decode",
        "--disaggregation-decode-polling-interval",
        "1",
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("args should parse");
    let transfer_executor = MooncakeKvCacheTransferExecutor::new(
        RecordingMooncakeBackend::completed(),
        MooncakeKvCacheLayout {
            source_base_addr: 0x3000,
            page_size_bytes: 64,
            target_base_offset: 0,
        },
        MooncakeTransferTarget { target_id: 17 },
    );
    let service = build_bootstrap_pd_grpc_router_service(
        &args,
        DecodeBootstrapRegistry::default(),
        transfer_executor,
    );

    let mut stream = service
        .text_generate(Request::new(TextGenerateRequest {
            text: "hi".to_string(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            options: Some(RequestOptions {
                request_id: Some("bootstrap-pd".to_string()),
                stream: true,
                data_parallel_rank: 0,
                trace_headers: Default::default(),
            }),
            disaggregated_params: Some(
                sglang_srt::proto::sglang::runtime::v1::DisaggregatedParams {
                    bootstrap_host: "10.0.0.9".to_string(),
                    bootstrap_port: 8998,
                    bootstrap_room: 41,
                },
            ),
        }))
        .await
        .expect("PD bootstrap service should poll transfer and generate")
        .into_inner();

    let first = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("first response")
        .expect("first response should be ok");
    let second = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("second response")
        .expect("second response should be ok");

    assert_eq!(first.request_id, "bootstrap-pd");
    assert!(matches!(first.body, Some(Body::Chunk(_))));
    assert_eq!(second.request_id, "bootstrap-pd");
    assert!(matches!(second.body, Some(Body::Complete(_))));
    assert!(
        tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
            .await
            .is_none()
    );
}

#[tokio::test]
async fn bootstrap_fake_pd_grpc_router_service_uses_decode_transfer_path() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--grpc-mode",
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "fake",
        "--disaggregation-decode-polling-interval",
        "1",
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("args should parse");
    let service = build_bootstrap_fake_pd_grpc_router_service(&args);

    let mut stream = service
        .text_generate(Request::new(TextGenerateRequest {
            text: "hi".to_string(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            options: Some(RequestOptions {
                request_id: Some("bootstrap-fake-pd".to_string()),
                stream: true,
                data_parallel_rank: 1,
                trace_headers: Default::default(),
            }),
            disaggregated_params: Some(
                sglang_srt::proto::sglang::runtime::v1::DisaggregatedParams {
                    bootstrap_host: "10.0.0.9".to_string(),
                    bootstrap_port: 8998,
                    bootstrap_room: 42,
                },
            ),
        }))
        .await
        .expect("fake PD bootstrap service should generate")
        .into_inner();

    let first = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("first response")
        .expect("first response should be ok");
    let second = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("second response")
        .expect("second response should be ok");

    assert_eq!(first.request_id, "bootstrap-fake-pd");
    assert!(matches!(first.body, Some(Body::Chunk(_))));
    assert_eq!(second.request_id, "bootstrap-fake-pd");
    assert!(matches!(second.body, Some(Body::Complete(_))));
}

#[tokio::test]
async fn bootstrap_pd_grpc_router_service_applies_max_running_requests() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--grpc-mode",
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "fake",
        "--max-running-requests",
        "1",
    ])
    .expect("args should parse");
    let service = build_bootstrap_fake_pd_grpc_router_service(&args);
    let runtime = service
        .runtime()
        .lock()
        .expect("runtime lock should be held");

    assert_eq!(runtime.engine().scheduler().max_running_requests(), Some(1));
}

#[tokio::test]
async fn launch_grpc_server_rejects_unsupported_bootstrap_pd_backend() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--grpc-mode",
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--kv-cache-dtype",
        "bfloat16",
        "--kv-cache-num-layers",
        "61",
        "--kv-cache-kv-heads",
        "1",
        "--kv-cache-head-dim",
        "512",
    ])
    .expect("args should parse");

    let error = launch_grpc_server(args)
        .await
        .expect_err("unsupported PD backend should fail before serving");

    assert_eq!(
        error,
        ServerLaunchError::UnsupportedBootstrapPdRuntime {
            mode: DisaggregationMode::Decode,
            transfer_backend: TransferBackend::Mooncake,
        }
    );
}

#[tokio::test]
async fn launch_grpc_server_requires_kv_model_layout_for_mooncake_decode() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--grpc-mode",
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--kv-cache-dtype",
        "bfloat16",
    ])
    .expect("args should parse");

    let error = launch_grpc_server(args)
        .await
        .expect_err("missing Mooncake KV layout should fail before serving");

    assert_eq!(
        error,
        ServerLaunchError::PdConfig(PdConfigError::MissingMooncakeKvCacheModelLayout)
    );
}

#[derive(Default)]
struct RecordingMooncakeBackend {
    submitted_batches: usize,
    statuses: Vec<MooncakeTransferStatusCode>,
    freed_batches: Vec<MooncakeBatchId>,
}

impl RecordingMooncakeBackend {
    fn completed() -> Self {
        Self {
            submitted_batches: 0,
            statuses: vec![MooncakeTransferStatusCode::Completed],
            freed_batches: Vec::new(),
        }
    }
}

impl MooncakeTransferSubmitter for RecordingMooncakeBackend {
    fn submit_transfer(
        &mut self,
        requests: &mut [MooncakeTransferRequest],
    ) -> Result<MooncakeBatchId, MooncakeError> {
        assert!(!requests.is_empty());
        self.submitted_batches += 1;
        Ok(700 + self.submitted_batches as MooncakeBatchId - 1)
    }
}

impl MooncakeTransferStatusReader for RecordingMooncakeBackend {
    fn transfer_status(
        &mut self,
        _batch_id: MooncakeBatchId,
        task_id: usize,
    ) -> Result<MooncakeTransferStatus, MooncakeError> {
        let status = self
            .statuses
            .get(task_id)
            .or_else(|| self.statuses.last())
            .copied()
            .expect("recording Mooncake backend needs at least one status");
        Ok(MooncakeTransferStatus {
            status: status as i32,
            transferred_bytes: 0,
        })
    }
}

impl MooncakeBatchReleaser for RecordingMooncakeBackend {
    fn free_batch(&mut self, batch_id: MooncakeBatchId) -> Result<(), MooncakeError> {
        self.freed_batches.push(batch_id);
        Ok(())
    }
}

fn word_level_tokenizer_json() -> &'static str {
    r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "Whitespace"
  },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "[UNK]": 0,
      "hello": 1,
      "world": 2
    },
    "unk_token": "[UNK]"
  }
}"#
}

fn deepseek_v4_model_config_json() -> &'static str {
    r#"{
  "model_type": "deepseek_v4",
  "vocab_size": 129280,
  "max_position_embeddings": 163840,
  "num_hidden_layers": 43
}"#
}

fn write_safetensors_file(
    path: &std::path::Path,
    tensors: &[(&str, &str, &[usize], [usize; 2])],
    payload: &[u8],
) -> std::io::Result<()> {
    let mut fields = Vec::new();
    for (name, dtype, shape, data_offsets) in tensors {
        let shape = shape
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        fields.push(format!(
            r#""{name}":{{"dtype":"{dtype}","shape":[{shape}],"data_offsets":[{},{}]}}"#,
            data_offsets[0], data_offsets[1]
        ));
    }
    let header = format!("{{{}}}", fields.join(","));
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(payload);
    fs::write(path, bytes)
}

fn temp_model_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("sglang-rs-{name}-{}", std::process::id()))
}
