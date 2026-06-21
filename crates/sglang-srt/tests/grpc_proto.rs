use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use prost::Message;
use tonic::codegen::tokio_stream::StreamExt;
use tonic::{Code, Request};

use sglang_srt::cache::{CachePageAllocator, RadixCache};
use sglang_srt::cli::ServerArgs;
use sglang_srt::engine::Engine;
use sglang_srt::grpc::{
    GrpcRouterService, SGLANG_RUNTIME_FILE_DESCRIPTOR_SET, router_protocol_error_to_status,
};
use sglang_srt::proto::sglang::runtime::v1::generate_response::Body;
use sglang_srt::proto::sglang::runtime::v1::sglang_service_server::SglangService;
use sglang_srt::proto::sglang::runtime::v1::{
    AbortRequest, ClassifyRequest, ContinueGenerationRequest, DetokenizeRequest,
    DisaggregatedParams, EmbedRequest, FlushCacheRequest, GenerateRequest, GetLoadRequest,
    GetModelInfoRequest, GetServerInfoRequest, ListModelsRequest, OpenAiJsonRequest,
    PauseGenerationRequest, RequestOptions, SamplingParams, StartProfileRequest,
    StopProfileRequest, TextEmbedRequest, TextGenerateRequest, TokenizeRequest, TokenizedInput,
    UpdateWeightVersionRequest, UpdateWeightsFromDiskRequest,
};
use sglang_srt::router::{RouterProtocolError, RouterRuntime};
use sglang_srt::scheduler::{ScheduleBatch, ScheduledRequest, Scheduler};
use sglang_srt::tokenizer::ByteTokenizer;
use sglang_srt::transfer::{
    DecodeBootstrapRegistry, DecodeBootstrapSession, KvTransferModelWorker, MooncakeBatchId,
    MooncakeBatchReleaser, MooncakeError, MooncakeKvCacheLayout, MooncakeKvCacheTransferExecutor,
    MooncakeTransferRequest, MooncakeTransferStatus, MooncakeTransferStatusCode,
    MooncakeTransferStatusReader, MooncakeTransferSubmitter, MooncakeTransferTarget,
};
use sglang_srt::types::{
    BootstrapRoom, DisaggregatedParams as RuntimeDisaggregatedParams, RequestId,
    SamplingParams as RuntimeSamplingParams,
};
use sglang_srt::worker::{
    BatchGeneratedTokens, FallibleModelWorker, GeneratedToken, ModelWorker, WorkerExecutionError,
    WorkerWeightUpdateRequest,
};

#[derive(Default)]
struct GrpcTwoStepWorker;

impl ModelWorker for GrpcTwoStepWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        let token = match batch.forward_mode() {
            sglang_srt::scheduler::ForwardMode::Prefill => GeneratedToken::unfinished(vec![42]),
            sglang_srt::scheduler::ForwardMode::Decode => GeneratedToken::finished(vec![43]),
        };

        BatchGeneratedTokens::from_batch(batch, vec![token])
            .expect("output shape should match batch")
    }
}

#[derive(Default)]
struct GrpcReloadingWorker {
    updates: Vec<WorkerWeightUpdateRequest>,
}

impl FallibleModelWorker for GrpcReloadingWorker {
    fn try_generate_batch(
        &mut self,
        batch: &ScheduleBatch,
    ) -> Result<BatchGeneratedTokens, WorkerExecutionError> {
        let token = match batch.forward_mode() {
            sglang_srt::scheduler::ForwardMode::Prefill => GeneratedToken::unfinished(vec![42]),
            sglang_srt::scheduler::ForwardMode::Decode => GeneratedToken::finished(vec![43]),
        };
        Ok(BatchGeneratedTokens::from_batch(batch, vec![token])
            .expect("output shape should match batch"))
    }

    fn update_weights_from_disk(
        &mut self,
        request: &WorkerWeightUpdateRequest,
    ) -> Result<(), WorkerExecutionError> {
        self.updates.push(request.clone());
        Ok(())
    }
}

fn unique_profile_dir() -> std::path::PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "sglang-rs-grpc-profile-{}-{suffix}",
        std::process::id()
    ))
}

fn unique_weight_update_model_dir() -> std::path::PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "sglang-rs-grpc-weight-update-{}-{suffix}",
        std::process::id()
    ))
}

fn write_minimal_generic_model_artifacts(model_dir: &Path) {
    fs::create_dir_all(model_dir).expect("model directory should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "architectures": ["TinyForCausalLM"],
  "model_type": "tiny",
  "vocab_size": 128,
  "max_position_embeddings": 4096,
  "eos_token_id": [2, 3]
}"#,
    )
    .expect("config should be written");
    write_minimal_safetensors_file(&model_dir.join("model.safetensors"));
}

fn write_minimal_safetensors_file(path: &Path) {
    let header =
        br#"{"model.embed_tokens.weight":{"dtype":"F32","shape":[1,1],"data_offsets":[0,4]}}"#;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&0.0f32.to_le_bytes());
    fs::write(path, bytes).expect("safetensors shard should be written");
}

#[test]
fn generated_proto_generate_request_round_trips_with_prost() {
    let request = GenerateRequest {
        input_ids: vec![101, 202, 303],
        original_text: "hello".to_string(),
        sampling_params: Some(SamplingParams {
            max_new_tokens: Some(16),
            temperature: Some(0.7),
            top_p: Some(0.95),
            stop_token_id: Some(2),
            stop_token_ids: vec![3, 4],
            ignore_eos: Some(true),
            ..Default::default()
        }),
        options: Some(RequestOptions {
            request_id: Some("grpc-rid".to_string()),
            stream: true,
            data_parallel_rank: 0,
            trace_headers: [("traceparent".to_string(), "00-abc".to_string())].into(),
        }),
        disaggregated_params: Some(DisaggregatedParams {
            bootstrap_host: "10.0.0.7".to_string(),
            bootstrap_port: 8998,
            bootstrap_room: i64::MAX as u64,
        }),
    };

    let mut bytes = Vec::new();
    request
        .encode(&mut bytes)
        .expect("generated request should encode");
    let decoded =
        GenerateRequest::decode(bytes.as_slice()).expect("generated request should decode");

    assert_eq!(decoded.input_ids, vec![101, 202, 303]);
    let sampling_params = decoded.sampling_params.expect("sampling params");
    assert_eq!(sampling_params.max_new_tokens, Some(16));
    assert_eq!(sampling_params.stop_token_id, Some(2));
    assert_eq!(sampling_params.stop_token_ids, vec![3, 4]);
    assert_eq!(sampling_params.ignore_eos, Some(true));
    assert_eq!(
        decoded
            .options
            .expect("request options")
            .trace_headers
            .get("traceparent"),
        Some(&"00-abc".to_string())
    );
    assert_eq!(
        decoded
            .disaggregated_params
            .expect("disaggregated params")
            .bootstrap_room,
        i64::MAX as u64
    );
}

#[test]
fn grpc_reflection_descriptor_registers_runtime_service() {
    let reflection_service = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(SGLANG_RUNTIME_FILE_DESCRIPTOR_SET)
        .build_v1();

    assert!(reflection_service.is_ok());
    assert!(
        SGLANG_RUNTIME_FILE_DESCRIPTOR_SET
            .windows(b"SglangService".len())
            .any(|window| window == b"SglangService")
    );
}

#[test]
fn router_protocol_errors_map_to_tonic_status_codes() {
    let invalid_argument =
        router_protocol_error_to_status(RouterProtocolError::InvalidIntegerSamplingParam {
            field: "max_new_tokens",
            value: 0,
            expected: "positive",
        });
    let resource_exhausted =
        router_protocol_error_to_status(RouterProtocolError::ContextOverflow {
            input_tokens: 3,
            max_new_tokens: 4,
            max_context_tokens: 6,
        });

    assert_eq!(invalid_argument.code(), Code::InvalidArgument);
    assert_eq!(resource_exhausted.code(), Code::ResourceExhausted);
}

#[tokio::test]
async fn grpc_generate_streams_router_runtime_outputs() {
    let service = GrpcRouterService::from_engine(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));

    let mut stream = service
        .generate(Request::new(GenerateRequest {
            input_ids: vec![1, 2, 3],
            original_text: String::new(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            options: Some(RequestOptions {
                request_id: Some("grpc-generate".to_string()),
                stream: true,
                data_parallel_rank: 0,
                trace_headers: Default::default(),
            }),
            disaggregated_params: None,
        }))
        .await
        .expect("grpc generate should execute")
        .into_inner();

    let first = stream
        .next()
        .await
        .expect("first response")
        .expect("first response ok");
    let second = stream
        .next()
        .await
        .expect("second response")
        .expect("second response ok");

    assert_eq!(first.request_id, "grpc-generate");
    assert_eq!(
        first.body,
        Some(Body::Chunk(
            sglang_srt::proto::sglang::runtime::v1::GenerateStreamChunk {
                token_ids: vec![42],
                text: String::new(),
                prompt_tokens: 3,
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
                output_ids: vec![42, 43],
                text: String::new(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 3,
                completion_tokens: 2,
                cached_tokens: 0,
                index: 0,
            }
        ))
    );
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn grpc_generate_maps_running_request_limit_to_resource_exhausted() {
    let mut scheduler = Scheduler::new(GrpcTwoStepWorker).with_max_running_requests(Some(1));
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("active"),
        vec![1],
        RuntimeSamplingParams::new(2),
    ));
    scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should occupy the active slot");
    let service = GrpcRouterService::from_engine(Engine::new(ByteTokenizer, scheduler));

    let result = service
        .generate(Request::new(GenerateRequest {
            input_ids: vec![9],
            original_text: String::new(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            options: Some(RequestOptions {
                request_id: Some("over-capacity".to_string()),
                stream: true,
                data_parallel_rank: 0,
                trace_headers: Default::default(),
            }),
            disaggregated_params: None,
        }))
        .await;

    let Err(error) = result else {
        panic!("capacity backpressure should be surfaced as grpc error");
    };
    assert_eq!(error.code(), Code::ResourceExhausted);
}

#[tokio::test]
async fn grpc_generate_non_stream_returns_only_complete_response() {
    let service = GrpcRouterService::from_engine(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));

    let mut stream = service
        .generate(Request::new(GenerateRequest {
            input_ids: vec![1, 2, 3],
            original_text: String::new(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            options: Some(RequestOptions {
                request_id: Some("grpc-non-stream".to_string()),
                stream: false,
                data_parallel_rank: 0,
                trace_headers: Default::default(),
            }),
            disaggregated_params: None,
        }))
        .await
        .expect("grpc generate should execute")
        .into_inner();

    let response = stream
        .next()
        .await
        .expect("complete response")
        .expect("complete response ok");

    assert_eq!(response.request_id, "grpc-non-stream");
    assert_eq!(
        response.body,
        Some(Body::Complete(
            sglang_srt::proto::sglang::runtime::v1::GenerateComplete {
                output_ids: vec![42, 43],
                text: String::new(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 3,
                completion_tokens: 2,
                cached_tokens: 0,
                index: 0,
            }
        ))
    );
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn grpc_text_embed_returns_proto_embeddings() {
    let service = GrpcRouterService::from_engine(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));

    let response = service
        .text_embed(Request::new(TextEmbedRequest {
            texts: vec!["rust pd router".to_string(), "python gateway".to_string()],
            options: None,
        }))
        .await
        .expect("text embed should execute")
        .into_inner();

    assert_eq!(response.embeddings.len(), 2);
    assert_eq!(response.embeddings[0].index, 0);
    assert_eq!(response.embeddings[1].index, 1);
    assert_eq!(response.embeddings[0].values.len(), 8);
    assert_eq!(response.embeddings[1].values.len(), 8);
    assert_ne!(response.embeddings[0].values, response.embeddings[1].values);
    let usage = response.usage.expect("usage");
    assert!(usage.prompt_tokens > 0);
    assert_eq!(usage.completion_tokens, 0);
    assert_eq!(usage.total_tokens, usage.prompt_tokens);
}

#[tokio::test]
async fn grpc_embed_returns_proto_embeddings_for_tokenized_inputs() {
    let service = GrpcRouterService::from_engine(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));

    let response = service
        .embed(Request::new(EmbedRequest {
            inputs: vec![
                TokenizedInput {
                    original_text: "rust".to_string(),
                    input_ids: vec![1, 2, 3],
                },
                TokenizedInput {
                    original_text: "gateway".to_string(),
                    input_ids: vec![4, 5],
                },
            ],
            options: None,
        }))
        .await
        .expect("embed should execute")
        .into_inner();

    assert_eq!(response.embeddings.len(), 2);
    assert_eq!(response.embeddings[0].index, 0);
    assert_eq!(response.embeddings[1].index, 1);
    assert_eq!(response.embeddings[0].values.len(), 8);
    assert_eq!(response.embeddings[1].values.len(), 8);
    let usage = response.usage.expect("usage");
    assert_eq!(usage.prompt_tokens, 5);
    assert_eq!(usage.completion_tokens, 0);
    assert_eq!(usage.total_tokens, 5);
}

#[tokio::test]
async fn grpc_classify_returns_proto_classifications() {
    let service = GrpcRouterService::from_engine(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));
    let labels = vec!["cpu".to_string(), "gpu".to_string(), "router".to_string()];

    let response = service
        .classify(Request::new(ClassifyRequest {
            inputs: vec![
                TokenizedInput {
                    original_text: "rust".to_string(),
                    input_ids: vec![1, 2, 3],
                },
                TokenizedInput {
                    original_text: "python".to_string(),
                    input_ids: vec![4, 5, 6, 7],
                },
            ],
            labels: labels.clone(),
            options: None,
        }))
        .await
        .expect("classify should execute")
        .into_inner();

    assert_eq!(response.classifications.len(), 2);
    assert_eq!(response.classifications[0].index, 0);
    assert_eq!(response.classifications[1].index, 1);
    for classification in &response.classifications {
        assert!(labels.contains(&classification.label));
        assert!(classification.score.is_finite());
        assert!((0.0..=1.0).contains(&classification.score));
    }
    let usage = response.usage.expect("usage");
    assert_eq!(usage.prompt_tokens, 7);
    assert_eq!(usage.completion_tokens, 0);
    assert_eq!(usage.total_tokens, 7);
}

#[tokio::test]
async fn grpc_chat_complete_returns_openai_chat_json() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny",
        "--grpc-mode",
    ])
    .expect("args should parse");
    let runtime = RouterRuntime::new(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));
    let service = GrpcRouterService::with_server_args(runtime, &args);
    let payload = serde_json::json!({
        "model": "tiny",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 2,
    });

    let mut stream = service
        .chat_complete(Request::new(OpenAiJsonRequest {
            json: serde_json::to_vec(&payload).expect("payload should serialize"),
            options: None,
        }))
        .await
        .expect("chat complete should execute")
        .into_inner();
    let response = stream
        .next()
        .await
        .expect("chat complete response")
        .expect("chat complete response ok");
    assert!(stream.next().await.is_none());

    let body: serde_json::Value =
        serde_json::from_slice(&response.json).expect("response should be JSON");
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "tiny");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
}

#[tokio::test]
async fn grpc_chat_complete_streams_openai_chat_chunks() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny",
        "--grpc-mode",
    ])
    .expect("args should parse");
    let runtime = RouterRuntime::new(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));
    let service = GrpcRouterService::with_server_args(runtime, &args);
    let payload = serde_json::json!({
        "model": "tiny",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 2,
        "stream": true,
    });

    let mut stream = service
        .chat_complete(Request::new(OpenAiJsonRequest {
            json: serde_json::to_vec(&payload).expect("payload should serialize"),
            options: None,
        }))
        .await
        .expect("streaming chat complete should execute")
        .into_inner();
    let first = stream
        .next()
        .await
        .expect("first chat chunk")
        .expect("first chat chunk ok");
    let second = stream
        .next()
        .await
        .expect("final chat chunk")
        .expect("final chat chunk ok");
    assert!(stream.next().await.is_none());

    let first: serde_json::Value =
        serde_json::from_slice(&first.json).expect("first chunk should be JSON");
    assert_eq!(first["object"], "chat.completion.chunk");
    assert_eq!(first["model"], "tiny");
    assert!(first["choices"][0]["delta"].get("content").is_some());
    assert!(first["choices"][0]["finish_reason"].is_null());

    let second: serde_json::Value =
        serde_json::from_slice(&second.json).expect("final chunk should be JSON");
    assert_eq!(second["object"], "chat.completion.chunk");
    assert_eq!(second["model"], "tiny");
    assert_eq!(second["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn grpc_complete_returns_openai_completion_json() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny",
        "--grpc-mode",
    ])
    .expect("args should parse");
    let runtime = RouterRuntime::new(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));
    let service = GrpcRouterService::with_server_args(runtime, &args);
    let payload = serde_json::json!({
        "model": "tiny",
        "prompt": "hi",
        "max_tokens": 2,
    });

    let mut stream = service
        .complete(Request::new(OpenAiJsonRequest {
            json: serde_json::to_vec(&payload).expect("payload should serialize"),
            options: None,
        }))
        .await
        .expect("complete should execute")
        .into_inner();
    let response = stream
        .next()
        .await
        .expect("completion response")
        .expect("completion response ok");
    assert!(stream.next().await.is_none());

    let body: serde_json::Value =
        serde_json::from_slice(&response.json).expect("response should be JSON");
    assert_eq!(body["object"], "text_completion");
    assert_eq!(body["model"], "tiny");
    assert_eq!(body["choices"][0]["text"], "*+");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn grpc_complete_streams_openai_completion_chunks() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny",
        "--grpc-mode",
    ])
    .expect("args should parse");
    let runtime = RouterRuntime::new(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));
    let service = GrpcRouterService::with_server_args(runtime, &args);
    let payload = serde_json::json!({
        "model": "tiny",
        "prompt": "hi",
        "max_tokens": 2,
        "stream": true,
    });

    let mut stream = service
        .complete(Request::new(OpenAiJsonRequest {
            json: serde_json::to_vec(&payload).expect("payload should serialize"),
            options: None,
        }))
        .await
        .expect("streaming complete should execute")
        .into_inner();
    let first = stream
        .next()
        .await
        .expect("first completion chunk")
        .expect("first completion chunk ok");
    let second = stream
        .next()
        .await
        .expect("final completion chunk")
        .expect("final completion chunk ok");
    assert!(stream.next().await.is_none());

    let first: serde_json::Value =
        serde_json::from_slice(&first.json).expect("first chunk should be JSON");
    assert_eq!(first["object"], "text_completion");
    assert_eq!(first["choices"][0]["text"], "*");
    assert!(first["choices"][0]["finish_reason"].is_null());

    let second: serde_json::Value =
        serde_json::from_slice(&second.json).expect("final chunk should be JSON");
    assert_eq!(second["object"], "text_completion");
    assert_eq!(second["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn grpc_rerank_returns_raw_worker_results() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny",
        "--grpc-mode",
    ])
    .expect("args should parse");
    let runtime = RouterRuntime::new(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));
    let service = GrpcRouterService::with_server_args(runtime, &args);
    let payload = serde_json::json!({
        "model": "tiny",
        "query": "rust pd router",
        "documents": [
            "python gateway only",
            "rust pd router transfers kv cache",
            "router"
        ],
    });

    let response = service
        .rerank(Request::new(OpenAiJsonRequest {
            json: serde_json::to_vec(&payload).expect("payload should serialize"),
            options: None,
        }))
        .await
        .expect("rerank should execute")
        .into_inner();

    let body: serde_json::Value =
        serde_json::from_slice(&response.json).expect("response should be JSON");
    let results = body.as_array().expect("worker should return raw list");
    assert_eq!(results.len(), 3);
    assert_eq!(results[0]["index"], 1);
    assert_eq!(results[0]["document"], "rust pd router transfers kv cache");
    assert_eq!(results[1]["index"], 2);
    assert_eq!(results[2]["index"], 0);
    assert!(
        results[0]["score"].as_f64().unwrap() > results[1]["score"].as_f64().unwrap(),
        "more overlapping tokens should score higher: {results:?}"
    );
}

#[tokio::test]
async fn grpc_score_returns_openai_scoring_json() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny",
        "--grpc-mode",
    ])
    .expect("args should parse");
    let runtime = RouterRuntime::new(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));
    let service = GrpcRouterService::with_server_args(runtime, &args);
    let payload = serde_json::json!({
        "model": "tiny",
        "query": "rust pd router",
        "items": [
            "rust pd router transfers kv cache",
            "unrelated python gateway"
        ],
        "label_token_ids": [1, 2, 3],
        "apply_softmax": true
    });

    let response = service
        .score(Request::new(OpenAiJsonRequest {
            json: serde_json::to_vec(&payload).expect("payload should serialize"),
            options: None,
        }))
        .await
        .expect("score should execute")
        .into_inner();

    let body: serde_json::Value =
        serde_json::from_slice(&response.json).expect("response should be JSON");
    assert_eq!(body["object"], "scoring");
    assert_eq!(body["model"], "tiny");
    assert!(body["pooled_hidden_states"].is_null());
    let scores = body["scores"].as_array().expect("scores");
    assert_eq!(scores.len(), 2);
    for row in scores {
        let row = row.as_array().expect("score row");
        assert_eq!(row.len(), 3);
        let sum = row.iter().map(|value| value.as_f64().unwrap()).sum::<f64>();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "softmax score row should sum to 1: {row:?}"
        );
    }
    assert!(body["usage"]["prompt_tokens"].as_i64().unwrap() > 0);
    assert_eq!(body["usage"]["completion_tokens"], 0);
    assert_eq!(
        body["usage"]["total_tokens"],
        body["usage"]["prompt_tokens"]
    );
}

#[tokio::test]
async fn grpc_open_ai_embed_returns_embedding_json() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny",
        "--grpc-mode",
    ])
    .expect("args should parse");
    let runtime = RouterRuntime::new(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));
    let service = GrpcRouterService::with_server_args(runtime, &args);
    let payload = serde_json::json!({
        "model": "tiny",
        "input": ["rust pd router", "python gateway"],
        "dimensions": 4,
    });

    let response = service
        .open_ai_embed(Request::new(OpenAiJsonRequest {
            json: serde_json::to_vec(&payload).expect("payload should serialize"),
            options: None,
        }))
        .await
        .expect("OpenAI embed should execute")
        .into_inner();

    let body: serde_json::Value =
        serde_json::from_slice(&response.json).expect("response should be JSON");
    assert_eq!(body["object"], "list");
    assert_eq!(body["model"], "tiny");
    assert_eq!(body["data"].as_array().unwrap().len(), 2);
    assert_eq!(body["data"][0]["index"], 0);
    assert_eq!(body["data"][0]["embedding"].as_array().unwrap().len(), 4);
    assert_ne!(body["data"][0]["embedding"], body["data"][1]["embedding"]);
    assert!(body["usage"]["prompt_tokens"].as_i64().unwrap() > 0);
}

#[tokio::test]
async fn grpc_open_ai_classify_returns_classify_json() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny",
        "--grpc-mode",
    ])
    .expect("args should parse");
    let runtime = RouterRuntime::new(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));
    let service = GrpcRouterService::with_server_args(runtime, &args);
    let payload = serde_json::json!({
        "model": "tiny",
        "input": ["rust pd router", "python gateway"],
    });

    let response = service
        .open_ai_classify(Request::new(OpenAiJsonRequest {
            json: serde_json::to_vec(&payload).expect("payload should serialize"),
            options: None,
        }))
        .await
        .expect("OpenAI classify should execute")
        .into_inner();

    let body: serde_json::Value =
        serde_json::from_slice(&response.json).expect("response should be JSON");
    assert_eq!(body["object"], "list");
    assert_eq!(body["model"], "tiny");
    assert!(body["id"].as_str().unwrap().starts_with("classify-"));
    assert_eq!(body["data"].as_array().unwrap().len(), 2);
    assert_eq!(body["data"][0]["index"], 0);
    assert!(
        body["data"][0]["label"]
            .as_str()
            .unwrap()
            .starts_with("LABEL_")
    );
    assert_eq!(body["data"][0]["num_classes"], 3);
    assert_eq!(body["data"][0]["probs"].as_array().unwrap().len(), 3);
    assert_ne!(body["data"][0]["probs"], body["data"][1]["probs"]);
    assert!(body["usage"]["prompt_tokens"].as_i64().unwrap() > 0);
}

#[tokio::test]
async fn grpc_generate_can_poll_pd_transfer_before_decode() {
    let worker = KvTransferModelWorker::new(
        GrpcTwoStepWorker,
        grpc_registry_with_session("grpc-pd-poll", 31),
        MooncakeKvCacheTransferExecutor::new(
            RecordingMooncakeBackend::completed(),
            MooncakeKvCacheLayout {
                source_base_addr: 0x1000,
                page_size_bytes: 64,
                target_base_offset: 0,
            },
            MooncakeTransferTarget { target_id: 9 },
        ),
    );
    let service = GrpcRouterService::from_engine(Engine::new(
        ByteTokenizer,
        Scheduler::with_cache_resources(worker, RadixCache::default(), CachePageAllocator::new(3)),
    ))
    .with_max_transfer_polls(1);

    let mut stream = service
        .generate(Request::new(GenerateRequest {
            input_ids: vec![1, 2],
            original_text: String::new(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            options: Some(RequestOptions {
                request_id: Some("grpc-pd-poll".to_string()),
                stream: true,
                data_parallel_rank: 0,
                trace_headers: Default::default(),
            }),
            disaggregated_params: Some(
                sglang_srt::proto::sglang::runtime::v1::DisaggregatedParams {
                    bootstrap_host: "10.0.0.7".to_string(),
                    bootstrap_port: 8998,
                    bootstrap_room: 31,
                },
            ),
        }))
        .await
        .expect("grpc generate should poll transfer and execute")
        .into_inner();

    let first = stream
        .next()
        .await
        .expect("first response")
        .expect("first response ok");
    let second = stream
        .next()
        .await
        .expect("second response")
        .expect("second response ok");

    assert!(matches!(first.body, Some(Body::Chunk(_))));
    assert!(matches!(second.body, Some(Body::Complete(_))));
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn grpc_text_generate_can_poll_pd_transfer_before_decode() {
    let worker = KvTransferModelWorker::new(
        GrpcTwoStepWorker,
        grpc_registry_with_session("grpc-text-pd-poll", 32),
        MooncakeKvCacheTransferExecutor::new(
            RecordingMooncakeBackend::completed(),
            MooncakeKvCacheLayout {
                source_base_addr: 0x2000,
                page_size_bytes: 64,
                target_base_offset: 0,
            },
            MooncakeTransferTarget { target_id: 10 },
        ),
    );
    let service = GrpcRouterService::from_engine(Engine::new(
        ByteTokenizer,
        Scheduler::with_cache_resources(worker, RadixCache::default(), CachePageAllocator::new(3)),
    ))
    .with_max_transfer_polls(1);

    let mut stream = service
        .text_generate(Request::new(TextGenerateRequest {
            text: "Hi".to_string(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            options: Some(RequestOptions {
                request_id: Some("grpc-text-pd-poll".to_string()),
                stream: true,
                data_parallel_rank: 0,
                trace_headers: Default::default(),
            }),
            disaggregated_params: Some(
                sglang_srt::proto::sglang::runtime::v1::DisaggregatedParams {
                    bootstrap_host: "10.0.0.7".to_string(),
                    bootstrap_port: 8998,
                    bootstrap_room: 32,
                },
            ),
        }))
        .await
        .expect("grpc text generate should poll transfer and execute")
        .into_inner();

    let first = stream
        .next()
        .await
        .expect("first response")
        .expect("first response ok");
    let second = stream
        .next()
        .await
        .expect("second response")
        .expect("second response ok");

    assert_eq!(first.request_id, "grpc-text-pd-poll");
    assert!(matches!(first.body, Some(Body::Chunk(_))));
    assert_eq!(second.request_id, "grpc-text-pd-poll");
    assert!(matches!(second.body, Some(Body::Complete(_))));
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn grpc_text_generate_tokenizes_prompt_and_streams_decoded_text() {
    let service = GrpcRouterService::from_engine(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));

    let mut stream = service
        .text_generate(Request::new(TextGenerateRequest {
            text: "Hello".to_string(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            options: Some(RequestOptions {
                request_id: Some("grpc-text".to_string()),
                stream: true,
                data_parallel_rank: 0,
                trace_headers: Default::default(),
            }),
            disaggregated_params: None,
        }))
        .await
        .expect("grpc text generate should execute")
        .into_inner();

    let first = stream
        .next()
        .await
        .expect("first response")
        .expect("first response ok");
    let second = stream
        .next()
        .await
        .expect("second response")
        .expect("second response ok");

    assert_eq!(
        first.body,
        Some(Body::Chunk(
            sglang_srt::proto::sglang::runtime::v1::GenerateStreamChunk {
                token_ids: vec![42],
                text: "*".to_string(),
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
                output_ids: vec![42, 43],
                text: "*+".to_string(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 5,
                completion_tokens: 2,
                cached_tokens: 0,
                index: 0,
            }
        ))
    );
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn grpc_generate_maps_router_protocol_errors_to_status() {
    let service = GrpcRouterService::from_engine(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));

    let error = match service
        .generate(Request::new(GenerateRequest {
            input_ids: Vec::new(),
            original_text: String::new(),
            sampling_params: None,
            options: None,
            disaggregated_params: None,
        }))
        .await
    {
        Ok(_) => panic!("empty input should be rejected before dispatch"),
        Err(error) => error,
    };

    assert_eq!(error.code(), Code::InvalidArgument);
    assert!(error.message().contains("empty router tokenized input"));
}

#[tokio::test]
async fn grpc_text_generate_maps_empty_text_to_invalid_argument() {
    let service = GrpcRouterService::from_engine(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));

    let error = match service
        .text_generate(Request::new(TextGenerateRequest {
            text: String::new(),
            sampling_params: None,
            options: None,
            disaggregated_params: None,
        }))
        .await
    {
        Ok(_) => panic!("empty text should be rejected"),
        Err(error) => error,
    };

    assert_eq!(error.code(), Code::InvalidArgument);
    assert!(error.message().contains("empty router text input"));
}

#[tokio::test]
async fn grpc_tokenize_uses_router_tokenizer() {
    let service = GrpcRouterService::from_engine(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));

    let response = service
        .tokenize(Request::new(TokenizeRequest {
            text: "Hello".to_string(),
            add_special_tokens: true,
        }))
        .await
        .expect("tokenize should execute")
        .into_inner();

    assert_eq!(response.token_ids, vec![72, 101, 108, 108, 111]);
    assert_eq!(response.count, 5);
}

#[tokio::test]
async fn grpc_detokenize_uses_router_tokenizer() {
    let service = GrpcRouterService::from_engine(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));

    let response = service
        .detokenize(Request::new(DetokenizeRequest {
            token_ids: vec![72, 101, 108, 108, 111],
            skip_special_tokens: true,
        }))
        .await
        .expect("detokenize should execute")
        .into_inner();

    assert_eq!(response.text, "Hello");
}

#[tokio::test]
async fn grpc_detokenize_maps_tokenizer_errors_to_invalid_argument() {
    let service = GrpcRouterService::from_engine(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));

    let error = service
        .detokenize(Request::new(DetokenizeRequest {
            token_ids: vec![u32::from(u8::MAX) + 1],
            skip_special_tokens: false,
        }))
        .await
        .expect_err("invalid token ids should be rejected");

    assert_eq!(error.code(), Code::InvalidArgument);
    assert!(error.message().contains("not valid UTF-8"));
}

#[tokio::test]
async fn grpc_get_model_info_reports_configured_server_args() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "meta-llama/Llama-3.1-8B-Instruct",
        "--served-model-name",
        "llama3",
        "--tokenizer-path",
        "hf-tokenizer",
    ])
    .expect("server args should parse");
    let runtime = RouterRuntime::new(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));
    let service = GrpcRouterService::with_server_args(runtime, &args);

    let response = service
        .get_model_info(Request::new(GetModelInfoRequest {}))
        .await
        .expect("model info should execute")
        .into_inner();

    assert_eq!(response.model_path, "meta-llama/Llama-3.1-8B-Instruct");
    assert_eq!(response.tokenizer_path, "hf-tokenizer");
    assert_eq!(response.served_model_name, "llama3");
    assert!(response.is_generation);
    assert_eq!(response.preferred_sampling_params_json, "{}");
}

#[tokio::test]
async fn grpc_list_models_returns_configured_model_info() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "Qwen/Qwen3-4B",
        "--served-model-name",
        "qwen3",
    ])
    .expect("server args should parse");
    let runtime = RouterRuntime::new(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));
    let service = GrpcRouterService::with_server_args(runtime, &args);

    let response = service
        .list_models(Request::new(ListModelsRequest {}))
        .await
        .expect("list models should execute")
        .into_inner();

    assert_eq!(response.models.len(), 1);
    assert_eq!(response.models[0].model_path, "Qwen/Qwen3-4B");
    assert_eq!(response.models[0].tokenizer_path, "Qwen/Qwen3-4B");
    assert_eq!(response.models[0].served_model_name, "qwen3");
}

#[tokio::test]
async fn grpc_model_info_requires_configured_metadata() {
    let service = GrpcRouterService::from_engine(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));

    let error = service
        .get_model_info(Request::new(GetModelInfoRequest {}))
        .await
        .expect_err("model info should require configured metadata");

    assert_eq!(error.code(), Code::FailedPrecondition);
    assert!(error.message().contains("model info is not configured"));
}

#[tokio::test]
async fn grpc_get_server_info_reports_rust_runtime() {
    let service = GrpcRouterService::from_engine(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));

    let response = service
        .get_server_info(Request::new(GetServerInfoRequest {}))
        .await
        .expect("server info should execute")
        .into_inner();

    assert_eq!(response.version, env!("CARGO_PKG_VERSION"));
    assert_eq!(response.runtime, "sglang-rs");
    assert_eq!(
        response.attributes.get("transport"),
        Some(&"tonic-grpc".to_string())
    );
}

#[tokio::test]
async fn grpc_get_server_info_reports_pd_prefill_attributes() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "grpc-prefill-info",
        "--grpc-mode",
        "--host",
        "0.0.0.0",
        "--dist-init-addr",
        "10.95.250.21:6676",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-bootstrap-port",
        "8200",
        "--disaggregation-zmq-ports",
        "7000-7001",
        "--tp-size",
        "2",
        "--dp-size",
        "1",
        "--page-size",
        "64",
    ])
    .expect("args should parse");
    let service = GrpcRouterService::with_server_args(
        RouterRuntime::new(Engine::new(
            ByteTokenizer,
            Scheduler::new(GrpcTwoStepWorker),
        )),
        &args,
    );

    let response = service
        .get_server_info(Request::new(GetServerInfoRequest {}))
        .await
        .expect("server info should execute")
        .into_inner();

    assert_eq!(
        response.attributes.get("served_model_name"),
        Some(&"grpc-prefill-info".to_string())
    );
    assert_eq!(
        response.attributes.get("disaggregation_mode"),
        Some(&"prefill".to_string())
    );
    assert_eq!(
        response.attributes.get("disaggregation_bootstrap_port"),
        Some(&"8200".to_string())
    );
    assert_eq!(
        response.attributes.get("kv_events.publisher"),
        Some(&"zmq".to_string())
    );
    assert_eq!(
        response.attributes.get("kv_events.endpoint_host"),
        Some(&"10.95.250.21".to_string())
    );
    assert_eq!(
        response.attributes.get("kv_events.endpoint_port_base"),
        Some(&"7000".to_string())
    );
    assert_eq!(
        response.attributes.get("kv_events.block_size"),
        Some(&"64".to_string())
    );
    assert_eq!(
        response.attributes.get("kv_events.dp_size"),
        Some(&"1".to_string())
    );
}

#[tokio::test]
async fn grpc_get_load_reports_scheduler_metrics() {
    let mut scheduler = Scheduler::with_cache_resources(
        GrpcTwoStepWorker,
        RadixCache::default(),
        CachePageAllocator::new(4),
    );
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("load-waiting"),
        vec![1, 2, 3],
        RuntimeSamplingParams::new(1),
    ));
    let runtime = RouterRuntime::new(Engine::new(ByteTokenizer, scheduler));
    let service = GrpcRouterService::new(runtime);

    let response = service
        .get_load(Request::new(GetLoadRequest {}))
        .await
        .expect("get load should execute")
        .into_inner();

    assert_eq!(response.waiting_queue_depth, 1);
    assert_eq!(response.decode_queue_depth, 0);
    assert_eq!(response.available_cache_pages, Some(4));
}

#[tokio::test]
async fn grpc_flush_cache_calls_router_runtime() {
    let service = GrpcRouterService::from_engine(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));

    let response = service
        .flush_cache(Request::new(FlushCacheRequest {}))
        .await
        .expect("flush cache should execute")
        .into_inner();

    assert!(response.success);
    assert_eq!(response.message, "cache flushed");
}

#[tokio::test]
async fn grpc_profile_start_stop_writes_trace_file() {
    let service = GrpcRouterService::from_engine(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));
    let output_dir = unique_profile_dir();

    let start = service
        .start_profile(Request::new(StartProfileRequest {
            output_dir: Some(output_dir.to_string_lossy().to_string()),
        }))
        .await
        .expect("start profile should execute")
        .into_inner();
    assert!(start.success);
    assert!(start.message.contains("profile started"));

    let duplicate = service
        .start_profile(Request::new(StartProfileRequest { output_dir: None }))
        .await
        .expect_err("duplicate profile start should be rejected");
    assert_eq!(duplicate.code(), Code::AlreadyExists);

    let stop = service
        .stop_profile(Request::new(StopProfileRequest {}))
        .await
        .expect("stop profile should execute")
        .into_inner();
    assert!(stop.success);
    assert!(stop.message.contains("profile stopped"));

    let entries = fs::read_dir(&output_dir)
        .expect("profile output directory should exist")
        .collect::<Result<Vec<_>, _>>()
        .expect("profile directory should be readable");
    assert_eq!(entries.len(), 1);
    let profile: serde_json::Value = serde_json::from_slice(
        &fs::read(entries[0].path()).expect("profile file should be readable"),
    )
    .expect("profile file should contain JSON");
    assert_eq!(profile["profile"]["transport"], "tonic-grpc");
    assert!(profile["profile"]["duration_ms"].as_u64().is_some());
    assert!(
        profile["profile"]["started_unix_ms"].as_u64().unwrap()
            <= profile["profile"]["stopped_unix_ms"].as_u64().unwrap()
    );

    fs::remove_dir_all(output_dir).expect("profile temp directory should clean up");
}

#[tokio::test]
async fn grpc_update_weights_from_disk_validates_artifacts_and_updates_model_info() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "old-model",
        "--served-model-name",
        "tiny",
    ])
    .expect("server args should parse");
    let service = GrpcRouterService::with_server_args(
        RouterRuntime::new(Engine::new(
            ByteTokenizer,
            Scheduler::new(GrpcReloadingWorker::default()),
        )),
        &args,
    );
    let model_dir = unique_weight_update_model_dir();
    write_minimal_generic_model_artifacts(&model_dir);

    let response = service
        .update_weights_from_disk(Request::new(UpdateWeightsFromDiskRequest {
            model_path: model_dir.to_string_lossy().to_string(),
            load_format: Some("safetensors".to_string()),
        }))
        .await
        .expect("local safetensors artifacts should update model metadata")
        .into_inner();
    assert!(response.success);
    assert!(response.message.contains("registered"));

    let model_info = service
        .get_model_info(Request::new(GetModelInfoRequest {}))
        .await
        .expect("updated model info should be readable")
        .into_inner();
    assert_eq!(model_info.model_path, model_dir.to_string_lossy());
    assert_eq!(model_info.tokenizer_path, model_dir.to_string_lossy());
    assert_eq!(model_info.served_model_name, "tiny");
    assert_eq!(model_info.model_type, "tiny");
    assert_eq!(model_info.eos_token_ids, vec![2, 3]);
    assert_eq!(model_info.vocab_size, 128);
    assert_eq!(model_info.max_context_length, 4096);
    assert!(model_info.weight_version.starts_with("safetensors-sha256:"));

    fs::remove_dir_all(model_dir).expect("model temp directory should clean up");
}

#[tokio::test]
async fn grpc_update_weights_from_disk_rejects_invalid_requests() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "old-model",
        "--served-model-name",
        "tiny",
    ])
    .expect("server args should parse");
    let service = GrpcRouterService::with_server_args(
        RouterRuntime::new(Engine::new(
            ByteTokenizer,
            Scheduler::new(GrpcTwoStepWorker),
        )),
        &args,
    );

    let empty_path = service
        .update_weights_from_disk(Request::new(UpdateWeightsFromDiskRequest {
            model_path: "  ".to_string(),
            load_format: None,
        }))
        .await
        .expect_err("empty model paths should be rejected");
    assert_eq!(empty_path.code(), Code::InvalidArgument);
    assert!(empty_path.message().contains("model_path"));

    let unsupported_format = service
        .update_weights_from_disk(Request::new(UpdateWeightsFromDiskRequest {
            model_path: "some-model".to_string(),
            load_format: Some("gguf".to_string()),
        }))
        .await
        .expect_err("unsupported load formats should be rejected");
    assert_eq!(unsupported_format.code(), Code::InvalidArgument);
    assert!(unsupported_format.message().contains("load_format"));
}

#[tokio::test]
async fn grpc_update_weight_version_updates_model_info() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "old-model",
        "--served-model-name",
        "tiny",
    ])
    .expect("server args should parse");
    let service = GrpcRouterService::with_server_args(
        RouterRuntime::new(Engine::new(
            ByteTokenizer,
            Scheduler::new(GrpcTwoStepWorker),
        )),
        &args,
    );

    let response = service
        .update_weight_version(Request::new(UpdateWeightVersionRequest {
            new_version: "grpc-checkpoint-7".to_string(),
            abort_all_requests: Some(false),
        }))
        .await
        .expect("weight version update should execute")
        .into_inner();
    assert!(response.success);
    assert!(response.message.contains("grpc-checkpoint-7"));

    let model_info = service
        .get_model_info(Request::new(GetModelInfoRequest {}))
        .await
        .expect("updated model info should be readable")
        .into_inner();
    assert_eq!(model_info.weight_version, "grpc-checkpoint-7");
}

#[tokio::test]
async fn grpc_abort_removes_queued_request() {
    let mut scheduler = Scheduler::new(GrpcTwoStepWorker);
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("grpc-abort"),
        vec![1, 2, 3],
        RuntimeSamplingParams::new(1),
    ));
    let runtime = RouterRuntime::new(Engine::new(ByteTokenizer, scheduler));
    let service = GrpcRouterService::new(runtime);

    let response = service
        .abort(Request::new(AbortRequest {
            request_id: "grpc-abort".to_string(),
            abort_all: false,
        }))
        .await
        .expect("abort should execute")
        .into_inner();

    assert!(response.success);
    assert_eq!(response.message, "request aborted");

    let load = service
        .get_load(Request::new(GetLoadRequest {}))
        .await
        .expect("load should execute")
        .into_inner();

    assert_eq!(load.waiting_queue_depth, 0);

    let missing = service
        .abort(Request::new(AbortRequest {
            request_id: "missing".to_string(),
            abort_all: false,
        }))
        .await
        .expect("abort for missing request should execute")
        .into_inner();

    assert!(!missing.success);
    assert_eq!(missing.message, "request not found");
}

#[tokio::test]
async fn grpc_abort_all_removes_all_queued_requests() {
    let mut scheduler = Scheduler::new(GrpcTwoStepWorker);
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("grpc-abort-a"),
        vec![1],
        RuntimeSamplingParams::new(1),
    ));
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("grpc-abort-b"),
        vec![2],
        RuntimeSamplingParams::new(1),
    ));
    let runtime = RouterRuntime::new(Engine::new(ByteTokenizer, scheduler));
    let service = GrpcRouterService::new(runtime);

    let response = service
        .abort(Request::new(AbortRequest {
            request_id: String::new(),
            abort_all: true,
        }))
        .await
        .expect("abort_all should execute")
        .into_inner();

    assert!(response.success);
    assert_eq!(response.message, "aborted 2 request(s)");

    let load = service
        .get_load(Request::new(GetLoadRequest {}))
        .await
        .expect("load should execute")
        .into_inner();

    assert_eq!(load.waiting_queue_depth, 0);
}

#[tokio::test]
async fn grpc_abort_rejects_empty_request_id() {
    let service = GrpcRouterService::from_engine(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));

    let error = service
        .abort(Request::new(AbortRequest {
            request_id: String::new(),
            abort_all: false,
        }))
        .await
        .expect_err("empty request id should be rejected");

    assert_eq!(error.code(), Code::InvalidArgument);
    assert!(error.message().contains("missing router request id"));
}

#[tokio::test]
async fn grpc_pause_generation_rejects_generate_until_continued() {
    let service = GrpcRouterService::from_engine(Engine::new(
        ByteTokenizer,
        Scheduler::new(GrpcTwoStepWorker),
    ));

    let pause_response = service
        .pause_generation(Request::new(PauseGenerationRequest {}))
        .await
        .expect("pause generation should execute")
        .into_inner();

    assert!(pause_response.success);
    assert_eq!(pause_response.message, "generation paused");

    let paused_error = match service
        .generate(Request::new(GenerateRequest {
            input_ids: vec![1, 2, 3],
            original_text: String::new(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            options: Some(RequestOptions {
                request_id: Some("grpc-paused".to_string()),
                stream: false,
                data_parallel_rank: 0,
                trace_headers: Default::default(),
            }),
            disaggregated_params: None,
        }))
        .await
    {
        Ok(_) => panic!("paused generation should be rejected"),
        Err(error) => error,
    };

    assert_eq!(paused_error.code(), Code::FailedPrecondition);
    assert!(paused_error.message().contains("generation is paused"));

    let continue_response = service
        .continue_generation(Request::new(ContinueGenerationRequest {}))
        .await
        .expect("continue generation should execute")
        .into_inner();

    assert!(continue_response.success);
    assert_eq!(continue_response.message, "generation continued");

    let mut stream = service
        .generate(Request::new(GenerateRequest {
            input_ids: vec![1, 2, 3],
            original_text: String::new(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            options: Some(RequestOptions {
                request_id: Some("grpc-continued".to_string()),
                stream: false,
                data_parallel_rank: 0,
                trace_headers: Default::default(),
            }),
            disaggregated_params: None,
        }))
        .await
        .expect("continued generation should execute")
        .into_inner();

    let response = stream
        .next()
        .await
        .expect("complete response")
        .expect("complete response ok");

    assert_eq!(response.request_id, "grpc-continued");
    assert!(matches!(response.body, Some(Body::Complete(_))));
    assert!(stream.next().await.is_none());
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
        Ok(500 + self.submitted_batches as MooncakeBatchId - 1)
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

fn grpc_registry_with_session(
    request_id: &str,
    bootstrap_room: BootstrapRoom,
) -> DecodeBootstrapRegistry {
    let mut registry = DecodeBootstrapRegistry::default();
    registry
        .register(DecodeBootstrapSession::new(
            RequestId::from(request_id),
            RuntimeDisaggregatedParams {
                bootstrap_host: "10.0.0.7".to_string(),
                bootstrap_port: 8998,
                bootstrap_room,
            },
            0,
        ))
        .expect("session should register");
    registry
}
