use prost::Message;
use tonic::codegen::tokio_stream::StreamExt;
use tonic::{Code, Request};

use sglang_srt::engine::Engine;
use sglang_srt::grpc::{GrpcRouterService, router_protocol_error_to_status};
use sglang_srt::proto::sglang::runtime::v1::generate_response::Body;
use sglang_srt::proto::sglang::runtime::v1::sglang_service_server::SglangService;
use sglang_srt::proto::sglang::runtime::v1::{
    FlushCacheRequest, GenerateRequest, RequestOptions, SamplingParams,
};
use sglang_srt::router::RouterProtocolError;
use sglang_srt::scheduler::{ScheduleBatch, Scheduler};
use sglang_srt::tokenizer::ByteTokenizer;
use sglang_srt::worker::{BatchGeneratedTokens, GeneratedToken, ModelWorker};

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

#[test]
fn generated_proto_generate_request_round_trips_with_prost() {
    let request = GenerateRequest {
        input_ids: vec![101, 202, 303],
        original_text: "hello".to_string(),
        sampling_params: Some(SamplingParams {
            max_new_tokens: Some(16),
            temperature: Some(0.7),
            top_p: Some(0.95),
            ..Default::default()
        }),
        options: Some(RequestOptions {
            request_id: Some("grpc-rid".to_string()),
            stream: true,
            data_parallel_rank: 0,
            trace_headers: [("traceparent".to_string(), "00-abc".to_string())].into(),
        }),
        disaggregated_params: None,
    };

    let mut bytes = Vec::new();
    request
        .encode(&mut bytes)
        .expect("generated request should encode");
    let decoded =
        GenerateRequest::decode(bytes.as_slice()).expect("generated request should decode");

    assert_eq!(decoded.input_ids, vec![101, 202, 303]);
    assert_eq!(
        decoded
            .sampling_params
            .expect("sampling params")
            .max_new_tokens,
        Some(16)
    );
    assert_eq!(
        decoded
            .options
            .expect("request options")
            .trace_headers
            .get("traceparent"),
        Some(&"00-abc".to_string())
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
