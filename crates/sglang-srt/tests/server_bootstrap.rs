use tonic::Request;

use sglang_srt::cli::ServerArgs;
use sglang_srt::proto::sglang::runtime::v1::generate_response::Body;
use sglang_srt::proto::sglang::runtime::v1::sglang_service_server::SglangService;
use sglang_srt::proto::sglang::runtime::v1::{
    GetModelInfoRequest, RequestOptions, SamplingParams, TextGenerateRequest,
};
use sglang_srt::server::{build_bootstrap_grpc_router_service, grpc_listen_addr};

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
async fn bootstrap_grpc_router_service_generates_through_model_runner() {
    let args = ServerArgs::parse_from(["serve", "--model-path", "dummy", "--grpc-mode"])
        .expect("args should parse");
    let service = build_bootstrap_grpc_router_service(&args);

    let mut stream = service
        .text_generate(Request::new(TextGenerateRequest {
            text: "hello".to_string(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(1),
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
