use prost::Message;
use tonic::Code;

use sglang_srt::grpc::router_protocol_error_to_status;
use sglang_srt::proto::sglang::runtime::v1::{GenerateRequest, RequestOptions, SamplingParams};
use sglang_srt::router::RouterProtocolError;

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
