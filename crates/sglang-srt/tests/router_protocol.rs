use sglang_srt::cli::ServerArgs;
use sglang_srt::engine::Engine;
use sglang_srt::router::{
    DEFAULT_MAX_NEW_TOKENS, RouterGenerateComplete, RouterGenerateRequest, RouterGenerateResponse,
    RouterGenerateResponseBody, RouterGenerateStreamChunk, RouterGetModelInfoResponse,
    RouterHealthCheckResponse, RouterProtocolError, RouterRuntime, RouterSamplingParams,
    RouterTokenizedInput,
};
use sglang_srt::scheduler::{ScheduleBatch, Scheduler};
use sglang_srt::tokenizer::ByteTokenizer;
use sglang_srt::types::{RequestId, SamplingParams, TokenGenerateOutput};
use sglang_srt::worker::{BatchGeneratedTokens, GeneratedToken, ModelWorker};

#[derive(Default)]
struct RouterEchoWorker {
    seen_input_ids: Vec<u32>,
}

impl ModelWorker for RouterEchoWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        let request = &batch.requests()[0];
        self.seen_input_ids = request.input_ids().to_vec();
        BatchGeneratedTokens::from_batch(batch, vec![GeneratedToken::finished(vec![42, 43])])
            .expect("output shape should match batch")
    }
}

#[test]
fn router_generate_request_maps_to_tokenized_engine_request() {
    let request = RouterGenerateRequest {
        request_id: "router-rid".to_string(),
        tokenized: Some(RouterTokenizedInput {
            original_text: "hello".to_string(),
            input_ids: vec![101, 202, 303],
        }),
        sampling_params: Some(RouterSamplingParams {
            max_new_tokens: Some(7),
        }),
        stream: true,
        data_parallel_rank: 2,
        trace_headers: [("traceparent".to_string(), "00-abc".to_string())].into(),
    };

    let token_request = request
        .try_into_token_generate_request()
        .expect("router request should map");

    assert_eq!(token_request.request_id, RequestId::from("router-rid"));
    assert_eq!(token_request.input_ids, vec![101, 202, 303]);
    assert_eq!(token_request.sampling, SamplingParams { max_new_tokens: 7 });
}

#[test]
fn router_sampling_params_default_to_sglang_max_new_tokens() {
    let request = RouterGenerateRequest {
        request_id: "default-sampling".to_string(),
        tokenized: Some(RouterTokenizedInput {
            original_text: String::new(),
            input_ids: vec![1],
        }),
        sampling_params: None,
        stream: false,
        data_parallel_rank: 0,
        trace_headers: Default::default(),
    };

    let token_request = request
        .try_into_token_generate_request()
        .expect("router request should map");

    assert_eq!(
        token_request.sampling,
        SamplingParams {
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS
        }
    );
}

#[test]
fn router_generate_request_rejects_missing_tokenized_input() {
    let request = RouterGenerateRequest {
        request_id: "missing-tokenized".to_string(),
        tokenized: None,
        sampling_params: None,
        stream: false,
        data_parallel_rank: 0,
        trace_headers: Default::default(),
    };

    let error = request
        .try_into_token_generate_request()
        .expect_err("missing tokenized input should be rejected");

    assert_eq!(error, RouterProtocolError::MissingTokenizedInput);
}

#[test]
fn tokenized_engine_output_maps_to_router_generate_stream_chunk() {
    let output = TokenGenerateOutput {
        request_id: RequestId::from("router-rid"),
        output_ids: vec![7, 8, 9],
        finished: false,
    };

    let response = RouterGenerateResponse::from_token_generate_output(output, 5);

    assert_eq!(response.request_id, "router-rid");
    assert_eq!(
        response.body,
        RouterGenerateResponseBody::Chunk(RouterGenerateStreamChunk {
            token_ids: vec![7, 8, 9],
            prompt_tokens: 5,
            completion_tokens: 3,
            cached_tokens: 0,
            index: 0,
        })
    );
}

#[test]
fn tokenized_engine_finished_output_maps_to_router_generate_complete() {
    let output = TokenGenerateOutput {
        request_id: RequestId::from("router-rid"),
        output_ids: vec![7, 8, 9],
        finished: true,
    };

    let response = RouterGenerateResponse::from_token_generate_output(output, 5);

    assert_eq!(
        response.body,
        RouterGenerateResponseBody::Complete(RouterGenerateComplete {
            output_ids: vec![7, 8, 9],
            finish_reason: "stop".to_string(),
            prompt_tokens: 5,
            completion_tokens: 3,
            cached_tokens: 0,
            index: 0,
        })
    );
}

#[test]
fn router_health_check_reports_ready_worker() {
    let response = RouterHealthCheckResponse::healthy();

    assert!(response.healthy);
    assert_eq!(response.message, "ready");
}

#[test]
fn router_model_info_uses_sglang_server_args_for_worker_registration() {
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

    let response = RouterGetModelInfoResponse::from_server_args(&args);

    assert_eq!(response.model_path, "meta-llama/Llama-3.1-8B-Instruct");
    assert_eq!(response.tokenizer_path, "hf-tokenizer");
    assert!(response.is_generation);
    assert_eq!(response.served_model_name, "llama3");
    assert_eq!(response.preferred_sampling_params, "{}");
}

#[test]
fn router_runtime_executes_generate_request_through_engine() {
    let tokenizer = ByteTokenizer::default();
    let scheduler = Scheduler::new(RouterEchoWorker::default());
    let engine = Engine::new(tokenizer, scheduler);
    let mut runtime = RouterRuntime::new(engine);

    let response = runtime
        .generate(RouterGenerateRequest {
            request_id: "router-exec".to_string(),
            tokenized: Some(RouterTokenizedInput {
                original_text: "ignored because router already tokenized".to_string(),
                input_ids: vec![9, 8, 7],
            }),
            sampling_params: Some(RouterSamplingParams {
                max_new_tokens: Some(2),
            }),
            stream: true,
            data_parallel_rank: 0,
            trace_headers: Default::default(),
        })
        .expect("router request should execute");

    assert_eq!(response.request_id, "router-exec");
    assert_eq!(
        response.body,
        RouterGenerateResponseBody::Complete(RouterGenerateComplete {
            output_ids: vec![42, 43],
            finish_reason: "stop".to_string(),
            prompt_tokens: 3,
            completion_tokens: 2,
            cached_tokens: 0,
            index: 0,
        })
    );
    assert_eq!(
        runtime.engine().scheduler().worker().seen_input_ids,
        vec![9, 8, 7]
    );
}
