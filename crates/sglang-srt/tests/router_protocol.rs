use sglang_srt::cache::{
    CachePageAllocator, KvBlockPrefixIndex, KvCacheWorkerId, KvCacheWorkerSnapshot, RadixCache,
    compute_sglang_block_hashes,
};
use sglang_srt::cli::ServerArgs;
use sglang_srt::engine::Engine;
use sglang_srt::router::{
    DEFAULT_MAX_NEW_TOKENS, RouterDisaggregatedParams, RouterFlushCacheResponse,
    RouterGenerateComplete, RouterGenerateRequest, RouterGenerateResponse,
    RouterGenerateResponseBody, RouterGenerateStreamChunk, RouterGetModelInfoResponse,
    RouterHealthCheckResponse, RouterProtocolError, RouterRuntime, RouterSamplingParams,
    RouterStatusCode, RouterTextGenerateRequest, RouterTokenizedInput, RouterValidationConfig,
};
use sglang_srt::scheduler::{ScheduleBatch, ScheduledRequest, Scheduler};
use sglang_srt::tokenizer::ByteTokenizer;
use sglang_srt::types::{RequestId, SamplingParams, TokenGenerateOutput};
use sglang_srt::worker::{
    BatchGeneratedTokens, DecodeRequestState, FallibleModelWorker, GeneratedToken, ModelWorker,
    WorkerExecutionError, WorkerWeightUpdateRequest,
};

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

#[derive(Default)]
struct TwoStepRouterWorker {
    seen_modes: Vec<sglang_srt::scheduler::ForwardMode>,
}

#[derive(Default)]
struct AlwaysUnfinishedRouterWorker;

impl ModelWorker for AlwaysUnfinishedRouterWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        BatchGeneratedTokens::from_batch(batch, vec![GeneratedToken::unfinished(vec![42])])
            .expect("output shape should match batch")
    }
}

impl ModelWorker for TwoStepRouterWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        self.seen_modes.push(batch.forward_mode());
        let token = match batch.forward_mode() {
            sglang_srt::scheduler::ForwardMode::Prefill => GeneratedToken::unfinished(vec![42]),
            sglang_srt::scheduler::ForwardMode::Decode => GeneratedToken::finished(vec![43]),
        };

        BatchGeneratedTokens::from_batch(batch, vec![token])
            .expect("output shape should match batch")
    }
}

#[derive(Default)]
struct RecordingWeightUpdateWorker {
    updates: Vec<WorkerWeightUpdateRequest>,
}

impl FallibleModelWorker for RecordingWeightUpdateWorker {
    fn try_generate_batch(
        &mut self,
        batch: &ScheduleBatch,
    ) -> Result<BatchGeneratedTokens, WorkerExecutionError> {
        Ok(
            BatchGeneratedTokens::from_batch(batch, vec![GeneratedToken::finished(vec![42])])
                .expect("output shape should match batch"),
        )
    }

    fn decode_request_state(
        &self,
        _request: &ScheduledRequest,
    ) -> Result<DecodeRequestState, WorkerExecutionError> {
        Ok(DecodeRequestState::Ready)
    }

    fn update_weights_from_disk(
        &mut self,
        request: &WorkerWeightUpdateRequest,
    ) -> Result<(), WorkerExecutionError> {
        self.updates.push(request.clone());
        Ok(())
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
            ..Default::default()
        }),
        disaggregated_params: None,
        stream: true,
        data_parallel_rank: 2,
        trace_headers: [("traceparent".to_string(), "00-abc".to_string())].into(),
    };

    let token_request = request
        .try_into_token_generate_request()
        .expect("router request should map");

    assert_eq!(token_request.request_id, RequestId::from("router-rid"));
    assert_eq!(token_request.input_ids, vec![101, 202, 303]);
    assert_eq!(token_request.sampling, SamplingParams::new(7));
    assert_eq!(token_request.data_parallel_rank, 2);
    assert!(token_request.disaggregated_params.is_none());
}

#[test]
fn router_generate_request_generates_request_id_when_missing() {
    let first = RouterGenerateRequest {
        request_id: String::new(),
        tokenized: Some(RouterTokenizedInput {
            original_text: String::new(),
            input_ids: vec![101],
        }),
        sampling_params: Some(RouterSamplingParams {
            max_new_tokens: Some(1),
            ..Default::default()
        }),
        disaggregated_params: None,
        stream: false,
        data_parallel_rank: 0,
        trace_headers: Default::default(),
    }
    .try_into_token_generate_request()
    .expect("router should generate a request id");

    let second = RouterGenerateRequest {
        request_id: String::new(),
        tokenized: Some(RouterTokenizedInput {
            original_text: String::new(),
            input_ids: vec![202],
        }),
        sampling_params: Some(RouterSamplingParams {
            max_new_tokens: Some(1),
            ..Default::default()
        }),
        disaggregated_params: None,
        stream: false,
        data_parallel_rank: 0,
        trace_headers: Default::default(),
    }
    .try_into_token_generate_request()
    .expect("router should generate a request id");

    assert!(first.request_id.as_str().starts_with("sglang-rs-"));
    assert!(second.request_id.as_str().starts_with("sglang-rs-"));
    assert_ne!(first.request_id, second.request_id);
}

#[test]
fn router_generate_request_preserves_disaggregated_params_for_pd() {
    let request = RouterGenerateRequest {
        request_id: "router-pd".to_string(),
        tokenized: Some(RouterTokenizedInput {
            original_text: String::new(),
            input_ids: vec![101, 202],
        }),
        sampling_params: None,
        disaggregated_params: Some(RouterDisaggregatedParams {
            bootstrap_host: "10.0.0.7".to_string(),
            bootstrap_port: 8998,
            bootstrap_room: 123,
        }),
        stream: true,
        data_parallel_rank: 0,
        trace_headers: Default::default(),
    };

    let token_request = request
        .try_into_token_generate_request()
        .expect("router request should map");

    assert_eq!(
        token_request.disaggregated_params,
        Some(sglang_srt::types::DisaggregatedParams {
            bootstrap_host: "10.0.0.7".to_string(),
            bootstrap_port: 8998,
            bootstrap_room: 123,
        })
    );
}

#[test]
fn router_generate_request_accepts_sgl_router_63_bit_bootstrap_room() {
    let bootstrap_room = i64::MAX as u64;
    let request = RouterGenerateRequest {
        request_id: "router-pd-wide-room".to_string(),
        tokenized: Some(RouterTokenizedInput {
            original_text: String::new(),
            input_ids: vec![101, 202],
        }),
        sampling_params: None,
        disaggregated_params: Some(RouterDisaggregatedParams {
            bootstrap_host: "10.0.0.7".to_string(),
            bootstrap_port: 8998,
            bootstrap_room,
        }),
        stream: true,
        data_parallel_rank: 0,
        trace_headers: Default::default(),
    };

    let token_request = request
        .try_into_token_generate_request()
        .expect("63-bit sgl-router bootstrap_room should map");

    assert_eq!(
        token_request
            .disaggregated_params
            .expect("disaggregated params")
            .bootstrap_room,
        bootstrap_room
    );
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
        disaggregated_params: None,
        stream: false,
        data_parallel_rank: 0,
        trace_headers: Default::default(),
    };

    let token_request = request
        .try_into_token_generate_request()
        .expect("router request should map");

    assert_eq!(
        token_request.sampling,
        SamplingParams::new(DEFAULT_MAX_NEW_TOKENS)
    );
}

#[test]
fn router_generate_request_selects_cache_aware_prefill_worker_from_tokens() {
    let mut index = KvBlockPrefixIndex::default();
    let worker_a = kv_worker("http://prefill-a:30000", 0);
    let worker_b = kv_worker("http://prefill-b:30000", 0);
    let input_ids = vec![101, 202, 303, 404];
    let block_hashes = compute_sglang_block_hashes(&input_ids, 2);
    index.insert(&worker_b, &block_hashes);
    let request = RouterGenerateRequest {
        request_id: "router-cache-aware".to_string(),
        tokenized: Some(RouterTokenizedInput {
            original_text: "hello".to_string(),
            input_ids,
        }),
        sampling_params: None,
        disaggregated_params: None,
        stream: false,
        data_parallel_rank: 0,
        trace_headers: Default::default(),
    };

    let selected = request
        .select_cache_aware_prefill_worker(
            &index,
            &kv_workers_with_loads([(&worker_a, 0), (&worker_b, 9)]),
            2,
            0.5,
        )
        .expect("router request should be valid")
        .expect("selector should choose a worker");

    assert_eq!(selected, worker_b);
}

#[test]
fn router_generate_request_cache_aware_selection_rejects_missing_tokenized_input() {
    let request = RouterGenerateRequest {
        request_id: "router-cache-aware-missing".to_string(),
        tokenized: None,
        sampling_params: None,
        disaggregated_params: None,
        stream: false,
        data_parallel_rank: 0,
        trace_headers: Default::default(),
    };

    let error = request
        .select_cache_aware_prefill_worker(&KvBlockPrefixIndex::default(), &[], 2, 0.5)
        .expect_err("missing tokenized input should be rejected");

    assert_eq!(error, RouterProtocolError::MissingTokenizedInput);
}

#[test]
fn router_generate_request_rejects_missing_tokenized_input() {
    let request = RouterGenerateRequest {
        request_id: "missing-tokenized".to_string(),
        tokenized: None,
        sampling_params: None,
        disaggregated_params: None,
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
fn router_generate_request_rejects_empty_tokenized_input() {
    let request = RouterGenerateRequest {
        request_id: "empty-tokenized".to_string(),
        tokenized: Some(RouterTokenizedInput {
            original_text: String::new(),
            input_ids: Vec::new(),
        }),
        sampling_params: None,
        disaggregated_params: None,
        stream: false,
        data_parallel_rank: 0,
        trace_headers: Default::default(),
    };

    let error = request
        .try_into_token_generate_request()
        .expect_err("empty tokenized input should be rejected");

    assert_eq!(error, RouterProtocolError::EmptyTokenizedInput);
    assert_eq!(error.status_code(), RouterStatusCode::InvalidArgument);
}

#[test]
fn router_generate_request_rejects_non_positive_max_new_tokens() {
    let request = RouterGenerateRequest {
        request_id: "bad-sampling".to_string(),
        tokenized: Some(RouterTokenizedInput {
            original_text: String::new(),
            input_ids: vec![1],
        }),
        sampling_params: Some(RouterSamplingParams {
            max_new_tokens: Some(0),
            ..Default::default()
        }),
        disaggregated_params: None,
        stream: false,
        data_parallel_rank: 0,
        trace_headers: Default::default(),
    };

    let error = request
        .try_into_token_generate_request()
        .expect_err("zero max_new_tokens should be rejected");

    assert_eq!(
        error,
        RouterProtocolError::InvalidIntegerSamplingParam {
            field: "max_new_tokens",
            value: 0,
            expected: "positive",
        }
    );
    assert_eq!(error.status_code(), RouterStatusCode::InvalidArgument);
}

#[test]
fn router_generate_request_accepts_valid_extended_sampling_params() {
    let request = RouterGenerateRequest {
        request_id: "extended-sampling".to_string(),
        tokenized: Some(RouterTokenizedInput {
            original_text: String::new(),
            input_ids: vec![1],
        }),
        sampling_params: Some(RouterSamplingParams {
            max_new_tokens: Some(8),
            temperature: Some(0.7),
            top_p: Some(0.95),
            top_k: Some(40),
            min_p: Some(0.0),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
            repetition_penalty: Some(1.0),
            stop_token_id: None,
            stop_token_ids: Vec::new(),
            ignore_eos: None,
            n: Some(1),
            best_of: Some(1),
        }),
        disaggregated_params: None,
        stream: false,
        data_parallel_rank: 0,
        trace_headers: Default::default(),
    };

    let token_request = request
        .try_into_token_generate_request()
        .expect("valid extended sampling params should map");

    assert_eq!(token_request.sampling.max_new_tokens, 8);
    assert_eq!(token_request.sampling.temperature, Some(0.7));
    assert_eq!(token_request.sampling.top_p, Some(0.95));
    assert_eq!(token_request.sampling.top_k, Some(40));
    assert_eq!(token_request.sampling.min_p, Some(0.0));
}

#[test]
fn router_generate_request_preserves_stop_token_id_sampling_param() {
    let request = RouterGenerateRequest {
        request_id: "stop-token".to_string(),
        tokenized: Some(RouterTokenizedInput {
            original_text: String::new(),
            input_ids: vec![1],
        }),
        sampling_params: Some(RouterSamplingParams {
            max_new_tokens: Some(8),
            stop_token_id: Some(2),
            ..Default::default()
        }),
        disaggregated_params: None,
        stream: false,
        data_parallel_rank: 0,
        trace_headers: Default::default(),
    };

    let token_request = request
        .try_into_token_generate_request()
        .expect("stop_token_id should map into runtime sampling params");

    assert_eq!(token_request.sampling.stop_token_ids, vec![2]);
}

#[test]
fn router_generate_request_preserves_stop_token_ids_sampling_param() {
    let request = RouterGenerateRequest {
        request_id: "stop-token-list".to_string(),
        tokenized: Some(RouterTokenizedInput {
            original_text: String::new(),
            input_ids: vec![1],
        }),
        sampling_params: Some(RouterSamplingParams {
            max_new_tokens: Some(8),
            stop_token_ids: vec![2, 3],
            ..Default::default()
        }),
        disaggregated_params: None,
        stream: false,
        data_parallel_rank: 0,
        trace_headers: Default::default(),
    };

    let token_request = request
        .try_into_token_generate_request()
        .expect("stop_token_ids should map into runtime sampling params");

    assert_eq!(token_request.sampling.stop_token_ids, vec![2, 3]);
}

#[test]
fn router_generate_request_accepts_top_k_minus_one_as_unbounded_sampling() {
    let request = RouterGenerateRequest {
        request_id: "top-k-disabled".to_string(),
        tokenized: Some(RouterTokenizedInput {
            original_text: String::new(),
            input_ids: vec![1],
        }),
        sampling_params: Some(RouterSamplingParams {
            max_new_tokens: Some(8),
            top_k: Some(-1),
            ..Default::default()
        }),
        disaggregated_params: None,
        stream: false,
        data_parallel_rank: 0,
        trace_headers: Default::default(),
    };

    let token_request = request
        .try_into_token_generate_request()
        .expect("top_k=-1 should disable top-k filtering like upstream SGLang");

    assert_eq!(token_request.sampling.top_k, Some(-1));
}

#[test]
fn router_generate_request_rejects_invalid_float_sampling_params() {
    let request = RouterGenerateRequest {
        request_id: "bad-top-p".to_string(),
        tokenized: Some(RouterTokenizedInput {
            original_text: String::new(),
            input_ids: vec![1],
        }),
        sampling_params: Some(RouterSamplingParams {
            top_p: Some(1.1),
            ..Default::default()
        }),
        disaggregated_params: None,
        stream: false,
        data_parallel_rank: 0,
        trace_headers: Default::default(),
    };

    let error = request
        .try_into_token_generate_request()
        .expect_err("top_p outside [0, 1] should be rejected");

    assert_eq!(
        error,
        RouterProtocolError::InvalidFloatSamplingParam {
            field: "top_p",
            value: 1.1,
            expected: "finite and in (0, 1]",
        }
    );
    assert_eq!(error.status_code(), RouterStatusCode::InvalidArgument);
}

#[test]
fn router_generate_request_rejects_zero_top_p() {
    let request = RouterGenerateRequest {
        request_id: "zero-top-p".to_string(),
        tokenized: Some(RouterTokenizedInput {
            original_text: String::new(),
            input_ids: vec![1],
        }),
        sampling_params: Some(RouterSamplingParams {
            top_p: Some(0.0),
            ..Default::default()
        }),
        disaggregated_params: None,
        stream: false,
        data_parallel_rank: 0,
        trace_headers: Default::default(),
    };

    let error = request
        .try_into_token_generate_request()
        .expect_err("top_p=0 should be rejected");

    assert_eq!(
        error,
        RouterProtocolError::InvalidFloatSamplingParam {
            field: "top_p",
            value: 0.0,
            expected: "finite and in (0, 1]",
        }
    );
    assert_eq!(error.status_code(), RouterStatusCode::InvalidArgument);
}

#[test]
fn router_generate_request_rejects_invalid_integer_sampling_params() {
    let request = RouterGenerateRequest {
        request_id: "bad-n".to_string(),
        tokenized: Some(RouterTokenizedInput {
            original_text: String::new(),
            input_ids: vec![1],
        }),
        sampling_params: Some(RouterSamplingParams {
            n: Some(0),
            ..Default::default()
        }),
        disaggregated_params: None,
        stream: false,
        data_parallel_rank: 0,
        trace_headers: Default::default(),
    };

    let error = request
        .try_into_token_generate_request()
        .expect_err("n must be positive");

    assert_eq!(
        error,
        RouterProtocolError::InvalidIntegerSamplingParam {
            field: "n",
            value: 0,
            expected: "positive",
        }
    );
    assert_eq!(error.status_code(), RouterStatusCode::InvalidArgument);
}

#[test]
fn router_generate_request_rejects_input_over_model_request_limit() {
    let request = RouterGenerateRequest {
        request_id: "too-long".to_string(),
        tokenized: Some(RouterTokenizedInput {
            original_text: String::new(),
            input_ids: vec![1, 2, 3],
        }),
        sampling_params: Some(RouterSamplingParams {
            max_new_tokens: Some(1),
            ..Default::default()
        }),
        disaggregated_params: None,
        stream: false,
        data_parallel_rank: 0,
        trace_headers: Default::default(),
    };

    let error = request
        .try_into_token_generate_request_with_config(RouterValidationConfig {
            max_context_tokens: None,
            max_request_input_tokens: Some(2),
        })
        .expect_err("input over model request limit should be rejected");

    assert_eq!(
        error,
        RouterProtocolError::InputTooLong {
            input_tokens: 3,
            max_request_input_tokens: 2,
        }
    );
    assert_eq!(error.status_code(), RouterStatusCode::ResourceExhausted);
}

#[test]
fn router_generate_request_rejects_context_overflow() {
    let request = RouterGenerateRequest {
        request_id: "context-overflow".to_string(),
        tokenized: Some(RouterTokenizedInput {
            original_text: String::new(),
            input_ids: vec![1, 2, 3],
        }),
        sampling_params: Some(RouterSamplingParams {
            max_new_tokens: Some(4),
            ..Default::default()
        }),
        disaggregated_params: None,
        stream: false,
        data_parallel_rank: 0,
        trace_headers: Default::default(),
    };

    let error = request
        .try_into_token_generate_request_with_config(RouterValidationConfig {
            max_context_tokens: Some(6),
            max_request_input_tokens: None,
        })
        .expect_err("input plus output budget over context should be rejected");

    assert_eq!(
        error,
        RouterProtocolError::ContextOverflow {
            input_tokens: 3,
            max_new_tokens: 4,
            max_context_tokens: 6,
        }
    );
    assert_eq!(error.status_code(), RouterStatusCode::ResourceExhausted);
}

#[test]
fn tokenized_engine_output_maps_to_router_generate_stream_chunk() {
    let output = TokenGenerateOutput {
        request_id: RequestId::from("router-rid"),
        output_ids: vec![7, 8, 9],
        cached_tokens: 2,
        finished: false,
    };

    let response = RouterGenerateResponse::from_token_generate_output(output, 5);

    assert_eq!(response.request_id, "router-rid");
    assert_eq!(
        response.body,
        RouterGenerateResponseBody::Chunk(RouterGenerateStreamChunk {
            token_ids: vec![7, 8, 9],
            text: String::new(),
            prompt_tokens: 5,
            completion_tokens: 3,
            cached_tokens: 2,
            index: 0,
        })
    );
}

#[test]
fn tokenized_engine_finished_output_maps_to_router_generate_complete() {
    let output = TokenGenerateOutput {
        request_id: RequestId::from("router-rid"),
        output_ids: vec![7, 8, 9],
        cached_tokens: 2,
        finished: true,
    };

    let response = RouterGenerateResponse::from_token_generate_output(output, 5);

    assert_eq!(
        response.body,
        RouterGenerateResponseBody::Complete(RouterGenerateComplete {
            output_ids: vec![7, 8, 9],
            text: String::new(),
            finish_reason: "stop".to_string(),
            prompt_tokens: 5,
            completion_tokens: 3,
            cached_tokens: 2,
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
    assert_eq!(response.routed_expert_expected_group_count, 0);
    assert_eq!(response.routed_expert_actual_group_count, 0);
    assert_eq!(response.routed_expert_expected_weight_count, 0);
    assert_eq!(response.routed_expert_actual_weight_count, 0);
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
                ..Default::default()
            }),
            disaggregated_params: None,
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
            text: String::new(),
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

#[test]
fn router_runtime_reports_prefix_cache_hits_as_cached_tokens() {
    let tokenizer = ByteTokenizer::default();
    let scheduler = Scheduler::with_cache_resources(
        RouterEchoWorker::default(),
        RadixCache::default(),
        CachePageAllocator::new(8),
    );
    let engine = Engine::new(tokenizer, scheduler);
    let mut runtime = RouterRuntime::new(engine);

    let first_request = RouterGenerateRequest {
        request_id: "cache-warmup".to_string(),
        tokenized: Some(RouterTokenizedInput {
            original_text: String::new(),
            input_ids: vec![9, 8, 7],
        }),
        sampling_params: Some(RouterSamplingParams {
            max_new_tokens: Some(2),
            ..Default::default()
        }),
        disaggregated_params: None,
        stream: false,
        data_parallel_rank: 0,
        trace_headers: Default::default(),
    };
    runtime
        .generate(first_request)
        .expect("warmup request should populate prefix cache");

    let response = runtime
        .generate(RouterGenerateRequest {
            request_id: "cache-hit".to_string(),
            tokenized: Some(RouterTokenizedInput {
                original_text: String::new(),
                input_ids: vec![9, 8, 7],
            }),
            sampling_params: Some(RouterSamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            disaggregated_params: None,
            stream: false,
            data_parallel_rank: 0,
            trace_headers: Default::default(),
        })
        .expect("cache hit request should execute");

    assert_eq!(
        response.body,
        RouterGenerateResponseBody::Complete(RouterGenerateComplete {
            output_ids: vec![42, 43],
            text: String::new(),
            finish_reason: "stop".to_string(),
            prompt_tokens: 3,
            completion_tokens: 2,
            cached_tokens: 3,
            index: 0,
        })
    );
}

#[test]
fn router_runtime_rejects_invalid_request_before_engine_dispatch() {
    let tokenizer = ByteTokenizer::default();
    let scheduler = Scheduler::new(RouterEchoWorker::default());
    let engine = Engine::new(tokenizer, scheduler);
    let mut runtime = RouterRuntime::with_validation_config(
        engine,
        RouterValidationConfig {
            max_context_tokens: Some(4),
            max_request_input_tokens: None,
        },
    );

    let error = runtime
        .generate(RouterGenerateRequest {
            request_id: "runtime-reject".to_string(),
            tokenized: Some(RouterTokenizedInput {
                original_text: String::new(),
                input_ids: vec![1, 2, 3],
            }),
            sampling_params: Some(RouterSamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            disaggregated_params: None,
            stream: false,
            data_parallel_rank: 0,
            trace_headers: Default::default(),
        })
        .expect_err("context overflow should be rejected by router runtime");

    assert!(matches!(
        error,
        sglang_srt::router::RouterRuntimeError::Protocol(RouterProtocolError::ContextOverflow {
            input_tokens: 3,
            max_new_tokens: 2,
            max_context_tokens: 4,
        })
    ));
    assert!(
        runtime
            .engine()
            .scheduler()
            .worker()
            .seen_input_ids
            .is_empty()
    );
}

#[test]
fn router_runtime_pause_generation_rejects_valid_requests_before_dispatch() {
    let tokenizer = ByteTokenizer::default();
    let scheduler = Scheduler::new(RouterEchoWorker::default());
    let engine = Engine::new(tokenizer, scheduler);
    let mut runtime = RouterRuntime::new(engine);

    let pause_response = runtime.pause_generation();

    assert!(pause_response.success);
    assert_eq!(pause_response.message, "generation paused");

    let error = runtime
        .generate(RouterGenerateRequest {
            request_id: "paused".to_string(),
            tokenized: Some(RouterTokenizedInput {
                original_text: String::new(),
                input_ids: vec![1, 2, 3],
            }),
            sampling_params: Some(RouterSamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            disaggregated_params: None,
            stream: false,
            data_parallel_rank: 0,
            trace_headers: Default::default(),
        })
        .expect_err("paused runtime should reject generation before dispatch");

    assert!(matches!(
        error,
        sglang_srt::router::RouterRuntimeError::Protocol(RouterProtocolError::GenerationPaused)
    ));
    assert!(
        runtime
            .engine()
            .scheduler()
            .worker()
            .seen_input_ids
            .is_empty()
    );

    let continue_response = runtime.continue_generation();

    assert!(continue_response.success);
    assert_eq!(continue_response.message, "generation continued");

    let response = runtime
        .generate(RouterGenerateRequest {
            request_id: "continued".to_string(),
            tokenized: Some(RouterTokenizedInput {
                original_text: String::new(),
                input_ids: vec![9, 8, 7],
            }),
            sampling_params: Some(RouterSamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            disaggregated_params: None,
            stream: false,
            data_parallel_rank: 0,
            trace_headers: Default::default(),
        })
        .expect("continued runtime should dispatch generation");

    assert_eq!(response.request_id, "continued");
    assert_eq!(
        runtime.engine().scheduler().worker().seen_input_ids,
        vec![9, 8, 7]
    );
}

#[test]
fn router_runtime_abort_request_removes_queued_request() {
    let tokenizer = ByteTokenizer::default();
    let mut scheduler = Scheduler::new(RouterEchoWorker::default());
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("abort-me"),
        vec![1, 2, 3],
        SamplingParams::new(1),
    ));
    let engine = Engine::new(tokenizer, scheduler);
    let mut runtime = RouterRuntime::new(engine);

    let response = runtime
        .abort_request("abort-me")
        .expect("abort request id should be valid");

    assert!(response.success);
    assert_eq!(response.message, "request aborted");
    assert_eq!(runtime.load().waiting_queue_depth, 0);

    let missing = runtime
        .abort_request("missing")
        .expect("missing request id should still be valid");

    assert!(!missing.success);
    assert_eq!(missing.message, "request not found");

    let error = runtime
        .abort_request("")
        .expect_err("empty request id should be rejected");

    assert_eq!(error, RouterProtocolError::MissingRequestId);
    assert_eq!(error.status_code(), RouterStatusCode::InvalidArgument);
}

#[test]
fn router_runtime_reports_running_request_limit_as_resource_exhausted_without_queuing_request() {
    let tokenizer = ByteTokenizer::default();
    let mut scheduler =
        Scheduler::new(AlwaysUnfinishedRouterWorker).with_max_running_requests(Some(1));
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("active"),
        vec![1],
        SamplingParams::new(2),
    ));
    scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should occupy the active slot");
    let engine = Engine::new(tokenizer, scheduler);
    let mut runtime = RouterRuntime::new(engine);

    let error = runtime
        .generate_stream(RouterGenerateRequest {
            request_id: "over-capacity".to_string(),
            tokenized: Some(RouterTokenizedInput {
                original_text: String::new(),
                input_ids: vec![9],
            }),
            sampling_params: Some(RouterSamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            disaggregated_params: None,
            stream: true,
            data_parallel_rank: 0,
            trace_headers: Default::default(),
        })
        .expect_err("capacity backpressure should reject the request");

    assert!(matches!(
        error,
        sglang_srt::router::RouterRuntimeError::Protocol(
            RouterProtocolError::RunningRequestLimitReached {
                max_running_requests: 1
            }
        )
    ));
    assert_eq!(
        RouterProtocolError::RunningRequestLimitReached {
            max_running_requests: 1
        }
        .status_code(),
        RouterStatusCode::ResourceExhausted
    );
    assert_eq!(runtime.load().waiting_queue_depth, 0);
    assert_eq!(runtime.load().decode_queue_depth, 1);
}

#[test]
fn router_runtime_streams_prefill_chunks_and_final_complete_response() {
    let tokenizer = ByteTokenizer::default();
    let scheduler = Scheduler::new(TwoStepRouterWorker::default());
    let engine = Engine::new(tokenizer, scheduler);
    let mut runtime = RouterRuntime::new(engine);

    let responses = runtime
        .generate_stream(RouterGenerateRequest {
            request_id: "router-stream".to_string(),
            tokenized: Some(RouterTokenizedInput {
                original_text: String::new(),
                input_ids: vec![1, 2, 3],
            }),
            sampling_params: Some(RouterSamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            disaggregated_params: None,
            stream: true,
            data_parallel_rank: 0,
            trace_headers: Default::default(),
        })
        .expect("router stream should execute");

    assert_eq!(
        responses,
        vec![
            RouterGenerateResponse {
                request_id: "router-stream".to_string(),
                body: RouterGenerateResponseBody::Chunk(RouterGenerateStreamChunk {
                    token_ids: vec![42],
                    text: String::new(),
                    prompt_tokens: 3,
                    completion_tokens: 1,
                    cached_tokens: 0,
                    index: 0,
                }),
            },
            RouterGenerateResponse {
                request_id: "router-stream".to_string(),
                body: RouterGenerateResponseBody::Complete(RouterGenerateComplete {
                    output_ids: vec![42, 43],
                    text: String::new(),
                    finish_reason: "stop".to_string(),
                    prompt_tokens: 3,
                    completion_tokens: 2,
                    cached_tokens: 0,
                    index: 0,
                }),
            },
        ]
    );
    assert_eq!(
        runtime.engine().scheduler().worker().seen_modes,
        vec![
            sglang_srt::scheduler::ForwardMode::Prefill,
            sglang_srt::scheduler::ForwardMode::Decode,
        ]
    );
}

#[test]
fn router_runtime_non_stream_generate_returns_only_final_complete_response() {
    let tokenizer = ByteTokenizer::default();
    let scheduler = Scheduler::new(TwoStepRouterWorker::default());
    let engine = Engine::new(tokenizer, scheduler);
    let mut runtime = RouterRuntime::new(engine);

    let responses = runtime
        .generate_stream(RouterGenerateRequest {
            request_id: "router-non-stream".to_string(),
            tokenized: Some(RouterTokenizedInput {
                original_text: String::new(),
                input_ids: vec![1, 2, 3],
            }),
            sampling_params: Some(RouterSamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            disaggregated_params: None,
            stream: false,
            data_parallel_rank: 0,
            trace_headers: Default::default(),
        })
        .expect("router non-stream request should execute");

    assert_eq!(
        responses,
        vec![RouterGenerateResponse {
            request_id: "router-non-stream".to_string(),
            body: RouterGenerateResponseBody::Complete(RouterGenerateComplete {
                output_ids: vec![42, 43],
                text: String::new(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 3,
                completion_tokens: 2,
                cached_tokens: 0,
                index: 0,
            }),
        }]
    );
    assert_eq!(
        runtime.engine().scheduler().worker().seen_modes,
        vec![
            sglang_srt::scheduler::ForwardMode::Prefill,
            sglang_srt::scheduler::ForwardMode::Decode,
        ]
    );
}

#[test]
fn router_runtime_text_generate_tokenizes_prompt_and_decodes_output_text() {
    let tokenizer = ByteTokenizer::default();
    let scheduler = Scheduler::new(RouterEchoWorker::default());
    let engine = Engine::new(tokenizer, scheduler);
    let mut runtime = RouterRuntime::new(engine);

    let responses = runtime
        .generate_text_stream(RouterTextGenerateRequest {
            request_id: "router-text".to_string(),
            text: "abc".to_string(),
            sampling_params: Some(RouterSamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            disaggregated_params: None,
            stream: true,
            data_parallel_rank: 0,
            trace_headers: Default::default(),
        })
        .expect("text request should execute");

    assert_eq!(
        responses,
        vec![RouterGenerateResponse {
            request_id: "router-text".to_string(),
            body: RouterGenerateResponseBody::Complete(RouterGenerateComplete {
                output_ids: vec![42, 43],
                text: "*+".to_string(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 3,
                completion_tokens: 2,
                cached_tokens: 0,
                index: 0,
            }),
        }]
    );
    assert_eq!(
        runtime.engine().scheduler().worker().seen_input_ids,
        vec![97, 98, 99]
    );
}

#[test]
fn router_runtime_text_generate_rejects_empty_text_before_dispatch() {
    let tokenizer = ByteTokenizer::default();
    let scheduler = Scheduler::new(RouterEchoWorker::default());
    let engine = Engine::new(tokenizer, scheduler);
    let mut runtime = RouterRuntime::new(engine);

    let error = runtime
        .generate_text_stream(RouterTextGenerateRequest {
            request_id: "router-empty-text".to_string(),
            text: String::new(),
            sampling_params: None,
            disaggregated_params: None,
            stream: true,
            data_parallel_rank: 0,
            trace_headers: Default::default(),
        })
        .expect_err("empty text should be rejected");

    assert!(matches!(
        error,
        sglang_srt::router::RouterRuntimeError::Protocol(RouterProtocolError::EmptyTextInput)
    ));
    assert!(
        runtime
            .engine()
            .scheduler()
            .worker()
            .seen_input_ids
            .is_empty()
    );
}

#[test]
fn router_runtime_flush_cache_calls_scheduler_and_reports_success() {
    let tokenizer = ByteTokenizer::default();
    let scheduler = Scheduler::with_cache_resources(
        RouterEchoWorker::default(),
        RadixCache::default(),
        CachePageAllocator::new(4),
    );
    let engine = Engine::new(tokenizer, scheduler);
    let mut runtime = RouterRuntime::new(engine);

    runtime
        .generate(RouterGenerateRequest {
            request_id: "flush-seed".to_string(),
            tokenized: Some(RouterTokenizedInput {
                original_text: String::new(),
                input_ids: vec![1, 2],
            }),
            sampling_params: Some(RouterSamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            disaggregated_params: None,
            stream: false,
            data_parallel_rank: 0,
            trace_headers: Default::default(),
        })
        .expect("seed request should execute");

    assert_eq!(
        runtime.engine().scheduler().available_cache_pages(),
        Some(2)
    );

    let response = runtime.flush_cache();

    assert_eq!(
        response,
        RouterFlushCacheResponse {
            success: true,
            message: "cache flushed".to_string(),
        }
    );
    assert_eq!(
        runtime.engine().scheduler().available_cache_pages(),
        Some(4)
    );
}

#[test]
fn router_runtime_flush_cache_reports_failure_when_decode_requests_are_active() {
    let tokenizer = ByteTokenizer::default();
    let mut scheduler = Scheduler::with_cache_resources(
        AlwaysUnfinishedRouterWorker,
        RadixCache::default(),
        CachePageAllocator::new(4),
    );
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("active"),
        vec![1, 2],
        SamplingParams::new(2),
    ));
    scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should dispatch");

    let engine = Engine::new(tokenizer, scheduler);
    let mut runtime = RouterRuntime::new(engine);

    let response = runtime.flush_cache();

    assert!(!response.success);
    assert!(response.message.is_empty());
    assert_eq!(
        runtime.engine().scheduler().available_cache_pages(),
        Some(2)
    );
}

#[test]
fn router_runtime_update_weights_from_disk_calls_worker_reload_hook() {
    let tokenizer = ByteTokenizer::default();
    let scheduler = Scheduler::new(RecordingWeightUpdateWorker::default());
    let engine = Engine::new(tokenizer, scheduler);
    let mut runtime = RouterRuntime::new(engine);

    runtime
        .update_weights_from_disk(WorkerWeightUpdateRequest {
            model_path: "/tmp/new-model".into(),
            load_format: Some("safetensors".into()),
            weight_version: "safetensors-sha256:abc".into(),
        })
        .expect("idle runtime should update worker weights");

    assert_eq!(
        runtime.engine().scheduler().worker().updates,
        vec![WorkerWeightUpdateRequest {
            model_path: "/tmp/new-model".into(),
            load_format: Some("safetensors".into()),
            weight_version: "safetensors-sha256:abc".into(),
        }]
    );
}

#[test]
fn router_runtime_update_weights_from_disk_rejects_when_requests_are_queued() {
    let tokenizer = ByteTokenizer::default();
    let mut scheduler = Scheduler::new(RecordingWeightUpdateWorker::default());
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("queued"),
        vec![1, 2],
        SamplingParams::new(2),
    ));
    let engine = Engine::new(tokenizer, scheduler);
    let mut runtime = RouterRuntime::new(engine);

    let error = runtime
        .update_weights_from_disk(WorkerWeightUpdateRequest {
            model_path: "/tmp/new-model".into(),
            load_format: Some("safetensors".into()),
            weight_version: "safetensors-sha256:abc".into(),
        })
        .expect_err("queued requests should prevent weight updates");

    assert!(
        error
            .to_string()
            .contains("cannot update weights while requests are running or waiting")
    );
    assert!(runtime.engine().scheduler().worker().updates.is_empty());
}

fn kv_worker(endpoint: &str, dp_rank: u32) -> KvCacheWorkerId {
    KvCacheWorkerId {
        endpoint: endpoint.to_string(),
        dp_rank,
    }
}

fn kv_workers_with_loads<const N: usize>(
    workers: [(&KvCacheWorkerId, usize); N],
) -> Vec<KvCacheWorkerSnapshot> {
    workers
        .into_iter()
        .map(|(id, active_load)| KvCacheWorkerSnapshot {
            id: id.clone(),
            active_load,
        })
        .collect()
}
