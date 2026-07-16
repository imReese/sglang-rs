use sglang_srt::engine::Engine;
use sglang_srt::router::{
    RouterGenerateComplete, RouterGenerateRequest, RouterGenerateResponse,
    RouterGenerateResponseBody, RouterGenerateStreamChunk, RouterRuntime, RouterSamplingParams,
    RouterTokenizedInput,
};
use sglang_srt::scheduler::{
    ForwardMode, RequestStage, ScheduleBatch, ScheduledRequest, Scheduler,
};
use sglang_srt::tokenizer::ByteTokenizer;
use sglang_srt::types::{RequestId, SamplingParams};
use sglang_srt::worker::{
    BatchGeneratedTokens, FallibleModelWorker, GeneratedToken, ModelWorker, PdModelWorkers,
    WorkerExecutionError, WorkerWeightUpdateRequest,
};

#[derive(Default)]
struct PrefillOnlyWorker {
    seen_modes: Vec<ForwardMode>,
}

impl ModelWorker for PrefillOnlyWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        self.seen_modes.push(batch.forward_mode());
        assert_eq!(batch.forward_mode(), ForwardMode::Prefill);

        let tokens = batch
            .requests()
            .iter()
            .map(|request| {
                assert_eq!(request.stage(), RequestStage::PrefillForward);
                GeneratedToken::unfinished(vec![11])
            })
            .collect();

        BatchGeneratedTokens::from_batch(batch, tokens).expect("output shape should match batch")
    }
}

#[derive(Default)]
struct DecodeOnlyWorker {
    seen_modes: Vec<ForwardMode>,
}

impl ModelWorker for DecodeOnlyWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        self.seen_modes.push(batch.forward_mode());
        assert_eq!(batch.forward_mode(), ForwardMode::Decode);

        let tokens = batch
            .requests()
            .iter()
            .map(|request| {
                assert_eq!(request.stage(), RequestStage::DecodeForward);
                assert_eq!(request.output_ids(), &[11]);
                GeneratedToken::finished(vec![12])
            })
            .collect();

        BatchGeneratedTokens::from_batch(batch, tokens).expect("output shape should match batch")
    }
}

#[derive(Default)]
struct WeightUpdateRecordingWorker {
    updates: Vec<WorkerWeightUpdateRequest>,
}

impl FallibleModelWorker for WeightUpdateRecordingWorker {
    fn try_generate_batch(
        &mut self,
        batch: &ScheduleBatch,
    ) -> Result<BatchGeneratedTokens, WorkerExecutionError> {
        BatchGeneratedTokens::from_batch(
            batch,
            batch
                .requests()
                .iter()
                .map(|_| GeneratedToken::finished(vec![1]))
                .collect(),
        )
        .map_err(WorkerExecutionError::from)
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
fn pd_workers_route_prefill_and_decode_batches_to_separate_executors() {
    let workers = PdModelWorkers::new(PrefillOnlyWorker::default(), DecodeOnlyWorker::default());
    let mut scheduler = Scheduler::new(workers);
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("pd-request"),
        vec![1, 2, 3],
        SamplingParams::new(2),
    ));

    let prefill_outputs = scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should dispatch");
    let decode_outputs = scheduler
        .dispatch_decode_batch(1)
        .expect("decode should dispatch");

    assert_eq!(prefill_outputs[0].token_ids, vec![11]);
    assert!(!prefill_outputs[0].finished);
    assert_eq!(decode_outputs[0].token_ids, vec![12]);
    assert!(decode_outputs[0].finished);
    assert_eq!(
        scheduler.worker().prefill().seen_modes,
        vec![ForwardMode::Prefill]
    );
    assert_eq!(
        scheduler.worker().decode().seen_modes,
        vec![ForwardMode::Decode]
    );
}

#[test]
fn pd_workers_forward_weight_updates_to_prefill_and_decode_workers() {
    let mut workers = PdModelWorkers::new(
        WeightUpdateRecordingWorker::default(),
        WeightUpdateRecordingWorker::default(),
    );
    let request = WorkerWeightUpdateRequest {
        model_path: "/models/next".to_string(),
        load_format: Some("safetensors".to_string()),
        weight_version: "v2".to_string(),
    };

    workers
        .update_weights_from_disk(&request)
        .expect("PD workers should forward reload requests");

    assert_eq!(workers.prefill().updates, vec![request.clone()]);
    assert_eq!(workers.decode().updates, vec![request]);
}

#[test]
fn router_runtime_streams_pd_worker_outputs_from_prefill_then_decode() {
    let workers = PdModelWorkers::new(PrefillOnlyWorker::default(), DecodeOnlyWorker::default());
    let scheduler = Scheduler::new(workers);
    let engine = Engine::new(ByteTokenizer, scheduler);
    let mut runtime = RouterRuntime::new(engine);

    let responses = runtime
        .generate_stream(RouterGenerateRequest {
            request_id: "router-pd".to_string(),
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
        .expect("router request should stream through split PD workers");

    assert_eq!(
        responses,
        vec![
            RouterGenerateResponse {
                request_id: "router-pd".to_string(),
                body: RouterGenerateResponseBody::Chunk(RouterGenerateStreamChunk {
                    token_ids: vec![11],
                    text: String::new(),
                    prompt_tokens: 3,
                    completion_tokens: 1,
                    cached_tokens: 0,
                    index: 0,
                }),
            },
            RouterGenerateResponse {
                request_id: "router-pd".to_string(),
                body: RouterGenerateResponseBody::Complete(RouterGenerateComplete {
                    output_ids: vec![11, 12],
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
        runtime.engine().scheduler().worker().prefill().seen_modes,
        vec![ForwardMode::Prefill]
    );
    assert_eq!(
        runtime.engine().scheduler().worker().decode().seen_modes,
        vec![ForwardMode::Decode]
    );
}
