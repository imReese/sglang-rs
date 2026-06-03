use sglang_srt::cache::{CachePageId, RadixCache};
use sglang_srt::model_executor::{ForwardModel, ModelForwardOutput, ModelRunner, ModelWorkerBatch};
use sglang_srt::scheduler::{
    ForwardMode, ScheduleBatch, ScheduledRequest, Scheduler, SchedulerError,
};
use sglang_srt::types::{DisaggregatedParams, FAKE_BOOTSTRAP_HOST, RequestId, SamplingParams};
use sglang_srt::worker::{BatchGeneratedTokens, GeneratedToken, ModelWorker};

#[derive(Default)]
struct NoopWorker;

impl ModelWorker for NoopWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        BatchGeneratedTokens::from_batch(
            batch,
            batch
                .requests()
                .iter()
                .map(|_| GeneratedToken::finished(vec![0]))
                .collect(),
        )
        .expect("output shape should match batch")
    }
}

#[derive(Default)]
struct UnfinishedPrefillWorker;

impl ModelWorker for UnfinishedPrefillWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        BatchGeneratedTokens::from_batch(
            batch,
            batch
                .requests()
                .iter()
                .map(|_| GeneratedToken::unfinished(vec![99]))
                .collect(),
        )
        .expect("output shape should match batch")
    }
}

#[derive(Default)]
struct PreparingWorker {
    seen_worker_batch: Option<ModelWorkerBatch>,
}

impl ModelWorker for PreparingWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        self.seen_worker_batch = Some(ModelWorkerBatch::from_schedule_batch(batch));
        BatchGeneratedTokens::from_batch(
            batch,
            batch
                .requests()
                .iter()
                .map(|_| GeneratedToken::finished(vec![1]))
                .collect(),
        )
        .expect("output shape should match batch")
    }
}

#[test]
fn model_worker_batch_for_prefill_flattens_uncached_tokens_with_positions() {
    let mut cache = RadixCache::default();
    cache
        .insert(&[10, 11], &[CachePageId::from(100), CachePageId::from(101)])
        .expect("insert should succeed");
    let mut scheduler = Scheduler::with_prefix_cache(NoopWorker, cache);
    enqueue_request(&mut scheduler, "req-a", &[10, 11, 12, 13]);
    enqueue_request(&mut scheduler, "req-b", &[20, 21]);

    let batch = scheduler
        .next_prefill_batch(8)
        .expect("batch should be available");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&batch);

    assert_eq!(worker_batch.forward_mode(), ForwardMode::Prefill);
    assert_eq!(
        worker_batch.request_ids(),
        &[RequestId::from("req-a"), RequestId::from("req-b")]
    );
    assert_eq!(worker_batch.input_ids(), &[12, 13, 20, 21]);
    assert_eq!(worker_batch.positions(), &[2, 3, 0, 1]);
    assert_eq!(worker_batch.sequence_lengths(), &[4, 2]);
    assert_eq!(worker_batch.request_offsets(), &[0, 2]);
    assert_eq!(
        worker_batch.prefix_cache_pages()[0],
        &[CachePageId::from(100), CachePageId::from(101)]
    );
    assert!(worker_batch.prefix_cache_pages()[1].is_empty());
}

#[test]
fn model_worker_batch_for_decode_uses_last_output_token_and_next_position() {
    let mut scheduler = Scheduler::new(UnfinishedPrefillWorker);
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("req-decode"),
        vec![1, 2, 3],
        SamplingParams { max_new_tokens: 4 },
    ));

    scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should dispatch");
    let decode_batch = scheduler
        .next_decode_batch(1)
        .expect("decode batch should be available");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&decode_batch);

    assert_eq!(worker_batch.forward_mode(), ForwardMode::Decode);
    assert_eq!(worker_batch.request_ids(), &[RequestId::from("req-decode")]);
    assert_eq!(worker_batch.input_ids(), &[99]);
    assert_eq!(worker_batch.positions(), &[3]);
    assert_eq!(worker_batch.sequence_lengths(), &[4]);
    assert_eq!(worker_batch.request_offsets(), &[0]);
}

#[test]
fn worker_can_prepare_model_worker_batch_from_scheduler_dispatch() {
    let mut scheduler = Scheduler::new(PreparingWorker::default());
    enqueue_request(&mut scheduler, "req-a", &[1, 2]);
    enqueue_request(&mut scheduler, "req-b", &[3]);

    scheduler
        .dispatch_prefill_batch(2)
        .expect("prefill should dispatch");

    let worker_batch = scheduler
        .worker()
        .seen_worker_batch
        .as_ref()
        .expect("worker should prepare model batch");

    assert_eq!(worker_batch.forward_mode(), ForwardMode::Prefill);
    assert_eq!(
        worker_batch.request_ids(),
        &[RequestId::from("req-a"), RequestId::from("req-b")]
    );
    assert_eq!(worker_batch.input_ids(), &[1, 2, 3]);
    assert_eq!(worker_batch.request_offsets(), &[0, 2]);
}

#[test]
fn model_worker_batch_exposes_pd_bootstrap_metadata() {
    let request = ScheduledRequest::new(
        RequestId::from("pd-bootstrap"),
        vec![1, 2, 3],
        SamplingParams { max_new_tokens: 2 },
    )
    .with_disaggregated_params(Some(DisaggregatedParams {
        bootstrap_host: "10.0.0.7".to_string(),
        bootstrap_port: 8998,
        bootstrap_room: 123,
    }));
    let mut scheduler = Scheduler::new(NoopWorker);
    scheduler.enqueue(request);

    let batch = scheduler
        .next_prefill_batch(1)
        .expect("prefill batch should be created");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&batch);

    assert_eq!(
        worker_batch.disaggregated_params(),
        &[Some(DisaggregatedParams {
            bootstrap_host: "10.0.0.7".to_string(),
            bootstrap_port: 8998,
            bootstrap_room: 123,
        })]
    );
}

#[test]
fn model_worker_batch_exposes_data_parallel_rank_for_pd_bootstrap() {
    let request = ScheduledRequest::new(
        RequestId::from("pd-dp-rank"),
        vec![1, 2, 3],
        SamplingParams { max_new_tokens: 1 },
    )
    .with_data_parallel_rank(7);
    let mut scheduler = Scheduler::new(NoopWorker);
    scheduler.enqueue(request);

    let batch = scheduler
        .next_prefill_batch(1)
        .expect("prefill batch should be created");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&batch);

    assert_eq!(worker_batch.data_parallel_ranks(), &[7]);
}

#[test]
fn scheduled_request_detects_fake_bootstrap_radix_cache_skip() {
    let request = ScheduledRequest::new(
        RequestId::from("fake-bootstrap"),
        vec![1, 2, 3],
        SamplingParams { max_new_tokens: 1 },
    )
    .with_disaggregated_params(Some(DisaggregatedParams {
        bootstrap_host: FAKE_BOOTSTRAP_HOST.to_string(),
        bootstrap_port: 8998,
        bootstrap_room: 0,
    }));

    assert!(request.skips_radix_cache_insert());
}

#[derive(Default)]
struct RecordingForwardModel {
    seen_input_ids: Vec<u32>,
    seen_positions: Vec<usize>,
    seen_forward_modes: Vec<ForwardMode>,
}

impl ForwardModel for RecordingForwardModel {
    fn forward(&mut self, batch: &ModelWorkerBatch) -> ModelForwardOutput {
        self.seen_input_ids = batch.input_ids().to_vec();
        self.seen_positions = batch.positions().to_vec();
        self.seen_forward_modes.push(batch.forward_mode());

        ModelForwardOutput::new(vec![vec![0.1, 9.0, 0.2], vec![0.0, 0.5, 8.0]])
            .expect("logits should be rectangular")
    }
}

#[test]
fn model_runner_calls_forward_and_returns_argmax_tokens() {
    let mut scheduler = Scheduler::new(ModelRunner::new(RecordingForwardModel::default()));
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("req-a"),
        vec![10, 11],
        SamplingParams { max_new_tokens: 1 },
    ));
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("req-b"),
        vec![20],
        SamplingParams { max_new_tokens: 1 },
    ));

    let outputs = scheduler
        .dispatch_prefill_batch(2)
        .expect("prefill should dispatch through model runner");

    assert_eq!(outputs[0].token_ids, vec![1]);
    assert_eq!(outputs[1].token_ids, vec![2]);
    assert!(outputs.iter().all(|output| output.finished));
    assert_eq!(scheduler.worker().model().seen_input_ids, vec![10, 11, 20]);
    assert_eq!(scheduler.worker().model().seen_positions, vec![0, 1, 0]);
    assert_eq!(
        scheduler.worker().model().seen_forward_modes,
        vec![ForwardMode::Prefill]
    );
}

#[derive(Default)]
struct TwoStepForwardModel {
    seen_forward_modes: Vec<ForwardMode>,
}

impl ForwardModel for TwoStepForwardModel {
    fn forward(&mut self, batch: &ModelWorkerBatch) -> ModelForwardOutput {
        self.seen_forward_modes.push(batch.forward_mode());
        match batch.forward_mode() {
            ForwardMode::Prefill => {
                ModelForwardOutput::new(vec![vec![0.0, 4.0, 0.1]]).expect("valid logits")
            }
            ForwardMode::Decode => {
                ModelForwardOutput::new(vec![vec![0.0, 0.1, 4.0]]).expect("valid logits")
            }
        }
    }
}

#[test]
fn model_runner_requeues_decode_until_scheduler_reaches_max_new_tokens() {
    let mut scheduler = Scheduler::new(ModelRunner::new(TwoStepForwardModel::default()));
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("runner-decode"),
        vec![10, 11],
        SamplingParams { max_new_tokens: 2 },
    ));

    let prefill_outputs = scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should dispatch through model runner");
    let decode_outputs = scheduler
        .dispatch_decode_batch(1)
        .expect("decode should dispatch through model runner");

    assert_eq!(prefill_outputs[0].token_ids, vec![1]);
    assert!(!prefill_outputs[0].finished);
    assert_eq!(decode_outputs[0].token_ids, vec![2]);
    assert!(decode_outputs[0].finished);
    assert_eq!(
        scheduler.worker().model().seen_forward_modes,
        vec![ForwardMode::Prefill, ForwardMode::Decode]
    );
}

#[derive(Default)]
struct EmptyForwardModel;

impl ForwardModel for EmptyForwardModel {
    fn forward(&mut self, _batch: &ModelWorkerBatch) -> ModelForwardOutput {
        ModelForwardOutput::new(Vec::new()).expect("empty batch logits are constructible")
    }
}

#[test]
fn model_runner_returns_forward_errors_without_panicking() {
    let mut scheduler = Scheduler::new(ModelRunner::new(EmptyForwardModel));
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("runner-error"),
        vec![10, 11],
        SamplingParams { max_new_tokens: 1 },
    ));

    let error = scheduler
        .dispatch_prefill_batch(1)
        .expect_err("empty model output should become a scheduler error");

    assert!(matches!(error, SchedulerError::Worker(_)));
    assert!(
        error
            .to_string()
            .contains("model forward output count (0) must match request count (1)")
    );
}

fn enqueue_request<W>(scheduler: &mut Scheduler<W>, request_id: &str, input_ids: &[u32]) {
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from(request_id),
        input_ids.to_vec(),
        SamplingParams { max_new_tokens: 4 },
    ));
}
