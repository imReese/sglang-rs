use sglang_srt::cache::{CachePageId, RadixCache};
use sglang_srt::model_executor::{
    ForwardModel, LogitSampler, ModelForwardError, ModelForwardOutput, ModelRunner,
    ModelWorkerBatch, SamplingRandomSource,
};
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
    assert_eq!(worker_batch.cached_token_counts(), &[2, 0]);
    assert_eq!(worker_batch.input_token_counts(), &[2, 2]);
    assert_eq!(
        worker_batch.prefix_cache_pages()[0],
        &[CachePageId::from(100), CachePageId::from(101)]
    );
    assert!(worker_batch.prefix_cache_pages()[1].is_empty());
}

#[test]
fn model_worker_batch_exposes_last_input_token_per_prefill_request() {
    let mut cache = RadixCache::default();
    cache
        .insert(&[10], &[CachePageId::from(100)])
        .expect("insert should succeed");
    let mut scheduler = Scheduler::with_prefix_cache(NoopWorker, cache);
    enqueue_request(&mut scheduler, "req-a", &[10, 11, 12]);
    enqueue_request(&mut scheduler, "req-b", &[20, 21, 22, 23]);

    let batch = scheduler
        .next_prefill_batch(8)
        .expect("batch should be available");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&batch);

    assert_eq!(worker_batch.input_ids(), &[11, 12, 20, 21, 22, 23]);
    assert_eq!(worker_batch.last_input_token_ids(), vec![12, 23]);
}

#[test]
fn model_worker_batch_for_decode_uses_last_output_token_and_next_position() {
    let mut scheduler = Scheduler::new(UnfinishedPrefillWorker);
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("req-decode"),
        vec![1, 2, 3],
        SamplingParams::new(4),
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
    assert_eq!(worker_batch.cached_token_counts(), &[0]);
    assert_eq!(worker_batch.input_token_counts(), &[1]);
    assert_eq!(worker_batch.sequence_token_ids(), &[1, 2, 3, 99]);
    assert_eq!(worker_batch.sequence_offsets(), &[0]);
    assert_eq!(worker_batch.sequence_token_counts(), &[4]);
}

#[test]
fn model_worker_batch_exposes_last_input_token_per_decode_request() {
    let mut scheduler = Scheduler::new(UnfinishedPrefillWorker);
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("decode-a"),
        vec![1, 2],
        SamplingParams::new(4),
    ));
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("decode-b"),
        vec![3],
        SamplingParams::new(4),
    ));

    scheduler
        .dispatch_prefill_batch(2)
        .expect("prefill should dispatch");
    let decode_batch = scheduler
        .next_decode_batch(2)
        .expect("decode batch should be available");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&decode_batch);

    assert_eq!(worker_batch.input_ids(), &[99, 99]);
    assert_eq!(worker_batch.last_input_token_ids(), vec![99, 99]);
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
        SamplingParams::new(2),
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
        SamplingParams::new(1),
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
        SamplingParams::new(1),
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
    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        self.seen_input_ids = batch.input_ids().to_vec();
        self.seen_positions = batch.positions().to_vec();
        self.seen_forward_modes.push(batch.forward_mode());

        ModelForwardOutput::new(vec![vec![0.1, 9.0, 0.2], vec![0.0, 0.5, 8.0]])
    }
}

#[test]
fn model_runner_calls_forward_and_returns_argmax_tokens() {
    let mut scheduler = Scheduler::new(ModelRunner::new(RecordingForwardModel::default()));
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("req-a"),
        vec![10, 11],
        SamplingParams::new(1),
    ));
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("req-b"),
        vec![20],
        SamplingParams::new(1),
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

struct FixedSamplingRandomSource {
    values: Vec<f32>,
}

impl SamplingRandomSource for FixedSamplingRandomSource {
    fn next_unit_f32(&mut self) -> f32 {
        self.values.remove(0)
    }
}

#[derive(Default)]
struct SamplingForwardModel;

impl ForwardModel for SamplingForwardModel {
    fn forward(
        &mut self,
        _batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        ModelForwardOutput::new(vec![vec![-0.223_143_55, -1.609_438]])
    }
}

#[test]
fn model_runner_samples_from_temperature_scaled_logits() {
    let sampler = LogitSampler::new(FixedSamplingRandomSource { values: vec![0.85] });
    let mut scheduler = Scheduler::new(ModelRunner::with_sampler(SamplingForwardModel, sampler));
    let mut sampling = SamplingParams::new(1);
    sampling.temperature = Some(1.0);
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("sampled"),
        vec![10, 11],
        sampling,
    ));

    let outputs = scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should dispatch through model runner");

    assert_eq!(outputs[0].token_ids, vec![1]);
}

#[test]
fn model_runner_finishes_request_when_sampled_token_matches_stop_token() {
    let mut scheduler = Scheduler::new(ModelRunner::new(TwoStepForwardModel::default()));
    let mut sampling = SamplingParams::new(4);
    sampling.stop_token_ids = vec![1];
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("runner-stop-token"),
        vec![10, 11],
        sampling,
    ));

    let outputs = scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should dispatch through model runner");

    assert_eq!(outputs[0].token_ids, vec![1]);
    assert!(outputs[0].finished);
    assert_eq!(scheduler.decode_queue_depth(), 0);
}

#[test]
fn model_runner_ignores_stop_tokens_when_ignore_eos_is_enabled() {
    let mut scheduler = Scheduler::new(ModelRunner::new(TwoStepForwardModel::default()));
    let mut sampling = SamplingParams::new(2);
    sampling.stop_token_ids = vec![1];
    sampling.ignore_eos = true;
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("runner-ignore-eos"),
        vec![10, 11],
        sampling,
    ));

    let outputs = scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should dispatch through model runner");

    assert_eq!(outputs[0].token_ids, vec![1]);
    assert!(!outputs[0].finished);
    assert_eq!(scheduler.decode_queue_depth(), 1);
}

#[derive(Default)]
struct FlattenedPrefillLogitsModel;

impl ForwardModel for FlattenedPrefillLogitsModel {
    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        assert_eq!(batch.input_ids(), &[10, 11, 20, 21, 22]);
        ModelForwardOutput::from_token_logits(
            batch,
            vec![
                vec![9.0, 0.1, 0.2],
                vec![0.1, 9.0, 0.2],
                vec![0.2, 0.3, 9.0],
                vec![0.4, 0.5, 9.0],
                vec![0.5, 9.0, 0.6],
            ],
        )
    }
}

#[test]
fn model_runner_samples_last_token_logits_from_flattened_prefill_output() {
    let mut scheduler = Scheduler::new(ModelRunner::new(FlattenedPrefillLogitsModel));
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("prefill-a"),
        vec![10, 11],
        SamplingParams::new(1),
    ));
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("prefill-b"),
        vec![20, 21, 22],
        SamplingParams::new(1),
    ));

    let outputs = scheduler
        .dispatch_prefill_batch(2)
        .expect("prefill should dispatch through model runner");

    assert_eq!(outputs[0].token_ids, vec![1]);
    assert_eq!(outputs[1].token_ids, vec![1]);
}

#[test]
fn model_forward_output_rejects_flattened_logits_with_wrong_token_count() {
    let mut scheduler = Scheduler::new(NoopWorker);
    enqueue_request(&mut scheduler, "prefill-a", &[10, 11]);
    enqueue_request(&mut scheduler, "prefill-b", &[20]);
    let batch = scheduler
        .next_prefill_batch(2)
        .expect("batch should be available");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&batch);

    let error =
        ModelForwardOutput::from_token_logits(&worker_batch, vec![vec![9.0, 0.1], vec![0.1, 9.0]])
            .expect_err("token logits must match flattened input token count");

    assert_eq!(
        error,
        ModelForwardError::TokenLogitCountMismatch {
            token_count: 3,
            logit_count: 2,
        }
    );
}

#[test]
fn model_forward_output_rejects_flattened_logits_for_request_without_input_tokens() {
    let mut cache = RadixCache::default();
    cache
        .insert(&[10, 11], &[CachePageId::from(100), CachePageId::from(101)])
        .expect("insert should succeed");
    let mut scheduler = Scheduler::with_prefix_cache(NoopWorker, cache);
    enqueue_request(&mut scheduler, "prefill-a", &[10, 11]);
    let batch = scheduler
        .next_prefill_batch(1)
        .expect("batch should be available");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&batch);

    let error = ModelForwardOutput::from_token_logits(&worker_batch, Vec::new())
        .expect_err("fully cached requests should not index missing token logits");

    assert_eq!(
        error,
        ModelForwardError::MissingRequestTokenLogits { request_index: 0 }
    );
}

#[derive(Default)]
struct TwoStepForwardModel {
    seen_forward_modes: Vec<ForwardMode>,
}

impl ForwardModel for TwoStepForwardModel {
    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        self.seen_forward_modes.push(batch.forward_mode());
        match batch.forward_mode() {
            ForwardMode::Prefill => ModelForwardOutput::new(vec![vec![0.0, 4.0, 0.1]]),
            ForwardMode::Decode => ModelForwardOutput::new(vec![vec![0.0, 0.1, 4.0]]),
        }
    }
}

#[test]
fn model_runner_requeues_decode_until_scheduler_reaches_max_new_tokens() {
    let mut scheduler = Scheduler::new(ModelRunner::new(TwoStepForwardModel::default()));
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("runner-decode"),
        vec![10, 11],
        SamplingParams::new(2),
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
    fn forward(
        &mut self,
        _batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        ModelForwardOutput::new(Vec::new())
    }
}

#[derive(Default)]
struct ExtraForwardModel;

impl ForwardModel for ExtraForwardModel {
    fn forward(
        &mut self,
        _batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        ModelForwardOutput::new(vec![vec![0.0, 1.0], vec![1.0, 0.0]])
    }
}

#[test]
fn model_runner_rejects_extra_forward_outputs() {
    let mut scheduler = Scheduler::new(ModelRunner::new(ExtraForwardModel));
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("runner-extra"),
        vec![10, 11],
        SamplingParams::new(1),
    ));

    let error = scheduler
        .dispatch_prefill_batch(1)
        .expect_err("extra model output should become a scheduler error");

    assert!(matches!(error, SchedulerError::Worker(_)));
    assert!(
        error
            .to_string()
            .contains("model forward output count (2) must match request count (1)")
    );
}

#[test]
fn model_runner_returns_forward_errors_without_panicking() {
    let mut scheduler = Scheduler::new(ModelRunner::new(EmptyForwardModel));
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("runner-error"),
        vec![10, 11],
        SamplingParams::new(1),
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

#[derive(Default)]
struct FailingForwardModel;

impl ForwardModel for FailingForwardModel {
    fn forward(
        &mut self,
        _batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        Err(ModelForwardError::Runtime(
            "DeepSeek V4 CPU fallback does not support tensor dtype F8_E4M3".to_string(),
        ))
    }
}

#[test]
fn model_runner_propagates_forward_runtime_errors() {
    let mut scheduler = Scheduler::new(ModelRunner::new(FailingForwardModel));
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("runner-forward-error"),
        vec![10, 11],
        SamplingParams::new(1),
    ));

    let error = scheduler
        .dispatch_prefill_batch(1)
        .expect_err("fallible model forward errors should become scheduler errors");

    assert!(matches!(error, SchedulerError::Worker(_)));
    assert!(
        error
            .to_string()
            .contains("DeepSeek V4 CPU fallback does not support tensor dtype F8_E4M3")
    );
}

fn enqueue_request<W>(scheduler: &mut Scheduler<W>, request_id: &str, input_ids: &[u32]) {
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from(request_id),
        input_ids.to_vec(),
        SamplingParams::new(4),
    ));
}
