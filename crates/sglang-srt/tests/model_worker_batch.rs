use sglang_srt::cache::{CachePageId, RadixCache};
use sglang_srt::model_executor::ModelWorkerBatch;
use sglang_srt::scheduler::{ForwardMode, ScheduleBatch, ScheduledRequest, Scheduler};
use sglang_srt::types::{RequestId, SamplingParams};
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

fn enqueue_request<W>(scheduler: &mut Scheduler<W>, request_id: &str, input_ids: &[u32]) {
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from(request_id),
        input_ids.to_vec(),
        SamplingParams { max_new_tokens: 4 },
    ));
}
