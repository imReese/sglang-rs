use sglang_srt::cache::{CachePageId, RadixCache};
use sglang_srt::scheduler::{
    ForwardMode, RequestStage, ScheduleBatch, ScheduledRequest, Scheduler,
};
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
struct RecordingBatchWorker {
    seen_request_ids: Vec<RequestId>,
}

impl ModelWorker for RecordingBatchWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        self.seen_request_ids = batch
            .requests()
            .iter()
            .map(|request| request.request_id().clone())
            .collect();
        BatchGeneratedTokens::from_batch(
            batch,
            batch
                .requests()
                .iter()
                .enumerate()
                .map(|(index, _)| GeneratedToken::finished(vec![index as u32 + 10]))
                .collect(),
        )
        .expect("output shape should match batch")
    }
}

struct UnfinishedWorker;

impl ModelWorker for UnfinishedWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        BatchGeneratedTokens::from_batch(
            batch,
            batch
                .requests()
                .iter()
                .map(|_| GeneratedToken::unfinished(vec![42]))
                .collect(),
        )
        .expect("output shape should match batch")
    }
}

#[test]
fn next_prefill_batch_applies_batch_limit_and_preserves_queue_order() {
    let mut scheduler = Scheduler::new(NoopWorker);
    enqueue_request(&mut scheduler, "req-a", &[1, 2]);
    enqueue_request(&mut scheduler, "req-b", &[3, 4]);
    enqueue_request(&mut scheduler, "req-c", &[5, 6]);

    let batch = scheduler
        .next_prefill_batch(2)
        .expect("batch should be available");

    assert_eq!(batch.forward_mode(), ForwardMode::Prefill);
    assert_eq!(batch.batch_size(), 2);
    assert_eq!(scheduler.waiting_queue_depth(), 1);
    assert_eq!(batch.requests()[0].request_id(), &RequestId::from("req-a"));
    assert_eq!(batch.requests()[1].request_id(), &RequestId::from("req-b"));
    assert_eq!(batch.requests()[0].stage(), RequestStage::PrefillForward);
    assert_eq!(batch.requests()[1].stage(), RequestStage::PrefillForward);
}

#[test]
fn next_prefill_batch_attaches_prefix_cache_matches_to_each_request() {
    let mut cache = RadixCache::default();
    cache
        .insert(&[7, 8], &[CachePageId::from(700), CachePageId::from(701)])
        .expect("insert should succeed");
    cache
        .insert(&[9], &[CachePageId::from(900)])
        .expect("insert should succeed");
    let mut scheduler = Scheduler::with_prefix_cache(NoopWorker, cache);

    enqueue_request(&mut scheduler, "req-a", &[7, 8, 10]);
    enqueue_request(&mut scheduler, "req-b", &[9, 11]);

    let batch = scheduler
        .next_prefill_batch(8)
        .expect("batch should be available");

    assert_eq!(
        batch.requests()[0].prefix_cache_pages(),
        &[CachePageId::from(700), CachePageId::from(701)]
    );
    assert_eq!(batch.requests()[0].uncached_input_ids(), &[10]);
    assert_eq!(
        batch.requests()[1].prefix_cache_pages(),
        &[CachePageId::from(900)]
    );
    assert_eq!(batch.requests()[1].uncached_input_ids(), &[11]);
}

#[test]
fn next_prefill_batch_with_token_budget_stops_before_exceeding_uncached_token_budget() {
    let mut scheduler = Scheduler::new(NoopWorker);
    enqueue_request(&mut scheduler, "req-a", &[1, 2]);
    enqueue_request(&mut scheduler, "req-b", &[3, 4, 5]);
    enqueue_request(&mut scheduler, "req-c", &[6]);

    let batch = scheduler
        .next_prefill_batch_with_token_budget(8, 4)
        .expect("batch should be available");

    assert_eq!(batch.batch_size(), 1);
    assert_eq!(batch.total_uncached_tokens(), 2);
    assert_eq!(batch.requests()[0].request_id(), &RequestId::from("req-a"));
    assert_eq!(scheduler.waiting_queue_depth(), 2);
}

#[test]
fn next_prefill_batch_with_token_budget_counts_only_uncached_tokens_after_prefix_match() {
    let mut cache = RadixCache::default();
    cache
        .insert(&[10, 11], &[CachePageId::from(100), CachePageId::from(101)])
        .expect("insert should succeed");
    let mut scheduler = Scheduler::with_prefix_cache(NoopWorker, cache);
    enqueue_request(&mut scheduler, "req-a", &[10, 11, 12]);
    enqueue_request(&mut scheduler, "req-b", &[13, 14]);

    let batch = scheduler
        .next_prefill_batch_with_token_budget(8, 3)
        .expect("batch should be available");

    assert_eq!(batch.batch_size(), 2);
    assert_eq!(batch.total_uncached_tokens(), 3);
    assert_eq!(batch.requests()[0].uncached_input_ids(), &[12]);
    assert_eq!(batch.requests()[1].uncached_input_ids(), &[13, 14]);
}

#[test]
fn next_prefill_batch_with_token_budget_includes_first_oversized_request_to_avoid_starvation() {
    let mut scheduler = Scheduler::new(NoopWorker);
    enqueue_request(&mut scheduler, "req-a", &[1, 2, 3, 4, 5]);
    enqueue_request(&mut scheduler, "req-b", &[6]);

    let batch = scheduler
        .next_prefill_batch_with_token_budget(8, 3)
        .expect("batch should be available");

    assert_eq!(batch.batch_size(), 1);
    assert_eq!(batch.total_uncached_tokens(), 5);
    assert_eq!(batch.requests()[0].request_id(), &RequestId::from("req-a"));
    assert_eq!(scheduler.waiting_queue_depth(), 1);
}

#[test]
fn abort_request_removes_waiting_request_by_id() {
    let mut scheduler = Scheduler::new(NoopWorker);
    enqueue_request(&mut scheduler, "req-a", &[1]);
    enqueue_request(&mut scheduler, "req-b", &[2]);
    enqueue_request(&mut scheduler, "req-c", &[3]);

    assert!(scheduler.abort_request(&RequestId::from("req-b")));
    assert_eq!(scheduler.waiting_queue_depth(), 2);
    assert!(!scheduler.abort_request(&RequestId::from("missing")));

    let batch = scheduler
        .next_prefill_batch(8)
        .expect("remaining requests should batch");

    assert_eq!(batch.requests()[0].request_id(), &RequestId::from("req-a"));
    assert_eq!(batch.requests()[1].request_id(), &RequestId::from("req-c"));
}

#[test]
fn abort_request_removes_decode_request_by_id() {
    let mut scheduler = Scheduler::new(UnfinishedWorker);
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("decode-a"),
        vec![1],
        SamplingParams { max_new_tokens: 2 },
    ));

    scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should leave request in decode queue");

    assert_eq!(scheduler.decode_queue_depth(), 1);
    assert!(scheduler.abort_request(&RequestId::from("decode-a")));
    assert_eq!(scheduler.decode_queue_depth(), 0);
}

#[test]
fn dispatch_prefill_batch_sends_batch_to_worker_and_returns_outputs_by_request() {
    let mut scheduler = Scheduler::new(RecordingBatchWorker::default());
    enqueue_request(&mut scheduler, "req-a", &[1]);
    enqueue_request(&mut scheduler, "req-b", &[2]);

    let outputs = scheduler
        .dispatch_prefill_batch(2)
        .expect("dispatch should succeed");

    assert_eq!(
        scheduler.worker().seen_request_ids,
        vec![RequestId::from("req-a"), RequestId::from("req-b")]
    );
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].request_id, RequestId::from("req-a"));
    assert_eq!(outputs[0].token_ids, vec![10]);
    assert_eq!(outputs[1].request_id, RequestId::from("req-b"));
    assert_eq!(outputs[1].token_ids, vec![11]);
}

#[test]
fn dispatch_prefill_batch_with_token_budget_sends_only_budgeted_requests_to_worker() {
    let mut scheduler = Scheduler::new(RecordingBatchWorker::default());
    enqueue_request(&mut scheduler, "req-a", &[1, 2]);
    enqueue_request(&mut scheduler, "req-b", &[3, 4, 5]);

    let outputs = scheduler
        .dispatch_prefill_batch_with_token_budget(8, 2)
        .expect("dispatch should succeed");

    assert_eq!(
        scheduler.worker().seen_request_ids,
        vec![RequestId::from("req-a")]
    );
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].request_id, RequestId::from("req-a"));
    assert_eq!(scheduler.waiting_queue_depth(), 1);
}

fn enqueue_request<W>(scheduler: &mut Scheduler<W>, request_id: &str, input_ids: &[u32]) {
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from(request_id),
        input_ids.to_vec(),
        SamplingParams { max_new_tokens: 1 },
    ));
}
