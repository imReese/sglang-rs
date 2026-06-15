use sglang_srt::cache::{CacheAllocationError, CachePageAllocator, CachePageId, RadixCache};
use sglang_srt::model_executor::ModelWorkerBatch;
use sglang_srt::scheduler::{
    ForwardMode, ScheduleBatch, ScheduledRequest, Scheduler, SchedulerError,
};
use sglang_srt::types::{DisaggregatedParams, FAKE_BOOTSTRAP_HOST, RequestId, SamplingParams};
use sglang_srt::worker::{
    BatchGeneratedTokens, FallibleModelWorker, GeneratedToken, ModelWorker, WorkerExecutionError,
    WorkerOutputError,
};

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
struct UnfinishedWorker;

impl ModelWorker for UnfinishedWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        BatchGeneratedTokens::from_batch(
            batch,
            batch
                .requests()
                .iter()
                .map(|_| GeneratedToken::unfinished(vec![0]))
                .collect(),
        )
        .expect("output shape should match batch")
    }
}

#[derive(Default)]
struct FailingWorker;

impl FallibleModelWorker for FailingWorker {
    fn try_generate_batch(
        &mut self,
        batch: &ScheduleBatch,
    ) -> Result<BatchGeneratedTokens, WorkerExecutionError> {
        Err(WorkerOutputError::BatchSizeMismatch {
            request_count: batch.batch_size(),
            output_count: 0,
        }
        .into())
    }
}

#[derive(Default)]
struct DecodeFailingWorker;

impl FallibleModelWorker for DecodeFailingWorker {
    fn try_generate_batch(
        &mut self,
        batch: &ScheduleBatch,
    ) -> Result<BatchGeneratedTokens, WorkerExecutionError> {
        if batch.forward_mode() == ForwardMode::Decode {
            return Err(WorkerOutputError::BatchSizeMismatch {
                request_count: batch.batch_size(),
                output_count: 0,
            }
            .into());
        }

        Ok(BatchGeneratedTokens::from_batch(
            batch,
            batch
                .requests()
                .iter()
                .map(|_| GeneratedToken::unfinished(vec![0]))
                .collect(),
        )
        .expect("output shape should match batch"))
    }
}

#[test]
fn prefill_batch_allocates_cache_pages_for_uncached_tokens() {
    let mut scheduler = Scheduler::with_cache_resources(
        NoopWorker,
        RadixCache::default(),
        CachePageAllocator::new(4),
    );
    enqueue_request(&mut scheduler, "req-a", &[1, 2]);
    enqueue_request(&mut scheduler, "req-b", &[3]);

    let batch = scheduler
        .next_prefill_batch(2)
        .expect("batch should be available");

    assert_eq!(
        batch.requests()[0].allocated_cache_pages(),
        &[CachePageId::from(0), CachePageId::from(1)]
    );
    assert_eq!(
        batch.requests()[1].allocated_cache_pages(),
        &[CachePageId::from(2)]
    );
    assert_eq!(scheduler.available_cache_pages(), Some(1));
}

#[test]
fn model_worker_batch_exposes_flattened_output_cache_pages_for_prefill() {
    let mut scheduler = Scheduler::with_cache_resources(
        NoopWorker,
        RadixCache::default(),
        CachePageAllocator::new(4),
    );
    enqueue_request(&mut scheduler, "req-a", &[1, 2]);
    enqueue_request(&mut scheduler, "req-b", &[3]);

    let batch = scheduler
        .next_prefill_batch(2)
        .expect("batch should be available");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&batch);

    assert_eq!(
        worker_batch.out_cache_pages(),
        &[
            CachePageId::from(0),
            CachePageId::from(1),
            CachePageId::from(2)
        ]
    );
}

#[test]
fn decode_batch_allocates_output_cache_page_and_keeps_sequence_cache_pages() {
    let mut scheduler = Scheduler::with_cache_resources(
        UnfinishedWorker,
        RadixCache::default(),
        CachePageAllocator::new(4),
    );
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("req-decode-cache-page"),
        vec![1, 2],
        SamplingParams::new(3),
    ));

    scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should dispatch and leave request in decode");

    let batch = scheduler
        .next_decode_batch(1)
        .expect("decode batch should allocate a cache page");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&batch);

    assert_eq!(
        batch.requests()[0].allocated_cache_pages(),
        &[CachePageId::from(2)]
    );
    assert_eq!(
        batch.requests()[0].sequence_cache_pages(),
        &[
            CachePageId::from(0),
            CachePageId::from(1),
            CachePageId::from(2)
        ]
    );
    assert_eq!(worker_batch.out_cache_pages(), &[CachePageId::from(2)]);
    assert_eq!(
        worker_batch.sequence_cache_pages(),
        &[
            CachePageId::from(0),
            CachePageId::from(1),
            CachePageId::from(2)
        ]
    );
    assert_eq!(scheduler.available_cache_pages(), Some(1));
}

#[test]
fn successful_prefill_dispatch_publishes_allocated_pages_to_radix_cache_for_future_prefix_hits() {
    let mut scheduler = Scheduler::with_cache_resources(
        NoopWorker,
        RadixCache::default(),
        CachePageAllocator::new(8),
    );
    enqueue_request(&mut scheduler, "req-a", &[1, 2, 3]);

    scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should dispatch");

    enqueue_request(&mut scheduler, "req-b", &[1, 2, 3, 4]);
    let batch = scheduler
        .next_prefill_batch(1)
        .expect("batch should be available");

    assert_eq!(
        batch.requests()[0].prefix_cache_pages(),
        &[
            CachePageId::from(0),
            CachePageId::from(1),
            CachePageId::from(2)
        ]
    );
    assert_eq!(batch.requests()[0].uncached_input_ids(), &[4]);
    assert_eq!(
        batch.requests()[0].allocated_cache_pages(),
        &[CachePageId::from(3)]
    );
}

#[test]
fn fake_bootstrap_prefill_dispatch_does_not_publish_pages_to_radix_cache() {
    let mut scheduler = Scheduler::with_cache_resources(
        NoopWorker,
        RadixCache::default(),
        CachePageAllocator::new(8),
    );
    scheduler.enqueue(
        ScheduledRequest::new(
            RequestId::from("req-fake-bootstrap"),
            vec![1, 2, 3],
            SamplingParams::new(1),
        )
        .with_disaggregated_params(Some(DisaggregatedParams {
            bootstrap_host: FAKE_BOOTSTRAP_HOST.to_string(),
            bootstrap_port: 8998,
            bootstrap_room: 0,
        })),
    );

    scheduler
        .dispatch_prefill_batch(1)
        .expect("fake bootstrap prefill should dispatch");

    enqueue_request(&mut scheduler, "req-normal", &[1, 2, 3, 4]);
    let batch = scheduler
        .next_prefill_batch(1)
        .expect("batch should be available");

    assert!(batch.requests()[0].prefix_cache_pages().is_empty());
    assert_eq!(batch.requests()[0].uncached_input_ids(), &[1, 2, 3, 4]);
}

#[test]
fn worker_failure_releases_prefill_cache_pages() {
    let mut scheduler = Scheduler::with_cache_resources(
        FailingWorker,
        RadixCache::default(),
        CachePageAllocator::new(3),
    );
    enqueue_request(&mut scheduler, "req-fail", &[1, 2]);

    let error = scheduler
        .dispatch_prefill_batch(1)
        .expect_err("worker failure should be returned");

    assert_eq!(
        error.to_string(),
        "worker execution error: worker output error: batch output count (0) must match request count (1)"
    );
    assert_eq!(scheduler.available_cache_pages(), Some(3));
}

#[test]
fn worker_failure_releases_decode_output_cache_page() {
    let mut scheduler = Scheduler::with_cache_resources(
        DecodeFailingWorker,
        RadixCache::default(),
        CachePageAllocator::new(3),
    );
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("req-decode-fail"),
        vec![1, 2],
        SamplingParams::new(3),
    ));

    scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should dispatch and leave request in decode");
    assert_eq!(scheduler.available_cache_pages(), Some(1));

    let error = scheduler
        .dispatch_decode_batch(1)
        .expect_err("decode worker failure should be returned");

    assert_eq!(
        error.to_string(),
        "worker execution error: worker output error: batch output count (0) must match request count (1)"
    );
    assert_eq!(scheduler.available_cache_pages(), Some(1));
}

#[test]
fn allocation_failure_leaves_request_queued_and_pages_available() {
    let mut scheduler = Scheduler::with_cache_resources(
        NoopWorker,
        RadixCache::default(),
        CachePageAllocator::new(2),
    );
    enqueue_request(&mut scheduler, "req-too-large", &[1, 2, 3]);

    let result = scheduler.next_prefill_batch(1);

    assert_eq!(
        result,
        Err(SchedulerError::CacheAllocation(
            CacheAllocationError::OutOfPages {
                requested: 3,
                available: 2
            }
        ))
    );
    assert_eq!(scheduler.waiting_queue_depth(), 1);
    assert_eq!(scheduler.available_cache_pages(), Some(2));
}

#[test]
fn allocation_failure_rolls_back_pages_and_preserves_queue_order_for_partial_batch() {
    let mut scheduler = Scheduler::with_cache_resources(
        NoopWorker,
        RadixCache::default(),
        CachePageAllocator::new(3),
    );
    enqueue_request(&mut scheduler, "req-a", &[1, 2]);
    enqueue_request(&mut scheduler, "req-b", &[3, 4]);

    let result = scheduler.next_prefill_batch(2);

    assert_eq!(
        result,
        Err(SchedulerError::CacheAllocation(
            CacheAllocationError::OutOfPages {
                requested: 2,
                available: 1
            }
        ))
    );
    assert_eq!(scheduler.waiting_queue_depth(), 2);
    assert_eq!(scheduler.available_cache_pages(), Some(3));

    let batch = scheduler
        .next_prefill_batch_with_token_budget(1, usize::MAX)
        .expect("first request should still be queued first");
    assert_eq!(batch.requests()[0].request_id(), &RequestId::from("req-a"));
    assert_eq!(
        batch.requests()[0].allocated_cache_pages(),
        &[CachePageId::from(0), CachePageId::from(1)]
    );
}

#[test]
fn flush_cache_clears_prefix_matches_and_restores_allocator_when_no_decode_is_active() {
    let mut scheduler = Scheduler::with_cache_resources(
        NoopWorker,
        RadixCache::default(),
        CachePageAllocator::new(4),
    );
    enqueue_request(&mut scheduler, "req-a", &[1, 2]);

    scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should dispatch");
    assert_eq!(scheduler.available_cache_pages(), Some(2));

    assert!(scheduler.flush_cache());

    assert_eq!(scheduler.available_cache_pages(), Some(4));
    enqueue_request(&mut scheduler, "req-b", &[1, 2, 3]);
    let batch = scheduler
        .next_prefill_batch(1)
        .expect("batch should be available");

    assert!(batch.requests()[0].prefix_cache_pages().is_empty());
    assert_eq!(batch.requests()[0].uncached_input_ids(), &[1, 2, 3]);
    assert_eq!(
        batch.requests()[0].allocated_cache_pages(),
        &[
            CachePageId::from(0),
            CachePageId::from(1),
            CachePageId::from(2)
        ]
    );
}

#[test]
fn flush_cache_rejects_when_decode_requests_are_active() {
    let mut scheduler = Scheduler::with_cache_resources(
        UnfinishedWorker,
        RadixCache::default(),
        CachePageAllocator::new(4),
    );
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("req-active"),
        vec![1, 2],
        SamplingParams::new(2),
    ));

    scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should dispatch");

    assert_eq!(scheduler.decode_queue_depth(), 1);
    assert!(!scheduler.flush_cache());
    assert_eq!(scheduler.available_cache_pages(), Some(2));
}

#[test]
fn flush_cache_rejects_when_waiting_requests_are_queued() {
    let mut scheduler = Scheduler::with_cache_resources(
        NoopWorker,
        RadixCache::default(),
        CachePageAllocator::new(4),
    );
    enqueue_request(&mut scheduler, "req-waiting", &[1, 2]);

    assert!(!scheduler.flush_cache());
    assert_eq!(scheduler.waiting_queue_depth(), 1);
    assert_eq!(scheduler.available_cache_pages(), Some(4));
}

fn enqueue_request<W>(scheduler: &mut Scheduler<W>, request_id: &str, input_ids: &[u32]) {
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from(request_id),
        input_ids.to_vec(),
        SamplingParams::new(1),
    ));
}
