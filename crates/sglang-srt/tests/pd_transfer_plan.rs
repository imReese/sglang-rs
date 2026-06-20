use sglang_srt::cache::{CachePageAllocator, CachePageId, RadixCache};
use sglang_srt::engine::{Engine, RuntimeError};
use sglang_srt::model_executor::ModelWorkerBatch;
use sglang_srt::pd_bootstrap::{MooncakeBootstrapKvCacheTransferExecutor, PrefillBootstrapService};
use sglang_srt::router::{RouterRuntime, RouterTransferPollResponse};
use sglang_srt::scheduler::{ScheduleBatch, ScheduledRequest, Scheduler, SchedulerError};
use sglang_srt::tokenizer::ByteTokenizer;
use sglang_srt::transfer::{
    DecodeBootstrapMetadataPublishSummary, DecodeBootstrapPublisher, DecodeBootstrapRegistry,
    DecodeBootstrapSession, FakeKvCacheTransferExecutor, KvCacheTransferError,
    KvCacheTransferExecutor, KvCacheTransferPlan, KvCacheTransferPlanError, KvCacheTransferSpan,
    KvPoll, KvTransferModelWorker, LocalSnapshotTransferPdModelWorkers, MooncakeBatchId,
    MooncakeBatchReleaser, MooncakeError, MooncakeKvCacheLayout, MooncakeKvCacheTransferExecutor,
    MooncakeOpcode, MooncakeRemoteKvLayout, MooncakeSessionTargetResolver, MooncakeSubmittedBatch,
    MooncakeTransferRequest, MooncakeTransferStatus, MooncakeTransferStatusCode,
    MooncakeTransferStatusReader, MooncakeTransferSubmitter, MooncakeTransferTarget,
    MooncakeTransferTargetResolver, TransferableKvCacheMemory, TransferableKvCacheRegion,
    build_mooncake_kv_transfer_requests, build_mooncake_remote_kv_transfer_requests,
    execute_kv_cache_transfer_plan, is_decode_request_kv_ready, poll_mooncake_transfer_batches,
};
use sglang_srt::types::{
    BootstrapRoom, DisaggregatedParams, RequestId, SamplingParams, TokenGenerateRequest,
};
use sglang_srt::worker::{
    BatchGeneratedTokens, FallibleModelWorker, GeneratedToken, ModelWorker, WorkerExecutionError,
    WorkerWeightUpdateRequest,
};
use std::time::Duration;

#[derive(Default)]
struct FinishedWorker;

impl ModelWorker for FinishedWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
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

#[derive(Default)]
struct UnfinishedWorker;

impl ModelWorker for UnfinishedWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        BatchGeneratedTokens::from_batch(
            batch,
            batch
                .requests()
                .iter()
                .map(|_| GeneratedToken::unfinished(vec![1]))
                .collect(),
        )
        .expect("output shape should match batch")
    }
}

#[derive(Default)]
struct ReloadRecordingWorker {
    updates: Vec<WorkerWeightUpdateRequest>,
}

impl FallibleModelWorker for ReloadRecordingWorker {
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

impl sglang_srt::transfer::KvCachePageSnapshotProvider for ReloadRecordingWorker {
    type Snapshot = CachePageId;

    fn export_kv_cache_pages(
        &self,
        cache_pages: &[CachePageId],
    ) -> Result<Vec<Self::Snapshot>, KvCacheTransferError> {
        Ok(cache_pages.to_vec())
    }
}

impl sglang_srt::transfer::KvCachePageSnapshotImporter for ReloadRecordingWorker {
    type Snapshot = CachePageId;

    fn import_kv_cache_pages(
        &mut self,
        _snapshots: Vec<Self::Snapshot>,
    ) -> Result<(), KvCacheTransferError> {
        Ok(())
    }
}

#[test]
fn transfer_plan_uses_uncached_prefill_pages_as_pd_delta() {
    let mut prefix_cache = RadixCache::default();
    prefix_cache
        .insert(&[10, 11], &[CachePageId::from(100), CachePageId::from(101)])
        .expect("prefix cache should insert");
    let mut scheduler =
        Scheduler::with_cache_resources(FinishedWorker, prefix_cache, CachePageAllocator::new(8));
    scheduler.enqueue(
        ScheduledRequest::new(
            RequestId::from("pd-delta"),
            vec![10, 11, 12, 13],
            SamplingParams::new(1),
        )
        .with_disaggregated_params(Some(disaggregated_params(77)))
        .with_data_parallel_rank(3),
    );

    let batch = scheduler
        .next_prefill_batch(1)
        .expect("prefill batch should be available");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&batch);
    let transfer_plan = KvCacheTransferPlan::from_prefill_worker_batch(&worker_batch)
        .expect("transfer plan should build from prefill batch");

    assert_eq!(transfer_plan.len(), 1);
    let span = &transfer_plan.spans()[0];
    assert_eq!(span.request_id(), &RequestId::from("pd-delta"));
    assert_eq!(span.disaggregated_params(), &disaggregated_params(77));
    assert_eq!(span.bootstrap_room(), 77);
    assert_eq!(span.data_parallel_rank(), 3);
    assert_eq!(span.token_offset(), 2);
    assert_eq!(span.token_count(), 2);
    assert_eq!(
        span.cache_pages(),
        &[CachePageId::from(0), CachePageId::from(1)]
    );
    assert!(!span.is_noop());
}

#[test]
fn transfer_plan_keeps_noop_span_when_decode_radix_cache_satisfies_full_prefix() {
    let mut prefix_cache = RadixCache::default();
    prefix_cache
        .insert(
            &[20, 21, 22],
            &[
                CachePageId::from(200),
                CachePageId::from(201),
                CachePageId::from(202),
            ],
        )
        .expect("prefix cache should insert");
    let mut scheduler =
        Scheduler::with_cache_resources(FinishedWorker, prefix_cache, CachePageAllocator::new(2));
    scheduler.enqueue(
        ScheduledRequest::new(
            RequestId::from("pd-noop"),
            vec![20, 21, 22],
            SamplingParams::new(1),
        )
        .with_disaggregated_params(Some(disaggregated_params(88))),
    );

    let batch = scheduler
        .next_prefill_batch(1)
        .expect("prefill batch should be available");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&batch);
    let transfer_plan = KvCacheTransferPlan::from_prefill_worker_batch(&worker_batch)
        .expect("transfer plan should build from fully cached prefill");

    assert_eq!(transfer_plan.len(), 1);
    let span = &transfer_plan.spans()[0];
    assert_eq!(span.token_offset(), 3);
    assert_eq!(span.token_count(), 0);
    assert!(span.cache_pages().is_empty());
    assert!(span.is_noop());
}

#[test]
fn transfer_plan_skips_non_pd_prefill_requests_but_consumes_their_cache_pages() {
    let mut scheduler = Scheduler::with_cache_resources(
        FinishedWorker,
        RadixCache::default(),
        CachePageAllocator::new(8),
    );
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("local"),
        vec![1, 2],
        SamplingParams::new(1),
    ));
    scheduler.enqueue(
        ScheduledRequest::new(RequestId::from("pd"), vec![3, 4, 5], SamplingParams::new(1))
            .with_disaggregated_params(Some(disaggregated_params(99))),
    );

    let batch = scheduler
        .next_prefill_batch(2)
        .expect("prefill batch should be available");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&batch);
    let transfer_plan = KvCacheTransferPlan::from_prefill_worker_batch(&worker_batch)
        .expect("transfer plan should build");

    assert_eq!(transfer_plan.len(), 1);
    let span = &transfer_plan.spans()[0];
    assert_eq!(span.request_id(), &RequestId::from("pd"));
    assert_eq!(
        span.cache_pages(),
        &[
            CachePageId::from(2),
            CachePageId::from(3),
            CachePageId::from(4)
        ]
    );
}

#[test]
fn transfer_plan_rejects_decode_worker_batches() {
    let mut scheduler = Scheduler::new(UnfinishedWorker);
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("decode"),
        vec![1, 2],
        SamplingParams::new(2),
    ));
    scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should dispatch");
    let decode_batch = scheduler
        .next_decode_batch(1)
        .expect("decode batch should be available");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&decode_batch);

    let error = KvCacheTransferPlan::from_prefill_worker_batch(&worker_batch)
        .expect_err("decode batch should not create prefill transfer plan");

    assert_eq!(error, KvCacheTransferPlanError::NonPrefillBatch);
}

#[test]
fn executor_marks_non_noop_transfer_spans_success_after_submit() {
    let transfer_plan = transfer_plan_for_request("pd-submit", &[10, 11, 12], Some(1), 4);
    let mut registry = registry_with_session("pd-submit", 4);
    let mut executor = RecordingTransferExecutor::default();

    let summary = execute_kv_cache_transfer_plan(&mut registry, &mut executor, &transfer_plan)
        .expect("transfer plan should execute");

    assert_eq!(summary.submitted_spans(), 1);
    assert_eq!(summary.noop_spans(), 0);
    assert_eq!(executor.seen_rooms, vec![4]);
    assert_eq!(
        registry.get(4).expect("session should remain").status(),
        KvPoll::Success
    );
}

#[test]
fn fake_transfer_executor_marks_spans_success_inline() {
    let transfer_plan = transfer_plan_for_request("pd-fake", &[1, 2], None, 27);
    let mut registry = registry_with_session("pd-fake", 27);
    let mut executor = FakeKvCacheTransferExecutor::default();

    let summary = execute_kv_cache_transfer_plan(&mut registry, &mut executor, &transfer_plan)
        .expect("fake transfer should execute inline");

    assert_eq!(summary.submitted_spans(), 1);
    assert_eq!(executor.transferred_rooms(), &[27]);
    assert_eq!(
        registry.get(27).expect("session should remain").status(),
        KvPoll::Success
    );
}

#[test]
fn executor_marks_noop_transfer_spans_success_without_submit() {
    let transfer_plan = transfer_plan_for_request("pd-noop-exec", &[20, 21], Some(2), 5);
    let mut registry = registry_with_session("pd-noop-exec", 5);
    let mut executor = RecordingTransferExecutor::default();

    let summary = execute_kv_cache_transfer_plan(&mut registry, &mut executor, &transfer_plan)
        .expect("noop transfer plan should execute");

    assert_eq!(summary.submitted_spans(), 0);
    assert_eq!(summary.noop_spans(), 1);
    assert!(executor.seen_rooms.is_empty());
    assert_eq!(
        registry.get(5).expect("session should remain").status(),
        KvPoll::Success
    );
}

#[test]
fn executor_reports_missing_bootstrap_room_before_submit() {
    let transfer_plan = transfer_plan_for_request("pd-missing-room", &[30, 31], None, 6);
    let mut registry = DecodeBootstrapRegistry::default();
    let mut executor = RecordingTransferExecutor::default();

    let error = execute_kv_cache_transfer_plan(&mut registry, &mut executor, &transfer_plan)
        .expect_err("missing room should fail");

    assert_eq!(
        error,
        KvCacheTransferError::Registry(
            sglang_srt::transfer::DecodeBootstrapRegistryError::MissingBootstrapRoom(6)
        )
    );
    assert!(executor.seen_rooms.is_empty());
}

#[test]
fn executor_marks_span_failed_when_transfer_submit_fails() {
    let transfer_plan = transfer_plan_for_request("pd-fail", &[40, 41], None, 7);
    let mut registry = registry_with_session("pd-fail", 7);
    let mut executor = RecordingTransferExecutor {
        fail_room: Some(7),
        ..Default::default()
    };

    let error = execute_kv_cache_transfer_plan(&mut registry, &mut executor, &transfer_plan)
        .expect_err("transfer failure should propagate");

    assert_eq!(
        error,
        KvCacheTransferError::Runtime("submit failed for room 7".to_string())
    );
    assert_eq!(executor.seen_rooms, vec![7]);
    assert_eq!(
        registry.get(7).expect("session should remain").status(),
        KvPoll::Failed
    );
}

#[test]
fn transfer_model_worker_submits_pd_prefill_transfer_during_scheduler_dispatch() {
    let worker = KvTransferModelWorker::new(
        FinishedWorker,
        registry_with_session("pd-dispatch", 15),
        RecordingTransferExecutor::default(),
    );
    let mut scheduler =
        Scheduler::with_cache_resources(worker, RadixCache::default(), CachePageAllocator::new(4));
    scheduler.enqueue(
        ScheduledRequest::new(
            RequestId::from("pd-dispatch"),
            vec![1, 2, 3],
            SamplingParams::new(1),
        )
        .with_disaggregated_params(Some(disaggregated_params(15))),
    );

    let outputs = scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill dispatch should transfer KV");

    assert_eq!(outputs[0].token_ids, vec![1]);
    let worker = scheduler.worker();
    assert_eq!(
        worker
            .last_transfer_summary()
            .expect("transfer summary should be recorded")
            .submitted_spans(),
        1
    );
    assert_eq!(worker.transfer_executor().seen_rooms, vec![15]);
    assert!(worker.registry().get(15).is_none());
}

#[test]
fn transfer_model_worker_forwards_weight_updates_to_inner_worker() {
    let mut worker = KvTransferModelWorker::new(
        ReloadRecordingWorker::default(),
        DecodeBootstrapRegistry::default(),
        RecordingTransferExecutor::default(),
    );
    let request = WorkerWeightUpdateRequest {
        model_path: "/models/reloaded".to_string(),
        load_format: Some("safetensors".to_string()),
        weight_version: "weights-v2".to_string(),
    };

    worker
        .update_weights_from_disk(&request)
        .expect("transfer wrapper should forward reload requests");

    assert_eq!(worker.worker().updates, vec![request]);
}

#[test]
fn local_snapshot_transfer_pd_workers_forward_weight_updates_to_both_workers() {
    let mut workers = LocalSnapshotTransferPdModelWorkers::new(
        ReloadRecordingWorker::default(),
        ReloadRecordingWorker::default(),
    );
    let request = WorkerWeightUpdateRequest {
        model_path: "/models/local-snapshot-reloaded".to_string(),
        load_format: Some("safetensors".to_string()),
        weight_version: "snapshot-v2".to_string(),
    };

    workers
        .update_weights_from_disk(&request)
        .expect("local snapshot wrapper should forward reload requests");

    assert_eq!(workers.prefill().updates, vec![request.clone()]);
    assert_eq!(workers.decode().updates, vec![request]);
}

#[test]
fn transfer_model_worker_registers_pd_prefill_session_before_transfer() {
    let worker = KvTransferModelWorker::new(
        UnfinishedWorker,
        DecodeBootstrapRegistry::default(),
        RecordingTransferExecutor::default(),
    );
    let mut scheduler =
        Scheduler::with_cache_resources(worker, RadixCache::default(), CachePageAllocator::new(4));
    scheduler.enqueue(
        ScheduledRequest::new(
            RequestId::from("pd-auto-register"),
            vec![1, 2, 3],
            SamplingParams::new(2),
        )
        .with_disaggregated_params(Some(disaggregated_params(26)))
        .with_data_parallel_rank(2),
    );

    scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill dispatch should auto-register and transfer KV");

    let worker = scheduler.worker();
    let session = worker
        .registry()
        .get(26)
        .expect("bootstrap session should be registered from request metadata");
    assert_eq!(session.request_id(), &RequestId::from("pd-auto-register"));
    assert_eq!(session.data_parallel_rank(), 2);
    assert_eq!(session.status(), KvPoll::Success);
    assert_eq!(worker.transfer_executor().seen_rooms, vec![26]);
}

#[test]
fn transfer_model_worker_publishes_decode_bootstrap_metadata_from_allocated_pages() {
    let worker = KvTransferModelWorker::new(
        UnfinishedWorker,
        DecodeBootstrapRegistry::default(),
        RecordingTransferExecutor::default(),
    )
    .with_decode_bootstrap_publisher(RecordingDecodeBootstrapPublisher::default());
    let mut scheduler =
        Scheduler::with_cache_resources(worker, RadixCache::default(), CachePageAllocator::new(4));
    scheduler.enqueue(
        ScheduledRequest::new(
            RequestId::from("pd-decode-publish"),
            vec![1, 2, 3],
            SamplingParams::new(2),
        )
        .with_disaggregated_params(Some(DisaggregatedParams {
            bootstrap_host: "10.0.0.8".to_string(),
            bootstrap_port: 8200,
            bootstrap_room: 84,
        }))
        .with_data_parallel_rank(3),
    );

    scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill dispatch should publish decode bootstrap metadata");

    let worker = scheduler.worker();
    let publisher = worker.decode_bootstrap_publisher();
    assert_eq!(publisher.published.len(), 1);
    assert_eq!(
        publisher.published[0],
        PublishedDecodeBootstrapSpan {
            request_id: RequestId::from("pd-decode-publish"),
            bootstrap_addr: "10.0.0.8:8200".to_string(),
            bootstrap_room: 84,
            prefill_dp_rank: 3,
            dst_kv_indices: vec![0, 1, 2],
            decode_prefix_len: Some(3),
        }
    );
    assert_eq!(worker.transfer_executor().seen_rooms, vec![84]);
}

#[test]
fn transfer_model_worker_decode_side_bootstrap_only_publishes_without_transfer_submit() {
    let worker = KvTransferModelWorker::new(
        UnfinishedWorker,
        DecodeBootstrapRegistry::default(),
        RecordingTransferExecutor::default(),
    )
    .with_decode_bootstrap_publisher(RecordingDecodeBootstrapPublisher::default())
    .with_decode_side_bootstrap_only();
    let mut scheduler =
        Scheduler::with_cache_resources(worker, RadixCache::default(), CachePageAllocator::new(4));
    scheduler.enqueue(
        ScheduledRequest::new(
            RequestId::from("pd-decode-bootstrap-only"),
            vec![1, 2, 3],
            SamplingParams::new(2),
        )
        .with_disaggregated_params(Some(DisaggregatedParams {
            bootstrap_host: "10.0.0.8".to_string(),
            bootstrap_port: 8200,
            bootstrap_room: 85,
        }))
        .with_data_parallel_rank(3),
    );

    scheduler
        .dispatch_prefill_batch(1)
        .expect("decode-side prefill dispatch should publish bootstrap metadata");

    let worker = scheduler.worker();
    assert_eq!(worker.decode_bootstrap_publisher().published.len(), 1);
    assert!(
        worker.transfer_executor().seen_rooms.is_empty(),
        "decode side must not submit Mooncake transfers"
    );
    assert_eq!(
        worker
            .registry()
            .get(85)
            .expect("decode-side session should be registered")
            .status(),
        KvPoll::Success
    );
    assert_eq!(
        worker
            .last_transfer_summary()
            .expect("bootstrap-only summary should be recorded")
            .submitted_spans(),
        0
    );
}

#[test]
fn transfer_model_worker_skips_non_pd_prefill_transfer() {
    let worker = KvTransferModelWorker::new(
        FinishedWorker,
        DecodeBootstrapRegistry::default(),
        RecordingTransferExecutor::default(),
    );
    let mut scheduler =
        Scheduler::with_cache_resources(worker, RadixCache::default(), CachePageAllocator::new(4));
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("local-dispatch"),
        vec![1, 2],
        SamplingParams::new(1),
    ));

    scheduler
        .dispatch_prefill_batch(1)
        .expect("local prefill should dispatch");

    let worker = scheduler.worker();
    assert_eq!(
        worker
            .last_transfer_summary()
            .expect("empty transfer summary should be recorded")
            .submitted_spans(),
        0
    );
    assert!(worker.transfer_executor().seen_rooms.is_empty());
}

#[test]
fn transfer_model_worker_propagates_transfer_failure_and_scheduler_releases_pages() {
    let worker = KvTransferModelWorker::new(
        FinishedWorker,
        registry_with_session("pd-transfer-fail", 16),
        RecordingTransferExecutor {
            fail_room: Some(16),
            ..Default::default()
        },
    );
    let mut scheduler =
        Scheduler::with_cache_resources(worker, RadixCache::default(), CachePageAllocator::new(3));
    scheduler.enqueue(
        ScheduledRequest::new(
            RequestId::from("pd-transfer-fail"),
            vec![1, 2],
            SamplingParams::new(1),
        )
        .with_disaggregated_params(Some(disaggregated_params(16))),
    );

    let error = scheduler
        .dispatch_prefill_batch(1)
        .expect_err("transfer failure should fail prefill dispatch");

    assert_eq!(
        error,
        SchedulerError::Worker(WorkerExecutionError::Runtime(
            "KV transfer execution failed: KV cache transfer runtime error: submit failed for room 16"
                .to_string()
        ))
    );
    assert_eq!(scheduler.available_cache_pages(), Some(3));
    assert_eq!(
        scheduler
            .worker()
            .registry()
            .get(16)
            .expect("bootstrap session should remain")
            .status(),
        KvPoll::Failed
    );
}

#[test]
fn transfer_model_worker_blocks_default_decode_dispatch_until_kv_success() {
    let worker = KvTransferModelWorker::new(
        UnfinishedWorker,
        registry_with_session("pd-decode-wait", 19),
        MooncakeKvCacheTransferExecutor::new(
            RecordingMooncakeSubmitter::default(),
            MooncakeKvCacheLayout {
                source_base_addr: 0x4000,
                page_size_bytes: 64,
                target_base_offset: 0,
            },
            MooncakeTransferTarget { target_id: 9 },
        ),
    );
    let mut scheduler =
        Scheduler::with_cache_resources(worker, RadixCache::default(), CachePageAllocator::new(3));
    scheduler.enqueue(
        ScheduledRequest::new(
            RequestId::from("pd-decode-wait"),
            vec![1, 2],
            SamplingParams::new(2),
        )
        .with_disaggregated_params(Some(disaggregated_params(19))),
    );
    scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should submit async KV transfer");

    let error = scheduler
        .dispatch_decode_batch(1)
        .expect_err("decode should wait for async KV transfer");

    assert_eq!(
        error,
        SchedulerError::DecodeNotReady {
            request_id: RequestId::from("pd-decode-wait")
        }
    );
    assert_eq!(scheduler.decode_queue_depth(), 1);

    scheduler
        .worker_mut()
        .registry_mut()
        .update_status(19, KvPoll::Success)
        .expect("status should update");
    let outputs = scheduler
        .dispatch_decode_batch(1)
        .expect("decode should dispatch after KV success");

    assert_eq!(outputs[0].request_id, RequestId::from("pd-decode-wait"));
    assert_eq!(outputs[0].token_ids, vec![1]);
}

#[test]
fn transfer_model_worker_fails_default_decode_dispatch_when_kv_failed() {
    let worker = KvTransferModelWorker::new(
        UnfinishedWorker,
        registry_with_session("pd-decode-failed", 20),
        MooncakeKvCacheTransferExecutor::new(
            RecordingMooncakeSubmitter::default(),
            MooncakeKvCacheLayout {
                source_base_addr: 0x5000,
                page_size_bytes: 64,
                target_base_offset: 0,
            },
            MooncakeTransferTarget { target_id: 10 },
        ),
    );
    let mut scheduler =
        Scheduler::with_cache_resources(worker, RadixCache::default(), CachePageAllocator::new(2));
    scheduler.enqueue(
        ScheduledRequest::new(
            RequestId::from("pd-decode-failed"),
            vec![1, 2],
            SamplingParams::new(2),
        )
        .with_disaggregated_params(Some(disaggregated_params(20))),
    );
    scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should submit async KV transfer");
    scheduler
        .worker_mut()
        .registry_mut()
        .update_status(20, KvPoll::Failed)
        .expect("status should update");

    let error = scheduler
        .dispatch_decode_batch(1)
        .expect_err("failed KV transfer should fail decode dispatch");

    assert_eq!(
        error,
        SchedulerError::Worker(WorkerExecutionError::Runtime(
            "KV transfer failed for bootstrap room 20".to_string()
        ))
    );
    assert_eq!(scheduler.decode_queue_depth(), 1);
}

#[test]
fn scheduler_abort_removes_pd_bootstrap_session_for_decode_request() {
    let worker = KvTransferModelWorker::new(
        UnfinishedWorker,
        DecodeBootstrapRegistry::default(),
        RecordingTransferExecutor::default(),
    );
    let mut scheduler =
        Scheduler::with_cache_resources(worker, RadixCache::default(), CachePageAllocator::new(2));
    scheduler.enqueue(
        ScheduledRequest::new(
            RequestId::from("pd-abort-cleanup"),
            vec![1, 2],
            SamplingParams::new(2),
        )
        .with_disaggregated_params(Some(disaggregated_params(31))),
    );
    scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should register bootstrap session and queue decode");
    assert!(scheduler.worker().registry().get(31).is_some());

    assert!(scheduler.abort_request(&RequestId::from("pd-abort-cleanup")));

    assert!(scheduler.worker().registry().get(31).is_none());
    assert_eq!(scheduler.decode_queue_depth(), 0);
}

#[test]
fn engine_token_generation_waits_when_pd_decode_kv_is_not_ready() {
    let worker = KvTransferModelWorker::new(
        UnfinishedWorker,
        registry_with_session("engine-pd-wait", 21),
        MooncakeKvCacheTransferExecutor::new(
            RecordingMooncakeSubmitter::default(),
            MooncakeKvCacheLayout {
                source_base_addr: 0x6000,
                page_size_bytes: 64,
                target_base_offset: 0,
            },
            MooncakeTransferTarget { target_id: 11 },
        ),
    );
    let scheduler =
        Scheduler::with_cache_resources(worker, RadixCache::default(), CachePageAllocator::new(3));
    let mut engine = Engine::new(ByteTokenizer::default(), scheduler);

    let error = engine
        .generate_tokens(TokenGenerateRequest {
            request_id: RequestId::from("engine-pd-wait"),
            input_ids: vec![1, 2],
            sampling: SamplingParams::new(2),
            disaggregated_params: Some(disaggregated_params(21)),
            data_parallel_rank: 0,
        })
        .expect_err("engine should surface pending KV transfer as decode not ready");

    assert!(matches!(
        error,
        RuntimeError::Scheduler(SchedulerError::DecodeNotReady { request_id })
            if request_id == RequestId::from("engine-pd-wait")
    ));
    assert_eq!(engine.scheduler().decode_queue_depth(), 1);
}

#[test]
fn engine_poll_transfers_updates_registry_and_unblocks_decode_dispatch() {
    let backend = RecordingMooncakeBackend::completed();
    let worker = KvTransferModelWorker::new(
        UnfinishedWorker,
        registry_with_session("engine-poll", 22),
        MooncakeKvCacheTransferExecutor::new(
            backend,
            MooncakeKvCacheLayout {
                source_base_addr: 0x7000,
                page_size_bytes: 64,
                target_base_offset: 0,
            },
            MooncakeTransferTarget { target_id: 12 },
        ),
    );
    let scheduler =
        Scheduler::with_cache_resources(worker, RadixCache::default(), CachePageAllocator::new(3));
    let mut engine = Engine::new(ByteTokenizer::default(), scheduler);

    let error = engine
        .generate_tokens(TokenGenerateRequest {
            request_id: RequestId::from("engine-poll"),
            input_ids: vec![1, 2],
            sampling: SamplingParams::new(2),
            disaggregated_params: Some(disaggregated_params(22)),
            data_parallel_rank: 0,
        })
        .expect_err("engine should wait for pending transfer");

    assert!(matches!(
        error,
        RuntimeError::Scheduler(SchedulerError::DecodeNotReady { .. })
    ));

    let summary = engine
        .poll_transfers()
        .expect("polling transfer should complete submitted batch");
    assert_eq!(summary.completed_batches(), 1);
    assert_eq!(summary.pending_batches(), 0);

    let outputs = engine
        .scheduler_mut()
        .dispatch_decode_batch(1)
        .expect("decode should dispatch after poll");
    assert_eq!(outputs[0].request_id, RequestId::from("engine-poll"));
    assert_eq!(outputs[0].token_ids, vec![1]);
}

#[test]
fn engine_token_generation_can_poll_transfer_and_continue_decode() {
    let backend = RecordingMooncakeBackend::completed();
    let worker = KvTransferModelWorker::new(
        UnfinishedWorker,
        registry_with_session("engine-poll-inline", 24),
        MooncakeKvCacheTransferExecutor::new(
            backend,
            MooncakeKvCacheLayout {
                source_base_addr: 0x9000,
                page_size_bytes: 64,
                target_base_offset: 0,
            },
            MooncakeTransferTarget { target_id: 14 },
        ),
    );
    let scheduler =
        Scheduler::with_cache_resources(worker, RadixCache::default(), CachePageAllocator::new(3));
    let mut engine = Engine::new(ByteTokenizer::default(), scheduler);

    let output = engine
        .generate_tokens_with_transfer_polling(
            TokenGenerateRequest {
                request_id: RequestId::from("engine-poll-inline"),
                input_ids: vec![1, 2],
                sampling: SamplingParams::new(2),
                disaggregated_params: Some(disaggregated_params(24)),
                data_parallel_rank: 0,
            },
            1,
        )
        .expect("engine should poll completed transfer and continue decode");

    assert_eq!(output.request_id, RequestId::from("engine-poll-inline"));
    assert_eq!(output.output_ids, vec![1, 1]);
    assert!(output.finished);
    assert_eq!(engine.scheduler().decode_queue_depth(), 0);
    assert!(engine.scheduler().worker().registry().get(24).is_none());
}

#[test]
fn router_runtime_poll_transfers_exposes_control_plane_counts() {
    let backend = RecordingMooncakeBackend::completed();
    let worker = KvTransferModelWorker::new(
        UnfinishedWorker,
        registry_with_session("router-poll", 23),
        MooncakeKvCacheTransferExecutor::new(
            backend,
            MooncakeKvCacheLayout {
                source_base_addr: 0x8000,
                page_size_bytes: 64,
                target_base_offset: 0,
            },
            MooncakeTransferTarget { target_id: 13 },
        ),
    );
    let scheduler =
        Scheduler::with_cache_resources(worker, RadixCache::default(), CachePageAllocator::new(3));
    let engine = Engine::new(ByteTokenizer::default(), scheduler);
    let mut runtime = RouterRuntime::new(engine);

    let error = runtime
        .generate_stream(sglang_srt::router::RouterGenerateRequest {
            request_id: "router-poll".to_string(),
            tokenized: Some(sglang_srt::router::RouterTokenizedInput {
                original_text: String::new(),
                input_ids: vec![1, 2],
            }),
            sampling_params: Some(sglang_srt::router::RouterSamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            disaggregated_params: Some(sglang_srt::router::RouterDisaggregatedParams {
                bootstrap_host: "10.0.0.7".to_string(),
                bootstrap_port: 8998,
                bootstrap_room: 23,
            }),
            stream: true,
            data_parallel_rank: 0,
            trace_headers: Default::default(),
        })
        .expect_err("router generation should wait for pending transfer");

    assert!(matches!(
        error,
        sglang_srt::router::RouterRuntimeError::Runtime(RuntimeError::Scheduler(
            SchedulerError::DecodeNotReady { .. }
        ))
    ));

    let response = runtime
        .poll_transfers()
        .expect("router should expose transfer polling");
    assert_eq!(
        response,
        RouterTransferPollResponse {
            completed_batches: 1,
            pending_batches: 0,
        }
    );
}

#[test]
fn router_runtime_stream_can_poll_transfer_and_continue_decode() {
    let backend = RecordingMooncakeBackend::completed();
    let worker = KvTransferModelWorker::new(
        UnfinishedWorker,
        registry_with_session("router-poll-inline", 25),
        MooncakeKvCacheTransferExecutor::new(
            backend,
            MooncakeKvCacheLayout {
                source_base_addr: 0xa000,
                page_size_bytes: 64,
                target_base_offset: 0,
            },
            MooncakeTransferTarget { target_id: 15 },
        ),
    );
    let scheduler =
        Scheduler::with_cache_resources(worker, RadixCache::default(), CachePageAllocator::new(3));
    let engine = Engine::new(ByteTokenizer::default(), scheduler);
    let mut runtime = RouterRuntime::new(engine);

    let responses = runtime
        .generate_stream_with_transfer_polling(
            sglang_srt::router::RouterGenerateRequest {
                request_id: "router-poll-inline".to_string(),
                tokenized: Some(sglang_srt::router::RouterTokenizedInput {
                    original_text: String::new(),
                    input_ids: vec![1, 2],
                }),
                sampling_params: Some(sglang_srt::router::RouterSamplingParams {
                    max_new_tokens: Some(2),
                    ..Default::default()
                }),
                disaggregated_params: Some(sglang_srt::router::RouterDisaggregatedParams {
                    bootstrap_host: "10.0.0.7".to_string(),
                    bootstrap_port: 8998,
                    bootstrap_room: 25,
                }),
                stream: true,
                data_parallel_rank: 0,
                trace_headers: Default::default(),
            },
            1,
        )
        .expect("router should poll completed transfer and continue streaming");

    assert_eq!(responses.len(), 2);
    assert!(matches!(
        responses[0].body,
        sglang_srt::router::RouterGenerateResponseBody::Chunk(_)
    ));
    assert!(matches!(
        responses[1].body,
        sglang_srt::router::RouterGenerateResponseBody::Complete(_)
    ));
}

#[test]
fn decode_batch_ready_check_keeps_pd_request_queued_until_kv_success() {
    let mut scheduler = Scheduler::new(UnfinishedWorker);
    scheduler.enqueue(
        ScheduledRequest::new(
            RequestId::from("decode-waits"),
            vec![1, 2],
            SamplingParams::new(2),
        )
        .with_disaggregated_params(Some(disaggregated_params(17))),
    );
    scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should queue decode");
    let mut registry = registry_with_session("decode-waits", 17);
    registry
        .update_status(17, KvPoll::Transferring)
        .expect("status should update");

    let error = scheduler
        .next_decode_batch_with_ready_check(1, |request| {
            is_decode_request_kv_ready(request, &registry).expect("registry lookup should succeed")
        })
        .expect_err("decode should wait while KV is transferring");

    assert_eq!(
        error,
        SchedulerError::DecodeNotReady {
            request_id: RequestId::from("decode-waits")
        }
    );
    assert_eq!(scheduler.decode_queue_depth(), 1);

    registry
        .update_status(17, KvPoll::Success)
        .expect("status should update");
    let outputs = scheduler
        .dispatch_decode_batch_with_ready_check(1, |request| {
            is_decode_request_kv_ready(request, &registry).expect("registry lookup should succeed")
        })
        .expect("decode should dispatch after KV success");

    assert_eq!(outputs[0].request_id, RequestId::from("decode-waits"));
    assert_eq!(outputs[0].token_ids, vec![1]);
}

#[test]
fn decode_kv_ready_check_reports_missing_bootstrap_session() {
    let mut scheduler = Scheduler::new(UnfinishedWorker);
    scheduler.enqueue(
        ScheduledRequest::new(
            RequestId::from("missing-session"),
            vec![1, 2],
            SamplingParams::new(2),
        )
        .with_disaggregated_params(Some(disaggregated_params(18))),
    );
    scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should queue decode");
    let registry = DecodeBootstrapRegistry::default();
    let mut ready_error = None;

    let error = scheduler
        .next_decode_batch_with_ready_check(1, |request| {
            match is_decode_request_kv_ready(request, &registry) {
                Ok(ready) => ready,
                Err(error) => {
                    ready_error = Some(error);
                    false
                }
            }
        })
        .expect_err("missing bootstrap session should block decode");

    assert_eq!(
        error,
        SchedulerError::DecodeNotReady {
            request_id: RequestId::from("missing-session")
        }
    );
    assert_eq!(
        ready_error,
        Some(KvCacheTransferError::Registry(
            sglang_srt::transfer::DecodeBootstrapRegistryError::MissingBootstrapRoom(18)
        ))
    );
    assert_eq!(scheduler.decode_queue_depth(), 1);
}

#[test]
fn transferable_kv_memory_builds_prefill_and_decode_mooncake_layouts() {
    let memory = TransferableKvCacheMemory::new(
        vec![TransferableKvCacheRegion {
            base_addr: 0x1000,
            byte_len: 4096,
            page_size_bytes: 128,
        }],
        128,
    )
    .expect("memory should be valid");

    assert_eq!(
        memory.prefill_layout(0x200),
        MooncakeKvCacheLayout {
            source_base_addr: 0x1000,
            page_size_bytes: 128,
            target_base_offset: 0x200,
        }
    );
    assert_eq!(
        memory.decode_remote_layout(&[4, 5]),
        MooncakeRemoteKvLayout {
            dst_kv_ptrs: vec![0x1000],
            dst_kv_indices: vec![4, 5],
            dst_kv_item_len: 128,
        }
    );
}

#[test]
fn transferable_kv_memory_rejects_zero_base_address() {
    let error = TransferableKvCacheMemory::new(
        vec![TransferableKvCacheRegion {
            base_addr: 0,
            byte_len: 128,
            page_size_bytes: 128,
        }],
        128,
    )
    .expect_err("zero address should be rejected");

    assert!(
        error.to_string().contains("base address must be non-zero"),
        "{error}"
    );
}

#[test]
fn mooncake_request_builder_maps_cache_pages_to_source_and_target_offsets() {
    let transfer_plan = transfer_plan_for_request("pd-mooncake", &[50, 51, 52], Some(1), 8);
    let span = &transfer_plan.spans()[0];

    let requests = build_mooncake_kv_transfer_requests(
        span,
        MooncakeKvCacheLayout {
            source_base_addr: 0x1000,
            page_size_bytes: 256,
            target_base_offset: 0x8000,
        },
        MooncakeTransferTarget { target_id: 42 },
    )
    .expect("mooncake requests should build");

    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].opcode, MooncakeOpcode::Write as i32);
    assert_eq!(requests[0].source as usize, 0x1000);
    assert_eq!(requests[0].target_id, 42);
    assert_eq!(requests[0].target_offset, 0x8000 + 256);
    assert_eq!(requests[0].length, 256);
    assert_eq!(requests[1].source as usize, 0x1000 + 256);
    assert_eq!(requests[1].target_offset, 0x8000 + 512);
}

#[test]
fn mooncake_remote_request_builder_uses_decode_kv_ptrs_and_indices() {
    let transfer_plan = transfer_plan_for_request("pd-mooncake-remote", &[50, 51, 52], Some(1), 8);
    let span = &transfer_plan.spans()[0];

    let requests = build_mooncake_remote_kv_transfer_requests(
        span,
        MooncakeKvCacheLayout {
            source_base_addr: 0x1000,
            page_size_bytes: 256,
            target_base_offset: 0,
        },
        MooncakeTransferTarget { target_id: 42 },
        &MooncakeRemoteKvLayout {
            dst_kv_ptrs: vec![0x9000],
            dst_kv_indices: vec![7, 8],
            dst_kv_item_len: 256,
        },
    )
    .expect("mooncake remote requests should build");

    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].source as usize, 0x1000);
    assert_eq!(requests[0].target_id, 42);
    assert_eq!(requests[0].target_offset, 0x9000 + 7 * 256);
    assert_eq!(requests[0].length, 256);
    assert_eq!(requests[1].source as usize, 0x1000 + 256);
    assert_eq!(requests[1].target_offset, 0x9000 + 8 * 256);
}

#[test]
fn mooncake_remote_request_builder_rejects_partial_page_layout() {
    let transfer_plan = transfer_plan_for_request("pd-mooncake-partial", &[50], None, 8);
    let span = &transfer_plan.spans()[0];

    let error = build_mooncake_remote_kv_transfer_requests(
        span,
        MooncakeKvCacheLayout {
            source_base_addr: 0x1000,
            page_size_bytes: 256,
            target_base_offset: 0,
        },
        MooncakeTransferTarget { target_id: 42 },
        &MooncakeRemoteKvLayout {
            dst_kv_ptrs: vec![0x9000],
            dst_kv_indices: vec![7],
            dst_kv_item_len: 128,
        },
    )
    .expect_err("partial remote KV layout should be rejected");

    assert_eq!(
        error.to_string(),
        "Mooncake remote KV item layout requires exactly 256 bytes but has 128 bytes"
    );
}

#[test]
fn mooncake_executor_submits_built_requests_through_transfer_submitter() {
    let transfer_plan = transfer_plan_for_request("pd-mooncake-submit", &[60, 61], None, 9);
    let mut registry = registry_with_session("pd-mooncake-submit", 9);
    let mut executor = MooncakeKvCacheTransferExecutor::new(
        RecordingMooncakeSubmitter::default(),
        MooncakeKvCacheLayout {
            source_base_addr: 0x2000,
            page_size_bytes: 128,
            target_base_offset: 0x9000,
        },
        MooncakeTransferTarget { target_id: 7 },
    );

    let summary = execute_kv_cache_transfer_plan(&mut registry, &mut executor, &transfer_plan)
        .expect("mooncake executor should submit");

    assert_eq!(summary.submitted_spans(), 1);
    assert_eq!(executor.submitted_batches(), &[100]);
    let submitted_requests = &executor.submitter().submitted_requests;
    assert_eq!(submitted_requests.len(), 1);
    assert_eq!(submitted_requests[0].len(), 2);
    assert_eq!(submitted_requests[0][0].source as usize, 0x2000);
    assert_eq!(submitted_requests[0][1].source as usize, 0x2000 + 128);
    assert_eq!(submitted_requests[0][0].target_offset, 0x9000);
    assert_eq!(submitted_requests[0][1].target_offset, 0x9000 + 128);
    assert_eq!(
        registry.get(9).expect("session should remain").status(),
        KvPoll::Transferring
    );

    let submitted_transfers = executor.submitted_transfers().to_vec();
    let mut reader = RecordingMooncakeStatusReader::completed();
    let poll_summary =
        poll_mooncake_transfer_batches(&mut registry, &mut reader, &submitted_transfers)
            .expect("completed status should update registry");

    assert_eq!(poll_summary.completed_batches(), 1);
    assert_eq!(poll_summary.pending_batches(), 0);
    assert_eq!(
        registry.get(9).expect("session should remain").status(),
        KvPoll::Success
    );
}

#[test]
fn mooncake_executor_uses_remote_kv_layout_for_bootstrap_room() {
    let transfer_plan =
        transfer_plan_for_request("pd-mooncake-remote-submit", &[60, 61, 62], Some(1), 9);
    let mut registry = registry_with_session("pd-mooncake-remote-submit", 9);
    let mut executor = MooncakeKvCacheTransferExecutor::with_remote_kv_layouts(
        RecordingMooncakeSubmitter::default(),
        MooncakeKvCacheLayout {
            source_base_addr: 0x2000,
            page_size_bytes: 128,
            target_base_offset: 0xdead_0000,
        },
        MooncakeTransferTarget { target_id: 7 },
        vec![(
            9,
            MooncakeRemoteKvLayout {
                dst_kv_ptrs: vec![0x9000],
                dst_kv_indices: vec![4, 5],
                dst_kv_item_len: 128,
            },
        )],
    );

    execute_kv_cache_transfer_plan(&mut registry, &mut executor, &transfer_plan)
        .expect("mooncake executor should submit with remote layout");

    let submitted_requests = &executor.submitter().submitted_requests;
    assert_eq!(submitted_requests.len(), 1);
    assert_eq!(submitted_requests[0].len(), 2);
    assert_eq!(submitted_requests[0][0].source as usize, 0x2000);
    assert_eq!(submitted_requests[0][0].target_offset, 0x9000 + 4 * 128);
    assert_eq!(submitted_requests[0][1].source as usize, 0x2000 + 128);
    assert_eq!(submitted_requests[0][1].target_offset, 0x9000 + 5 * 128);
}

#[test]
fn mooncake_executor_submits_remote_kv_layouts_for_each_session_in_room() {
    let transfer_plan = transfer_plan_for_request("pd-mooncake-remote-multi", &[60, 61], None, 9);
    let mut registry = registry_with_session("pd-mooncake-remote-multi", 9);
    let mut executor =
        MooncakeKvCacheTransferExecutor::with_target_resolver_and_remote_kv_session_layouts(
            RecordingMooncakeSubmitter::default(),
            MooncakeKvCacheLayout {
                source_base_addr: 0x2000,
                page_size_bytes: 128,
                target_base_offset: 0xdead_0000,
            },
            SessionTargetResolver {
                targets: vec![("session-a".to_string(), 7), ("session-b".to_string(), 8)],
            },
            vec![
                (
                    9,
                    "session-a".to_string(),
                    MooncakeRemoteKvLayout {
                        dst_kv_ptrs: vec![0x9000],
                        dst_kv_indices: vec![4, 5],
                        dst_kv_item_len: 128,
                    },
                ),
                (
                    9,
                    "session-b".to_string(),
                    MooncakeRemoteKvLayout {
                        dst_kv_ptrs: vec![0xa000],
                        dst_kv_indices: vec![6, 7],
                        dst_kv_item_len: 128,
                    },
                ),
            ],
        );

    execute_kv_cache_transfer_plan(&mut registry, &mut executor, &transfer_plan)
        .expect("mooncake executor should submit all remote session layouts");

    let submitted_requests = &executor.submitter().submitted_requests;
    assert_eq!(submitted_requests.len(), 1);
    assert_eq!(submitted_requests[0].len(), 4);
    assert_eq!(submitted_requests[0][0].target_id, 7);
    assert_eq!(submitted_requests[0][0].target_offset, 0x9000 + 4 * 128);
    assert_eq!(submitted_requests[0][1].target_id, 7);
    assert_eq!(submitted_requests[0][1].target_offset, 0x9000 + 5 * 128);
    assert_eq!(submitted_requests[0][2].target_id, 8);
    assert_eq!(submitted_requests[0][2].target_offset, 0xa000 + 6 * 128);
    assert_eq!(submitted_requests[0][3].target_id, 8);
    assert_eq!(submitted_requests[0][3].target_offset, 0xa000 + 7 * 128);
}

#[test]
fn mooncake_bootstrap_executor_refreshes_remote_layouts_before_submit() {
    let transfer_plan =
        transfer_plan_for_request("pd-mooncake-bootstrap-refresh", &[60, 61], None, 34);
    let mut registry = registry_with_session("pd-mooncake-bootstrap-refresh", 34);
    let bootstrap_service = PrefillBootstrapService::default();
    {
        let mut state = bootstrap_service
            .state()
            .lock()
            .expect("bootstrap state lock should be held");
        state
            .ingest_mooncake_bootstrap_frame(&kv_args_frame("session-a", &[0x9000], 128))
            .expect("KVArgs frame should parse");
        state
            .ingest_mooncake_bootstrap_frame(&transfer_metadata_frame(34, "session-a", &[4, 5]))
            .expect("transfer metadata frame should parse");
    }
    let inner = MooncakeKvCacheTransferExecutor::with_target_resolver(
        RecordingMooncakeSubmitter::default(),
        MooncakeKvCacheLayout {
            source_base_addr: 0x2000,
            page_size_bytes: 128,
            target_base_offset: 0xdead_0000,
        },
        SessionTargetResolver {
            targets: vec![("session-a".to_string(), 7)],
        },
    );
    let mut executor = MooncakeBootstrapKvCacheTransferExecutor::new(bootstrap_service, inner);

    execute_kv_cache_transfer_plan(&mut registry, &mut executor, &transfer_plan)
        .expect("bootstrap-backed Mooncake executor should submit");

    let submitted_requests = &executor.inner().submitter().submitted_requests;
    assert_eq!(submitted_requests.len(), 1);
    assert_eq!(submitted_requests[0].len(), 2);
    assert_eq!(submitted_requests[0][0].target_id, 7);
    assert_eq!(submitted_requests[0][0].target_offset, 0x9000 + 4 * 128);
    assert_eq!(submitted_requests[0][1].target_id, 7);
    assert_eq!(submitted_requests[0][1].target_offset, 0x9000 + 5 * 128);
}

#[test]
fn mooncake_bootstrap_executor_waits_for_delayed_remote_layouts_before_submit() {
    let transfer_plan =
        transfer_plan_for_request("pd-mooncake-bootstrap-wait", &[62, 63], None, 35);
    let mut registry = registry_with_session("pd-mooncake-bootstrap-wait", 35);
    let bootstrap_service = PrefillBootstrapService::default();
    let delayed_service = bootstrap_service.clone();
    let delayed_insert = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(25));
        let mut state = delayed_service
            .state()
            .lock()
            .expect("bootstrap state lock should be held");
        state
            .ingest_mooncake_bootstrap_frame(&kv_args_frame("session-delayed", &[0xb000], 128))
            .expect("KVArgs frame should parse");
        state
            .ingest_mooncake_bootstrap_frame(&transfer_metadata_frame(
                35,
                "session-delayed",
                &[6, 7],
            ))
            .expect("transfer metadata frame should parse");
    });
    let inner = MooncakeKvCacheTransferExecutor::with_target_resolver(
        RecordingMooncakeSubmitter::default(),
        MooncakeKvCacheLayout {
            source_base_addr: 0x2200,
            page_size_bytes: 128,
            target_base_offset: 0xdead_0000,
        },
        SessionTargetResolver {
            targets: vec![("session-delayed".to_string(), 8)],
        },
    );
    let mut executor = MooncakeBootstrapKvCacheTransferExecutor::new(bootstrap_service, inner)
        .with_metadata_wait_timeout(Duration::from_secs(1));

    execute_kv_cache_transfer_plan(&mut registry, &mut executor, &transfer_plan)
        .expect("bootstrap-backed Mooncake executor should wait for delayed metadata");
    delayed_insert
        .join()
        .expect("delayed bootstrap metadata insert should join");

    let submitted_requests = &executor.inner().submitter().submitted_requests;
    assert_eq!(submitted_requests.len(), 1);
    assert_eq!(submitted_requests[0].len(), 2);
    assert_eq!(submitted_requests[0][0].target_id, 8);
    assert_eq!(submitted_requests[0][0].target_offset, 0xb000 + 6 * 128);
    assert_eq!(submitted_requests[0][1].target_id, 8);
    assert_eq!(submitted_requests[0][1].target_offset, 0xb000 + 7 * 128);
}

#[test]
fn mooncake_executor_clears_completed_submitted_transfers_after_poll() {
    let transfer_plan = transfer_plan_for_request("pd-mooncake-cleanup", &[61, 62], None, 28);
    let mut registry = registry_with_session("pd-mooncake-cleanup", 28);
    let mut executor = MooncakeKvCacheTransferExecutor::new(
        RecordingMooncakeBackend::completed(),
        MooncakeKvCacheLayout {
            source_base_addr: 0x2100,
            page_size_bytes: 128,
            target_base_offset: 0x9100,
        },
        MooncakeTransferTarget { target_id: 8 },
    );
    execute_kv_cache_transfer_plan(&mut registry, &mut executor, &transfer_plan)
        .expect("mooncake executor should submit");
    assert_eq!(executor.submitted_transfers().len(), 1);

    let summary = executor
        .poll_submitted_transfers(&mut registry)
        .expect("completed transfer should poll successfully");

    assert_eq!(summary.completed_batches(), 1);
    assert!(executor.submitted_transfers().is_empty());
    assert!(executor.submitted_batches().is_empty());
    assert_eq!(executor.submitter().freed_batches, vec![300]);
}

#[test]
fn mooncake_executor_keeps_pending_submitted_transfers_after_poll() {
    let transfer_plan =
        transfer_plan_for_request("pd-mooncake-pending-cleanup", &[63, 64], None, 29);
    let mut registry = registry_with_session("pd-mooncake-pending-cleanup", 29);
    let mut executor = MooncakeKvCacheTransferExecutor::new(
        RecordingMooncakeBackend::pending(),
        MooncakeKvCacheLayout {
            source_base_addr: 0x2200,
            page_size_bytes: 128,
            target_base_offset: 0x9200,
        },
        MooncakeTransferTarget { target_id: 8 },
    );
    execute_kv_cache_transfer_plan(&mut registry, &mut executor, &transfer_plan)
        .expect("mooncake executor should submit");

    let summary = executor
        .poll_submitted_transfers(&mut registry)
        .expect("pending transfer should poll successfully");

    assert_eq!(summary.pending_batches(), 1);
    assert_eq!(executor.submitted_batches(), &[300]);
    assert_eq!(executor.submitted_transfers().len(), 1);
    assert!(executor.submitter().freed_batches.is_empty());
}

#[test]
fn mooncake_executor_clears_failed_submitted_transfers_after_poll_error() {
    let transfer_plan =
        transfer_plan_for_request("pd-mooncake-failed-cleanup", &[65, 66], None, 30);
    let mut registry = registry_with_session("pd-mooncake-failed-cleanup", 30);
    let mut executor = MooncakeKvCacheTransferExecutor::new(
        RecordingMooncakeBackend::failed(),
        MooncakeKvCacheLayout {
            source_base_addr: 0x2300,
            page_size_bytes: 128,
            target_base_offset: 0x9300,
        },
        MooncakeTransferTarget { target_id: 8 },
    );
    execute_kv_cache_transfer_plan(&mut registry, &mut executor, &transfer_plan)
        .expect("mooncake executor should submit");

    let error = executor
        .poll_submitted_transfers(&mut registry)
        .expect_err("failed transfer should return polling error");

    assert_eq!(
        error,
        KvCacheTransferError::Runtime(
            "Mooncake transfer batch 300 task 0 failed with status 6".to_string()
        )
    );
    assert!(executor.submitted_transfers().is_empty());
    assert!(executor.submitted_batches().is_empty());
    assert_eq!(executor.submitter().freed_batches, vec![300]);
    assert_eq!(
        registry.get(30).expect("session should remain").status(),
        KvPoll::Failed
    );
}

#[test]
fn mooncake_executor_maps_request_build_errors_to_transfer_runtime_errors() {
    let transfer_plan = transfer_plan_for_request("pd-bad-layout", &[70], None, 10);
    let mut registry = registry_with_session("pd-bad-layout", 10);
    let mut executor = MooncakeKvCacheTransferExecutor::new(
        RecordingMooncakeSubmitter::default(),
        MooncakeKvCacheLayout {
            source_base_addr: 0x2000,
            page_size_bytes: 0,
            target_base_offset: 0x9000,
        },
        MooncakeTransferTarget { target_id: 7 },
    );

    let error = execute_kv_cache_transfer_plan(&mut registry, &mut executor, &transfer_plan)
        .expect_err("bad layout should fail before submit");

    assert_eq!(
        error,
        KvCacheTransferError::Runtime("Mooncake KV page size must be non-zero".to_string())
    );
    assert!(executor.submitter().submitted_requests.is_empty());
    assert_eq!(
        registry.get(10).expect("session should remain").status(),
        KvPoll::Failed
    );
}

#[test]
fn mooncake_executor_resolves_target_per_bootstrap_room() {
    let mut scheduler = Scheduler::with_cache_resources(
        FinishedWorker,
        RadixCache::default(),
        CachePageAllocator::new(4),
    );
    scheduler.enqueue(
        ScheduledRequest::new(
            RequestId::from("room-a"),
            vec![1, 2],
            SamplingParams::new(1),
        )
        .with_disaggregated_params(Some(disaggregated_params(11))),
    );
    scheduler.enqueue(
        ScheduledRequest::new(
            RequestId::from("room-b"),
            vec![3, 4],
            SamplingParams::new(1),
        )
        .with_disaggregated_params(Some(disaggregated_params(12))),
    );
    let batch = scheduler
        .next_prefill_batch(2)
        .expect("prefill batch should be available");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&batch);
    let transfer_plan =
        KvCacheTransferPlan::from_prefill_worker_batch(&worker_batch).expect("plan should build");
    let mut registry = DecodeBootstrapRegistry::default();
    registry
        .register(DecodeBootstrapSession::new(
            RequestId::from("room-a"),
            disaggregated_params(11),
            0,
        ))
        .expect("room-a should register");
    registry
        .register(DecodeBootstrapSession::new(
            RequestId::from("room-b"),
            disaggregated_params(12),
            0,
        ))
        .expect("room-b should register");
    let mut executor = MooncakeKvCacheTransferExecutor::with_target_resolver(
        RecordingMooncakeSubmitter::default(),
        MooncakeKvCacheLayout {
            source_base_addr: 0x3000,
            page_size_bytes: 64,
            target_base_offset: 0,
        },
        RoomTargetResolver {
            targets: vec![(11, 101), (12, 202)],
        },
    );

    let summary = execute_kv_cache_transfer_plan(&mut registry, &mut executor, &transfer_plan)
        .expect("mooncake executor should submit both rooms");

    assert_eq!(summary.submitted_spans(), 2);
    assert_eq!(executor.submitted_transfers().len(), 2);
    assert_eq!(
        executor
            .submitter()
            .submitted_requests
            .iter()
            .map(|requests| requests[0].target_id)
            .collect::<Vec<_>>(),
        vec![101, 202]
    );
}

#[test]
fn mooncake_session_target_resolver_opens_segment_once_per_session() {
    let mut scheduler = Scheduler::with_cache_resources(
        FinishedWorker,
        RadixCache::default(),
        CachePageAllocator::new(4),
    );
    scheduler.enqueue(
        ScheduledRequest::new(
            RequestId::from("session-room"),
            vec![1, 2],
            SamplingParams::new(1),
        )
        .with_disaggregated_params(Some(disaggregated_params(33))),
    );
    let batch = scheduler
        .next_prefill_batch(1)
        .expect("prefill batch should be available");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&batch);
    let transfer_plan =
        KvCacheTransferPlan::from_prefill_worker_batch(&worker_batch).expect("plan should build");
    let mut registry = DecodeBootstrapRegistry::default();
    registry
        .register(DecodeBootstrapSession::new(
            RequestId::from("session-room"),
            disaggregated_params(33),
            0,
        ))
        .expect("session should register");
    let mut executor = MooncakeKvCacheTransferExecutor::with_target_resolver(
        RecordingMooncakeSubmitter::default(),
        MooncakeKvCacheLayout {
            source_base_addr: 0x5000,
            page_size_bytes: 64,
            target_base_offset: 0,
        },
        MooncakeSessionTargetResolver::new(
            RecordingSegmentOpener::new(vec![("decode-session-a".to_string(), 404)]),
            vec![(33, "decode-session-a".to_string())],
        ),
    );

    let first_summary =
        execute_kv_cache_transfer_plan(&mut registry, &mut executor, &transfer_plan)
            .expect("first transfer should open segment and submit");
    let second_summary =
        execute_kv_cache_transfer_plan(&mut registry, &mut executor, &transfer_plan)
            .expect("second transfer should reuse cached target and submit");

    assert_eq!(first_summary.submitted_spans(), 1);
    assert_eq!(second_summary.submitted_spans(), 1);
    assert_eq!(
        executor
            .submitter()
            .submitted_requests
            .iter()
            .map(|requests| requests[0].target_id)
            .collect::<Vec<_>>(),
        vec![404, 404]
    );
    assert_eq!(
        executor.target_resolver().opener().opened_segments(),
        &["decode-session-a".to_string()]
    );
}

#[test]
fn poll_mooncake_transfer_batches_reports_pending_without_changing_status() {
    let mut registry = registry_with_session("pd-pending", 13);
    registry
        .update_status(13, KvPoll::Transferring)
        .expect("status should update");
    let mut reader = RecordingMooncakeStatusReader::pending();
    let batches = vec![MooncakeSubmittedBatch::new(13, 200, 2)];

    let summary = poll_mooncake_transfer_batches(&mut registry, &mut reader, &batches)
        .expect("pending status should not fail");

    assert_eq!(summary.completed_batches(), 0);
    assert_eq!(summary.pending_batches(), 1);
    assert_eq!(
        registry.get(13).expect("session should remain").status(),
        KvPoll::Transferring
    );
}

#[test]
fn poll_mooncake_transfer_batches_marks_failed_status_and_returns_error() {
    let mut registry = registry_with_session("pd-status-fail", 14);
    registry
        .update_status(14, KvPoll::Transferring)
        .expect("status should update");
    let mut reader = RecordingMooncakeStatusReader {
        statuses: vec![
            MooncakeTransferStatusCode::Completed,
            MooncakeTransferStatusCode::Failed,
        ],
    };
    let batches = vec![MooncakeSubmittedBatch::new(14, 201, 2)];

    let error = poll_mooncake_transfer_batches(&mut registry, &mut reader, &batches)
        .expect_err("failed status should propagate");

    assert_eq!(
        error,
        KvCacheTransferError::Runtime(
            "Mooncake transfer batch 201 task 1 failed with status 6".to_string()
        )
    );
    assert_eq!(
        registry.get(14).expect("session should remain").status(),
        KvPoll::Failed
    );
}

#[derive(Default)]
struct RecordingTransferExecutor {
    seen_rooms: Vec<BootstrapRoom>,
    fail_room: Option<BootstrapRoom>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PublishedDecodeBootstrapSpan {
    request_id: RequestId,
    bootstrap_addr: String,
    bootstrap_room: BootstrapRoom,
    prefill_dp_rank: i32,
    dst_kv_indices: Vec<i32>,
    decode_prefix_len: Option<usize>,
}

#[derive(Default)]
struct RecordingDecodeBootstrapPublisher {
    published: Vec<PublishedDecodeBootstrapSpan>,
}

impl DecodeBootstrapPublisher for RecordingDecodeBootstrapPublisher {
    fn publish_decode_bootstrap_metadata(
        &mut self,
        plan: &KvCacheTransferPlan,
    ) -> Result<DecodeBootstrapMetadataPublishSummary, String> {
        for span in plan.spans() {
            self.published.push(PublishedDecodeBootstrapSpan {
                request_id: span.request_id().clone(),
                bootstrap_addr: format!(
                    "{}:{}",
                    span.disaggregated_params().bootstrap_host,
                    span.disaggregated_params().bootstrap_port
                ),
                bootstrap_room: span.bootstrap_room(),
                prefill_dp_rank: span.data_parallel_rank(),
                dst_kv_indices: span
                    .cache_pages()
                    .iter()
                    .map(|page| page.as_usize() as i32)
                    .collect(),
                decode_prefix_len: Some(span.token_offset() + span.token_count()),
            });
        }

        Ok(DecodeBootstrapMetadataPublishSummary {
            published_spans: plan.len(),
        })
    }
}

impl KvCacheTransferExecutor for RecordingTransferExecutor {
    fn transfer_span(&mut self, span: &KvCacheTransferSpan) -> Result<(), KvCacheTransferError> {
        self.seen_rooms.push(span.bootstrap_room());
        if self.fail_room == Some(span.bootstrap_room()) {
            return Err(KvCacheTransferError::Runtime(format!(
                "submit failed for room {}",
                span.bootstrap_room()
            )));
        }

        Ok(())
    }
}

#[derive(Default)]
struct RecordingMooncakeSubmitter {
    submitted_requests: Vec<Vec<MooncakeTransferRequest>>,
    freed_batches: Vec<MooncakeBatchId>,
}

impl MooncakeTransferSubmitter for RecordingMooncakeSubmitter {
    fn submit_transfer(
        &mut self,
        requests: &mut [MooncakeTransferRequest],
    ) -> Result<MooncakeBatchId, MooncakeError> {
        self.submitted_requests.push(requests.to_vec());
        Ok(100 + self.submitted_requests.len() as MooncakeBatchId - 1)
    }
}

impl MooncakeTransferStatusReader for RecordingMooncakeSubmitter {
    fn transfer_status(
        &mut self,
        _batch_id: MooncakeBatchId,
        _task_id: usize,
    ) -> Result<MooncakeTransferStatus, MooncakeError> {
        Ok(MooncakeTransferStatus {
            status: MooncakeTransferStatusCode::Completed as i32,
            transferred_bytes: 0,
        })
    }
}

impl MooncakeBatchReleaser for RecordingMooncakeSubmitter {
    fn free_batch(&mut self, batch_id: MooncakeBatchId) -> Result<(), MooncakeError> {
        self.freed_batches.push(batch_id);
        Ok(())
    }
}

#[derive(Default)]
struct RecordingMooncakeBackend {
    submitted_requests: Vec<Vec<MooncakeTransferRequest>>,
    statuses: Vec<MooncakeTransferStatusCode>,
    freed_batches: Vec<MooncakeBatchId>,
}

impl RecordingMooncakeBackend {
    fn completed() -> Self {
        Self {
            submitted_requests: Vec::new(),
            statuses: vec![MooncakeTransferStatusCode::Completed],
            freed_batches: Vec::new(),
        }
    }

    fn pending() -> Self {
        Self {
            submitted_requests: Vec::new(),
            statuses: vec![MooncakeTransferStatusCode::Pending],
            freed_batches: Vec::new(),
        }
    }

    fn failed() -> Self {
        Self {
            submitted_requests: Vec::new(),
            statuses: vec![MooncakeTransferStatusCode::Failed],
            freed_batches: Vec::new(),
        }
    }
}

impl MooncakeTransferSubmitter for RecordingMooncakeBackend {
    fn submit_transfer(
        &mut self,
        requests: &mut [MooncakeTransferRequest],
    ) -> Result<MooncakeBatchId, MooncakeError> {
        self.submitted_requests.push(requests.to_vec());
        Ok(300 + self.submitted_requests.len() as MooncakeBatchId - 1)
    }
}

impl MooncakeTransferStatusReader for RecordingMooncakeBackend {
    fn transfer_status(
        &mut self,
        _batch_id: MooncakeBatchId,
        task_id: usize,
    ) -> Result<MooncakeTransferStatus, MooncakeError> {
        let status = self
            .statuses
            .get(task_id)
            .or_else(|| self.statuses.last())
            .copied()
            .expect("recording Mooncake backend needs at least one status");
        Ok(MooncakeTransferStatus {
            status: status as i32,
            transferred_bytes: 0,
        })
    }
}

impl MooncakeBatchReleaser for RecordingMooncakeBackend {
    fn free_batch(&mut self, batch_id: MooncakeBatchId) -> Result<(), MooncakeError> {
        self.freed_batches.push(batch_id);
        Ok(())
    }
}

struct RoomTargetResolver {
    targets: Vec<(BootstrapRoom, i32)>,
}

struct SessionTargetResolver {
    targets: Vec<(String, i32)>,
}

struct RecordingSegmentOpener {
    targets: Vec<(String, i32)>,
    opened_segments: Vec<String>,
}

impl RecordingSegmentOpener {
    fn new(targets: Vec<(String, i32)>) -> Self {
        Self {
            targets,
            opened_segments: Vec::new(),
        }
    }

    fn opened_segments(&self) -> &[String] {
        &self.opened_segments
    }
}

impl sglang_srt::transfer::MooncakeSegmentOpener for RecordingSegmentOpener {
    fn open_segment(&mut self, segment: &str) -> Result<i32, MooncakeError> {
        self.opened_segments.push(segment.to_string());
        self.targets
            .iter()
            .find(|(candidate, _)| candidate == segment)
            .map(|(_, target_id)| *target_id)
            .ok_or_else(|| MooncakeError::OpenSegmentFailed(segment.to_string()))
    }
}

impl MooncakeTransferTargetResolver for RoomTargetResolver {
    fn resolve_target(
        &mut self,
        span: &KvCacheTransferSpan,
    ) -> Result<MooncakeTransferTarget, KvCacheTransferError> {
        self.targets
            .iter()
            .find(|(room, _)| *room == span.bootstrap_room())
            .map(|(_, target_id)| MooncakeTransferTarget {
                target_id: *target_id,
            })
            .ok_or_else(|| {
                KvCacheTransferError::Runtime(format!(
                    "missing target for room {}",
                    span.bootstrap_room()
                ))
            })
    }
}

impl MooncakeTransferTargetResolver for SessionTargetResolver {
    fn resolve_target(
        &mut self,
        span: &KvCacheTransferSpan,
    ) -> Result<MooncakeTransferTarget, KvCacheTransferError> {
        Err(KvCacheTransferError::Runtime(format!(
            "room-only target resolution is not valid for room {}",
            span.bootstrap_room()
        )))
    }

    fn resolve_session_target(
        &mut self,
        _span: &KvCacheTransferSpan,
        session_id: &str,
    ) -> Result<MooncakeTransferTarget, KvCacheTransferError> {
        self.targets
            .iter()
            .find(|(session, _)| session == session_id)
            .map(|(_, target_id)| MooncakeTransferTarget {
                target_id: *target_id,
            })
            .ok_or_else(|| {
                KvCacheTransferError::Runtime(format!("missing target for session {session_id}"))
            })
    }
}

struct RecordingMooncakeStatusReader {
    statuses: Vec<MooncakeTransferStatusCode>,
}

impl RecordingMooncakeStatusReader {
    fn completed() -> Self {
        Self {
            statuses: vec![MooncakeTransferStatusCode::Completed],
        }
    }

    fn pending() -> Self {
        Self {
            statuses: vec![MooncakeTransferStatusCode::Pending],
        }
    }
}

impl MooncakeTransferStatusReader for RecordingMooncakeStatusReader {
    fn transfer_status(
        &mut self,
        _batch_id: MooncakeBatchId,
        task_id: usize,
    ) -> Result<MooncakeTransferStatus, MooncakeError> {
        let status = self
            .statuses
            .get(task_id)
            .or_else(|| self.statuses.last())
            .copied()
            .expect("recording status reader needs at least one status");
        Ok(MooncakeTransferStatus {
            status: status as i32,
            transferred_bytes: 0,
        })
    }
}

fn transfer_plan_for_request(
    request_id: &str,
    input_ids: &[u32],
    cached_prefix_len: Option<usize>,
    bootstrap_room: BootstrapRoom,
) -> KvCacheTransferPlan {
    let mut prefix_cache = RadixCache::default();
    if let Some(cached_prefix_len) = cached_prefix_len {
        let prefix_tokens = &input_ids[..cached_prefix_len];
        let prefix_pages = (0..cached_prefix_len)
            .map(|index| CachePageId::from(100 + index))
            .collect::<Vec<_>>();
        prefix_cache
            .insert(prefix_tokens, &prefix_pages)
            .expect("prefix cache should insert");
    }

    let mut scheduler = Scheduler::with_cache_resources(
        FinishedWorker,
        prefix_cache,
        CachePageAllocator::new(input_ids.len()),
    );
    scheduler.enqueue(
        ScheduledRequest::new(
            RequestId::from(request_id),
            input_ids.to_vec(),
            SamplingParams::new(1),
        )
        .with_disaggregated_params(Some(disaggregated_params(bootstrap_room))),
    );

    let batch = scheduler
        .next_prefill_batch(1)
        .expect("prefill batch should be available");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&batch);
    KvCacheTransferPlan::from_prefill_worker_batch(&worker_batch)
        .expect("transfer plan should build")
}

fn registry_with_session(
    request_id: &str,
    bootstrap_room: BootstrapRoom,
) -> DecodeBootstrapRegistry {
    let mut registry = DecodeBootstrapRegistry::default();
    registry
        .register(DecodeBootstrapSession::new(
            RequestId::from(request_id),
            disaggregated_params(bootstrap_room),
            0,
        ))
        .expect("session should register");
    registry
}

fn kv_args_frame(session_id: &str, dst_kv_ptrs: &[u64], dst_kv_item_len: usize) -> Vec<Vec<u8>> {
    vec![
        b"None".to_vec(),
        b"10.0.0.9".to_vec(),
        b"41001".to_vec(),
        session_id.as_bytes().to_vec(),
        pack_u64s(dst_kv_ptrs),
        pack_u64s(&[]),
        pack_list_of_buffers(&[]),
        b"1".to_vec(),
        b"8".to_vec(),
        dst_kv_item_len.to_string().into_bytes(),
    ]
}

fn transfer_metadata_frame(
    room: BootstrapRoom,
    session_id: &str,
    dst_kv_indices: &[i32],
) -> Vec<Vec<u8>> {
    vec![
        room.to_string().into_bytes(),
        b"10.0.0.9".to_vec(),
        b"41001".to_vec(),
        session_id.as_bytes().to_vec(),
        pack_i32s(dst_kv_indices),
        b"11".to_vec(),
        pack_list_of_buffers(&[]),
        b"1".to_vec(),
        b"64".to_vec(),
    ]
}

fn pack_u64s(values: &[u64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn pack_i32s(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn pack_list_of_buffers(buffers: &[Vec<u8>]) -> Vec<u8> {
    let mut packed = Vec::new();
    packed.extend_from_slice(&(buffers.len() as u32).to_le_bytes());
    for buffer in buffers {
        packed.extend_from_slice(&(buffer.len() as u32).to_le_bytes());
        packed.extend_from_slice(buffer);
    }
    packed
}

fn disaggregated_params(bootstrap_room: BootstrapRoom) -> DisaggregatedParams {
    DisaggregatedParams {
        bootstrap_host: "10.0.0.7".to_string(),
        bootstrap_port: 8998,
        bootstrap_room,
    }
}
