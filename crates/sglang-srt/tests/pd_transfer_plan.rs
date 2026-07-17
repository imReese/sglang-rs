#![cfg(feature = "test-support")]

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use sglang_srt::cache::{CachePageAllocator, CachePageId, RadixCache};
use sglang_srt::model_executor::ModelWorkerBatch;
use sglang_srt::scheduler::{ScheduleBatch, ScheduledRequest, Scheduler};
use sglang_srt::transfer::{
    DecodeBootstrapRegistry, DecodeBootstrapSession, KvCacheMemoryLocation, KvCacheTransferError,
    KvCacheTransferPlan, KvPoll, KvTransferBackend, KvTransferCompletionEvent, KvTransferPoll,
    MooncakeBatchId, MooncakeBatchReleaser, MooncakeBufferEntry, MooncakeError,
    MooncakeKvCacheLayout, MooncakeKvCacheTransferExecutor, MooncakeMemoryRegistrar,
    MooncakeTransferRequest, MooncakeTransferStatus, MooncakeTransferStatusCode,
    MooncakeTransferStatusReader, MooncakeTransferSubmitter, MooncakeTransferTarget,
    TransferableKvCacheMemory, TransferableKvCacheRegion, apply_transfer_completion_events,
    execute_kv_cache_transfer_plan,
};
use sglang_srt::types::{DisaggregatedParams, RequestId, SamplingParams};
use sglang_srt::worker::{BatchGeneratedTokens, GeneratedToken, ModelWorker};

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
        .expect("worker output should match batch")
    }
}

#[derive(Default)]
struct RecordingBackend {
    pending: Vec<(u64, String)>,
    registered: bool,
    canceled: Vec<u64>,
    shutdown: bool,
}

impl KvTransferBackend for RecordingBackend {
    fn register(&mut self, _memory: TransferableKvCacheMemory) -> Result<(), KvCacheTransferError> {
        self.registered = true;
        Ok(())
    }

    fn submit(
        &mut self,
        span: &sglang_srt::transfer::KvCacheTransferSpan,
    ) -> Result<(), KvCacheTransferError> {
        self.pending
            .push((span.bootstrap_room(), span.descriptor_checksum()));
        Ok(())
    }

    fn poll(&mut self) -> Result<KvTransferPoll, KvCacheTransferError> {
        let events = self
            .pending
            .drain(..)
            .map(|(room, checksum)| KvTransferCompletionEvent::completed(room, checksum))
            .collect();
        Ok(KvTransferPoll::new(events, 0))
    }

    fn cancel(&mut self, bootstrap_room: u64) -> Result<(), KvCacheTransferError> {
        self.pending.retain(|(room, _)| *room != bootstrap_room);
        self.canceled.push(bootstrap_room);
        Ok(())
    }

    fn shutdown(&mut self) -> Result<(), KvCacheTransferError> {
        self.pending.clear();
        self.shutdown = true;
        Ok(())
    }
}

#[test]
fn scheduler_builds_transfer_delta_from_uncached_tokens() {
    let mut prefix_cache = RadixCache::default();
    prefix_cache
        .insert(&[10, 11], &[CachePageId::from(20), CachePageId::from(21)])
        .expect("prefix should insert");
    let mut scheduler =
        Scheduler::with_cache_resources(FinishedWorker, prefix_cache, CachePageAllocator::new(8));
    scheduler.enqueue(
        ScheduledRequest::new(
            RequestId::from("request"),
            vec![10, 11, 12, 13],
            SamplingParams::new(1),
        )
        .with_disaggregated_params(Some(disaggregated_params(7))),
    );

    let batch = scheduler
        .next_prefill_batch(1)
        .expect("prefill batch should exist");
    let plan = KvCacheTransferPlan::from_prefill_worker_batch(
        &ModelWorkerBatch::from_schedule_batch(&batch),
    )
    .expect("transfer plan should build");

    let span = &plan.spans()[0];
    assert_eq!(span.token_offset(), 2);
    assert_eq!(span.token_count(), 2);
    assert_eq!(span.cache_pages().len(), 2);
}

#[test]
fn completion_events_are_applied_by_orchestration_not_backend_poll() {
    let plan = simple_plan(8);
    let mut registry = registry_with_session(8);
    let mut backend = RecordingBackend::default();

    execute_kv_cache_transfer_plan(&mut registry, &mut backend, &plan)
        .expect("submit should succeed");
    assert_eq!(
        registry.get(8).expect("session").status(),
        KvPoll::Transferring
    );

    let poll = backend.poll().expect("poll should succeed");
    assert_eq!(
        registry.get(8).expect("session").status(),
        KvPoll::Transferring
    );
    let summary = apply_transfer_completion_events(&mut registry, poll)
        .expect("orchestration should apply event");

    assert_eq!(summary.completed_batches(), 1);
    assert_eq!(registry.get(8).expect("session").status(), KvPoll::Success);
}

#[test]
fn explicit_cancel_and_shutdown_are_observable() {
    let mut backend = RecordingBackend::default();
    backend.cancel(9).expect("cancel should succeed");
    backend.shutdown().expect("shutdown should succeed");

    assert_eq!(backend.canceled, vec![9]);
    assert!(backend.shutdown);
}

#[test]
fn mooncake_registers_nexus_descriptor_and_unregisters_on_shutdown() {
    let state = Arc::new(Mutex::new(MooncakeState::default()));
    let io = RecordingMooncakeIo {
        state: state.clone(),
    };
    let memory = transferable_memory();
    let layout = MooncakeKvCacheLayout {
        source_base_addr: memory.regions()[0].base_addr,
        page_size_bytes: memory.page_size_bytes(),
        target_base_offset: 0,
    };
    let mut backend = MooncakeKvCacheTransferExecutor::new(
        io.clone(),
        layout,
        MooncakeTransferTarget { target_id: 3 },
    )
    .with_memory_registrar(io);
    backend
        .register(memory)
        .expect("descriptor should register");

    let mut registry = registry_with_session(10);
    execute_kv_cache_transfer_plan(&mut registry, &mut backend, &simple_plan(10))
        .expect("Mooncake submit should succeed");
    let poll = backend.poll().expect("Mooncake poll should succeed");
    apply_transfer_completion_events(&mut registry, poll).expect("completion event should apply");
    backend
        .shutdown()
        .expect("Mooncake shutdown should succeed");

    let state = state.lock().expect("state lock");
    assert_eq!(state.registered, 1);
    assert_eq!(state.submitted, 1);
    assert_eq!(state.freed, 1);
    assert_eq!(state.unregistered, 1);
    assert_eq!(registry.get(10).expect("session").status(), KvPoll::Success);
}

fn simple_plan(room: u64) -> KvCacheTransferPlan {
    let mut scheduler = Scheduler::with_cache_resources(
        FinishedWorker,
        RadixCache::default(),
        CachePageAllocator::new(4),
    );
    scheduler.enqueue(
        ScheduledRequest::new(request_id(room), vec![1, 2], SamplingParams::new(1))
            .with_disaggregated_params(Some(disaggregated_params(room))),
    );
    let batch = scheduler
        .next_prefill_batch(1)
        .expect("prefill batch should exist");
    KvCacheTransferPlan::from_prefill_worker_batch(&ModelWorkerBatch::from_schedule_batch(&batch))
        .expect("transfer plan should build")
}

fn registry_with_session(room: u64) -> DecodeBootstrapRegistry {
    let mut registry = DecodeBootstrapRegistry::default();
    registry
        .register(DecodeBootstrapSession::new(
            request_id(room),
            disaggregated_params(room),
            0,
        ))
        .expect("session should register");
    registry
}

fn disaggregated_params(room: u64) -> DisaggregatedParams {
    DisaggregatedParams {
        bootstrap_host: "127.0.0.1".to_string(),
        bootstrap_port: 8998,
        bootstrap_room: room,
    }
}

fn request_id(room: u64) -> RequestId {
    let value = format!("request-{room}");
    RequestId::from(value.as_str())
}

fn transferable_memory() -> TransferableKvCacheMemory {
    TransferableKvCacheMemory::new(
        vec![TransferableKvCacheRegion {
            base_addr: 0x1000,
            byte_len: 0x1000,
            page_size_bytes: 32,
        }],
        32,
        KvCacheMemoryLocation::Cuda { device_id: 2 },
    )
    .expect("descriptor should be valid")
}

#[derive(Default)]
struct MooncakeState {
    registered: usize,
    unregistered: usize,
    submitted: usize,
    freed: usize,
}

#[derive(Clone)]
struct RecordingMooncakeIo {
    state: Arc<Mutex<MooncakeState>>,
}

impl MooncakeMemoryRegistrar for RecordingMooncakeIo {
    fn register_memory_batch(
        &mut self,
        _buffers: &mut [MooncakeBufferEntry],
        _location: &str,
    ) -> Result<(), MooncakeError> {
        self.state.lock().expect("state lock").registered += 1;
        Ok(())
    }

    fn unregister_memory_batch(&mut self, _addrs: &mut [*mut c_void]) -> Result<(), MooncakeError> {
        self.state.lock().expect("state lock").unregistered += 1;
        Ok(())
    }
}

impl MooncakeTransferSubmitter for RecordingMooncakeIo {
    fn submit_transfer(
        &mut self,
        requests: &mut [MooncakeTransferRequest],
    ) -> Result<MooncakeBatchId, MooncakeError> {
        assert!(!requests.is_empty());
        self.state.lock().expect("state lock").submitted += 1;
        Ok(42)
    }
}

impl MooncakeTransferStatusReader for RecordingMooncakeIo {
    fn transfer_status(
        &mut self,
        _batch_id: MooncakeBatchId,
        _task_id: usize,
    ) -> Result<MooncakeTransferStatus, MooncakeError> {
        Ok(MooncakeTransferStatus {
            status: MooncakeTransferStatusCode::Completed as i32,
            transferred_bytes: 32,
        })
    }
}

impl MooncakeBatchReleaser for RecordingMooncakeIo {
    fn free_batch(&mut self, _batch_id: MooncakeBatchId) -> Result<(), MooncakeError> {
        self.state.lock().expect("state lock").freed += 1;
        Ok(())
    }
}
