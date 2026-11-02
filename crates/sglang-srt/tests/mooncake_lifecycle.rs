#![cfg(feature = "test-support")]

use std::ffi::c_void;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use sglang_srt::cache::{CachePageAllocator, RadixCache};
use sglang_srt::model_executor::ModelWorkerBatch;
use sglang_srt::scheduler::{PrefixCachePolicy, ScheduleBatch, ScheduledRequest, Scheduler};
use sglang_srt::transfer::*;
use sglang_srt::types::{DisaggregatedParams, RequestId, SamplingParams};
use sglang_srt::worker::{BatchGeneratedTokens, GeneratedToken, ModelWorker};

struct Unfinished;
impl ModelWorker for Unfinished {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        BatchGeneratedTokens::from_batch(
            batch,
            batch
                .requests()
                .iter()
                .map(|_| GeneratedToken::unfinished(vec![1]))
                .collect(),
        )
        .expect("valid output")
    }
}

fn request(room: u64) -> ScheduledRequest {
    ScheduledRequest::new(
        RequestId::from(format!("request-{room}").as_str()),
        vec![1, 2],
        SamplingParams::new(2),
    )
    .with_disaggregated_params(Some(DisaggregatedParams {
        bootstrap_host: "127.0.0.1".into(),
        bootstrap_port: 8998,
        bootstrap_room: room,
    }))
}

fn plan(room: u64) -> KvCacheTransferPlan {
    let mut scheduler = Scheduler::with_cache_resources(
        Unfinished,
        RadixCache::default(),
        CachePageAllocator::new(4),
    );
    scheduler.enqueue(request(room));
    KvCacheTransferPlan::from_prefill_worker_batch(&ModelWorkerBatch::from_schedule_batch(
        &scheduler.next_prefill_batch(1).expect("batch"),
    ))
    .expect("plan")
}

#[derive(Default)]
struct NativeState {
    batches: usize,
    tail_polls: usize,
    busy: bool,
    query_error_once: bool,
    unregister_error_once: bool,
    freed: Vec<usize>,
    events: Vec<&'static str>,
}

#[derive(Clone)]
struct StrictNative(Arc<Mutex<NativeState>>);
impl MooncakeMemoryRegistrar for StrictNative {
    fn register_memory_batch(
        &mut self,
        _: &mut [MooncakeBufferEntry],
        _: &str,
    ) -> Result<(), MooncakeError> {
        self.0.lock().expect("state").events.push("register");
        Ok(())
    }
    fn unregister_memory_batch(&mut self, _: &mut [*mut c_void]) -> Result<(), MooncakeError> {
        let mut state = self.0.lock().expect("state");
        assert_eq!(
            state.freed.len(),
            state.batches,
            "unregister requires every batch to be drained and freed"
        );
        if state.unregister_error_once {
            state.unregister_error_once = false;
            return Err(MooncakeError::UnregisterMemoryFailed(-1));
        }
        state.events.push("unregister");
        Ok(())
    }
}
impl MooncakeTransferSubmitter for StrictNative {
    fn submit_transfer(
        &mut self,
        requests: &mut [MooncakeTransferRequest],
    ) -> Result<MooncakeBatchId, MooncakeError> {
        assert_eq!(requests.len(), 2);
        let mut state = self.0.lock().expect("state");
        state.batches += 1;
        state.events.push("submit");
        Ok(state.batches as MooncakeBatchId)
    }
}
impl MooncakeTransferStatusReader for StrictNative {
    fn transfer_status(
        &mut self,
        batch: MooncakeBatchId,
        task: usize,
    ) -> Result<MooncakeTransferStatus, MooncakeError> {
        let mut state = self.0.lock().expect("state");
        if batch == 2 && state.query_error_once {
            state.query_error_once = false;
            return Err(MooncakeError::StatusQueryFailed(-1));
        }
        let status = if state.busy && task == 0 {
            MooncakeTransferStatusCode::Failed
        } else if state.busy && task == 1 {
            state.tail_polls += 1;
            if state.tail_polls < 3 {
                MooncakeTransferStatusCode::Pending
            } else {
                MooncakeTransferStatusCode::Completed
            }
        } else {
            MooncakeTransferStatusCode::Completed
        };
        state.events.push("status");
        Ok(MooncakeTransferStatus {
            status: status as i32,
            transferred_bytes: 32,
        })
    }
}
impl MooncakeBatchReleaser for StrictNative {
    fn free_batch(&mut self, batch: MooncakeBatchId) -> Result<(), MooncakeError> {
        let mut state = self.0.lock().expect("state");
        assert!(
            !state.busy || state.tail_polls >= 3,
            "native freeBatchID must reject pending tasks"
        );
        assert!(
            !state.freed.contains(&(batch as usize)),
            "batch cannot be freed twice"
        );
        state.freed.push(batch as usize);
        state.events.push("free");
        Ok(())
    }
}

fn backend(state: &Arc<Mutex<NativeState>>) -> MooncakeKvCacheTransferExecutor<StrictNative> {
    let io = StrictNative(Arc::clone(state));
    let memory = TransferableKvCacheMemory::new(
        vec![TransferableKvCacheRegion {
            base_addr: 0x1000,
            byte_len: 128,
            page_size_bytes: 32,
        }],
        32,
        KvCacheMemoryLocation::Cpu { numa_node: 0 },
    )
    .expect("descriptor");
    let mut backend = MooncakeKvCacheTransferExecutor::new(
        io.clone(),
        MooncakeKvCacheLayout {
            source_base_addr: 0x1000,
            page_size_bytes: 32,
            target_base_offset: 0,
        },
        MooncakeTransferTarget { target_id: 3 },
    )
    .with_memory_registrar(io);
    backend.register(memory).expect("register");
    backend
}

#[test]
fn cancellation_and_drop_drain_every_task_before_free_and_unregister() {
    for explicit in [false, true] {
        let state = Arc::new(Mutex::new(NativeState {
            busy: true,
            ..NativeState::default()
        }));
        {
            let mut backend = backend(&state);
            backend.submit(&plan(5).spans()[0]).expect("submit");
            if explicit {
                backend.cancel(5).expect("drain mixed failed/pending batch");
                backend.cancel(5).expect("repeat cancel");
                backend.shutdown().expect("unregister after drain");
            }
        }
        let state = state.lock().expect("state");
        assert_eq!(state.freed, vec![1]);
        assert_eq!(state.events.last(), Some(&"unregister"));
        assert_eq!(state.tail_polls, 3);
    }
}

#[test]
fn poll_preserves_completion_when_a_later_batch_status_query_fails() {
    let state = Arc::new(Mutex::new(NativeState {
        query_error_once: true,
        ..NativeState::default()
    }));
    let mut backend = backend(&state);
    backend.submit(&plan(7).spans()[0]).expect("first batch");
    backend.submit(&plan(8).spans()[0]).expect("second batch");
    assert!(backend.poll().is_err());
    assert_eq!(backend.submitted_transfers().len(), 1);
    let poll = backend.poll().expect("retry query");
    assert_eq!(
        poll.events()
            .iter()
            .map(|event| event.bootstrap_room())
            .collect::<Vec<_>>(),
        vec![7, 8]
    );
    assert!(
        backend
            .poll()
            .expect("no duplicate events")
            .events()
            .is_empty()
    );
    backend.shutdown().expect("shutdown");
}

#[test]
fn shutdown_keeps_registration_lease_when_unregister_fails_and_can_retry() {
    let state = Arc::new(Mutex::new(NativeState {
        unregister_error_once: true,
        ..NativeState::default()
    }));
    let mut backend = backend(&state);
    backend.submit(&plan(10).spans()[0]).expect("submit");
    assert!(backend.shutdown().is_err());
    assert_eq!(state.lock().expect("state").freed, vec![1]);
    backend
        .shutdown()
        .expect("retry unregister without freeing batch twice");
    assert_eq!(
        state.lock().expect("state").events.last(),
        Some(&"unregister")
    );
}

#[test]
fn batch_cleanup_failure_retains_every_request_until_drain() {
    let safe = Arc::new(AtomicBool::new(false));
    let worker = KvTransferModelWorker::new(
        Unfinished,
        DecodeBootstrapRegistry::default(),
        CancelGate(Arc::clone(&safe)),
    );
    let mut scheduler = Scheduler::with_cache_resources_and_policy(
        worker,
        RadixCache::default(),
        CachePageAllocator::new(4),
        PrefixCachePolicy::Disabled,
    );
    scheduler.enqueue(request(21));
    scheduler.enqueue(request(22));
    let batch = scheduler.next_prefill_batch(2).expect("prefill batch");
    let output = BatchGeneratedTokens::from_batch(
        &batch,
        vec![
            GeneratedToken::finished(vec![1]),
            GeneratedToken::finished(vec![1]),
        ],
    )
    .expect("finished output");
    assert!(scheduler.complete_batch(batch, output).is_err());
    assert_eq!(scheduler.decode_queue_depth(), 2);
    assert_eq!(scheduler.available_cache_pages(), Some(0));
    safe.store(true, Ordering::SeqCst);
    assert_eq!(scheduler.shutdown().expect("drain both requests"), 2);
    assert_eq!(scheduler.available_cache_pages(), Some(4));
}

struct CancelGate(Arc<AtomicBool>);
impl KvTransferBackend for CancelGate {
    fn register(&mut self, _: TransferableKvCacheMemory) -> Result<(), KvCacheTransferError> {
        Err(KvCacheTransferError::Runtime(
            "unit-test cancellation gate has no native memory".into(),
        ))
    }
    fn submit(&mut self, _: &KvCacheTransferSpan) -> Result<(), KvCacheTransferError> {
        Ok(())
    }
    fn poll(&mut self) -> Result<KvTransferPoll, KvCacheTransferError> {
        Ok(KvTransferPoll::new(Vec::new(), 1))
    }
    fn cancel(&mut self, _: u64) -> Result<(), KvCacheTransferError> {
        if self.0.load(Ordering::SeqCst) {
            Ok(())
        } else {
            Err(KvCacheTransferError::Runtime(
                "sender is still writing".into(),
            ))
        }
    }
    fn shutdown(&mut self) -> Result<(), KvCacheTransferError> {
        assert!(self.0.load(Ordering::SeqCst));
        Ok(())
    }
}

#[test]
fn failed_cancel_keeps_request_and_pages_reserved_until_retry_drains() {
    let safe = Arc::new(AtomicBool::new(false));
    let worker = KvTransferModelWorker::new(
        Unfinished,
        DecodeBootstrapRegistry::default(),
        CancelGate(Arc::clone(&safe)),
    );
    let mut scheduler = Scheduler::with_cache_resources_and_policy(
        worker,
        RadixCache::default(),
        CachePageAllocator::new(4),
        PrefixCachePolicy::Disabled,
    );
    scheduler.enqueue(request(12));
    scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill submitted");
    assert!(!scheduler.abort_request(&request(12).request_id().clone()));
    assert_eq!(scheduler.available_cache_pages(), Some(2));
    assert_eq!(scheduler.decode_queue_depth(), 1);
    assert!(scheduler.shutdown().is_err());
    safe.store(true, Ordering::SeqCst);
    assert!(scheduler.abort_request(request(12).request_id()));
    assert_eq!(scheduler.available_cache_pages(), Some(4));
    scheduler
        .shutdown()
        .expect("clean shutdown after sender drain");
}
