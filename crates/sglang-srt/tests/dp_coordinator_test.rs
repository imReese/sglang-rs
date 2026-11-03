// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use sglang_srt::cache::CachePageAllocator;
use sglang_srt::dp_coordinator::DpCoordinator;
use sglang_srt::dp_scheduler::DpScheduler;
use sglang_srt::dp_worker_pool::DpWorkerPool;
use sglang_srt::parallel_topology::{Parallel4DConfig, Parallel4DTopology};
use sglang_srt::scheduler::ScheduleBatch;
use sglang_srt::tokenizer::ByteTokenizer;
use sglang_srt::types::{GenerateRequest, RequestId, SamplingParams};
use sglang_srt::worker::{
    BatchGeneratedTokens, FallibleModelWorker, GeneratedToken, WorkerExecutionError,
    WorkerWeightUpdateRequest,
};

struct CountingWorker {
    rank: usize,
    batches_executed: Arc<AtomicUsize>,
}

impl FallibleModelWorker for CountingWorker {
    fn try_generate_batch(
        &mut self,
        batch: &ScheduleBatch,
    ) -> Result<BatchGeneratedTokens, WorkerExecutionError> {
        self.batches_executed.fetch_add(1, Ordering::SeqCst);
        let tokens = batch
            .requests()
            .iter()
            .map(|_| GeneratedToken::finished(vec![48 + self.rank as u32])) // '0' + rank
            .collect();
        BatchGeneratedTokens::from_batch(batch, tokens).map_err(WorkerExecutionError::from)
    }

    fn update_weights_from_disk(
        &mut self,
        _req: &WorkerWeightUpdateRequest,
    ) -> Result<(), WorkerExecutionError> {
        Ok(())
    }
}

#[test]
fn test_4_worker_dp_coordinator_continuous_batching() {
    let dp_size = 4;
    let config = Parallel4DConfig::new(1, dp_size, 1, dp_size, 1).unwrap();
    let topology = Parallel4DTopology::new(config).unwrap();

    let batches_executed = Arc::new(AtomicUsize::new(0));

    let pool = DpWorkerPool::launch(dp_size, {
        let batches = Arc::clone(&batches_executed);
        move |rank| {
            Ok::<_, String>(CountingWorker {
                rank,
                batches_executed: Arc::clone(&batches),
            })
        }
    })
    .unwrap();

    let page_size = 4;
    let pages_per_rank = 64;
    let allocators: Vec<_> = (0..dp_size)
        .map(|_| CachePageAllocator::with_page_size(pages_per_rank, page_size).unwrap())
        .collect();

    let max_batch_size = 16;
    let scheduler = DpScheduler::new(pool, allocators, max_batch_size, page_size).unwrap();

    let mut engine = DpCoordinator::new(ByteTokenizer, scheduler, topology);

    assert!(engine.is_idle());

    // 1. Submit first request on Rank 0
    let prompt1 = "common_prefix_1234_request_A".to_string();
    let id1 = engine.enqueue_request(GenerateRequest {
        request_id: RequestId::from("req_1"),
        prompt: prompt1.clone(),
        sampling: SamplingParams::new(1),
    });
    assert_eq!(id1, RequestId::from("req_1"));

    // Step until req_1 finishes
    let mut finished_req1 = false;
    while !finished_req1 {
        let outputs = engine.step().unwrap();
        for out in outputs {
            if out.request_id == RequestId::from("req_1") {
                assert!(out.finished);
                finished_req1 = true;
            }
        }
        if !finished_req1 && !engine.is_idle() {
            engine.scheduler_mut().step_blocking().unwrap();
        }
    }

    assert!(batches_executed.load(Ordering::SeqCst) >= 1);

    // Verify prefix cache has stored the prefix
    let prompt1_ids = engine.tokenize(&prompt1);
    assert!(
        engine
            .scheduler()
            .prefix_cache()
            .match_prefix(&prompt1_ids)
            .matched_token_count
            > 0
    );

    // 2. Submit second request sharing the identical prompt prefix:
    // It should hit prefix cache immediately, warming from Rank 0's prefill!
    let prompt2 = format!("{prompt1}_extension_B");
    engine.enqueue_request(GenerateRequest {
        request_id: RequestId::from("req_2"),
        prompt: prompt2,
        sampling: SamplingParams::new(1),
    });

    let mut finished_req2 = false;
    while !finished_req2 {
        let outputs = engine.step().unwrap();
        for out in outputs {
            if out.request_id == RequestId::from("req_2") {
                assert!(out.finished);
                finished_req2 = true;
            }
        }
        if !finished_req2 && !engine.is_idle() {
            engine.scheduler_mut().step_blocking().unwrap();
        }
    }

    assert!(engine.is_idle());

    // 3. Test clean shutdown of all 4 workers
    engine.shutdown().expect("shutdown all 4 workers cleanly");
}
