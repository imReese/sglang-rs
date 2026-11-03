// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, VecDeque};
use std::fmt;

use crate::cache::{CacheAllocationError, CachePageAllocator, RadixCache};
use crate::dp_worker_pool::{DpWorkerEvent, DpWorkerPool, DpWorkerPoolError};
use crate::scheduler::{
    ForwardMode, RequestStage, ScheduleBatch, ScheduledOutput, ScheduledRequest,
};
use crate::types::RequestId;
use crate::worker::WorkerExecutionError;

/// Centralized multi-worker scheduler managing unified request queues,
/// a single global shared prefix cache, and concurrent worker execution ranks.
pub struct MultiWorkerScheduler {
    waiting_queue: VecDeque<ScheduledRequest>,
    decode_queue: VecDeque<ScheduledRequest>,
    prefix_cache: RadixCache,
    allocators: Vec<CachePageAllocator>,
    worker_pool: DpWorkerPool,
    running_batches: BTreeMap<u64, (usize, ScheduleBatch)>,
    finished_outputs: Vec<ScheduledOutput>,
    max_batch_size: usize,
    page_size: usize,
}

impl MultiWorkerScheduler {
    pub fn new(
        worker_pool: DpWorkerPool,
        allocators: Vec<CachePageAllocator>,
        max_batch_size: usize,
        page_size: usize,
    ) -> Result<Self, MultiWorkerSchedulerError> {
        if worker_pool.dp_size() != allocators.len() {
            return Err(MultiWorkerSchedulerError::AllocatorCountMismatch {
                dp_size: worker_pool.dp_size(),
                allocators: allocators.len(),
            });
        }
        if max_batch_size == 0 {
            return Err(MultiWorkerSchedulerError::ZeroBatchSize);
        }
        if page_size == 0 {
            return Err(MultiWorkerSchedulerError::ZeroPageSize);
        }

        Ok(Self {
            waiting_queue: VecDeque::new(),
            decode_queue: VecDeque::new(),
            prefix_cache: RadixCache::default(),
            allocators,
            worker_pool,
            running_batches: BTreeMap::new(),
            finished_outputs: Vec::new(),
            max_batch_size,
            page_size,
        })
    }

    pub fn dp_size(&self) -> usize {
        self.worker_pool.dp_size()
    }

    pub fn waiting_queue_len(&self) -> usize {
        self.waiting_queue.len()
    }

    pub fn decode_queue_len(&self) -> usize {
        self.decode_queue.len()
    }

    pub fn prefix_cache(&self) -> &RadixCache {
        &self.prefix_cache
    }

    pub fn enqueue(&mut self, request: ScheduledRequest) {
        self.waiting_queue.push_back(request);
    }

    pub fn is_idle(&self) -> bool {
        self.waiting_queue.is_empty()
            && self.decode_queue.is_empty()
            && self.running_batches.is_empty()
    }

    pub fn running_batch_count(&self) -> usize {
        self.running_batches.len()
    }

    pub fn abort_request(&mut self, request_id: &RequestId) -> bool {
        let mut aborted = false;
        let original_waiting = self.waiting_queue.len();
        self.waiting_queue.retain(|r| r.request_id() != request_id);
        if self.waiting_queue.len() < original_waiting {
            aborted = true;
        }

        self.decode_queue.retain(|r| {
            if r.request_id() == request_id {
                aborted = true;
                let rank_idx = r.data_parallel_rank() as usize;
                if rank_idx < self.allocators.len() {
                    self.allocators[rank_idx].release(r.allocated_cache_pages());
                }
                false
            } else {
                true
            }
        });

        aborted
    }

    pub fn flush_cache(&mut self) {
        self.prefix_cache = RadixCache::default();
        for alloc in &mut self.allocators {
            alloc.reset();
        }
    }

    pub fn shutdown(&mut self) -> Result<(), MultiWorkerSchedulerError> {
        self.worker_pool.shutdown()?;
        Ok(())
    }

    /// Single non-blocking tick of the multi-worker scheduling loop:
    ///
    /// 1. Drains completed events from the worker pool.
    /// 2. Updates request output tokens and registers completed prefills into the SHARED prefix cache.
    /// 3. Forms and dispatches new batches to any currently idle worker ranks.
    pub fn step(&mut self) -> Result<usize, MultiWorkerSchedulerError> {
        let events = self.worker_pool.poll_events();
        let event_count = events.len();

        for event in events {
            match event {
                DpWorkerEvent::BatchCompleted {
                    sequence,
                    rank,
                    result,
                } => {
                    let (_, batch) = self
                        .running_batches
                        .remove(&sequence)
                        .ok_or(MultiWorkerSchedulerError::UnknownSequence { sequence, rank })?;
                    let generated = result.map_err(MultiWorkerSchedulerError::Worker)?;
                    self.handle_batch_completion(rank, batch, generated)?;
                }
                DpWorkerEvent::FatalError { rank, error } => {
                    return Err(MultiWorkerSchedulerError::RankFatal { rank, error });
                }
                DpWorkerEvent::WeightsUpdated { .. } => {}
            }
        }

        // Dispatch new work to any idle ranks
        for rank in self.worker_pool.idle_ranks() {
            if let Some(batch) = self.form_batch_for_rank(rank)? {
                let seq = self.worker_pool.submit_batch(rank, batch.clone())?;
                self.running_batches.insert(seq, (rank, batch));
            }
        }

        Ok(event_count)
    }

    /// Blocks until at least one batch completion event arrives and processes it.
    pub fn step_blocking(&mut self) -> Result<(), MultiWorkerSchedulerError> {
        let event = self.worker_pool.wait_for_event()?;
        match event {
            DpWorkerEvent::BatchCompleted {
                sequence,
                rank,
                result,
            } => {
                let (_, batch) = self
                    .running_batches
                    .remove(&sequence)
                    .ok_or(MultiWorkerSchedulerError::UnknownSequence { sequence, rank })?;
                let generated = result.map_err(MultiWorkerSchedulerError::Worker)?;
                self.handle_batch_completion(rank, batch, generated)?;
            }
            DpWorkerEvent::FatalError { rank, error } => {
                return Err(MultiWorkerSchedulerError::RankFatal { rank, error });
            }
            DpWorkerEvent::WeightsUpdated { .. } => {}
        }

        // Try to dispatch to any newly freed ranks
        for rank in self.worker_pool.idle_ranks() {
            if let Some(batch) = self.form_batch_for_rank(rank)? {
                let seq = self.worker_pool.submit_batch(rank, batch.clone())?;
                self.running_batches.insert(seq, (rank, batch));
            }
        }

        Ok(())
    }

    /// Drains all completed requests from the scheduler.
    pub fn drain_finished_outputs(&mut self) -> Vec<ScheduledOutput> {
        std::mem::take(&mut self.finished_outputs)
    }

    fn handle_batch_completion(
        &mut self,
        rank: usize,
        batch: ScheduleBatch,
        generated: crate::worker::BatchGeneratedTokens,
    ) -> Result<(), MultiWorkerSchedulerError> {
        let is_prefill = batch.forward_mode() == ForwardMode::Prefill;

        for (mut request, token) in batch
            .into_requests()
            .into_iter()
            .zip(generated.into_tokens())
        {
            if is_prefill {
                // Insert completed prompt into the global shared prefix cache!
                if !request.skips_radix_cache_insert() {
                    let _ = self
                        .prefix_cache
                        .insert(request.input_ids(), request.sequence_cache_pages());
                }
            }

            request.append_output_ids(token.token_ids());

            if token.is_finished() || request.reached_max_new_tokens() {
                // Request finished: free allocated pages on this rank
                if let Some(allocator) = self.allocators.get_mut(rank) {
                    allocator.release(request.allocated_cache_pages());
                }
                self.finished_outputs.push(ScheduledOutput {
                    request_id: request.request_id().clone(),
                    token_ids: request.output_ids().to_vec(),
                    cached_tokens: request.cached_token_count(),
                    finished: true,
                });
            } else {
                // Request continues: move to decode queue pinned to this rank
                request.set_stage(RequestStage::DecodeWaiting);
                request.set_data_parallel_rank(rank as i32);
                self.decode_queue.push_back(request);
            }
        }

        Ok(())
    }

    fn form_batch_for_rank(
        &mut self,
        rank: usize,
    ) -> Result<Option<ScheduleBatch>, MultiWorkerSchedulerError> {
        // 1. Prioritize decode requests pinned to this rank
        let mut decode_requests = Vec::new();
        let mut i = 0;
        while i < self.decode_queue.len() && decode_requests.len() < self.max_batch_size {
            if self.decode_queue[i].data_parallel_rank() == rank as i32 {
                if let Some(mut req) = self.decode_queue.remove(i) {
                    req.set_stage(RequestStage::DecodeForward);
                    decode_requests.push(req);
                }
            } else {
                i += 1;
            }
        }

        if !decode_requests.is_empty() {
            return Ok(Some(ScheduleBatch::decode(decode_requests)));
        }

        // 2. Otherwise form prefill batch from global waiting queue
        if self.waiting_queue.is_empty() {
            return Ok(None);
        }

        let allocator = self
            .allocators
            .get_mut(rank)
            .ok_or(MultiWorkerSchedulerError::InvalidRank(rank))?;

        let mut prefill_requests = Vec::new();

        while prefill_requests.len() < self.max_batch_size {
            let Some(mut req) = self.waiting_queue.pop_front() else {
                break;
            };

            // Match against the unified shared RadixCache
            let prefix_match = self
                .prefix_cache
                .match_prefix_page_aligned(req.input_ids(), self.page_size);

            let cached_token_count = prefix_match.matched_token_count;
            let uncached_count = prefix_match.remaining_input_ids.len();

            let allocated = match allocator.allocate(uncached_count) {
                Ok(pages) => pages,
                Err(CacheAllocationError::OutOfSlots { .. }) => {
                    // Not enough cache pages on this rank: put back and stop batch
                    self.waiting_queue.push_front(req);
                    break;
                }
                Err(err) => return Err(MultiWorkerSchedulerError::CacheAllocation(err)),
            };

            let mut seq_pages =
                Vec::with_capacity(prefix_match.cache_pages.len() + allocated.len());
            seq_pages.extend_from_slice(&prefix_match.cache_pages);
            seq_pages.extend_from_slice(&allocated);

            req.set_forward_cache_pages(allocated.clone());
            req.set_allocated_cache_pages(allocated);
            req.set_sequence_cache_pages(seq_pages);
            req.set_cache_hit_token_count(cached_token_count);
            req.apply_prefix_match(prefix_match);
            req.set_stage(RequestStage::PrefillForward);
            req.set_data_parallel_rank(rank as i32);

            prefill_requests.push(req);
        }

        if prefill_requests.is_empty() {
            Ok(None)
        } else {
            Ok(Some(ScheduleBatch::prefill(prefill_requests)))
        }
    }
}

/// Errors occurring during multi-worker scheduler operations.
#[derive(Debug)]
pub enum MultiWorkerSchedulerError {
    ZeroBatchSize,
    ZeroPageSize,
    AllocatorCountMismatch { dp_size: usize, allocators: usize },
    InvalidRank(usize),
    UnknownSequence { sequence: u64, rank: usize },
    RankFatal { rank: usize, error: String },
    Worker(WorkerExecutionError),
    WorkerPool(DpWorkerPoolError),
    CacheAllocation(CacheAllocationError),
}

impl fmt::Display for MultiWorkerSchedulerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroBatchSize => f.write_str("max batch size must be positive"),
            Self::ZeroPageSize => f.write_str("page size must be positive"),
            Self::AllocatorCountMismatch {
                dp_size,
                allocators,
            } => write!(
                f,
                "allocator count ({allocators}) must match worker pool DP size ({dp_size})"
            ),
            Self::InvalidRank(r) => write!(f, "rank {r} is invalid"),
            Self::UnknownSequence { sequence, rank } => {
                write!(f, "unknown batch sequence {sequence} from rank {rank}")
            }
            Self::RankFatal { rank, error } => write!(f, "fatal error on rank {rank}: {error}"),
            Self::Worker(e) => write!(f, "worker error: {e}"),
            Self::WorkerPool(e) => write!(f, "worker pool error: {e}"),
            Self::CacheAllocation(e) => write!(f, "cache allocation error: {e}"),
        }
    }
}

impl std::error::Error for MultiWorkerSchedulerError {}

impl From<DpWorkerPoolError> for MultiWorkerSchedulerError {
    fn from(e: DpWorkerPoolError) -> Self {
        Self::WorkerPool(e)
    }
}

impl From<CacheAllocationError> for MultiWorkerSchedulerError {
    fn from(e: CacheAllocationError) -> Self {
        Self::CacheAllocation(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SamplingParams;
    use crate::worker::{
        BatchGeneratedTokens, FallibleModelWorker, GeneratedToken, WorkerWeightUpdateRequest,
    };

    struct TestRankWorker {
        rank: usize,
    }

    impl FallibleModelWorker for TestRankWorker {
        fn try_generate_batch(
            &mut self,
            batch: &ScheduleBatch,
        ) -> Result<BatchGeneratedTokens, WorkerExecutionError> {
            let tokens = batch
                .requests()
                .iter()
                .map(|_| GeneratedToken::finished(vec![self.rank as u32 + 200]))
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
    fn test_multi_worker_shared_prefix_cache_across_dp_ranks() {
        let pool = DpWorkerPool::launch(2, |rank| Ok::<_, String>(TestRankWorker { rank }))
            .expect("launch pool of 2 ranks");

        // 2 allocators, 64 pages each, page_size = 4
        let allocators = vec![
            CachePageAllocator::with_page_size(64, 4).expect("allocator 0"),
            CachePageAllocator::with_page_size(64, 4).expect("allocator 1"),
        ];

        let mut scheduler = MultiWorkerScheduler::new(pool, allocators, 4, 4)
            .expect("create multi-worker scheduler");

        // Request 1: prompt has 8 tokens (2 pages of 4 tokens)
        let prompt1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let req1 = ScheduledRequest::new(
            RequestId::from("req1"),
            prompt1.clone(),
            SamplingParams::new(1),
        );
        scheduler.enqueue(req1);

        // Step 1: Dispatches Request 1 to Rank 0 (prefill)
        scheduler.step().expect("step 1");
        assert_eq!(scheduler.waiting_queue_len(), 0);

        // Wait for Rank 0 to finish Request 1
        scheduler.step_blocking().expect("step blocking for req 1");

        let outputs1 = scheduler.drain_finished_outputs();
        assert_eq!(outputs1.len(), 1);
        assert_eq!(outputs1[0].request_id, RequestId::from("req1"));
        assert_eq!(outputs1[0].cached_tokens, 0); // first request had 0 cached tokens

        // Now Request 2 arrives: shares the SAME 8-token prefix, plus 4 new tokens!
        let mut prompt2 = prompt1.clone();
        prompt2.extend_from_slice(&[9, 10, 11, 12]);

        let req2 = ScheduledRequest::new(RequestId::from("req2"), prompt2, SamplingParams::new(1));
        scheduler.enqueue(req2);

        // Step 2: Now Rank 1 is idle, so it receives Request 2!
        scheduler.step().expect("step 2");

        // Wait for Rank 1 to finish Request 2
        scheduler.step_blocking().expect("step blocking for req 2");

        let outputs2 = scheduler.drain_finished_outputs();
        assert_eq!(outputs2.len(), 1);
        assert_eq!(outputs2[0].request_id, RequestId::from("req2"));

        // VERIFICATION OF THE ARCHITECTURE:
        // Rank 1 hit 8 cached tokens from the prefix computed by Rank 0!
        assert_eq!(outputs2[0].cached_tokens, 8);
    }
}
