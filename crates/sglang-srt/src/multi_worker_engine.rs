// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

use crate::multi_worker_scheduler::{MultiWorkerScheduler, MultiWorkerSchedulerError};
use crate::parallel_topology::Parallel4DTopology;
use crate::scheduler::ScheduledRequest;
use crate::tokenizer::{ChatTemplateInput, Tokenizer, TokenizerError};
use crate::types::{
    GenerateOutput, GenerateRequest, RequestId, TokenGenerateOutput, TokenGenerateRequest,
};
use crate::worker::WorkerExecutionError;

/// Error conditions arising during multi-worker engine orchestration.
#[derive(Debug)]
pub enum MultiWorkerEngineError {
    Scheduler(MultiWorkerSchedulerError),
    Tokenizer(TokenizerError),
    Worker(WorkerExecutionError),
    InvalidState(String),
}

impl fmt::Display for MultiWorkerEngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scheduler(err) => write!(f, "scheduler error: {err}"),
            Self::Tokenizer(err) => write!(f, "tokenizer error: {err}"),
            Self::Worker(err) => write!(f, "worker error: {err}"),
            Self::InvalidState(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for MultiWorkerEngineError {}

impl From<MultiWorkerSchedulerError> for MultiWorkerEngineError {
    fn from(err: MultiWorkerSchedulerError) -> Self {
        Self::Scheduler(err)
    }
}

impl From<TokenizerError> for MultiWorkerEngineError {
    fn from(err: TokenizerError) -> Self {
        Self::Tokenizer(err)
    }
}

impl From<WorkerExecutionError> for MultiWorkerEngineError {
    fn from(err: WorkerExecutionError) -> Self {
        Self::Worker(err)
    }
}

/// Unified multi-worker engine coordinating tokenization, 4D distributed topology,
/// and a centralized multi-worker continuous batching scheduler.
pub struct MultiWorkerEngine<T> {
    tokenizer: T,
    scheduler: MultiWorkerScheduler,
    topology: Parallel4DTopology,
}

impl<T> MultiWorkerEngine<T> {
    pub fn new(
        tokenizer: T,
        scheduler: MultiWorkerScheduler,
        topology: Parallel4DTopology,
    ) -> Self {
        Self {
            tokenizer,
            scheduler,
            topology,
        }
    }

    pub fn topology(&self) -> &Parallel4DTopology {
        &self.topology
    }

    pub fn scheduler(&self) -> &MultiWorkerScheduler {
        &self.scheduler
    }

    pub fn scheduler_mut(&mut self) -> &mut MultiWorkerScheduler {
        &mut self.scheduler
    }

    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }

    pub fn is_idle(&self) -> bool {
        self.scheduler.is_idle()
    }

    pub fn waiting_queue_depth(&self) -> usize {
        self.scheduler.waiting_queue_len()
    }

    pub fn decode_queue_depth(&self) -> usize {
        self.scheduler.decode_queue_len()
    }

    pub fn running_batches(&self) -> usize {
        self.scheduler.running_batch_count()
    }

    pub fn abort_request(&mut self, request_id: &RequestId) -> bool {
        self.scheduler.abort_request(request_id)
    }

    pub fn flush_cache(&mut self) {
        self.scheduler.flush_cache();
    }

    pub fn shutdown(&mut self) -> Result<(), MultiWorkerEngineError> {
        self.scheduler.shutdown()?;
        Ok(())
    }
}

impl<T> MultiWorkerEngine<T>
where
    T: Tokenizer,
{
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    pub fn detokenize(&self, token_ids: &[u32]) -> Result<String, TokenizerError> {
        self.tokenizer.decode(token_ids)
    }

    pub fn apply_chat_template(&self, input: &ChatTemplateInput) -> Result<String, TokenizerError> {
        self.tokenizer.apply_chat_template(input)
    }

    /// Enqueues a high-level text generation request into the multi-worker scheduler.
    pub fn enqueue_request(&mut self, request: GenerateRequest) -> RequestId {
        let input_ids = self.tokenizer.encode(&request.prompt);
        let req_id = request.request_id.clone();
        let scheduled = ScheduledRequest::new(request.request_id, input_ids, request.sampling);
        self.scheduler.enqueue(scheduled);
        req_id
    }

    /// Enqueues a raw token-level generation request into the multi-worker scheduler.
    pub fn enqueue_token_request(&mut self, request: TokenGenerateRequest) -> RequestId {
        let req_id = request.request_id.clone();
        let mut scheduled =
            ScheduledRequest::new(request.request_id, request.input_ids, request.sampling);
        scheduled.set_data_parallel_rank(request.data_parallel_rank);
        self.scheduler.enqueue(scheduled);
        req_id
    }

    /// Performs a non-blocking step on the scheduler, draining completed batches,
    /// scheduling ready work onto idle worker ranks, and returning newly finished text outputs.
    pub fn step(&mut self) -> Result<Vec<GenerateOutput>, MultiWorkerEngineError> {
        self.scheduler.step()?;
        let finished = self.scheduler.drain_finished_outputs();
        let mut outputs = Vec::with_capacity(finished.len());

        for item in finished {
            let text = self.tokenizer.decode(&item.token_ids)?;
            outputs.push(GenerateOutput {
                request_id: item.request_id,
                text,
                finished: item.finished,
            });
        }

        Ok(outputs)
    }

    /// Performs a non-blocking step and returns raw token outputs without string decoding.
    pub fn step_tokens(&mut self) -> Result<Vec<TokenGenerateOutput>, MultiWorkerEngineError> {
        self.scheduler.step()?;
        let finished = self.scheduler.drain_finished_outputs();
        let mut outputs = Vec::with_capacity(finished.len());

        for item in finished {
            outputs.push(TokenGenerateOutput {
                request_id: item.request_id,
                output_ids: item.token_ids,
                cached_tokens: item.cached_tokens,
                finished: item.finished,
            });
        }

        Ok(outputs)
    }

    /// Synchronously runs a single generation request until completion.
    pub fn generate(
        &mut self,
        request: GenerateRequest,
    ) -> Result<GenerateOutput, MultiWorkerEngineError> {
        let target_id = request.request_id.clone();
        self.enqueue_request(request);

        loop {
            // Step scheduler non-blockingly or block for events
            self.scheduler.step()?;
            let finished = self.scheduler.drain_finished_outputs();
            for item in finished {
                if item.request_id == target_id {
                    let text = self.tokenizer.decode(&item.token_ids)?;
                    return Ok(GenerateOutput {
                        request_id: item.request_id,
                        text,
                        finished: item.finished,
                    });
                }
            }
            if !self.scheduler.is_idle() {
                self.scheduler.step_blocking()?;
            } else {
                return Err(MultiWorkerEngineError::InvalidState(format!(
                    "request {} disappeared before generating completion",
                    target_id.as_str()
                )));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::CachePageAllocator;
    use crate::dp_worker_pool::DpWorkerPool;
    use crate::parallel_topology::Parallel4DConfig;
    use crate::tokenizer::ByteTokenizer;
    use crate::types::SamplingParams;
    use crate::worker::{
        BatchGeneratedTokens, FallibleModelWorker, GeneratedToken, WorkerWeightUpdateRequest,
    };

    struct MockWorker {
        _rank: usize,
    }

    impl FallibleModelWorker for MockWorker {
        fn try_generate_batch(
            &mut self,
            batch: &crate::scheduler::ScheduleBatch,
        ) -> Result<BatchGeneratedTokens, WorkerExecutionError> {
            let tokens = batch
                .requests()
                .iter()
                .map(|_| GeneratedToken::finished(vec![33])) // ASCII '!'
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
    fn test_multi_worker_engine_end_to_end() {
        let dp_size = 2;
        let config = Parallel4DConfig::new(1, dp_size, 1, dp_size, 1).unwrap();
        let topology = Parallel4DTopology::new(config).unwrap();

        let pool =
            DpWorkerPool::launch(dp_size, |rank| Ok::<_, String>(MockWorker { _rank: rank }))
                .unwrap();

        let allocators = vec![
            CachePageAllocator::with_page_size(64, 4).unwrap(),
            CachePageAllocator::with_page_size(64, 4).unwrap(),
        ];

        let scheduler = MultiWorkerScheduler::new(pool, allocators, 16, 4).unwrap();

        let mut engine = MultiWorkerEngine::new(ByteTokenizer, scheduler, topology);

        assert!(engine.is_idle());

        let output = engine
            .generate(GenerateRequest {
                request_id: RequestId::from("req_test_1"),
                prompt: "Hello".to_string(),
                sampling: SamplingParams::new(1),
            })
            .expect("generation succeeds");

        assert_eq!(output.request_id, RequestId::from("req_test_1"));
        assert_eq!(output.text, "!");
        assert!(output.finished);

        // Verify engine cleans up and returns to idle
        assert!(engine.is_idle());
    }
}
