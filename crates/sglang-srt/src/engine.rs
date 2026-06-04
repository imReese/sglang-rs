use std::fmt;

use crate::scheduler::{ScheduledOutput, ScheduledRequest, Scheduler, SchedulerError};
use crate::tokenizer::{Tokenizer, TokenizerError};
use crate::transfer::{KvCacheTransferError, KvTransferPoller, MooncakeTransferPollSummary};
use crate::types::{
    GenerateOutput, GenerateRequest, RequestId, TokenGenerateOutput, TokenGenerateRequest,
};
use crate::worker::WorkerExecutor;

#[derive(Debug)]
pub enum RuntimeError {
    Scheduler(SchedulerError),
    Tokenizer(TokenizerError),
    Transfer(KvCacheTransferError),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scheduler(error) => write!(formatter, "scheduler error: {error}"),
            Self::Tokenizer(error) => write!(formatter, "tokenizer error: {error}"),
            Self::Transfer(error) => write!(formatter, "transfer error: {error}"),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<SchedulerError> for RuntimeError {
    fn from(value: SchedulerError) -> Self {
        Self::Scheduler(value)
    }
}

impl From<TokenizerError> for RuntimeError {
    fn from(value: TokenizerError) -> Self {
        Self::Tokenizer(value)
    }
}

impl From<KvCacheTransferError> for RuntimeError {
    fn from(value: KvCacheTransferError) -> Self {
        Self::Transfer(value)
    }
}

pub struct Engine<T, W> {
    tokenizer: T,
    scheduler: Scheduler<W>,
}

impl<T, W> Engine<T, W> {
    pub fn new(tokenizer: T, scheduler: Scheduler<W>) -> Self {
        Self {
            tokenizer,
            scheduler,
        }
    }

    pub fn scheduler(&self) -> &Scheduler<W> {
        &self.scheduler
    }

    pub fn scheduler_mut(&mut self) -> &mut Scheduler<W> {
        &mut self.scheduler
    }

    pub fn flush_cache(&mut self) -> bool {
        self.scheduler.flush_cache()
    }

    pub fn abort_request(&mut self, request_id: &RequestId) -> bool {
        self.scheduler.abort_request(request_id)
    }
}

impl<T, W> Engine<T, W>
where
    W: KvTransferPoller,
{
    pub fn poll_transfers(&mut self) -> Result<MooncakeTransferPollSummary, RuntimeError> {
        Ok(self.scheduler.worker_mut().poll_transfers()?)
    }
}

impl<T, W> Engine<T, W>
where
    T: Tokenizer,
{
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    pub fn detokenize(&self, token_ids: &[u32]) -> Result<String, TokenizerError> {
        self.tokenizer.decode(token_ids)
    }
}

impl<T, W> Engine<T, W>
where
    T: Tokenizer,
    W: WorkerExecutor,
{
    pub fn generate(&mut self, request: GenerateRequest) -> Result<GenerateOutput, RuntimeError> {
        let input_ids = self.tokenizer.encode(&request.prompt);
        let output = self.generate_scheduled(ScheduledRequest::new(
            request.request_id,
            input_ids,
            request.sampling,
        ))?;

        let text = self.tokenizer.decode(&output.token_ids)?;

        Ok(GenerateOutput {
            request_id: output.request_id,
            text,
            finished: output.finished,
        })
    }

    pub fn generate_tokens(
        &mut self,
        request: TokenGenerateRequest,
    ) -> Result<TokenGenerateOutput, RuntimeError> {
        let output = self.generate_scheduled(
            ScheduledRequest::new(request.request_id, request.input_ids, request.sampling)
                .with_disaggregated_params(request.disaggregated_params)
                .with_data_parallel_rank(request.data_parallel_rank),
        )?;

        Ok(TokenGenerateOutput {
            request_id: output.request_id,
            output_ids: output.token_ids,
            cached_tokens: output.cached_tokens,
            finished: output.finished,
        })
    }

    pub fn generate_token_stream(
        &mut self,
        request: TokenGenerateRequest,
    ) -> Result<Vec<TokenGenerateOutput>, RuntimeError> {
        let outputs = self.generate_scheduled_stream(
            ScheduledRequest::new(request.request_id, request.input_ids, request.sampling)
                .with_disaggregated_params(request.disaggregated_params)
                .with_data_parallel_rank(request.data_parallel_rank),
        )?;

        Ok(outputs
            .into_iter()
            .map(|output| TokenGenerateOutput {
                request_id: output.request_id,
                output_ids: output.token_ids,
                cached_tokens: output.cached_tokens,
                finished: output.finished,
            })
            .collect())
    }

    fn generate_scheduled(
        &mut self,
        request: ScheduledRequest,
    ) -> Result<ScheduledOutput, RuntimeError> {
        let outputs = self.generate_scheduled_stream(request)?;
        let mut output_ids = Vec::new();
        let mut final_output = None;

        for output in outputs {
            output_ids.extend_from_slice(&output.token_ids);
            final_output = Some(output);
        }

        let final_output = final_output.ok_or(SchedulerError::EmptyQueue)?;

        Ok(ScheduledOutput {
            request_id: final_output.request_id,
            token_ids: output_ids,
            cached_tokens: final_output.cached_tokens,
            finished: final_output.finished,
        })
    }

    fn generate_scheduled_stream(
        &mut self,
        request: ScheduledRequest,
    ) -> Result<Vec<ScheduledOutput>, RuntimeError> {
        self.scheduler.enqueue(request);
        let mut scheduled_output = self.scheduler.dispatch_next()?;
        let mut outputs = vec![scheduled_output.clone()];

        while !scheduled_output.finished {
            scheduled_output = self.next_decode_output()?;
            outputs.push(scheduled_output.clone());
        }

        Ok(outputs)
    }

    fn next_decode_output(&mut self) -> Result<ScheduledOutput, RuntimeError> {
        self.scheduler
            .dispatch_decode_batch(1)?
            .pop()
            .ok_or(SchedulerError::EmptyQueue.into())
    }
}

impl<T, W> Engine<T, W>
where
    T: Tokenizer,
    W: WorkerExecutor + KvTransferPoller,
{
    pub fn generate_tokens_with_transfer_polling(
        &mut self,
        request: TokenGenerateRequest,
        max_transfer_polls: usize,
    ) -> Result<TokenGenerateOutput, RuntimeError> {
        let outputs = self.generate_scheduled_stream_with_transfer_polling(
            ScheduledRequest::new(request.request_id, request.input_ids, request.sampling)
                .with_disaggregated_params(request.disaggregated_params)
                .with_data_parallel_rank(request.data_parallel_rank),
            max_transfer_polls,
        )?;
        let mut output_ids = Vec::new();
        let mut final_output = None;

        for output in outputs {
            output_ids.extend_from_slice(&output.token_ids);
            final_output = Some(output);
        }

        let final_output = final_output.ok_or(SchedulerError::EmptyQueue)?;

        Ok(TokenGenerateOutput {
            request_id: final_output.request_id,
            output_ids,
            cached_tokens: final_output.cached_tokens,
            finished: final_output.finished,
        })
    }

    pub fn generate_token_stream_with_transfer_polling(
        &mut self,
        request: TokenGenerateRequest,
        max_transfer_polls: usize,
    ) -> Result<Vec<TokenGenerateOutput>, RuntimeError> {
        let outputs = self.generate_scheduled_stream_with_transfer_polling(
            ScheduledRequest::new(request.request_id, request.input_ids, request.sampling)
                .with_disaggregated_params(request.disaggregated_params)
                .with_data_parallel_rank(request.data_parallel_rank),
            max_transfer_polls,
        )?;

        Ok(outputs
            .into_iter()
            .map(|output| TokenGenerateOutput {
                request_id: output.request_id,
                output_ids: output.token_ids,
                cached_tokens: output.cached_tokens,
                finished: output.finished,
            })
            .collect())
    }

    fn generate_scheduled_stream_with_transfer_polling(
        &mut self,
        request: ScheduledRequest,
        max_transfer_polls: usize,
    ) -> Result<Vec<ScheduledOutput>, RuntimeError> {
        self.scheduler.enqueue(request);
        let mut scheduled_output = self.scheduler.dispatch_next()?;
        let mut outputs = vec![scheduled_output.clone()];
        let mut remaining_transfer_polls = max_transfer_polls;

        while !scheduled_output.finished {
            scheduled_output =
                self.next_decode_output_with_transfer_polling(&mut remaining_transfer_polls)?;
            outputs.push(scheduled_output.clone());
        }

        Ok(outputs)
    }

    fn next_decode_output_with_transfer_polling(
        &mut self,
        remaining_transfer_polls: &mut usize,
    ) -> Result<ScheduledOutput, RuntimeError> {
        loop {
            match self.scheduler.dispatch_decode_batch(1) {
                Ok(mut outputs) => {
                    return outputs.pop().ok_or(SchedulerError::EmptyQueue.into());
                }
                Err(SchedulerError::DecodeNotReady { .. }) if *remaining_transfer_polls > 0 => {
                    *remaining_transfer_polls -= 1;
                    self.poll_transfers()?;
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
}
