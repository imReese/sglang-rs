use std::fmt;

use crate::scheduler::{ScheduledOutput, ScheduledRequest, Scheduler, SchedulerError};
use crate::tokenizer::{ChatTemplateInput, Tokenizer, TokenizerError};
use crate::transfer::{KvCacheTransferError, MooncakeTransferPollSummary};
use crate::types::{
    GenerateOutput, GenerateRequest, RequestId, TokenGenerateOutput, TokenGenerateRequest,
};
use crate::worker::{WorkerExecutionError, WorkerExecutor, WorkerWeightUpdateRequest};

#[derive(Debug)]
pub enum RuntimeError {
    Scheduler(SchedulerError),
    Tokenizer(TokenizerError),
    Transfer(KvCacheTransferError),
    Worker(WorkerExecutionError),
    InvalidState(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scheduler(error) => write!(formatter, "scheduler error: {error}"),
            Self::Tokenizer(error) => write!(formatter, "tokenizer error: {error}"),
            Self::Transfer(error) => write!(formatter, "transfer error: {error}"),
            Self::Worker(error) => write!(formatter, "worker error: {error}"),
            Self::InvalidState(error) => write!(formatter, "{error}"),
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

impl From<WorkerExecutionError> for RuntimeError {
    fn from(value: WorkerExecutionError) -> Self {
        Self::Worker(value)
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

    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }

    pub fn flush_cache(&mut self) -> bool {
        self.scheduler.flush_cache()
    }

    pub fn abort_request(&mut self, request_id: &RequestId) -> bool
    where
        W: WorkerExecutor,
    {
        self.scheduler.abort_request(request_id)
    }

    pub fn abort_all_requests(&mut self) -> usize
    where
        W: WorkerExecutor,
    {
        self.scheduler.abort_all_requests()
    }
}

impl<T, W> Engine<T, W>
where
    W: WorkerExecutor,
{
    pub fn poll_transfers(&mut self) -> Result<MooncakeTransferPollSummary, RuntimeError> {
        Ok(self.scheduler.poll_transfers()?)
    }

    pub fn update_weights_from_disk(
        &mut self,
        request: &WorkerWeightUpdateRequest,
    ) -> Result<(), RuntimeError> {
        if self.scheduler.waiting_queue_depth() > 0 || self.scheduler.decode_queue_depth() > 0 {
            return Err(RuntimeError::InvalidState(
                "cannot update weights while requests are running or waiting".to_string(),
            ));
        }

        self.scheduler
            .worker_mut()
            .update_weights_from_disk(request)?;
        self.flush_cache();
        Ok(())
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

    pub fn apply_chat_template(&self, input: &ChatTemplateInput) -> Result<String, TokenizerError> {
        self.tokenizer.apply_chat_template(input)
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
        let mut outputs = Vec::new();
        self.generate_token_stream_with_sink(request, |output| {
            outputs.push(output);
            true
        })?;
        Ok(outputs)
    }

    pub fn generate_token_stream_with_sink<F>(
        &mut self,
        request: TokenGenerateRequest,
        mut sink: F,
    ) -> Result<(), RuntimeError>
    where
        F: FnMut(TokenGenerateOutput) -> bool,
    {
        self.generate_scheduled_stream_with_sink(
            ScheduledRequest::new(request.request_id, request.input_ids, request.sampling)
                .with_disaggregated_params(request.disaggregated_params)
                .with_data_parallel_rank(request.data_parallel_rank),
            |output| {
                sink(TokenGenerateOutput {
                    request_id: output.request_id,
                    output_ids: output.token_ids,
                    cached_tokens: output.cached_tokens,
                    finished: output.finished,
                })
            },
        )
    }

    pub fn generate_token_batch_stream(
        &mut self,
        requests: Vec<TokenGenerateRequest>,
    ) -> Result<Vec<Vec<TokenGenerateOutput>>, RuntimeError> {
        let scheduled_requests = requests
            .into_iter()
            .map(|request| {
                ScheduledRequest::new(request.request_id, request.input_ids, request.sampling)
                    .with_disaggregated_params(request.disaggregated_params)
                    .with_data_parallel_rank(request.data_parallel_rank)
            })
            .collect::<Vec<_>>();
        let outputs = self.generate_scheduled_batch_stream(scheduled_requests)?;
        Ok(group_token_outputs_by_request(outputs))
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
        let mut outputs = Vec::new();
        self.generate_scheduled_stream_with_sink(request, |output| {
            outputs.push(output);
            true
        })?;
        Ok(outputs)
    }

    fn generate_scheduled_stream_with_sink<F>(
        &mut self,
        request: ScheduledRequest,
        mut sink: F,
    ) -> Result<(), RuntimeError>
    where
        F: FnMut(ScheduledOutput) -> bool,
    {
        let request_id = request.request_id().clone();
        self.scheduler.enqueue(request);
        let mut scheduled_output = match self.scheduler.dispatch_next() {
            Ok(output) => output,
            Err(error) => {
                self.scheduler.abort_request(&request_id);
                return Err(error.into());
            }
        };
        loop {
            let finished = scheduled_output.finished;
            if !sink(scheduled_output) {
                self.scheduler.abort_request(&request_id);
                return Ok(());
            }
            if finished {
                return Ok(());
            }
            scheduled_output = match self.next_decode_output() {
                Ok(output) => output,
                Err(error) => {
                    if !matches!(
                        error,
                        RuntimeError::Scheduler(SchedulerError::DecodeNotReady { .. })
                    ) {
                        self.scheduler.abort_request(&request_id);
                    }
                    return Err(error);
                }
            };
        }
    }

    fn generate_scheduled_batch_stream(
        &mut self,
        requests: Vec<ScheduledRequest>,
    ) -> Result<Vec<Vec<ScheduledOutput>>, RuntimeError> {
        let batch_size = requests.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let request_ids = requests
            .iter()
            .map(|request| request.request_id().clone())
            .collect::<Vec<_>>();
        for request in requests {
            self.scheduler.enqueue(request);
        }

        let prefill_outputs = match self.scheduler.dispatch_prefill_batch(batch_size) {
            Ok(outputs) => outputs,
            Err(error) => {
                for request_id in &request_ids {
                    self.scheduler.abort_request(request_id);
                }
                return Err(error.into());
            }
        };
        let mut grouped = vec![Vec::new(); batch_size];
        let mut active =
            record_batch_outputs(&mut grouped, (0..batch_size).collect(), prefill_outputs);

        while !active.is_empty() {
            let decode_outputs = self.scheduler.dispatch_decode_batch(active.len())?;
            active = record_batch_outputs(&mut grouped, active, decode_outputs);
        }

        Ok(grouped)
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
    W: WorkerExecutor,
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
        let mut outputs = Vec::new();
        self.generate_token_stream_with_transfer_polling_sink(
            request,
            max_transfer_polls,
            |output| {
                outputs.push(output);
                true
            },
        )?;
        Ok(outputs)
    }

    pub fn generate_token_stream_with_transfer_polling_sink<F>(
        &mut self,
        request: TokenGenerateRequest,
        max_transfer_polls: usize,
        mut sink: F,
    ) -> Result<(), RuntimeError>
    where
        F: FnMut(TokenGenerateOutput) -> bool,
    {
        self.generate_scheduled_stream_with_transfer_polling_sink(
            ScheduledRequest::new(request.request_id, request.input_ids, request.sampling)
                .with_disaggregated_params(request.disaggregated_params)
                .with_data_parallel_rank(request.data_parallel_rank),
            max_transfer_polls,
            |output| {
                sink(TokenGenerateOutput {
                    request_id: output.request_id,
                    output_ids: output.token_ids,
                    cached_tokens: output.cached_tokens,
                    finished: output.finished,
                })
            },
        )
    }

    pub fn generate_token_batch_stream_with_transfer_polling(
        &mut self,
        requests: Vec<TokenGenerateRequest>,
        max_transfer_polls: usize,
    ) -> Result<Vec<Vec<TokenGenerateOutput>>, RuntimeError> {
        let scheduled_requests = requests
            .into_iter()
            .map(|request| {
                ScheduledRequest::new(request.request_id, request.input_ids, request.sampling)
                    .with_disaggregated_params(request.disaggregated_params)
                    .with_data_parallel_rank(request.data_parallel_rank)
            })
            .collect::<Vec<_>>();
        let outputs = self.generate_scheduled_batch_stream_with_transfer_polling(
            scheduled_requests,
            max_transfer_polls,
        )?;
        Ok(group_token_outputs_by_request(outputs))
    }

    fn generate_scheduled_stream_with_transfer_polling(
        &mut self,
        request: ScheduledRequest,
        max_transfer_polls: usize,
    ) -> Result<Vec<ScheduledOutput>, RuntimeError> {
        let mut outputs = Vec::new();
        self.generate_scheduled_stream_with_transfer_polling_sink(
            request,
            max_transfer_polls,
            |output| {
                outputs.push(output);
                true
            },
        )?;
        Ok(outputs)
    }

    fn generate_scheduled_stream_with_transfer_polling_sink<F>(
        &mut self,
        request: ScheduledRequest,
        max_transfer_polls: usize,
        mut sink: F,
    ) -> Result<(), RuntimeError>
    where
        F: FnMut(ScheduledOutput) -> bool,
    {
        let request_id = request.request_id().clone();
        self.scheduler.enqueue(request);
        let mut scheduled_output = match self.scheduler.dispatch_next() {
            Ok(output) => output,
            Err(error) => {
                self.scheduler.abort_request(&request_id);
                return Err(error.into());
            }
        };
        let mut remaining_transfer_polls = max_transfer_polls;

        loop {
            let finished = scheduled_output.finished;
            if !sink(scheduled_output) {
                self.scheduler.abort_request(&request_id);
                return Ok(());
            }
            if finished {
                return Ok(());
            }
            scheduled_output = match self
                .next_decode_output_with_transfer_polling(&mut remaining_transfer_polls)
            {
                Ok(output) => output,
                Err(error) => {
                    if !matches!(
                        error,
                        RuntimeError::Scheduler(SchedulerError::DecodeNotReady { .. })
                    ) {
                        self.scheduler.abort_request(&request_id);
                    }
                    return Err(error);
                }
            };
        }
    }

    fn generate_scheduled_batch_stream_with_transfer_polling(
        &mut self,
        requests: Vec<ScheduledRequest>,
        max_transfer_polls: usize,
    ) -> Result<Vec<Vec<ScheduledOutput>>, RuntimeError> {
        let batch_size = requests.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let request_ids = requests
            .iter()
            .map(|request| request.request_id().clone())
            .collect::<Vec<_>>();
        for request in requests {
            self.scheduler.enqueue(request);
        }

        let prefill_outputs = match self.scheduler.dispatch_prefill_batch(batch_size) {
            Ok(outputs) => outputs,
            Err(error) => {
                for request_id in &request_ids {
                    self.scheduler.abort_request(request_id);
                }
                return Err(error.into());
            }
        };
        let mut grouped = vec![Vec::new(); batch_size];
        let mut active =
            record_batch_outputs(&mut grouped, (0..batch_size).collect(), prefill_outputs);
        let mut remaining_transfer_polls = max_transfer_polls;

        while !active.is_empty() {
            let decode_outputs = loop {
                match self.scheduler.dispatch_decode_batch(active.len()) {
                    Ok(outputs) => break outputs,
                    Err(SchedulerError::DecodeNotReady { .. }) if remaining_transfer_polls > 0 => {
                        remaining_transfer_polls -= 1;
                        self.poll_transfers()?;
                    }
                    Err(error) => return Err(error.into()),
                }
            };
            active = record_batch_outputs(&mut grouped, active, decode_outputs);
        }

        Ok(grouped)
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

fn record_batch_outputs(
    grouped: &mut [Vec<ScheduledOutput>],
    active_indices: Vec<usize>,
    outputs: Vec<ScheduledOutput>,
) -> Vec<usize> {
    let dispatched = outputs.len();
    let mut next_active = active_indices
        .iter()
        .skip(dispatched)
        .copied()
        .collect::<Vec<_>>();
    let mut requeued = Vec::new();

    for (index, output) in active_indices.into_iter().zip(outputs) {
        if !output.finished {
            requeued.push(index);
        }
        grouped[index].push(output);
    }

    next_active.extend(requeued);
    next_active
}

fn group_token_outputs_by_request(
    outputs: Vec<Vec<ScheduledOutput>>,
) -> Vec<Vec<TokenGenerateOutput>> {
    outputs
        .into_iter()
        .map(|outputs| {
            outputs
                .into_iter()
                .map(|output| TokenGenerateOutput {
                    request_id: output.request_id,
                    output_ids: output.token_ids,
                    cached_tokens: output.cached_tokens,
                    finished: output.finished,
                })
                .collect()
        })
        .collect()
}
