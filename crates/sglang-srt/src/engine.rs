use std::fmt;

use crate::scheduler::{ScheduledOutput, ScheduledRequest, Scheduler, SchedulerError};
use crate::tokenizer::{Tokenizer, TokenizerError};
use crate::types::{GenerateOutput, GenerateRequest, TokenGenerateOutput, TokenGenerateRequest};
use crate::worker::ModelWorker;

#[derive(Debug)]
pub enum RuntimeError {
    Scheduler(SchedulerError),
    Tokenizer(TokenizerError),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scheduler(error) => write!(formatter, "scheduler error: {error}"),
            Self::Tokenizer(error) => write!(formatter, "tokenizer error: {error}"),
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
}

impl<T, W> Engine<T, W>
where
    T: Tokenizer,
    W: ModelWorker,
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
        let output = self.generate_scheduled(ScheduledRequest::new(
            request.request_id,
            request.input_ids,
            request.sampling,
        ))?;

        Ok(TokenGenerateOutput {
            request_id: output.request_id,
            output_ids: output.token_ids,
            finished: output.finished,
        })
    }

    fn generate_scheduled(
        &mut self,
        request: ScheduledRequest,
    ) -> Result<ScheduledOutput, RuntimeError> {
        self.scheduler.enqueue(request);
        let mut scheduled_output = self.scheduler.dispatch_next()?;
        let request_id = scheduled_output.request_id.clone();
        let mut output_ids = Vec::new();
        output_ids.extend_from_slice(&scheduled_output.token_ids);

        while !scheduled_output.finished {
            scheduled_output = self.next_decode_output()?;
            output_ids.extend_from_slice(&scheduled_output.token_ids);
        }

        Ok(ScheduledOutput {
            request_id,
            token_ids: output_ids,
            finished: scheduled_output.finished,
        })
    }

    fn next_decode_output(&mut self) -> Result<ScheduledOutput, RuntimeError> {
        self.scheduler
            .dispatch_decode_batch(1)?
            .pop()
            .ok_or(SchedulerError::EmptyQueue.into())
    }
}
