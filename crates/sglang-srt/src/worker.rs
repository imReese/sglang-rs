use std::fmt;

use crate::scheduler::{ForwardMode, ScheduleBatch, ScheduledOutput, ScheduledRequest};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeneratedToken {
    token_ids: Vec<u32>,
    finished: bool,
}

impl GeneratedToken {
    pub fn finished(token_ids: Vec<u32>) -> Self {
        Self {
            token_ids,
            finished: true,
        }
    }

    pub fn unfinished(token_ids: Vec<u32>) -> Self {
        Self {
            token_ids,
            finished: false,
        }
    }

    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchGeneratedTokens {
    tokens: Vec<GeneratedToken>,
}

impl BatchGeneratedTokens {
    pub fn from_batch(
        batch: &ScheduleBatch,
        tokens: Vec<GeneratedToken>,
    ) -> Result<Self, WorkerOutputError> {
        if batch.batch_size() != tokens.len() {
            return Err(WorkerOutputError::BatchSizeMismatch {
                request_count: batch.batch_size(),
                output_count: tokens.len(),
            });
        }

        Ok(Self { tokens })
    }

    pub fn into_scheduled_outputs(self, batch: ScheduleBatch) -> Vec<ScheduledOutput> {
        batch
            .into_requests()
            .into_iter()
            .zip(self.tokens)
            .map(|(request, generated)| ScheduledOutput {
                cached_tokens: request.cached_token_count(),
                request_id: request.into_request_id(),
                token_ids: generated.token_ids().to_vec(),
                finished: generated.is_finished(),
            })
            .collect()
    }

    pub(crate) fn into_tokens(self) -> Vec<GeneratedToken> {
        self.tokens
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum WorkerOutputError {
    BatchSizeMismatch {
        request_count: usize,
        output_count: usize,
    },
}

impl fmt::Display for WorkerOutputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BatchSizeMismatch {
                request_count,
                output_count,
            } => write!(
                formatter,
                "batch output count ({output_count}) must match request count ({request_count})"
            ),
        }
    }
}

impl std::error::Error for WorkerOutputError {}

#[derive(Debug, Eq, PartialEq)]
pub enum WorkerExecutionError {
    Output(WorkerOutputError),
    Runtime(String),
}

impl fmt::Display for WorkerExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Output(error) => write!(formatter, "worker output error: {error}"),
            Self::Runtime(error) => write!(formatter, "worker runtime error: {error}"),
        }
    }
}

impl std::error::Error for WorkerExecutionError {}

impl From<WorkerOutputError> for WorkerExecutionError {
    fn from(value: WorkerOutputError) -> Self {
        Self::Output(value)
    }
}

pub trait ModelWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens;
}

pub trait FallibleModelWorker {
    fn try_generate_batch(
        &mut self,
        batch: &ScheduleBatch,
    ) -> Result<BatchGeneratedTokens, WorkerExecutionError>;

    fn decode_request_state(
        &self,
        _request: &ScheduledRequest,
    ) -> Result<DecodeRequestState, WorkerExecutionError> {
        Ok(DecodeRequestState::Ready)
    }
}

pub trait WorkerExecutor {
    fn execute_batch(
        &mut self,
        batch: &ScheduleBatch,
    ) -> Result<BatchGeneratedTokens, WorkerExecutionError>;

    fn decode_request_state(
        &self,
        request: &ScheduledRequest,
    ) -> Result<DecodeRequestState, WorkerExecutionError>;
}

impl<W> FallibleModelWorker for W
where
    W: ModelWorker,
{
    fn try_generate_batch(
        &mut self,
        batch: &ScheduleBatch,
    ) -> Result<BatchGeneratedTokens, WorkerExecutionError> {
        Ok(self.generate_batch(batch))
    }
}

impl<W> WorkerExecutor for W
where
    W: FallibleModelWorker,
{
    fn execute_batch(
        &mut self,
        batch: &ScheduleBatch,
    ) -> Result<BatchGeneratedTokens, WorkerExecutionError> {
        self.try_generate_batch(batch)
    }

    fn decode_request_state(
        &self,
        request: &ScheduledRequest,
    ) -> Result<DecodeRequestState, WorkerExecutionError> {
        FallibleModelWorker::decode_request_state(self, request)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecodeRequestState {
    Ready,
    Pending,
    Failed(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PdModelWorkers<P, D> {
    prefill: P,
    decode: D,
}

impl<P, D> PdModelWorkers<P, D> {
    pub fn new(prefill: P, decode: D) -> Self {
        Self { prefill, decode }
    }

    pub fn prefill(&self) -> &P {
        &self.prefill
    }

    pub fn prefill_mut(&mut self) -> &mut P {
        &mut self.prefill
    }

    pub fn decode(&self) -> &D {
        &self.decode
    }

    pub fn decode_mut(&mut self) -> &mut D {
        &mut self.decode
    }
}

impl<P, D> FallibleModelWorker for PdModelWorkers<P, D>
where
    P: FallibleModelWorker,
    D: FallibleModelWorker,
{
    fn try_generate_batch(
        &mut self,
        batch: &ScheduleBatch,
    ) -> Result<BatchGeneratedTokens, WorkerExecutionError> {
        match batch.forward_mode() {
            ForwardMode::Prefill => self.prefill.try_generate_batch(batch),
            ForwardMode::Decode => self.decode.try_generate_batch(batch),
        }
    }
}
