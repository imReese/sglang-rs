use std::fmt;

use crate::scheduler::{ForwardMode, ScheduleBatch, ScheduledOutput};

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

pub trait ModelWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens;
}

pub trait WorkerExecutor {
    fn execute_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens;
}

impl<W> WorkerExecutor for W
where
    W: ModelWorker,
{
    fn execute_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        self.generate_batch(batch)
    }
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

impl<P, D> WorkerExecutor for PdModelWorkers<P, D>
where
    P: ModelWorker,
    D: ModelWorker,
{
    fn execute_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        match batch.forward_mode() {
            ForwardMode::Prefill => self.prefill.generate_batch(batch),
            ForwardMode::Decode => self.decode.generate_batch(batch),
        }
    }
}
