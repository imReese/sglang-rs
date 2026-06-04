use crate::cache::CachePageId;
use crate::scheduler::{ForwardMode, ScheduleBatch, ScheduledRequest};
use crate::types::{DisaggregatedParams, RequestId};
use std::fmt;

use crate::worker::{
    BatchGeneratedTokens, FallibleModelWorker, GeneratedToken, WorkerExecutionError,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelWorkerBatch {
    forward_mode: ForwardMode,
    request_ids: Vec<RequestId>,
    input_ids: Vec<u32>,
    positions: Vec<usize>,
    sequence_lengths: Vec<usize>,
    request_offsets: Vec<usize>,
    cached_token_counts: Vec<usize>,
    input_token_counts: Vec<usize>,
    prefix_cache_pages: Vec<Vec<CachePageId>>,
    out_cache_pages: Vec<CachePageId>,
    disaggregated_params: Vec<Option<DisaggregatedParams>>,
    data_parallel_ranks: Vec<i32>,
}

impl ModelWorkerBatch {
    pub fn from_schedule_batch(batch: &ScheduleBatch) -> Self {
        let mut worker_batch = Self {
            forward_mode: batch.forward_mode(),
            request_ids: Vec::with_capacity(batch.batch_size()),
            input_ids: Vec::new(),
            positions: Vec::new(),
            sequence_lengths: Vec::with_capacity(batch.batch_size()),
            request_offsets: Vec::with_capacity(batch.batch_size()),
            cached_token_counts: Vec::with_capacity(batch.batch_size()),
            input_token_counts: Vec::with_capacity(batch.batch_size()),
            prefix_cache_pages: Vec::with_capacity(batch.batch_size()),
            out_cache_pages: Vec::new(),
            disaggregated_params: Vec::with_capacity(batch.batch_size()),
            data_parallel_ranks: Vec::with_capacity(batch.batch_size()),
        };

        for request in batch.requests() {
            worker_batch.push_request(batch.forward_mode(), request);
        }

        worker_batch
    }

    pub fn forward_mode(&self) -> ForwardMode {
        self.forward_mode
    }

    pub fn request_ids(&self) -> &[RequestId] {
        &self.request_ids
    }

    pub fn input_ids(&self) -> &[u32] {
        &self.input_ids
    }

    pub fn positions(&self) -> &[usize] {
        &self.positions
    }

    pub fn sequence_lengths(&self) -> &[usize] {
        &self.sequence_lengths
    }

    pub fn request_offsets(&self) -> &[usize] {
        &self.request_offsets
    }

    pub fn cached_token_counts(&self) -> &[usize] {
        &self.cached_token_counts
    }

    pub fn input_token_counts(&self) -> &[usize] {
        &self.input_token_counts
    }

    pub fn prefix_cache_pages(&self) -> &[Vec<CachePageId>] {
        &self.prefix_cache_pages
    }

    pub fn out_cache_pages(&self) -> &[CachePageId] {
        &self.out_cache_pages
    }

    pub fn disaggregated_params(&self) -> &[Option<DisaggregatedParams>] {
        &self.disaggregated_params
    }

    pub fn data_parallel_ranks(&self) -> &[i32] {
        &self.data_parallel_ranks
    }

    fn push_request(&mut self, forward_mode: ForwardMode, request: &ScheduledRequest) {
        self.request_ids.push(request.request_id().clone());
        self.request_offsets.push(self.input_ids.len());
        self.cached_token_counts.push(request.cached_token_count());
        self.prefix_cache_pages
            .push(request.prefix_cache_pages().to_vec());
        self.disaggregated_params
            .push(request.disaggregated_params().cloned());
        self.data_parallel_ranks.push(request.data_parallel_rank());

        match forward_mode {
            ForwardMode::Prefill => self.push_prefill_request(request),
            ForwardMode::Decode => self.push_decode_request(request),
        }
    }

    fn push_prefill_request(&mut self, request: &ScheduledRequest) {
        let prefix_len = request.prefix_cache_pages().len();
        let uncached_input_ids = request.uncached_input_ids();

        self.input_ids.extend_from_slice(uncached_input_ids);
        self.input_token_counts.push(uncached_input_ids.len());
        self.out_cache_pages
            .extend_from_slice(request.allocated_cache_pages());
        self.positions
            .extend(prefix_len..prefix_len + uncached_input_ids.len());
        self.sequence_lengths.push(request.input_ids().len());
    }

    fn push_decode_request(&mut self, request: &ScheduledRequest) {
        let decode_token = request.output_ids().last().copied().unwrap_or_default();

        self.input_ids.push(decode_token);
        self.input_token_counts.push(1);
        self.positions
            .push(request.input_ids().len() + request.output_ids().len() - 1);
        self.sequence_lengths
            .push(request.input_ids().len() + request.output_ids().len());
    }
}

pub trait ForwardModel {
    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelForwardOutput {
    logits: Vec<Vec<f32>>,
}

impl ModelForwardOutput {
    pub fn new(logits: Vec<Vec<f32>>) -> Result<Self, ModelForwardError> {
        let Some(first_row) = logits.first() else {
            return Ok(Self { logits });
        };
        let vocab_size = first_row.len();
        if vocab_size == 0 {
            return Err(ModelForwardError::EmptyVocabulary);
        }
        if logits.iter().any(|row| row.len() != vocab_size) {
            return Err(ModelForwardError::RaggedLogits);
        }

        Ok(Self { logits })
    }

    pub fn logits(&self) -> &[Vec<f32>] {
        &self.logits
    }

    fn into_argmax_tokens(self) -> Result<Vec<u32>, ModelForwardError> {
        self.logits
            .into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, left), (_, right)| left.total_cmp(right))
                    .map(|(index, _)| index as u32)
                    .ok_or(ModelForwardError::EmptyVocabulary)
            })
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelForwardError {
    EmptyVocabulary,
    RaggedLogits,
    Runtime(String),
    BatchSizeMismatch {
        request_count: usize,
        output_count: usize,
    },
}

impl fmt::Display for ModelForwardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyVocabulary => {
                formatter.write_str("model forward output has empty vocabulary")
            }
            Self::RaggedLogits => formatter.write_str("model forward output logits are ragged"),
            Self::Runtime(message) => formatter.write_str(message),
            Self::BatchSizeMismatch {
                request_count,
                output_count,
            } => write!(
                formatter,
                "model forward output count ({output_count}) must match request count ({request_count})"
            ),
        }
    }
}

impl std::error::Error for ModelForwardError {}

pub struct ModelRunner<M> {
    model: M,
}

impl<M> ModelRunner<M> {
    pub fn new(model: M) -> Self {
        Self { model }
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }
}

impl<M> FallibleModelWorker for ModelRunner<M>
where
    M: ForwardModel,
{
    fn try_generate_batch(
        &mut self,
        batch: &ScheduleBatch,
    ) -> Result<BatchGeneratedTokens, WorkerExecutionError> {
        let worker_batch = ModelWorkerBatch::from_schedule_batch(batch);
        let forward_output = self
            .model
            .forward(&worker_batch)
            .map_err(|error| WorkerExecutionError::Runtime(error.to_string()))?;
        let token_ids = forward_output
            .into_argmax_tokens()
            .map_err(|error| WorkerExecutionError::Runtime(error.to_string()))?;

        if token_ids.len() != batch.batch_size() {
            return Err(WorkerExecutionError::Runtime(
                ModelForwardError::BatchSizeMismatch {
                    request_count: batch.batch_size(),
                    output_count: token_ids.len(),
                }
                .to_string(),
            ));
        }

        Ok(BatchGeneratedTokens::from_batch(
            batch,
            token_ids
                .into_iter()
                .map(|token_id| GeneratedToken::unfinished(vec![token_id]))
                .collect(),
        )?)
    }
}
