use crate::cache::CachePageId;
use crate::scheduler::{ForwardMode, ScheduleBatch, ScheduledRequest};
use crate::types::{DisaggregatedParams, RequestId, SamplingParams};
use rand::RngExt as _;
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

    pub fn last_input_token_ids(&self) -> Vec<u32> {
        self.request_offsets
            .iter()
            .zip(&self.input_token_counts)
            .map(|(offset, token_count)| self.input_ids[offset + token_count - 1])
            .collect()
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
        validate_logits(&logits)?;
        Ok(Self { logits })
    }

    pub fn from_token_logits(
        batch: &ModelWorkerBatch,
        token_logits: Vec<Vec<f32>>,
    ) -> Result<Self, ModelForwardError> {
        validate_logits(&token_logits)?;
        if token_logits.len() != batch.input_ids().len() {
            return Err(ModelForwardError::TokenLogitCountMismatch {
                token_count: batch.input_ids().len(),
                logit_count: token_logits.len(),
            });
        }

        let mut logits = Vec::with_capacity(batch.request_ids().len());
        for (request_index, (offset, token_count)) in batch
            .request_offsets()
            .iter()
            .zip(batch.input_token_counts())
            .enumerate()
        {
            if *token_count == 0 {
                return Err(ModelForwardError::MissingRequestTokenLogits { request_index });
            }
            logits.push(token_logits[*offset + *token_count - 1].clone());
        }

        Ok(Self { logits })
    }

    pub fn logits(&self) -> &[Vec<f32>] {
        &self.logits
    }

    fn into_logits(self) -> Vec<Vec<f32>> {
        self.logits
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
    TokenLogitCountMismatch {
        token_count: usize,
        logit_count: usize,
    },
    MissingRequestTokenLogits {
        request_index: usize,
    },
    InvalidProbabilityDistribution {
        request_index: usize,
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
            Self::TokenLogitCountMismatch {
                token_count,
                logit_count,
            } => write!(
                formatter,
                "model forward token logit count ({logit_count}) must match input token count ({token_count})"
            ),
            Self::MissingRequestTokenLogits { request_index } => write!(
                formatter,
                "model forward request {request_index} has no input token logits"
            ),
            Self::InvalidProbabilityDistribution { request_index } => write!(
                formatter,
                "model forward request {request_index} produced an invalid sampling distribution"
            ),
        }
    }
}

impl std::error::Error for ModelForwardError {}

fn validate_logits(logits: &[Vec<f32>]) -> Result<(), ModelForwardError> {
    let Some(first_row) = logits.first() else {
        return Ok(());
    };
    let vocab_size = first_row.len();
    if vocab_size == 0 {
        return Err(ModelForwardError::EmptyVocabulary);
    }
    if logits.iter().any(|row| row.len() != vocab_size) {
        return Err(ModelForwardError::RaggedLogits);
    }

    Ok(())
}

pub trait SamplingRandomSource {
    fn next_unit_f32(&mut self) -> f32;
}

#[derive(Clone, Debug, Default)]
pub struct SystemRandomSource;

impl SamplingRandomSource for SystemRandomSource {
    fn next_unit_f32(&mut self) -> f32 {
        rand::rng().random::<f32>()
    }
}

#[derive(Clone, Debug)]
pub struct LogitSampler<R = SystemRandomSource> {
    random: R,
}

impl<R> LogitSampler<R> {
    pub fn new(random: R) -> Self {
        Self { random }
    }
}

impl Default for LogitSampler<SystemRandomSource> {
    fn default() -> Self {
        Self::new(SystemRandomSource)
    }
}

impl<R> LogitSampler<R>
where
    R: SamplingRandomSource,
{
    fn sample(
        &mut self,
        output: ModelForwardOutput,
        batch: &ScheduleBatch,
    ) -> Result<Vec<u32>, ModelForwardError> {
        let logits = output.into_logits();
        if logits.len() != batch.batch_size() {
            return Err(ModelForwardError::BatchSizeMismatch {
                request_count: batch.batch_size(),
                output_count: logits.len(),
            });
        }

        logits
            .into_iter()
            .zip(batch.requests())
            .enumerate()
            .map(|(request_index, (logits, request))| {
                self.sample_row(request_index, &logits, request.sampling())
            })
            .collect()
    }

    fn sample_row(
        &mut self,
        request_index: usize,
        logits: &[f32],
        sampling: &SamplingParams,
    ) -> Result<u32, ModelForwardError> {
        if sampling.temperature.is_none()
            && sampling.top_p.is_none()
            && sampling.top_k.is_none()
            && sampling.min_p.is_none()
        {
            return argmax_token(logits);
        }

        let temperature = sampling.temperature.unwrap_or(1.0);
        if temperature <= f32::EPSILON || sampling.top_k == Some(1) {
            return argmax_token(logits);
        }

        let max_scaled_logit = logits
            .iter()
            .map(|logit| logit / temperature)
            .max_by(f32::total_cmp)
            .ok_or(ModelForwardError::EmptyVocabulary)?;
        let mut candidates = logits
            .iter()
            .enumerate()
            .map(|(token_id, logit)| (token_id, ((logit / temperature) - max_scaled_logit).exp()))
            .collect::<Vec<_>>();

        if candidates
            .iter()
            .any(|(_, probability)| !probability.is_finite() || *probability < 0.0)
        {
            return Err(ModelForwardError::InvalidProbabilityDistribution { request_index });
        }

        candidates.sort_by(|(_, left), (_, right)| right.total_cmp(left));

        if let Some(top_k) = sampling.top_k.and_then(|top_k| usize::try_from(top_k).ok())
            && top_k > 0
            && top_k < candidates.len()
        {
            candidates.truncate(top_k);
        }

        let top_p = sampling.top_p.unwrap_or(1.0);
        let min_p = sampling.min_p.unwrap_or(0.0);
        let max_probability = candidates
            .first()
            .map(|(_, probability)| *probability)
            .ok_or(ModelForwardError::EmptyVocabulary)?;
        let min_probability = max_probability * min_p;
        let mut cumulative_probability = 0.0;
        let mut filtered = Vec::with_capacity(candidates.len());

        for (token_id, probability) in candidates {
            let keep_for_top_p = cumulative_probability <= top_p;
            cumulative_probability += probability;

            if keep_for_top_p && probability >= min_probability {
                filtered.push((token_id, probability));
            }
        }

        let total_probability = filtered
            .iter()
            .map(|(_, probability)| *probability)
            .sum::<f32>();
        if !total_probability.is_finite() || total_probability <= 0.0 {
            return Err(ModelForwardError::InvalidProbabilityDistribution { request_index });
        }

        let random = self.next_bounded_unit_f32();
        let target = random * total_probability;
        let mut cumulative_probability = 0.0;
        for (token_id, probability) in filtered.iter() {
            cumulative_probability += *probability;
            if target < cumulative_probability {
                return Ok(*token_id as u32);
            }
        }

        filtered
            .last()
            .map(|(token_id, _)| *token_id as u32)
            .ok_or(ModelForwardError::InvalidProbabilityDistribution { request_index })
    }

    fn next_bounded_unit_f32(&mut self) -> f32 {
        let random = self.random.next_unit_f32();
        if random.is_finite() {
            random.clamp(0.0, f32::from_bits(0x3f7f_ffff))
        } else {
            0.0
        }
    }
}

fn argmax_token(logits: &[f32]) -> Result<u32, ModelForwardError> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index as u32)
        .ok_or(ModelForwardError::EmptyVocabulary)
}

pub struct ModelRunner<M, S = LogitSampler<SystemRandomSource>> {
    model: M,
    sampler: S,
}

impl<M> ModelRunner<M, LogitSampler<SystemRandomSource>> {
    pub fn new(model: M) -> Self {
        Self::with_sampler(model, LogitSampler::default())
    }
}

impl<M, S> ModelRunner<M, S> {
    pub fn with_sampler(model: M, sampler: S) -> Self {
        Self { model, sampler }
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }
}

impl<M, S> FallibleModelWorker for ModelRunner<M, S>
where
    M: ForwardModel,
    S: LogitSamplerLike,
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
        let token_ids = self
            .sampler
            .sample(forward_output, batch)
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

pub trait LogitSamplerLike {
    fn sample(
        &mut self,
        output: ModelForwardOutput,
        batch: &ScheduleBatch,
    ) -> Result<Vec<u32>, ModelForwardError>;
}

impl<R> LogitSamplerLike for LogitSampler<R>
where
    R: SamplingRandomSource,
{
    fn sample(
        &mut self,
        output: ModelForwardOutput,
        batch: &ScheduleBatch,
    ) -> Result<Vec<u32>, ModelForwardError> {
        LogitSampler::sample(self, output, batch)
    }
}
