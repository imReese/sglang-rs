use crate::cache::CachePageId;
use crate::scheduler::{ForwardMode, ScheduleBatch, ScheduledRequest};
use crate::types::{DisaggregatedParams, RequestId, SamplingParams};
use rand::RngExt as _;
use std::fmt;

use crate::model_artifacts::{
    LocalModelArtifacts, LocalModelCheckpointCatalog, ModelArtifactError,
    SafetensorsQuantizedLinearWeightCatalog, SafetensorsQuantizedLinearWeightSpan,
    SafetensorsTensorData,
};
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
    sequence_token_ids: Vec<u32>,
    sequence_offsets: Vec<usize>,
    sequence_token_counts: Vec<usize>,
    sequence_cache_pages: Vec<CachePageId>,
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
            sequence_token_ids: Vec::new(),
            sequence_offsets: Vec::with_capacity(batch.batch_size()),
            sequence_token_counts: Vec::with_capacity(batch.batch_size()),
            sequence_cache_pages: Vec::new(),
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

    pub fn sequence_token_ids(&self) -> &[u32] {
        &self.sequence_token_ids
    }

    pub fn sequence_offsets(&self) -> &[usize] {
        &self.sequence_offsets
    }

    pub fn sequence_token_counts(&self) -> &[usize] {
        &self.sequence_token_counts
    }

    pub fn sequence_cache_pages(&self) -> &[CachePageId] {
        &self.sequence_cache_pages
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
        self.sequence_offsets.push(self.sequence_token_ids.len());
        self.sequence_token_ids
            .extend_from_slice(request.input_ids());
        self.sequence_token_counts.push(request.input_ids().len());
        self.sequence_cache_pages
            .extend_from_slice(request.sequence_cache_pages());

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
        self.sequence_offsets.push(self.sequence_token_ids.len());
        self.sequence_token_ids
            .extend_from_slice(request.input_ids());
        self.sequence_token_ids
            .extend_from_slice(request.output_ids());
        self.sequence_token_counts
            .push(request.input_ids().len() + request.output_ids().len());
        self.sequence_cache_pages
            .extend_from_slice(request.sequence_cache_pages());

        self.input_ids.push(decode_token);
        self.input_token_counts.push(1);
        self.out_cache_pages
            .extend_from_slice(request.allocated_cache_pages());
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
pub struct CpuEmbeddingLmModel {
    token_embeddings: Vec<f32>,
    lm_head: Vec<f32>,
    vocab_size: usize,
    hidden_size: usize,
}

impl CpuEmbeddingLmModel {
    pub fn from_local_model_artifacts(
        artifacts: &LocalModelArtifacts,
    ) -> Result<Option<Self>, ModelArtifactError> {
        if artifacts.config().model_type.as_deref() != Some("sglang_embedding_lm") {
            return Ok(None);
        }

        let vocab_size = artifacts.config().vocab_size.ok_or_else(|| {
            invalid_cpu_lm_artifact(artifacts, "missing vocab_size for CPU embedding LM")
        })?;
        let hidden_size = artifacts.config().hidden_size.ok_or_else(|| {
            invalid_cpu_lm_artifact(artifacts, "missing hidden_size for CPU embedding LM")
        })?;
        let checkpoint = LocalModelCheckpointCatalog::from_local_model_artifacts(artifacts)?;
        let quantized_linears = checkpoint.quantized_linear_weights();
        let token_embeddings = required_float_matrix(
            artifacts,
            quantized_linears,
            "model.embed_tokens.weight",
            vocab_size,
            hidden_size,
        )?;
        let lm_head = required_float_matrix(
            artifacts,
            quantized_linears,
            "lm_head.weight",
            vocab_size,
            hidden_size,
        )?;

        Ok(Some(Self {
            token_embeddings,
            lm_head,
            vocab_size,
            hidden_size,
        }))
    }

    fn logits_for_token(&self, token_id: u32) -> Result<Vec<f32>, ModelForwardError> {
        let token_id = usize::try_from(token_id).map_err(|_| {
            ModelForwardError::Runtime(format!("token id {token_id} does not fit usize"))
        })?;
        if token_id >= self.vocab_size {
            return Err(ModelForwardError::Runtime(format!(
                "token id {token_id} is outside CPU embedding LM vocabulary {}",
                self.vocab_size
            )));
        }

        let hidden_offset = token_id * self.hidden_size;
        let hidden = &self.token_embeddings[hidden_offset..hidden_offset + self.hidden_size];
        let mut logits = Vec::with_capacity(self.vocab_size);
        for output_token_id in 0..self.vocab_size {
            let row_offset = output_token_id * self.hidden_size;
            let row = &self.lm_head[row_offset..row_offset + self.hidden_size];
            logits.push(dot_product(hidden, row));
        }
        Ok(logits)
    }
}

impl ForwardModel for CpuEmbeddingLmModel {
    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        let logits = batch
            .last_input_token_ids()
            .into_iter()
            .map(|token_id| self.logits_for_token(token_id))
            .collect::<Result<Vec<_>, _>>()?;
        ModelForwardOutput::new(logits)
    }
}

fn required_float_matrix(
    artifacts: &LocalModelArtifacts,
    quantized_linears: &SafetensorsQuantizedLinearWeightCatalog,
    tensor_name: &str,
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>, ModelArtifactError> {
    if let Some(weight_span) = quantized_linears.span(tensor_name) {
        return required_scaled_quantized_float_matrix(
            artifacts,
            tensor_name,
            weight_span,
            rows,
            columns,
        );
    }

    required_unscaled_float_matrix(artifacts, tensor_name, rows, columns)
}

fn required_unscaled_float_matrix(
    artifacts: &LocalModelArtifacts,
    tensor_name: &str,
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>, ModelArtifactError> {
    let tensor = artifacts
        .safetensors()
        .read_tensor(tensor_name)?
        .ok_or_else(|| {
            invalid_cpu_lm_artifact(
                artifacts,
                format!("missing CPU embedding LM tensor {tensor_name}"),
            )
        })?;

    validate_cpu_lm_tensor_shape(artifacts, tensor_name, &tensor, rows, columns)?;
    tensor.decode_f32_values().map_err(|error| {
        invalid_cpu_lm_artifact(
            artifacts,
            format!("failed to decode CPU embedding LM tensor {tensor_name}: {error}"),
        )
    })
}

fn required_scaled_quantized_float_matrix(
    artifacts: &LocalModelArtifacts,
    tensor_name: &str,
    weight_span: &SafetensorsQuantizedLinearWeightSpan,
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>, ModelArtifactError> {
    let weight = artifacts
        .safetensors()
        .read_tensor(&weight_span.tensor_name)?
        .ok_or_else(|| {
            invalid_cpu_lm_artifact(
                artifacts,
                format!(
                    "missing CPU embedding LM quantized weight tensor {}",
                    weight_span.tensor_name
                ),
            )
        })?;
    validate_cpu_lm_tensor_shape(artifacts, tensor_name, &weight, rows, columns)?;
    let mut values = decode_cpu_lm_tensor_values(artifacts, &weight_span.tensor_name, &weight)?;

    let scale = artifacts
        .safetensors()
        .read_tensor(&weight_span.scale_tensor_name)?
        .ok_or_else(|| {
            invalid_cpu_lm_artifact(
                artifacts,
                format!(
                    "missing CPU embedding LM quantized scale tensor {}",
                    weight_span.scale_tensor_name
                ),
            )
        })?;
    let scale_values =
        decode_cpu_lm_tensor_values(artifacts, &weight_span.scale_tensor_name, &scale)?;
    apply_cpu_lm_matrix_scale(
        artifacts,
        tensor_name,
        &mut values,
        rows,
        columns,
        &scale.metadata.shape,
        &scale_values,
    )?;

    Ok(values)
}

fn decode_cpu_lm_tensor_values(
    artifacts: &LocalModelArtifacts,
    tensor_name: &str,
    tensor: &SafetensorsTensorData,
) -> Result<Vec<f32>, ModelArtifactError> {
    tensor.decode_f32_values().map_err(|error| {
        invalid_cpu_lm_artifact(
            artifacts,
            format!("failed to decode CPU embedding LM tensor {tensor_name}: {error}"),
        )
    })
}

fn apply_cpu_lm_matrix_scale(
    artifacts: &LocalModelArtifacts,
    tensor_name: &str,
    values: &mut [f32],
    rows: usize,
    columns: usize,
    scale_shape: &[usize],
    scale_values: &[f32],
) -> Result<(), ModelArtifactError> {
    match scale_shape {
        [1] | [1, 1] => {
            let scale = single_cpu_lm_scale(artifacts, tensor_name, scale_values)?;
            for value in values {
                *value *= scale;
            }
            Ok(())
        }
        [scale_rows] if *scale_rows == rows => {
            validate_cpu_lm_row_scale_count(artifacts, tensor_name, rows, scale_values)?;
            apply_cpu_lm_row_scales(values, columns, scale_values);
            Ok(())
        }
        [scale_rows, 1] if *scale_rows == rows => {
            validate_cpu_lm_row_scale_count(artifacts, tensor_name, rows, scale_values)?;
            apply_cpu_lm_row_scales(values, columns, scale_values);
            Ok(())
        }
        _ => Err(invalid_cpu_lm_artifact(
            artifacts,
            format!(
                "CPU embedding LM tensor {tensor_name} quantization scale shape {scale_shape:?} \
                 does not match scalar, [{rows}], or [{rows}, 1]"
            ),
        )),
    }
}

fn validate_cpu_lm_row_scale_count(
    artifacts: &LocalModelArtifacts,
    tensor_name: &str,
    rows: usize,
    scale_values: &[f32],
) -> Result<(), ModelArtifactError> {
    if scale_values.len() != rows {
        return Err(invalid_cpu_lm_artifact(
            artifacts,
            format!(
                "CPU embedding LM tensor {tensor_name} row quantization scale has {} values but expected {rows}",
                scale_values.len()
            ),
        ));
    }
    Ok(())
}

fn single_cpu_lm_scale(
    artifacts: &LocalModelArtifacts,
    tensor_name: &str,
    scale_values: &[f32],
) -> Result<f32, ModelArtifactError> {
    match scale_values {
        [scale] => Ok(*scale),
        _ => Err(invalid_cpu_lm_artifact(
            artifacts,
            format!(
                "CPU embedding LM tensor {tensor_name} scalar quantization scale has {} values",
                scale_values.len()
            ),
        )),
    }
}

fn apply_cpu_lm_row_scales(values: &mut [f32], columns: usize, scale_values: &[f32]) {
    for (row, scale) in values.chunks_mut(columns).zip(scale_values) {
        for value in row {
            *value *= *scale;
        }
    }
}

fn validate_cpu_lm_tensor_shape(
    artifacts: &LocalModelArtifacts,
    tensor_name: &str,
    tensor: &SafetensorsTensorData,
    rows: usize,
    columns: usize,
) -> Result<(), ModelArtifactError> {
    if tensor.metadata.shape != [rows, columns] {
        return Err(invalid_cpu_lm_artifact(
            artifacts,
            format!(
                "CPU embedding LM tensor {tensor_name} shape {:?} does not match expected [{rows}, {columns}]",
                tensor.metadata.shape
            ),
        ));
    }
    Ok(())
}

fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn invalid_cpu_lm_artifact(
    artifacts: &LocalModelArtifacts,
    message: impl Into<String>,
) -> ModelArtifactError {
    ModelArtifactError::InvalidSafetensorsData {
        path: artifacts.model_path().to_path_buf(),
        message: message.into(),
    }
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
                .zip(batch.requests())
                .map(|(token_id, request)| {
                    if !request.sampling().ignore_eos
                        && request.sampling().stop_token_ids.contains(&token_id)
                    {
                        GeneratedToken::finished(vec![token_id])
                    } else {
                        GeneratedToken::unfinished(vec![token_id])
                    }
                })
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
