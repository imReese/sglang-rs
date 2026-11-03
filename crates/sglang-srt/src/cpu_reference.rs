use std::collections::HashMap;
use std::fmt;

use sglang_kernel::cpu::{
    GroupedQueryAttentionShape, gemma_rms_norm, grouped_query_attention, linear, rms_norm,
    silu_and_mul,
};
use sglang_kernel::rotary::{
    RotaryEmbeddingConfig, RotaryEmbeddingStyle, apply_rotary_embedding_inplace,
};

use crate::backend::{RuntimeCapability, RuntimeDtype};
use crate::cache::CachePageId;
use crate::kv_cache::{KvCacheDtype, PagedKvCacheLayout};
use crate::model_artifacts::{LocalModelArtifacts, ModelArtifactError};
use crate::model_executor::{
    ModelForwardError, ModelForwardOutput, ModelWorkerBatch, validate_model_worker_batch,
};
use crate::model_runtime::BackendModelExecutor;
use crate::models::{
    AttentionArchitecture, DenseDecoderActivation, DenseDecoderExecutionPlan,
    DenseDecoderLayerWeightNames, FeedForwardArchitecture, ModelDefinition,
    ModelExecutionArchitecture,
};
use crate::runtime_kv_cache::{ActiveKvCache, RuntimeKvCache};
use crate::transfer::{KvCacheTransferError, TransferableKvCacheMemory};

#[derive(Debug)]
pub(crate) struct CpuReferenceDenseDecoder {
    plan: DenseDecoderExecutionPlan,
    query_head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    intermediate_size: usize,
    token_embeddings: FloatMatrix,
    final_norm: Vec<f32>,
    lm_head: Option<FloatMatrix>,
    layers: Vec<DenseDecoderLayerWeights>,
}

impl CpuReferenceDenseDecoder {
    pub(crate) fn load(
        definition: &ModelDefinition,
        artifacts: &LocalModelArtifacts,
    ) -> Result<Self, CpuReferenceDenseDecoderError> {
        definition.validate_dense_decoder_checkpoint(artifacts)?;
        let plan = definition
            .dense_decoder()
            .ok_or_else(|| {
                CpuReferenceDenseDecoderError::Unsupported(
                    "model definition does not include a dense decoder execution plan".to_string(),
                )
            })?
            .clone();
        let ModelExecutionArchitecture::Transformer {
            attention:
                AttentionArchitecture::MultiHead {
                    num_attention_heads,
                    num_key_value_heads,
                    head_dim,
                },
            feed_forward: FeedForwardArchitecture::Dense { intermediate_size },
        } = definition.execution()
        else {
            return Err(CpuReferenceDenseDecoderError::Unsupported(
                "CPU reference dense decoder requires multi-head attention and dense feed-forward components"
                    .to_string(),
            ));
        };

        let query_size = checked_product(num_attention_heads, head_dim, "query projection")?;
        let kv_size = checked_product(num_key_value_heads, head_dim, "KV projection")?;
        let token_embeddings = FloatMatrix::load(
            artifacts,
            &plan.weights.token_embeddings,
            plan.vocab_size,
            plan.hidden_size,
        )?;
        let final_norm = load_vector(artifacts, &plan.weights.final_norm, plan.hidden_size)?;
        let lm_head = plan
            .weights
            .lm_head
            .as_deref()
            .map(|name| FloatMatrix::load(artifacts, name, plan.vocab_size, plan.hidden_size))
            .transpose()?;
        let layers = plan
            .weights
            .layers
            .iter()
            .map(|names| {
                DenseDecoderLayerWeights::load(
                    artifacts,
                    names,
                    plan.hidden_size,
                    query_size,
                    kv_size,
                    head_dim,
                    intermediate_size,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            plan,
            query_head_count: num_attention_heads,
            kv_head_count: num_key_value_heads,
            head_dim,
            intermediate_size,
            token_embeddings,
            final_norm,
            lm_head,
            layers,
        })
    }

    pub(crate) fn runtime_capability(&self) -> RuntimeCapability {
        RuntimeCapability::cpu_reference("cpu-reference-dense-decoder", false)
    }

    pub(crate) fn execution_dtype(&self) -> RuntimeDtype {
        RuntimeDtype::F32
    }

    fn forward_token(
        &mut self,
        kv_cache: &mut CpuReferenceKvCache,
        token_id: u32,
        position: usize,
        output_slot: CachePageId,
        sequence_slots: &[CachePageId],
    ) -> Result<Vec<f32>, CpuReferenceDenseDecoderError> {
        if position >= self.plan.max_position_embeddings {
            return Err(CpuReferenceDenseDecoderError::Execution(format!(
                "token position {position} exceeds max_position_embeddings {}",
                self.plan.max_position_embeddings
            )));
        }
        if sequence_slots.get(position) != Some(&output_slot) {
            return Err(CpuReferenceDenseDecoderError::Execution(format!(
                "output KV slot {} does not match sequence slot at position {position}",
                output_slot.as_usize()
            )));
        }

        let mut hidden = self.token_embeddings.row(token_id)?;
        let rotary_embedding =
            RotaryEmbeddingConfig::standard(self.plan.rope_theta, RotaryEmbeddingStyle::Neox)
                .and_then(|rotary| {
                    rotary.validate(self.head_dim)?;
                    Ok(rotary)
                })
                .map_err(|error| CpuReferenceDenseDecoderError::Execution(error.to_string()))?;
        for layer_index in 0..self.layers.len() {
            hidden = forward_layer(
                &self.layers[layer_index],
                kv_cache,
                &hidden,
                DenseDecoderForwardContext {
                    layer_index,
                    position,
                    output_slot,
                    sequence_slots: &sequence_slots[..=position],
                    shape: DenseDecoderShape {
                        hidden_size: self.plan.hidden_size,
                        intermediate_size: self.intermediate_size,
                        query_head_count: self.query_head_count,
                        kv_head_count: self.kv_head_count,
                        head_dim: self.head_dim,
                    },
                    rms_norm_eps: self.plan.rms_norm_eps,
                    rotary_embedding,
                    activation: self.plan.activation,
                },
            )?;
        }

        let hidden = rms_norm(
            &hidden,
            &self.final_norm,
            1,
            self.plan.hidden_size,
            self.plan.rms_norm_eps,
        )
        .map_err(kernel_error)?;
        let lm_head = self.lm_head.as_ref().unwrap_or(&self.token_embeddings);
        lm_head.project(&hidden, 1)
    }
}

impl BackendModelExecutor<RuntimeKvCache<CpuReferenceKvCache>> for CpuReferenceDenseDecoder {
    fn runtime_capability(&self) -> RuntimeCapability {
        CpuReferenceDenseDecoder::runtime_capability(self)
    }

    fn execution_dtype(&self) -> RuntimeDtype {
        CpuReferenceDenseDecoder::execution_dtype(self)
    }

    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
        resources: &mut RuntimeKvCache<CpuReferenceKvCache>,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        let kv_cache = resources.allocation_mut();
        validate_batch(batch).map_err(model_forward_error)?;
        let mut logits = Vec::with_capacity(batch.request_ids().len());
        for request_index in 0..batch.request_ids().len() {
            let input_offset = batch.request_offsets()[request_index];
            let input_count = batch.input_token_counts()[request_index];
            let sequence_offset = batch.sequence_offsets()[request_index];
            let sequence_count = batch.sequence_token_counts()[request_index];
            let sequence_slots =
                &batch.sequence_cache_pages()[sequence_offset..sequence_offset + sequence_count];
            let mut request_logits = None;
            for input_index in input_offset..input_offset + input_count {
                request_logits = Some(
                    self.forward_token(
                        kv_cache,
                        batch.input_ids()[input_index],
                        batch.positions()[input_index],
                        batch.out_cache_pages()[input_index],
                        sequence_slots,
                    )
                    .map_err(model_forward_error)?,
                );
            }
            logits.push(request_logits.ok_or_else(|| {
                ModelForwardError::Runtime(format!(
                    "dense decoder request {request_index} has no input tokens"
                ))
            })?);
        }
        ModelForwardOutput::new(logits)
    }
}

#[derive(Clone, Copy)]
struct DenseDecoderShape {
    hidden_size: usize,
    intermediate_size: usize,
    query_head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
}

#[derive(Clone, Copy)]
struct DenseDecoderForwardContext<'a> {
    layer_index: usize,
    position: usize,
    output_slot: CachePageId,
    sequence_slots: &'a [CachePageId],
    shape: DenseDecoderShape,
    rms_norm_eps: f32,
    rotary_embedding: RotaryEmbeddingConfig,
    activation: DenseDecoderActivation,
}

fn forward_layer(
    layer: &DenseDecoderLayerWeights,
    kv_cache: &mut CpuReferenceKvCache,
    hidden: &[f32],
    context: DenseDecoderForwardContext<'_>,
) -> Result<Vec<f32>, CpuReferenceDenseDecoderError> {
    let DenseDecoderForwardContext {
        layer_index,
        position,
        output_slot,
        sequence_slots,
        shape,
        rms_norm_eps,
        rotary_embedding,
        activation,
    } = context;
    let normalized = rms_norm(
        hidden,
        &layer.input_norm,
        1,
        shape.hidden_size,
        rms_norm_eps,
    )
    .map_err(kernel_error)?;
    let attention_output = forward_full_attention(
        &layer.attention,
        kv_cache,
        &normalized,
        FullAttentionForwardContext {
            cache_layer_index: layer_index,
            position,
            output_slot,
            sequence_slots,
            shape: FullAttentionShape {
                query_head_count: shape.query_head_count,
                kv_head_count: shape.kv_head_count,
                head_dim: shape.head_dim,
                rotary_dim: shape.head_dim,
                output_gate: false,
            },
            rms_norm_eps,
            rotary_embedding,
            qk_normalization: CpuNormalization::Standard,
        },
    )?;
    let after_attention = add_residual(hidden, &attention_output)?;

    let normalized = rms_norm(
        &after_attention,
        &layer.post_attention_norm,
        1,
        shape.hidden_size,
        rms_norm_eps,
    )
    .map_err(kernel_error)?;
    let feed_forward =
        layer
            .feed_forward
            .forward(&normalized, shape.intermediate_size, activation)?;
    add_residual(&after_attention, &feed_forward)
}

pub(crate) fn validate_batch(
    batch: &ModelWorkerBatch,
) -> Result<(), CpuReferenceDenseDecoderError> {
    validate_model_worker_batch(batch)
        .map_err(|error| CpuReferenceDenseDecoderError::Execution(error.to_string()))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CpuNormalization {
    Standard,
    Gemma,
}

pub(crate) fn apply_normalization(
    input: &[f32],
    weight: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
    normalization: CpuNormalization,
) -> Result<Vec<f32>, CpuReferenceDenseDecoderError> {
    match normalization {
        CpuNormalization::Standard => rms_norm(input, weight, rows, cols, eps),
        CpuNormalization::Gemma => gemma_rms_norm(input, weight, rows, cols, eps),
    }
    .map_err(kernel_error)
}

pub(crate) struct FullAttentionWeightNamesRef<'a> {
    pub(crate) query_weight: &'a str,
    pub(crate) query_bias: Option<&'a str>,
    pub(crate) query_norm: Option<&'a str>,
    pub(crate) key_weight: &'a str,
    pub(crate) key_bias: Option<&'a str>,
    pub(crate) key_norm: Option<&'a str>,
    pub(crate) value_weight: &'a str,
    pub(crate) value_bias: Option<&'a str>,
    pub(crate) output_weight: &'a str,
    pub(crate) output_bias: Option<&'a str>,
}

#[derive(Clone, Copy)]
pub(crate) struct FullAttentionShape {
    pub(crate) query_head_count: usize,
    pub(crate) kv_head_count: usize,
    pub(crate) head_dim: usize,
    pub(crate) rotary_dim: usize,
    pub(crate) output_gate: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct FullAttentionForwardContext<'a> {
    pub(crate) cache_layer_index: usize,
    pub(crate) position: usize,
    pub(crate) output_slot: CachePageId,
    pub(crate) sequence_slots: &'a [CachePageId],
    pub(crate) shape: FullAttentionShape,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rotary_embedding: RotaryEmbeddingConfig,
    pub(crate) qk_normalization: CpuNormalization,
}

#[derive(Debug)]
pub(crate) struct FullAttentionWeights {
    query: FloatMatrix,
    query_norm: Option<Vec<f32>>,
    key: FloatMatrix,
    key_norm: Option<Vec<f32>>,
    value: FloatMatrix,
    output: FloatMatrix,
}

impl FullAttentionWeights {
    pub(crate) fn load(
        artifacts: &LocalModelArtifacts,
        names: FullAttentionWeightNamesRef<'_>,
        hidden_size: usize,
        projected_query_size: usize,
        query_size: usize,
        kv_size: usize,
        head_dim: usize,
    ) -> Result<Self, CpuReferenceDenseDecoderError> {
        Ok(Self {
            query: FloatMatrix::load_with_bias(
                artifacts,
                names.query_weight,
                names.query_bias,
                projected_query_size,
                hidden_size,
            )?,
            query_norm: names
                .query_norm
                .map(|name| load_vector(artifacts, name, head_dim))
                .transpose()?,
            key: FloatMatrix::load_with_bias(
                artifacts,
                names.key_weight,
                names.key_bias,
                kv_size,
                hidden_size,
            )?,
            key_norm: names
                .key_norm
                .map(|name| load_vector(artifacts, name, head_dim))
                .transpose()?,
            value: FloatMatrix::load_with_bias(
                artifacts,
                names.value_weight,
                names.value_bias,
                kv_size,
                hidden_size,
            )?,
            output: FloatMatrix::load_with_bias(
                artifacts,
                names.output_weight,
                names.output_bias,
                hidden_size,
                query_size,
            )?,
        })
    }
}

pub(crate) fn forward_full_attention(
    weights: &FullAttentionWeights,
    kv_cache: &mut CpuReferenceKvCache,
    hidden: &[f32],
    context: FullAttentionForwardContext<'_>,
) -> Result<Vec<f32>, CpuReferenceDenseDecoderError> {
    let FullAttentionForwardContext {
        cache_layer_index,
        position,
        output_slot,
        sequence_slots,
        shape,
        rms_norm_eps,
        rotary_embedding,
        qk_normalization,
    } = context;
    let projected_query = weights.query.project(hidden, 1)?;
    let query_size = checked_product(
        shape.query_head_count,
        shape.head_dim,
        "full attention query",
    )?;
    let (mut query, output_gate) = if shape.output_gate {
        if projected_query.len() != query_size * 2 {
            return Err(CpuReferenceDenseDecoderError::Execution(format!(
                "gated query width {} does not match expected {}",
                projected_query.len(),
                query_size * 2
            )));
        }
        let mut query = Vec::with_capacity(query_size);
        let mut gate = Vec::with_capacity(query_size);
        for head in 0..shape.query_head_count {
            let offset = head * shape.head_dim * 2;
            query.extend_from_slice(&projected_query[offset..offset + shape.head_dim]);
            gate.extend_from_slice(
                &projected_query[offset + shape.head_dim..offset + shape.head_dim * 2],
            );
        }
        (query, Some(gate))
    } else {
        (projected_query, None)
    };
    let mut key = weights.key.project(hidden, 1)?;
    let value = weights.value.project(hidden, 1)?;
    if let Some(weight) = &weights.query_norm {
        query = apply_normalization(
            &query,
            weight,
            shape.query_head_count,
            shape.head_dim,
            rms_norm_eps,
            qk_normalization,
        )?;
    }
    if let Some(weight) = &weights.key_norm {
        key = apply_normalization(
            &key,
            weight,
            shape.kv_head_count,
            shape.head_dim,
            rms_norm_eps,
            qk_normalization,
        )?;
    }
    apply_rotary_embedding_inplace(
        &mut query,
        shape.query_head_count,
        shape.head_dim,
        shape.rotary_dim,
        position,
        rotary_embedding,
    )
    .map_err(|error| CpuReferenceDenseDecoderError::Execution(error.to_string()))?;
    apply_rotary_embedding_inplace(
        &mut key,
        shape.kv_head_count,
        shape.head_dim,
        shape.rotary_dim,
        position,
        rotary_embedding,
    )
    .map_err(|error| CpuReferenceDenseDecoderError::Execution(error.to_string()))?;
    kv_cache.write(cache_layer_index, output_slot, key, value)?;
    let (keys, values) = kv_cache.gather(cache_layer_index, sequence_slots)?;
    let mut attention = grouped_query_attention(
        &query,
        &keys,
        &values,
        GroupedQueryAttentionShape {
            token_count: sequence_slots.len(),
            query_head_count: shape.query_head_count,
            kv_head_count: shape.kv_head_count,
            head_dim: shape.head_dim,
            scale: (shape.head_dim as f32).sqrt().recip(),
        },
    )
    .map_err(kernel_error)?;
    if let Some(gate) = output_gate {
        for (value, gate) in attention.iter_mut().zip(gate) {
            *value *= 1.0 / (1.0 + (-gate).exp());
        }
    }
    weights.output.project(&attention, 1)
}

#[derive(Debug)]
pub(crate) struct DenseFeedForwardWeights {
    gate: FloatMatrix,
    up: FloatMatrix,
    down: FloatMatrix,
}

impl DenseFeedForwardWeights {
    pub(crate) fn from_values(
        hidden_size: usize,
        intermediate_size: usize,
        gate: Vec<f32>,
        up: Vec<f32>,
        down: Vec<f32>,
    ) -> Result<Self, CpuReferenceDenseDecoderError> {
        Ok(Self {
            gate: FloatMatrix::try_from_values(intermediate_size, hidden_size, gate)?,
            up: FloatMatrix::try_from_values(intermediate_size, hidden_size, up)?,
            down: FloatMatrix::try_from_values(hidden_size, intermediate_size, down)?,
        })
    }

    pub(crate) fn load(
        artifacts: &LocalModelArtifacts,
        gate_weight: &str,
        up_weight: &str,
        down_weight: &str,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self, CpuReferenceDenseDecoderError> {
        Ok(Self {
            gate: FloatMatrix::load(artifacts, gate_weight, intermediate_size, hidden_size)?,
            up: FloatMatrix::load(artifacts, up_weight, intermediate_size, hidden_size)?,
            down: FloatMatrix::load(artifacts, down_weight, hidden_size, intermediate_size)?,
        })
    }

    pub(crate) fn forward(
        &self,
        input: &[f32],
        intermediate_size: usize,
        activation: DenseDecoderActivation,
    ) -> Result<Vec<f32>, CpuReferenceDenseDecoderError> {
        let gate = self.gate.project(input, 1)?;
        let up = self.up.project(input, 1)?;
        let activated = match activation {
            DenseDecoderActivation::Silu => silu_and_mul(&gate, &up).map_err(kernel_error)?,
        };
        if activated.len() != intermediate_size {
            return Err(CpuReferenceDenseDecoderError::Execution(format!(
                "dense activation width {} does not match intermediate size {intermediate_size}",
                activated.len()
            )));
        }
        self.down.project(&activated, 1)
    }
}

#[derive(Debug)]
struct DenseDecoderLayerWeights {
    input_norm: Vec<f32>,
    attention: FullAttentionWeights,
    post_attention_norm: Vec<f32>,
    feed_forward: DenseFeedForwardWeights,
}

impl DenseDecoderLayerWeights {
    fn load(
        artifacts: &LocalModelArtifacts,
        names: &DenseDecoderLayerWeightNames,
        hidden_size: usize,
        query_size: usize,
        kv_size: usize,
        head_dim: usize,
        intermediate_size: usize,
    ) -> Result<Self, CpuReferenceDenseDecoderError> {
        Ok(Self {
            input_norm: load_vector(artifacts, &names.input_norm, hidden_size)?,
            attention: FullAttentionWeights::load(
                artifacts,
                FullAttentionWeightNamesRef {
                    query_weight: &names.query_weight,
                    query_bias: names.query_bias.as_deref(),
                    query_norm: names.query_norm.as_deref(),
                    key_weight: &names.key_weight,
                    key_bias: names.key_bias.as_deref(),
                    key_norm: names.key_norm.as_deref(),
                    value_weight: &names.value_weight,
                    value_bias: names.value_bias.as_deref(),
                    output_weight: &names.output_weight,
                    output_bias: names.output_bias.as_deref(),
                },
                hidden_size,
                query_size,
                query_size,
                kv_size,
                head_dim,
            )?,
            post_attention_norm: load_vector(artifacts, &names.post_attention_norm, hidden_size)?,
            feed_forward: DenseFeedForwardWeights::load(
                artifacts,
                &names.gate_weight,
                &names.up_weight,
                &names.down_weight,
                hidden_size,
                intermediate_size,
            )?,
        })
    }
}

#[derive(Debug)]
pub(crate) struct FloatMatrix {
    rows: usize,
    columns: usize,
    values: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl FloatMatrix {
    fn try_from_values(
        rows: usize,
        columns: usize,
        values: Vec<f32>,
    ) -> Result<Self, CpuReferenceDenseDecoderError> {
        let expected = rows.checked_mul(columns).ok_or_else(|| {
            CpuReferenceDenseDecoderError::Unsupported(
                "float matrix element count overflowed".to_string(),
            )
        })?;
        if values.len() != expected {
            return Err(CpuReferenceDenseDecoderError::Unsupported(format!(
                "float matrix has {} values, expected {expected} for shape [{rows}, {columns}]",
                values.len()
            )));
        }
        Ok(Self {
            rows,
            columns,
            values,
            bias: None,
        })
    }

    #[cfg(test)]
    pub(crate) fn from_values(rows: usize, columns: usize, values: Vec<f32>) -> Self {
        Self::try_from_values(rows, columns, values).expect("test matrix shape should be valid")
    }

    pub(crate) fn load(
        artifacts: &LocalModelArtifacts,
        tensor_name: &str,
        rows: usize,
        columns: usize,
    ) -> Result<Self, CpuReferenceDenseDecoderError> {
        Self::load_with_bias(artifacts, tensor_name, None, rows, columns)
    }

    pub(crate) fn load_with_bias(
        artifacts: &LocalModelArtifacts,
        tensor_name: &str,
        bias_name: Option<&str>,
        rows: usize,
        columns: usize,
    ) -> Result<Self, CpuReferenceDenseDecoderError> {
        let values = load_tensor(artifacts, tensor_name, &[rows, columns])?;
        let bias = bias_name
            .map(|name| load_vector(artifacts, name, rows))
            .transpose()?;
        Ok(Self {
            rows,
            columns,
            values,
            bias,
        })
    }

    pub(crate) fn project(
        &self,
        input: &[f32],
        input_rows: usize,
    ) -> Result<Vec<f32>, CpuReferenceDenseDecoderError> {
        linear(
            input,
            &self.values,
            self.bias.as_deref(),
            input_rows,
            self.columns,
            self.rows,
        )
        .map_err(kernel_error)
    }

    pub(crate) fn row(&self, row: u32) -> Result<Vec<f32>, CpuReferenceDenseDecoderError> {
        let row = usize::try_from(row).map_err(|_| {
            CpuReferenceDenseDecoderError::Execution(format!("token id {row} does not fit usize"))
        })?;
        if row >= self.rows {
            return Err(CpuReferenceDenseDecoderError::Execution(format!(
                "token id {row} is outside vocabulary {}",
                self.rows
            )));
        }
        Ok(self.values[row * self.columns..(row + 1) * self.columns].to_vec())
    }
}

#[derive(Debug)]
pub(crate) struct CpuReferenceKvCache {
    layout: PagedKvCacheLayout,
    layers: Vec<HashMap<usize, KvSlot>>,
    key_width: usize,
    value_width: usize,
}

impl CpuReferenceKvCache {
    pub(crate) fn new(layout: PagedKvCacheLayout) -> Result<Self, CpuReferenceDenseDecoderError> {
        let runtime = layout.runtime();
        if runtime.dtype != KvCacheDtype::Float32 {
            return Err(CpuReferenceDenseDecoderError::Unsupported(format!(
                "CPU reference KV cache requires f32 storage, received {:?}",
                runtime.dtype
            )));
        }
        let key_bytes = layout.tensor_token_size_bytes(0).map_err(|error| {
            CpuReferenceDenseDecoderError::Unsupported(format!(
                "invalid CPU reference key tensor layout: {error}"
            ))
        })?;
        let value_bytes = layout.tensor_token_size_bytes(1).map_err(|error| {
            CpuReferenceDenseDecoderError::Unsupported(format!(
                "invalid CPU reference value tensor layout: {error}"
            ))
        })?;
        if !key_bytes.is_multiple_of(std::mem::size_of::<f32>())
            || !value_bytes.is_multiple_of(std::mem::size_of::<f32>())
        {
            return Err(CpuReferenceDenseDecoderError::Unsupported(format!(
                "CPU reference KV tensor byte widths {key_bytes}/{value_bytes} are not f32 aligned"
            )));
        }
        Ok(Self {
            layout,
            layers: (0..runtime.num_layers).map(|_| HashMap::new()).collect(),
            key_width: key_bytes / std::mem::size_of::<f32>(),
            value_width: value_bytes / std::mem::size_of::<f32>(),
        })
    }

    pub(crate) fn write(
        &mut self,
        layer_index: usize,
        slot: CachePageId,
        key: Vec<f32>,
        value: Vec<f32>,
    ) -> Result<(), CpuReferenceDenseDecoderError> {
        if key.len() != self.key_width || value.len() != self.value_width {
            return Err(CpuReferenceDenseDecoderError::Execution(format!(
                "KV write width {}/{} does not match expected {}/{}",
                key.len(),
                value.len(),
                self.key_width,
                self.value_width
            )));
        }
        let layer = self.layers.get_mut(layer_index).ok_or_else(|| {
            CpuReferenceDenseDecoderError::Execution(format!(
                "KV cache layer {layer_index} is out of range"
            ))
        })?;
        layer.insert(slot.as_usize(), KvSlot { key, value });
        Ok(())
    }

    pub(crate) fn gather(
        &self,
        layer_index: usize,
        slots: &[CachePageId],
    ) -> Result<(Vec<f32>, Vec<f32>), CpuReferenceDenseDecoderError> {
        let layer = self.layers.get(layer_index).ok_or_else(|| {
            CpuReferenceDenseDecoderError::Execution(format!(
                "KV cache layer {layer_index} is out of range"
            ))
        })?;
        let key_capacity = slots.len().checked_mul(self.key_width).ok_or_else(|| {
            CpuReferenceDenseDecoderError::Execution("KV gather size overflowed".to_string())
        })?;
        let value_capacity = slots.len().checked_mul(self.value_width).ok_or_else(|| {
            CpuReferenceDenseDecoderError::Execution("KV gather size overflowed".to_string())
        })?;
        let mut keys = Vec::with_capacity(key_capacity);
        let mut values = Vec::with_capacity(value_capacity);
        for slot in slots {
            let entry = layer.get(&slot.as_usize()).ok_or_else(|| {
                CpuReferenceDenseDecoderError::Execution(format!(
                    "KV cache layer {layer_index} physical slot {} is not initialized",
                    slot.as_usize()
                ))
            })?;
            keys.extend_from_slice(&entry.key);
            values.extend_from_slice(&entry.value);
        }
        Ok((keys, values))
    }
}

impl ActiveKvCache for CpuReferenceKvCache {
    fn backend(&self) -> crate::backend::RuntimeBackend {
        crate::backend::RuntimeBackend::Cpu
    }

    fn layout(&self) -> PagedKvCacheLayout {
        self.layout
    }

    fn transferable_memory(&self) -> Result<TransferableKvCacheMemory, KvCacheTransferError> {
        Err(KvCacheTransferError::Runtime(
            "CPU reference active KV memory is not a production transfer or offload tier"
                .to_string(),
        ))
    }
}

#[derive(Debug)]
struct KvSlot {
    key: Vec<f32>,
    value: Vec<f32>,
}

pub(crate) fn load_vector(
    artifacts: &LocalModelArtifacts,
    tensor_name: &str,
    length: usize,
) -> Result<Vec<f32>, CpuReferenceDenseDecoderError> {
    load_tensor(artifacts, tensor_name, &[length])
}

pub(crate) fn load_tensor(
    artifacts: &LocalModelArtifacts,
    tensor_name: &str,
    expected_shape: &[usize],
) -> Result<Vec<f32>, CpuReferenceDenseDecoderError> {
    let tensor = artifacts
        .safetensors()
        .read_tensor(tensor_name)?
        .ok_or_else(|| {
            CpuReferenceDenseDecoderError::Artifact(ModelArtifactError::InvalidSafetensorsData {
                path: artifacts.model_path().to_path_buf(),
                message: format!("missing dense decoder tensor {tensor_name}"),
            })
        })?;
    if tensor.metadata.shape != expected_shape {
        return Err(CpuReferenceDenseDecoderError::Artifact(
            ModelArtifactError::InvalidSafetensorsData {
                path: artifacts.model_path().to_path_buf(),
                message: format!(
                    "dense decoder tensor {tensor_name} shape {:?} does not match expected {expected_shape:?}",
                    tensor.metadata.shape
                ),
            },
        ));
    }
    tensor.decode_f32_values().map_err(|error| {
        CpuReferenceDenseDecoderError::Artifact(ModelArtifactError::InvalidSafetensorsData {
            path: artifacts.model_path().to_path_buf(),
            message: format!("cannot load dense decoder tensor {tensor_name}: {error}"),
        })
    })
}

pub(crate) fn add_residual(
    residual: &[f32],
    update: &[f32],
) -> Result<Vec<f32>, CpuReferenceDenseDecoderError> {
    if residual.len() != update.len() {
        return Err(CpuReferenceDenseDecoderError::Execution(format!(
            "residual width {} does not match update width {}",
            residual.len(),
            update.len()
        )));
    }
    Ok(residual
        .iter()
        .zip(update)
        .map(|(residual, update)| residual + update)
        .collect())
}

pub(crate) fn checked_product(
    left: usize,
    right: usize,
    name: &str,
) -> Result<usize, CpuReferenceDenseDecoderError> {
    left.checked_mul(right).ok_or_else(|| {
        CpuReferenceDenseDecoderError::Unsupported(format!("{name} size overflowed"))
    })
}

pub(crate) fn kernel_error(error: impl fmt::Display) -> CpuReferenceDenseDecoderError {
    CpuReferenceDenseDecoderError::Execution(error.to_string())
}

pub(crate) fn model_forward_error(error: CpuReferenceDenseDecoderError) -> ModelForwardError {
    ModelForwardError::Runtime(error.to_string())
}

#[derive(Debug)]
pub(crate) enum CpuReferenceDenseDecoderError {
    Artifact(ModelArtifactError),
    Unsupported(String),
    Execution(String),
}

impl fmt::Display for CpuReferenceDenseDecoderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Artifact(error) => write!(formatter, "{error}"),
            Self::Unsupported(message) | Self::Execution(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for CpuReferenceDenseDecoderError {}

impl From<ModelArtifactError> for CpuReferenceDenseDecoderError {
    fn from(value: ModelArtifactError) -> Self {
        Self::Artifact(value)
    }
}

#[cfg(test)]
mod pd_tests {
    use std::sync::{Arc, Condvar, Mutex};

    use super::*;
    use crate::backend_model::paged_kv_cache_layout_for_model;
    use crate::cache::{CachePageAllocator, RadixCache};
    use crate::model_executor::{ForwardModel, KvCacheAllocationConfig, ModelRunner};
    use crate::model_registry::ModelRegistry;
    use crate::mooncake_control::{MooncakeDecodeReceiver, completion_frames, send_frames};
    use crate::scheduler::{ForwardMode, PrefixCachePolicy, ScheduledRequest, Scheduler};
    use crate::transfer::{
        DecodeBootstrapMetadataPublishSummary, DecodeBootstrapPublisher, DecodeBootstrapRegistry,
        KvCacheTransferPlan, KvCacheTransferSpan, KvTransferBackend, KvTransferCompletionEvent,
        KvTransferModelWorker, KvTransferPoll,
    };
    use crate::types::{DisaggregatedParams, RequestId, SamplingParams};

    mod fixture {
        include!("../tests/support/qwen3_nonzero.rs");
    }

    type Resources = Arc<Mutex<RuntimeKvCache<CpuReferenceKvCache>>>;
    type Logits = Arc<Mutex<Vec<(ForwardMode, Vec<Vec<f32>>)>>>;

    struct ReferenceModel {
        decoder: CpuReferenceDenseDecoder,
        cache: Resources,
        logits: Logits,
    }

    impl ForwardModel for ReferenceModel {
        fn forward(
            &mut self,
            batch: &ModelWorkerBatch,
        ) -> Result<ModelForwardOutput, ModelForwardError> {
            let output = self
                .decoder
                .forward(batch, &mut *self.cache.lock().expect("cache"))?;
            self.logits
                .lock()
                .expect("logits")
                .push((batch.forward_mode(), output.logits().to_vec()));
            Ok(output)
        }
    }

    fn model(path: &std::path::Path) -> (ModelRunner<ReferenceModel>, Resources, Logits) {
        let artifacts = LocalModelArtifacts::from_model_path(path).expect("load fixture");
        let definition = ModelRegistry
            .definition(path, artifacts.config())
            .expect("Qwen adapter definition");
        let layout = paged_kv_cache_layout_for_model(
            definition.kv_cache_layout().expect("KV geometry"),
            KvCacheDtype::Float32,
            KvCacheAllocationConfig {
                slot_capacity: 64,
                page_size: 1,
            },
        )
        .expect("layout");
        let cache = Arc::new(Mutex::new(RuntimeKvCache::new(
            CpuReferenceKvCache::new(layout).expect("KV storage"),
        )));
        let logits = Arc::new(Mutex::new(Vec::new()));
        let model = ReferenceModel {
            decoder: CpuReferenceDenseDecoder::load(&definition, &artifacts)
                .expect("full Transformer"),
            cache: Arc::clone(&cache),
            logits: Arc::clone(&logits),
        };
        (ModelRunner::new(model), cache, logits)
    }

    struct Destination {
        slots: Vec<CachePageId>,
        endpoint: String,
    }
    type Rendezvous = Arc<(Mutex<Option<Destination>>, Condvar)>;

    // A test-only host payload copy exercises the real reference decoder. It does
    // not implement or claim Mooncake's CUDA/RDMA data plane.
    struct ReferencePageCopy {
        source: Resources,
        destination: Resources,
        rendezvous: Rendezvous,
        corrupt: bool,
        pending: Vec<KvTransferCompletionEvent>,
    }

    impl KvTransferBackend for ReferencePageCopy {
        fn register(&mut self, _: TransferableKvCacheMemory) -> Result<(), KvCacheTransferError> {
            Err(KvCacheTransferError::Runtime(
                "reference HashMap storage has no production memory descriptor".into(),
            ))
        }
        fn submit(&mut self, span: &KvCacheTransferSpan) -> Result<(), KvCacheTransferError> {
            let (lock, ready) = &*self.rendezvous;
            let destination = ready
                .wait_while(lock.lock().expect("rendezvous"), |value| value.is_none())
                .expect("destination ready");
            let destination = destination.as_ref().expect("destination");
            assert_ne!(span.cache_pages(), destination.slots);
            let mut source = self.source.lock().expect("source KV");
            let source = source.allocation_mut();
            let mut target_resources = self.destination.lock().expect("destination KV");
            let target = target_resources.allocation_mut();
            for layer in 0..source.layout.runtime().num_layers {
                for (src, dst) in span.cache_pages().iter().zip(&destination.slots) {
                    let (mut key, mut value) =
                        source.gather(layer, &[*src]).expect("real prefill KV");
                    assert!(key.iter().any(|v| *v != 0.0));
                    assert!(value.iter().any(|v| *v != 0.0));
                    if self.corrupt {
                        key.fill(0.0);
                        value.fill(0.0);
                    }
                    target
                        .write(layer, *dst, key, value)
                        .expect("copy reference KV");
                }
            }
            drop(target_resources);
            let frames = completion_frames(span.bootstrap_room(), span.first_token(), None, 0)
                .expect("community control frames");
            send_frames(destination.endpoint.clone(), frames)
                .map_err(KvCacheTransferError::Runtime)?;
            self.pending.push(KvTransferCompletionEvent::completed(
                span.bootstrap_room(),
                span.descriptor_checksum(),
            ));
            Ok(())
        }
        fn poll(&mut self) -> Result<KvTransferPoll, KvCacheTransferError> {
            Ok(KvTransferPoll::new(std::mem::take(&mut self.pending), 0))
        }
        fn cancel(&mut self, room: u64) -> Result<(), KvCacheTransferError> {
            self.pending.retain(|event| event.bootstrap_room() != room);
            Ok(())
        }
        fn shutdown(&mut self) -> Result<(), KvCacheTransferError> {
            self.pending.clear();
            Ok(())
        }
    }

    struct ReferenceReceive {
        receiver: MooncakeDecodeReceiver,
        rendezvous: Rendezvous,
    }

    impl DecodeBootstrapPublisher for ReferenceReceive {
        fn publish_decode_bootstrap_metadata(
            &mut self,
            plan: &KvCacheTransferPlan,
        ) -> Result<DecodeBootstrapMetadataPublishSummary, String> {
            assert_eq!(plan.len(), 1);
            let span = &plan.spans()[0];
            self.receiver.register(span.bootstrap_room())?;
            *self.rendezvous.0.lock().expect("rendezvous") = Some(Destination {
                slots: span.cache_pages().to_vec(),
                endpoint: format!("tcp://127.0.0.1:{}", self.receiver.port()),
            });
            self.rendezvous.1.notify_all();
            Ok(DecodeBootstrapMetadataPublishSummary { published_spans: 1 })
        }
        fn receive_prefill_token(&mut self, room: u64) -> Result<u32, String> {
            self.receiver.receive(room)
        }
        fn cancel_receive(&mut self, room: u64) -> Result<(), String> {
            self.receiver.wait_drained(room)
        }
    }

    struct ReceiveOnly;
    impl KvTransferBackend for ReceiveOnly {
        fn register(&mut self, _: TransferableKvCacheMemory) -> Result<(), KvCacheTransferError> {
            unreachable!("reference receive has no native registration")
        }
        fn submit(&mut self, _: &KvCacheTransferSpan) -> Result<(), KvCacheTransferError> {
            panic!("decode must not submit a local prefill transfer")
        }
        fn poll(&mut self) -> Result<KvTransferPoll, KvCacheTransferError> {
            panic!("decode receives sender completion")
        }
        fn cancel(&mut self, _: u64) -> Result<(), KvCacheTransferError> {
            Ok(())
        }
        fn shutdown(&mut self) -> Result<(), KvCacheTransferError> {
            Ok(())
        }
    }

    fn request(pd: bool, id: &str) -> ScheduledRequest {
        let mut sampling = SamplingParams::new(4);
        sampling.ignore_eos = true;
        ScheduledRequest::new(RequestId::from(id), vec![1, 2, 3], sampling)
            .with_disaggregated_params(pd.then(|| DisaggregatedParams {
                bootstrap_host: "127.0.0.1".into(),
                bootstrap_port: 8998,
                bootstrap_room: 51,
            }))
    }

    fn run_split(
        path: &std::path::Path,
        corrupt: bool,
    ) -> (Vec<u32>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let (prefill, source, prefill_logits) = model(path);
        let (decode, destination, decode_logits) = model(path);
        let rendezvous = Arc::new((Mutex::new(None), Condvar::new()));
        let copy = ReferencePageCopy {
            source,
            destination,
            rendezvous: Arc::clone(&rendezvous),
            corrupt,
            pending: Vec::new(),
        };
        let prefill = KvTransferModelWorker::new(prefill, DecodeBootstrapRegistry::default(), copy)
            .with_prefill_only();
        let mut sender = Scheduler::with_cache_resources_and_policy(
            prefill,
            RadixCache::default(),
            CachePageAllocator::new(64),
            PrefixCachePolicy::Disabled,
        );
        sender.enqueue(request(true, "P"));
        let sender = std::thread::spawn(move || {
            let output = sender
                .dispatch_prefill_batch(1)
                .expect("prefill completes after copy");
            assert!(output[0].finished);
            assert_eq!(sender.available_cache_pages(), Some(64));
        });
        let receive = ReferenceReceive {
            receiver: MooncakeDecodeReceiver::start("127.0.0.1").expect("receiver"),
            rendezvous,
        };
        let decode =
            KvTransferModelWorker::new(decode, DecodeBootstrapRegistry::default(), ReceiveOnly)
                .with_decode_bootstrap_publisher(receive)
                .with_decode_side_bootstrap_only();
        let mut allocator = CachePageAllocator::new(64);
        let _held = allocator
            .allocate(7)
            .expect("offset destination physical pages");
        let mut receiver = Scheduler::with_cache_resources_and_policy(
            decode,
            RadixCache::default(),
            allocator,
            PrefixCachePolicy::Disabled,
        );
        receiver.enqueue(request(true, "D"));
        let first = receiver
            .dispatch_prefill_batch(1)
            .expect("receive first token");
        assert!(
            decode_logits.lock().expect("logits").is_empty(),
            "decode did not execute local prefill"
        );
        let mut tokens = first[0].token_ids.clone();
        for _ in 0..3 {
            tokens.extend(
                receiver
                    .dispatch_decode_batch(1)
                    .expect("real cached decode")[0]
                    .token_ids
                    .clone(),
            );
        }
        sender.join().expect("sender joins");
        assert_eq!(receiver.available_cache_pages(), Some(57));
        let p = prefill_logits
            .lock()
            .expect("prefill logits")
            .iter()
            .map(|(_, rows)| rows[0].clone())
            .collect();
        let d = decode_logits
            .lock()
            .expect("decode logits")
            .iter()
            .map(|(mode, rows)| {
                assert_eq!(*mode, ForwardMode::Decode);
                rows[0].clone()
            })
            .collect();
        (tokens, p, d)
    }

    #[test]
    fn nonzero_qwen_pd_matches_centralized_logits_and_detects_corrupt_kv() {
        let path = std::env::temp_dir().join(format!("sglang-pd-logits-{}", std::process::id()));
        fixture::write_model(&path);
        let (central, _, central_logits) = model(&path);
        let mut scheduler = Scheduler::with_cache_resources_and_policy(
            central,
            RadixCache::default(),
            CachePageAllocator::new(64),
            PrefixCachePolicy::Disabled,
        );
        scheduler.enqueue(request(false, "central"));
        let mut expected = scheduler
            .dispatch_prefill_batch(1)
            .expect("central prefill")[0]
            .token_ids
            .clone();
        for _ in 0..3 {
            expected.extend(
                scheduler.dispatch_decode_batch(1).expect("central decode")[0]
                    .token_ids
                    .clone(),
            );
        }
        let expected_logits = central_logits
            .lock()
            .expect("logits")
            .iter()
            .map(|(_, rows)| rows[0].clone())
            .collect::<Vec<_>>();
        let (actual, p, d) = run_split(&path, false);
        assert_eq!(actual, expected);
        for (actual, expected) in p.iter().chain(&d).zip(&expected_logits) {
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() < 1e-6, "PD logit {a} != central {b}");
            }
        }
        let (_, _, corrupt) = run_split(&path, true);
        assert!(
            corrupt[0]
                .iter()
                .zip(&expected_logits[1])
                .any(|(a, b)| (a - b).abs() > 1e-4),
            "the oracle must detect incorrect transferred KV"
        );
        std::fs::remove_dir_all(path).expect("remove fixture");
    }
}
