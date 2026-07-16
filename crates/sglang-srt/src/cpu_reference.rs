use std::collections::HashMap;
use std::fmt;

use sglang_kernel::cpu::{
    GroupedQueryAttentionShape, apply_neox_rope_inplace, grouped_query_attention, linear, rms_norm,
    silu_and_mul,
};

use crate::backend::{RuntimeCapability, RuntimeDtype};
use crate::cache::CachePageId;
use crate::model_artifacts::{LocalModelArtifacts, ModelArtifactError};
use crate::model_executor::{
    ForwardModel, ModelForwardError, ModelForwardOutput, ModelWorkerBatch,
};
use crate::models::{
    AttentionArchitecture, DenseDecoderActivation, DenseDecoderExecutionPlan,
    DenseDecoderLayerWeightNames, FeedForwardArchitecture, ModelDefinition,
    ModelExecutionArchitecture,
};

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
    kv_cache: CpuReferenceKvCache,
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
                    intermediate_size,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let kv_cache = CpuReferenceKvCache::new(layers.len(), kv_size);

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
            kv_cache,
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
        for layer_index in 0..self.layers.len() {
            hidden = forward_layer(
                &self.layers[layer_index],
                &mut self.kv_cache,
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
                    rope_theta: self.plan.rope_theta,
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

impl ForwardModel for CpuReferenceDenseDecoder {
    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
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
    rope_theta: f32,
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
        rope_theta,
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
    let mut query = layer.query.project(&normalized, 1)?;
    let mut key = layer.key.project(&normalized, 1)?;
    let value = layer.value.project(&normalized, 1)?;
    apply_neox_rope_inplace(
        &mut query,
        shape.query_head_count,
        shape.head_dim,
        position,
        rope_theta,
    )
    .map_err(kernel_error)?;
    apply_neox_rope_inplace(
        &mut key,
        shape.kv_head_count,
        shape.head_dim,
        position,
        rope_theta,
    )
    .map_err(kernel_error)?;
    kv_cache.write(layer_index, output_slot, key, value)?;
    let (keys, values) = kv_cache.gather(layer_index, sequence_slots)?;
    let attention = grouped_query_attention(
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
    let attention_output = layer.output.project(&attention, 1)?;
    let after_attention = add_residual(hidden, &attention_output)?;

    let normalized = rms_norm(
        &after_attention,
        &layer.post_attention_norm,
        1,
        shape.hidden_size,
        rms_norm_eps,
    )
    .map_err(kernel_error)?;
    let gate = layer.gate.project(&normalized, 1)?;
    let up = layer.up.project(&normalized, 1)?;
    let activated = match activation {
        DenseDecoderActivation::Silu => silu_and_mul(&gate, &up).map_err(kernel_error)?,
    };
    if activated.len() != shape.intermediate_size {
        return Err(CpuReferenceDenseDecoderError::Execution(format!(
            "dense activation width {} does not match intermediate size {}",
            activated.len(),
            shape.intermediate_size
        )));
    }
    let feed_forward = layer.down.project(&activated, 1)?;
    add_residual(&after_attention, &feed_forward)
}

fn validate_batch(batch: &ModelWorkerBatch) -> Result<(), CpuReferenceDenseDecoderError> {
    let request_count = batch.request_ids().len();
    for (name, actual) in [
        ("request_offsets", batch.request_offsets().len()),
        ("input_token_counts", batch.input_token_counts().len()),
        ("sequence_offsets", batch.sequence_offsets().len()),
        ("sequence_token_counts", batch.sequence_token_counts().len()),
    ] {
        if actual != request_count {
            return Err(CpuReferenceDenseDecoderError::Execution(format!(
                "batch field {name} has length {actual}, expected {request_count}"
            )));
        }
    }
    if batch.input_ids().len() != batch.positions().len()
        || batch.input_ids().len() != batch.out_cache_pages().len()
    {
        return Err(CpuReferenceDenseDecoderError::Execution(format!(
            "dense decoder input/position/output-slot lengths differ: {}/{}/{}",
            batch.input_ids().len(),
            batch.positions().len(),
            batch.out_cache_pages().len()
        )));
    }
    if batch.sequence_token_ids().len() != batch.sequence_cache_pages().len() {
        return Err(CpuReferenceDenseDecoderError::Execution(format!(
            "dense decoder sequence token/slot lengths differ: {}/{}",
            batch.sequence_token_ids().len(),
            batch.sequence_cache_pages().len()
        )));
    }
    for request_index in 0..request_count {
        let input_end = batch.request_offsets()[request_index]
            .checked_add(batch.input_token_counts()[request_index])
            .ok_or_else(|| {
                CpuReferenceDenseDecoderError::Execution("batch input range overflowed".to_string())
            })?;
        let sequence_end = batch.sequence_offsets()[request_index]
            .checked_add(batch.sequence_token_counts()[request_index])
            .ok_or_else(|| {
                CpuReferenceDenseDecoderError::Execution(
                    "batch sequence range overflowed".to_string(),
                )
            })?;
        if input_end > batch.input_ids().len() || sequence_end > batch.sequence_cache_pages().len()
        {
            return Err(CpuReferenceDenseDecoderError::Execution(format!(
                "dense decoder request {request_index} batch ranges exceed flattened storage"
            )));
        }
    }
    Ok(())
}

#[derive(Debug)]
struct DenseDecoderLayerWeights {
    input_norm: Vec<f32>,
    query: FloatMatrix,
    key: FloatMatrix,
    value: FloatMatrix,
    output: FloatMatrix,
    post_attention_norm: Vec<f32>,
    gate: FloatMatrix,
    up: FloatMatrix,
    down: FloatMatrix,
}

impl DenseDecoderLayerWeights {
    fn load(
        artifacts: &LocalModelArtifacts,
        names: &DenseDecoderLayerWeightNames,
        hidden_size: usize,
        query_size: usize,
        kv_size: usize,
        intermediate_size: usize,
    ) -> Result<Self, CpuReferenceDenseDecoderError> {
        Ok(Self {
            input_norm: load_vector(artifacts, &names.input_norm, hidden_size)?,
            query: FloatMatrix::load_with_bias(
                artifacts,
                &names.query_weight,
                names.query_bias.as_deref(),
                query_size,
                hidden_size,
            )?,
            key: FloatMatrix::load_with_bias(
                artifacts,
                &names.key_weight,
                names.key_bias.as_deref(),
                kv_size,
                hidden_size,
            )?,
            value: FloatMatrix::load_with_bias(
                artifacts,
                &names.value_weight,
                names.value_bias.as_deref(),
                kv_size,
                hidden_size,
            )?,
            output: FloatMatrix::load(artifacts, &names.output_weight, hidden_size, query_size)?,
            post_attention_norm: load_vector(artifacts, &names.post_attention_norm, hidden_size)?,
            gate: FloatMatrix::load(
                artifacts,
                &names.gate_weight,
                intermediate_size,
                hidden_size,
            )?,
            up: FloatMatrix::load(artifacts, &names.up_weight, intermediate_size, hidden_size)?,
            down: FloatMatrix::load(
                artifacts,
                &names.down_weight,
                hidden_size,
                intermediate_size,
            )?,
        })
    }
}

#[derive(Debug)]
struct FloatMatrix {
    rows: usize,
    columns: usize,
    values: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl FloatMatrix {
    fn load(
        artifacts: &LocalModelArtifacts,
        tensor_name: &str,
        rows: usize,
        columns: usize,
    ) -> Result<Self, CpuReferenceDenseDecoderError> {
        Self::load_with_bias(artifacts, tensor_name, None, rows, columns)
    }

    fn load_with_bias(
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

    fn project(
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

    fn row(&self, row: u32) -> Result<Vec<f32>, CpuReferenceDenseDecoderError> {
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
struct CpuReferenceKvCache {
    layers: Vec<HashMap<usize, KvSlot>>,
    kv_width: usize,
}

impl CpuReferenceKvCache {
    fn new(layer_count: usize, kv_width: usize) -> Self {
        Self {
            layers: (0..layer_count).map(|_| HashMap::new()).collect(),
            kv_width,
        }
    }

    fn write(
        &mut self,
        layer_index: usize,
        slot: CachePageId,
        key: Vec<f32>,
        value: Vec<f32>,
    ) -> Result<(), CpuReferenceDenseDecoderError> {
        if key.len() != self.kv_width || value.len() != self.kv_width {
            return Err(CpuReferenceDenseDecoderError::Execution(format!(
                "KV write width {}/{} does not match expected {}",
                key.len(),
                value.len(),
                self.kv_width
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

    fn gather(
        &self,
        layer_index: usize,
        slots: &[CachePageId],
    ) -> Result<(Vec<f32>, Vec<f32>), CpuReferenceDenseDecoderError> {
        let layer = self.layers.get(layer_index).ok_or_else(|| {
            CpuReferenceDenseDecoderError::Execution(format!(
                "KV cache layer {layer_index} is out of range"
            ))
        })?;
        let capacity = slots.len().checked_mul(self.kv_width).ok_or_else(|| {
            CpuReferenceDenseDecoderError::Execution("KV gather size overflowed".to_string())
        })?;
        let mut keys = Vec::with_capacity(capacity);
        let mut values = Vec::with_capacity(capacity);
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

#[derive(Debug)]
struct KvSlot {
    key: Vec<f32>,
    value: Vec<f32>,
}

fn load_vector(
    artifacts: &LocalModelArtifacts,
    tensor_name: &str,
    length: usize,
) -> Result<Vec<f32>, CpuReferenceDenseDecoderError> {
    load_tensor(artifacts, tensor_name, &[length])
}

fn load_tensor(
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

fn add_residual(
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

fn checked_product(
    left: usize,
    right: usize,
    name: &str,
) -> Result<usize, CpuReferenceDenseDecoderError> {
    left.checked_mul(right).ok_or_else(|| {
        CpuReferenceDenseDecoderError::Unsupported(format!("{name} size overflowed"))
    })
}

fn kernel_error(error: impl fmt::Display) -> CpuReferenceDenseDecoderError {
    CpuReferenceDenseDecoderError::Execution(error.to_string())
}

fn model_forward_error(error: CpuReferenceDenseDecoderError) -> ModelForwardError {
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
