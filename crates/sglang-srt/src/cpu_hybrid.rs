use std::collections::HashMap;

use sglang_kernel::cpu::{
    GatedDeltaRuleShape, causal_depthwise_conv1d_step, gated_delta_rule_step, rms_norm,
};

use crate::backend::{RuntimeCapability, RuntimeDtype};
use crate::cache::CachePageId;
use crate::cpu_reference::{
    CpuNormalization, CpuReferenceDenseDecoderError, CpuReferenceKvCache, DenseFeedForwardWeights,
    FloatMatrix, FullAttentionForwardContext, FullAttentionShape, FullAttentionWeightNamesRef,
    FullAttentionWeights, add_residual, apply_normalization, checked_product,
    forward_full_attention, kernel_error, load_tensor, load_vector, model_forward_error,
    validate_batch,
};
use crate::model_artifacts::LocalModelArtifacts;
use crate::model_executor::{
    ForwardModel, ModelForwardError, ModelForwardOutput, ModelWorkerBatch,
};
use crate::models::{
    GatedDeltaNetWeightNames, HybridDecoderExecutionPlan, HybridDecoderLayerKind,
    HybridDecoderLayerWeightNames, HybridFullAttentionWeightNames, ModelDefinition,
};
use crate::runtime_kv_cache::ModelExecutionResources;
use crate::types::RequestId;

#[derive(Debug)]
pub(crate) struct CpuReferenceHybridDecoder {
    plan: HybridDecoderExecutionPlan,
    token_embeddings: FloatMatrix,
    final_norm: Vec<f32>,
    lm_head: Option<FloatMatrix>,
    layers: Vec<HybridDecoderLayerWeights>,
    request_states: HashMap<String, HybridRequestState>,
}

impl CpuReferenceHybridDecoder {
    pub(crate) fn load(
        definition: &ModelDefinition,
        artifacts: &LocalModelArtifacts,
    ) -> Result<Self, CpuReferenceDenseDecoderError> {
        definition.validate_hybrid_decoder_checkpoint(artifacts)?;
        let plan = definition
            .hybrid_decoder()
            .ok_or_else(|| {
                CpuReferenceDenseDecoderError::Unsupported(
                    "model definition does not include a hybrid decoder execution plan".to_string(),
                )
            })?
            .clone();
        let query_size = checked_product(
            plan.num_attention_heads,
            plan.attention_head_dim,
            "hybrid query projection",
        )?;
        let projected_query_size = checked_product(
            query_size,
            if plan.attention_output_gate { 2 } else { 1 },
            "hybrid gated query projection",
        )?;
        let kv_size = checked_product(
            plan.num_key_value_heads,
            plan.attention_head_dim,
            "hybrid KV projection",
        )?;
        let linear_key_size = checked_product(
            plan.linear_num_key_heads,
            plan.linear_key_head_dim,
            "linear key projection",
        )?;
        let linear_value_size = checked_product(
            plan.linear_num_value_heads,
            plan.linear_value_head_dim,
            "linear value projection",
        )?;
        let conv_dim = linear_key_size
            .checked_mul(2)
            .and_then(|size| size.checked_add(linear_value_size))
            .ok_or_else(|| {
                CpuReferenceDenseDecoderError::Unsupported(
                    "linear convolution size overflowed".to_string(),
                )
            })?;

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
                HybridDecoderLayerWeights::load(
                    artifacts,
                    names,
                    &plan,
                    HybridLoadGeometry {
                        projected_query_size,
                        query_size,
                        kv_size,
                        conv_dim,
                        linear_value_size,
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            plan,
            token_embeddings,
            final_norm,
            lm_head,
            layers,
            request_states: HashMap::new(),
        })
    }

    pub(crate) fn runtime_capability(&self) -> RuntimeCapability {
        RuntimeCapability::cpu_reference("cpu-reference-hybrid-decoder", false)
    }

    pub(crate) fn execution_dtype(&self) -> RuntimeDtype {
        RuntimeDtype::F32
    }

    fn forward_token(
        &mut self,
        kv_cache: &mut CpuReferenceKvCache,
        request_id: &RequestId,
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

        let recurrent_layer_count = self
            .layers
            .iter()
            .filter(|layer| matches!(layer.mixer, HybridMixerWeights::GatedDeltaNet { .. }))
            .count();
        let request_state = match self.request_states.entry(request_id.as_str().to_string()) {
            std::collections::hash_map::Entry::Vacant(entry) if position == 0 => {
                entry.insert(HybridRequestState::new(
                    recurrent_layer_count,
                    self.plan.linear_conv_kernel_dim,
                    self.plan.linear_num_key_heads,
                    self.plan.linear_num_value_heads,
                    self.plan.linear_key_head_dim,
                    self.plan.linear_value_head_dim,
                )?)
            }
            std::collections::hash_map::Entry::Vacant(_) => {
                return Err(CpuReferenceDenseDecoderError::Execution(format!(
                    "hybrid request {} has no recurrent state at position {position}; prefix-state restoration is not available",
                    request_id.as_str()
                )));
            }
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
        };
        if request_state.next_position != position {
            return Err(CpuReferenceDenseDecoderError::Execution(format!(
                "hybrid request {} expected position {} but received {position}",
                request_id.as_str(),
                request_state.next_position
            )));
        }

        let mut hidden = self.token_embeddings.row(token_id)?;
        for layer in &self.layers {
            let normalized = apply_normalization(
                &hidden,
                &layer.input_norm,
                1,
                self.plan.hidden_size,
                self.plan.rms_norm_eps,
                CpuNormalization::Gemma,
            )?;
            let mixer_output = match &layer.mixer {
                HybridMixerWeights::FullAttention {
                    cache_layer_index,
                    weights,
                } => forward_full_attention(
                    weights,
                    kv_cache,
                    &normalized,
                    FullAttentionForwardContext {
                        cache_layer_index: *cache_layer_index,
                        position,
                        output_slot,
                        sequence_slots: &sequence_slots[..=position],
                        shape: FullAttentionShape {
                            query_head_count: self.plan.num_attention_heads,
                            kv_head_count: self.plan.num_key_value_heads,
                            head_dim: self.plan.attention_head_dim,
                            rotary_dim: self.plan.rotary_dim,
                            output_gate: self.plan.attention_output_gate,
                        },
                        rms_norm_eps: self.plan.rms_norm_eps,
                        rope_theta: self.plan.rope_theta,
                        qk_normalization: CpuNormalization::Gemma,
                    },
                )?,
                HybridMixerWeights::GatedDeltaNet {
                    state_layer_index,
                    weights,
                } => weights.forward(
                    &normalized,
                    &mut request_state.recurrent_layers[*state_layer_index],
                    &self.plan,
                )?,
            };
            let after_mixer = add_residual(&hidden, &mixer_output)?;
            let normalized = apply_normalization(
                &after_mixer,
                &layer.post_attention_norm,
                1,
                self.plan.hidden_size,
                self.plan.rms_norm_eps,
                CpuNormalization::Gemma,
            )?;
            let feed_forward = layer.feed_forward.forward(
                &normalized,
                self.plan.intermediate_size,
                self.plan.activation,
            )?;
            hidden = add_residual(&after_mixer, &feed_forward)?;
        }
        request_state.next_position += 1;

        let hidden = apply_normalization(
            &hidden,
            &self.final_norm,
            1,
            self.plan.hidden_size,
            self.plan.rms_norm_eps,
            CpuNormalization::Gemma,
        )?;
        self.lm_head
            .as_ref()
            .unwrap_or(&self.token_embeddings)
            .project(&hidden, 1)
    }
}

impl ForwardModel for CpuReferenceHybridDecoder {
    fn forward(
        &mut self,
        _batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        Err(ModelForwardError::Runtime(
            "CPU reference hybrid decoder requires ModelRunner-owned active KV memory".to_string(),
        ))
    }

    fn forward_with_resources(
        &mut self,
        batch: &ModelWorkerBatch,
        resources: ModelExecutionResources<'_>,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        let kv_cache = resources
            .active_kv_cache()
            .ok_or_else(|| {
                ModelForwardError::Runtime(
                    "CPU reference hybrid decoder has no active KV allocation".to_string(),
                )
            })?
            .as_any_mut()
            .downcast_mut::<CpuReferenceKvCache>()
            .ok_or_else(|| {
                ModelForwardError::Runtime(
                    "CPU reference hybrid decoder received a non-CPU active KV allocation"
                        .to_string(),
                )
            })?;
        validate_batch(batch).map_err(model_forward_error)?;
        let mut logits = Vec::with_capacity(batch.request_ids().len());
        for request_index in 0..batch.request_ids().len() {
            if batch.cached_token_counts()[request_index] != 0 {
                return Err(ModelForwardError::Runtime(format!(
                    "hybrid request {} received {} cached prefix tokens, but recurrent state snapshots are not implemented",
                    batch.request_ids()[request_index].as_str(),
                    batch.cached_token_counts()[request_index]
                )));
            }
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
                        &batch.request_ids()[request_index],
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
                    "hybrid decoder request {request_index} has no input tokens"
                ))
            })?);
        }
        ModelForwardOutput::new(logits)
    }

    fn complete_request(&mut self, request_id: &RequestId) {
        self.request_states.remove(request_id.as_str());
    }
}

#[derive(Debug)]
struct HybridDecoderLayerWeights {
    input_norm: Vec<f32>,
    mixer: HybridMixerWeights,
    post_attention_norm: Vec<f32>,
    feed_forward: DenseFeedForwardWeights,
}

#[derive(Clone, Copy)]
struct HybridLoadGeometry {
    projected_query_size: usize,
    query_size: usize,
    kv_size: usize,
    conv_dim: usize,
    linear_value_size: usize,
}

impl HybridDecoderLayerWeights {
    fn load(
        artifacts: &LocalModelArtifacts,
        names: &HybridDecoderLayerWeightNames,
        plan: &HybridDecoderExecutionPlan,
        geometry: HybridLoadGeometry,
    ) -> Result<Self, CpuReferenceDenseDecoderError> {
        let HybridLoadGeometry {
            projected_query_size,
            query_size,
            kv_size,
            conv_dim,
            linear_value_size,
        } = geometry;
        let mixer = match &names.mixer {
            HybridDecoderLayerKind::FullAttention {
                cache_layer_index,
                weights,
            } => HybridMixerWeights::FullAttention {
                cache_layer_index: *cache_layer_index,
                weights: load_full_attention(
                    artifacts,
                    weights,
                    plan.hidden_size,
                    projected_query_size,
                    query_size,
                    kv_size,
                    plan.attention_head_dim,
                )?,
            },
            HybridDecoderLayerKind::GatedDeltaNet {
                state_layer_index,
                weights,
            } => HybridMixerWeights::GatedDeltaNet {
                state_layer_index: *state_layer_index,
                weights: GatedDeltaNetWeights::load(
                    artifacts,
                    weights,
                    plan,
                    conv_dim,
                    linear_value_size,
                )?,
            },
        };
        Ok(Self {
            input_norm: load_vector(artifacts, &names.input_norm, plan.hidden_size)?,
            mixer,
            post_attention_norm: load_vector(
                artifacts,
                &names.post_attention_norm,
                plan.hidden_size,
            )?,
            feed_forward: DenseFeedForwardWeights::load(
                artifacts,
                &names.feed_forward.gate_weight,
                &names.feed_forward.up_weight,
                &names.feed_forward.down_weight,
                plan.hidden_size,
                plan.intermediate_size,
            )?,
        })
    }
}

#[derive(Debug)]
enum HybridMixerWeights {
    FullAttention {
        cache_layer_index: usize,
        weights: FullAttentionWeights,
    },
    GatedDeltaNet {
        state_layer_index: usize,
        weights: GatedDeltaNetWeights,
    },
}

fn load_full_attention(
    artifacts: &LocalModelArtifacts,
    names: &HybridFullAttentionWeightNames,
    hidden_size: usize,
    projected_query_size: usize,
    query_size: usize,
    kv_size: usize,
    head_dim: usize,
) -> Result<FullAttentionWeights, CpuReferenceDenseDecoderError> {
    FullAttentionWeights::load(
        artifacts,
        FullAttentionWeightNamesRef {
            query_weight: &names.query_weight,
            query_bias: None,
            query_norm: Some(&names.query_norm),
            key_weight: &names.key_weight,
            key_bias: None,
            key_norm: Some(&names.key_norm),
            value_weight: &names.value_weight,
            value_bias: None,
            output_weight: &names.output_weight,
            output_bias: None,
        },
        hidden_size,
        projected_query_size,
        query_size,
        kv_size,
        head_dim,
    )
}

#[derive(Debug)]
struct GatedDeltaNetWeights {
    a_log: Vec<f32>,
    conv1d_weight: Vec<f32>,
    dt_bias: Vec<f32>,
    in_proj_a: FloatMatrix,
    in_proj_b: FloatMatrix,
    in_proj_qkv: FloatMatrix,
    in_proj_z: FloatMatrix,
    norm_weight: Vec<f32>,
    output: FloatMatrix,
}

impl GatedDeltaNetWeights {
    fn load(
        artifacts: &LocalModelArtifacts,
        names: &GatedDeltaNetWeightNames,
        plan: &HybridDecoderExecutionPlan,
        conv_dim: usize,
        linear_value_size: usize,
    ) -> Result<Self, CpuReferenceDenseDecoderError> {
        Ok(Self {
            a_log: load_vector(artifacts, &names.a_log, plan.linear_num_value_heads)?,
            conv1d_weight: load_tensor(
                artifacts,
                &names.conv1d_weight,
                &[conv_dim, 1, plan.linear_conv_kernel_dim],
            )?,
            dt_bias: load_vector(artifacts, &names.dt_bias, plan.linear_num_value_heads)?,
            in_proj_a: FloatMatrix::load(
                artifacts,
                &names.in_proj_a_weight,
                plan.linear_num_value_heads,
                plan.hidden_size,
            )?,
            in_proj_b: FloatMatrix::load(
                artifacts,
                &names.in_proj_b_weight,
                plan.linear_num_value_heads,
                plan.hidden_size,
            )?,
            in_proj_qkv: FloatMatrix::load(
                artifacts,
                &names.in_proj_qkv_weight,
                conv_dim,
                plan.hidden_size,
            )?,
            in_proj_z: FloatMatrix::load(
                artifacts,
                &names.in_proj_z_weight,
                linear_value_size,
                plan.hidden_size,
            )?,
            norm_weight: load_vector(artifacts, &names.norm_weight, plan.linear_value_head_dim)?,
            output: FloatMatrix::load(
                artifacts,
                &names.output_weight,
                plan.hidden_size,
                linear_value_size,
            )?,
        })
    }

    fn forward(
        &self,
        hidden: &[f32],
        state: &mut GatedDeltaLayerState,
        plan: &HybridDecoderExecutionPlan,
    ) -> Result<Vec<f32>, CpuReferenceDenseDecoderError> {
        let mut qkv = self.in_proj_qkv.project(hidden, 1)?;
        qkv = causal_depthwise_conv1d_step(
            &qkv,
            &self.conv1d_weight,
            &mut state.conv,
            qkv.len(),
            plan.linear_conv_kernel_dim,
        )
        .map_err(kernel_error)?;
        for value in &mut qkv {
            *value *= 1.0 / (1.0 + (-*value).exp());
        }

        let key_size = checked_product(
            plan.linear_num_key_heads,
            plan.linear_key_head_dim,
            "linear key",
        )?;
        let value_size = checked_product(
            plan.linear_num_value_heads,
            plan.linear_value_head_dim,
            "linear value",
        )?;
        let mut query = qkv[..key_size].to_vec();
        let mut key = qkv[key_size..key_size * 2].to_vec();
        let value = &qkv[key_size * 2..key_size * 2 + value_size];
        l2_normalize_heads(
            &mut query,
            plan.linear_num_key_heads,
            plan.linear_key_head_dim,
            (plan.linear_key_head_dim as f32).sqrt().recip(),
        )?;
        l2_normalize_heads(
            &mut key,
            plan.linear_num_key_heads,
            plan.linear_key_head_dim,
            1.0,
        )?;

        let a = self.in_proj_a.project(hidden, 1)?;
        let b = self.in_proj_b.project(hidden, 1)?;
        let decay = a
            .iter()
            .zip(&self.dt_bias)
            .zip(&self.a_log)
            .map(|((a, dt_bias), a_log)| {
                let time = softplus(*a + *dt_bias);
                (-a_log.exp() * time).exp()
            })
            .collect::<Vec<_>>();
        let beta = b
            .into_iter()
            .map(|value| 1.0 / (1.0 + (-value).exp()))
            .collect::<Vec<_>>();
        let core = gated_delta_rule_step(
            &query,
            &key,
            value,
            &decay,
            &beta,
            &mut state.recurrent,
            GatedDeltaRuleShape {
                key_head_count: plan.linear_num_key_heads,
                value_head_count: plan.linear_num_value_heads,
                key_head_dim: plan.linear_key_head_dim,
                value_head_dim: plan.linear_value_head_dim,
            },
        )
        .map_err(kernel_error)?;
        let mut normalized = rms_norm(
            &core,
            &self.norm_weight,
            plan.linear_num_value_heads,
            plan.linear_value_head_dim,
            plan.rms_norm_eps,
        )
        .map_err(kernel_error)?;
        let z = self.in_proj_z.project(hidden, 1)?;
        for (value, gate) in normalized.iter_mut().zip(z) {
            *value *= gate / (1.0 + (-gate).exp());
        }
        self.output.project(&normalized, 1)
    }
}

#[derive(Debug)]
struct HybridRequestState {
    next_position: usize,
    recurrent_layers: Vec<GatedDeltaLayerState>,
}

impl HybridRequestState {
    fn new(
        layer_count: usize,
        conv_kernel_dim: usize,
        key_head_count: usize,
        value_head_count: usize,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Result<Self, CpuReferenceDenseDecoderError> {
        let conv_dim = key_head_count
            .checked_mul(key_head_dim)
            .and_then(|size| size.checked_mul(2))
            .and_then(|size| {
                value_head_count
                    .checked_mul(value_head_dim)
                    .and_then(|value_size| size.checked_add(value_size))
            })
            .ok_or_else(|| {
                CpuReferenceDenseDecoderError::Unsupported(
                    "hybrid convolution state size overflowed".to_string(),
                )
            })?;
        let conv_state_size = conv_kernel_dim
            .checked_sub(1)
            .and_then(|history| history.checked_mul(conv_dim))
            .ok_or_else(|| {
                CpuReferenceDenseDecoderError::Unsupported(
                    "hybrid convolution state size overflowed".to_string(),
                )
            })?;
        let recurrent_state_size = value_head_count
            .checked_mul(key_head_dim)
            .and_then(|size| size.checked_mul(value_head_dim))
            .ok_or_else(|| {
                CpuReferenceDenseDecoderError::Unsupported(
                    "hybrid recurrent state size overflowed".to_string(),
                )
            })?;
        Ok(Self {
            next_position: 0,
            recurrent_layers: (0..layer_count)
                .map(|_| GatedDeltaLayerState {
                    conv: vec![0.0; conv_state_size],
                    recurrent: vec![0.0; recurrent_state_size],
                })
                .collect(),
        })
    }
}

#[derive(Debug)]
struct GatedDeltaLayerState {
    conv: Vec<f32>,
    recurrent: Vec<f32>,
}

fn l2_normalize_heads(
    values: &mut [f32],
    head_count: usize,
    head_dim: usize,
    output_scale: f32,
) -> Result<(), CpuReferenceDenseDecoderError> {
    if values.len() != head_count * head_dim {
        return Err(CpuReferenceDenseDecoderError::Execution(format!(
            "L2 normalization width {} does not match {head_count} heads x {head_dim}",
            values.len()
        )));
    }
    for head in 0..head_count {
        let offset = head * head_dim;
        let row = &mut values[offset..offset + head_dim];
        let norm = (row.iter().map(|value| value * value).sum::<f32>() + 1e-6).sqrt();
        for value in row {
            *value = *value / norm * output_scale;
        }
    }
    Ok(())
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}
