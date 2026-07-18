use std::collections::HashMap;

use nexus_transfer::{KvCacheMemoryProvider, TransferableKvCacheMemory};
use sglang_kernel::cpu::{
    GatedDeltaRuleShape, KeyGatedDeltaRuleShape, apply_partial_neox_rope_inplace,
    causal_depthwise_conv1d_step, gated_delta_rule_step, key_gated_delta_rule_step, rms_norm,
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
use crate::model_executor::{ModelForwardError, ModelForwardOutput, ModelWorkerBatch};
use crate::model_runtime::{BackendExecutionResources, BackendModelExecutor};
use crate::models::{
    DecoderNormalization, GatedDeltaNetWeightNames, HybridDecoderExecutionPlan,
    HybridDecoderLayerKind, HybridDecoderLayerWeightNames, HybridFeedForward,
    HybridFullAttentionConfig, HybridFullAttentionWeightNames, HybridLinearAttentionConfig,
    HybridMultiLatentAttentionWeightNames, KeyGatedDeltaWeightNames, ModelDefinition,
    MoeFeedForwardConfig, MoeFeedForwardWeightNames, RecurrentStateLayout, RouterActivation,
};
use crate::runtime_kv_cache::{RuntimeKvCache, RuntimeKvCacheMetadata};
use crate::transfer::KvCacheTransferError;
use crate::types::RequestId;

#[derive(Debug)]
pub(crate) struct CpuHybridExecutionResources {
    active_kv_cache: RuntimeKvCache<CpuReferenceKvCache>,
    recurrent_state: CpuHybridStatePool,
}

impl CpuHybridExecutionResources {
    pub(crate) fn new(
        active_kv_cache: CpuReferenceKvCache,
        recurrent_state_layout: RecurrentStateLayout,
    ) -> Result<Self, CpuReferenceDenseDecoderError> {
        Ok(Self {
            active_kv_cache: RuntimeKvCache::new(active_kv_cache),
            recurrent_state: CpuHybridStatePool::new(recurrent_state_layout)?,
        })
    }

    fn parts_mut(&mut self) -> (&mut CpuReferenceKvCache, &mut CpuHybridStatePool) {
        (
            self.active_kv_cache.allocation_mut(),
            &mut self.recurrent_state,
        )
    }
}

impl KvCacheMemoryProvider for CpuHybridExecutionResources {
    type Error = KvCacheTransferError;

    fn transferable_kv_cache_memory(&self) -> Result<TransferableKvCacheMemory, Self::Error> {
        self.active_kv_cache.transferable_kv_cache_memory()
    }
}

impl RuntimeKvCacheMetadata for CpuHybridExecutionResources {
    fn active_kv_cache_layout(&self) -> Option<crate::kv_cache::PagedKvCacheLayout> {
        self.active_kv_cache.active_kv_cache_layout()
    }
}

impl BackendExecutionResources for CpuHybridExecutionResources {
    fn runtime_backend(&self) -> crate::backend::RuntimeBackend {
        crate::backend::RuntimeBackend::Cpu
    }

    fn recurrent_state_layout(&self) -> Option<RecurrentStateLayout> {
        Some(self.recurrent_state.layout())
    }
}

#[derive(Debug)]
struct CpuHybridStatePool {
    layout: RecurrentStateLayout,
    requests: HashMap<String, HybridRequestState>,
}

impl CpuHybridStatePool {
    fn new(layout: RecurrentStateLayout) -> Result<Self, CpuReferenceDenseDecoderError> {
        if layout.layer_count == 0
            || layout.conv_elements_per_layer().is_none()
            || layout.temporal_elements_per_layer().is_none()
            || layout.elements_per_request().is_none()
        {
            return Err(CpuReferenceDenseDecoderError::Unsupported(
                "hybrid recurrent state layout is empty or overflowed".to_string(),
            ));
        }
        Ok(Self {
            layout,
            requests: HashMap::new(),
        })
    }

    fn layout(&self) -> RecurrentStateLayout {
        self.layout
    }

    fn request_state_mut(
        &mut self,
        request_id: &RequestId,
        position: usize,
    ) -> Result<&mut HybridRequestState, CpuReferenceDenseDecoderError> {
        match self.requests.entry(request_id.as_str().to_string()) {
            std::collections::hash_map::Entry::Vacant(entry) if position == 0 => {
                Ok(entry.insert(HybridRequestState::new(self.layout)?))
            }
            std::collections::hash_map::Entry::Vacant(_) => {
                Err(CpuReferenceDenseDecoderError::Execution(format!(
                    "hybrid request {} has no recurrent state at position {position}; prefix-state restoration is not available",
                    request_id.as_str()
                )))
            }
            std::collections::hash_map::Entry::Occupied(entry) => Ok(entry.into_mut()),
        }
    }

    fn complete_request(&mut self, request_id: &RequestId) {
        self.requests.remove(request_id.as_str());
    }
}

#[derive(Debug)]
pub(crate) struct CpuReferenceHybridDecoder {
    plan: HybridDecoderExecutionPlan,
    token_embeddings: FloatMatrix,
    final_norm: Vec<f32>,
    lm_head: Option<FloatMatrix>,
    layers: Vec<HybridDecoderLayerWeights>,
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
            .map(|names| HybridDecoderLayerWeights::load(artifacts, names, &plan))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            plan,
            token_embeddings,
            final_norm,
            lm_head,
            layers,
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
        resources: &mut CpuHybridExecutionResources,
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

        let (kv_cache, state_pool) = resources.parts_mut();
        let request_state = state_pool.request_state_mut(request_id, position)?;
        if request_state.next_position != position {
            return Err(CpuReferenceDenseDecoderError::Execution(format!(
                "hybrid request {} expected position {} but received {position}",
                request_id.as_str(),
                request_state.next_position
            )));
        }

        let normalization_kind = cpu_normalization(self.plan.normalization);
        let mut hidden = self.token_embeddings.row(token_id)?;
        for layer in &self.layers {
            let normalized = apply_normalization(
                &hidden,
                &layer.input_norm,
                1,
                self.plan.hidden_size,
                self.plan.rms_norm_eps,
                normalization_kind,
            )?;
            let mixer_output = match &layer.mixer {
                HybridMixerWeights::FullAttention {
                    cache_layer_index,
                    weights,
                } => {
                    let HybridFullAttentionConfig::MultiHead {
                        num_attention_heads,
                        num_key_value_heads,
                        head_dim,
                        rotary_dim,
                        output_gate,
                    } = self.plan.full_attention
                    else {
                        return Err(CpuReferenceDenseDecoderError::Execution(
                            "multi-head weights were paired with a multi-latent plan".to_string(),
                        ));
                    };
                    forward_full_attention(
                        weights,
                        kv_cache,
                        &normalized,
                        FullAttentionForwardContext {
                            cache_layer_index: *cache_layer_index,
                            position,
                            output_slot,
                            sequence_slots: &sequence_slots[..=position],
                            shape: FullAttentionShape {
                                query_head_count: num_attention_heads,
                                kv_head_count: num_key_value_heads,
                                head_dim,
                                rotary_dim,
                                output_gate,
                            },
                            rms_norm_eps: self.plan.rms_norm_eps,
                            rope_theta: self.plan.rope_theta,
                            qk_normalization: normalization_kind,
                        },
                    )?
                }
                HybridMixerWeights::GatedDeltaNet {
                    state_layer_index,
                    weights,
                } => weights.forward(
                    &normalized,
                    &mut request_state.recurrent_layers[*state_layer_index],
                    self.plan.linear_attention,
                    self.plan.rms_norm_eps,
                )?,
                HybridMixerWeights::MultiLatentAttention {
                    cache_layer_index,
                    weights,
                } => weights.forward(
                    &normalized,
                    kv_cache,
                    MultiLatentAttentionForwardContext {
                        cache_layer_index: *cache_layer_index,
                        position,
                        output_slot,
                        sequence_slots: &sequence_slots[..=position],
                        config: self.plan.full_attention,
                        rms_norm_eps: self.plan.rms_norm_eps,
                        rope_theta: self.plan.rope_theta,
                    },
                )?,
                HybridMixerWeights::KeyGatedDelta {
                    state_layer_index,
                    weights,
                } => weights.forward(
                    &normalized,
                    &mut request_state.recurrent_layers[*state_layer_index],
                    self.plan.linear_attention,
                    self.plan.rms_norm_eps,
                )?,
            };
            let after_mixer = add_residual(&hidden, &mixer_output)?;
            let normalized = apply_normalization(
                &after_mixer,
                &layer.post_attention_norm,
                1,
                self.plan.hidden_size,
                self.plan.rms_norm_eps,
                normalization_kind,
            )?;
            let feed_forward = layer
                .feed_forward
                .forward(&normalized, self.plan.activation)?;
            hidden = add_residual(&after_mixer, &feed_forward)?;
        }
        request_state.next_position += 1;

        let hidden = apply_normalization(
            &hidden,
            &self.final_norm,
            1,
            self.plan.hidden_size,
            self.plan.rms_norm_eps,
            normalization_kind,
        )?;
        self.lm_head
            .as_ref()
            .unwrap_or(&self.token_embeddings)
            .project(&hidden, 1)
    }
}

impl BackendModelExecutor<CpuHybridExecutionResources> for CpuReferenceHybridDecoder {
    fn runtime_capability(&self) -> RuntimeCapability {
        CpuReferenceHybridDecoder::runtime_capability(self)
    }

    fn execution_dtype(&self) -> RuntimeDtype {
        CpuReferenceHybridDecoder::execution_dtype(self)
    }

    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
        resources: &mut CpuHybridExecutionResources,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
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
                        resources,
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

    fn complete_request(
        &mut self,
        resources: &mut CpuHybridExecutionResources,
        request_id: &RequestId,
    ) {
        resources.recurrent_state.complete_request(request_id);
    }
}

#[derive(Debug)]
struct HybridDecoderLayerWeights {
    input_norm: Vec<f32>,
    mixer: HybridMixerWeights,
    post_attention_norm: Vec<f32>,
    feed_forward: HybridFeedForwardWeights,
}

impl HybridDecoderLayerWeights {
    fn load(
        artifacts: &LocalModelArtifacts,
        names: &HybridDecoderLayerWeightNames,
        plan: &HybridDecoderExecutionPlan,
    ) -> Result<Self, CpuReferenceDenseDecoderError> {
        let mixer = match &names.mixer {
            HybridDecoderLayerKind::FullAttention {
                cache_layer_index,
                weights,
            } => {
                let HybridFullAttentionConfig::MultiHead {
                    num_attention_heads,
                    num_key_value_heads,
                    head_dim,
                    output_gate,
                    ..
                } = plan.full_attention
                else {
                    return Err(CpuReferenceDenseDecoderError::Unsupported(
                        "multi-head weights require a multi-head hybrid plan".to_string(),
                    ));
                };
                let query_size =
                    checked_product(num_attention_heads, head_dim, "hybrid query projection")?;
                let projected_query_size = checked_product(
                    query_size,
                    if output_gate { 2 } else { 1 },
                    "hybrid gated query projection",
                )?;
                let kv_size =
                    checked_product(num_key_value_heads, head_dim, "hybrid KV projection")?;
                HybridMixerWeights::FullAttention {
                    cache_layer_index: *cache_layer_index,
                    weights: Box::new(load_full_attention(
                        artifacts,
                        weights,
                        plan.hidden_size,
                        projected_query_size,
                        query_size,
                        kv_size,
                        head_dim,
                    )?),
                }
            }
            HybridDecoderLayerKind::GatedDeltaNet {
                state_layer_index,
                weights,
            } => {
                let config = plan.linear_attention;
                HybridMixerWeights::GatedDeltaNet {
                    state_layer_index: *state_layer_index,
                    weights: Box::new(GatedDeltaNetWeights::load(
                        artifacts,
                        weights,
                        plan.hidden_size,
                        config,
                    )?),
                }
            }
            HybridDecoderLayerKind::MultiLatentAttention {
                cache_layer_index,
                weights,
            } => HybridMixerWeights::MultiLatentAttention {
                cache_layer_index: *cache_layer_index,
                weights: Box::new(MultiLatentAttentionWeights::load(
                    artifacts,
                    weights,
                    plan.hidden_size,
                    plan.full_attention,
                )?),
            },
            HybridDecoderLayerKind::KeyGatedDelta {
                state_layer_index,
                weights,
            } => HybridMixerWeights::KeyGatedDelta {
                state_layer_index: *state_layer_index,
                weights: Box::new(KeyGatedDeltaWeights::load(
                    artifacts,
                    weights,
                    plan.hidden_size,
                    plan.linear_attention,
                )?),
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
            feed_forward: HybridFeedForwardWeights::load(
                artifacts,
                &names.feed_forward,
                plan.hidden_size,
            )?,
        })
    }
}

#[derive(Debug)]
enum HybridMixerWeights {
    FullAttention {
        cache_layer_index: usize,
        weights: Box<FullAttentionWeights>,
    },
    GatedDeltaNet {
        state_layer_index: usize,
        weights: Box<GatedDeltaNetWeights>,
    },
    MultiLatentAttention {
        cache_layer_index: usize,
        weights: Box<MultiLatentAttentionWeights>,
    },
    KeyGatedDelta {
        state_layer_index: usize,
        weights: Box<KeyGatedDeltaWeights>,
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

fn cpu_normalization(normalization: DecoderNormalization) -> CpuNormalization {
    match normalization {
        DecoderNormalization::Rms => CpuNormalization::Standard,
        DecoderNormalization::GemmaRms => CpuNormalization::Gemma,
    }
}

#[derive(Debug)]
enum HybridFeedForwardWeights {
    Dense {
        intermediate_size: usize,
        weights: DenseFeedForwardWeights,
    },
    MixtureOfExperts(MoeFeedForwardWeights),
}

impl HybridFeedForwardWeights {
    fn load(
        artifacts: &LocalModelArtifacts,
        feed_forward: &HybridFeedForward,
        hidden_size: usize,
    ) -> Result<Self, CpuReferenceDenseDecoderError> {
        match feed_forward {
            HybridFeedForward::Dense {
                intermediate_size,
                weights,
            } => Ok(Self::Dense {
                intermediate_size: *intermediate_size,
                weights: DenseFeedForwardWeights::load(
                    artifacts,
                    &weights.gate_weight,
                    &weights.up_weight,
                    &weights.down_weight,
                    hidden_size,
                    *intermediate_size,
                )?,
            }),
            HybridFeedForward::MixtureOfExperts { config, weights } => Ok(Self::MixtureOfExperts(
                MoeFeedForwardWeights::load(artifacts, config, weights, hidden_size)?,
            )),
        }
    }

    fn forward(
        &self,
        hidden: &[f32],
        activation: crate::models::DenseDecoderActivation,
    ) -> Result<Vec<f32>, CpuReferenceDenseDecoderError> {
        match self {
            Self::Dense {
                intermediate_size,
                weights,
            } => weights.forward(hidden, *intermediate_size, activation),
            Self::MixtureOfExperts(weights) => weights.forward(hidden, activation),
        }
    }
}

#[derive(Debug)]
struct MoeFeedForwardWeights {
    config: MoeFeedForwardConfig,
    gate: FloatMatrix,
    correction_bias: Option<Vec<f32>>,
    experts: Vec<DenseFeedForwardWeights>,
    shared_expert: Option<DenseFeedForwardWeights>,
}

impl MoeFeedForwardWeights {
    fn load(
        artifacts: &LocalModelArtifacts,
        config: &MoeFeedForwardConfig,
        names: &MoeFeedForwardWeightNames,
        hidden_size: usize,
    ) -> Result<Self, CpuReferenceDenseDecoderError> {
        if config.routed_expert_count == 0
            || config.experts_per_token == 0
            || config.experts_per_token > config.routed_expert_count
        {
            return Err(CpuReferenceDenseDecoderError::Unsupported(format!(
                "invalid MoE routed expert geometry: {} experts with top-k {}",
                config.routed_expert_count, config.experts_per_token
            )));
        }
        if config.expert_group_count == 0
            || !config
                .routed_expert_count
                .is_multiple_of(config.expert_group_count)
            || config.selected_expert_group_count == 0
            || config.selected_expert_group_count > config.expert_group_count
        {
            return Err(CpuReferenceDenseDecoderError::Unsupported(format!(
                "invalid MoE expert grouping: {} experts across {} groups selecting {}",
                config.routed_expert_count,
                config.expert_group_count,
                config.selected_expert_group_count
            )));
        }
        if !config.routed_scaling_factor.is_finite() || config.routed_scaling_factor <= 0.0 {
            return Err(CpuReferenceDenseDecoderError::Unsupported(format!(
                "MoE routed scaling factor {} must be finite and positive",
                config.routed_scaling_factor
            )));
        }
        if names.experts.len() != config.routed_expert_count {
            return Err(CpuReferenceDenseDecoderError::Unsupported(format!(
                "MoE weight map has {} experts but config requires {}",
                names.experts.len(),
                config.routed_expert_count
            )));
        }
        let experts = names
            .experts
            .iter()
            .map(|expert| {
                DenseFeedForwardWeights::load(
                    artifacts,
                    &expert.gate_weight,
                    &expert.up_weight,
                    &expert.down_weight,
                    hidden_size,
                    config.expert_intermediate_size,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let shared_intermediate_size = config
            .expert_intermediate_size
            .checked_mul(config.shared_expert_count)
            .ok_or_else(|| {
                CpuReferenceDenseDecoderError::Unsupported(
                    "shared expert intermediate size overflowed".to_string(),
                )
            })?;
        let shared_expert = match (config.shared_expert_count, &names.shared_expert) {
            (0, None) => None,
            (0, Some(_)) => {
                return Err(CpuReferenceDenseDecoderError::Unsupported(
                    "MoE weight map provides a shared expert but config disables it".to_string(),
                ));
            }
            (_, None) => {
                return Err(CpuReferenceDenseDecoderError::Unsupported(
                    "MoE config requires shared experts but the weight map has none".to_string(),
                ));
            }
            (_, Some(shared)) => Some(DenseFeedForwardWeights::load(
                artifacts,
                &shared.gate_weight,
                &shared.up_weight,
                &shared.down_weight,
                hidden_size,
                shared_intermediate_size,
            )?),
        };
        Ok(Self {
            config: config.clone(),
            gate: FloatMatrix::load(
                artifacts,
                &names.gate_weight,
                config.routed_expert_count,
                hidden_size,
            )?,
            correction_bias: names
                .correction_bias
                .as_deref()
                .map(|name| load_vector(artifacts, name, config.routed_expert_count))
                .transpose()?,
            experts,
            shared_expert,
        })
    }

    fn forward(
        &self,
        hidden: &[f32],
        activation: crate::models::DenseDecoderActivation,
    ) -> Result<Vec<f32>, CpuReferenceDenseDecoderError> {
        let logits = self.gate.project(hidden, 1)?;
        let probabilities = match self.config.router_activation {
            RouterActivation::Sigmoid => logits
                .iter()
                .map(|value| 1.0 / (1.0 + (-value).exp()))
                .collect::<Vec<_>>(),
            RouterActivation::Softmax => normalized_exponentials(&logits)?,
        };
        let selection_scores = probabilities
            .iter()
            .enumerate()
            .map(|(expert, probability)| {
                probability
                    + self
                        .correction_bias
                        .as_ref()
                        .map_or(0.0, |bias| bias[expert])
            })
            .collect::<Vec<_>>();
        let experts_per_group = self.config.routed_expert_count / self.config.expert_group_count;
        let mut group_scores = (0..self.config.expert_group_count)
            .map(|group| {
                let start = group * experts_per_group;
                let mut scores = selection_scores[start..start + experts_per_group].to_vec();
                scores.sort_by(|left, right| right.total_cmp(left));
                scores.into_iter().take(2).sum::<f32>()
            })
            .enumerate()
            .collect::<Vec<_>>();
        group_scores.sort_by(|(left_group, left), (right_group, right)| {
            right
                .total_cmp(left)
                .then_with(|| left_group.cmp(right_group))
        });
        let selected_groups = group_scores
            .into_iter()
            .take(self.config.selected_expert_group_count)
            .map(|(group, _)| group)
            .collect::<Vec<_>>();
        let mut selected = selection_scores
            .iter()
            .copied()
            .enumerate()
            .filter(|(expert, _)| selected_groups.contains(&(expert / experts_per_group)))
            .collect::<Vec<_>>();
        selected.sort_by(|(left_expert, left), (right_expert, right)| {
            right
                .total_cmp(left)
                .then_with(|| left_expert.cmp(right_expert))
        });
        selected.truncate(self.config.experts_per_token);

        let normalizer = if self.config.renormalize {
            let sum = selected
                .iter()
                .map(|(expert, _)| probabilities[*expert])
                .sum::<f32>();
            if !sum.is_finite() || sum <= 0.0 {
                return Err(CpuReferenceDenseDecoderError::Execution(
                    "MoE routed probability normalization is invalid".to_string(),
                ));
            }
            sum
        } else {
            1.0
        };
        let mut output = vec![0.0; hidden.len()];
        for (expert, _) in selected {
            let expert_output = self.experts[expert].forward(
                hidden,
                self.config.expert_intermediate_size,
                activation,
            )?;
            let weight = probabilities[expert] / normalizer * self.config.routed_scaling_factor;
            for (output, expert_value) in output.iter_mut().zip(expert_output) {
                *output += weight * expert_value;
            }
        }
        if let Some(shared_expert) = &self.shared_expert {
            let intermediate_size = self
                .config
                .expert_intermediate_size
                .checked_mul(self.config.shared_expert_count)
                .ok_or_else(|| {
                    CpuReferenceDenseDecoderError::Execution(
                        "shared expert intermediate size overflowed".to_string(),
                    )
                })?;
            let shared = shared_expert.forward(hidden, intermediate_size, activation)?;
            for (output, shared_value) in output.iter_mut().zip(shared) {
                *output += shared_value;
            }
        }
        Ok(output)
    }
}

#[derive(Debug)]
struct MultiLatentAttentionWeights {
    query: FloatMatrix,
    kv_a: FloatMatrix,
    kv_a_norm: Vec<f32>,
    kv_b: FloatMatrix,
    output: FloatMatrix,
}

#[derive(Clone, Copy)]
struct MultiLatentAttentionForwardContext<'a> {
    cache_layer_index: usize,
    position: usize,
    output_slot: CachePageId,
    sequence_slots: &'a [CachePageId],
    config: HybridFullAttentionConfig,
    rms_norm_eps: f32,
    rope_theta: f32,
}

impl MultiLatentAttentionWeights {
    fn load(
        artifacts: &LocalModelArtifacts,
        names: &HybridMultiLatentAttentionWeightNames,
        hidden_size: usize,
        config: HybridFullAttentionConfig,
    ) -> Result<Self, CpuReferenceDenseDecoderError> {
        let HybridFullAttentionConfig::MultiLatent {
            num_attention_heads,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            value_head_dim,
            ..
        } = config
        else {
            return Err(CpuReferenceDenseDecoderError::Unsupported(
                "MLA weights require a multi-latent attention plan".to_string(),
            ));
        };
        let query_head_dim = qk_nope_head_dim
            .checked_add(qk_rope_head_dim)
            .ok_or_else(|| {
                CpuReferenceDenseDecoderError::Unsupported(
                    "MLA query head size overflowed".to_string(),
                )
            })?;
        let query_size = checked_product(num_attention_heads, query_head_dim, "MLA query")?;
        let kv_a_size = kv_lora_rank.checked_add(qk_rope_head_dim).ok_or_else(|| {
            CpuReferenceDenseDecoderError::Unsupported(
                "MLA compressed KV size overflowed".to_string(),
            )
        })?;
        let expanded_head_dim = qk_nope_head_dim
            .checked_add(value_head_dim)
            .ok_or_else(|| {
                CpuReferenceDenseDecoderError::Unsupported(
                    "MLA expanded KV head size overflowed".to_string(),
                )
            })?;
        let expanded_size =
            checked_product(num_attention_heads, expanded_head_dim, "MLA expanded KV")?;
        let output_size = checked_product(num_attention_heads, value_head_dim, "MLA output")?;
        Ok(Self {
            query: FloatMatrix::load(artifacts, &names.query_weight, query_size, hidden_size)?,
            kv_a: FloatMatrix::load(artifacts, &names.kv_a_weight, kv_a_size, hidden_size)?,
            kv_a_norm: load_vector(artifacts, &names.kv_a_norm, kv_lora_rank)?,
            kv_b: FloatMatrix::load(artifacts, &names.kv_b_weight, expanded_size, kv_lora_rank)?,
            output: FloatMatrix::load(artifacts, &names.output_weight, hidden_size, output_size)?,
        })
    }

    fn forward(
        &self,
        hidden: &[f32],
        kv_cache: &mut CpuReferenceKvCache,
        context: MultiLatentAttentionForwardContext<'_>,
    ) -> Result<Vec<f32>, CpuReferenceDenseDecoderError> {
        let MultiLatentAttentionForwardContext {
            cache_layer_index,
            position,
            output_slot,
            sequence_slots,
            config,
            rms_norm_eps,
            rope_theta,
        } = context;
        let HybridFullAttentionConfig::MultiLatent {
            num_attention_heads,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            value_head_dim,
            skip_rope,
        } = config
        else {
            return Err(CpuReferenceDenseDecoderError::Execution(
                "MLA weights were paired with a multi-head attention plan".to_string(),
            ));
        };
        let query_head_dim = qk_nope_head_dim + qk_rope_head_dim;
        let query = self.query.project(hidden, 1)?;
        let compressed = self.kv_a.project(hidden, 1)?;
        let latent = apply_normalization(
            &compressed[..kv_lora_rank],
            &self.kv_a_norm,
            1,
            kv_lora_rank,
            rms_norm_eps,
            CpuNormalization::Standard,
        )?;
        let mut rope_key = compressed[kv_lora_rank..].to_vec();
        let mut query_nope = Vec::with_capacity(num_attention_heads * qk_nope_head_dim);
        let mut query_rope = Vec::with_capacity(num_attention_heads * qk_rope_head_dim);
        for head in 0..num_attention_heads {
            let offset = head * query_head_dim;
            query_nope.extend_from_slice(&query[offset..offset + qk_nope_head_dim]);
            query_rope
                .extend_from_slice(&query[offset + qk_nope_head_dim..offset + query_head_dim]);
        }
        if !skip_rope {
            apply_partial_neox_rope_inplace(
                &mut query_rope,
                num_attention_heads,
                qk_rope_head_dim,
                qk_rope_head_dim,
                position,
                rope_theta,
            )
            .map_err(kernel_error)?;
            apply_partial_neox_rope_inplace(
                &mut rope_key,
                1,
                qk_rope_head_dim,
                qk_rope_head_dim,
                position,
                rope_theta,
            )
            .map_err(kernel_error)?;
        }
        let mut cache_key = latent.clone();
        cache_key.extend_from_slice(&rope_key);
        kv_cache.write(cache_layer_index, output_slot, cache_key, latent.clone())?;
        let (cached_keys, cached_latents) = kv_cache.gather(cache_layer_index, sequence_slots)?;
        let key_width = kv_lora_rank + qk_rope_head_dim;
        let expanded_head_dim = qk_nope_head_dim + value_head_dim;
        let mut attention = vec![0.0; num_attention_heads * value_head_dim];
        let scale = (query_head_dim as f32).sqrt().recip();
        for head in 0..num_attention_heads {
            let query_nope_offset = head * qk_nope_head_dim;
            let query_rope_offset = head * qk_rope_head_dim;
            let mut scores = Vec::with_capacity(sequence_slots.len());
            let mut expanded_values = Vec::with_capacity(sequence_slots.len() * value_head_dim);
            for token in 0..sequence_slots.len() {
                let latent_offset = token * kv_lora_rank;
                let expanded = self.kv_b.project(
                    &cached_latents[latent_offset..latent_offset + kv_lora_rank],
                    1,
                )?;
                let expanded_offset = head * expanded_head_dim;
                let key_offset = token * key_width + kv_lora_rank;
                let nope_score = query_nope
                    [query_nope_offset..query_nope_offset + qk_nope_head_dim]
                    .iter()
                    .zip(&expanded[expanded_offset..expanded_offset + qk_nope_head_dim])
                    .map(|(query, key)| query * key)
                    .sum::<f32>();
                let rope_score = query_rope
                    [query_rope_offset..query_rope_offset + qk_rope_head_dim]
                    .iter()
                    .zip(&cached_keys[key_offset..key_offset + qk_rope_head_dim])
                    .map(|(query, key)| query * key)
                    .sum::<f32>();
                scores.push((nope_score + rope_score) * scale);
                expanded_values.extend_from_slice(
                    &expanded
                        [expanded_offset + qk_nope_head_dim..expanded_offset + expanded_head_dim],
                );
            }
            let probabilities = normalized_exponentials(&scores)?;
            let output_offset = head * value_head_dim;
            for (token, probability) in probabilities.into_iter().enumerate() {
                for dimension in 0..value_head_dim {
                    attention[output_offset + dimension] +=
                        probability * expanded_values[token * value_head_dim + dimension];
                }
            }
        }
        self.output.project(&attention, 1)
    }
}

#[derive(Debug)]
struct KeyGatedDeltaWeights {
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    query: FloatMatrix,
    key: FloatMatrix,
    value: FloatMatrix,
    beta: FloatMatrix,
    forget_a: FloatMatrix,
    forget_b: FloatMatrix,
    gate_a: FloatMatrix,
    gate_b: FloatMatrix,
    conv_weight: Vec<f32>,
    output_norm: Vec<f32>,
    output: FloatMatrix,
}

impl KeyGatedDeltaWeights {
    fn load(
        artifacts: &LocalModelArtifacts,
        names: &KeyGatedDeltaWeightNames,
        hidden_size: usize,
        config: HybridLinearAttentionConfig,
    ) -> Result<Self, CpuReferenceDenseDecoderError> {
        let HybridLinearAttentionConfig::KeyGatedDelta {
            conv_kernel_dim,
            key_head_dim,
            value_head_dim,
            num_heads,
        } = config
        else {
            return Err(CpuReferenceDenseDecoderError::Unsupported(
                "KDA weights require a key-gated delta attention plan".to_string(),
            ));
        };
        let key_size = checked_product(num_heads, key_head_dim, "KDA key")?;
        let value_size = checked_product(num_heads, value_head_dim, "KDA value")?;
        let query_conv = load_tensor(
            artifacts,
            &names.query_conv_weight,
            &[key_size, conv_kernel_dim],
        )?;
        let key_conv = load_tensor(
            artifacts,
            &names.key_conv_weight,
            &[key_size, conv_kernel_dim],
        )?;
        let value_conv = load_tensor(
            artifacts,
            &names.value_conv_weight,
            &[value_size, conv_kernel_dim],
        )?;
        let mut conv_weight =
            Vec::with_capacity(query_conv.len() + key_conv.len() + value_conv.len());
        conv_weight.extend(query_conv);
        conv_weight.extend(key_conv);
        conv_weight.extend(value_conv);
        Ok(Self {
            a_log: load_tensor(artifacts, &names.a_log, &[1, 1, num_heads, 1])?,
            dt_bias: load_vector(artifacts, &names.dt_bias, key_size)?,
            query: FloatMatrix::load(artifacts, &names.query_weight, key_size, hidden_size)?,
            key: FloatMatrix::load(artifacts, &names.key_weight, key_size, hidden_size)?,
            value: FloatMatrix::load(artifacts, &names.value_weight, value_size, hidden_size)?,
            beta: FloatMatrix::load(artifacts, &names.beta_weight, num_heads, hidden_size)?,
            forget_a: FloatMatrix::load(
                artifacts,
                &names.forget_a_weight,
                key_head_dim,
                hidden_size,
            )?,
            forget_b: FloatMatrix::load(artifacts, &names.forget_b_weight, key_size, key_head_dim)?,
            gate_a: FloatMatrix::load(artifacts, &names.gate_a_weight, key_head_dim, hidden_size)?,
            gate_b: FloatMatrix::load(artifacts, &names.gate_b_weight, value_size, key_head_dim)?,
            conv_weight,
            output_norm: load_vector(artifacts, &names.output_norm, value_head_dim)?,
            output: FloatMatrix::load(artifacts, &names.output_weight, hidden_size, value_size)?,
        })
    }

    fn forward(
        &self,
        hidden: &[f32],
        state: &mut LinearAttentionLayerState,
        config: HybridLinearAttentionConfig,
        rms_norm_eps: f32,
    ) -> Result<Vec<f32>, CpuReferenceDenseDecoderError> {
        let HybridLinearAttentionConfig::KeyGatedDelta {
            conv_kernel_dim,
            key_head_dim,
            value_head_dim,
            num_heads,
        } = config
        else {
            return Err(CpuReferenceDenseDecoderError::Execution(
                "KDA weights were paired with gated-delta net attention".to_string(),
            ));
        };
        let key_size = num_heads * key_head_dim;
        let value_size = num_heads * value_head_dim;
        let mut qkv = self.query.project(hidden, 1)?;
        qkv.extend(self.key.project(hidden, 1)?);
        qkv.extend(self.value.project(hidden, 1)?);
        let mut qkv = causal_depthwise_conv1d_step(
            &qkv,
            &self.conv_weight,
            &mut state.conv,
            key_size * 2 + value_size,
            conv_kernel_dim,
        )
        .map_err(kernel_error)?;
        for value in &mut qkv {
            *value *= 1.0 / (1.0 + (-*value).exp());
        }
        let mut query = qkv[..key_size].to_vec();
        let mut key = qkv[key_size..key_size * 2].to_vec();
        let value = &qkv[key_size * 2..];
        l2_normalize_heads(
            &mut query,
            num_heads,
            key_head_dim,
            (key_head_dim as f32).sqrt().recip(),
        )?;
        l2_normalize_heads(&mut key, num_heads, key_head_dim, 1.0)?;

        let raw_forget = self
            .forget_b
            .project(&self.forget_a.project(hidden, 1)?, 1)?;
        let decay = raw_forget
            .iter()
            .zip(&self.dt_bias)
            .enumerate()
            .map(|(index, (raw, bias))| {
                let head = index / key_head_dim;
                (-self.a_log[head].exp() * softplus(*raw + *bias)).exp()
            })
            .collect::<Vec<_>>();
        let beta = self
            .beta
            .project(hidden, 1)?
            .into_iter()
            .map(|value| 1.0 / (1.0 + (-value).exp()))
            .collect::<Vec<_>>();
        let core = key_gated_delta_rule_step(
            &query,
            &key,
            value,
            &decay,
            &beta,
            &mut state.recurrent,
            KeyGatedDeltaRuleShape {
                head_count: num_heads,
                key_head_dim,
                value_head_dim,
            },
        )
        .map_err(kernel_error)?;
        let mut normalized = rms_norm(
            &core,
            &self.output_norm,
            num_heads,
            value_head_dim,
            rms_norm_eps,
        )
        .map_err(kernel_error)?;
        let gate = self.gate_b.project(&self.gate_a.project(hidden, 1)?, 1)?;
        for (value, gate) in normalized.iter_mut().zip(gate) {
            *value *= 1.0 / (1.0 + (-gate).exp());
        }
        self.output.project(&normalized, 1)
    }
}

fn normalized_exponentials(values: &[f32]) -> Result<Vec<f32>, CpuReferenceDenseDecoderError> {
    let maximum = values
        .iter()
        .copied()
        .max_by(f32::total_cmp)
        .ok_or_else(|| {
            CpuReferenceDenseDecoderError::Execution(
                "cannot normalize an empty probability vector".to_string(),
            )
        })?;
    let mut probabilities = values
        .iter()
        .map(|value| (*value - maximum).exp())
        .collect::<Vec<_>>();
    let normalizer = probabilities.iter().sum::<f32>();
    if !normalizer.is_finite() || normalizer <= 0.0 {
        return Err(CpuReferenceDenseDecoderError::Execution(
            "probability normalization is invalid".to_string(),
        ));
    }
    for probability in &mut probabilities {
        *probability /= normalizer;
    }
    Ok(probabilities)
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
        hidden_size: usize,
        config: HybridLinearAttentionConfig,
    ) -> Result<Self, CpuReferenceDenseDecoderError> {
        let HybridLinearAttentionConfig::GatedDeltaNet {
            conv_kernel_dim,
            key_head_dim,
            value_head_dim,
            num_key_heads,
            num_value_heads,
        } = config
        else {
            return Err(CpuReferenceDenseDecoderError::Unsupported(
                "GDN weights require a gated-delta linear attention plan".to_string(),
            ));
        };
        let linear_key_size = checked_product(num_key_heads, key_head_dim, "linear key")?;
        let linear_value_size = checked_product(num_value_heads, value_head_dim, "linear value")?;
        let conv_dim = linear_key_size
            .checked_mul(2)
            .and_then(|size| size.checked_add(linear_value_size))
            .ok_or_else(|| {
                CpuReferenceDenseDecoderError::Unsupported(
                    "linear convolution size overflowed".to_string(),
                )
            })?;
        Ok(Self {
            a_log: load_vector(artifacts, &names.a_log, num_value_heads)?,
            conv1d_weight: load_tensor(
                artifacts,
                &names.conv1d_weight,
                &[conv_dim, 1, conv_kernel_dim],
            )?,
            dt_bias: load_vector(artifacts, &names.dt_bias, num_value_heads)?,
            in_proj_a: FloatMatrix::load(
                artifacts,
                &names.in_proj_a_weight,
                num_value_heads,
                hidden_size,
            )?,
            in_proj_b: FloatMatrix::load(
                artifacts,
                &names.in_proj_b_weight,
                num_value_heads,
                hidden_size,
            )?,
            in_proj_qkv: FloatMatrix::load(
                artifacts,
                &names.in_proj_qkv_weight,
                conv_dim,
                hidden_size,
            )?,
            in_proj_z: FloatMatrix::load(
                artifacts,
                &names.in_proj_z_weight,
                linear_value_size,
                hidden_size,
            )?,
            norm_weight: load_vector(artifacts, &names.norm_weight, value_head_dim)?,
            output: FloatMatrix::load(
                artifacts,
                &names.output_weight,
                hidden_size,
                linear_value_size,
            )?,
        })
    }

    fn forward(
        &self,
        hidden: &[f32],
        state: &mut LinearAttentionLayerState,
        config: HybridLinearAttentionConfig,
        rms_norm_eps: f32,
    ) -> Result<Vec<f32>, CpuReferenceDenseDecoderError> {
        let HybridLinearAttentionConfig::GatedDeltaNet {
            conv_kernel_dim,
            key_head_dim,
            value_head_dim,
            num_key_heads,
            num_value_heads,
        } = config
        else {
            return Err(CpuReferenceDenseDecoderError::Execution(
                "GDN weights were paired with key-gated delta attention".to_string(),
            ));
        };
        let mut qkv = self.in_proj_qkv.project(hidden, 1)?;
        qkv = causal_depthwise_conv1d_step(
            &qkv,
            &self.conv1d_weight,
            &mut state.conv,
            qkv.len(),
            conv_kernel_dim,
        )
        .map_err(kernel_error)?;
        for value in &mut qkv {
            *value *= 1.0 / (1.0 + (-*value).exp());
        }

        let key_size = checked_product(num_key_heads, key_head_dim, "linear key")?;
        let value_size = checked_product(num_value_heads, value_head_dim, "linear value")?;
        let mut query = qkv[..key_size].to_vec();
        let mut key = qkv[key_size..key_size * 2].to_vec();
        let value = &qkv[key_size * 2..key_size * 2 + value_size];
        l2_normalize_heads(
            &mut query,
            num_key_heads,
            key_head_dim,
            (key_head_dim as f32).sqrt().recip(),
        )?;
        l2_normalize_heads(&mut key, num_key_heads, key_head_dim, 1.0)?;

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
                key_head_count: num_key_heads,
                value_head_count: num_value_heads,
                key_head_dim,
                value_head_dim,
            },
        )
        .map_err(kernel_error)?;
        let mut normalized = rms_norm(
            &core,
            &self.norm_weight,
            num_value_heads,
            value_head_dim,
            rms_norm_eps,
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
    recurrent_layers: Vec<LinearAttentionLayerState>,
}

impl HybridRequestState {
    fn new(layout: RecurrentStateLayout) -> Result<Self, CpuReferenceDenseDecoderError> {
        let conv_state_size = layout.conv_elements_per_layer().ok_or_else(|| {
            CpuReferenceDenseDecoderError::Unsupported(
                "hybrid convolution state size overflowed".to_string(),
            )
        })?;
        let recurrent_state_size = layout.temporal_elements_per_layer().ok_or_else(|| {
            CpuReferenceDenseDecoderError::Unsupported(
                "hybrid recurrent state size overflowed".to_string(),
            )
        })?;
        Ok(Self {
            next_position: 0,
            recurrent_layers: (0..layout.layer_count)
                .map(|_| LinearAttentionLayerState {
                    conv: vec![0.0; conv_state_size],
                    recurrent: vec![0.0; recurrent_state_size],
                })
                .collect(),
        })
    }
}

#[derive(Debug)]
struct LinearAttentionLayerState {
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

#[cfg(test)]
mod tests {
    use super::CpuHybridStatePool;
    use crate::models::RecurrentStateLayout;
    use crate::types::RequestId;

    fn state_layout() -> RecurrentStateLayout {
        RecurrentStateLayout {
            layer_count: 2,
            conv_kernel_dim: 3,
            key_head_count: 2,
            value_head_count: 2,
            key_head_dim: 4,
            value_head_dim: 4,
        }
    }

    #[test]
    fn backend_owned_hybrid_state_survives_decode_and_is_released_on_completion() {
        let mut pool = CpuHybridStatePool::new(state_layout()).expect("valid state pool");
        let request_id = RequestId::from("hybrid-state-lifecycle");

        let initial = pool
            .request_state_mut(&request_id, 0)
            .expect("prefill allocates request state");
        initial.next_position = 1;
        initial.recurrent_layers[0].conv[0] = 3.0;
        initial.recurrent_layers[0].recurrent[0] = 5.0;

        let decode = pool
            .request_state_mut(&request_id, 1)
            .expect("decode reuses backend-owned request state");
        assert_eq!(decode.recurrent_layers[0].conv[0], 3.0);
        assert_eq!(decode.recurrent_layers[0].recurrent[0], 5.0);

        pool.complete_request(&request_id);
        let error = pool
            .request_state_mut(&request_id, 1)
            .expect_err("completed state must not remain addressable");
        assert!(error.to_string().contains("has no recurrent state"));

        let reused = pool
            .request_state_mut(&request_id, 0)
            .expect("a new request may reuse the identifier after completion");
        assert_eq!(reused.next_position, 0);
        assert_eq!(reused.recurrent_layers[0].conv[0], 0.0);
        assert_eq!(reused.recurrent_layers[0].recurrent[0], 0.0);
    }
}
