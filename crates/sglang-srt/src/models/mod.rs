mod deepseek;
mod glm;
mod kimi_linear;
mod mla_moe_weights;
mod qwen;
mod qwen3_5;

use std::fmt;
use std::path::Path;

use crate::backend::{RuntimeDtype, RuntimeRequirements};
use crate::kv_cache::KvCacheModelLayout;
use crate::model_artifacts::{
    CheckpointTensorRequirement, CheckpointTopology, HfModelConfig, LocalModelArtifacts,
    ModelArtifactError,
};

pub(crate) use deepseek::DEEPSEEK_V4_ADAPTER;
pub(crate) use glm::GLM_MOE_DSA_ADAPTER;
pub(crate) use kimi_linear::KIMI_LINEAR_ADAPTER;
pub(crate) use qwen::{QWEN2_ADAPTER, QWEN3_ADAPTER};
pub(crate) use qwen3_5::QWEN3_5_ADAPTER;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionArchitecture {
    None,
    MultiHead {
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
    },
    MultiLatent {
        num_attention_heads: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        value_head_dim: usize,
    },
    Hybrid {
        num_attention_heads: usize,
        num_key_value_heads: usize,
        attention_head_dim: usize,
        linear_num_key_heads: usize,
        linear_num_value_heads: usize,
        linear_key_head_dim: usize,
        linear_value_head_dim: usize,
    },
    HybridMultiLatent {
        num_attention_heads: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        value_head_dim: usize,
        linear_num_heads: usize,
        linear_key_head_dim: usize,
        linear_value_head_dim: usize,
    },
}

impl AttentionArchitecture {
    pub fn family(self) -> AttentionFamily {
        match self {
            Self::None => AttentionFamily::None,
            Self::MultiHead { .. } => AttentionFamily::MultiHead,
            Self::MultiLatent { .. } => AttentionFamily::MultiLatent,
            Self::Hybrid { .. } | Self::HybridMultiLatent { .. } => AttentionFamily::Hybrid,
        }
    }

    fn validate_tensor_parallel(self, tensor_parallel_size: usize) -> Result<(), String> {
        let (num_attention_heads, head_counts) = match self {
            Self::None => return Ok(()),
            Self::MultiHead {
                num_attention_heads,
                num_key_value_heads,
                ..
            } => (num_attention_heads, vec![("KV", num_key_value_heads)]),
            Self::MultiLatent {
                num_attention_heads,
                ..
            } => (num_attention_heads, Vec::new()),
            Self::Hybrid {
                num_attention_heads,
                num_key_value_heads,
                linear_num_key_heads,
                linear_num_value_heads,
                ..
            } => (
                num_attention_heads,
                vec![
                    ("KV", num_key_value_heads),
                    ("linear key", linear_num_key_heads),
                    ("linear value", linear_num_value_heads),
                ],
            ),
            Self::HybridMultiLatent {
                num_attention_heads,
                linear_num_heads,
                ..
            } => (num_attention_heads, vec![("linear", linear_num_heads)]),
        };

        if !num_attention_heads.is_multiple_of(tensor_parallel_size) {
            return Err(format!(
                "attention head count {num_attention_heads} must be divisible by tensor parallel size {tensor_parallel_size}"
            ));
        }

        for (name, head_count) in head_counts {
            let valid = if head_count >= tensor_parallel_size {
                head_count.is_multiple_of(tensor_parallel_size)
            } else {
                tensor_parallel_size.is_multiple_of(head_count)
            };
            if !valid {
                return Err(format!(
                    "{name} head count {head_count} must shard across or replicate evenly over tensor parallel size {tensor_parallel_size}"
                ));
            }
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionFamily {
    None,
    MultiHead,
    MultiLatent,
    Hybrid,
}

impl fmt::Display for AttentionFamily {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::None => "none",
            Self::MultiHead => "multi-head attention",
            Self::MultiLatent => "multi-latent attention",
            Self::Hybrid => "hybrid full and linear attention",
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeedForwardArchitecture {
    None,
    Dense {
        intermediate_size: usize,
    },
    MixtureOfExperts {
        routed_expert_count: usize,
        experts_per_token: usize,
        shared_expert_count: usize,
        expert_intermediate_size: usize,
    },
}

impl FeedForwardArchitecture {
    pub fn family(self) -> FeedForwardFamily {
        match self {
            Self::None => FeedForwardFamily::None,
            Self::Dense { .. } => FeedForwardFamily::Dense,
            Self::MixtureOfExperts { .. } => FeedForwardFamily::MixtureOfExperts,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeedForwardFamily {
    None,
    Dense,
    MixtureOfExperts,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DenseDecoderActivation {
    Silu,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DenseDecoderLayerWeightNames {
    pub(crate) input_norm: String,
    pub(crate) query_weight: String,
    pub(crate) query_bias: Option<String>,
    pub(crate) query_norm: Option<String>,
    pub(crate) key_weight: String,
    pub(crate) key_bias: Option<String>,
    pub(crate) key_norm: Option<String>,
    pub(crate) value_weight: String,
    pub(crate) value_bias: Option<String>,
    pub(crate) output_weight: String,
    pub(crate) output_bias: Option<String>,
    pub(crate) post_attention_norm: String,
    pub(crate) gate_weight: String,
    pub(crate) up_weight: String,
    pub(crate) down_weight: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DenseDecoderWeightNames {
    pub(crate) token_embeddings: String,
    pub(crate) final_norm: String,
    pub(crate) lm_head: Option<String>,
    pub(crate) layers: Vec<DenseDecoderLayerWeightNames>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DenseDecoderExecutionPlan {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_theta: f32,
    pub(crate) activation: DenseDecoderActivation,
    pub(crate) weights: DenseDecoderWeightNames,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DenseFeedForwardWeightNames {
    pub(crate) gate_weight: String,
    pub(crate) up_weight: String,
    pub(crate) down_weight: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct HybridFullAttentionWeightNames {
    pub(crate) query_weight: String,
    pub(crate) query_norm: String,
    pub(crate) key_weight: String,
    pub(crate) key_norm: String,
    pub(crate) value_weight: String,
    pub(crate) output_weight: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct HybridMultiLatentAttentionWeightNames {
    pub(crate) query_weight: String,
    pub(crate) kv_a_weight: String,
    pub(crate) kv_a_norm: String,
    pub(crate) kv_b_weight: String,
    pub(crate) output_weight: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GatedDeltaNetWeightNames {
    pub(crate) a_log: String,
    pub(crate) conv1d_weight: String,
    pub(crate) dt_bias: String,
    pub(crate) in_proj_a_weight: String,
    pub(crate) in_proj_b_weight: String,
    pub(crate) in_proj_qkv_weight: String,
    pub(crate) in_proj_z_weight: String,
    pub(crate) norm_weight: String,
    pub(crate) output_weight: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KeyGatedDeltaWeightNames {
    pub(crate) a_log: String,
    pub(crate) dt_bias: String,
    pub(crate) query_weight: String,
    pub(crate) key_weight: String,
    pub(crate) value_weight: String,
    pub(crate) beta_weight: String,
    pub(crate) forget_a_weight: String,
    pub(crate) forget_b_weight: String,
    pub(crate) gate_a_weight: String,
    pub(crate) gate_b_weight: String,
    pub(crate) query_conv_weight: String,
    pub(crate) key_conv_weight: String,
    pub(crate) value_conv_weight: String,
    pub(crate) output_norm: String,
    pub(crate) output_weight: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecoderNormalization {
    Rms,
    GemmaRms,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HybridFullAttentionConfig {
    MultiHead {
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        output_gate: bool,
    },
    MultiLatent {
        num_attention_heads: usize,
        kv_lora_rank: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        value_head_dim: usize,
        skip_rope: bool,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HybridLinearAttentionConfig {
    GatedDeltaNet {
        conv_kernel_dim: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        num_key_heads: usize,
        num_value_heads: usize,
    },
    KeyGatedDelta {
        conv_kernel_dim: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        num_heads: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RouterActivation {
    Sigmoid,
    Softmax,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MoeFeedForwardConfig {
    pub(crate) routed_expert_count: usize,
    pub(crate) experts_per_token: usize,
    pub(crate) expert_group_count: usize,
    pub(crate) selected_expert_group_count: usize,
    pub(crate) shared_expert_count: usize,
    pub(crate) expert_intermediate_size: usize,
    pub(crate) renormalize: bool,
    pub(crate) router_activation: RouterActivation,
    pub(crate) routed_scaling_factor: f32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MoeFeedForwardWeightNames {
    pub(crate) gate_weight: String,
    pub(crate) correction_bias: Option<String>,
    pub(crate) experts: Vec<DenseFeedForwardWeightNames>,
    pub(crate) shared_expert: Option<DenseFeedForwardWeightNames>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum HybridFeedForward {
    Dense {
        intermediate_size: usize,
        weights: DenseFeedForwardWeightNames,
    },
    MixtureOfExperts {
        config: MoeFeedForwardConfig,
        weights: MoeFeedForwardWeightNames,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum HybridDecoderLayerKind {
    FullAttention {
        cache_layer_index: usize,
        weights: HybridFullAttentionWeightNames,
    },
    GatedDeltaNet {
        state_layer_index: usize,
        weights: GatedDeltaNetWeightNames,
    },
    MultiLatentAttention {
        cache_layer_index: usize,
        weights: HybridMultiLatentAttentionWeightNames,
    },
    KeyGatedDelta {
        state_layer_index: usize,
        weights: KeyGatedDeltaWeightNames,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HybridDecoderLayerWeightNames {
    pub(crate) input_norm: String,
    pub(crate) mixer: HybridDecoderLayerKind,
    pub(crate) post_attention_norm: String,
    pub(crate) feed_forward: HybridFeedForward,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HybridDecoderWeightNames {
    pub(crate) token_embeddings: String,
    pub(crate) final_norm: String,
    pub(crate) lm_head: Option<String>,
    pub(crate) layers: Vec<HybridDecoderLayerWeightNames>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HybridDecoderExecutionPlan {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_theta: f32,
    pub(crate) normalization: DecoderNormalization,
    pub(crate) full_attention: HybridFullAttentionConfig,
    pub(crate) linear_attention: HybridLinearAttentionConfig,
    pub(crate) activation: DenseDecoderActivation,
    pub(crate) weights: HybridDecoderWeightNames,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelCacheArchitecture {
    None,
    PagedKv,
    HybridState {
        full_attention_layer_count: usize,
        recurrent_state_layer_count: usize,
    },
}

impl ModelCacheArchitecture {
    pub fn supports_radix_prefix_cache(self) -> bool {
        matches!(self, Self::PagedKv)
    }

    pub fn supports_kv_only_transfer(self) -> bool {
        !matches!(self, Self::HybridState { .. })
    }
}

impl fmt::Display for FeedForwardFamily {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::None => "none",
            Self::Dense => "dense feed-forward",
            Self::MixtureOfExperts => "mixture-of-experts",
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelExecutionArchitecture {
    Transformer {
        attention: AttentionArchitecture,
        feed_forward: FeedForwardArchitecture,
    },
}

impl ModelExecutionArchitecture {
    pub fn attention_family(self) -> AttentionFamily {
        match self {
            Self::Transformer { attention, .. } => attention.family(),
        }
    }

    pub fn feed_forward_family(self) -> FeedForwardFamily {
        match self {
            Self::Transformer { feed_forward, .. } => feed_forward.family(),
        }
    }

    fn validate_tensor_parallel(self, tensor_parallel_size: usize) -> Result<(), String> {
        match self {
            Self::Transformer { attention, .. } => {
                attention.validate_tensor_parallel(tensor_parallel_size)
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelDefinition {
    architecture: &'static str,
    model_type: Option<String>,
    vocab_size: Option<usize>,
    max_context_length: Option<usize>,
    eos_token_ids: Vec<u32>,
    execution: ModelExecutionArchitecture,
    supported_dtypes: Vec<RuntimeDtype>,
    kv_cache_layout: Option<KvCacheModelLayout>,
    cache_architecture: ModelCacheArchitecture,
    dense_decoder: Option<DenseDecoderExecutionPlan>,
    hybrid_decoder: Option<HybridDecoderExecutionPlan>,
    checkpoint_topology: CheckpointTopology,
}

impl ModelDefinition {
    pub(crate) fn new(
        architecture: &'static str,
        config: &HfModelConfig,
        execution: ModelExecutionArchitecture,
        supported_dtypes: Vec<RuntimeDtype>,
        kv_cache_layout: Option<KvCacheModelLayout>,
    ) -> Self {
        debug_assert!(!supported_dtypes.is_empty());
        Self {
            architecture,
            model_type: config.model_type.clone(),
            vocab_size: None,
            max_context_length: None,
            eos_token_ids: config.eos_token_ids.clone(),
            execution,
            supported_dtypes,
            kv_cache_layout,
            cache_architecture: if kv_cache_layout.is_some() {
                ModelCacheArchitecture::PagedKv
            } else {
                ModelCacheArchitecture::None
            },
            dense_decoder: None,
            hybrid_decoder: None,
            checkpoint_topology: CheckpointTopology::default(),
        }
    }

    pub(crate) fn with_serving_metadata(
        mut self,
        vocab_size: usize,
        max_context_length: usize,
    ) -> Self {
        self.vocab_size = Some(vocab_size);
        self.max_context_length = Some(max_context_length);
        self
    }

    pub(crate) fn with_checkpoint_topology(mut self, topology: CheckpointTopology) -> Self {
        self.checkpoint_topology = topology;
        self
    }

    pub(crate) fn with_dense_decoder(mut self, plan: DenseDecoderExecutionPlan) -> Self {
        self.dense_decoder = Some(plan);
        self
    }

    pub(crate) fn with_hybrid_decoder(mut self, plan: HybridDecoderExecutionPlan) -> Self {
        let full_attention_layer_count = plan
            .weights
            .layers
            .iter()
            .filter(|layer| {
                matches!(
                    layer.mixer,
                    HybridDecoderLayerKind::FullAttention { .. }
                        | HybridDecoderLayerKind::MultiLatentAttention { .. }
                )
            })
            .count();
        let recurrent_state_layer_count = plan.weights.layers.len() - full_attention_layer_count;
        self.cache_architecture = ModelCacheArchitecture::HybridState {
            full_attention_layer_count,
            recurrent_state_layer_count,
        };
        self.hybrid_decoder = Some(plan);
        self
    }

    pub fn architecture(&self) -> &'static str {
        self.architecture
    }

    pub fn model_type(&self) -> Option<&str> {
        self.model_type.as_deref()
    }

    pub fn vocab_size(&self) -> Option<usize> {
        self.vocab_size
    }

    pub fn max_context_length(&self) -> Option<usize> {
        self.max_context_length
    }

    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    pub fn checkpoint_topology(&self) -> &CheckpointTopology {
        &self.checkpoint_topology
    }

    pub(crate) fn validate_checkpoint(
        &self,
        artifacts: &LocalModelArtifacts,
    ) -> Result<(), ModelArtifactError> {
        artifacts.validate_checkpoint_topology(&self.checkpoint_topology)
    }

    pub fn execution(&self) -> ModelExecutionArchitecture {
        self.execution
    }

    pub fn supported_dtypes(&self) -> &[RuntimeDtype] {
        &self.supported_dtypes
    }

    pub fn kv_cache_layout(&self) -> Option<KvCacheModelLayout> {
        self.kv_cache_layout
    }

    pub fn cache_architecture(&self) -> ModelCacheArchitecture {
        self.cache_architecture
    }

    pub(crate) fn dense_decoder(&self) -> Option<&DenseDecoderExecutionPlan> {
        self.dense_decoder.as_ref()
    }

    pub(crate) fn hybrid_decoder(&self) -> Option<&HybridDecoderExecutionPlan> {
        self.hybrid_decoder.as_ref()
    }

    pub(crate) fn validate_dense_decoder_checkpoint(
        &self,
        artifacts: &LocalModelArtifacts,
    ) -> Result<(), ModelArtifactError> {
        let Some(plan) = self.dense_decoder() else {
            return Err(invalid_dense_decoder_checkpoint(
                artifacts,
                "model definition does not include a dense decoder execution plan",
            ));
        };
        let ModelExecutionArchitecture::Transformer {
            attention:
                AttentionArchitecture::MultiHead {
                    num_attention_heads,
                    num_key_value_heads,
                    head_dim,
                },
            feed_forward: FeedForwardArchitecture::Dense { intermediate_size },
        } = self.execution
        else {
            return Err(invalid_dense_decoder_checkpoint(
                artifacts,
                "dense decoder plan requires multi-head attention and dense feed-forward components",
            ));
        };
        let query_size = num_attention_heads.checked_mul(head_dim).ok_or_else(|| {
            invalid_dense_decoder_checkpoint(artifacts, "query projection size overflowed")
        })?;
        let kv_size = num_key_value_heads.checked_mul(head_dim).ok_or_else(|| {
            invalid_dense_decoder_checkpoint(artifacts, "KV projection size overflowed")
        })?;

        require_tensor_shape(
            artifacts,
            &plan.weights.token_embeddings,
            &[plan.vocab_size, plan.hidden_size],
        )?;
        require_tensor_shape(artifacts, &plan.weights.final_norm, &[plan.hidden_size])?;
        if let Some(lm_head) = &plan.weights.lm_head {
            require_tensor_shape(artifacts, lm_head, &[plan.vocab_size, plan.hidden_size])?;
        }
        if plan.weights.layers.is_empty() {
            return Err(invalid_dense_decoder_checkpoint(
                artifacts,
                "dense decoder weight map has no layers",
            ));
        }

        for layer in &plan.weights.layers {
            require_tensor_shape(artifacts, &layer.input_norm, &[plan.hidden_size])?;
            require_tensor_shape(
                artifacts,
                &layer.query_weight,
                &[query_size, plan.hidden_size],
            )?;
            require_optional_tensor_shape(artifacts, layer.query_bias.as_deref(), &[query_size])?;
            require_optional_tensor_shape(artifacts, layer.query_norm.as_deref(), &[head_dim])?;
            require_tensor_shape(artifacts, &layer.key_weight, &[kv_size, plan.hidden_size])?;
            require_optional_tensor_shape(artifacts, layer.key_bias.as_deref(), &[kv_size])?;
            require_optional_tensor_shape(artifacts, layer.key_norm.as_deref(), &[head_dim])?;
            require_tensor_shape(artifacts, &layer.value_weight, &[kv_size, plan.hidden_size])?;
            require_optional_tensor_shape(artifacts, layer.value_bias.as_deref(), &[kv_size])?;
            require_tensor_shape(
                artifacts,
                &layer.output_weight,
                &[plan.hidden_size, query_size],
            )?;
            require_optional_tensor_shape(
                artifacts,
                layer.output_bias.as_deref(),
                &[plan.hidden_size],
            )?;
            require_tensor_shape(artifacts, &layer.post_attention_norm, &[plan.hidden_size])?;
            require_tensor_shape(
                artifacts,
                &layer.gate_weight,
                &[intermediate_size, plan.hidden_size],
            )?;
            require_tensor_shape(
                artifacts,
                &layer.up_weight,
                &[intermediate_size, plan.hidden_size],
            )?;
            require_tensor_shape(
                artifacts,
                &layer.down_weight,
                &[plan.hidden_size, intermediate_size],
            )?;
        }
        Ok(())
    }

    pub(crate) fn validate_hybrid_decoder_checkpoint(
        &self,
        artifacts: &LocalModelArtifacts,
    ) -> Result<(), ModelArtifactError> {
        let Some(plan) = self.hybrid_decoder() else {
            return Err(invalid_dense_decoder_checkpoint(
                artifacts,
                "model definition does not include a hybrid decoder execution plan",
            ));
        };
        if plan.weights.layers.is_empty() {
            return Err(invalid_dense_decoder_checkpoint(
                artifacts,
                "hybrid decoder weight map has no layers",
            ));
        }
        artifacts.validate_checkpoint_topology(&self.checkpoint_topology)
    }

    pub fn runtime_requirements<'a>(
        &self,
        execution_dtype: RuntimeDtype,
        tensor_parallel_size: usize,
        requested_attention_backend: Option<&'a str>,
    ) -> RuntimeRequirements<'a> {
        RuntimeRequirements {
            requires_forward: true,
            dtype: Some(execution_dtype),
            attention_backend: requested_attention_backend,
            tensor_parallel_size,
            requires_kv_cache_registration: false,
            requires_mooncake: false,
        }
    }

    pub fn validate_tensor_parallel(&self, tensor_parallel_size: usize) -> Result<(), String> {
        if tensor_parallel_size == 0 {
            return Err("tensor parallel size must be non-zero".to_string());
        }
        self.execution
            .validate_tensor_parallel(tensor_parallel_size)
    }
}

pub(crate) fn dense_decoder_checkpoint_topology(
    architecture: &'static str,
    execution: ModelExecutionArchitecture,
    plan: &DenseDecoderExecutionPlan,
) -> Result<CheckpointTopology, ModelAdapterError> {
    let ModelExecutionArchitecture::Transformer {
        attention:
            AttentionArchitecture::MultiHead {
                num_attention_heads,
                num_key_value_heads,
                head_dim,
            },
        feed_forward: FeedForwardArchitecture::Dense { intermediate_size },
    } = execution
    else {
        return Err(ModelAdapterError::invalid(
            architecture,
            "dense decoder checkpoint topology requires multi-head attention and dense feed-forward components",
        ));
    };
    let query_size = num_attention_heads.checked_mul(head_dim).ok_or_else(|| {
        ModelAdapterError::invalid(architecture, "query projection size overflowed")
    })?;
    let kv_size = num_key_value_heads
        .checked_mul(head_dim)
        .ok_or_else(|| ModelAdapterError::invalid(architecture, "KV projection size overflowed"))?;
    if plan.weights.layers.is_empty() {
        return Err(ModelAdapterError::invalid(
            architecture,
            "dense decoder checkpoint topology has no layers",
        ));
    }

    let mut tensors = vec![
        CheckpointTensorRequirement::with_shape(
            &plan.weights.token_embeddings,
            [plan.vocab_size, plan.hidden_size],
        ),
        CheckpointTensorRequirement::with_shape(&plan.weights.final_norm, [plan.hidden_size]),
    ];
    if let Some(lm_head) = &plan.weights.lm_head {
        tensors.push(CheckpointTensorRequirement::with_shape(
            lm_head,
            [plan.vocab_size, plan.hidden_size],
        ));
    }
    for layer in &plan.weights.layers {
        tensors.extend([
            CheckpointTensorRequirement::with_shape(&layer.input_norm, [plan.hidden_size]),
            CheckpointTensorRequirement::with_shape(
                &layer.query_weight,
                [query_size, plan.hidden_size],
            ),
            CheckpointTensorRequirement::with_shape(&layer.key_weight, [kv_size, plan.hidden_size]),
            CheckpointTensorRequirement::with_shape(
                &layer.value_weight,
                [kv_size, plan.hidden_size],
            ),
            CheckpointTensorRequirement::with_shape(
                &layer.output_weight,
                [plan.hidden_size, query_size],
            ),
            CheckpointTensorRequirement::with_shape(&layer.post_attention_norm, [plan.hidden_size]),
            CheckpointTensorRequirement::with_shape(
                &layer.gate_weight,
                [intermediate_size, plan.hidden_size],
            ),
            CheckpointTensorRequirement::with_shape(
                &layer.up_weight,
                [intermediate_size, plan.hidden_size],
            ),
            CheckpointTensorRequirement::with_shape(
                &layer.down_weight,
                [plan.hidden_size, intermediate_size],
            ),
        ]);
        for (name, shape) in [
            (layer.query_bias.as_ref(), vec![query_size]),
            (layer.query_norm.as_ref(), vec![head_dim]),
            (layer.key_bias.as_ref(), vec![kv_size]),
            (layer.key_norm.as_ref(), vec![head_dim]),
            (layer.value_bias.as_ref(), vec![kv_size]),
            (layer.output_bias.as_ref(), vec![plan.hidden_size]),
        ] {
            if let Some(name) = name {
                tensors.push(CheckpointTensorRequirement::with_shape(name, shape));
            }
        }
    }
    Ok(CheckpointTopology::new(tensors))
}

pub(crate) fn hybrid_decoder_checkpoint_topology(
    architecture: &'static str,
    plan: &HybridDecoderExecutionPlan,
) -> Result<CheckpointTopology, ModelAdapterError> {
    if plan.weights.layers.is_empty() {
        return Err(ModelAdapterError::invalid(
            architecture,
            "hybrid decoder checkpoint topology has no layers",
        ));
    }
    let HybridFullAttentionConfig::MultiHead {
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        output_gate,
        ..
    } = plan.full_attention
    else {
        return Err(ModelAdapterError::invalid(
            architecture,
            "multi-head/GDN checkpoint topology cannot validate multi-latent attention",
        ));
    };
    let HybridLinearAttentionConfig::GatedDeltaNet {
        conv_kernel_dim,
        key_head_dim,
        value_head_dim,
        num_key_heads,
        num_value_heads,
    } = plan.linear_attention
    else {
        return Err(ModelAdapterError::invalid(
            architecture,
            "multi-head/GDN checkpoint topology cannot validate key-gated delta attention",
        ));
    };
    let query_size = num_attention_heads.checked_mul(head_dim).ok_or_else(|| {
        ModelAdapterError::invalid(architecture, "query projection size overflowed")
    })?;
    let projected_query_size = query_size
        .checked_mul(if output_gate { 2 } else { 1 })
        .ok_or_else(|| {
            ModelAdapterError::invalid(architecture, "gated query projection size overflowed")
        })?;
    let kv_size = num_key_value_heads
        .checked_mul(head_dim)
        .ok_or_else(|| ModelAdapterError::invalid(architecture, "KV projection size overflowed"))?;
    let linear_key_size = num_key_heads
        .checked_mul(key_head_dim)
        .ok_or_else(|| ModelAdapterError::invalid(architecture, "linear key size overflowed"))?;
    let linear_value_size = num_value_heads
        .checked_mul(value_head_dim)
        .ok_or_else(|| ModelAdapterError::invalid(architecture, "linear value size overflowed"))?;
    let conv_dim = linear_key_size
        .checked_mul(2)
        .and_then(|size| size.checked_add(linear_value_size))
        .ok_or_else(|| {
            ModelAdapterError::invalid(architecture, "linear convolution size overflowed")
        })?;

    let mut tensors = vec![
        CheckpointTensorRequirement::with_shape(
            &plan.weights.token_embeddings,
            [plan.vocab_size, plan.hidden_size],
        ),
        CheckpointTensorRequirement::with_shape(&plan.weights.final_norm, [plan.hidden_size]),
    ];
    if let Some(lm_head) = &plan.weights.lm_head {
        tensors.push(CheckpointTensorRequirement::with_shape(
            lm_head,
            [plan.vocab_size, plan.hidden_size],
        ));
    }
    for layer in &plan.weights.layers {
        tensors.extend([
            CheckpointTensorRequirement::with_shape(&layer.input_norm, [plan.hidden_size]),
            CheckpointTensorRequirement::with_shape(&layer.post_attention_norm, [plan.hidden_size]),
        ]);
        let HybridFeedForward::Dense {
            intermediate_size,
            weights: feed_forward,
        } = &layer.feed_forward
        else {
            return Err(ModelAdapterError::invalid(
                architecture,
                "multi-head/GDN checkpoint topology requires dense feed-forward layers",
            ));
        };
        tensors.extend([
            CheckpointTensorRequirement::with_shape(
                &feed_forward.gate_weight,
                [*intermediate_size, plan.hidden_size],
            ),
            CheckpointTensorRequirement::with_shape(
                &feed_forward.up_weight,
                [*intermediate_size, plan.hidden_size],
            ),
            CheckpointTensorRequirement::with_shape(
                &feed_forward.down_weight,
                [plan.hidden_size, *intermediate_size],
            ),
        ]);
        match &layer.mixer {
            HybridDecoderLayerKind::FullAttention { weights, .. } => tensors.extend([
                CheckpointTensorRequirement::with_shape(
                    &weights.query_weight,
                    [projected_query_size, plan.hidden_size],
                ),
                CheckpointTensorRequirement::with_shape(&weights.query_norm, [head_dim]),
                CheckpointTensorRequirement::with_shape(
                    &weights.key_weight,
                    [kv_size, plan.hidden_size],
                ),
                CheckpointTensorRequirement::with_shape(&weights.key_norm, [head_dim]),
                CheckpointTensorRequirement::with_shape(
                    &weights.value_weight,
                    [kv_size, plan.hidden_size],
                ),
                CheckpointTensorRequirement::with_shape(
                    &weights.output_weight,
                    [plan.hidden_size, query_size],
                ),
            ]),
            HybridDecoderLayerKind::GatedDeltaNet { weights, .. } => tensors.extend([
                CheckpointTensorRequirement::with_shape(&weights.a_log, [num_value_heads]),
                CheckpointTensorRequirement::with_shape(
                    &weights.conv1d_weight,
                    [conv_dim, 1, conv_kernel_dim],
                ),
                CheckpointTensorRequirement::with_shape(&weights.dt_bias, [num_value_heads]),
                CheckpointTensorRequirement::with_shape(
                    &weights.in_proj_a_weight,
                    [num_value_heads, plan.hidden_size],
                ),
                CheckpointTensorRequirement::with_shape(
                    &weights.in_proj_b_weight,
                    [num_value_heads, plan.hidden_size],
                ),
                CheckpointTensorRequirement::with_shape(
                    &weights.in_proj_qkv_weight,
                    [conv_dim, plan.hidden_size],
                ),
                CheckpointTensorRequirement::with_shape(
                    &weights.in_proj_z_weight,
                    [linear_value_size, plan.hidden_size],
                ),
                CheckpointTensorRequirement::with_shape(&weights.norm_weight, [value_head_dim]),
                CheckpointTensorRequirement::with_shape(
                    &weights.output_weight,
                    [plan.hidden_size, linear_value_size],
                ),
            ]),
            HybridDecoderLayerKind::MultiLatentAttention { .. }
            | HybridDecoderLayerKind::KeyGatedDelta { .. } => {
                return Err(ModelAdapterError::invalid(
                    architecture,
                    "multi-head/GDN checkpoint topology received an incompatible mixer",
                ));
            }
        }
    }
    Ok(CheckpointTopology::new(tensors))
}

fn require_optional_tensor_shape(
    artifacts: &LocalModelArtifacts,
    tensor_name: Option<&str>,
    expected_shape: &[usize],
) -> Result<(), ModelArtifactError> {
    match tensor_name {
        Some(tensor_name) => require_tensor_shape(artifacts, tensor_name, expected_shape),
        None => Ok(()),
    }
}

fn require_tensor_shape(
    artifacts: &LocalModelArtifacts,
    tensor_name: &str,
    expected_shape: &[usize],
) -> Result<(), ModelArtifactError> {
    let metadata = artifacts
        .safetensors()
        .tensor_metadata(tensor_name)?
        .ok_or_else(|| {
            invalid_dense_decoder_checkpoint(
                artifacts,
                format!("missing dense decoder checkpoint tensor {tensor_name}"),
            )
        })?;
    if metadata.shape != expected_shape {
        return Err(invalid_dense_decoder_checkpoint(
            artifacts,
            format!(
                "dense decoder tensor {tensor_name} shape {:?} does not match expected {expected_shape:?}",
                metadata.shape
            ),
        ));
    }
    Ok(())
}

fn invalid_dense_decoder_checkpoint(
    artifacts: &LocalModelArtifacts,
    message: impl Into<String>,
) -> ModelArtifactError {
    ModelArtifactError::InvalidSafetensorsData {
        path: artifacts.model_path().to_path_buf(),
        message: message.into(),
    }
}

pub(crate) trait ModelAdapter: Sync {
    fn architectures(&self) -> &'static [&'static str];

    fn build_definition(
        &self,
        model_path: &Path,
        config: &HfModelConfig,
    ) -> Result<ModelDefinition, ModelAdapterError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ModelAdapterError {
    message: String,
}

impl ModelAdapterError {
    pub(crate) fn missing_field(architecture: &str, field: &str) -> Self {
        Self {
            message: format!("{architecture} requires config field {field}"),
        }
    }

    pub(crate) fn invalid(architecture: &str, message: impl Into<String>) -> Self {
        Self {
            message: format!("{architecture}: {}", message.into()),
        }
    }
}

impl fmt::Display for ModelAdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

pub(crate) fn required_usize(
    architecture: &str,
    field: &str,
    value: Option<usize>,
) -> Result<usize, ModelAdapterError> {
    let value = value.ok_or_else(|| ModelAdapterError::missing_field(architecture, field))?;
    if value == 0 {
        return Err(ModelAdapterError::invalid(
            architecture,
            format!("config field {field} must be non-zero"),
        ));
    }
    Ok(value)
}
