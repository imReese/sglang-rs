mod deepseek;
mod embedding;
mod glm;
mod mla_moe_weights;
mod qwen;
mod qwen3_5;

use std::fmt;
use std::path::Path;

use crate::backend::{RuntimeDtype, RuntimeRequirements};
use crate::kv_cache::KvCacheModelLayout;
use crate::model_artifacts::{HfModelConfig, LocalModelArtifacts, ModelArtifactError};

pub(crate) use deepseek::DEEPSEEK_V4_ADAPTER;
pub(crate) use embedding::EMBEDDING_LM_ADAPTER;
pub(crate) use glm::GLM_MOE_DSA_ADAPTER;
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
}

impl AttentionArchitecture {
    pub fn family(self) -> AttentionFamily {
        match self {
            Self::None => AttentionFamily::None,
            Self::MultiHead { .. } => AttentionFamily::MultiHead,
            Self::MultiLatent { .. } => AttentionFamily::MultiLatent,
            Self::Hybrid { .. } => AttentionFamily::Hybrid,
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
pub(crate) enum HybridDecoderLayerKind {
    FullAttention {
        cache_layer_index: usize,
        weights: HybridFullAttentionWeightNames,
    },
    GatedDeltaNet {
        state_layer_index: usize,
        weights: GatedDeltaNetWeightNames,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct HybridDecoderLayerWeightNames {
    pub(crate) input_norm: String,
    pub(crate) mixer: HybridDecoderLayerKind,
    pub(crate) post_attention_norm: String,
    pub(crate) feed_forward: DenseFeedForwardWeightNames,
}

#[derive(Clone, Debug, Eq, PartialEq)]
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
    pub(crate) intermediate_size: usize,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_theta: f32,
    pub(crate) rotary_dim: usize,
    pub(crate) attention_output_gate: bool,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) attention_head_dim: usize,
    pub(crate) linear_conv_kernel_dim: usize,
    pub(crate) linear_key_head_dim: usize,
    pub(crate) linear_value_head_dim: usize,
    pub(crate) linear_num_key_heads: usize,
    pub(crate) linear_num_value_heads: usize,
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
    Embedding,
    Transformer {
        attention: AttentionArchitecture,
        feed_forward: FeedForwardArchitecture,
    },
}

impl ModelExecutionArchitecture {
    pub fn attention_family(self) -> AttentionFamily {
        match self {
            Self::Embedding => AttentionFamily::None,
            Self::Transformer { attention, .. } => attention.family(),
        }
    }

    pub fn feed_forward_family(self) -> FeedForwardFamily {
        match self {
            Self::Embedding => FeedForwardFamily::None,
            Self::Transformer { feed_forward, .. } => feed_forward.family(),
        }
    }

    fn validate_tensor_parallel(self, tensor_parallel_size: usize) -> Result<(), String> {
        match self {
            Self::Embedding if tensor_parallel_size != 1 => Err(
                "embedding reference execution supports tensor parallel size 1 only".to_string(),
            ),
            Self::Embedding => Ok(()),
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
    execution: ModelExecutionArchitecture,
    supported_dtypes: Vec<RuntimeDtype>,
    kv_cache_layout: Option<KvCacheModelLayout>,
    cache_architecture: ModelCacheArchitecture,
    dense_decoder: Option<DenseDecoderExecutionPlan>,
    hybrid_decoder: Option<HybridDecoderExecutionPlan>,
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
        }
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
            .filter(|layer| matches!(layer.mixer, HybridDecoderLayerKind::FullAttention { .. }))
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

        let query_size = plan
            .num_attention_heads
            .checked_mul(plan.attention_head_dim)
            .ok_or_else(|| {
                invalid_dense_decoder_checkpoint(artifacts, "query projection size overflowed")
            })?;
        let projected_query_size = query_size
            .checked_mul(if plan.attention_output_gate { 2 } else { 1 })
            .ok_or_else(|| {
                invalid_dense_decoder_checkpoint(
                    artifacts,
                    "gated query projection size overflowed",
                )
            })?;
        let kv_size = plan
            .num_key_value_heads
            .checked_mul(plan.attention_head_dim)
            .ok_or_else(|| {
                invalid_dense_decoder_checkpoint(artifacts, "KV projection size overflowed")
            })?;
        let linear_key_size = plan
            .linear_num_key_heads
            .checked_mul(plan.linear_key_head_dim)
            .ok_or_else(|| {
                invalid_dense_decoder_checkpoint(artifacts, "linear key size overflowed")
            })?;
        let linear_value_size = plan
            .linear_num_value_heads
            .checked_mul(plan.linear_value_head_dim)
            .ok_or_else(|| {
                invalid_dense_decoder_checkpoint(artifacts, "linear value size overflowed")
            })?;
        let conv_dim = linear_key_size
            .checked_mul(2)
            .and_then(|size| size.checked_add(linear_value_size))
            .ok_or_else(|| {
                invalid_dense_decoder_checkpoint(artifacts, "linear convolution size overflowed")
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

        for layer in &plan.weights.layers {
            require_tensor_shape(artifacts, &layer.input_norm, &[plan.hidden_size])?;
            require_tensor_shape(artifacts, &layer.post_attention_norm, &[plan.hidden_size])?;
            require_tensor_shape(
                artifacts,
                &layer.feed_forward.gate_weight,
                &[plan.intermediate_size, plan.hidden_size],
            )?;
            require_tensor_shape(
                artifacts,
                &layer.feed_forward.up_weight,
                &[plan.intermediate_size, plan.hidden_size],
            )?;
            require_tensor_shape(
                artifacts,
                &layer.feed_forward.down_weight,
                &[plan.hidden_size, plan.intermediate_size],
            )?;

            match &layer.mixer {
                HybridDecoderLayerKind::FullAttention { weights, .. } => {
                    require_tensor_shape(
                        artifacts,
                        &weights.query_weight,
                        &[projected_query_size, plan.hidden_size],
                    )?;
                    require_tensor_shape(
                        artifacts,
                        &weights.query_norm,
                        &[plan.attention_head_dim],
                    )?;
                    require_tensor_shape(
                        artifacts,
                        &weights.key_weight,
                        &[kv_size, plan.hidden_size],
                    )?;
                    require_tensor_shape(artifacts, &weights.key_norm, &[plan.attention_head_dim])?;
                    require_tensor_shape(
                        artifacts,
                        &weights.value_weight,
                        &[kv_size, plan.hidden_size],
                    )?;
                    require_tensor_shape(
                        artifacts,
                        &weights.output_weight,
                        &[plan.hidden_size, query_size],
                    )?;
                }
                HybridDecoderLayerKind::GatedDeltaNet { weights, .. } => {
                    require_tensor_shape(
                        artifacts,
                        &weights.a_log,
                        &[plan.linear_num_value_heads],
                    )?;
                    require_tensor_shape(
                        artifacts,
                        &weights.conv1d_weight,
                        &[conv_dim, 1, plan.linear_conv_kernel_dim],
                    )?;
                    require_tensor_shape(
                        artifacts,
                        &weights.dt_bias,
                        &[plan.linear_num_value_heads],
                    )?;
                    require_tensor_shape(
                        artifacts,
                        &weights.in_proj_a_weight,
                        &[plan.linear_num_value_heads, plan.hidden_size],
                    )?;
                    require_tensor_shape(
                        artifacts,
                        &weights.in_proj_b_weight,
                        &[plan.linear_num_value_heads, plan.hidden_size],
                    )?;
                    require_tensor_shape(
                        artifacts,
                        &weights.in_proj_qkv_weight,
                        &[conv_dim, plan.hidden_size],
                    )?;
                    require_tensor_shape(
                        artifacts,
                        &weights.in_proj_z_weight,
                        &[linear_value_size, plan.hidden_size],
                    )?;
                    require_tensor_shape(
                        artifacts,
                        &weights.norm_weight,
                        &[plan.linear_value_head_dim],
                    )?;
                    require_tensor_shape(
                        artifacts,
                        &weights.output_weight,
                        &[plan.hidden_size, linear_value_size],
                    )?;
                }
            }
        }
        Ok(())
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

    fn validate_checkpoint(
        &self,
        artifacts: &LocalModelArtifacts,
    ) -> Result<(), ModelArtifactError>;
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
