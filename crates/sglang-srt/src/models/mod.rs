mod deepseek;
mod embedding;
mod glm;
mod qwen;

use std::fmt;
use std::path::Path;

use crate::backend::{RuntimeDtype, RuntimeRequirements};
use crate::model_artifacts::{HfModelConfig, LocalModelArtifacts, ModelArtifactError};
use crate::transfer::KvCacheModelLayout;

pub(crate) use deepseek::DEEPSEEK_V4_ADAPTER;
pub(crate) use embedding::EMBEDDING_LM_ADAPTER;
pub(crate) use glm::GLM_MOE_DSA_ADAPTER;
pub(crate) use qwen::QWEN2_ADAPTER;

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
}

impl AttentionArchitecture {
    pub fn family(self) -> AttentionFamily {
        match self {
            Self::None => AttentionFamily::None,
            Self::MultiHead { .. } => AttentionFamily::MultiHead,
            Self::MultiLatent { .. } => AttentionFamily::MultiLatent,
        }
    }

    fn validate_tensor_parallel(self, tensor_parallel_size: usize) -> Result<(), String> {
        let (num_attention_heads, num_key_value_heads) = match self {
            Self::None => return Ok(()),
            Self::MultiHead {
                num_attention_heads,
                num_key_value_heads,
                ..
            } => (num_attention_heads, Some(num_key_value_heads)),
            Self::MultiLatent {
                num_attention_heads,
                ..
            } => (num_attention_heads, None),
        };

        if !num_attention_heads.is_multiple_of(tensor_parallel_size) {
            return Err(format!(
                "attention head count {num_attention_heads} must be divisible by tensor parallel size {tensor_parallel_size}"
            ));
        }

        if let Some(num_key_value_heads) = num_key_value_heads {
            let valid = if num_key_value_heads >= tensor_parallel_size {
                num_key_value_heads.is_multiple_of(tensor_parallel_size)
            } else {
                tensor_parallel_size.is_multiple_of(num_key_value_heads)
            };
            if !valid {
                return Err(format!(
                    "KV head count {num_key_value_heads} must shard across or replicate evenly over tensor parallel size {tensor_parallel_size}"
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
}

impl fmt::Display for AttentionFamily {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::None => "none",
            Self::MultiHead => "multi-head attention",
            Self::MultiLatent => "multi-latent attention",
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelDefinition {
    architecture: &'static str,
    model_type: Option<String>,
    execution: ModelExecutionArchitecture,
    dtype: RuntimeDtype,
    kv_cache_layout: Option<KvCacheModelLayout>,
}

impl ModelDefinition {
    pub(crate) fn new(
        architecture: &'static str,
        config: &HfModelConfig,
        execution: ModelExecutionArchitecture,
        dtype: RuntimeDtype,
        kv_cache_layout: Option<KvCacheModelLayout>,
    ) -> Self {
        Self {
            architecture,
            model_type: config.model_type.clone(),
            execution,
            dtype,
            kv_cache_layout,
        }
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

    pub fn dtype(&self) -> RuntimeDtype {
        self.dtype
    }

    pub fn kv_cache_layout(&self) -> Option<KvCacheModelLayout> {
        self.kv_cache_layout
    }

    pub fn runtime_requirements<'a>(
        &self,
        tensor_parallel_size: usize,
        requested_attention_backend: Option<&'a str>,
    ) -> RuntimeRequirements<'a> {
        RuntimeRequirements {
            requires_forward: true,
            dtype: Some(self.dtype),
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
