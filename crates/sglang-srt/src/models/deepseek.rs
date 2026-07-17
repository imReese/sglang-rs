use std::path::Path;

use crate::backend::RuntimeDtype;
use crate::kv_cache::KvCacheModelLayout;
use crate::model_artifacts::{HfModelConfig, LocalModelArtifacts, ModelArtifactError};

use super::mla_moe_weights::validate_deepseek_checkpoint;
use super::{
    AttentionArchitecture, FeedForwardArchitecture, ModelAdapter, ModelAdapterError,
    ModelDefinition, ModelExecutionArchitecture, required_usize,
};

pub(crate) const DEEPSEEK_V4_ARCHITECTURE: &str = "DeepseekV4ForCausalLM";
pub(crate) static DEEPSEEK_V4_ADAPTER: DeepSeekV4Adapter = DeepSeekV4Adapter;

pub(crate) struct DeepSeekV4Adapter;

impl ModelAdapter for DeepSeekV4Adapter {
    fn architectures(&self) -> &'static [&'static str] {
        &[DEEPSEEK_V4_ARCHITECTURE]
    }

    fn build_definition(
        &self,
        _model_path: &Path,
        config: &HfModelConfig,
    ) -> Result<ModelDefinition, ModelAdapterError> {
        build_mla_moe_definition(DEEPSEEK_V4_ARCHITECTURE, config)
    }

    fn validate_checkpoint(
        &self,
        artifacts: &LocalModelArtifacts,
    ) -> Result<(), ModelArtifactError> {
        validate_deepseek_checkpoint(artifacts)
    }
}

pub(super) fn build_mla_moe_definition(
    architecture: &'static str,
    config: &HfModelConfig,
) -> Result<ModelDefinition, ModelAdapterError> {
    let num_layers = required_usize(architecture, "num_hidden_layers", config.num_hidden_layers)?;
    let num_attention_heads = required_usize(
        architecture,
        "num_attention_heads",
        config.num_attention_heads,
    )?;
    let qk_nope_head_dim =
        required_usize(architecture, "qk_nope_head_dim", config.qk_nope_head_dim)?;
    let qk_rope_head_dim =
        required_usize(architecture, "qk_rope_head_dim", config.qk_rope_head_dim)?;
    let value_head_dim = required_usize(architecture, "v_head_dim", config.v_head_dim)?;
    let routed_expert_count =
        required_usize(architecture, "n_routed_experts", config.n_routed_experts)?;
    let experts_per_token = required_usize(
        architecture,
        "num_experts_per_tok",
        config.num_experts_per_tok,
    )?;
    let expert_intermediate_size = required_usize(
        architecture,
        "moe_intermediate_size",
        config.moe_intermediate_size,
    )?;
    if experts_per_token > routed_expert_count {
        return Err(ModelAdapterError::invalid(
            architecture,
            format!(
                "num_experts_per_tok ({experts_per_token}) exceeds n_routed_experts ({routed_expert_count})"
            ),
        ));
    }

    let kv_cache_layout =
        KvCacheModelLayout::packed_mla(num_layers, qk_nope_head_dim, qk_rope_head_dim)
            .map_err(|error| ModelAdapterError::invalid(architecture, error.to_string()))?;

    Ok(ModelDefinition::new(
        architecture,
        config,
        ModelExecutionArchitecture::Transformer {
            attention: AttentionArchitecture::MultiLatent {
                num_attention_heads,
                qk_nope_head_dim,
                qk_rope_head_dim,
                value_head_dim,
            },
            feed_forward: FeedForwardArchitecture::MixtureOfExperts {
                routed_expert_count,
                experts_per_token,
                shared_expert_count: config.n_shared_experts.unwrap_or(0),
                expert_intermediate_size,
            },
        },
        vec![RuntimeDtype::Bf16],
        Some(kv_cache_layout),
    ))
}
