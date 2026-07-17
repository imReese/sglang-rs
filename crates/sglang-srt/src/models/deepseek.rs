use std::path::Path;

use serde::Deserialize;

use crate::backend::RuntimeDtype;
use crate::kv_cache::KvCacheModelLayout;
use crate::model_artifacts::HfModelConfig;

use super::mla_moe_weights::{MlaMoeCheckpointFlavor, checkpoint_topology};
use super::{
    AttentionArchitecture, FeedForwardArchitecture, ModelAdapter, ModelAdapterError,
    ModelDefinition, ModelExecutionArchitecture, required_usize,
};

pub(crate) const DEEPSEEK_V4_ARCHITECTURE: &str = "DeepseekV4ForCausalLM";
pub(crate) static DEEPSEEK_V4_ADAPTER: DeepSeekV4Adapter = DeepSeekV4Adapter;

pub(crate) struct DeepSeekV4Adapter;

#[derive(Clone, Debug, Default, Deserialize)]
pub(super) struct MlaMoeConfig {
    pub(super) vocab_size: Option<usize>,
    pub(super) max_position_embeddings: Option<usize>,
    pub(super) num_hidden_layers: Option<usize>,
    pub(super) hidden_size: Option<usize>,
    pub(super) num_attention_heads: Option<usize>,
    pub(super) qk_nope_head_dim: Option<usize>,
    pub(super) qk_rope_head_dim: Option<usize>,
    pub(super) v_head_dim: Option<usize>,
    pub(super) n_routed_experts: Option<usize>,
    pub(super) n_shared_experts: Option<usize>,
    pub(super) num_experts_per_tok: Option<usize>,
    pub(super) moe_intermediate_size: Option<usize>,
    pub(super) first_k_dense_replace: Option<usize>,
    pub(super) moe_layer_freq: Option<usize>,
    pub(super) hc_mult: Option<usize>,
}

impl MlaMoeConfig {
    pub(super) fn is_moe_layer(&self, layer_id: usize, num_layers: usize) -> bool {
        if layer_id >= num_layers || self.n_routed_experts.is_none() {
            return false;
        }
        let first_dense = self.first_k_dense_replace.unwrap_or(0);
        let frequency = self.moe_layer_freq.unwrap_or(1);
        frequency > 0 && layer_id >= first_dense && layer_id.is_multiple_of(frequency)
    }
}

impl ModelAdapter for DeepSeekV4Adapter {
    fn architectures(&self) -> &'static [&'static str] {
        &[DEEPSEEK_V4_ARCHITECTURE]
    }

    fn build_definition(
        &self,
        _model_path: &Path,
        config: &HfModelConfig,
    ) -> Result<ModelDefinition, ModelAdapterError> {
        build_mla_moe_definition(
            DEEPSEEK_V4_ARCHITECTURE,
            config,
            MlaMoeCheckpointFlavor::DeepSeek,
        )
    }
}

pub(super) fn build_mla_moe_definition(
    architecture: &'static str,
    hf_config: &HfModelConfig,
    checkpoint_flavor: MlaMoeCheckpointFlavor,
) -> Result<ModelDefinition, ModelAdapterError> {
    let config: MlaMoeConfig = hf_config.parse_text_config().map_err(|error| {
        ModelAdapterError::invalid(
            architecture,
            format!("invalid shared MLA/MoE config document: {error}"),
        )
    })?;
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
    let vocab_size = required_usize(architecture, "vocab_size", config.vocab_size)?;
    let max_context_length = config.max_position_embeddings.unwrap_or(32_768);
    if max_context_length == 0 {
        return Err(ModelAdapterError::invalid(
            architecture,
            "max_position_embeddings must be non-zero",
        ));
    }
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
    let checkpoint_topology = checkpoint_topology(architecture, &config, checkpoint_flavor)?;

    Ok(ModelDefinition::new(
        architecture,
        hf_config,
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
    )
    .with_serving_metadata(vocab_size, max_context_length)
    .with_checkpoint_topology(checkpoint_topology))
}
