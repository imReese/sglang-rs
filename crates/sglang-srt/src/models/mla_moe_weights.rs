use crate::model_artifacts::{CheckpointTensorRequirement, CheckpointTopology};

use super::deepseek::MlaMoeConfig;
use super::{ModelAdapterError, required_usize};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MlaMoeCheckpointFlavor {
    DeepSeek,
    GlmDsa,
}

pub(super) fn checkpoint_topology(
    architecture: &'static str,
    config: &MlaMoeConfig,
    flavor: MlaMoeCheckpointFlavor,
) -> Result<CheckpointTopology, ModelAdapterError> {
    let num_layers = required_usize(architecture, "num_hidden_layers", config.num_hidden_layers)?;
    let routed_expert_count =
        required_usize(architecture, "n_routed_experts", config.n_routed_experts)?;
    if config.moe_layer_freq == Some(0) {
        return Err(ModelAdapterError::invalid(
            architecture,
            "moe_layer_freq must be non-zero",
        ));
    }

    let mut tensors = vec![
        CheckpointTensorRequirement::present("model.embed_tokens.weight"),
        CheckpointTensorRequirement::present("model.norm.weight"),
        CheckpointTensorRequirement::present("lm_head.weight"),
    ];
    if flavor == MlaMoeCheckpointFlavor::DeepSeek {
        let hidden_size = required_usize(architecture, "hidden_size", config.hidden_size)?;
        let hc_mult = required_usize(architecture, "hc_mult", config.hc_mult)?;
        let hc_dim = hc_mult.checked_mul(hidden_size).ok_or_else(|| {
            ModelAdapterError::invalid(architecture, "HC head dimension overflowed")
        })?;
        tensors.extend([
            CheckpointTensorRequirement::with_shape("model.hc_head_fn", [hc_mult, hc_dim]),
            CheckpointTensorRequirement::with_shape("model.hc_head_base", [hc_mult]),
            CheckpointTensorRequirement::with_shape("model.hc_head_scale", [1]),
        ]);
    }

    let mut routed_coordinates = Vec::new();
    for layer_id in 0..num_layers {
        let prefix = format!("model.layers.{layer_id}");
        let attention_suffixes: &[&str] = match flavor {
            MlaMoeCheckpointFlavor::DeepSeek => &[
                "self_attn.wq_a.weight",
                "self_attn.wq_b.weight",
                "self_attn.wkv.weight",
                "self_attn.q_norm.weight",
                "self_attn.kv_norm.weight",
                "self_attn.wo_a.weight",
                "self_attn.wo_b.weight",
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "hc_attn_fn",
                "hc_attn_base",
                "hc_attn_scale",
                "hc_ffn_fn",
                "hc_ffn_base",
                "hc_ffn_scale",
            ],
            MlaMoeCheckpointFlavor::GlmDsa => &[
                "self_attn.q_a_proj.weight",
                "self_attn.q_a_layernorm.weight",
                "self_attn.q_b_proj.weight",
                "self_attn.kv_a_proj_with_mqa.weight",
                "self_attn.kv_a_layernorm.weight",
                "self_attn.kv_b_proj.weight",
                "self_attn.o_proj.weight",
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
            ],
        };
        tensors.extend(
            attention_suffixes
                .iter()
                .map(|suffix| CheckpointTensorRequirement::present(format!("{prefix}.{suffix}"))),
        );

        if config.is_moe_layer(layer_id, num_layers) {
            tensors.push(CheckpointTensorRequirement::present(format!(
                "{prefix}.mlp.gate.weight"
            )));
            routed_coordinates
                .extend((0..routed_expert_count).map(|expert_id| (layer_id, expert_id)));
        } else {
            let dense_suffixes: &[&str] = match flavor {
                MlaMoeCheckpointFlavor::DeepSeek => {
                    &["mlp.gate_up_proj.weight", "mlp.down_proj.weight"]
                }
                MlaMoeCheckpointFlavor::GlmDsa => &[
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp.down_proj.weight",
                ],
            };
            tensors.extend(
                dense_suffixes.iter().map(|suffix| {
                    CheckpointTensorRequirement::present(format!("{prefix}.{suffix}"))
                }),
            );
        }
    }

    Ok(CheckpointTopology::new(tensors).with_routed_experts(routed_coordinates, 3))
}
