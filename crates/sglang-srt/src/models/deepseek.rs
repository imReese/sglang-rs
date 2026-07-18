use std::path::Path;

use serde::Deserialize;

use crate::backend::RuntimeDtype;
use crate::kv_cache::KvCacheModelLayout;
use crate::model_artifacts::{CheckpointTensorRequirement, CheckpointTopology, HfModelConfig};

use super::mla_moe_weights::{
    MlaMoeCheckpointFlavor, checkpoint_topology as mla_moe_metadata_topology,
};
use super::{
    AttentionArchitecture, DecoderNormalization, DenseDecoderActivation,
    DenseFeedForwardWeightNames, FeedForwardArchitecture, HybridDecoderExecutionPlan,
    HybridDecoderLayerKind, HybridDecoderLayerWeightNames, HybridDecoderWeightNames,
    HybridFeedForward, HybridFullAttentionConfig, HybridMultiLatentAttentionWeightNames,
    ModelAdapter, ModelAdapterError, ModelDefinition, ModelExecutionArchitecture,
    MoeFeedForwardConfig, MoeFeedForwardWeightNames, MultiLatentQueryConfig,
    MultiLatentQueryWeightNames, RouterActivation, required_usize,
};

pub(crate) const DEEPSEEK_V3_ARCHITECTURE: &str = "DeepseekV3ForCausalLM";
pub(crate) const DEEPSEEK_V4_ARCHITECTURE: &str = "DeepseekV4ForCausalLM";
pub(crate) static DEEPSEEK_V3_ADAPTER: DeepSeekV3Adapter = DeepSeekV3Adapter;
pub(crate) static DEEPSEEK_V4_ADAPTER: DeepSeekV4Adapter = DeepSeekV4Adapter;

pub(crate) struct DeepSeekV3Adapter;
pub(crate) struct DeepSeekV4Adapter;

#[derive(Clone, Debug, Default, Deserialize)]
pub(super) struct MlaMoeConfig {
    pub(super) vocab_size: Option<usize>,
    pub(super) max_position_embeddings: Option<usize>,
    pub(super) num_hidden_layers: Option<usize>,
    pub(super) hidden_size: Option<usize>,
    pub(super) intermediate_size: Option<usize>,
    pub(super) num_attention_heads: Option<usize>,
    pub(super) q_lora_rank: Option<usize>,
    pub(super) kv_lora_rank: Option<usize>,
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
    pub(super) n_group: Option<usize>,
    pub(super) topk_group: Option<usize>,
    pub(super) norm_topk_prob: Option<bool>,
    pub(super) routed_scaling_factor: Option<f64>,
    pub(super) scoring_func: Option<String>,
    pub(super) topk_method: Option<String>,
    pub(super) rms_norm_eps: Option<f64>,
    pub(super) rope_theta: Option<f64>,
    pub(super) rope_scaling: Option<serde_json::Value>,
    pub(super) hidden_act: Option<String>,
    pub(super) tie_word_embeddings: Option<bool>,
    pub(super) attention_bias: Option<bool>,
    pub(super) num_nextn_predict_layers: Option<usize>,
    pub(super) quantization_config: Option<serde_json::Value>,
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

impl ModelAdapter for DeepSeekV3Adapter {
    fn architectures(&self) -> &'static [&'static str] {
        &[DEEPSEEK_V3_ARCHITECTURE]
    }

    fn build_definition(
        &self,
        _model_path: &Path,
        config: &HfModelConfig,
    ) -> Result<ModelDefinition, ModelAdapterError> {
        build_deepseek_v3_definition(config)
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

fn build_deepseek_v3_definition(
    hf_config: &HfModelConfig,
) -> Result<ModelDefinition, ModelAdapterError> {
    let config: MlaMoeConfig = hf_config.parse_text_config().map_err(|error| {
        ModelAdapterError::invalid(
            DEEPSEEK_V3_ARCHITECTURE,
            format!("invalid DeepSeek V3 MLA/MoE config document: {error}"),
        )
    })?;
    reject_unsupported_deepseek_v3_features(&config)?;

    let vocab_size = required_usize(DEEPSEEK_V3_ARCHITECTURE, "vocab_size", config.vocab_size)?;
    let hidden_size = required_usize(DEEPSEEK_V3_ARCHITECTURE, "hidden_size", config.hidden_size)?;
    let dense_intermediate_size = required_usize(
        DEEPSEEK_V3_ARCHITECTURE,
        "intermediate_size",
        config.intermediate_size,
    )?;
    let num_layers = required_usize(
        DEEPSEEK_V3_ARCHITECTURE,
        "num_hidden_layers",
        config.num_hidden_layers,
    )?;
    let num_attention_heads = required_usize(
        DEEPSEEK_V3_ARCHITECTURE,
        "num_attention_heads",
        config.num_attention_heads,
    )?;
    let query = match config.q_lora_rank {
        Some(0) => {
            return Err(ModelAdapterError::invalid(
                DEEPSEEK_V3_ARCHITECTURE,
                "q_lora_rank must be non-zero when present",
            ));
        }
        Some(rank) => MultiLatentQueryConfig::LowRank { rank },
        None => MultiLatentQueryConfig::Direct,
    };
    let kv_lora_rank = required_usize(
        DEEPSEEK_V3_ARCHITECTURE,
        "kv_lora_rank",
        config.kv_lora_rank,
    )?;
    let qk_nope_head_dim = required_usize(
        DEEPSEEK_V3_ARCHITECTURE,
        "qk_nope_head_dim",
        config.qk_nope_head_dim,
    )?;
    let qk_rope_head_dim = required_usize(
        DEEPSEEK_V3_ARCHITECTURE,
        "qk_rope_head_dim",
        config.qk_rope_head_dim,
    )?;
    let value_head_dim = required_usize(DEEPSEEK_V3_ARCHITECTURE, "v_head_dim", config.v_head_dim)?;
    let routed_expert_count = required_usize(
        DEEPSEEK_V3_ARCHITECTURE,
        "n_routed_experts",
        config.n_routed_experts,
    )?;
    let experts_per_token = required_usize(
        DEEPSEEK_V3_ARCHITECTURE,
        "num_experts_per_tok",
        config.num_experts_per_tok,
    )?;
    let expert_intermediate_size = required_usize(
        DEEPSEEK_V3_ARCHITECTURE,
        "moe_intermediate_size",
        config.moe_intermediate_size,
    )?;
    let first_dense_layer_count = config.first_k_dense_replace.unwrap_or(0);
    if first_dense_layer_count > num_layers {
        return Err(ModelAdapterError::invalid(
            DEEPSEEK_V3_ARCHITECTURE,
            "first_k_dense_replace exceeds num_hidden_layers",
        ));
    }
    let moe_layer_frequency = required_usize(
        DEEPSEEK_V3_ARCHITECTURE,
        "moe_layer_freq",
        config.moe_layer_freq,
    )?;
    let expert_group_count = required_usize(DEEPSEEK_V3_ARCHITECTURE, "n_group", config.n_group)?;
    let selected_expert_group_count =
        required_usize(DEEPSEEK_V3_ARCHITECTURE, "topk_group", config.topk_group)?;
    if experts_per_token > routed_expert_count {
        return Err(ModelAdapterError::invalid(
            DEEPSEEK_V3_ARCHITECTURE,
            format!(
                "num_experts_per_tok ({experts_per_token}) exceeds n_routed_experts ({routed_expert_count})"
            ),
        ));
    }
    if !routed_expert_count.is_multiple_of(expert_group_count)
        || selected_expert_group_count > expert_group_count
    {
        return Err(ModelAdapterError::invalid(
            DEEPSEEK_V3_ARCHITECTURE,
            "n_routed_experts must divide evenly into n_group and topk_group must not exceed n_group",
        ));
    }
    let router_activation = match config.scoring_func.as_deref() {
        Some("sigmoid") => RouterActivation::Sigmoid,
        Some("softmax") => RouterActivation::Softmax,
        Some(scoring) => {
            return Err(ModelAdapterError::invalid(
                DEEPSEEK_V3_ARCHITECTURE,
                format!("unsupported scoring_func {scoring}"),
            ));
        }
        None => {
            return Err(ModelAdapterError::missing_field(
                DEEPSEEK_V3_ARCHITECTURE,
                "scoring_func",
            ));
        }
    };
    let has_correction_bias = match config.topk_method.as_deref() {
        Some("noaux_tc") => true,
        Some("greedy" | "group_limited_greedy") => false,
        Some(method) => {
            return Err(ModelAdapterError::invalid(
                DEEPSEEK_V3_ARCHITECTURE,
                format!("unsupported topk_method {method}"),
            ));
        }
        None => {
            return Err(ModelAdapterError::missing_field(
                DEEPSEEK_V3_ARCHITECTURE,
                "topk_method",
            ));
        }
    };
    let renormalize = config.norm_topk_prob.ok_or_else(|| {
        ModelAdapterError::missing_field(DEEPSEEK_V3_ARCHITECTURE, "norm_topk_prob")
    })?;
    let routed_scaling_factor = positive_float(
        DEEPSEEK_V3_ARCHITECTURE,
        "routed_scaling_factor",
        config.routed_scaling_factor,
    )?;
    let rms_norm_eps = positive_float(
        DEEPSEEK_V3_ARCHITECTURE,
        "rms_norm_eps",
        config.rms_norm_eps,
    )?;
    let rope_theta = positive_float(DEEPSEEK_V3_ARCHITECTURE, "rope_theta", config.rope_theta)?;
    let max_position_embeddings = required_usize(
        DEEPSEEK_V3_ARCHITECTURE,
        "max_position_embeddings",
        config.max_position_embeddings,
    )?;
    let shared_expert_count = config.n_shared_experts.unwrap_or(0);
    let moe_config = MoeFeedForwardConfig {
        routed_expert_count,
        experts_per_token,
        expert_group_count,
        selected_expert_group_count,
        shared_expert_count,
        expert_intermediate_size,
        renormalize,
        router_activation,
        routed_scaling_factor,
    };
    let weights = deepseek_v3_weight_names(DeepSeekV3WeightLayout {
        num_layers,
        first_dense_layer_count,
        moe_layer_frequency,
        dense_intermediate_size,
        moe_config: &moe_config,
        query,
        has_correction_bias,
        tied_embeddings: config.tie_word_embeddings.unwrap_or(false),
    });
    let plan = HybridDecoderExecutionPlan {
        vocab_size,
        hidden_size,
        max_position_embeddings,
        rms_norm_eps,
        rope_theta,
        normalization: DecoderNormalization::Rms,
        full_attention: HybridFullAttentionConfig::MultiLatent {
            num_attention_heads,
            query,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            value_head_dim,
            skip_rope: false,
        },
        linear_attention: None,
        activation: DenseDecoderActivation::Silu,
        weights,
    };
    let topology = deepseek_v3_checkpoint_topology(&plan)?;
    let mla_key_width = checked_add(
        DEEPSEEK_V3_ARCHITECTURE,
        kv_lora_rank,
        qk_rope_head_dim,
        "MLA key width",
    )?;
    let kv_cache_layout =
        KvCacheModelLayout::tensor_pair(num_layers, 1, mla_key_width, 1, kv_lora_rank).map_err(
            |error| ModelAdapterError::invalid(DEEPSEEK_V3_ARCHITECTURE, error.to_string()),
        )?;
    let execution = ModelExecutionArchitecture::Transformer {
        attention: AttentionArchitecture::MultiLatent {
            num_attention_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            value_head_dim,
        },
        feed_forward: FeedForwardArchitecture::MixtureOfExperts {
            routed_expert_count,
            experts_per_token,
            shared_expert_count,
            expert_intermediate_size,
        },
    };

    ModelDefinition::new(
        DEEPSEEK_V3_ARCHITECTURE,
        hf_config,
        execution,
        vec![RuntimeDtype::F32, RuntimeDtype::Bf16],
        Some(kv_cache_layout),
    )
    .with_serving_metadata(vocab_size, max_position_embeddings)
    .with_checkpoint_topology(topology)
    .with_hybrid_decoder(plan)
}

fn reject_unsupported_deepseek_v3_features(config: &MlaMoeConfig) -> Result<(), ModelAdapterError> {
    if config.hidden_act.as_deref() != Some("silu") {
        return Err(ModelAdapterError::invalid(
            DEEPSEEK_V3_ARCHITECTURE,
            format!("hidden_act must be silu; found {:?}", config.hidden_act),
        ));
    }
    if config.attention_bias.unwrap_or(false) {
        return Err(ModelAdapterError::invalid(
            DEEPSEEK_V3_ARCHITECTURE,
            "attention_bias=true is not implemented by the shared MLA component",
        ));
    }
    if config
        .rope_scaling
        .as_ref()
        .is_some_and(|value| !value.is_null())
    {
        return Err(ModelAdapterError::invalid(
            DEEPSEEK_V3_ARCHITECTURE,
            "rope_scaling/YaRN is not implemented by the shared MLA component",
        ));
    }
    if config
        .quantization_config
        .as_ref()
        .is_some_and(|value| !value.is_null())
    {
        return Err(ModelAdapterError::invalid(
            DEEPSEEK_V3_ARCHITECTURE,
            "quantized DeepSeek/Kimi checkpoints are not implemented; an unquantized checkpoint is required",
        ));
    }
    if config.num_nextn_predict_layers.unwrap_or(0) != 0 {
        return Err(ModelAdapterError::invalid(
            DEEPSEEK_V3_ARCHITECTURE,
            "next-token prediction layers are not implemented",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct DeepSeekV3WeightLayout<'a> {
    num_layers: usize,
    first_dense_layer_count: usize,
    moe_layer_frequency: usize,
    dense_intermediate_size: usize,
    moe_config: &'a MoeFeedForwardConfig,
    query: MultiLatentQueryConfig,
    has_correction_bias: bool,
    tied_embeddings: bool,
}

fn deepseek_v3_weight_names(layout: DeepSeekV3WeightLayout<'_>) -> HybridDecoderWeightNames {
    let DeepSeekV3WeightLayout {
        num_layers,
        first_dense_layer_count,
        moe_layer_frequency,
        dense_intermediate_size,
        moe_config,
        query,
        has_correction_bias,
        tied_embeddings,
    } = layout;
    let layers = (0..num_layers)
        .map(|layer_id| {
            let prefix = format!("model.layers.{layer_id}");
            let mixer = HybridDecoderLayerKind::MultiLatentAttention {
                cache_layer_index: layer_id,
                weights: HybridMultiLatentAttentionWeightNames {
                    query: match query {
                        MultiLatentQueryConfig::Direct => MultiLatentQueryWeightNames::Direct {
                            weight: format!("{prefix}.self_attn.q_proj.weight"),
                        },
                        MultiLatentQueryConfig::LowRank { .. } => {
                            MultiLatentQueryWeightNames::LowRank {
                                a_weight: format!("{prefix}.self_attn.q_a_proj.weight"),
                                a_norm: format!("{prefix}.self_attn.q_a_layernorm.weight"),
                                b_weight: format!("{prefix}.self_attn.q_b_proj.weight"),
                            }
                        }
                    },
                    kv_a_weight: format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
                    kv_a_norm: format!("{prefix}.self_attn.kv_a_layernorm.weight"),
                    kv_b_weight: format!("{prefix}.self_attn.kv_b_proj.weight"),
                    output_weight: format!("{prefix}.self_attn.o_proj.weight"),
                },
            };
            let is_moe =
                layer_id >= first_dense_layer_count && layer_id.is_multiple_of(moe_layer_frequency);
            let feed_forward = if is_moe {
                HybridFeedForward::MixtureOfExperts {
                    config: moe_config.clone(),
                    weights: MoeFeedForwardWeightNames {
                        gate_weight: format!("{prefix}.mlp.gate.weight"),
                        correction_bias: has_correction_bias
                            .then(|| format!("{prefix}.mlp.gate.e_score_correction_bias")),
                        experts: (0..moe_config.routed_expert_count)
                            .map(|expert| DenseFeedForwardWeightNames {
                                gate_weight: format!("{prefix}.mlp.experts.{expert}.w1.weight"),
                                down_weight: format!("{prefix}.mlp.experts.{expert}.w2.weight"),
                                up_weight: format!("{prefix}.mlp.experts.{expert}.w3.weight"),
                            })
                            .collect(),
                        shared_expert: (moe_config.shared_expert_count > 0).then(|| {
                            DenseFeedForwardWeightNames {
                                gate_weight: format!(
                                    "{prefix}.mlp.shared_experts.gate_proj.weight"
                                ),
                                up_weight: format!("{prefix}.mlp.shared_experts.up_proj.weight"),
                                down_weight: format!(
                                    "{prefix}.mlp.shared_experts.down_proj.weight"
                                ),
                            }
                        }),
                    },
                }
            } else {
                HybridFeedForward::Dense {
                    intermediate_size: dense_intermediate_size,
                    weights: DenseFeedForwardWeightNames {
                        gate_weight: format!("{prefix}.mlp.gate_proj.weight"),
                        up_weight: format!("{prefix}.mlp.up_proj.weight"),
                        down_weight: format!("{prefix}.mlp.down_proj.weight"),
                    },
                }
            };
            HybridDecoderLayerWeightNames {
                input_norm: format!("{prefix}.input_layernorm.weight"),
                mixer,
                post_attention_norm: format!("{prefix}.post_attention_layernorm.weight"),
                feed_forward,
            }
        })
        .collect();
    HybridDecoderWeightNames {
        token_embeddings: "model.embed_tokens.weight".to_string(),
        final_norm: "model.norm.weight".to_string(),
        lm_head: (!tied_embeddings).then(|| "lm_head.weight".to_string()),
        layers,
    }
}

fn deepseek_v3_checkpoint_topology(
    plan: &HybridDecoderExecutionPlan,
) -> Result<CheckpointTopology, ModelAdapterError> {
    let HybridFullAttentionConfig::MultiLatent {
        num_attention_heads,
        query,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        value_head_dim,
        ..
    } = plan.full_attention
    else {
        return Err(ModelAdapterError::invalid(
            DEEPSEEK_V3_ARCHITECTURE,
            "DeepSeek V3 checkpoint topology requires MLA",
        ));
    };
    let query_head_dim = checked_add(
        DEEPSEEK_V3_ARCHITECTURE,
        qk_nope_head_dim,
        qk_rope_head_dim,
        "MLA query head",
    )?;
    let query_size = checked_product(
        DEEPSEEK_V3_ARCHITECTURE,
        num_attention_heads,
        query_head_dim,
        "MLA query",
    )?;
    let compressed_kv_size = checked_add(
        DEEPSEEK_V3_ARCHITECTURE,
        kv_lora_rank,
        qk_rope_head_dim,
        "MLA compressed KV",
    )?;
    let expanded_head_dim = checked_add(
        DEEPSEEK_V3_ARCHITECTURE,
        qk_nope_head_dim,
        value_head_dim,
        "MLA expanded KV head",
    )?;
    let expanded_kv_size = checked_product(
        DEEPSEEK_V3_ARCHITECTURE,
        num_attention_heads,
        expanded_head_dim,
        "MLA expanded KV",
    )?;
    let attention_output_size = checked_product(
        DEEPSEEK_V3_ARCHITECTURE,
        num_attention_heads,
        value_head_dim,
        "MLA output",
    )?;
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
    let mut routed_coordinates = Vec::new();
    for (layer_id, layer) in plan.weights.layers.iter().enumerate() {
        tensors.extend([
            CheckpointTensorRequirement::with_shape(&layer.input_norm, [plan.hidden_size]),
            CheckpointTensorRequirement::with_shape(&layer.post_attention_norm, [plan.hidden_size]),
        ]);
        let HybridDecoderLayerKind::MultiLatentAttention { weights, .. } = &layer.mixer else {
            return Err(ModelAdapterError::invalid(
                DEEPSEEK_V3_ARCHITECTURE,
                "DeepSeek V3 decoder plan contains a non-MLA mixer",
            ));
        };
        match (&query, &weights.query) {
            (MultiLatentQueryConfig::Direct, MultiLatentQueryWeightNames::Direct { weight }) => {
                tensors.push(CheckpointTensorRequirement::with_shape(
                    weight,
                    [query_size, plan.hidden_size],
                ));
            }
            (
                MultiLatentQueryConfig::LowRank { rank },
                MultiLatentQueryWeightNames::LowRank {
                    a_weight,
                    a_norm,
                    b_weight,
                },
            ) => tensors.extend([
                CheckpointTensorRequirement::with_shape(a_weight, [*rank, plan.hidden_size]),
                CheckpointTensorRequirement::with_shape(a_norm, [*rank]),
                CheckpointTensorRequirement::with_shape(b_weight, [query_size, *rank]),
            ]),
            _ => {
                return Err(ModelAdapterError::invalid(
                    DEEPSEEK_V3_ARCHITECTURE,
                    "MLA query config does not match its checkpoint weight mapping",
                ));
            }
        }
        tensors.extend([
            CheckpointTensorRequirement::with_shape(
                &weights.kv_a_weight,
                [compressed_kv_size, plan.hidden_size],
            ),
            CheckpointTensorRequirement::with_shape(&weights.kv_a_norm, [kv_lora_rank]),
            CheckpointTensorRequirement::with_shape(
                &weights.kv_b_weight,
                [expanded_kv_size, kv_lora_rank],
            ),
            CheckpointTensorRequirement::with_shape(
                &weights.output_weight,
                [plan.hidden_size, attention_output_size],
            ),
        ]);
        match &layer.feed_forward {
            HybridFeedForward::Dense {
                intermediate_size,
                weights,
            } => tensors.extend(dense_feed_forward_requirements(
                weights,
                plan.hidden_size,
                *intermediate_size,
            )),
            HybridFeedForward::MixtureOfExperts { config, weights } => {
                tensors.push(CheckpointTensorRequirement::with_shape(
                    &weights.gate_weight,
                    [config.routed_expert_count, plan.hidden_size],
                ));
                if let Some(correction_bias) = &weights.correction_bias {
                    tensors.push(CheckpointTensorRequirement::with_shape(
                        correction_bias,
                        [config.routed_expert_count],
                    ));
                }
                for (expert_id, expert) in weights.experts.iter().enumerate() {
                    tensors.extend(dense_feed_forward_requirements(
                        expert,
                        plan.hidden_size,
                        config.expert_intermediate_size,
                    ));
                    routed_coordinates.push((layer_id, expert_id));
                }
                if let Some(shared) = &weights.shared_expert {
                    let shared_intermediate_size = checked_product(
                        DEEPSEEK_V3_ARCHITECTURE,
                        config.expert_intermediate_size,
                        config.shared_expert_count,
                        "shared expert intermediate",
                    )?;
                    tensors.extend(dense_feed_forward_requirements(
                        shared,
                        plan.hidden_size,
                        shared_intermediate_size,
                    ));
                }
            }
        }
    }
    Ok(CheckpointTopology::new(tensors).with_routed_experts(routed_coordinates, 3))
}

fn dense_feed_forward_requirements(
    weights: &DenseFeedForwardWeightNames,
    hidden_size: usize,
    intermediate_size: usize,
) -> [CheckpointTensorRequirement; 3] {
    [
        CheckpointTensorRequirement::with_shape(
            &weights.gate_weight,
            [intermediate_size, hidden_size],
        ),
        CheckpointTensorRequirement::with_shape(
            &weights.up_weight,
            [intermediate_size, hidden_size],
        ),
        CheckpointTensorRequirement::with_shape(
            &weights.down_weight,
            [hidden_size, intermediate_size],
        ),
    ]
}

fn positive_float(
    architecture: &'static str,
    field: &str,
    value: Option<f64>,
) -> Result<f32, ModelAdapterError> {
    let value = value.ok_or_else(|| ModelAdapterError::missing_field(architecture, field))?;
    if !value.is_finite() || value <= 0.0 || value > f32::MAX as f64 {
        return Err(ModelAdapterError::invalid(
            architecture,
            format!("{field} must be a finite positive f32 value"),
        ));
    }
    Ok(value as f32)
}

fn checked_add(
    architecture: &'static str,
    left: usize,
    right: usize,
    component: &str,
) -> Result<usize, ModelAdapterError> {
    left.checked_add(right).ok_or_else(|| {
        ModelAdapterError::invalid(architecture, format!("{component} size overflowed"))
    })
}

fn checked_product(
    architecture: &'static str,
    left: usize,
    right: usize,
    component: &str,
) -> Result<usize, ModelAdapterError> {
    left.checked_mul(right).ok_or_else(|| {
        ModelAdapterError::invalid(architecture, format!("{component} size overflowed"))
    })
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
    let checkpoint_topology = mla_moe_metadata_topology(architecture, &config, checkpoint_flavor)?;

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
