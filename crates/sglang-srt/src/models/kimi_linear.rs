use std::collections::BTreeSet;
use std::path::Path;

use serde::Deserialize;
use sglang_kernel::rotary::{RotaryEmbeddingConfig, RotaryEmbeddingStyle};

use crate::backend::RuntimeDtype;
use crate::kv_cache::KvCacheModelLayout;
use crate::model_artifacts::{CheckpointTensorRequirement, CheckpointTopology, HfModelConfig};

use super::{
    AttentionArchitecture, DecoderNormalization, DenseDecoderActivation,
    DenseFeedForwardWeightNames, FeedForwardArchitecture, HybridDecoderExecutionPlan,
    HybridDecoderLayerKind, HybridDecoderLayerWeightNames, HybridDecoderWeightNames,
    HybridFeedForward, HybridFullAttentionConfig, HybridLinearAttentionConfig,
    HybridMultiLatentAttentionWeightNames, KeyGatedDeltaWeightNames, ModelAdapter,
    ModelAdapterError, ModelDefinition, ModelExecutionArchitecture, MoeFeedForwardConfig,
    MoeFeedForwardWeightNames, MultiLatentQueryConfig, MultiLatentQueryWeightNames,
    RouterActivation, required_usize,
};

pub(crate) const KIMI_LINEAR_ARCHITECTURE: &str = "KimiLinearForCausalLM";
pub(crate) static KIMI_LINEAR_ADAPTER: KimiLinearAdapter = KimiLinearAdapter;

pub(crate) struct KimiLinearAdapter;

#[derive(Clone, Debug, Default, Deserialize)]
struct KimiLinearConfig {
    model_type: Option<String>,
    vocab_size: Option<usize>,
    model_max_length: Option<usize>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    hidden_act: Option<String>,
    rms_norm_eps: Option<f64>,
    rope_theta: Option<f64>,
    rope_scaling: Option<serde_json::Value>,
    tie_word_embeddings: Option<bool>,
    moe_intermediate_size: Option<usize>,
    moe_renormalize: Option<bool>,
    moe_router_activation_func: Option<String>,
    #[serde(alias = "n_routed_experts")]
    num_experts: Option<usize>,
    #[serde(alias = "num_experts_per_tok")]
    num_experts_per_token: Option<usize>,
    #[serde(alias = "n_shared_experts")]
    num_shared_experts: Option<usize>,
    routed_scaling_factor: Option<f64>,
    first_k_dense_replace: Option<usize>,
    moe_layer_freq: Option<usize>,
    use_grouped_topk: Option<bool>,
    num_expert_group: Option<usize>,
    topk_group: Option<usize>,
    q_lora_rank: Option<usize>,
    kv_lora_rank: Option<usize>,
    qk_nope_head_dim: Option<usize>,
    qk_rope_head_dim: Option<usize>,
    v_head_dim: Option<usize>,
    mla_use_nope: Option<bool>,
    num_nextn_predict_layers: Option<usize>,
    linear_attn_config: Option<KimiLinearAttentionConfig>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct KimiLinearAttentionConfig {
    head_dim: Option<usize>,
    num_heads: Option<usize>,
    short_conv_kernel_size: Option<usize>,
    kda_layers: Option<Vec<usize>>,
    full_attn_layers: Option<Vec<usize>>,
}

impl ModelAdapter for KimiLinearAdapter {
    fn architectures(&self) -> &'static [&'static str] {
        &[KIMI_LINEAR_ARCHITECTURE]
    }

    fn build_definition(
        &self,
        _model_path: &Path,
        config: &HfModelConfig,
    ) -> Result<ModelDefinition, ModelAdapterError> {
        build_definition(config)
    }
}

fn build_definition(hf_config: &HfModelConfig) -> Result<ModelDefinition, ModelAdapterError> {
    let config: KimiLinearConfig = hf_config.parse_text_config().map_err(|error| {
        ModelAdapterError::invalid(
            KIMI_LINEAR_ARCHITECTURE,
            format!("invalid Kimi Linear config document: {error}"),
        )
    })?;
    if config.model_type.as_deref() != Some("kimi_linear") {
        return Err(ModelAdapterError::invalid(
            KIMI_LINEAR_ARCHITECTURE,
            format!(
                "model_type must be kimi_linear; found {:?}",
                config.model_type
            ),
        ));
    }
    if config.hidden_act.as_deref() != Some("silu") {
        return Err(ModelAdapterError::invalid(
            KIMI_LINEAR_ARCHITECTURE,
            format!("hidden_act must be silu; found {:?}", config.hidden_act),
        ));
    }
    if config
        .rope_scaling
        .as_ref()
        .is_some_and(|value| !value.is_null())
    {
        return Err(ModelAdapterError::invalid(
            KIMI_LINEAR_ARCHITECTURE,
            "rope_scaling is not implemented by the shared MLA component",
        ));
    }
    if config.mla_use_nope != Some(true) {
        return Err(ModelAdapterError::invalid(
            KIMI_LINEAR_ARCHITECTURE,
            "mla_use_nope=true is required by the shared Kimi MLA component",
        ));
    }
    if config.num_nextn_predict_layers.unwrap_or(0) != 0 {
        return Err(ModelAdapterError::invalid(
            KIMI_LINEAR_ARCHITECTURE,
            "next-token prediction layers are not implemented",
        ));
    }

    let vocab_size = required_usize(KIMI_LINEAR_ARCHITECTURE, "vocab_size", config.vocab_size)?;
    let hidden_size = required_usize(KIMI_LINEAR_ARCHITECTURE, "hidden_size", config.hidden_size)?;
    let mla_query = match config.q_lora_rank {
        Some(0) => {
            return Err(ModelAdapterError::invalid(
                KIMI_LINEAR_ARCHITECTURE,
                "q_lora_rank must be non-zero when present",
            ));
        }
        Some(rank) => MultiLatentQueryConfig::LowRank { rank },
        None => MultiLatentQueryConfig::Direct,
    };
    let dense_intermediate_size = required_usize(
        KIMI_LINEAR_ARCHITECTURE,
        "intermediate_size",
        config.intermediate_size,
    )?;
    let num_layers = required_usize(
        KIMI_LINEAR_ARCHITECTURE,
        "num_hidden_layers",
        config.num_hidden_layers,
    )?;
    let num_attention_heads = required_usize(
        KIMI_LINEAR_ARCHITECTURE,
        "num_attention_heads",
        config.num_attention_heads,
    )?;
    let num_key_value_heads = config.num_key_value_heads.unwrap_or(num_attention_heads);
    if num_key_value_heads != num_attention_heads {
        return Err(ModelAdapterError::invalid(
            KIMI_LINEAR_ARCHITECTURE,
            format!(
                "num_key_value_heads {num_key_value_heads} must equal num_attention_heads {num_attention_heads} for Kimi MLA"
            ),
        ));
    }
    let kv_lora_rank = required_usize(
        KIMI_LINEAR_ARCHITECTURE,
        "kv_lora_rank",
        config.kv_lora_rank,
    )?;
    let qk_nope_head_dim = required_usize(
        KIMI_LINEAR_ARCHITECTURE,
        "qk_nope_head_dim",
        config.qk_nope_head_dim,
    )?;
    let qk_rope_head_dim = required_usize(
        KIMI_LINEAR_ARCHITECTURE,
        "qk_rope_head_dim",
        config.qk_rope_head_dim,
    )?;
    let value_head_dim = required_usize(KIMI_LINEAR_ARCHITECTURE, "v_head_dim", config.v_head_dim)?;
    let routed_expert_count =
        required_usize(KIMI_LINEAR_ARCHITECTURE, "num_experts", config.num_experts)?;
    let experts_per_token = required_usize(
        KIMI_LINEAR_ARCHITECTURE,
        "num_experts_per_token",
        config.num_experts_per_token,
    )?;
    if experts_per_token > routed_expert_count {
        return Err(ModelAdapterError::invalid(
            KIMI_LINEAR_ARCHITECTURE,
            format!(
                "num_experts_per_token {experts_per_token} exceeds num_experts {routed_expert_count}"
            ),
        ));
    }
    let expert_intermediate_size = required_usize(
        KIMI_LINEAR_ARCHITECTURE,
        "moe_intermediate_size",
        config.moe_intermediate_size,
    )?;
    let shared_expert_count = config.num_shared_experts.unwrap_or(0);
    let first_dense_layer_count = config.first_k_dense_replace.unwrap_or(0);
    if first_dense_layer_count > num_layers {
        return Err(ModelAdapterError::invalid(
            KIMI_LINEAR_ARCHITECTURE,
            "first_k_dense_replace exceeds num_hidden_layers",
        ));
    }
    let moe_layer_frequency = required_usize(
        KIMI_LINEAR_ARCHITECTURE,
        "moe_layer_freq",
        config.moe_layer_freq,
    )?;
    let expert_group_count = required_usize(
        KIMI_LINEAR_ARCHITECTURE,
        "num_expert_group",
        config.num_expert_group,
    )?;
    let selected_expert_group_count =
        required_usize(KIMI_LINEAR_ARCHITECTURE, "topk_group", config.topk_group)?;
    if !routed_expert_count.is_multiple_of(expert_group_count)
        || selected_expert_group_count > expert_group_count
    {
        return Err(ModelAdapterError::invalid(
            KIMI_LINEAR_ARCHITECTURE,
            "num_experts must divide evenly into num_expert_group and topk_group must not exceed it",
        ));
    }
    if config.use_grouped_topk == Some(false)
        && (expert_group_count != 1 || selected_expert_group_count != 1)
    {
        return Err(ModelAdapterError::invalid(
            KIMI_LINEAR_ARCHITECTURE,
            "use_grouped_topk=false is only valid with one expert group",
        ));
    }
    let router_activation = match config.moe_router_activation_func.as_deref() {
        Some("sigmoid") => RouterActivation::Sigmoid,
        Some("softmax") => RouterActivation::Softmax,
        Some(activation) => {
            return Err(ModelAdapterError::invalid(
                KIMI_LINEAR_ARCHITECTURE,
                format!("unsupported MoE router activation {activation}"),
            ));
        }
        None => {
            return Err(ModelAdapterError::missing_field(
                KIMI_LINEAR_ARCHITECTURE,
                "moe_router_activation_func",
            ));
        }
    };
    let routed_scaling_factor =
        positive_float("routed_scaling_factor", config.routed_scaling_factor)?;
    let rms_norm_eps = positive_float("rms_norm_eps", config.rms_norm_eps)?;
    let rope_theta = positive_float("rope_theta", config.rope_theta)?;
    let max_position_embeddings = required_usize(
        KIMI_LINEAR_ARCHITECTURE,
        "model_max_length",
        config.model_max_length,
    )?;
    let rotary_embedding =
        RotaryEmbeddingConfig::standard(rope_theta, RotaryEmbeddingStyle::Interleaved)
            .and_then(|rotary| {
                rotary.validate(qk_rope_head_dim)?;
                Ok(rotary)
            })
            .map_err(|error| {
                ModelAdapterError::invalid(KIMI_LINEAR_ARCHITECTURE, error.to_string())
            })?;

    let linear = config.linear_attn_config.as_ref().ok_or_else(|| {
        ModelAdapterError::missing_field(KIMI_LINEAR_ARCHITECTURE, "linear_attn_config")
    })?;
    let linear_head_dim = required_usize(
        KIMI_LINEAR_ARCHITECTURE,
        "linear_attn_config.head_dim",
        linear.head_dim,
    )?;
    let linear_num_heads = required_usize(
        KIMI_LINEAR_ARCHITECTURE,
        "linear_attn_config.num_heads",
        linear.num_heads,
    )?;
    let linear_conv_kernel_dim = required_usize(
        KIMI_LINEAR_ARCHITECTURE,
        "linear_attn_config.short_conv_kernel_size",
        linear.short_conv_kernel_size,
    )?;
    let (kda_layers, full_attention_layers) = validate_layer_partition(linear, num_layers)?;

    let execution = ModelExecutionArchitecture::Transformer {
        attention: AttentionArchitecture::HybridMultiLatent {
            num_attention_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            value_head_dim,
            linear_num_heads,
            linear_key_head_dim: linear_head_dim,
            linear_value_head_dim: linear_head_dim,
        },
        feed_forward: FeedForwardArchitecture::MixtureOfExperts {
            routed_expert_count,
            experts_per_token,
            shared_expert_count,
            expert_intermediate_size,
        },
    };
    let moe_config = MoeFeedForwardConfig {
        routed_expert_count,
        experts_per_token,
        expert_group_count,
        selected_expert_group_count,
        shared_expert_count,
        expert_intermediate_size,
        renormalize: config.moe_renormalize.unwrap_or(true),
        router_activation,
        routed_scaling_factor,
        routed_expert_weight_format: super::RoutedExpertWeightFormat::Unquantized,
    };
    let weights = weight_names(
        num_layers,
        &kda_layers,
        FeedForwardWeightLayout {
            first_dense_layer_count,
            moe_layer_frequency,
            dense_intermediate_size,
            moe_config: &moe_config,
        },
        mla_query,
        config.tie_word_embeddings == Some(true),
    );
    let plan = HybridDecoderExecutionPlan {
        vocab_size,
        hidden_size,
        max_position_embeddings,
        rms_norm_eps,
        normalization: DecoderNormalization::Rms,
        full_attention: HybridFullAttentionConfig::MultiLatent {
            num_attention_heads,
            query: mla_query,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            value_head_dim,
            skip_rope: true,
            rotary_embedding,
        },
        linear_attention: Some(HybridLinearAttentionConfig::KeyGatedDelta {
            conv_kernel_dim: linear_conv_kernel_dim,
            key_head_dim: linear_head_dim,
            value_head_dim: linear_head_dim,
            num_heads: linear_num_heads,
        }),
        activation: DenseDecoderActivation::Silu,
        weights,
    };
    let topology = checkpoint_topology(&plan)?;
    let mla_key_width = checked_add(kv_lora_rank, qk_rope_head_dim, "MLA key width")?;
    let kv_cache_layout = KvCacheModelLayout::tensor_pair(
        full_attention_layers.len(),
        1,
        mla_key_width,
        1,
        kv_lora_rank,
    )
    .map_err(|error| ModelAdapterError::invalid(KIMI_LINEAR_ARCHITECTURE, error.to_string()))?;

    ModelDefinition::new(
        KIMI_LINEAR_ARCHITECTURE,
        hf_config,
        execution,
        vec![RuntimeDtype::F32, RuntimeDtype::Bf16],
        Some(kv_cache_layout),
    )
    .with_serving_metadata(vocab_size, max_position_embeddings)
    .with_checkpoint_topology(topology)
    .with_hybrid_decoder(plan)
}

fn validate_layer_partition(
    linear: &KimiLinearAttentionConfig,
    num_layers: usize,
) -> Result<(BTreeSet<usize>, BTreeSet<usize>), ModelAdapterError> {
    let kda_layers = layer_set("kda_layers", linear.kda_layers.as_deref(), num_layers)?;
    let full_attention_layers = layer_set(
        "full_attn_layers",
        linear.full_attn_layers.as_deref(),
        num_layers,
    )?;
    if kda_layers.is_empty() || full_attention_layers.is_empty() {
        return Err(ModelAdapterError::invalid(
            KIMI_LINEAR_ARCHITECTURE,
            "Kimi hybrid execution requires at least one KDA and one full-attention layer",
        ));
    }
    if let Some(layer) = kda_layers.intersection(&full_attention_layers).next() {
        return Err(ModelAdapterError::invalid(
            KIMI_LINEAR_ARCHITECTURE,
            format!("layer {layer} appears in both KDA and full-attention lists"),
        ));
    }
    let expected = (1..=num_layers).collect::<BTreeSet<_>>();
    let actual = kda_layers
        .union(&full_attention_layers)
        .copied()
        .collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(ModelAdapterError::invalid(
            KIMI_LINEAR_ARCHITECTURE,
            "KDA and full-attention layer lists must partition every 1-based model layer",
        ));
    }
    Ok((kda_layers, full_attention_layers))
}

fn layer_set(
    field: &str,
    layers: Option<&[usize]>,
    num_layers: usize,
) -> Result<BTreeSet<usize>, ModelAdapterError> {
    let layers = layers.ok_or_else(|| {
        ModelAdapterError::missing_field(
            KIMI_LINEAR_ARCHITECTURE,
            &format!("linear_attn_config.{field}"),
        )
    })?;
    let set = layers.iter().copied().collect::<BTreeSet<_>>();
    if set.len() != layers.len() {
        return Err(ModelAdapterError::invalid(
            KIMI_LINEAR_ARCHITECTURE,
            format!("linear_attn_config.{field} contains duplicate layers"),
        ));
    }
    if let Some(layer) = set
        .iter()
        .find(|layer| **layer == 0 || **layer > num_layers)
    {
        return Err(ModelAdapterError::invalid(
            KIMI_LINEAR_ARCHITECTURE,
            format!("linear_attn_config.{field} contains out-of-range layer {layer}"),
        ));
    }
    Ok(set)
}

fn positive_float(field: &str, value: Option<f64>) -> Result<f32, ModelAdapterError> {
    let value =
        value.ok_or_else(|| ModelAdapterError::missing_field(KIMI_LINEAR_ARCHITECTURE, field))?;
    if !value.is_finite() || value <= 0.0 || value > f32::MAX as f64 {
        return Err(ModelAdapterError::invalid(
            KIMI_LINEAR_ARCHITECTURE,
            format!("{field} must be a finite positive f32 value"),
        ));
    }
    Ok(value as f32)
}

#[derive(Clone, Copy)]
struct FeedForwardWeightLayout<'a> {
    first_dense_layer_count: usize,
    moe_layer_frequency: usize,
    dense_intermediate_size: usize,
    moe_config: &'a MoeFeedForwardConfig,
}

fn weight_names(
    num_layers: usize,
    kda_layers: &BTreeSet<usize>,
    feed_forward: FeedForwardWeightLayout<'_>,
    mla_query: MultiLatentQueryConfig,
    tied_embeddings: bool,
) -> HybridDecoderWeightNames {
    let FeedForwardWeightLayout {
        first_dense_layer_count,
        moe_layer_frequency,
        dense_intermediate_size,
        moe_config,
    } = feed_forward;
    let mut cache_layer_index = 0;
    let mut state_layer_index = 0;
    let layers = (0..num_layers)
        .map(|layer_id| {
            let prefix = format!("model.layers.{layer_id}");
            let mixer = if kda_layers.contains(&(layer_id + 1)) {
                let index = state_layer_index;
                state_layer_index += 1;
                HybridDecoderLayerKind::KeyGatedDelta {
                    state_layer_index: index,
                    weights: KeyGatedDeltaWeightNames {
                        a_log: format!("{prefix}.self_attn.A_log"),
                        dt_bias: format!("{prefix}.self_attn.dt_bias"),
                        query_weight: format!("{prefix}.self_attn.q_proj.weight"),
                        key_weight: format!("{prefix}.self_attn.k_proj.weight"),
                        value_weight: format!("{prefix}.self_attn.v_proj.weight"),
                        beta_weight: format!("{prefix}.self_attn.b_proj.weight"),
                        forget_a_weight: format!("{prefix}.self_attn.f_a_proj.weight"),
                        forget_b_weight: format!("{prefix}.self_attn.f_b_proj.weight"),
                        gate_a_weight: format!("{prefix}.self_attn.g_a_proj.weight"),
                        gate_b_weight: format!("{prefix}.self_attn.g_b_proj.weight"),
                        query_conv_weight: format!("{prefix}.self_attn.q_conv1d.weight"),
                        key_conv_weight: format!("{prefix}.self_attn.k_conv1d.weight"),
                        value_conv_weight: format!("{prefix}.self_attn.v_conv1d.weight"),
                        output_norm: format!("{prefix}.self_attn.o_norm.weight"),
                        output_weight: format!("{prefix}.self_attn.o_proj.weight"),
                    },
                }
            } else {
                let index = cache_layer_index;
                cache_layer_index += 1;
                HybridDecoderLayerKind::MultiLatentAttention {
                    cache_layer_index: index,
                    weights: HybridMultiLatentAttentionWeightNames {
                        query: match mla_query {
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
                }
            };
            let is_moe =
                layer_id >= first_dense_layer_count && layer_id.is_multiple_of(moe_layer_frequency);
            let feed_forward = if is_moe {
                HybridFeedForward::MixtureOfExperts {
                    config: moe_config.clone(),
                    weights: MoeFeedForwardWeightNames {
                        gate_weight: format!("{prefix}.mlp.gate.weight"),
                        correction_bias: Some(format!("{prefix}.mlp.gate.e_score_correction_bias")),
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

fn checkpoint_topology(
    plan: &HybridDecoderExecutionPlan,
) -> Result<CheckpointTopology, ModelAdapterError> {
    let HybridFullAttentionConfig::MultiLatent {
        num_attention_heads,
        query: full_attention_query,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        value_head_dim,
        ..
    } = plan.full_attention
    else {
        return Err(ModelAdapterError::invalid(
            KIMI_LINEAR_ARCHITECTURE,
            "Kimi checkpoint topology requires MLA",
        ));
    };
    let Some(HybridLinearAttentionConfig::KeyGatedDelta {
        conv_kernel_dim,
        key_head_dim,
        value_head_dim: linear_value_head_dim,
        num_heads: linear_num_heads,
    }) = plan.linear_attention
    else {
        return Err(ModelAdapterError::invalid(
            KIMI_LINEAR_ARCHITECTURE,
            "Kimi checkpoint topology requires KDA",
        ));
    };
    let query_head_dim = checked_add(qk_nope_head_dim, qk_rope_head_dim, "MLA query head")?;
    let query_size = checked_product(num_attention_heads, query_head_dim, "MLA query")?;
    let expanded_head_dim = checked_add(qk_nope_head_dim, value_head_dim, "MLA KV head")?;
    let expanded_kv_size =
        checked_product(num_attention_heads, expanded_head_dim, "MLA expanded KV")?;
    let attention_output_size = checked_product(num_attention_heads, value_head_dim, "MLA output")?;
    let linear_key_size = checked_product(linear_num_heads, key_head_dim, "KDA key")?;
    let linear_value_size = checked_product(linear_num_heads, linear_value_head_dim, "KDA value")?;
    let compressed_kv_size = checked_add(kv_lora_rank, qk_rope_head_dim, "MLA compressed KV")?;
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
        match &layer.mixer {
            HybridDecoderLayerKind::MultiLatentAttention { weights, .. } => {
                match (&full_attention_query, &weights.query) {
                    (
                        MultiLatentQueryConfig::Direct,
                        MultiLatentQueryWeightNames::Direct { weight },
                    ) => tensors.push(CheckpointTensorRequirement::with_shape(
                        weight,
                        [query_size, plan.hidden_size],
                    )),
                    (
                        MultiLatentQueryConfig::LowRank { rank },
                        MultiLatentQueryWeightNames::LowRank {
                            a_weight,
                            a_norm,
                            b_weight,
                        },
                    ) => tensors.extend([
                        CheckpointTensorRequirement::with_shape(
                            a_weight,
                            [*rank, plan.hidden_size],
                        ),
                        CheckpointTensorRequirement::with_shape(a_norm, [*rank]),
                        CheckpointTensorRequirement::with_shape(b_weight, [query_size, *rank]),
                    ]),
                    _ => {
                        return Err(ModelAdapterError::invalid(
                            KIMI_LINEAR_ARCHITECTURE,
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
            }
            HybridDecoderLayerKind::KeyGatedDelta { weights, .. } => tensors.extend([
                CheckpointTensorRequirement::with_shape(
                    &weights.a_log,
                    [1, 1, linear_num_heads, 1],
                ),
                CheckpointTensorRequirement::with_shape(&weights.dt_bias, [linear_key_size]),
                CheckpointTensorRequirement::with_shape(
                    &weights.query_weight,
                    [linear_key_size, plan.hidden_size],
                ),
                CheckpointTensorRequirement::with_shape(
                    &weights.key_weight,
                    [linear_key_size, plan.hidden_size],
                ),
                CheckpointTensorRequirement::with_shape(
                    &weights.value_weight,
                    [linear_value_size, plan.hidden_size],
                ),
                CheckpointTensorRequirement::with_shape(
                    &weights.beta_weight,
                    [linear_num_heads, plan.hidden_size],
                ),
                CheckpointTensorRequirement::with_shape(
                    &weights.forget_a_weight,
                    [key_head_dim, plan.hidden_size],
                ),
                CheckpointTensorRequirement::with_shape(
                    &weights.forget_b_weight,
                    [linear_key_size, key_head_dim],
                ),
                CheckpointTensorRequirement::with_shape(
                    &weights.gate_a_weight,
                    [key_head_dim, plan.hidden_size],
                ),
                CheckpointTensorRequirement::with_shape(
                    &weights.gate_b_weight,
                    [linear_value_size, key_head_dim],
                ),
                CheckpointTensorRequirement::with_shape(
                    &weights.query_conv_weight,
                    [linear_key_size, conv_kernel_dim],
                ),
                CheckpointTensorRequirement::with_shape(
                    &weights.key_conv_weight,
                    [linear_key_size, conv_kernel_dim],
                ),
                CheckpointTensorRequirement::with_shape(
                    &weights.value_conv_weight,
                    [linear_value_size, conv_kernel_dim],
                ),
                CheckpointTensorRequirement::with_shape(
                    &weights.output_norm,
                    [linear_value_head_dim],
                ),
                CheckpointTensorRequirement::with_shape(
                    &weights.output_weight,
                    [plan.hidden_size, linear_value_size],
                ),
            ]),
            _ => {
                return Err(ModelAdapterError::invalid(
                    KIMI_LINEAR_ARCHITECTURE,
                    "Kimi checkpoint topology received an incompatible mixer",
                ));
            }
        }
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

fn checked_add(left: usize, right: usize, component: &str) -> Result<usize, ModelAdapterError> {
    left.checked_add(right).ok_or_else(|| {
        ModelAdapterError::invalid(
            KIMI_LINEAR_ARCHITECTURE,
            format!("{component} size overflowed"),
        )
    })
}

fn checked_product(left: usize, right: usize, component: &str) -> Result<usize, ModelAdapterError> {
    left.checked_mul(right).ok_or_else(|| {
        ModelAdapterError::invalid(
            KIMI_LINEAR_ARCHITECTURE,
            format!("{component} size overflowed"),
        )
    })
}
