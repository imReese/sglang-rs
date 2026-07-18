use std::path::Path;

use serde::Deserialize;
use sglang_kernel::rotary::{RotaryEmbeddingConfig, RotaryEmbeddingStyle};

use crate::backend::RuntimeDtype;
use crate::kv_cache::KvCacheModelLayout;
use crate::model_artifacts::HfModelConfig;

use super::{
    AttentionArchitecture, DecoderNormalization, DenseDecoderActivation,
    DenseFeedForwardWeightNames, FeedForwardArchitecture, GatedDeltaNetWeightNames,
    HybridDecoderExecutionPlan, HybridDecoderLayerKind, HybridDecoderLayerWeightNames,
    HybridDecoderWeightNames, HybridFeedForward, HybridFullAttentionConfig,
    HybridFullAttentionWeightNames, HybridLinearAttentionConfig, ModelAdapter, ModelAdapterError,
    ModelDefinition, ModelExecutionArchitecture, hybrid_decoder_checkpoint_topology,
    required_usize,
};

pub(crate) const QWEN3_5_ARCHITECTURE: &str = "Qwen3_5ForConditionalGeneration";
pub(crate) static QWEN3_5_ADAPTER: Qwen3_5Adapter = Qwen3_5Adapter;

pub(crate) struct Qwen3_5Adapter;

#[derive(Clone, Debug, Default, Deserialize)]
struct Qwen3_5TextConfig {
    model_type: Option<String>,
    vocab_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    num_hidden_layers: Option<usize>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    hidden_act: Option<String>,
    attention_bias: Option<bool>,
    rms_norm_eps: Option<f64>,
    rope_theta: Option<f64>,
    rope_scaling: Option<serde_json::Value>,
    rope_parameters: Option<serde_json::Value>,
    partial_rotary_factor: Option<f64>,
    tie_word_embeddings: Option<bool>,
    layer_types: Vec<String>,
    full_attention_interval: Option<usize>,
    attn_output_gate: Option<bool>,
    linear_conv_kernel_dim: Option<usize>,
    linear_key_head_dim: Option<usize>,
    linear_value_head_dim: Option<usize>,
    linear_num_key_heads: Option<usize>,
    linear_num_value_heads: Option<usize>,
    output_gate_type: Option<String>,
    mamba_ssm_dtype: Option<String>,
}

impl ModelAdapter for Qwen3_5Adapter {
    fn architectures(&self) -> &'static [&'static str] {
        &[QWEN3_5_ARCHITECTURE]
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
    let config: Qwen3_5TextConfig = hf_config.parse_text_config().map_err(|error| {
        ModelAdapterError::invalid(
            QWEN3_5_ARCHITECTURE,
            format!("invalid Qwen3.5 hybrid config document: {error}"),
        )
    })?;
    if config.model_type.as_deref() != Some("qwen3_5_text") {
        return Err(ModelAdapterError::invalid(
            QWEN3_5_ARCHITECTURE,
            format!(
                "text_config.model_type must be qwen3_5_text; found {:?}. Qwen3.5 MoE and non-text backbones are not implemented",
                config.model_type
            ),
        ));
    }
    if config.attention_bias == Some(true) {
        return Err(ModelAdapterError::invalid(
            QWEN3_5_ARCHITECTURE,
            "attention_bias=true is not implemented by the shared hybrid decoder",
        ));
    }
    match config.hidden_act.as_deref() {
        Some("silu") => {}
        Some(activation) => {
            return Err(ModelAdapterError::invalid(
                QWEN3_5_ARCHITECTURE,
                format!("unsupported hidden_act {activation}; hybrid decoder requires silu"),
            ));
        }
        None => {
            return Err(ModelAdapterError::missing_field(
                QWEN3_5_ARCHITECTURE,
                "text_config.hidden_act",
            ));
        }
    }
    if !matches!(
        config.output_gate_type.as_deref(),
        None | Some("silu") | Some("swish")
    ) {
        return Err(ModelAdapterError::invalid(
            QWEN3_5_ARCHITECTURE,
            format!(
                "unsupported output_gate_type {:?}; hybrid decoder supports silu/swish",
                config.output_gate_type
            ),
        ));
    }
    if config
        .mamba_ssm_dtype
        .as_deref()
        .is_some_and(|dtype| dtype != "float32")
    {
        return Err(ModelAdapterError::invalid(
            QWEN3_5_ARCHITECTURE,
            format!(
                "unsupported mamba_ssm_dtype {:?}; recurrent state backend requires float32",
                config.mamba_ssm_dtype
            ),
        ));
    }

    validate_rope(&config)?;

    let vocab_size = required_usize(QWEN3_5_ARCHITECTURE, "vocab_size", config.vocab_size)?;
    let num_layers = required_usize(
        QWEN3_5_ARCHITECTURE,
        "num_hidden_layers",
        config.num_hidden_layers,
    )?;
    let hidden_size = required_usize(QWEN3_5_ARCHITECTURE, "hidden_size", config.hidden_size)?;
    let intermediate_size = required_usize(
        QWEN3_5_ARCHITECTURE,
        "intermediate_size",
        config.intermediate_size,
    )?;
    let num_attention_heads = required_usize(
        QWEN3_5_ARCHITECTURE,
        "num_attention_heads",
        config.num_attention_heads,
    )?;
    let num_key_value_heads = required_usize(
        QWEN3_5_ARCHITECTURE,
        "num_key_value_heads",
        config.num_key_value_heads,
    )?;
    let attention_head_dim = required_usize(QWEN3_5_ARCHITECTURE, "head_dim", config.head_dim)?;
    let linear_conv_kernel_dim = required_usize(
        QWEN3_5_ARCHITECTURE,
        "linear_conv_kernel_dim",
        config.linear_conv_kernel_dim,
    )?;
    let linear_key_head_dim = required_usize(
        QWEN3_5_ARCHITECTURE,
        "linear_key_head_dim",
        config.linear_key_head_dim,
    )?;
    let linear_value_head_dim = required_usize(
        QWEN3_5_ARCHITECTURE,
        "linear_value_head_dim",
        config.linear_value_head_dim,
    )?;
    let linear_num_key_heads = required_usize(
        QWEN3_5_ARCHITECTURE,
        "linear_num_key_heads",
        config.linear_num_key_heads,
    )?;
    let linear_num_value_heads = required_usize(
        QWEN3_5_ARCHITECTURE,
        "linear_num_value_heads",
        config.linear_num_value_heads,
    )?;
    if !num_attention_heads.is_multiple_of(num_key_value_heads) {
        return Err(ModelAdapterError::invalid(
            QWEN3_5_ARCHITECTURE,
            "num_attention_heads must be divisible by num_key_value_heads",
        ));
    }
    if !linear_num_value_heads.is_multiple_of(linear_num_key_heads) {
        return Err(ModelAdapterError::invalid(
            QWEN3_5_ARCHITECTURE,
            "linear_num_value_heads must be divisible by linear_num_key_heads",
        ));
    }

    let rms_norm_eps = required_positive_float(config.rms_norm_eps, "rms_norm_eps")?;
    let rope_theta = config
        .rope_theta
        .or_else(|| rope_parameter(&config, "rope_theta"))
        .unwrap_or(10_000.0);
    if !rope_theta.is_finite() || rope_theta <= 0.0 {
        return Err(ModelAdapterError::invalid(
            QWEN3_5_ARCHITECTURE,
            "rope_theta must be finite and positive",
        ));
    }
    let partial_rotary_factor = config
        .partial_rotary_factor
        .or_else(|| rope_parameter(&config, "partial_rotary_factor"))
        .unwrap_or(0.25);
    if !partial_rotary_factor.is_finite()
        || partial_rotary_factor <= 0.0
        || partial_rotary_factor > 1.0
    {
        return Err(ModelAdapterError::invalid(
            QWEN3_5_ARCHITECTURE,
            "partial_rotary_factor must be finite and in (0, 1]",
        ));
    }
    let rotary_width = attention_head_dim as f64 * partial_rotary_factor;
    let rotary_dim = rotary_width.round() as usize;
    if (rotary_width - rotary_dim as f64).abs() > f64::EPSILON
        || rotary_dim == 0
        || !rotary_dim.is_multiple_of(2)
    {
        return Err(ModelAdapterError::invalid(
            QWEN3_5_ARCHITECTURE,
            format!(
                "head_dim {attention_head_dim} * partial_rotary_factor {partial_rotary_factor} must produce a non-zero even rotary dimension"
            ),
        ));
    }
    let rotary_embedding =
        RotaryEmbeddingConfig::standard(rope_theta as f32, RotaryEmbeddingStyle::Neox)
            .and_then(|rotary| {
                rotary.validate(rotary_dim)?;
                Ok(rotary)
            })
            .map_err(|error| ModelAdapterError::invalid(QWEN3_5_ARCHITECTURE, error.to_string()))?;

    let layer_types = resolve_layer_types(&config, num_layers)?;
    let full_attention_layer_count = layer_types
        .iter()
        .filter(|layer_type| **layer_type == "full_attention")
        .count();
    let recurrent_state_layer_count = num_layers - full_attention_layer_count;
    if full_attention_layer_count == 0 || recurrent_state_layer_count == 0 {
        return Err(ModelAdapterError::invalid(
            QWEN3_5_ARCHITECTURE,
            "hybrid decoder requires at least one full_attention and one linear_attention layer",
        ));
    }

    let execution = ModelExecutionArchitecture::Transformer {
        attention: AttentionArchitecture::Hybrid {
            num_attention_heads,
            num_key_value_heads,
            attention_head_dim,
            linear_num_key_heads,
            linear_num_value_heads,
            linear_key_head_dim,
            linear_value_head_dim,
        },
        feed_forward: FeedForwardArchitecture::Dense { intermediate_size },
    };
    let weights = weight_names(
        &layer_types,
        config.tie_word_embeddings == Some(true),
        intermediate_size,
    );
    let max_position_embeddings = config.max_position_embeddings.unwrap_or(32_768);
    if max_position_embeddings == 0 {
        return Err(ModelAdapterError::invalid(
            QWEN3_5_ARCHITECTURE,
            "max_position_embeddings must be non-zero",
        ));
    }
    let plan = HybridDecoderExecutionPlan {
        vocab_size,
        hidden_size,
        max_position_embeddings,
        rms_norm_eps,
        normalization: DecoderNormalization::GemmaRms,
        full_attention: HybridFullAttentionConfig::MultiHead {
            num_attention_heads,
            num_key_value_heads,
            head_dim: attention_head_dim,
            rotary_dim,
            output_gate: config.attn_output_gate.unwrap_or(true),
            rotary_embedding,
        },
        linear_attention: Some(HybridLinearAttentionConfig::GatedDeltaNet {
            conv_kernel_dim: linear_conv_kernel_dim,
            key_head_dim: linear_key_head_dim,
            value_head_dim: linear_value_head_dim,
            num_key_heads: linear_num_key_heads,
            num_value_heads: linear_num_value_heads,
        }),
        activation: DenseDecoderActivation::Silu,
        weights,
    };
    let topology = hybrid_decoder_checkpoint_topology(QWEN3_5_ARCHITECTURE, &plan)?;

    ModelDefinition::new(
        QWEN3_5_ARCHITECTURE,
        hf_config,
        execution,
        vec![RuntimeDtype::F32, RuntimeDtype::Fp16, RuntimeDtype::Bf16],
        Some(KvCacheModelLayout::multi_tensor(
            full_attention_layer_count,
            num_key_value_heads,
            attention_head_dim,
            2,
        )),
    )
    .with_serving_metadata(vocab_size, max_position_embeddings)
    .with_checkpoint_topology(topology)
    .with_hybrid_decoder(plan)
}

fn resolve_layer_types(
    config: &Qwen3_5TextConfig,
    num_layers: usize,
) -> Result<Vec<&str>, ModelAdapterError> {
    let layer_types: Vec<&str> = if config.layer_types.is_empty() {
        let interval = required_usize(
            QWEN3_5_ARCHITECTURE,
            "full_attention_interval",
            config.full_attention_interval,
        )?;
        (0..num_layers)
            .map(|layer| {
                if (layer + 1).is_multiple_of(interval) {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect()
    } else {
        if config.layer_types.len() != num_layers {
            return Err(ModelAdapterError::invalid(
                QWEN3_5_ARCHITECTURE,
                format!(
                    "layer_types has {} entries but num_hidden_layers is {num_layers}",
                    config.layer_types.len()
                ),
            ));
        }
        config.layer_types.iter().map(String::as_str).collect()
    };
    if let Some(layer_type) = layer_types
        .iter()
        .find(|layer_type| !matches!(**layer_type, "full_attention" | "linear_attention"))
    {
        return Err(ModelAdapterError::invalid(
            QWEN3_5_ARCHITECTURE,
            format!("unsupported hybrid layer type {layer_type}"),
        ));
    }
    Ok(layer_types)
}

fn required_positive_float(value: Option<f64>, field: &str) -> Result<f32, ModelAdapterError> {
    let value =
        value.ok_or_else(|| ModelAdapterError::missing_field(QWEN3_5_ARCHITECTURE, field))?;
    if !value.is_finite() || value <= 0.0 {
        return Err(ModelAdapterError::invalid(
            QWEN3_5_ARCHITECTURE,
            format!("{field} must be finite and positive"),
        ));
    }
    Ok(value as f32)
}

fn rope_parameter(config: &Qwen3_5TextConfig, field: &str) -> Option<f64> {
    config
        .rope_parameters
        .as_ref()
        .and_then(|parameters| parameters.get(field))
        .and_then(serde_json::Value::as_f64)
}

fn validate_rope(config: &Qwen3_5TextConfig) -> Result<(), ModelAdapterError> {
    let Some(parameters) = config
        .rope_scaling
        .as_ref()
        .or(config.rope_parameters.as_ref())
    else {
        return Ok(());
    };
    let parameters = parameters.as_object().ok_or_else(|| {
        ModelAdapterError::invalid(QWEN3_5_ARCHITECTURE, "rope_parameters must be an object")
    })?;
    let rope_type = parameters
        .get("rope_type")
        .or_else(|| parameters.get("type"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("default");
    if rope_type != "default" {
        return Err(ModelAdapterError::invalid(
            QWEN3_5_ARCHITECTURE,
            format!("RoPE variant {rope_type} is not implemented by the hybrid decoder"),
        ));
    }
    Ok(())
}

fn weight_names(
    layer_types: &[&str],
    tied_embeddings: bool,
    intermediate_size: usize,
) -> HybridDecoderWeightNames {
    let mut cache_layer_index = 0;
    let mut state_layer_index = 0;
    let layers = layer_types
        .iter()
        .enumerate()
        .map(|(layer_id, layer_type)| {
            let prefix = format!("model.language_model.layers.{layer_id}");
            let mixer = match *layer_type {
                "full_attention" => {
                    let index = cache_layer_index;
                    cache_layer_index += 1;
                    HybridDecoderLayerKind::FullAttention {
                        cache_layer_index: index,
                        weights: HybridFullAttentionWeightNames {
                            query_weight: format!("{prefix}.self_attn.q_proj.weight"),
                            query_norm: format!("{prefix}.self_attn.q_norm.weight"),
                            key_weight: format!("{prefix}.self_attn.k_proj.weight"),
                            key_norm: format!("{prefix}.self_attn.k_norm.weight"),
                            value_weight: format!("{prefix}.self_attn.v_proj.weight"),
                            output_weight: format!("{prefix}.self_attn.o_proj.weight"),
                        },
                    }
                }
                "linear_attention" => {
                    let index = state_layer_index;
                    state_layer_index += 1;
                    HybridDecoderLayerKind::GatedDeltaNet {
                        state_layer_index: index,
                        weights: GatedDeltaNetWeightNames {
                            a_log: format!("{prefix}.linear_attn.A_log"),
                            conv1d_weight: format!("{prefix}.linear_attn.conv1d.weight"),
                            dt_bias: format!("{prefix}.linear_attn.dt_bias"),
                            in_proj_a_weight: format!("{prefix}.linear_attn.in_proj_a.weight"),
                            in_proj_b_weight: format!("{prefix}.linear_attn.in_proj_b.weight"),
                            in_proj_qkv_weight: format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                            in_proj_z_weight: format!("{prefix}.linear_attn.in_proj_z.weight"),
                            norm_weight: format!("{prefix}.linear_attn.norm.weight"),
                            output_weight: format!("{prefix}.linear_attn.out_proj.weight"),
                        },
                    }
                }
                _ => unreachable!("layer types were validated before weight mapping"),
            };
            HybridDecoderLayerWeightNames {
                input_norm: format!("{prefix}.input_layernorm.weight"),
                mixer,
                post_attention_norm: format!("{prefix}.post_attention_layernorm.weight"),
                feed_forward: HybridFeedForward::Dense {
                    intermediate_size,
                    weights: DenseFeedForwardWeightNames {
                        gate_weight: format!("{prefix}.mlp.gate_proj.weight"),
                        up_weight: format!("{prefix}.mlp.up_proj.weight"),
                        down_weight: format!("{prefix}.mlp.down_proj.weight"),
                    },
                },
            }
        })
        .collect();

    HybridDecoderWeightNames {
        token_embeddings: "model.language_model.embed_tokens.weight".to_string(),
        final_norm: "model.language_model.norm.weight".to_string(),
        lm_head: (!tied_embeddings).then(|| "lm_head.weight".to_string()),
        layers,
    }
}
