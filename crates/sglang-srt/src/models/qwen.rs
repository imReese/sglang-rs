use std::path::Path;

use serde::Deserialize;

use crate::backend::RuntimeDtype;
use crate::kv_cache::KvCacheModelLayout;
use crate::model_artifacts::HfModelConfig;

use super::{
    AttentionArchitecture, DenseDecoderActivation, DenseDecoderExecutionPlan,
    DenseDecoderLayerWeightNames, DenseDecoderWeightNames, FeedForwardArchitecture, ModelAdapter,
    ModelAdapterError, ModelDefinition, ModelExecutionArchitecture,
    dense_decoder_checkpoint_topology, required_usize,
};

pub(crate) const QWEN2_ARCHITECTURE: &str = "Qwen2ForCausalLM";
pub(crate) const QWEN3_ARCHITECTURE: &str = "Qwen3ForCausalLM";
pub(crate) static QWEN2_ADAPTER: Qwen2Adapter = Qwen2Adapter;
pub(crate) static QWEN3_ADAPTER: Qwen3Adapter = Qwen3Adapter;

pub(crate) struct Qwen2Adapter;
pub(crate) struct Qwen3Adapter;

#[derive(Clone, Debug, Default, Deserialize)]
struct QwenDenseConfig {
    vocab_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    num_hidden_layers: Option<usize>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_attention_heads: Option<usize>,
    original_num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    hidden_act: Option<String>,
    attention_bias: Option<bool>,
    rms_norm_eps: Option<f64>,
    rope_theta: Option<f64>,
    rope_scaling: Option<serde_json::Value>,
    rope_parameters: Option<serde_json::Value>,
    use_sliding_window: Option<bool>,
    sliding_window: Option<usize>,
    tie_word_embeddings: Option<bool>,
}

impl ModelAdapter for Qwen2Adapter {
    fn architectures(&self) -> &'static [&'static str] {
        &[QWEN2_ARCHITECTURE]
    }

    fn build_definition(
        &self,
        _model_path: &Path,
        config: &HfModelConfig,
    ) -> Result<ModelDefinition, ModelAdapterError> {
        build_qwen2_definition(config)
    }
}

impl ModelAdapter for Qwen3Adapter {
    fn architectures(&self) -> &'static [&'static str] {
        &[QWEN3_ARCHITECTURE]
    }

    fn build_definition(
        &self,
        _model_path: &Path,
        config: &HfModelConfig,
    ) -> Result<ModelDefinition, ModelAdapterError> {
        build_qwen3_definition(config)
    }
}

fn build_qwen2_definition(config: &HfModelConfig) -> Result<ModelDefinition, ModelAdapterError> {
    let typed = parse_qwen_config(config, QWEN2_ARCHITECTURE)?;
    build_qwen_definition(config, &typed, QWEN2_ARCHITECTURE, qwen2_weight_names)
}

fn build_qwen3_definition(config: &HfModelConfig) -> Result<ModelDefinition, ModelAdapterError> {
    let typed = parse_qwen_config(config, QWEN3_ARCHITECTURE)?;
    let attention_bias = typed.attention_bias.unwrap_or(false);
    build_qwen_definition(
        config,
        &typed,
        QWEN3_ARCHITECTURE,
        |num_layers, tied_embeddings| {
            qwen3_weight_names(num_layers, tied_embeddings, attention_bias)
        },
    )
}

fn parse_qwen_config(
    config: &HfModelConfig,
    architecture: &'static str,
) -> Result<QwenDenseConfig, ModelAdapterError> {
    config.parse_text_config().map_err(|error| {
        ModelAdapterError::invalid(
            architecture,
            format!("invalid Qwen dense config document: {error}"),
        )
    })
}

fn build_qwen_definition(
    hf_config: &HfModelConfig,
    config: &QwenDenseConfig,
    architecture: &'static str,
    weight_names: impl FnOnce(usize, bool) -> DenseDecoderWeightNames,
) -> Result<ModelDefinition, ModelAdapterError> {
    let vocab_size = required_usize(architecture, "vocab_size", config.vocab_size)?;
    let num_layers = required_usize(architecture, "num_hidden_layers", config.num_hidden_layers)?;
    let hidden_size = required_usize(architecture, "hidden_size", config.hidden_size)?;
    let intermediate_size =
        required_usize(architecture, "intermediate_size", config.intermediate_size)?;
    let num_attention_heads = required_usize(
        architecture,
        "num_attention_heads",
        config.num_attention_heads,
    )?;
    let num_key_value_heads = config.num_key_value_heads.unwrap_or(num_attention_heads);
    if num_key_value_heads == 0 || !num_attention_heads.is_multiple_of(num_key_value_heads) {
        return Err(ModelAdapterError::invalid(
            architecture,
            format!(
                "num_attention_heads ({num_attention_heads}) must be divisible by non-zero num_key_value_heads ({num_key_value_heads})"
            ),
        ));
    }
    let head_divisor = config
        .original_num_attention_heads
        .unwrap_or(num_attention_heads);
    let head_dim = match (config.original_num_attention_heads, config.head_dim) {
        (Some(_), _) | (None, None) if hidden_size.is_multiple_of(head_divisor) => {
            hidden_size / head_divisor
        }
        (Some(_), _) | (None, None) => {
            return Err(ModelAdapterError::invalid(
                architecture,
                format!(
                    "hidden_size ({hidden_size}) must be divisible by attention head divisor ({head_divisor})"
                ),
            ));
        }
        (None, Some(head_dim)) if head_dim > 0 => head_dim,
        (None, Some(_)) => {
            return Err(ModelAdapterError::invalid(
                architecture,
                "head_dim must be non-zero",
            ));
        }
    };
    if !head_dim.is_multiple_of(2) {
        return Err(ModelAdapterError::invalid(
            architecture,
            format!("NeoX RoPE requires an even head_dim, found {head_dim}"),
        ));
    }

    match config.hidden_act.as_deref() {
        Some("silu") => {}
        Some(hidden_act) => {
            return Err(ModelAdapterError::invalid(
                architecture,
                format!("unsupported hidden_act {hidden_act}; shared dense decoder requires silu"),
            ));
        }
        None => {
            return Err(ModelAdapterError::missing_field(architecture, "hidden_act"));
        }
    }
    if config.use_sliding_window == Some(true) {
        return Err(ModelAdapterError::invalid(
            architecture,
            format!(
                "sliding-window attention is not implemented (sliding_window={:?})",
                config.sliding_window
            ),
        ));
    }
    validate_default_rope(architecture, config)?;

    let rms_norm_eps = config
        .rms_norm_eps
        .ok_or_else(|| ModelAdapterError::missing_field(architecture, "rms_norm_eps"))?
        as f32;
    if !rms_norm_eps.is_finite() || rms_norm_eps < 0.0 {
        return Err(ModelAdapterError::invalid(
            architecture,
            "rms_norm_eps must be finite and non-negative",
        ));
    }
    let rope_theta = config
        .rope_theta
        .or_else(|| rope_parameter(config, "rope_theta"))
        .unwrap_or(1_000_000.0) as f32;
    if !rope_theta.is_finite() || rope_theta <= 0.0 {
        return Err(ModelAdapterError::invalid(
            architecture,
            "rope_theta must be finite and positive",
        ));
    }
    let max_position_embeddings = config.max_position_embeddings.unwrap_or(32_768);
    if max_position_embeddings == 0 {
        return Err(ModelAdapterError::invalid(
            architecture,
            "max_position_embeddings must be non-zero",
        ));
    }

    let execution = ModelExecutionArchitecture::Transformer {
        attention: AttentionArchitecture::MultiHead {
            num_attention_heads,
            num_key_value_heads,
            head_dim,
        },
        feed_forward: FeedForwardArchitecture::Dense { intermediate_size },
    };
    let plan = DenseDecoderExecutionPlan {
        vocab_size,
        hidden_size,
        max_position_embeddings,
        rms_norm_eps,
        rope_theta,
        activation: DenseDecoderActivation::Silu,
        weights: weight_names(num_layers, config.tie_word_embeddings == Some(true)),
    };
    let topology = dense_decoder_checkpoint_topology(architecture, execution, &plan)?;

    Ok(ModelDefinition::new(
        architecture,
        hf_config,
        execution,
        vec![RuntimeDtype::F32, RuntimeDtype::Fp16, RuntimeDtype::Bf16],
        Some(KvCacheModelLayout::multi_tensor(
            num_layers,
            num_key_value_heads,
            head_dim,
            2,
        )),
    )
    .with_serving_metadata(vocab_size, max_position_embeddings)
    .with_checkpoint_topology(topology)
    .with_dense_decoder(plan))
}

fn rope_parameter(config: &QwenDenseConfig, field: &str) -> Option<f64> {
    config
        .rope_parameters
        .as_ref()
        .and_then(|parameters| parameters.get(field))
        .and_then(serde_json::Value::as_f64)
}

fn validate_default_rope(
    architecture: &'static str,
    config: &QwenDenseConfig,
) -> Result<(), ModelAdapterError> {
    let Some(rope_scaling) = config
        .rope_scaling
        .as_ref()
        .or(config.rope_parameters.as_ref())
    else {
        return Ok(());
    };
    let Some(parameters) = rope_scaling.as_object() else {
        return Err(ModelAdapterError::invalid(
            architecture,
            "rope_scaling/rope_parameters must be an object",
        ));
    };
    let rope_type = parameters
        .get("rope_type")
        .or_else(|| parameters.get("type"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("default");
    if rope_type != "default"
        || parameters.contains_key("mrope_section")
        || parameters
            .get("use_fope")
            .and_then(serde_json::Value::as_bool)
            == Some(true)
    {
        return Err(ModelAdapterError::invalid(
            architecture,
            format!("RoPE variant {rope_type} is not implemented by the dense decoder backend"),
        ));
    }
    Ok(())
}

fn qwen2_weight_names(num_layers: usize, tied_embeddings: bool) -> DenseDecoderWeightNames {
    DenseDecoderWeightNames {
        token_embeddings: "model.embed_tokens.weight".to_string(),
        final_norm: "model.norm.weight".to_string(),
        lm_head: (!tied_embeddings).then(|| "lm_head.weight".to_string()),
        layers: (0..num_layers)
            .map(|layer_id| {
                let prefix = format!("model.layers.{layer_id}");
                DenseDecoderLayerWeightNames {
                    input_norm: format!("{prefix}.input_layernorm.weight"),
                    query_weight: format!("{prefix}.self_attn.q_proj.weight"),
                    query_bias: Some(format!("{prefix}.self_attn.q_proj.bias")),
                    query_norm: None,
                    key_weight: format!("{prefix}.self_attn.k_proj.weight"),
                    key_bias: Some(format!("{prefix}.self_attn.k_proj.bias")),
                    key_norm: None,
                    value_weight: format!("{prefix}.self_attn.v_proj.weight"),
                    value_bias: Some(format!("{prefix}.self_attn.v_proj.bias")),
                    output_weight: format!("{prefix}.self_attn.o_proj.weight"),
                    output_bias: None,
                    post_attention_norm: format!("{prefix}.post_attention_layernorm.weight"),
                    gate_weight: format!("{prefix}.mlp.gate_proj.weight"),
                    up_weight: format!("{prefix}.mlp.up_proj.weight"),
                    down_weight: format!("{prefix}.mlp.down_proj.weight"),
                }
            })
            .collect(),
    }
}

fn qwen3_weight_names(
    num_layers: usize,
    tied_embeddings: bool,
    attention_bias: bool,
) -> DenseDecoderWeightNames {
    DenseDecoderWeightNames {
        token_embeddings: "model.embed_tokens.weight".to_string(),
        final_norm: "model.norm.weight".to_string(),
        lm_head: (!tied_embeddings).then(|| "lm_head.weight".to_string()),
        layers: (0..num_layers)
            .map(|layer_id| {
                let prefix = format!("model.layers.{layer_id}");
                DenseDecoderLayerWeightNames {
                    input_norm: format!("{prefix}.input_layernorm.weight"),
                    query_weight: format!("{prefix}.self_attn.q_proj.weight"),
                    query_bias: attention_bias.then(|| format!("{prefix}.self_attn.q_proj.bias")),
                    query_norm: Some(format!("{prefix}.self_attn.q_norm.weight")),
                    key_weight: format!("{prefix}.self_attn.k_proj.weight"),
                    key_bias: attention_bias.then(|| format!("{prefix}.self_attn.k_proj.bias")),
                    key_norm: Some(format!("{prefix}.self_attn.k_norm.weight")),
                    value_weight: format!("{prefix}.self_attn.v_proj.weight"),
                    value_bias: attention_bias.then(|| format!("{prefix}.self_attn.v_proj.bias")),
                    output_weight: format!("{prefix}.self_attn.o_proj.weight"),
                    output_bias: attention_bias.then(|| format!("{prefix}.self_attn.o_proj.bias")),
                    post_attention_norm: format!("{prefix}.post_attention_layernorm.weight"),
                    gate_weight: format!("{prefix}.mlp.gate_proj.weight"),
                    up_weight: format!("{prefix}.mlp.up_proj.weight"),
                    down_weight: format!("{prefix}.mlp.down_proj.weight"),
                }
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::json;

    use super::*;
    use crate::model_artifacts::LocalModelArtifacts;

    #[test]
    fn qwen_adapter_validates_dense_decoder_weight_names_and_shapes() {
        let model_dir = temp_model_dir("qwen-complete-checkpoint");
        write_qwen_checkpoint(&model_dir, None);
        let artifacts =
            LocalModelArtifacts::from_model_path(&model_dir).expect("Qwen artifacts should load");

        build_qwen2_definition(artifacts.config())
            .expect("Qwen definition should build")
            .validate_checkpoint(&artifacts)
            .expect("complete Qwen dense decoder checkpoint should validate");

        fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
    }

    #[test]
    fn qwen_adapter_rejects_missing_dense_decoder_projection() {
        let model_dir = temp_model_dir("qwen-missing-projection");
        write_qwen_checkpoint(&model_dir, Some("model.layers.0.mlp.down_proj.weight"));
        let artifacts =
            LocalModelArtifacts::from_model_path(&model_dir).expect("Qwen artifacts should load");

        let error = build_qwen2_definition(artifacts.config())
            .expect("Qwen definition should build")
            .validate_checkpoint(&artifacts)
            .expect_err("missing Qwen projection must fail checkpoint validation");
        assert!(error.to_string().contains("mlp.down_proj.weight"));

        fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
    }

    fn write_qwen_checkpoint(model_dir: &Path, omitted_tensor: Option<&str>) {
        fs::create_dir_all(model_dir).expect("temp model dir should be created");
        fs::write(
            model_dir.join("config.json"),
            r#"{
  "architectures": ["Qwen2ForCausalLM"],
  "model_type": "qwen2",
  "vocab_size": 8,
  "num_hidden_layers": 1,
  "hidden_size": 4,
  "intermediate_size": 8,
  "num_attention_heads": 2,
  "num_key_value_heads": 1,
  "hidden_act": "silu",
  "rms_norm_eps": 0.000001,
  "rope_theta": 1000000.0,
  "max_position_embeddings": 32,
  "tie_word_embeddings": false
}"#,
        )
        .expect("config should be written");

        let tensors = [
            ("model.embed_tokens.weight", vec![8, 4]),
            ("model.norm.weight", vec![4]),
            ("lm_head.weight", vec![8, 4]),
            ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.q_proj.bias", vec![4]),
            ("model.layers.0.self_attn.k_proj.weight", vec![2, 4]),
            ("model.layers.0.self_attn.k_proj.bias", vec![2]),
            ("model.layers.0.self_attn.v_proj.weight", vec![2, 4]),
            ("model.layers.0.self_attn.v_proj.bias", vec![2]),
            ("model.layers.0.self_attn.o_proj.weight", vec![4, 4]),
            ("model.layers.0.input_layernorm.weight", vec![4]),
            ("model.layers.0.post_attention_layernorm.weight", vec![4]),
            ("model.layers.0.mlp.gate_proj.weight", vec![8, 4]),
            ("model.layers.0.mlp.up_proj.weight", vec![8, 4]),
            ("model.layers.0.mlp.down_proj.weight", vec![4, 8]),
        ];
        let included = tensors
            .into_iter()
            .filter(|(name, _)| Some(*name) != omitted_tensor)
            .collect::<Vec<_>>();
        let mut header = BTreeMap::new();
        let mut offset = 0;
        for (tensor_name, shape) in &included {
            let byte_len = shape.iter().product::<usize>() * std::mem::size_of::<f32>();
            header.insert(
                *tensor_name,
                json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [offset, offset + byte_len]
                }),
            );
            offset += byte_len;
        }
        let header = serde_json::to_vec(&header).expect("safetensors header should serialize");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.resize(bytes.len() + offset, 0);
        fs::write(model_dir.join("model.safetensors"), bytes)
            .expect("safetensors should be written");
    }

    fn temp_model_dir(name: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        std::env::temp_dir().join(format!("sglang-rs-{name}-{}-{nonce}", std::process::id()))
    }
}
