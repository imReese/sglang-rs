use std::path::Path;

use crate::backend::RuntimeDtype;
use crate::model_artifacts::{HfModelConfig, LocalModelArtifacts, ModelArtifactError};
use crate::transfer::KvCacheModelLayout;

use super::{
    AttentionArchitecture, FeedForwardArchitecture, ModelAdapter, ModelAdapterError,
    ModelDefinition, ModelExecutionArchitecture, required_usize,
};

pub(crate) const QWEN2_ARCHITECTURE: &str = "Qwen2ForCausalLM";
pub(crate) static QWEN2_ADAPTER: Qwen2Adapter = Qwen2Adapter;

pub(crate) struct Qwen2Adapter;

impl ModelAdapter for Qwen2Adapter {
    fn architectures(&self) -> &'static [&'static str] {
        &[QWEN2_ARCHITECTURE]
    }

    fn build_definition(
        &self,
        _model_path: &Path,
        config: &HfModelConfig,
    ) -> Result<ModelDefinition, ModelAdapterError> {
        let num_layers = required_usize(
            QWEN2_ARCHITECTURE,
            "num_hidden_layers",
            config.num_hidden_layers,
        )?;
        let hidden_size = required_usize(QWEN2_ARCHITECTURE, "hidden_size", config.hidden_size)?;
        let intermediate_size = required_usize(
            QWEN2_ARCHITECTURE,
            "intermediate_size",
            config.intermediate_size,
        )?;
        let num_attention_heads = required_usize(
            QWEN2_ARCHITECTURE,
            "num_attention_heads",
            config.num_attention_heads,
        )?;
        let num_key_value_heads = config.num_key_value_heads.unwrap_or(num_attention_heads);
        if num_key_value_heads == 0 {
            return Err(ModelAdapterError::invalid(
                QWEN2_ARCHITECTURE,
                "num_key_value_heads must be non-zero",
            ));
        }
        let head_dim = match config.head_dim {
            Some(head_dim) if head_dim > 0 => head_dim,
            Some(_) => {
                return Err(ModelAdapterError::invalid(
                    QWEN2_ARCHITECTURE,
                    "head_dim must be non-zero",
                ));
            }
            None if hidden_size.is_multiple_of(num_attention_heads) => {
                hidden_size / num_attention_heads
            }
            None => {
                return Err(ModelAdapterError::invalid(
                    QWEN2_ARCHITECTURE,
                    format!(
                        "hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
                    ),
                ));
            }
        };

        Ok(ModelDefinition::new(
            QWEN2_ARCHITECTURE,
            config,
            ModelExecutionArchitecture::Transformer {
                attention: AttentionArchitecture::MultiHead {
                    num_attention_heads,
                    num_key_value_heads,
                    head_dim,
                },
                feed_forward: FeedForwardArchitecture::Dense { intermediate_size },
            },
            RuntimeDtype::Bf16,
            Some(KvCacheModelLayout::multi_tensor(
                num_layers,
                num_key_value_heads,
                head_dim,
                2,
            )),
        ))
    }

    fn validate_checkpoint(
        &self,
        artifacts: &LocalModelArtifacts,
    ) -> Result<(), ModelArtifactError> {
        let config = artifacts.config();
        let num_layers = config.num_hidden_layers.ok_or_else(|| {
            invalid_checkpoint(artifacts, "Qwen2 config is missing num_hidden_layers")
        })?;
        let mut required = vec!["model.embed_tokens.weight", "model.norm.weight"];
        if config.tie_word_embeddings != Some(true) {
            required.push("lm_head.weight");
        }
        for tensor_name in required {
            require_tensor(artifacts, tensor_name)?;
        }

        const LAYER_TENSORS: &[&str] = &[
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ];
        for layer_id in 0..num_layers {
            for suffix in LAYER_TENSORS {
                require_tensor(artifacts, &format!("model.layers.{layer_id}.{suffix}"))?;
            }
        }
        Ok(())
    }
}

fn require_tensor(
    artifacts: &LocalModelArtifacts,
    tensor_name: &str,
) -> Result<(), ModelArtifactError> {
    if artifacts
        .safetensors()
        .tensor_metadata(tensor_name)?
        .is_none()
    {
        return Err(invalid_checkpoint(
            artifacts,
            format!("missing Qwen2 checkpoint tensor {tensor_name}"),
        ));
    }
    Ok(())
}

fn invalid_checkpoint(
    artifacts: &LocalModelArtifacts,
    message: impl Into<String>,
) -> ModelArtifactError {
    ModelArtifactError::InvalidSafetensorsData {
        path: artifacts.model_path().to_path_buf(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::json;

    use super::*;

    #[test]
    fn qwen_adapter_validates_real_dense_decoder_weight_names() {
        let model_dir = temp_model_dir("qwen-complete-checkpoint");
        write_qwen_checkpoint(&model_dir, None);
        let artifacts =
            LocalModelArtifacts::from_model_path(&model_dir).expect("Qwen artifacts should load");

        QWEN2_ADAPTER
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

        let error = QWEN2_ADAPTER
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
  "vocab_size": 2,
  "num_hidden_layers": 1,
  "hidden_size": 2,
  "intermediate_size": 4,
  "num_attention_heads": 2,
  "num_key_value_heads": 1,
  "tie_word_embeddings": false
}"#,
        )
        .expect("config should be written");

        let tensor_names = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ];
        let included = tensor_names
            .into_iter()
            .filter(|name| Some(*name) != omitted_tensor)
            .collect::<Vec<_>>();
        let mut header = BTreeMap::new();
        for (index, tensor_name) in included.iter().enumerate() {
            let start = index * std::mem::size_of::<f32>();
            header.insert(
                *tensor_name,
                json!({
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [start, start + std::mem::size_of::<f32>()]
                }),
            );
        }
        let header = serde_json::to_vec(&header).expect("safetensors header should serialize");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.resize(bytes.len() + included.len() * std::mem::size_of::<f32>(), 0);
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
