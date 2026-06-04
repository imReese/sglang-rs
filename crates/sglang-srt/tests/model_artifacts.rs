use std::fs;
use std::path::PathBuf;

use sglang_srt::model_artifacts::{LocalModelArtifacts, ModelArtifactError};

#[test]
fn local_model_artifacts_loads_deepseek_v4_config_and_indexed_safetensors() {
    let model_dir = temp_model_dir("indexed-safetensors");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(model_dir.join("config.json"), deepseek_v4_config_json())
        .expect("config should be written");
    fs::write(
        model_dir.join("model.safetensors.index.json"),
        safetensors_index_json(),
    )
    .expect("index should be written");
    write_safetensors_header(
        &model_dir.join("model-00001-of-00002.safetensors"),
        &[(
            "model.layers.0.ffn.experts.3.w1.weight",
            "F8_E4M3",
            &[16, 128],
            [0, 2048],
        )],
    )
    .expect("first shard should be written");
    write_safetensors_header(
        &model_dir.join("model-00002-of-00002.safetensors"),
        &[(
            "model.layers.0.self_attn.q_b_proj.weight",
            "BF16",
            &[128, 128],
            [0, 32768],
        )],
    )
    .expect("second shard should be written");

    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local artifacts should load");

    assert_eq!(artifacts.model_path(), model_dir.as_path());
    assert_eq!(
        artifacts.config().model_type.as_deref(),
        Some("deepseek_v4")
    );
    assert_eq!(artifacts.config().vocab_size, Some(129_280));
    assert_eq!(artifacts.config().max_position_embeddings, Some(163_840));
    assert_eq!(artifacts.config().num_hidden_layers, Some(43));
    assert_eq!(
        artifacts.safetensors().tensor_names(),
        &[
            "model.embed_tokens.weight".to_string(),
            "model.layers.0.ffn.experts.3.w1.weight".to_string(),
            "model.layers.0.self_attn.q_a_proj.weight".to_string(),
            "model.layers.0.self_attn.q_b_proj.weight".to_string(),
        ]
    );
    assert_eq!(
        artifacts
            .safetensors()
            .shard_for_tensor("model.layers.0.self_attn.q_b_proj.weight"),
        Some(model_dir.join("model-00002-of-00002.safetensors").as_path())
    );
    assert_eq!(
        artifacts.safetensors().shard_paths(),
        &[
            model_dir.join("model-00001-of-00002.safetensors"),
            model_dir.join("model-00002-of-00002.safetensors"),
        ]
    );
    assert_eq!(
        artifacts
            .safetensors()
            .tensor_metadata("model.layers.0.ffn.experts.3.w1.weight")
            .expect("indexed tensor metadata should read"),
        Some(sglang_srt::model_artifacts::SafetensorsTensorMetadata {
            dtype: "F8_E4M3".to_string(),
            shape: vec![16, 128],
            data_offsets: [0, 2048],
        })
    );
    assert_eq!(
        artifacts
            .safetensors()
            .probe_routed_expert_weight_dtype()
            .expect("routed expert dtype probe should read header"),
        Some("F8_E4M3".to_string())
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn local_model_artifacts_rejects_index_referencing_missing_shard() {
    let model_dir = temp_model_dir("missing-shard");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(model_dir.join("config.json"), deepseek_v4_config_json())
        .expect("config should be written");
    fs::write(
        model_dir.join("model.safetensors.index.json"),
        safetensors_index_json(),
    )
    .expect("index should be written");
    fs::write(model_dir.join("model-00001-of-00002.safetensors"), b"")
        .expect("first shard should be written");

    let error = LocalModelArtifacts::from_model_path(&model_dir)
        .expect_err("missing indexed shard should be rejected");

    assert_eq!(
        error,
        ModelArtifactError::MissingWeightShard {
            path: model_dir.join("model-00002-of-00002.safetensors"),
        }
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn local_model_artifacts_accepts_unindexed_safetensors_shards() {
    let model_dir = temp_model_dir("unindexed-safetensors");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(model_dir.join("config.json"), deepseek_v4_config_json())
        .expect("config should be written");
    write_safetensors_header(
        &model_dir.join("model.safetensors"),
        &[(
            "model.layers.1.mlp.experts.42.down_proj.weight",
            "U8",
            &[8, 16],
            [0, 128],
        )],
    )
    .expect("shard should be written");

    let artifacts = LocalModelArtifacts::from_model_path(&model_dir)
        .expect("single safetensors shard should load without index");

    assert!(artifacts.safetensors().tensor_names().is_empty());
    assert_eq!(
        artifacts.safetensors().shard_paths(),
        &[model_dir.join("model.safetensors")]
    );
    assert_eq!(
        artifacts
            .safetensors()
            .probe_routed_expert_weight_dtype()
            .expect("unindexed routed expert dtype probe should read header"),
        Some("U8".to_string())
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

fn deepseek_v4_config_json() -> &'static str {
    r#"{
  "model_type": "deepseek_v4",
  "architectures": ["DeepSeekV4ForCausalLM"],
  "vocab_size": 129280,
  "max_position_embeddings": 163840,
  "num_hidden_layers": 43,
  "num_key_value_heads": 1,
  "qk_nope_head_dim": 448,
  "qk_rope_head_dim": 64,
  "v_head_dim": 512
}"#
}

fn safetensors_index_json() -> &'static str {
    r#"{
  "metadata": {
    "total_size": 1024
  },
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.ffn.experts.3.w1.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.self_attn.q_a_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.self_attn.q_b_proj.weight": "model-00002-of-00002.safetensors"
  }
}"#
}

fn write_safetensors_header(
    path: &std::path::Path,
    tensors: &[(&str, &str, &[usize], [usize; 2])],
) -> std::io::Result<()> {
    let mut fields = Vec::new();
    for (name, dtype, shape, data_offsets) in tensors {
        let shape = shape
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        fields.push(format!(
            r#""{name}":{{"dtype":"{dtype}","shape":[{shape}],"data_offsets":[{},{}]}}"#,
            data_offsets[0], data_offsets[1]
        ));
    }
    let header = format!("{{{}}}", fields.join(","));
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    fs::write(path, bytes)
}

fn temp_model_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "sglang-rs-model-artifacts-{name}-{}",
        std::process::id()
    ))
}
