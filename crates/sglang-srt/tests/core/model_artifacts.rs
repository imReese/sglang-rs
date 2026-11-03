use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};
use sglang_srt::model_artifacts::{
    CheckpointTopology, HfModelConfig, LocalModelArtifacts, ModelArtifactError,
    SafetensorsRoutedExpertProjectionWeights,
};
use sglang_srt::model_registry::ModelRegistry;

#[test]
fn hf_config_keeps_routing_fields_and_raw_model_document() {
    let model_dir = temp_model_dir("config");
    write_json(
        &model_dir.join("config.json"),
        &json!({
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4
        }),
    );

    let config = HfModelConfig::from_model_path(&model_dir).expect("config should load");

    assert_eq!(config.architectures, vec!["Qwen3ForCausalLM"]);
    assert_eq!(config.model_type.as_deref(), Some("qwen3"));
    assert_eq!(config.raw_document()["num_hidden_layers"].as_u64(), Some(2));
    assert_eq!(
        config.raw_document()["num_key_value_heads"].as_u64(),
        Some(1)
    );
}

#[test]
fn checkpoint_catalog_exposes_generic_layer_tensor_spans() {
    let model_dir = temp_model_dir("layer-catalog");
    write_json(
        &model_dir.join("config.json"),
        &json!({
            "architectures": ["UnknownForCausalLM"],
            "model_type": "unknown",
            "num_hidden_layers": 1
        }),
    );
    write_safetensors(
        &model_dir.join("model.safetensors"),
        &[
            ("model.embed_tokens.weight".to_string(), vec![4, 2]),
            (
                "model.layers.0.self_attn.q_proj.weight".to_string(),
                vec![2, 2],
            ),
            ("model.layers.0.input_layernorm.weight".to_string(), vec![2]),
        ],
    );

    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local safetensors should load");
    let catalog = artifacts
        .checkpoint_catalog()
        .expect("generic checkpoint catalog should build");

    assert_eq!(catalog.model_path(), model_dir.as_path());
    assert_eq!(catalog.layer_tensors().tensor_count(), 2);
    let q_proj = catalog
        .layer_tensors()
        .span(0, "self_attn.q_proj.weight")
        .expect("layer tensor should be indexed");
    assert_eq!(q_proj.span.metadata.shape, vec![2, 2]);
}

#[test]
fn routed_expert_catalog_groups_complete_compressed_tensors_projections() {
    let model_dir = temp_model_dir("compressed-routed-expert-catalog");
    write_json(
        &model_dir.join("config.json"),
        &json!({
            "architectures": ["UnknownForCausalLM"],
            "model_type": "unknown"
        }),
    );
    let mut tensors = Vec::new();
    for projection in ["w1", "w2", "w3"] {
        let base = format!("model.layers.0.mlp.experts.0.{projection}");
        tensors.extend([
            (
                format!("{base}.weight_packed"),
                "I32",
                vec![1, 1],
                0x8888_8888_u32.to_le_bytes().to_vec(),
            ),
            (
                format!("{base}.weight_scale"),
                "F32",
                vec![1, 1],
                1.0_f32.to_le_bytes().to_vec(),
            ),
            (
                format!("{base}.weight_shape"),
                "I64",
                vec![2],
                [1_i64, 8].into_iter().flat_map(i64::to_le_bytes).collect(),
            ),
        ]);
    }
    write_typed_safetensors(&model_dir.join("model.safetensors"), &tensors);

    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("compressed fixture should load");
    let catalog = artifacts
        .routed_expert_weight_catalog()
        .expect("compressed routed-expert catalog should build");
    let group = catalog.group(0, 0).expect("layer 0 expert 0 group");
    assert_eq!(group.tensor_count(), 9);
    assert!(matches!(
        group.gate,
        SafetensorsRoutedExpertProjectionWeights::CompressedTensorsInt4(_)
    ));
    let coverage = artifacts
        .validate_routed_expert_checkpoint_coverage(
            &CheckpointTopology::new(Vec::new()).with_routed_experts([(0, 0)], 9),
        )
        .expect("compressed coverage should count all packed components");
    assert_eq!(coverage.actual_weight_count, 9);
}

#[test]
fn glm_and_deepseek_adapters_validate_their_own_weight_mappings() {
    for (name, architecture, model_type, family) in [
        (
            "glm",
            "GlmMoeDsaForCausalLM",
            "glm_moe_dsa",
            MlaFixtureFamily::Glm,
        ),
        (
            "deepseek",
            "DeepseekV4ForCausalLM",
            "deepseek_v4",
            MlaFixtureFamily::DeepSeek,
        ),
    ] {
        let model_dir = temp_model_dir(name);
        write_mla_fixture(&model_dir, architecture, model_type, family, false);
        let artifacts = LocalModelArtifacts::from_model_path(&model_dir)
            .expect("fixture artifacts should load");

        ModelRegistry
            .validate_checkpoint(&artifacts)
            .expect("model adapter should validate its weight mapping");
    }
}

#[test]
fn deepseek_adapter_rejects_missing_model_specific_tensor() {
    let model_dir = temp_model_dir("deepseek-missing");
    write_mla_fixture(
        &model_dir,
        "DeepseekV4ForCausalLM",
        "deepseek_v4",
        MlaFixtureFamily::DeepSeek,
        true,
    );
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("fixture artifacts should load");

    let error = ModelRegistry
        .validate_checkpoint(&artifacts)
        .expect_err("missing DeepSeek tensor must fail in the adapter");

    assert!(matches!(
        error,
        sglang_srt::model_registry::ModelRegistryError::ModelArtifact(
            ModelArtifactError::InvalidSafetensorsData { .. }
        )
    ));
    assert!(error.to_string().contains("model.hc_head_scale"));
}

#[derive(Clone, Copy)]
enum MlaFixtureFamily {
    Glm,
    DeepSeek,
}

fn write_mla_fixture(
    model_dir: &Path,
    architecture: &str,
    model_type: &str,
    family: MlaFixtureFamily,
    omit_hc_scale: bool,
) {
    let mut config = json!({
        "architectures": [architecture],
        "model_type": model_type,
        "vocab_size": 8,
        "hidden_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "qk_nope_head_dim": 64,
        "qk_rope_head_dim": 64,
        "v_head_dim": 64,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 4,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1
    });
    if matches!(family, MlaFixtureFamily::DeepSeek) {
        config["hc_mult"] = json!(1);
    }
    write_json(&model_dir.join("config.json"), &config);

    let mut tensors = vec![
        ("model.embed_tokens.weight".to_string(), vec![8, 128]),
        ("model.norm.weight".to_string(), vec![128]),
        ("lm_head.weight".to_string(), vec![8, 128]),
    ];
    match family {
        MlaFixtureFamily::Glm => {
            for suffix in [
                "self_attn.q_a_proj.weight",
                "self_attn.q_a_layernorm.weight",
                "self_attn.q_b_proj.weight",
                "self_attn.kv_a_proj_with_mqa.weight",
                "self_attn.kv_a_layernorm.weight",
                "self_attn.kv_b_proj.weight",
                "self_attn.o_proj.weight",
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ] {
                tensors.push((format!("model.layers.0.{suffix}"), vec![1]));
            }
        }
        MlaFixtureFamily::DeepSeek => {
            tensors.extend([
                ("model.hc_head_fn".to_string(), vec![1, 128]),
                ("model.hc_head_base".to_string(), vec![1]),
            ]);
            if !omit_hc_scale {
                tensors.push(("model.hc_head_scale".to_string(), vec![1]));
            }
            for suffix in [
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
                "mlp.gate_up_proj.weight",
                "mlp.down_proj.weight",
            ] {
                tensors.push((format!("model.layers.0.{suffix}"), vec![1]));
            }
        }
    }
    write_safetensors(&model_dir.join("model.safetensors"), &tensors);
}

fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>)]) {
    let mut header = BTreeMap::new();
    let mut data = Vec::new();
    for (name, shape) in tensors {
        let start = data.len();
        let element_count = shape.iter().product::<usize>();
        data.resize(start + element_count * 4, 0);
        header.insert(
            name.clone(),
            json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, data.len()]
            }),
        );
    }

    let mut header_bytes = serde_json::to_vec(&header).expect("header should serialize");
    while header_bytes.len() % 8 != 0 {
        header_bytes.push(b' ');
    }
    let mut bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header_bytes);
    bytes.extend(data);
    fs::write(path, bytes).expect("safetensors fixture should write");
}

fn write_typed_safetensors(path: &Path, tensors: &[(String, &'static str, Vec<usize>, Vec<u8>)]) {
    let mut header = BTreeMap::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = data.len();
        data.extend_from_slice(bytes);
        header.insert(
            name.clone(),
            json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, data.len()]
            }),
        );
    }
    let mut header_bytes = serde_json::to_vec(&header).expect("header should serialize");
    while header_bytes.len() % 8 != 0 {
        header_bytes.push(b' ');
    }
    let mut file = Vec::new();
    file.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    file.extend_from_slice(&header_bytes);
    file.extend_from_slice(&data);
    fs::write(path, file).expect("typed safetensors fixture should be written");
}

fn write_json(path: &Path, value: &Value) {
    fs::write(
        path,
        serde_json::to_vec(value).expect("JSON should serialize"),
    )
    .expect("JSON fixture should write");
}

fn temp_model_dir(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be valid")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("sglang-rs-{name}-{nonce}"));
    fs::create_dir_all(&path).expect("temporary model directory should be created");
    path
}
