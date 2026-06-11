use std::fs;
use std::path::{Path, PathBuf};

use sglang_srt::glm_runtime::{
    GlmMoeDsaF32KernelError, GlmMoeDsaFeedForwardTensorDescriptors, GlmMoeDsaRuntime,
    GlmMoeDsaTensorPlacementKind, GlmMoeDsaTensorShardSelection,
};
use sglang_srt::model_artifacts::LocalModelArtifacts;
use sglang_srt::model_artifacts::SafetensorsTensorDecodeError;
use sglang_srt::transfer::KvCacheDtype;

#[test]
fn glm_moe_dsa_runtime_builds_tensor_parallel_placement_plan() {
    let model_dir = temp_model_dir("glm-runtime-placement-plan");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_complete_glm_moe_dsa_checkpoint(&model_dir);

    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let runtime =
        GlmMoeDsaRuntime::from_local_model_artifacts(&artifacts).expect("runtime should build");

    assert_eq!(runtime.layer_count(), 1);
    assert_eq!(
        runtime
            .kv_cache_layout()
            .token_size_bytes(KvCacheDtype::Fp8E4M3)
            .expect("GLM multi-tensor layout should size"),
        8
    );
    assert_eq!(
        runtime.root_tensors().lm_head().tensor_name(),
        "lm_head.weight"
    );

    let layer0 = runtime.layers().first().expect("layer 0 should exist");
    let GlmMoeDsaFeedForwardTensorDescriptors::Moe { routed_experts, .. } = layer0.feed_forward()
    else {
        panic!("fixture layer should use MoE feed-forward descriptors");
    };
    assert_eq!(routed_experts[0].expert_id(), 0);
    assert_eq!(
        routed_experts[0].gate().tensor_name(),
        "model.layers.0.mlp.experts.0.gate_proj.weight"
    );

    let placement = runtime.tensor_parallel_placement_plan(2);
    assert_eq!(placement.tensor_parallel_size(), 2);
    assert_eq!(
        placement.kind_for("lm_head.weight"),
        Some(GlmMoeDsaTensorPlacementKind::VocabParallel { axis: 0 })
    );
    assert_eq!(
        placement.kind_for("model.layers.0.self_attn.q_b_proj.weight"),
        Some(GlmMoeDsaTensorPlacementKind::ColumnParallel { axis: 0 })
    );
    assert_eq!(
        placement.kind_for("model.layers.0.self_attn.o_proj.weight"),
        Some(GlmMoeDsaTensorPlacementKind::RowParallel { axis: 1 })
    );
    assert_eq!(
        placement.kind_for("model.layers.0.mlp.gate.weight"),
        Some(GlmMoeDsaTensorPlacementKind::Replicated)
    );

    let rank1 = placement
        .rank_shard_plan(1)
        .expect("rank 1 shard plan should build");
    assert_eq!(
        rank1.selection_for("lm_head.weight"),
        Some(GlmMoeDsaTensorShardSelection::Slice {
            axis: 0,
            range: 2..4
        })
    );
    assert_eq!(
        rank1.selection_for("model.layers.0.self_attn.o_proj.weight"),
        Some(GlmMoeDsaTensorShardSelection::Slice {
            axis: 1,
            range: 2..4
        })
    );
    assert_eq!(
        rank1.selection_for("model.layers.0.mlp.gate.weight"),
        Some(GlmMoeDsaTensorShardSelection::Full)
    );

    let loaded = runtime
        .load_tensor_parallel_shards(2)
        .expect("GLM runtime should load tensor-parallel shards");
    assert_eq!(loaded.rank_count(), 2);
    assert_eq!(loaded.layer_count(), 1);
    let rank1_lm_head = loaded
        .rank(1)
        .and_then(|rank| rank.tensor_shard("lm_head.weight"))
        .expect("rank 1 lm_head shard should be loaded");
    assert_eq!(rank1_lm_head.shape(), &[2, 4]);
    assert_eq!(
        rank1_lm_head.selection(),
        &GlmMoeDsaTensorShardSelection::Slice {
            axis: 0,
            range: 2..4
        }
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn glm_moe_dsa_f32_rank_computes_vocab_parallel_lm_head_partial_logits() {
    let model_dir = temp_model_dir("glm-runtime-tp-lm-head-logits");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_complete_glm_moe_dsa_checkpoint_with_dtype(&model_dir, "F32");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let runtime =
        GlmMoeDsaRuntime::from_local_model_artifacts(&artifacts).expect("runtime should build");
    let f32_runtime = runtime
        .load_tensor_parallel_shards(2)
        .expect("all TP rank shards should load")
        .decode_f32_tensor_parallel_shards()
        .expect("F32 cache should decode");

    let rank1_logits = f32_runtime
        .rank(1)
        .expect("rank 1 cache should exist")
        .lm_head_partial_logits(&[1.0, 0.5, 0.25, 0.125])
        .expect("rank 1 lm_head logits should compute");

    assert_eq!(rank1_logits, vec![(2, 53.875), (3, 61.375)]);

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn glm_moe_dsa_f32_rank_rejects_lm_head_hidden_size_mismatch() {
    let model_dir = temp_model_dir("glm-runtime-tp-lm-head-hidden-mismatch");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_complete_glm_moe_dsa_checkpoint_with_dtype(&model_dir, "F32");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let runtime =
        GlmMoeDsaRuntime::from_local_model_artifacts(&artifacts).expect("runtime should build");
    let f32_runtime = runtime
        .load_tensor_parallel_shards(2)
        .expect("all TP rank shards should load")
        .decode_f32_tensor_parallel_shards()
        .expect("F32 cache should decode");

    let error = f32_runtime
        .rank(1)
        .expect("rank 1 cache should exist")
        .lm_head_partial_logits(&[1.0])
        .expect_err("hidden size mismatch should be rejected");

    assert_eq!(
        error,
        GlmMoeDsaF32KernelError::HiddenSizeMismatch {
            tensor_name: "lm_head.weight".to_string(),
            expected: 4,
            actual: 1
        }
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn glm_moe_dsa_loaded_runtime_reports_unsupported_dtype_when_decoding_f32_cache() {
    let model_dir = temp_model_dir("glm-runtime-tp-f32-cache-u8");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_complete_glm_moe_dsa_checkpoint(&model_dir);
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let runtime =
        GlmMoeDsaRuntime::from_local_model_artifacts(&artifacts).expect("runtime should build");
    let loaded = runtime
        .load_tensor_parallel_shards(1)
        .expect("TP rank shards should load");

    let error = loaded
        .decode_f32_tensor_parallel_shards()
        .expect_err("U8 runtime should not decode into f32 cache");

    assert_eq!(
        error,
        SafetensorsTensorDecodeError::UnsupportedDtype {
            dtype: "U8".to_string()
        }
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

fn write_complete_glm_moe_dsa_checkpoint(model_dir: &Path) {
    write_complete_glm_moe_dsa_checkpoint_with_dtype(model_dir, "U8");
}

fn write_complete_glm_moe_dsa_checkpoint_with_dtype(model_dir: &Path, dtype: &str) {
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "glm_moe_dsa",
  "num_hidden_layers": 1,
  "hidden_size": 4,
  "num_attention_heads": 2,
  "num_key_value_heads": 2,
  "head_dim": 2,
  "n_routed_experts": 1,
  "first_k_dense_replace": 0,
  "moe_layer_freq": 1
}"#,
    )
    .expect("config should be written");
    let tensors = complete_glm_moe_dsa_tensor_shapes()
        .into_iter()
        .scan(0_usize, |offset, (name, shape)| {
            let element_count = shape.iter().product::<usize>();
            let byte_len = match dtype {
                "F32" => element_count * 4,
                "U8" => element_count,
                _ => panic!("test fixture dtype {dtype} is not supported"),
            };
            let tensor = (name, dtype, shape, [*offset, *offset + byte_len]);
            *offset += byte_len;
            Some(tensor)
        })
        .collect::<Vec<_>>();
    let payload_len = tensors
        .iter()
        .map(|(_, _, _, offsets)| offsets[1])
        .max()
        .unwrap_or(0);
    let payload = match dtype {
        "F32" => (0..payload_len / 4)
            .flat_map(|index| (index as f32).to_le_bytes())
            .collect::<Vec<_>>(),
        "U8" => (0..payload_len)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>(),
        _ => panic!("test fixture dtype {dtype} is not supported"),
    };

    write_safetensors_file(&model_dir.join("model.safetensors"), &tensors, &payload)
        .expect("safetensors shard should be written");
}

fn complete_glm_moe_dsa_tensor_shapes() -> Vec<(&'static str, Vec<usize>)> {
    vec![
        ("model.embed_tokens.weight", vec![4, 4]),
        ("model.norm.weight", vec![4]),
        ("lm_head.weight", vec![4, 4]),
        ("model.layers.0.self_attn.q_a_proj.weight", vec![4, 4]),
        ("model.layers.0.self_attn.q_a_layernorm.weight", vec![4]),
        ("model.layers.0.self_attn.q_b_proj.weight", vec![4, 4]),
        (
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            vec![4, 4],
        ),
        ("model.layers.0.self_attn.kv_a_layernorm.weight", vec![4]),
        ("model.layers.0.self_attn.kv_b_proj.weight", vec![4, 4]),
        ("model.layers.0.self_attn.o_proj.weight", vec![4, 4]),
        ("model.layers.0.input_layernorm.weight", vec![4]),
        ("model.layers.0.post_attention_layernorm.weight", vec![4]),
        ("model.layers.0.mlp.gate.weight", vec![1, 4]),
        ("model.layers.0.mlp.experts.0.gate_proj.weight", vec![4, 4]),
        ("model.layers.0.mlp.experts.0.down_proj.weight", vec![4, 4]),
        ("model.layers.0.mlp.experts.0.up_proj.weight", vec![4, 4]),
    ]
}

fn write_safetensors_file(
    path: &Path,
    tensors: &[(&str, &str, Vec<usize>, [usize; 2])],
    payload: &[u8],
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
    bytes.extend_from_slice(payload);
    fs::write(path, bytes)
}

fn temp_model_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("sglang-rs-{name}-{}", std::process::id()))
}
