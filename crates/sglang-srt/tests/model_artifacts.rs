use std::fs;
use std::path::PathBuf;

use sglang_srt::model_artifacts::{
    LocalModelArtifacts, ModelArtifactError, SafetensorsCheckpointFingerprintEntry,
    SafetensorsTensorData, SafetensorsTensorMetadata, SafetensorsTensorSpan,
};

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
fn safetensors_manifest_describes_indexed_tensor_byte_span_without_loading_bytes() {
    let model_dir = temp_model_dir("tensor-span");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(model_dir.join("config.json"), deepseek_v4_config_json())
        .expect("config should be written");
    fs::write(
        model_dir.join("model.safetensors.index.json"),
        safetensors_index_json(),
    )
    .expect("index should be written");
    let shard_path = model_dir.join("model-00001-of-00002.safetensors");
    let header_len = write_safetensors_file(
        &shard_path,
        &[("model.layers.0.ffn.experts.3.w1.weight", "U8", &[4], [2, 6])],
        &[9, 8, 1, 2, 3, 4, 7],
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

    assert_eq!(
        artifacts
            .safetensors()
            .tensor_span("model.layers.0.ffn.experts.3.w1.weight")
            .expect("indexed tensor span should read"),
        Some(SafetensorsTensorSpan {
            path: shard_path,
            metadata: SafetensorsTensorMetadata {
                dtype: "U8".to_string(),
                shape: vec![4],
                data_offsets: [2, 6],
            },
            absolute_byte_offset: 8 + header_len as u64 + 2,
            byte_len: 4,
        })
    );
    assert_eq!(
        artifacts
            .safetensors()
            .tensor_span("model.layers.404.mlp.down_proj.weight")
            .expect("unknown tensor should not read a shard"),
        None
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn safetensors_manifest_enumerates_indexed_tensor_spans_across_shards() {
    let model_dir = temp_model_dir("tensor-span-entries");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(model_dir.join("config.json"), deepseek_v4_config_json())
        .expect("config should be written");
    fs::write(
        model_dir.join("model.safetensors.index.json"),
        safetensors_index_json(),
    )
    .expect("index should be written");
    let first_shard = model_dir.join("model-00001-of-00002.safetensors");
    let first_header_len = write_safetensors_file(
        &first_shard,
        &[
            ("model.embed_tokens.weight", "U8", &[2], [0, 2]),
            ("model.layers.0.ffn.experts.3.w1.weight", "U8", &[3], [2, 5]),
            (
                "model.layers.0.self_attn.q_a_proj.weight",
                "U8",
                &[1],
                [5, 6],
            ),
        ],
        &[10, 11, 20, 21, 22, 30],
    )
    .expect("first shard should be written");
    let second_shard = model_dir.join("model-00002-of-00002.safetensors");
    let second_header_len = write_safetensors_file(
        &second_shard,
        &[(
            "model.layers.0.self_attn.q_b_proj.weight",
            "BF16",
            &[2],
            [0, 4],
        )],
        &[40, 41, 42, 43],
    )
    .expect("second shard should be written");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local artifacts should load");

    let entries = artifacts
        .safetensors()
        .tensor_span_entries()
        .expect("indexed tensor span entries should read");

    assert_eq!(
        entries
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        vec![
            "model.embed_tokens.weight",
            "model.layers.0.ffn.experts.3.w1.weight",
            "model.layers.0.self_attn.q_a_proj.weight",
            "model.layers.0.self_attn.q_b_proj.weight",
        ]
    );
    assert_eq!(entries[0].1.path, first_shard);
    assert_eq!(
        entries[0].1.absolute_byte_offset,
        8 + first_header_len as u64
    );
    assert_eq!(entries[0].1.byte_len, 2);
    assert_eq!(
        entries[1].1.absolute_byte_offset,
        8 + first_header_len as u64 + 2
    );
    assert_eq!(entries[1].1.byte_len, 3);
    assert_eq!(entries[3].1.path, second_shard);
    assert_eq!(
        entries[3].1.absolute_byte_offset,
        8 + second_header_len as u64
    );
    assert_eq!(
        entries[1]
            .1
            .read()
            .expect("span should read bytes")
            .expect("span should load tensor")
            .bytes,
        vec![20, 21, 22]
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn safetensors_manifest_reads_indexed_tensor_payload_bytes() {
    let model_dir = temp_model_dir("tensor-payload");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(model_dir.join("config.json"), deepseek_v4_config_json())
        .expect("config should be written");
    fs::write(
        model_dir.join("model.safetensors.index.json"),
        safetensors_index_json(),
    )
    .expect("index should be written");
    write_safetensors_file(
        &model_dir.join("model-00001-of-00002.safetensors"),
        &[("model.layers.0.ffn.experts.3.w1.weight", "U8", &[4], [2, 6])],
        &[9, 8, 1, 2, 3, 4, 7],
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

    assert_eq!(
        artifacts
            .safetensors()
            .read_tensor("model.layers.0.ffn.experts.3.w1.weight")
            .expect("indexed tensor payload should read"),
        Some(SafetensorsTensorData {
            metadata: SafetensorsTensorMetadata {
                dtype: "U8".to_string(),
                shape: vec![4],
                data_offsets: [2, 6],
            },
            bytes: vec![1, 2, 3, 4],
        })
    );
    let tensor = artifacts
        .safetensors()
        .read_tensor("model.layers.0.ffn.experts.3.w1.weight")
        .expect("indexed tensor payload should read")
        .expect("indexed tensor should exist");
    assert_eq!(tensor.element_count(), 4);
    assert_eq!(tensor.dtype_byte_width(), 1);
    assert_eq!(
        artifacts
            .safetensors()
            .read_tensor("model.layers.404.mlp.down_proj.weight")
            .expect("unknown tensor should not read a shard"),
        None
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn safetensors_tensor_span_computes_checksum_over_payload_slice() {
    let model_dir = temp_model_dir("tensor-span-checksum");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(model_dir.join("config.json"), deepseek_v4_config_json())
        .expect("config should be written");
    fs::write(
        model_dir.join("model.safetensors.index.json"),
        safetensors_index_json(),
    )
    .expect("index should be written");
    write_safetensors_file(
        &model_dir.join("model-00001-of-00002.safetensors"),
        &[("model.layers.0.ffn.experts.3.w1.weight", "U8", &[5], [1, 6])],
        &[99, 1, 2, 3, 4, 5, 88],
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
    let span = artifacts
        .safetensors()
        .tensor_span("model.layers.0.ffn.experts.3.w1.weight")
        .expect("indexed tensor span should read")
        .expect("indexed tensor should exist");

    assert_eq!(
        span.fnv1a64_checksum()
            .expect("span checksum should stream payload"),
        fnv1a64(&[1, 2, 3, 4, 5])
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn safetensors_manifest_builds_stable_checkpoint_fingerprint_entries() {
    let model_dir = temp_model_dir("checkpoint-fingerprint");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(model_dir.join("config.json"), deepseek_v4_config_json())
        .expect("config should be written");
    fs::write(
        model_dir.join("model.safetensors.index.json"),
        safetensors_index_json(),
    )
    .expect("index should be written");
    let first_shard = model_dir.join("model-00001-of-00002.safetensors");
    let first_header_len = write_safetensors_file(
        &first_shard,
        &[
            ("model.embed_tokens.weight", "U8", &[2], [0, 2]),
            ("model.layers.0.ffn.experts.3.w1.weight", "U8", &[3], [2, 5]),
            (
                "model.layers.0.self_attn.q_a_proj.weight",
                "U8",
                &[1],
                [5, 6],
            ),
        ],
        &[10, 11, 20, 21, 22, 30],
    )
    .expect("first shard should be written");
    let second_shard = model_dir.join("model-00002-of-00002.safetensors");
    let second_header_len = write_safetensors_file(
        &second_shard,
        &[(
            "model.layers.0.self_attn.q_b_proj.weight",
            "BF16",
            &[2],
            [0, 4],
        )],
        &[40, 41, 42, 43],
    )
    .expect("second shard should be written");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local artifacts should load");

    assert_eq!(
        artifacts
            .safetensors()
            .checkpoint_fingerprint_entries()
            .expect("checkpoint fingerprint should stream tensor spans"),
        vec![
            SafetensorsCheckpointFingerprintEntry {
                tensor_name: "model.embed_tokens.weight".to_string(),
                path: first_shard.clone(),
                dtype: "U8".to_string(),
                shape: vec![2],
                absolute_byte_offset: 8 + first_header_len as u64,
                byte_len: 2,
                fnv1a64: fnv1a64(&[10, 11]),
            },
            SafetensorsCheckpointFingerprintEntry {
                tensor_name: "model.layers.0.ffn.experts.3.w1.weight".to_string(),
                path: first_shard.clone(),
                dtype: "U8".to_string(),
                shape: vec![3],
                absolute_byte_offset: 8 + first_header_len as u64 + 2,
                byte_len: 3,
                fnv1a64: fnv1a64(&[20, 21, 22]),
            },
            SafetensorsCheckpointFingerprintEntry {
                tensor_name: "model.layers.0.self_attn.q_a_proj.weight".to_string(),
                path: first_shard,
                dtype: "U8".to_string(),
                shape: vec![1],
                absolute_byte_offset: 8 + first_header_len as u64 + 5,
                byte_len: 1,
                fnv1a64: fnv1a64(&[30]),
            },
            SafetensorsCheckpointFingerprintEntry {
                tensor_name: "model.layers.0.self_attn.q_b_proj.weight".to_string(),
                path: second_shard,
                dtype: "BF16".to_string(),
                shape: vec![2],
                absolute_byte_offset: 8 + second_header_len as u64,
                byte_len: 4,
                fnv1a64: fnv1a64(&[40, 41, 42, 43]),
            },
        ]
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn safetensors_manifest_rejects_tensor_payload_length_mismatching_shape_and_dtype() {
    let model_dir = temp_model_dir("tensor-payload-bad-shape-len");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(model_dir.join("config.json"), deepseek_v4_config_json())
        .expect("config should be written");
    fs::write(
        model_dir.join("model.safetensors.index.json"),
        safetensors_index_json(),
    )
    .expect("index should be written");
    let bad_shard = model_dir.join("model-00001-of-00002.safetensors");
    write_safetensors_file(
        &bad_shard,
        &[(
            "model.layers.0.ffn.experts.3.w1.weight",
            "BF16",
            &[4],
            [0, 4],
        )],
        &[1, 2, 3, 4],
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

    let error = artifacts
        .safetensors()
        .read_tensor("model.layers.0.ffn.experts.3.w1.weight")
        .expect_err("payload length mismatch should be rejected");

    assert!(
        matches!(
            error,
            ModelArtifactError::InvalidSafetensorsData { ref path, ref message }
                if path == &bad_shard
                    && message.contains("metadata expects 8 bytes")
                    && message.contains("data_offsets describe 4 bytes")
        ),
        "unexpected error: {error:?}"
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn safetensors_manifest_rejects_tensor_payload_offsets_past_shard_end() {
    let model_dir = temp_model_dir("tensor-payload-past-end");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(model_dir.join("config.json"), deepseek_v4_config_json())
        .expect("config should be written");
    fs::write(
        model_dir.join("model.safetensors.index.json"),
        safetensors_index_json(),
    )
    .expect("index should be written");
    let bad_shard = model_dir.join("model-00001-of-00002.safetensors");
    write_safetensors_file(
        &bad_shard,
        &[("model.layers.0.ffn.experts.3.w1.weight", "U8", &[8], [0, 8])],
        &[1, 2, 3, 4],
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

    let error = artifacts
        .safetensors()
        .read_tensor("model.layers.0.ffn.experts.3.w1.weight")
        .expect_err("out-of-bounds tensor payload should be rejected");

    assert!(
        matches!(
            error,
            ModelArtifactError::InvalidSafetensorsData { ref path, ref message }
                if path == &bad_shard && message.contains("extends past end of shard")
        ),
        "unexpected error: {error:?}"
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

#[test]
fn safetensors_manifest_finds_tensor_spans_in_unindexed_shards() {
    let model_dir = temp_model_dir("unindexed-tensor-spans");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(model_dir.join("config.json"), deepseek_v4_config_json())
        .expect("config should be written");
    let first_shard = model_dir.join("model-00001.safetensors");
    let first_header_len = write_safetensors_file(
        &first_shard,
        &[
            (
                "model.layers.1.mlp.experts.42.down_proj.weight",
                "U8",
                &[3],
                [0, 3],
            ),
            ("model.norm.weight", "BF16", &[2], [3, 7]),
        ],
        &[1, 2, 3, 4, 5, 6, 7],
    )
    .expect("first shard should be written");
    let second_shard = model_dir.join("model-00002.safetensors");
    let second_header_len = write_safetensors_file(
        &second_shard,
        &[("lm_head.weight", "U8", &[2], [0, 2])],
        &[8, 9],
    )
    .expect("second shard should be written");
    let artifacts = LocalModelArtifacts::from_model_path(&model_dir)
        .expect("unindexed safetensors shards should load");

    assert_eq!(
        artifacts
            .safetensors()
            .tensor_span("model.norm.weight")
            .expect("unindexed tensor span should read"),
        Some(SafetensorsTensorSpan {
            path: first_shard.clone(),
            metadata: SafetensorsTensorMetadata {
                dtype: "BF16".to_string(),
                shape: vec![2],
                data_offsets: [3, 7],
            },
            absolute_byte_offset: 8 + first_header_len as u64 + 3,
            byte_len: 4,
        })
    );
    let entries = artifacts
        .safetensors()
        .tensor_span_entries()
        .expect("unindexed tensor span entries should read");
    assert_eq!(
        entries
            .iter()
            .map(|(name, span)| (name.as_str(), span.path.clone(), span.absolute_byte_offset))
            .collect::<Vec<_>>(),
        vec![
            (
                "model.layers.1.mlp.experts.42.down_proj.weight",
                first_shard,
                8 + first_header_len as u64,
            ),
            (
                "model.norm.weight",
                model_dir.join("model-00001.safetensors"),
                8 + first_header_len as u64 + 3,
            ),
            ("lm_head.weight", second_shard, 8 + second_header_len as u64,),
        ]
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
) -> std::io::Result<usize> {
    write_safetensors_file(path, tensors, &[])
}

fn write_safetensors_file(
    path: &std::path::Path,
    tensors: &[(&str, &str, &[usize], [usize; 2])],
    payload: &[u8],
) -> std::io::Result<usize> {
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
    fs::write(path, bytes)?;
    Ok(header.len())
}

fn temp_model_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "sglang-rs-model-artifacts-{name}-{}",
        std::process::id()
    ))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}
