use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Mutex;

use sglang_srt::model_artifacts::{
    DeepSeekLayerFeedForwardCheckpointWeights, GlmMoeDsaLayerFeedForwardCheckpointWeights,
    HfModelConfig, LocalModelArtifacts, ModelArtifactError, SafetensorsCheckpointFingerprintEntry,
    SafetensorsLayerTensorCatalog, SafetensorsLayerTensorSpan, SafetensorsQuantizedLinearScaleKind,
    SafetensorsQuantizedLinearWeightSpan, SafetensorsRoutedExpertProjection,
    SafetensorsRoutedExpertWeightCatalog, SafetensorsRoutedExpertWeightGroup,
    SafetensorsRoutedExpertWeightSpan, SafetensorsTensorData, SafetensorsTensorMetadata,
    SafetensorsTensorSpan,
};

static HF_ENV_LOCK: Mutex<()> = Mutex::new(());

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
    assert_eq!(artifacts.config().eos_token_ids, vec![32, 100_001]);
    assert_eq!(artifacts.config().num_hidden_layers, Some(43));
    assert_eq!(artifacts.config().hidden_size, Some(7168));
    assert_eq!(artifacts.config().intermediate_size, Some(18_432));
    assert_eq!(artifacts.config().moe_intermediate_size, Some(2048));
    assert_eq!(artifacts.config().n_routed_experts, Some(256));
    assert_eq!(artifacts.config().n_shared_experts, Some(1));
    assert_eq!(artifacts.config().num_experts_per_tok, Some(8));
    assert_eq!(artifacts.config().first_k_dense_replace, Some(3));
    assert_eq!(artifacts.config().moe_layer_freq, Some(2));
    assert_eq!(artifacts.config().hc_mult, Some(4));
    assert_eq!(artifacts.config().hc_sinkhorn_iters, Some(20));
    assert_eq!(
        artifacts.config().rms_norm_eps.map(|value| value.get()),
        Some(1e-6)
    );
    assert_eq!(
        artifacts.config().rope_theta.map(|value| value.get()),
        Some(1_000_000.0)
    );
    assert_eq!(
        artifacts.config().hc_eps.map(|value| value.get()),
        Some(1e-6)
    );
    assert_eq!(artifacts.config().tie_word_embeddings, Some(false));
    assert_eq!(
        artifacts.config().moe_layer_ids(),
        vec![
            4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42
        ]
    );
    assert!(!artifacts.config().is_moe_layer(3));
    assert!(artifacts.config().is_moe_layer(4));
    assert!(!artifacts.config().is_moe_layer(43));
    assert_eq!(
        artifacts.config().expected_routed_expert_group_count(),
        Some(20 * 256)
    );
    assert_eq!(
        artifacts.config().expected_routed_expert_weight_count(),
        Some(20 * 256 * 3)
    );
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
fn local_model_artifacts_validates_routed_expert_checkpoint_coverage() {
    let model_dir = temp_model_dir("routed-expert-coverage");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "deepseek_v4",
  "num_hidden_layers": 2,
  "n_routed_experts": 2,
  "first_k_dense_replace": 0,
  "moe_layer_freq": 1
}"#,
    )
    .expect("config should be written");
    fs::write(
        model_dir.join("model.safetensors.index.json"),
        r#"{
  "weight_map": {
    "model.layers.0.ffn.experts.0.w1.weight": "model.safetensors",
    "model.layers.0.ffn.experts.0.w2.weight": "model.safetensors",
    "model.layers.0.ffn.experts.0.w3.weight": "model.safetensors",
    "model.layers.0.ffn.experts.1.w1.weight": "model.safetensors",
    "model.layers.0.ffn.experts.1.w2.weight": "model.safetensors",
    "model.layers.0.ffn.experts.1.w3.weight": "model.safetensors",
    "model.layers.1.ffn.experts.0.w1.weight": "model.safetensors",
    "model.layers.1.ffn.experts.0.w2.weight": "model.safetensors",
    "model.layers.1.ffn.experts.0.w3.weight": "model.safetensors",
    "model.layers.1.ffn.experts.1.w1.weight": "model.safetensors",
    "model.layers.1.ffn.experts.1.w2.weight": "model.safetensors",
    "model.layers.1.ffn.experts.1.w3.weight": "model.safetensors"
  }
}"#,
    )
    .expect("index should be written");
    write_safetensors_file(
        &model_dir.join("model.safetensors"),
        &[
            ("model.layers.0.ffn.experts.0.w1.weight", "U8", &[1], [0, 1]),
            ("model.layers.0.ffn.experts.0.w2.weight", "U8", &[1], [1, 2]),
            ("model.layers.0.ffn.experts.0.w3.weight", "U8", &[1], [2, 3]),
            ("model.layers.0.ffn.experts.1.w1.weight", "U8", &[1], [3, 4]),
            ("model.layers.0.ffn.experts.1.w2.weight", "U8", &[1], [4, 5]),
            ("model.layers.0.ffn.experts.1.w3.weight", "U8", &[1], [5, 6]),
            ("model.layers.1.ffn.experts.0.w1.weight", "U8", &[1], [6, 7]),
            ("model.layers.1.ffn.experts.0.w2.weight", "U8", &[1], [7, 8]),
            ("model.layers.1.ffn.experts.0.w3.weight", "U8", &[1], [8, 9]),
            (
                "model.layers.1.ffn.experts.1.w1.weight",
                "U8",
                &[1],
                [9, 10],
            ),
            (
                "model.layers.1.ffn.experts.1.w2.weight",
                "U8",
                &[1],
                [10, 11],
            ),
            (
                "model.layers.1.ffn.experts.1.w3.weight",
                "U8",
                &[1],
                [11, 12],
            ),
        ],
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    )
    .expect("shard should be written");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local artifacts should load");

    let summary = artifacts
        .validate_routed_expert_checkpoint_coverage()
        .expect("complete routed expert checkpoint should validate");

    assert_eq!(summary.expected_group_count, 4);
    assert_eq!(summary.actual_group_count, 4);
    assert_eq!(summary.expected_weight_count, 12);
    assert_eq!(summary.actual_weight_count, 12);

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn local_model_artifacts_builds_routed_expert_weight_catalog_for_lookup() {
    let model_dir = temp_model_dir("routed-expert-weight-catalog");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "deepseek_v4",
  "num_hidden_layers": 2,
  "n_routed_experts": 2,
  "first_k_dense_replace": 0,
  "moe_layer_freq": 1
}"#,
    )
    .expect("config should be written");
    let shard = model_dir.join("model.safetensors");
    let header_len = write_safetensors_file(
        &shard,
        &[
            ("model.layers.0.ffn.experts.0.w1.weight", "U8", &[1], [0, 1]),
            ("model.layers.0.ffn.experts.0.w2.weight", "U8", &[1], [1, 2]),
            ("model.layers.0.ffn.experts.0.w3.weight", "U8", &[1], [2, 3]),
            ("model.layers.0.ffn.experts.1.w1.weight", "U8", &[1], [3, 4]),
            ("model.layers.0.ffn.experts.1.w2.weight", "U8", &[1], [4, 5]),
            ("model.layers.0.ffn.experts.1.w3.weight", "U8", &[1], [5, 6]),
            ("model.layers.1.ffn.experts.0.w1.weight", "U8", &[1], [6, 7]),
            ("model.layers.1.ffn.experts.0.w2.weight", "U8", &[1], [7, 8]),
            ("model.layers.1.ffn.experts.0.w3.weight", "U8", &[1], [8, 9]),
            (
                "model.layers.1.ffn.experts.1.w1.weight",
                "U8",
                &[1],
                [9, 10],
            ),
            (
                "model.layers.1.ffn.experts.1.w2.weight",
                "U8",
                &[1],
                [10, 11],
            ),
            (
                "model.layers.1.ffn.experts.1.w3.weight",
                "U8",
                &[1],
                [11, 12],
            ),
        ],
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    )
    .expect("shard should be written");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local artifacts should load");

    let catalog = SafetensorsRoutedExpertWeightCatalog::from_local_model_artifacts(&artifacts)
        .expect("complete routed expert checkpoint should build catalog");

    assert_eq!(catalog.group_count(), 4);
    assert_eq!(
        catalog.coordinates().collect::<Vec<_>>(),
        vec![(0, 0), (0, 1), (1, 0), (1, 1)]
    );
    assert_eq!(catalog.layer_ids().collect::<Vec<_>>(), vec![0, 1]);
    let layer = catalog
        .layer(1)
        .expect("catalog should expose layer 1 expert weights");
    assert_eq!(layer.layer_id(), 1);
    assert_eq!(layer.expert_count(), 2);
    assert_eq!(layer.expert_ids().collect::<Vec<_>>(), vec![0, 1]);
    assert!(layer.group(0).is_some());
    assert!(layer.group(2).is_none());

    let group = catalog
        .group(1, 1)
        .expect("catalog should look up layer 1 expert 1");
    assert_eq!(group.layer_id, 1);
    assert_eq!(group.expert_id, 1);
    assert_eq!(group.gate.absolute_byte_offset, 8 + header_len as u64 + 9);
    assert_eq!(group.down.absolute_byte_offset, 8 + header_len as u64 + 10);
    assert_eq!(group.up.absolute_byte_offset, 8 + header_len as u64 + 11);
    assert!(catalog.group(9, 0).is_none());

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn local_model_artifacts_exposes_routed_expert_weight_catalog() {
    let model_dir = temp_model_dir("local-routed-expert-weight-catalog");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "deepseek_v4",
  "num_hidden_layers": 1,
  "hidden_size": 1,
  "hc_mult": 1,
  "n_routed_experts": 1,
  "first_k_dense_replace": 0,
  "moe_layer_freq": 1
}"#,
    )
    .expect("config should be written");
    write_safetensors_file(
        &model_dir.join("model.safetensors"),
        &[
            ("model.layers.0.ffn.experts.0.w1.weight", "U8", &[1], [0, 1]),
            ("model.layers.0.ffn.experts.0.w2.weight", "U8", &[1], [1, 2]),
            ("model.layers.0.ffn.experts.0.w3.weight", "U8", &[1], [2, 3]),
        ],
        &[1, 2, 3],
    )
    .expect("shard should be written");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local artifacts should load");

    let catalog = artifacts
        .routed_expert_weight_catalog()
        .expect("local artifacts should expose routed expert catalog");

    assert_eq!(catalog.group_count(), 1);
    assert!(catalog.group(0, 0).is_some());

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn local_model_artifacts_builds_checkpoint_catalog_for_layer_and_routed_weights() {
    let model_dir = temp_model_dir("local-checkpoint-catalog");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "deepseek_v4",
  "num_hidden_layers": 1,
  "hidden_size": 1,
  "hc_mult": 1,
  "n_routed_experts": 1,
  "first_k_dense_replace": 0,
  "moe_layer_freq": 1
}"#,
    )
    .expect("config should be written");
    write_safetensors_file(
        &model_dir.join("model.safetensors"),
        &[
            (
                "model.layers.0.self_attn.q_a_proj.weight",
                "U8",
                &[1],
                [0, 1],
            ),
            ("model.layers.0.ffn.experts.0.w1.weight", "U8", &[1], [1, 2]),
            ("model.layers.0.ffn.experts.0.w2.weight", "U8", &[1], [2, 3]),
            ("model.layers.0.ffn.experts.0.w3.weight", "U8", &[1], [3, 4]),
        ],
        &[1, 2, 3, 4],
    )
    .expect("shard should be written");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local artifacts should load");

    let catalog = artifacts
        .checkpoint_catalog()
        .expect("complete checkpoint catalog should build");

    assert_eq!(catalog.layer_tensors().tensor_count(), 4);
    assert!(
        catalog
            .layer_tensors()
            .span(0, "self_attn.q_a_proj.weight")
            .is_some()
    );
    assert_eq!(catalog.routed_experts().group_count(), 1);
    assert!(catalog.routed_experts().group(0, 0).is_some());

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn local_model_artifacts_dispatches_checkpoint_validation_by_model_type() {
    let deepseek_dir = temp_model_dir("deepseek-v4-dispatch-validation");
    fs::create_dir_all(&deepseek_dir).expect("temp model dir should be created");
    fs::write(
        deepseek_dir.join("config.json"),
        r#"{
  "model_type": "deepseek_v4",
  "num_hidden_layers": 0
}"#,
    )
    .expect("config should be written");
    write_safetensors_file(
        &deepseek_dir.join("model.safetensors"),
        &[
            ("model.embed_tokens.weight", "U8", &[1], [0, 1]),
            ("model.norm.weight", "U8", &[1], [1, 2]),
        ],
        &[1, 2],
    )
    .expect("shard should be written");
    let deepseek_artifacts = LocalModelArtifacts::from_model_path(&deepseek_dir)
        .expect("DeepSeek local artifacts should load");

    let error = deepseek_artifacts
        .validate_checkpoint_for_supported_model()
        .expect_err("DeepSeek V4 dispatch should enforce DeepSeek checkpoint structure");

    assert!(
        matches!(
            error,
            ModelArtifactError::InvalidSafetensorsData { ref path, ref message }
                if path == &deepseek_dir
                    && message.contains("missing DeepSeek model tensor")
                    && message.contains("lm_head.weight")
        ),
        "unexpected error: {error:?}"
    );

    let generic_dir = temp_model_dir("generic-dispatch-validation");
    fs::create_dir_all(&generic_dir).expect("temp model dir should be created");
    fs::write(
        generic_dir.join("config.json"),
        r#"{
  "model_type": "llama"
}"#,
    )
    .expect("config should be written");
    write_safetensors_file(
        &generic_dir.join("model.safetensors"),
        &[("model.embed_tokens.weight", "U8", &[1], [0, 1])],
        &[7],
    )
    .expect("shard should be written");
    let generic_artifacts = LocalModelArtifacts::from_model_path(&generic_dir)
        .expect("generic local artifacts should load");

    generic_artifacts
        .validate_checkpoint_for_supported_model()
        .expect("unknown model dispatch should keep generic artifact validation");

    let glm_dir = temp_model_dir("glm-moe-dsa-dispatch-validation");
    fs::create_dir_all(&glm_dir).expect("temp model dir should be created");
    fs::write(
        glm_dir.join("config.json"),
        r#"{
  "model_type": "glm_moe_dsa",
  "num_hidden_layers": 0
}"#,
    )
    .expect("config should be written");
    write_safetensors_file(
        &glm_dir.join("model.safetensors"),
        &[
            ("model.embed_tokens.weight", "U8", &[1], [0, 1]),
            ("model.norm.weight", "U8", &[1], [1, 2]),
        ],
        &[1, 2],
    )
    .expect("shard should be written");
    let glm_artifacts =
        LocalModelArtifacts::from_model_path(&glm_dir).expect("GLM local artifacts should load");

    let error = glm_artifacts
        .validate_checkpoint_for_supported_model()
        .expect_err("GLM-DSA dispatch should enforce CausalLM checkpoint roots");

    assert!(
        matches!(
            error,
            ModelArtifactError::InvalidSafetensorsData { ref path, ref message }
                if path == &glm_dir
                    && message.contains("missing GLM-DSA model tensor")
                    && message.contains("lm_head.weight")
        ),
        "unexpected error: {error:?}"
    );

    fs::remove_dir_all(deepseek_dir).expect("temp model dir should be removed");
    fs::remove_dir_all(generic_dir).expect("temp model dir should be removed");
    fs::remove_dir_all(glm_dir).expect("temp model dir should be removed");
}

#[test]
fn local_model_checkpoint_catalog_exposes_glm_moe_dsa_dense_layer_weights() {
    let model_dir = temp_model_dir("glm-moe-dsa-dense-layer-weights");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "glm_moe_dsa",
  "num_hidden_layers": 1,
  "hidden_size": 1
}"#,
    )
    .expect("config should be written");
    write_safetensors_file(
        &model_dir.join("model.safetensors"),
        &[
            ("model.embed_tokens.weight", "U8", &[1], [0, 1]),
            ("model.norm.weight", "U8", &[1], [1, 2]),
            ("lm_head.weight", "U8", &[1], [2, 3]),
            (
                "model.layers.0.self_attn.q_a_proj.weight",
                "U8",
                &[1],
                [3, 4],
            ),
            (
                "model.layers.0.self_attn.q_a_layernorm.weight",
                "U8",
                &[1],
                [4, 5],
            ),
            (
                "model.layers.0.self_attn.q_b_proj.weight",
                "U8",
                &[1],
                [5, 6],
            ),
            (
                "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                "U8",
                &[1],
                [6, 7],
            ),
            (
                "model.layers.0.self_attn.kv_a_layernorm.weight",
                "U8",
                &[1],
                [7, 8],
            ),
            (
                "model.layers.0.self_attn.kv_b_proj.weight",
                "U8",
                &[1],
                [8, 9],
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                "U8",
                &[1],
                [9, 10],
            ),
            (
                "model.layers.0.input_layernorm.weight",
                "U8",
                &[1],
                [10, 11],
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "U8",
                &[1],
                [11, 12],
            ),
            ("model.layers.0.mlp.gate_proj.weight", "U8", &[1], [12, 13]),
            ("model.layers.0.mlp.up_proj.weight", "U8", &[1], [13, 14]),
            ("model.layers.0.mlp.down_proj.weight", "U8", &[1], [14, 15]),
        ],
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    )
    .expect("shard should be written");

    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let checkpoint_catalog = artifacts
        .checkpoint_catalog()
        .expect("checkpoint catalog should build");
    let weights = checkpoint_catalog
        .glm_moe_dsa_model_weights()
        .expect("GLM-DSA checkpoint catalog should expose layer weights");

    assert_eq!(
        weights.token_embeddings().tensor_name,
        "model.embed_tokens.weight"
    );
    assert_eq!(weights.final_norm().tensor_name, "model.norm.weight");
    assert_eq!(weights.lm_head().tensor_name, "lm_head.weight");
    assert_eq!(weights.layer_count(), 1);

    let layer = weights.layer(0).expect("layer 0 should be present");
    assert_eq!(
        layer.q_a_proj().tensor_name,
        "model.layers.0.self_attn.q_a_proj.weight"
    );
    assert_eq!(
        layer.kv_b_proj().tensor_name,
        "model.layers.0.self_attn.kv_b_proj.weight"
    );
    assert_eq!(
        layer.o_proj().tensor_name,
        "model.layers.0.self_attn.o_proj.weight"
    );
    match layer.feed_forward() {
        GlmMoeDsaLayerFeedForwardCheckpointWeights::Dense {
            gate_proj,
            up_proj,
            down_proj,
        } => {
            assert_eq!(gate_proj.tensor_name, "model.layers.0.mlp.gate_proj.weight");
            assert_eq!(up_proj.tensor_name, "model.layers.0.mlp.up_proj.weight");
            assert_eq!(down_proj.tensor_name, "model.layers.0.mlp.down_proj.weight");
        }
        GlmMoeDsaLayerFeedForwardCheckpointWeights::Moe { .. } => {
            panic!("dense GLM-DSA layer should not expose MoE weights")
        }
    }

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn local_model_checkpoint_catalog_exposes_glm_moe_dsa_moe_layer_weights() {
    let model_dir = temp_model_dir("glm-moe-dsa-moe-layer-weights");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "glm_moe_dsa",
  "num_hidden_layers": 1,
  "n_routed_experts": 2,
  "first_k_dense_replace": 0,
  "moe_layer_freq": 1
}"#,
    )
    .expect("config should be written");
    let header_len = write_safetensors_file(
        &model_dir.join("model.safetensors"),
        &[
            ("model.embed_tokens.weight", "U8", &[1], [0, 1]),
            ("model.norm.weight", "U8", &[1], [1, 2]),
            ("lm_head.weight", "U8", &[1], [2, 3]),
            (
                "model.layers.0.self_attn.q_a_proj.weight",
                "U8",
                &[1],
                [3, 4],
            ),
            (
                "model.layers.0.self_attn.q_a_layernorm.weight",
                "U8",
                &[1],
                [4, 5],
            ),
            (
                "model.layers.0.self_attn.q_b_proj.weight",
                "U8",
                &[1],
                [5, 6],
            ),
            (
                "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                "U8",
                &[1],
                [6, 7],
            ),
            (
                "model.layers.0.self_attn.kv_a_layernorm.weight",
                "U8",
                &[1],
                [7, 8],
            ),
            (
                "model.layers.0.self_attn.kv_b_proj.weight",
                "U8",
                &[1],
                [8, 9],
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                "U8",
                &[1],
                [9, 10],
            ),
            (
                "model.layers.0.input_layernorm.weight",
                "U8",
                &[1],
                [10, 11],
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "U8",
                &[1],
                [11, 12],
            ),
            ("model.layers.0.mlp.gate.weight", "U8", &[1], [12, 13]),
            (
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                "U8",
                &[1],
                [13, 14],
            ),
            (
                "model.layers.0.mlp.experts.0.down_proj.weight",
                "U8",
                &[1],
                [14, 15],
            ),
            (
                "model.layers.0.mlp.experts.0.up_proj.weight",
                "U8",
                &[1],
                [15, 16],
            ),
            (
                "model.layers.0.mlp.experts.1.gate_proj.weight",
                "U8",
                &[1],
                [16, 17],
            ),
            (
                "model.layers.0.mlp.experts.1.down_proj.weight",
                "U8",
                &[1],
                [17, 18],
            ),
            (
                "model.layers.0.mlp.experts.1.up_proj.weight",
                "U8",
                &[1],
                [18, 19],
            ),
        ],
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        ],
    )
    .expect("shard should be written");

    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let checkpoint_catalog = artifacts
        .checkpoint_catalog()
        .expect("checkpoint catalog should build");
    let weights = checkpoint_catalog
        .glm_moe_dsa_model_weights()
        .expect("GLM-DSA checkpoint catalog should expose MoE layer weights");

    let layer = weights.layer(0).expect("layer 0 should be present");
    match layer.feed_forward() {
        GlmMoeDsaLayerFeedForwardCheckpointWeights::Moe {
            gate,
            routed_experts,
        } => {
            assert_eq!(gate.tensor_name, "model.layers.0.mlp.gate.weight");
            assert_eq!(routed_experts.layer_id(), 0);
            assert_eq!(routed_experts.expert_count(), 2);
            assert_eq!(routed_experts.expert_ids().collect::<Vec<_>>(), vec![0, 1]);
            let expert_1 = routed_experts
                .group(1)
                .expect("expert 1 weights should be present");
            assert_eq!(expert_1.layer_id, 0);
            assert_eq!(expert_1.expert_id, 1);
            assert_eq!(
                expert_1.gate.absolute_byte_offset,
                8 + header_len as u64 + 16
            );
            assert_eq!(
                expert_1.down.absolute_byte_offset,
                8 + header_len as u64 + 17
            );
            assert_eq!(expert_1.up.absolute_byte_offset, 8 + header_len as u64 + 18);
        }
        GlmMoeDsaLayerFeedForwardCheckpointWeights::Dense { .. } => {
            panic!("MoE GLM-DSA layer should expose routed expert weights")
        }
    }

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn local_model_checkpoint_catalog_exposes_deepseek_v4_model_weights() {
    let model_dir = temp_model_dir("deepseek-v4-model-weights");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "deepseek_v4",
  "num_hidden_layers": 1,
  "hidden_size": 1,
  "hc_mult": 1,
  "n_routed_experts": 1,
  "first_k_dense_replace": 0,
  "moe_layer_freq": 1
}"#,
    )
    .expect("config should be written");
    write_safetensors_file(
        &model_dir.join("model.safetensors"),
        &[
            ("model.embed_tokens.weight", "U8", &[1], [0, 1]),
            ("model.norm.weight", "U8", &[1], [1, 2]),
            ("lm_head.weight", "U8", &[1], [2, 3]),
            ("model.hc_head_fn", "U8", &[1, 1], [3, 4]),
            ("model.hc_head_base", "U8", &[1], [4, 5]),
            ("model.hc_head_scale", "U8", &[1], [5, 6]),
            ("model.layers.0.self_attn.wq_a.weight", "U8", &[1], [6, 7]),
            ("model.layers.0.self_attn.wq_b.weight", "U8", &[1], [7, 8]),
            ("model.layers.0.self_attn.wkv.weight", "U8", &[1], [8, 9]),
            (
                "model.layers.0.self_attn.q_norm.weight",
                "U8",
                &[1],
                [9, 10],
            ),
            (
                "model.layers.0.self_attn.kv_norm.weight",
                "U8",
                &[1],
                [10, 11],
            ),
            ("model.layers.0.self_attn.wo_a.weight", "U8", &[1], [11, 12]),
            ("model.layers.0.self_attn.wo_b.weight", "U8", &[1], [12, 13]),
            (
                "model.layers.0.input_layernorm.weight",
                "U8",
                &[1],
                [13, 14],
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "U8",
                &[1],
                [14, 15],
            ),
            ("model.layers.0.hc_attn_fn", "U8", &[1], [15, 16]),
            ("model.layers.0.hc_attn_base", "U8", &[1], [16, 17]),
            ("model.layers.0.hc_attn_scale", "U8", &[1], [17, 18]),
            ("model.layers.0.hc_ffn_fn", "U8", &[1], [18, 19]),
            ("model.layers.0.hc_ffn_base", "U8", &[1], [19, 20]),
            ("model.layers.0.hc_ffn_scale", "U8", &[1], [20, 21]),
            ("model.layers.0.mlp.gate.weight", "U8", &[1], [21, 22]),
            (
                "model.layers.0.ffn.experts.0.w1.weight",
                "U8",
                &[1],
                [22, 23],
            ),
            (
                "model.layers.0.ffn.experts.0.w2.weight",
                "U8",
                &[1],
                [23, 24],
            ),
            (
                "model.layers.0.ffn.experts.0.w3.weight",
                "U8",
                &[1],
                [24, 25],
            ),
        ],
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25,
        ],
    )
    .expect("shard should be written");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local artifacts should load");
    let checkpoint = artifacts
        .checkpoint_catalog()
        .expect("checkpoint catalog should build");

    let model = checkpoint
        .deepseek_model_weights()
        .expect("DeepSeek V4 model weights should be complete");

    assert_eq!(model.layer_count(), 1);
    assert_eq!(
        model.token_embeddings().tensor_name,
        "model.embed_tokens.weight"
    );
    assert_eq!(model.final_norm().tensor_name, "model.norm.weight");
    assert_eq!(model.lm_head().tensor_name, "lm_head.weight");
    assert_eq!(model.hc_head_fn().tensor_name, "model.hc_head_fn");
    assert_eq!(model.hc_head_base().tensor_name, "model.hc_head_base");
    assert_eq!(model.hc_head_scale().tensor_name, "model.hc_head_scale");
    assert_eq!(
        model
            .layer(0)
            .expect("model view should expose layer 0")
            .wq_a()
            .suffix,
        "self_attn.wq_a.weight"
    );
    assert!(model.layer(1).is_none());
    let loaded_roots = model
        .read_root_tensors()
        .expect("DeepSeek V4 root tensors should read");
    assert_eq!(loaded_roots.token_embeddings().bytes, vec![1]);
    assert_eq!(loaded_roots.final_norm().bytes, vec![2]);
    assert_eq!(loaded_roots.lm_head().bytes, vec![3]);
    assert_eq!(loaded_roots.hc_head_fn().bytes, vec![4]);
    assert_eq!(loaded_roots.hc_head_base().bytes, vec![5]);
    assert_eq!(loaded_roots.hc_head_scale().bytes, vec![6]);

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn local_model_checkpoint_catalog_rejects_missing_deepseek_v4_model_root_tensor() {
    let model_dir = temp_model_dir("deepseek-v4-model-weights-missing-root");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "deepseek_v4",
  "num_hidden_layers": 0
}"#,
    )
    .expect("config should be written");
    write_safetensors_file(
        &model_dir.join("model.safetensors"),
        &[
            ("model.embed_tokens.weight", "U8", &[1], [0, 1]),
            ("model.norm.weight", "U8", &[1], [1, 2]),
        ],
        &[1, 2],
    )
    .expect("shard should be written");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local artifacts should load");
    let checkpoint = artifacts
        .checkpoint_catalog()
        .expect("checkpoint catalog should build");

    let error = checkpoint
        .deepseek_model_weights()
        .expect_err("missing DeepSeek V4 root tensor should fail model view validation");

    assert!(
        matches!(
            error,
            ModelArtifactError::InvalidSafetensorsData { ref path, ref message }
                if path == &model_dir
                    && message.contains("missing DeepSeek model tensor")
                    && message.contains("lm_head.weight")
        ),
        "unexpected error: {error:?}"
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn local_model_checkpoint_catalog_rejects_missing_deepseek_v4_hc_head_tensor() {
    let model_dir = temp_model_dir("deepseek-v4-model-weights-missing-hc-head");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "deepseek_v4",
  "num_hidden_layers": 0
}"#,
    )
    .expect("config should be written");
    write_safetensors_file(
        &model_dir.join("model.safetensors"),
        &[
            ("model.embed_tokens.weight", "U8", &[1], [0, 1]),
            ("model.norm.weight", "U8", &[1], [1, 2]),
            ("lm_head.weight", "U8", &[1], [2, 3]),
        ],
        &[1, 2, 3],
    )
    .expect("shard should be written");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local artifacts should load");
    let checkpoint = artifacts
        .checkpoint_catalog()
        .expect("checkpoint catalog should build");

    let error = checkpoint
        .deepseek_model_weights()
        .expect_err("missing DeepSeek V4 HC head tensor should fail model view validation");

    assert!(
        matches!(
            error,
            ModelArtifactError::InvalidSafetensorsData { ref path, ref message }
                if path == &model_dir
                    && message.contains("missing DeepSeek model tensor")
                    && message.contains("model.hc_head_fn")
        ),
        "unexpected error: {error:?}"
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn local_model_checkpoint_catalog_rejects_deepseek_v4_hc_head_shape_mismatch() {
    let model_dir = temp_model_dir("deepseek-v4-model-weights-hc-head-shape");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "deepseek_v4",
  "num_hidden_layers": 0,
  "hidden_size": 2,
  "hc_mult": 2
}"#,
    )
    .expect("config should be written");
    write_safetensors_file(
        &model_dir.join("model.safetensors"),
        &[
            ("model.embed_tokens.weight", "U8", &[1], [0, 1]),
            ("model.norm.weight", "U8", &[1], [1, 2]),
            ("lm_head.weight", "U8", &[1], [2, 3]),
            ("model.hc_head_fn", "U8", &[3], [3, 6]),
            ("model.hc_head_base", "U8", &[2], [6, 8]),
            ("model.hc_head_scale", "U8", &[1], [8, 9]),
        ],
        &[1, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    .expect("shard should be written");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local artifacts should load");
    let checkpoint = artifacts
        .checkpoint_catalog()
        .expect("checkpoint catalog should build");

    let error = checkpoint
        .deepseek_model_weights()
        .expect_err("invalid DeepSeek V4 HC head tensor shape should fail model validation");

    assert!(
        matches!(
            error,
            ModelArtifactError::InvalidSafetensorsData { ref path, ref message }
                if path == &model_dir
                    && message.contains("DeepSeek model tensor model.hc_head_fn")
                    && message.contains("shape")
                    && message.contains("[2, 4]")
        ),
        "unexpected error: {error:?}"
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn local_model_checkpoint_catalog_exposes_deepseek_v4_moe_layer_weights() {
    let model_dir = temp_model_dir("deepseek-v4-layer-weights");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "deepseek_v4",
  "num_hidden_layers": 1,
  "n_routed_experts": 1,
  "first_k_dense_replace": 0,
  "moe_layer_freq": 1
}"#,
    )
    .expect("config should be written");
    write_safetensors_file(
        &model_dir.join("model.safetensors"),
        &[
            ("model.layers.0.self_attn.wq_a.weight", "U8", &[1], [0, 1]),
            ("model.layers.0.self_attn.wq_b.weight", "U8", &[1], [1, 2]),
            ("model.layers.0.self_attn.wkv.weight", "U8", &[1], [2, 3]),
            ("model.layers.0.self_attn.q_norm.weight", "U8", &[1], [3, 4]),
            (
                "model.layers.0.self_attn.kv_norm.weight",
                "U8",
                &[1],
                [4, 5],
            ),
            ("model.layers.0.self_attn.wo_a.weight", "U8", &[1], [5, 6]),
            ("model.layers.0.self_attn.wo_b.weight", "U8", &[1], [6, 7]),
            ("model.layers.0.input_layernorm.weight", "U8", &[1], [7, 8]),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "U8",
                &[1],
                [8, 9],
            ),
            ("model.layers.0.hc_attn_fn", "U8", &[1], [9, 10]),
            ("model.layers.0.hc_attn_base", "U8", &[1], [10, 11]),
            ("model.layers.0.hc_attn_scale", "U8", &[1], [11, 12]),
            ("model.layers.0.hc_ffn_fn", "U8", &[1], [12, 13]),
            ("model.layers.0.hc_ffn_base", "U8", &[1], [13, 14]),
            ("model.layers.0.hc_ffn_scale", "U8", &[1], [14, 15]),
            ("model.layers.0.mlp.gate.weight", "U8", &[1], [15, 16]),
            (
                "model.layers.0.ffn.experts.0.w1.weight",
                "U8",
                &[1],
                [16, 17],
            ),
            (
                "model.layers.0.ffn.experts.0.w2.weight",
                "U8",
                &[1],
                [17, 18],
            ),
            (
                "model.layers.0.ffn.experts.0.w3.weight",
                "U8",
                &[1],
                [18, 19],
            ),
        ],
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        ],
    )
    .expect("shard should be written");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local artifacts should load");
    let checkpoint = artifacts
        .checkpoint_catalog()
        .expect("checkpoint catalog should build");

    let layer = checkpoint
        .deepseek_layer_weights(0)
        .expect("DeepSeek V4 layer weights should be complete");

    assert_eq!(layer.layer_id(), 0);
    assert_eq!(layer.wq_a().suffix, "self_attn.wq_a.weight");
    assert_eq!(layer.wq_b().suffix, "self_attn.wq_b.weight");
    assert_eq!(layer.wkv().suffix, "self_attn.wkv.weight");
    assert_eq!(layer.q_norm().suffix, "self_attn.q_norm.weight");
    assert_eq!(layer.kv_norm().suffix, "self_attn.kv_norm.weight");
    assert_eq!(layer.wo_a().suffix, "self_attn.wo_a.weight");
    assert_eq!(layer.wo_b().suffix, "self_attn.wo_b.weight");
    assert_eq!(layer.input_layernorm().suffix, "input_layernorm.weight");
    assert_eq!(
        layer.post_attention_layernorm().suffix,
        "post_attention_layernorm.weight"
    );
    assert_eq!(layer.hc_attn_fn().suffix, "hc_attn_fn");
    assert_eq!(layer.hc_attn_base().suffix, "hc_attn_base");
    assert_eq!(layer.hc_attn_scale().suffix, "hc_attn_scale");
    assert_eq!(layer.hc_ffn_fn().suffix, "hc_ffn_fn");
    assert_eq!(layer.hc_ffn_base().suffix, "hc_ffn_base");
    assert_eq!(layer.hc_ffn_scale().suffix, "hc_ffn_scale");
    match layer.feed_forward() {
        DeepSeekLayerFeedForwardCheckpointWeights::Moe {
            gate,
            routed_experts,
        } => {
            assert_eq!(gate.suffix, "mlp.gate.weight");
            assert_eq!(routed_experts.expert_count(), 1);
        }
        DeepSeekLayerFeedForwardCheckpointWeights::Dense { .. } => {
            panic!("MoE layer should expose MoE feed-forward weights")
        }
    }

    let loaded = layer
        .read_tensors()
        .expect("DeepSeek V4 MoE layer tensors should read");
    assert_eq!(loaded.layer_id(), 0);
    assert_eq!(loaded.wq_a().bytes, vec![1]);
    match loaded.feed_forward() {
        sglang_srt::model_artifacts::DeepSeekLoadedLayerFeedForwardWeights::Moe {
            gate,
            routed_experts,
        } => {
            assert_eq!(gate.bytes, vec![16]);
            assert_eq!(routed_experts.len(), 1);
            assert_eq!(routed_experts[0].expert_id(), 0);
            assert_eq!(routed_experts[0].gate().bytes, vec![17]);
            assert_eq!(routed_experts[0].down().bytes, vec![18]);
            assert_eq!(routed_experts[0].up().bytes, vec![19]);
        }
        sglang_srt::model_artifacts::DeepSeekLoadedLayerFeedForwardWeights::Dense { .. } => {
            panic!("loaded MoE layer should expose loaded MoE feed-forward weights")
        }
    }

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn local_model_checkpoint_catalog_exposes_deepseek_v4_dense_layer_weights() {
    let model_dir = temp_model_dir("deepseek-v4-dense-layer-weights");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "deepseek_v4",
  "num_hidden_layers": 2,
  "n_routed_experts": 1,
  "first_k_dense_replace": 1,
  "moe_layer_freq": 1
}"#,
    )
    .expect("config should be written");
    write_safetensors_file(
        &model_dir.join("model.safetensors"),
        &[
            ("model.layers.0.self_attn.wq_a.weight", "U8", &[1], [0, 1]),
            ("model.layers.0.self_attn.wq_b.weight", "U8", &[1], [1, 2]),
            ("model.layers.0.self_attn.wkv.weight", "U8", &[1], [2, 3]),
            ("model.layers.0.self_attn.q_norm.weight", "U8", &[1], [3, 4]),
            (
                "model.layers.0.self_attn.kv_norm.weight",
                "U8",
                &[1],
                [4, 5],
            ),
            ("model.layers.0.self_attn.wo_a.weight", "U8", &[1], [5, 6]),
            ("model.layers.0.self_attn.wo_b.weight", "U8", &[1], [6, 7]),
            ("model.layers.0.input_layernorm.weight", "U8", &[1], [7, 8]),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "U8",
                &[1],
                [8, 9],
            ),
            ("model.layers.0.hc_attn_fn", "U8", &[1], [9, 10]),
            ("model.layers.0.hc_attn_base", "U8", &[1], [10, 11]),
            ("model.layers.0.hc_attn_scale", "U8", &[1], [11, 12]),
            ("model.layers.0.hc_ffn_fn", "U8", &[1], [12, 13]),
            ("model.layers.0.hc_ffn_base", "U8", &[1], [13, 14]),
            ("model.layers.0.hc_ffn_scale", "U8", &[1], [14, 15]),
            (
                "model.layers.0.mlp.gate_up_proj.weight",
                "U8",
                &[1],
                [15, 16],
            ),
            ("model.layers.0.mlp.down_proj.weight", "U8", &[1], [16, 17]),
            (
                "model.layers.1.ffn.experts.0.w1.weight",
                "U8",
                &[1],
                [17, 18],
            ),
            (
                "model.layers.1.ffn.experts.0.w2.weight",
                "U8",
                &[1],
                [18, 19],
            ),
            (
                "model.layers.1.ffn.experts.0.w3.weight",
                "U8",
                &[1],
                [19, 20],
            ),
        ],
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        ],
    )
    .expect("shard should be written");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local artifacts should load");
    let checkpoint = artifacts
        .checkpoint_catalog()
        .expect("checkpoint catalog should build");

    let layer = checkpoint
        .deepseek_layer_weights(0)
        .expect("DeepSeek V4 dense layer weights should be complete");

    assert_eq!(layer.layer_id(), 0);
    match layer.feed_forward() {
        DeepSeekLayerFeedForwardCheckpointWeights::Dense {
            gate_up_proj,
            down_proj,
        } => {
            assert_eq!(gate_up_proj.suffix, "mlp.gate_up_proj.weight");
            assert_eq!(down_proj.suffix, "mlp.down_proj.weight");
        }
        DeepSeekLayerFeedForwardCheckpointWeights::Moe { .. } => {
            panic!("dense layer should expose dense feed-forward weights")
        }
    }

    let loaded = layer
        .read_tensors()
        .expect("DeepSeek V4 dense layer tensors should read");
    assert_eq!(loaded.layer_id(), 0);
    assert_eq!(loaded.wq_a().bytes, vec![1]);
    assert_eq!(loaded.hc_ffn_scale().bytes, vec![15]);
    match loaded.feed_forward() {
        sglang_srt::model_artifacts::DeepSeekLoadedLayerFeedForwardWeights::Dense {
            gate_up_proj,
            down_proj,
        } => {
            assert_eq!(gate_up_proj.bytes, vec![16]);
            assert_eq!(down_proj.bytes, vec![17]);
        }
        sglang_srt::model_artifacts::DeepSeekLoadedLayerFeedForwardWeights::Moe { .. } => {
            panic!("loaded dense layer should expose loaded dense feed-forward weights")
        }
    }

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn local_model_checkpoint_catalog_rejects_missing_deepseek_v4_layer_tensor() {
    let model_dir = temp_model_dir("deepseek-v4-layer-weights-missing");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "deepseek_v4",
  "num_hidden_layers": 1,
  "n_routed_experts": 1,
  "first_k_dense_replace": 0,
  "moe_layer_freq": 1
}"#,
    )
    .expect("config should be written");
    write_safetensors_file(
        &model_dir.join("model.safetensors"),
        &[
            ("model.layers.0.self_attn.wq_a.weight", "U8", &[1], [0, 1]),
            ("model.layers.0.self_attn.wq_b.weight", "U8", &[1], [1, 2]),
            ("model.layers.0.self_attn.wkv.weight", "U8", &[1], [2, 3]),
            ("model.layers.0.self_attn.q_norm.weight", "U8", &[1], [3, 4]),
            (
                "model.layers.0.self_attn.kv_norm.weight",
                "U8",
                &[1],
                [4, 5],
            ),
            ("model.layers.0.self_attn.wo_a.weight", "U8", &[1], [5, 6]),
            ("model.layers.0.input_layernorm.weight", "U8", &[1], [6, 7]),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "U8",
                &[1],
                [7, 8],
            ),
            ("model.layers.0.hc_attn_fn", "U8", &[1], [8, 9]),
            ("model.layers.0.hc_attn_base", "U8", &[1], [9, 10]),
            ("model.layers.0.hc_attn_scale", "U8", &[1], [10, 11]),
            ("model.layers.0.hc_ffn_fn", "U8", &[1], [11, 12]),
            ("model.layers.0.hc_ffn_base", "U8", &[1], [12, 13]),
            ("model.layers.0.hc_ffn_scale", "U8", &[1], [13, 14]),
            ("model.layers.0.mlp.gate.weight", "U8", &[1], [14, 15]),
            (
                "model.layers.0.ffn.experts.0.w1.weight",
                "U8",
                &[1],
                [15, 16],
            ),
            (
                "model.layers.0.ffn.experts.0.w2.weight",
                "U8",
                &[1],
                [16, 17],
            ),
            (
                "model.layers.0.ffn.experts.0.w3.weight",
                "U8",
                &[1],
                [17, 18],
            ),
        ],
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        ],
    )
    .expect("shard should be written");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local artifacts should load");
    let checkpoint = artifacts
        .checkpoint_catalog()
        .expect("checkpoint catalog should build");

    let error = checkpoint
        .deepseek_layer_weights(0)
        .expect_err("missing DeepSeek V4 layer tensor should fail validation");

    assert!(
        matches!(
            error,
            ModelArtifactError::InvalidSafetensorsData { ref path, ref message }
                if path == &model_dir
                    && message.contains("missing DeepSeek layer 0 tensor")
                    && message.contains("self_attn.wo_b.weight")
        ),
        "unexpected error: {error:?}"
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn local_model_artifacts_rejects_missing_routed_expert_checkpoint_groups() {
    let model_dir = temp_model_dir("routed-expert-coverage-missing");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "deepseek_v4",
  "num_hidden_layers": 2,
  "n_routed_experts": 2,
  "first_k_dense_replace": 0,
  "moe_layer_freq": 1
}"#,
    )
    .expect("config should be written");
    fs::write(
        model_dir.join("model.safetensors.index.json"),
        r#"{
  "weight_map": {
    "model.layers.0.ffn.experts.0.w1.weight": "model.safetensors",
    "model.layers.0.ffn.experts.0.w2.weight": "model.safetensors",
    "model.layers.0.ffn.experts.0.w3.weight": "model.safetensors"
  }
}"#,
    )
    .expect("index should be written");
    write_safetensors_file(
        &model_dir.join("model.safetensors"),
        &[
            ("model.layers.0.ffn.experts.0.w1.weight", "U8", &[1], [0, 1]),
            ("model.layers.0.ffn.experts.0.w2.weight", "U8", &[1], [1, 2]),
            ("model.layers.0.ffn.experts.0.w3.weight", "U8", &[1], [2, 3]),
        ],
        &[1, 2, 3],
    )
    .expect("shard should be written");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local artifacts should load");

    let error = artifacts
        .validate_routed_expert_checkpoint_coverage()
        .expect_err("missing routed expert groups should fail coverage validation");

    assert!(
        matches!(
            error,
            ModelArtifactError::InvalidSafetensorsData { ref path, ref message }
                if path == &model_dir
                    && message.contains("expected 4 routed expert groups")
                    && message.contains("found 1")
        ),
        "unexpected error: {error:?}"
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn local_model_artifacts_rejects_mismatched_routed_expert_checkpoint_coordinates() {
    let model_dir = temp_model_dir("routed-expert-coverage-wrong-coordinates");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "deepseek_v4",
  "num_hidden_layers": 2,
  "n_routed_experts": 2,
  "first_k_dense_replace": 0,
  "moe_layer_freq": 1
}"#,
    )
    .expect("config should be written");
    write_safetensors_file(
        &model_dir.join("model.safetensors"),
        &[
            ("model.layers.0.ffn.experts.0.w1.weight", "U8", &[1], [0, 1]),
            ("model.layers.0.ffn.experts.0.w2.weight", "U8", &[1], [1, 2]),
            ("model.layers.0.ffn.experts.0.w3.weight", "U8", &[1], [2, 3]),
            ("model.layers.0.ffn.experts.1.w1.weight", "U8", &[1], [3, 4]),
            ("model.layers.0.ffn.experts.1.w2.weight", "U8", &[1], [4, 5]),
            ("model.layers.0.ffn.experts.1.w3.weight", "U8", &[1], [5, 6]),
            ("model.layers.9.ffn.experts.0.w1.weight", "U8", &[1], [6, 7]),
            ("model.layers.9.ffn.experts.0.w2.weight", "U8", &[1], [7, 8]),
            ("model.layers.9.ffn.experts.0.w3.weight", "U8", &[1], [8, 9]),
            (
                "model.layers.9.ffn.experts.1.w1.weight",
                "U8",
                &[1],
                [9, 10],
            ),
            (
                "model.layers.9.ffn.experts.1.w2.weight",
                "U8",
                &[1],
                [10, 11],
            ),
            (
                "model.layers.9.ffn.experts.1.w3.weight",
                "U8",
                &[1],
                [11, 12],
            ),
        ],
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    )
    .expect("shard should be written");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local artifacts should load");

    let error = artifacts
        .validate_routed_expert_checkpoint_coverage()
        .expect_err("mismatched routed expert coordinates should fail coverage validation");

    assert!(
        matches!(
            error,
            ModelArtifactError::InvalidSafetensorsData { ref path, ref message }
                if path == &model_dir
                    && message.contains("missing expected routed expert group layer 1 expert 0")
                    && message.contains("unexpected routed expert group layer 9 expert 0")
        ),
        "unexpected error: {error:?}"
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn local_model_artifacts_skips_routed_expert_coverage_for_dense_configs() {
    let model_dir = temp_model_dir("dense-checkpoint-coverage");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "llama",
  "num_hidden_layers": 2
}"#,
    )
    .expect("config should be written");
    write_safetensors_file(
        &model_dir.join("model.safetensors"),
        &[("model.layers.0.mlp.down_proj.weight", "U8", &[1], [0, 1])],
        &[1],
    )
    .expect("shard should be written");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local artifacts should load");

    let summary = artifacts
        .validate_routed_expert_checkpoint_coverage()
        .expect("dense config should not require routed expert coverage");

    assert_eq!(summary.expected_group_count, 0);
    assert_eq!(summary.actual_group_count, 0);
    assert_eq!(summary.expected_weight_count, 0);
    assert_eq!(summary.actual_weight_count, 0);

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn hf_model_config_treats_null_optional_moe_fields_as_absent() {
    let model_dir = temp_model_dir("null-optional-moe-config");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "deepseek_v4",
  "num_hidden_layers": 4,
  "n_routed_experts": null,
  "n_shared_experts": null,
  "num_experts_per_tok": null,
  "moe_intermediate_size": null
}"#,
    )
    .expect("config should be written");

    let config = HfModelConfig::from_model_path(&model_dir)
        .expect("null optional MoE fields should parse as absent");

    assert_eq!(config.n_routed_experts, None);
    assert_eq!(config.n_shared_experts, None);
    assert_eq!(config.num_experts_per_tok, None);
    assert_eq!(config.moe_intermediate_size, None);
    assert!(!config.is_moe_layer(0));
    assert_eq!(config.expected_routed_expert_group_count(), None);

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn hf_model_config_parses_glm_moe_routing_fields() {
    let model_dir = temp_model_dir("glm-moe-routing-config");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "glm_moe_dsa",
  "num_hidden_layers": 1,
  "n_routed_experts": 2,
  "num_experts_per_tok": 1,
  "norm_topk_prob": false,
  "routed_scaling_factor": 2.0
}"#,
    )
    .expect("config should be written");

    let config = HfModelConfig::from_model_path(&model_dir)
        .expect("GLM MoE routing config fields should parse");

    assert_eq!(config.num_experts_per_tok, Some(1));
    assert_eq!(config.norm_topk_prob, Some(false));
    assert_eq!(
        config.routed_scaling_factor.map(|value| value.get()),
        Some(2.0)
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn hf_model_config_loads_repo_id_from_huggingface_cache_snapshot() {
    let hub_dir = temp_model_dir("hf-cache-hub");
    let snapshot_dir = hub_dir
        .join("models--zai-org--GLM-5-FP8")
        .join("snapshots")
        .join("abc123");
    fs::create_dir_all(&snapshot_dir).expect("snapshot dir should be created");
    fs::create_dir_all(hub_dir.join("models--zai-org--GLM-5-FP8").join("refs"))
        .expect("refs dir should be created");
    fs::write(
        hub_dir
            .join("models--zai-org--GLM-5-FP8")
            .join("refs")
            .join("main"),
        "abc123\n",
    )
    .expect("main ref should be written");
    fs::write(
        snapshot_dir.join("config.json"),
        r#"{
  "model_type": "glm",
  "vocab_size": 151552,
  "num_hidden_layers": 48,
  "num_key_value_heads": 8,
  "head_dim": 128
}"#,
    )
    .expect("config should be written");

    let config = HfModelConfig::from_model_path_with_hf_cache("zai-org/GLM-5-FP8", &hub_dir)
        .expect("repo id should resolve through HF cache");

    assert_eq!(config.model_type.as_deref(), Some("glm"));
    assert_eq!(config.vocab_size, Some(151_552));
    assert_eq!(config.num_hidden_layers, Some(48));

    fs::remove_dir_all(hub_dir).expect("temp hub dir should be removed");
}

#[test]
fn hf_model_config_downloads_repo_id_config_when_cache_is_missing() {
    let _env_guard = HF_ENV_LOCK
        .lock()
        .expect("HF env lock should not be poisoned");
    let hf_home = temp_model_dir("hf-config-download-home");
    let endpoint = start_fake_hf_config_endpoint(glm_moe_dsa_config_json());
    let _hf_home = EnvVarRestore::set("HF_HOME", &hf_home);
    let _hf_hub_cache = EnvVarRestore::set("HUGGINGFACE_HUB_CACHE", hf_home.join("hub"));
    let _hf_endpoint = EnvVarRestore::set("HF_ENDPOINT", endpoint);

    let config = HfModelConfig::from_model_path("zai-org/GLM-5-FP8")
        .expect("repo id config should download through HF Hub");

    assert_eq!(config.model_type.as_deref(), Some("glm_moe_dsa"));
    assert_eq!(config.eos_token_ids, vec![151_329, 151_336, 151_338]);
    assert_eq!(config.num_hidden_layers, Some(78));
    assert_eq!(config.num_attention_heads, Some(64));
    assert_eq!(config.num_key_value_heads, Some(64));
    assert_eq!(config.head_dim, Some(64));

    fs::remove_dir_all(hf_home).expect("temp HF home should be removed");
}

#[test]
fn local_model_artifacts_load_repo_id_from_huggingface_cache_snapshot() {
    let hub_dir = temp_model_dir("hf-cache-artifacts-hub");
    let snapshot_dir = hub_dir
        .join("models--zai-org--GLM-5-FP8")
        .join("snapshots")
        .join("abc123");
    fs::create_dir_all(&snapshot_dir).expect("snapshot dir should be created");
    fs::create_dir_all(hub_dir.join("models--zai-org--GLM-5-FP8").join("refs"))
        .expect("refs dir should be created");
    fs::write(
        hub_dir
            .join("models--zai-org--GLM-5-FP8")
            .join("refs")
            .join("main"),
        "abc123\n",
    )
    .expect("main ref should be written");
    fs::write(snapshot_dir.join("config.json"), deepseek_v4_config_json())
        .expect("config should be written");
    fs::write(
        snapshot_dir.join("model.safetensors.index.json"),
        safetensors_index_json(),
    )
    .expect("index should be written");
    write_safetensors_file(
        &snapshot_dir.join("model-00001-of-00002.safetensors"),
        &[
            ("model.embed_tokens.weight", "U8", &[4], [0, 4]),
            ("model.layers.0.ffn.experts.3.w1.weight", "U8", &[4], [4, 8]),
            (
                "model.layers.0.self_attn.q_a_proj.weight",
                "U8",
                &[4],
                [8, 12],
            ),
        ],
        &[0; 12],
    )
    .expect("first shard should be written");
    write_safetensors_file(
        &snapshot_dir.join("model-00002-of-00002.safetensors"),
        &[(
            "model.layers.0.self_attn.q_b_proj.weight",
            "U8",
            &[4],
            [0, 4],
        )],
        &[0; 4],
    )
    .expect("second shard should be written");

    let artifacts =
        LocalModelArtifacts::from_model_path_with_hf_cache("zai-org/GLM-5-FP8", &hub_dir)
            .expect("repo id should resolve to cached HF snapshot artifacts");

    assert_eq!(artifacts.model_path(), snapshot_dir.as_path());
    assert_eq!(
        artifacts.config().model_type.as_deref(),
        Some("deepseek_v4")
    );
    assert_eq!(
        artifacts
            .safetensors()
            .shard_for_tensor("model.layers.0.self_attn.q_b_proj.weight"),
        Some(
            snapshot_dir
                .join("model-00002-of-00002.safetensors")
                .as_path()
        )
    );

    fs::remove_dir_all(hub_dir).expect("temp hub dir should be removed");
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
fn safetensors_manifest_indexes_layer_tensor_spans_by_suffix() {
    let model_dir = temp_model_dir("layer-tensor-spans");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(model_dir.join("config.json"), deepseek_v4_config_json())
        .expect("config should be written");
    let shard = model_dir.join("model.safetensors");
    let header_len = write_safetensors_file(
        &shard,
        &[
            ("model.embed_tokens.weight", "U8", &[2], [0, 2]),
            (
                "model.layers.0.self_attn.q_a_proj.weight",
                "U8",
                &[1],
                [2, 3],
            ),
            (
                "model.layers.0.self_attn.q_b_proj.weight",
                "U8",
                &[2],
                [3, 5],
            ),
            ("model.layers.0.ffn.experts.3.w1.weight", "U8", &[3], [5, 8]),
        ],
        &[10, 11, 20, 30, 31, 40, 41, 42],
    )
    .expect("shard should be written");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local artifacts should load");

    let layer_spans = artifacts
        .safetensors()
        .layer_tensor_spans()
        .expect("layer tensor spans should be indexed");

    assert_eq!(
        layer_spans
            .iter()
            .map(|entry| (entry.layer_id, entry.suffix.as_str()))
            .collect::<Vec<_>>(),
        vec![
            (0, "ffn.experts.3.w1.weight"),
            (0, "self_attn.q_a_proj.weight"),
            (0, "self_attn.q_b_proj.weight"),
        ]
    );
    assert_eq!(
        layer_spans[1],
        SafetensorsLayerTensorSpan {
            tensor_name: "model.layers.0.self_attn.q_a_proj.weight".to_string(),
            layer_id: 0,
            suffix: "self_attn.q_a_proj.weight".to_string(),
            span: SafetensorsTensorSpan {
                path: shard,
                metadata: SafetensorsTensorMetadata {
                    dtype: "U8".to_string(),
                    shape: vec![1],
                    data_offsets: [2, 3],
                },
                absolute_byte_offset: 8 + header_len as u64 + 2,
                byte_len: 1,
            },
        }
    );
    assert_eq!(
        artifacts
            .safetensors()
            .layer_tensor_span(0, "self_attn.q_b_proj.weight")
            .expect("layer tensor lookup should read spans")
            .map(|entry| (entry.layer_id, entry.suffix, entry.span.byte_len)),
        Some((0, "self_attn.q_b_proj.weight".to_string(), 2))
    );
    assert_eq!(
        artifacts
            .safetensors()
            .layer_tensor_span(0, "self_attn.nope.weight")
            .expect("missing layer tensor lookup should still read spans"),
        None
    );
    let catalog = SafetensorsLayerTensorCatalog::from_safetensors_manifest(artifacts.safetensors())
        .expect("layer tensor catalog should build from manifest");
    assert_eq!(catalog.tensor_count(), 3);
    assert_eq!(catalog.layer_ids().collect::<Vec<_>>(), vec![0]);
    assert_eq!(
        catalog.suffixes(0).collect::<Vec<_>>(),
        vec![
            "ffn.experts.3.w1.weight",
            "self_attn.q_a_proj.weight",
            "self_attn.q_b_proj.weight",
        ]
    );
    assert_eq!(
        catalog
            .span(0, "self_attn.q_b_proj.weight")
            .map(|entry| (entry.tensor_name.as_str(), entry.span.byte_len)),
        Some(("model.layers.0.self_attn.q_b_proj.weight", 2))
    );
    assert!(catalog.span(0, "self_attn.nope.weight").is_none());
    assert!(catalog.span(1, "self_attn.q_b_proj.weight").is_none());

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn safetensors_layer_tensor_catalog_rejects_duplicate_layer_suffixes() {
    let model_dir = temp_model_dir("duplicate-layer-tensor-suffixes");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(model_dir.join("config.json"), deepseek_v4_config_json())
        .expect("config should be written");
    let first_shard = model_dir.join("model-00001.safetensors");
    write_safetensors_file(
        &first_shard,
        &[(
            "model.layers.0.self_attn.q_a_proj.weight",
            "U8",
            &[1],
            [0, 1],
        )],
        &[1],
    )
    .expect("first shard should be written");
    let second_shard = model_dir.join("model-00002.safetensors");
    write_safetensors_file(
        &second_shard,
        &[(
            "model.layers.0.self_attn.q_a_proj.weight",
            "U8",
            &[1],
            [0, 1],
        )],
        &[2],
    )
    .expect("second shard should be written");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local artifacts should load");

    let error = artifacts
        .safetensors()
        .layer_tensor_catalog()
        .expect_err("duplicate layer tensor suffix should be rejected");

    assert!(
        matches!(
            error,
            ModelArtifactError::InvalidSafetensorsData { ref path, ref message }
                if path == &second_shard
                    && message.contains("duplicate layer tensor suffix")
                    && message.contains("layer 0")
                    && message.contains("self_attn.q_a_proj.weight")
        ),
        "unexpected error: {error:?}"
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
fn safetensors_tensor_data_decodes_common_float_dtypes_to_f32() {
    let cases = [
        (
            "F32",
            vec![2],
            [1.0_f32, -2.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
            vec![1.0, -2.0],
        ),
        (
            "BF16",
            vec![2],
            vec![0x80, 0x3f, 0x00, 0xc0],
            vec![1.0, -2.0],
        ),
        (
            "F16",
            vec![3],
            vec![0x00, 0x3c, 0x00, 0xc0, 0x00, 0x38],
            vec![1.0, -2.0, 0.5],
        ),
        (
            "F8_E4M3",
            vec![5],
            vec![0x00, 0x38, 0x40, 0xb8, 0x30],
            vec![0.0, 1.0, 2.0, -1.0, 0.5],
        ),
    ];

    for (dtype, shape, bytes, expected) in cases {
        let tensor = SafetensorsTensorData {
            metadata: SafetensorsTensorMetadata {
                dtype: dtype.to_string(),
                shape,
                data_offsets: [0, bytes.len()],
            },
            bytes,
        };

        let decoded = tensor
            .decode_f32_values()
            .expect("supported float dtype should decode");

        assert_eq!(decoded, expected, "dtype {dtype} should decode");
    }
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
fn safetensors_manifest_pairs_quantized_linear_weights_with_scale_tensors() {
    let model_dir = temp_model_dir("quantized-linear-weight-pairs");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "llama"
}"#,
    )
    .expect("config should be written");
    fs::write(
        model_dir.join("model.safetensors.index.json"),
        r#"{
  "weight_map": {
    "lm_head.weight": "model.safetensors",
    "lm_head.weight_scale_inv": "model.safetensors",
    "model.layers.0.self_attn.wq_a.weight": "model.safetensors",
    "model.layers.0.self_attn.wq_a.weight_scale_inv": "model.safetensors",
    "model.layers.0.self_attn.wq_b.weight": "model.safetensors",
    "model.layers.0.self_attn.wq_b.weight_scale": "model.safetensors",
    "model.layers.0.self_attn.wkv.weight_scale_inv": "model.safetensors"
  }
}"#,
    )
    .expect("index should be written");
    let shard = model_dir.join("model.safetensors");
    let header_len = write_safetensors_file(
        &shard,
        &[
            ("lm_head.weight", "F8_E4M3", &[2, 2], [0, 4]),
            ("lm_head.weight_scale_inv", "F32", &[1, 1], [4, 8]),
            (
                "model.layers.0.self_attn.wq_a.weight",
                "F8_E4M3",
                &[2, 2],
                [8, 12],
            ),
            (
                "model.layers.0.self_attn.wq_a.weight_scale_inv",
                "F32",
                &[1, 1],
                [12, 16],
            ),
            (
                "model.layers.0.self_attn.wq_b.weight",
                "F8_E4M3",
                &[2, 2],
                [16, 20],
            ),
            (
                "model.layers.0.self_attn.wq_b.weight_scale",
                "F32",
                &[1],
                [20, 24],
            ),
            (
                "model.layers.0.self_attn.wkv.weight_scale_inv",
                "F32",
                &[1, 1],
                [24, 28],
            ),
        ],
        &[0; 28],
    )
    .expect("shard should be written");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local artifacts should load");

    assert_eq!(
        artifacts
            .safetensors()
            .quantized_linear_weight_spans()
            .expect("quantized linear weights should pair with scale tensors"),
        vec![
            SafetensorsQuantizedLinearWeightSpan {
                tensor_name: "lm_head.weight".to_string(),
                scale_tensor_name: "lm_head.weight_scale_inv".to_string(),
                scale_kind: SafetensorsQuantizedLinearScaleKind::WeightScaleInv,
                weight: SafetensorsTensorSpan {
                    path: shard.clone(),
                    metadata: SafetensorsTensorMetadata {
                        dtype: "F8_E4M3".to_string(),
                        shape: vec![2, 2],
                        data_offsets: [0, 4],
                    },
                    absolute_byte_offset: 8 + header_len as u64,
                    byte_len: 4,
                },
                scale: SafetensorsTensorSpan {
                    path: shard.clone(),
                    metadata: SafetensorsTensorMetadata {
                        dtype: "F32".to_string(),
                        shape: vec![1, 1],
                        data_offsets: [4, 8],
                    },
                    absolute_byte_offset: 8 + header_len as u64 + 4,
                    byte_len: 4,
                },
            },
            SafetensorsQuantizedLinearWeightSpan {
                tensor_name: "model.layers.0.self_attn.wq_a.weight".to_string(),
                scale_tensor_name: "model.layers.0.self_attn.wq_a.weight_scale_inv".to_string(),
                scale_kind: SafetensorsQuantizedLinearScaleKind::WeightScaleInv,
                weight: SafetensorsTensorSpan {
                    path: shard.clone(),
                    metadata: SafetensorsTensorMetadata {
                        dtype: "F8_E4M3".to_string(),
                        shape: vec![2, 2],
                        data_offsets: [8, 12],
                    },
                    absolute_byte_offset: 8 + header_len as u64 + 8,
                    byte_len: 4,
                },
                scale: SafetensorsTensorSpan {
                    path: shard.clone(),
                    metadata: SafetensorsTensorMetadata {
                        dtype: "F32".to_string(),
                        shape: vec![1, 1],
                        data_offsets: [12, 16],
                    },
                    absolute_byte_offset: 8 + header_len as u64 + 12,
                    byte_len: 4,
                },
            },
            SafetensorsQuantizedLinearWeightSpan {
                tensor_name: "model.layers.0.self_attn.wq_b.weight".to_string(),
                scale_tensor_name: "model.layers.0.self_attn.wq_b.weight_scale".to_string(),
                scale_kind: SafetensorsQuantizedLinearScaleKind::WeightScale,
                weight: SafetensorsTensorSpan {
                    path: shard.clone(),
                    metadata: SafetensorsTensorMetadata {
                        dtype: "F8_E4M3".to_string(),
                        shape: vec![2, 2],
                        data_offsets: [16, 20],
                    },
                    absolute_byte_offset: 8 + header_len as u64 + 16,
                    byte_len: 4,
                },
                scale: SafetensorsTensorSpan {
                    path: shard,
                    metadata: SafetensorsTensorMetadata {
                        dtype: "F32".to_string(),
                        shape: vec![1],
                        data_offsets: [20, 24],
                    },
                    absolute_byte_offset: 8 + header_len as u64 + 20,
                    byte_len: 4,
                },
            },
        ]
    );
    let checkpoint = artifacts
        .checkpoint_catalog()
        .expect("checkpoint catalog should build");
    assert_eq!(
        checkpoint
            .quantized_linear_weights()
            .span("model.layers.0.self_attn.wq_a.weight")
            .expect("checkpoint catalog should expose quantized linear pair")
            .scale_tensor_name,
        "model.layers.0.self_attn.wq_a.weight_scale_inv"
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn safetensors_manifest_groups_routed_expert_weight_spans() {
    let model_dir = temp_model_dir("routed-expert-weight-spans");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(model_dir.join("config.json"), deepseek_v4_config_json())
        .expect("config should be written");
    fs::write(
        model_dir.join("model.safetensors.index.json"),
        r#"{
  "weight_map": {
    "model.layers.0.ffn.experts.3.w1.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.ffn.experts.3.w2.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.ffn.experts.3.w3.weight": "model-00002-of-00002.safetensors",
    "model.layers.1.mlp.experts.42.gate_proj.weight": "model-00002-of-00002.safetensors",
    "model.layers.1.mlp.experts.42.up_proj.weight": "model-00002-of-00002.safetensors",
    "model.layers.1.mlp.experts.42.down_proj.weight": "model-00002-of-00002.safetensors",
    "model.layers.1.mlp.shared_experts.gate_proj.weight": "model-00002-of-00002.safetensors",
    "model.layers.1.self_attn.q_b_proj.weight": "model-00002-of-00002.safetensors"
  }
}"#,
    )
    .expect("index should be written");
    let first_shard = model_dir.join("model-00001-of-00002.safetensors");
    let first_header_len = write_safetensors_file(
        &first_shard,
        &[
            ("model.layers.0.ffn.experts.3.w1.weight", "U8", &[2], [0, 2]),
            ("model.layers.0.ffn.experts.3.w2.weight", "U8", &[3], [2, 5]),
        ],
        &[10, 11, 20, 21, 22],
    )
    .expect("first shard should be written");
    let second_shard = model_dir.join("model-00002-of-00002.safetensors");
    let second_header_len = write_safetensors_file(
        &second_shard,
        &[
            ("model.layers.0.ffn.experts.3.w3.weight", "U8", &[1], [0, 1]),
            (
                "model.layers.1.mlp.experts.42.down_proj.weight",
                "U8",
                &[4],
                [1, 5],
            ),
            (
                "model.layers.1.mlp.experts.42.gate_proj.weight",
                "U8",
                &[2],
                [5, 7],
            ),
            (
                "model.layers.1.mlp.experts.42.up_proj.weight",
                "U8",
                &[3],
                [7, 10],
            ),
            (
                "model.layers.1.mlp.shared_experts.gate_proj.weight",
                "U8",
                &[1],
                [10, 11],
            ),
            (
                "model.layers.1.self_attn.q_b_proj.weight",
                "U8",
                &[1],
                [11, 12],
            ),
        ],
        &[30, 40, 41, 42, 43, 50, 51, 60, 61, 62, 70, 80],
    )
    .expect("second shard should be written");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local artifacts should load");

    assert_eq!(
        artifacts
            .safetensors()
            .routed_expert_weight_spans()
            .expect("routed expert weight spans should parse"),
        vec![
            SafetensorsRoutedExpertWeightSpan {
                tensor_name: "model.layers.0.ffn.experts.3.w1.weight".to_string(),
                layer_id: 0,
                expert_id: 3,
                projection: SafetensorsRoutedExpertProjection::Gate,
                span: SafetensorsTensorSpan {
                    path: first_shard.clone(),
                    metadata: SafetensorsTensorMetadata {
                        dtype: "U8".to_string(),
                        shape: vec![2],
                        data_offsets: [0, 2],
                    },
                    absolute_byte_offset: 8 + first_header_len as u64,
                    byte_len: 2,
                },
            },
            SafetensorsRoutedExpertWeightSpan {
                tensor_name: "model.layers.0.ffn.experts.3.w2.weight".to_string(),
                layer_id: 0,
                expert_id: 3,
                projection: SafetensorsRoutedExpertProjection::Down,
                span: SafetensorsTensorSpan {
                    path: first_shard,
                    metadata: SafetensorsTensorMetadata {
                        dtype: "U8".to_string(),
                        shape: vec![3],
                        data_offsets: [2, 5],
                    },
                    absolute_byte_offset: 8 + first_header_len as u64 + 2,
                    byte_len: 3,
                },
            },
            SafetensorsRoutedExpertWeightSpan {
                tensor_name: "model.layers.0.ffn.experts.3.w3.weight".to_string(),
                layer_id: 0,
                expert_id: 3,
                projection: SafetensorsRoutedExpertProjection::Up,
                span: SafetensorsTensorSpan {
                    path: second_shard.clone(),
                    metadata: SafetensorsTensorMetadata {
                        dtype: "U8".to_string(),
                        shape: vec![1],
                        data_offsets: [0, 1],
                    },
                    absolute_byte_offset: 8 + second_header_len as u64,
                    byte_len: 1,
                },
            },
            SafetensorsRoutedExpertWeightSpan {
                tensor_name: "model.layers.1.mlp.experts.42.down_proj.weight".to_string(),
                layer_id: 1,
                expert_id: 42,
                projection: SafetensorsRoutedExpertProjection::Down,
                span: SafetensorsTensorSpan {
                    path: second_shard.clone(),
                    metadata: SafetensorsTensorMetadata {
                        dtype: "U8".to_string(),
                        shape: vec![4],
                        data_offsets: [1, 5],
                    },
                    absolute_byte_offset: 8 + second_header_len as u64 + 1,
                    byte_len: 4,
                },
            },
            SafetensorsRoutedExpertWeightSpan {
                tensor_name: "model.layers.1.mlp.experts.42.gate_proj.weight".to_string(),
                layer_id: 1,
                expert_id: 42,
                projection: SafetensorsRoutedExpertProjection::Gate,
                span: SafetensorsTensorSpan {
                    path: second_shard.clone(),
                    metadata: SafetensorsTensorMetadata {
                        dtype: "U8".to_string(),
                        shape: vec![2],
                        data_offsets: [5, 7],
                    },
                    absolute_byte_offset: 8 + second_header_len as u64 + 5,
                    byte_len: 2,
                },
            },
            SafetensorsRoutedExpertWeightSpan {
                tensor_name: "model.layers.1.mlp.experts.42.up_proj.weight".to_string(),
                layer_id: 1,
                expert_id: 42,
                projection: SafetensorsRoutedExpertProjection::Up,
                span: SafetensorsTensorSpan {
                    path: second_shard,
                    metadata: SafetensorsTensorMetadata {
                        dtype: "U8".to_string(),
                        shape: vec![3],
                        data_offsets: [7, 10],
                    },
                    absolute_byte_offset: 8 + second_header_len as u64 + 7,
                    byte_len: 3,
                },
            },
        ]
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn safetensors_manifest_groups_complete_routed_expert_weight_triplets() {
    let model_dir = temp_model_dir("routed-expert-weight-triplets");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(model_dir.join("config.json"), deepseek_v4_config_json())
        .expect("config should be written");
    fs::write(
        model_dir.join("model.safetensors.index.json"),
        r#"{
  "weight_map": {
    "model.layers.0.ffn.experts.3.w1.weight": "model.safetensors",
    "model.layers.0.ffn.experts.3.w2.weight": "model.safetensors",
    "model.layers.0.ffn.experts.3.w3.weight": "model.safetensors",
    "model.layers.1.mlp.experts.42.gate_proj.weight": "model.safetensors",
    "model.layers.1.mlp.experts.42.up_proj.weight": "model.safetensors",
    "model.layers.1.mlp.experts.42.down_proj.weight": "model.safetensors"
  }
}"#,
    )
    .expect("index should be written");
    let shard = model_dir.join("model.safetensors");
    let header_len = write_safetensors_file(
        &shard,
        &[
            ("model.layers.0.ffn.experts.3.w1.weight", "U8", &[2], [0, 2]),
            ("model.layers.0.ffn.experts.3.w2.weight", "U8", &[3], [2, 5]),
            ("model.layers.0.ffn.experts.3.w3.weight", "U8", &[1], [5, 6]),
            (
                "model.layers.1.mlp.experts.42.down_proj.weight",
                "U8",
                &[4],
                [6, 10],
            ),
            (
                "model.layers.1.mlp.experts.42.gate_proj.weight",
                "U8",
                &[2],
                [10, 12],
            ),
            (
                "model.layers.1.mlp.experts.42.up_proj.weight",
                "U8",
                &[3],
                [12, 15],
            ),
        ],
        &[10, 11, 20, 21, 22, 30, 40, 41, 42, 43, 50, 51, 60, 61, 62],
    )
    .expect("shard should be written");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local artifacts should load");

    assert_eq!(
        artifacts
            .safetensors()
            .routed_expert_weight_groups()
            .expect("routed expert groups should validate complete triplets"),
        vec![
            SafetensorsRoutedExpertWeightGroup {
                layer_id: 0,
                expert_id: 3,
                gate: SafetensorsTensorSpan {
                    path: shard.clone(),
                    metadata: SafetensorsTensorMetadata {
                        dtype: "U8".to_string(),
                        shape: vec![2],
                        data_offsets: [0, 2],
                    },
                    absolute_byte_offset: 8 + header_len as u64,
                    byte_len: 2,
                },
                up: SafetensorsTensorSpan {
                    path: shard.clone(),
                    metadata: SafetensorsTensorMetadata {
                        dtype: "U8".to_string(),
                        shape: vec![1],
                        data_offsets: [5, 6],
                    },
                    absolute_byte_offset: 8 + header_len as u64 + 5,
                    byte_len: 1,
                },
                down: SafetensorsTensorSpan {
                    path: shard.clone(),
                    metadata: SafetensorsTensorMetadata {
                        dtype: "U8".to_string(),
                        shape: vec![3],
                        data_offsets: [2, 5],
                    },
                    absolute_byte_offset: 8 + header_len as u64 + 2,
                    byte_len: 3,
                },
            },
            SafetensorsRoutedExpertWeightGroup {
                layer_id: 1,
                expert_id: 42,
                gate: SafetensorsTensorSpan {
                    path: shard.clone(),
                    metadata: SafetensorsTensorMetadata {
                        dtype: "U8".to_string(),
                        shape: vec![2],
                        data_offsets: [10, 12],
                    },
                    absolute_byte_offset: 8 + header_len as u64 + 10,
                    byte_len: 2,
                },
                up: SafetensorsTensorSpan {
                    path: shard.clone(),
                    metadata: SafetensorsTensorMetadata {
                        dtype: "U8".to_string(),
                        shape: vec![3],
                        data_offsets: [12, 15],
                    },
                    absolute_byte_offset: 8 + header_len as u64 + 12,
                    byte_len: 3,
                },
                down: SafetensorsTensorSpan {
                    path: shard,
                    metadata: SafetensorsTensorMetadata {
                        dtype: "U8".to_string(),
                        shape: vec![4],
                        data_offsets: [6, 10],
                    },
                    absolute_byte_offset: 8 + header_len as u64 + 6,
                    byte_len: 4,
                },
            },
        ]
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn safetensors_manifest_rejects_incomplete_routed_expert_weight_triplets() {
    let model_dir = temp_model_dir("routed-expert-weight-triplet-missing");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(model_dir.join("config.json"), deepseek_v4_config_json())
        .expect("config should be written");
    fs::write(
        model_dir.join("model.safetensors.index.json"),
        r#"{
  "weight_map": {
    "model.layers.0.ffn.experts.3.w1.weight": "model.safetensors",
    "model.layers.0.ffn.experts.3.w2.weight": "model.safetensors"
  }
}"#,
    )
    .expect("index should be written");
    let shard = model_dir.join("model.safetensors");
    write_safetensors_file(
        &shard,
        &[
            ("model.layers.0.ffn.experts.3.w1.weight", "U8", &[2], [0, 2]),
            ("model.layers.0.ffn.experts.3.w2.weight", "U8", &[3], [2, 5]),
        ],
        &[10, 11, 20, 21, 22],
    )
    .expect("shard should be written");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("local artifacts should load");

    let error = artifacts
        .safetensors()
        .routed_expert_weight_groups()
        .expect_err("missing expert projection should fail group validation");

    assert!(
        matches!(
            error,
            ModelArtifactError::InvalidSafetensorsData { ref path, ref message }
                if path == &shard
                    && message.contains("layer 0 expert 3")
                    && message.contains("missing up projection")
        ),
        "unexpected error: {error:?}"
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
  "eos_token_id": [32, 100001],
  "vocab_size": 129280,
  "max_position_embeddings": 163840,
  "num_hidden_layers": 43,
  "hidden_size": 7168,
  "intermediate_size": 18432,
  "moe_intermediate_size": 2048,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "num_experts_per_tok": 8,
  "first_k_dense_replace": 3,
  "moe_layer_freq": 2,
  "hc_mult": 4,
  "hc_sinkhorn_iters": 20,
  "rms_norm_eps": 1e-6,
  "rope_theta": 1000000.0,
  "hc_eps": 1e-6,
  "tie_word_embeddings": false,
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

struct EnvVarRestore {
    name: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvVarRestore {
    fn set(name: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
        let previous = std::env::var_os(name);
        unsafe {
            std::env::set_var(name, value);
        }
        Self { name, previous }
    }
}

impl Drop for EnvVarRestore {
    fn drop(&mut self) {
        if let Some(value) = &self.previous {
            unsafe {
                std::env::set_var(self.name, value);
            }
        } else {
            unsafe {
                std::env::remove_var(self.name);
            }
        }
    }
}

fn start_fake_hf_config_endpoint(config: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("fake HF endpoint should bind");
    let addr = listener
        .local_addr()
        .expect("fake HF endpoint should have address");

    std::thread::spawn(move || {
        for request_id in 0..2 {
            let (mut stream, _) = listener.accept().expect("fake HF request should connect");
            let request = read_http_request(&mut stream);
            assert!(
                request.starts_with("GET /zai-org/GLM-5-FP8/resolve/main/config.json "),
                "unexpected fake HF request: {request:?}"
            );

            let body = if request_id == 0 {
                &config.as_bytes()[..1]
            } else {
                config.as_bytes()
            };
            let status = if request_id == 0 {
                "206 Partial Content"
            } else {
                "200 OK"
            };
            let content_range = if request_id == 0 {
                format!("bytes 0-0/{}", config.len())
            } else {
                format!("bytes 0-{}/{}", config.len() - 1, config.len())
            };
            let response = format!(
                "HTTP/1.1 {status}\r\n\
                 x-repo-commit: abc123\r\n\
                 etag: \"config-json\"\r\n\
                 content-range: {content_range}\r\n\
                 content-length: {}\r\n\
                 connection: close\r\n\
                 \r\n",
                body.len()
            );
            stream
                .write_all(response.as_bytes())
                .expect("fake HF response headers should write");
            stream
                .write_all(body)
                .expect("fake HF response body should write");
        }
    });

    format!("http://{addr}")
}

fn read_http_request(stream: &mut TcpStream) -> String {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 1];
    while stream
        .read(&mut buffer)
        .expect("fake HF request should read")
        == 1
    {
        request.push(buffer[0]);
        if request.ends_with(b"\r\n\r\n") {
            break;
        }
    }
    String::from_utf8(request).expect("fake HF request should be utf8")
}

fn glm_moe_dsa_config_json() -> &'static str {
    r#"{
  "model_type": "glm_moe_dsa",
  "eos_token_id": [151329, 151336, 151338],
  "num_hidden_layers": 78,
  "num_attention_heads": 64,
  "num_key_value_heads": 64,
  "head_dim": 64
}"#
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
