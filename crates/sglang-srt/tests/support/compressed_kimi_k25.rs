use std::fs;
use std::path::Path;

pub(crate) fn write_artifacts(model_dir: &Path) {
    fs::create_dir_all(model_dir).expect("temp model directory should be created");
    fs::write(model_dir.join("config.json"), CONFIG)
        .expect("compressed Kimi config should be written");
    fs::write(model_dir.join("tokenizer.json"), TOKENIZER)
        .expect("compressed Kimi tokenizer should be written");

    let model_prefix = "language_model.model";
    let mut tensors = Vec::<Tensor>::new();
    let mut embeddings = vec![0.0_f32; 3 * 32];
    embeddings[32] = 1.0;
    embeddings[2 * 32 + 1] = 1.0;
    push_f32(
        &mut tensors,
        format!("{model_prefix}.embed_tokens.weight"),
        vec![3, 32],
        embeddings,
    );
    push_f32(
        &mut tensors,
        format!("{model_prefix}.norm.weight"),
        vec![32],
        vec![1.0; 32],
    );
    let mut lm_head = vec![0.0_f32; 3 * 32];
    lm_head[32 + 1] = 1.0;
    lm_head[2 * 32] = 1.0;
    push_f32(
        &mut tensors,
        "language_model.lm_head.weight".to_string(),
        vec![3, 32],
        lm_head,
    );
    for layer_id in 0..2 {
        let prefix = format!("{model_prefix}.layers.{layer_id}");
        push_f32(
            &mut tensors,
            format!("{prefix}.input_layernorm.weight"),
            vec![32],
            vec![1.0; 32],
        );
        push_f32(
            &mut tensors,
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![32],
            vec![1.0; 32],
        );
        for (suffix, shape) in [
            ("self_attn.q_a_proj.weight", vec![2, 32]),
            ("self_attn.q_a_layernorm.weight", vec![2]),
            ("self_attn.q_b_proj.weight", vec![3, 2]),
            ("self_attn.kv_a_proj_with_mqa.weight", vec![4, 32]),
            ("self_attn.kv_a_layernorm.weight", vec![2]),
            ("self_attn.kv_b_proj.weight", vec![3, 2]),
            ("self_attn.o_proj.weight", vec![32, 2]),
        ] {
            let count = shape.iter().product();
            push_f32(
                &mut tensors,
                format!("{prefix}.{suffix}"),
                shape,
                vec![0.0; count],
            );
        }
    }
    for (suffix, shape) in [
        ("gate_proj", vec![2, 32]),
        ("up_proj", vec![2, 32]),
        ("down_proj", vec![32, 2]),
    ] {
        let count = shape.iter().product();
        push_f32(
            &mut tensors,
            format!("{model_prefix}.layers.0.mlp.{suffix}.weight"),
            shape,
            vec![0.0; count],
        );
    }
    push_f32(
        &mut tensors,
        format!("{model_prefix}.layers.1.mlp.gate.weight"),
        vec![1, 32],
        vec![0.0; 32],
    );
    push_f32(
        &mut tensors,
        format!("{model_prefix}.layers.1.mlp.gate.e_score_correction_bias"),
        vec![1],
        vec![0.0],
    );
    for projection in ["w1", "w2", "w3"] {
        let base = format!("{model_prefix}.layers.1.mlp.experts.0.{projection}");
        tensors.push(Tensor {
            name: format!("{base}.weight_packed"),
            dtype: "I32",
            shape: vec![32, 4],
            bytes: (0..128)
                .flat_map(|_| 0x8888_8888_u32.to_le_bytes())
                .collect(),
        });
        push_f32(
            &mut tensors,
            format!("{base}.weight_scale"),
            vec![32, 1],
            vec![1.0; 32],
        );
        tensors.push(Tensor {
            name: format!("{base}.weight_shape"),
            dtype: "I64",
            shape: vec![2],
            bytes: [32_i64, 32]
                .into_iter()
                .flat_map(i64::to_le_bytes)
                .collect(),
        });
    }
    for projection in ["gate_proj", "up_proj", "down_proj"] {
        push_f32(
            &mut tensors,
            format!("{model_prefix}.layers.1.mlp.shared_experts.{projection}.weight"),
            vec![32, 32],
            vec![0.0; 32 * 32],
        );
    }
    write_safetensors(&model_dir.join("model.safetensors"), &tensors);
}

struct Tensor {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn push_f32(tensors: &mut Vec<Tensor>, name: String, shape: Vec<usize>, values: Vec<f32>) {
    tensors.push(Tensor {
        name,
        dtype: "F32",
        shape,
        bytes: values.into_iter().flat_map(f32::to_le_bytes).collect(),
    });
}

fn write_safetensors(path: &Path, tensors: &[Tensor]) {
    let mut payload = Vec::new();
    let mut fields = Vec::new();
    for tensor in tensors {
        let start = payload.len();
        payload.extend_from_slice(&tensor.bytes);
        let shape = tensor
            .shape
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        fields.push(format!(
            r#""{}":{{"dtype":"{}","shape":[{}],"data_offsets":[{},{}]}}"#,
            tensor.name,
            tensor.dtype,
            shape,
            start,
            payload.len()
        ));
    }
    let header = format!("{{{}}}", fields.join(","));
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&payload);
    fs::write(path, bytes).expect("compressed Kimi safetensors should be written");
}

const CONFIG: &str = r#"{
  "architectures": ["KimiK25ForConditionalGeneration"],
  "model_type": "kimi_k25",
  "encoder_only": false,
  "vision_config": {"model_type": "kimi_k25", "hidden_size": 4},
  "text_config": {
    "architectures": ["DeepseekV3ForCausalLM"],
    "model_type": "kimi_k2",
    "vocab_size": 3,
    "max_position_embeddings": 32,
    "num_hidden_layers": 2,
    "hidden_size": 32,
    "intermediate_size": 2,
    "num_attention_heads": 1,
    "hidden_act": "silu",
    "rms_norm_eps": 0.00001,
    "rope_theta": 10000.0,
    "rope_scaling": null,
    "attention_bias": false,
    "tie_word_embeddings": false,
    "q_lora_rank": 2,
    "kv_lora_rank": 2,
    "qk_nope_head_dim": 1,
    "qk_rope_head_dim": 2,
    "v_head_dim": 2,
    "moe_intermediate_size": 32,
    "n_routed_experts": 1,
    "n_shared_experts": 1,
    "num_experts_per_tok": 1,
    "routed_scaling_factor": 1.0,
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": true,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "num_nextn_predict_layers": 0,
    "quantization_config": {
      "quant_method": "compressed-tensors",
      "format": "pack-quantized",
      "quantization_status": "compressed",
      "kv_cache_scheme": null,
      "config_groups": {"group_0": {
        "targets": ["Linear"],
        "input_activations": null,
        "output_activations": null,
        "weights": {
          "type": "int", "num_bits": 4, "group_size": 32,
          "strategy": "group", "symmetric": true, "dynamic": false,
          "observer": "minmax", "actorder": null, "block_structure": null
        }
      }},
      "ignore": [
        "re:.*self_attn.*", "re:.*shared_experts.*",
        "re:.*mlp\\.(gate|up|gate_up|down)_proj.*", "re:.*lm_head.*",
        "re:vision_tower.*", "re:mm_projector.*"
      ]
    }
  }
}"#;

const TOKENIZER: &str = r#"{
  "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
  "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null, "decoder": null,
  "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "hello": 1, "world": 2}, "unk_token": "[UNK]"}
}"#;
