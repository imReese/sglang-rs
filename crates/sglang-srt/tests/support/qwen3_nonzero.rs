use std::fs;
use std::path::Path;

pub fn write_model(path: &Path) {
    fs::create_dir_all(path).expect("create Qwen fixture");
    let config = serde_json::json!({
        "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
        "vocab_size": 8, "num_hidden_layers": 2, "hidden_size": 8,
        "intermediate_size": 12, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 4, "hidden_act": "silu", "attention_bias": false,
        "rms_norm_eps": 0.000001, "rope_theta": 10000.0,
        "max_position_embeddings": 128, "tie_word_embeddings": false
    });
    fs::write(
        path.join("config.json"),
        serde_json::to_vec(&config).expect("config JSON"),
    )
    .expect("write config");
    let tokenizer = serde_json::json!({
        "version":"1.0", "truncation":null, "padding":null, "added_tokens":[],
        "normalizer":null, "pre_tokenizer":{"type":"Whitespace"}, "post_processor":null,
        "decoder":null, "model":{"type":"WordLevel", "unk_token":"[UNK]",
            "vocab":{"[UNK]":0,"hello":1,"world":2,"a":3,"b":4,"c":5,"d":6,"e":7}}
    });
    fs::write(
        path.join("tokenizer.json"),
        serde_json::to_vec(&tokenizer).expect("tokenizer JSON"),
    )
    .expect("write tokenizer");
    let mut descriptors = vec![
        ("model.embed_tokens.weight".to_string(), vec![8, 8]),
        ("model.norm.weight".to_string(), vec![8]),
        ("lm_head.weight".to_string(), vec![8, 8]),
    ];
    for layer in 0..2 {
        for (name, shape) in [
            ("self_attn.q_proj.weight", vec![8, 8]),
            ("self_attn.k_proj.weight", vec![4, 8]),
            ("self_attn.v_proj.weight", vec![4, 8]),
            ("self_attn.o_proj.weight", vec![8, 8]),
            ("self_attn.q_norm.weight", vec![4]),
            ("self_attn.k_norm.weight", vec![4]),
            ("input_layernorm.weight", vec![8]),
            ("post_attention_layernorm.weight", vec![8]),
            ("mlp.gate_proj.weight", vec![12, 8]),
            ("mlp.up_proj.weight", vec![12, 8]),
            ("mlp.down_proj.weight", vec![8, 12]),
        ] {
            descriptors.push((format!("model.layers.{layer}.{name}"), shape));
        }
    }
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for (name, shape) in descriptors {
        let offset = payload.len();
        let seed = name.bytes().map(usize::from).sum::<usize>();
        for i in 0..shape.iter().product::<usize>() {
            let value = if shape.len() == 1 {
                1.0
            } else {
                ((i * 7 + seed) as f32 * 0.31).sin() * 0.23 + 0.013
            };
            payload.extend(value.to_le_bytes());
        }
        header.insert(name, serde_json::json!({"dtype":"F32", "shape":shape, "data_offsets":[offset,payload.len()]}));
    }
    let mut header = serde_json::to_vec(&header).expect("safetensors header");
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(payload);
    fs::write(path.join("model.safetensors"), bytes).expect("write nonzero weights");
}
