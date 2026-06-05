use std::fs;
use std::path::{Path, PathBuf};

use sglang_srt::cache::{CachePageAllocator, CachePageId, RadixCache};
use sglang_srt::deepseek_runtime::{DeepSeekV4FeedForwardTensorDescriptors, DeepSeekV4Runtime};
use sglang_srt::model_artifacts::LocalModelArtifacts;
use sglang_srt::model_executor::ModelWorkerBatch;
use sglang_srt::scheduler::{ForwardMode, ScheduleBatch, ScheduledRequest, Scheduler};
use sglang_srt::transfer::KvCacheDtype;
use sglang_srt::types::{DisaggregatedParams, RequestId, SamplingParams};
use sglang_srt::worker::{BatchGeneratedTokens, GeneratedToken, ModelWorker};

#[derive(Default)]
struct NoopWorker;

impl ModelWorker for NoopWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        BatchGeneratedTokens::from_batch(
            batch,
            batch
                .requests()
                .iter()
                .map(|_| GeneratedToken::finished(vec![0]))
                .collect(),
        )
        .expect("output shape should match batch")
    }
}

#[test]
fn deepseek_v4_runtime_builds_forward_plan_from_scheduler_batch() {
    let model_dir = temp_model_dir("deepseek-runtime-forward-plan");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_complete_deepseek_v4_checkpoint(&model_dir);
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("artifacts should load");
    let runtime =
        DeepSeekV4Runtime::from_local_model_artifacts(&artifacts).expect("runtime should build");

    let mut cache = RadixCache::default();
    cache
        .insert(&[10, 11], &[CachePageId::from(100), CachePageId::from(101)])
        .expect("prefix should insert");
    let mut scheduler =
        Scheduler::with_cache_resources(NoopWorker, cache, CachePageAllocator::new(16));
    scheduler.enqueue(
        ScheduledRequest::new(
            RequestId::from("pd-prefill"),
            vec![10, 11, 12, 13],
            SamplingParams::new(1),
        )
        .with_disaggregated_params(Some(DisaggregatedParams {
            bootstrap_host: "10.0.0.8".to_string(),
            bootstrap_port: 8200,
            bootstrap_room: 34,
        }))
        .with_data_parallel_rank(2),
    );
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("plain-prefill"),
        vec![20, 21],
        SamplingParams::new(1),
    ));

    let batch = scheduler
        .next_prefill_batch(8)
        .expect("prefill batch should be available");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&batch);
    let plan = runtime.forward_plan(&worker_batch);

    assert_eq!(runtime.layer_count(), 1);
    assert_eq!(
        runtime
            .kv_cache_layout()
            .token_size_bytes(KvCacheDtype::Fp8E4M3)
            .expect("DeepSeek V4 packed layout should size"),
        130
    );
    assert_eq!(
        runtime.root_tensors().lm_head().tensor_name(),
        "lm_head.weight"
    );
    let layer0 = runtime
        .layers()
        .first()
        .expect("layer descriptor should exist");
    let DeepSeekV4FeedForwardTensorDescriptors::Moe { routed_experts, .. } = layer0.feed_forward()
    else {
        panic!("fixture layer should use MoE feed-forward descriptors");
    };
    assert_eq!(routed_experts[0].expert_id(), 0);
    assert_eq!(
        routed_experts[0].gate().tensor_name(),
        "model.layers.0.ffn.experts.0.w1.weight"
    );
    assert_eq!(
        routed_experts[0].up().tensor_name(),
        "model.layers.0.ffn.experts.0.w3.weight"
    );
    assert_eq!(
        routed_experts[0].down().tensor_name(),
        "model.layers.0.ffn.experts.0.w2.weight"
    );
    assert_eq!(plan.forward_mode(), ForwardMode::Prefill);
    assert_eq!(
        plan.request_ids(),
        &[
            RequestId::from("pd-prefill"),
            RequestId::from("plain-prefill")
        ]
    );
    assert_eq!(plan.input_ids(), &[12, 13, 20, 21]);
    assert_eq!(plan.positions(), &[2, 3, 0, 1]);
    assert_eq!(plan.sequence_lengths(), &[4, 2]);
    assert_eq!(plan.request_offsets(), &[0, 2]);
    assert_eq!(plan.cached_token_counts(), &[2, 0]);
    assert_eq!(plan.input_token_counts(), &[2, 2]);
    assert_eq!(plan.out_cache_pages().len(), 4);
    assert_eq!(plan.data_parallel_ranks(), &[2, 0]);
    assert_eq!(plan.bootstrap_rooms(), &[Some(34), None]);

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

fn write_complete_deepseek_v4_checkpoint(model_dir: &Path) {
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "deepseek_v4",
  "num_hidden_layers": 1,
  "hidden_size": 1,
  "hc_mult": 1,
  "n_routed_experts": 1,
  "first_k_dense_replace": 0,
  "moe_layer_freq": 1,
  "num_key_value_heads": 1,
  "qk_nope_head_dim": 64,
  "qk_rope_head_dim": 32,
  "v_head_dim": 64
}"#,
    )
    .expect("config should be written");
    fs::write(
        model_dir.join("model.safetensors.index.json"),
        r#"{
  "weight_map": {
    "model.embed_tokens.weight": "model.safetensors",
    "model.norm.weight": "model.safetensors",
    "lm_head.weight": "model.safetensors",
    "model.hc_head_fn": "model.safetensors",
    "model.hc_head_base": "model.safetensors",
    "model.hc_head_scale": "model.safetensors",
    "model.layers.0.self_attn.wq_a.weight": "model.safetensors",
    "model.layers.0.self_attn.wq_b.weight": "model.safetensors",
    "model.layers.0.self_attn.wkv.weight": "model.safetensors",
    "model.layers.0.self_attn.q_norm.weight": "model.safetensors",
    "model.layers.0.self_attn.kv_norm.weight": "model.safetensors",
    "model.layers.0.self_attn.wo_a.weight": "model.safetensors",
    "model.layers.0.self_attn.wo_b.weight": "model.safetensors",
    "model.layers.0.input_layernorm.weight": "model.safetensors",
    "model.layers.0.post_attention_layernorm.weight": "model.safetensors",
    "model.layers.0.hc_attn_fn": "model.safetensors",
    "model.layers.0.hc_attn_base": "model.safetensors",
    "model.layers.0.hc_attn_scale": "model.safetensors",
    "model.layers.0.hc_ffn_fn": "model.safetensors",
    "model.layers.0.hc_ffn_base": "model.safetensors",
    "model.layers.0.hc_ffn_scale": "model.safetensors",
    "model.layers.0.mlp.gate.weight": "model.safetensors",
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
}

fn write_safetensors_file(
    path: &Path,
    tensors: &[(&str, &str, &[usize], [usize; 2])],
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
