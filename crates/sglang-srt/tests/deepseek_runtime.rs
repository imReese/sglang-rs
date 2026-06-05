use std::fs;
use std::path::{Path, PathBuf};

use sglang_srt::cache::{CachePageAllocator, CachePageId, RadixCache};
use sglang_srt::deepseek_runtime::{
    DeepSeekV4FeedForwardTensorDescriptors, DeepSeekV4Runtime, DeepSeekV4TensorPlacementKind,
    DeepSeekV4TensorShardLoadError, DeepSeekV4TensorShardPlanError, DeepSeekV4TensorShardSelection,
};
use sglang_srt::model_artifacts::{LocalModelArtifacts, SafetensorsTensorDecodeError};
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

#[derive(Default)]
struct UnfinishedPrefillWorker;

impl ModelWorker for UnfinishedPrefillWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        BatchGeneratedTokens::from_batch(
            batch,
            batch
                .requests()
                .iter()
                .map(|_| GeneratedToken::unfinished(vec![99]))
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
    assert_eq!(plan.request_spans().len(), 2);
    assert_eq!(
        plan.request_spans()[0].request_id(),
        &RequestId::from("pd-prefill")
    );
    assert_eq!(plan.request_spans()[0].token_range(), 0..2);
    assert_eq!(
        plan.request_spans()[0].prefix_cache_pages(),
        &[CachePageId::from(100), CachePageId::from(101)]
    );
    assert_eq!(plan.request_spans()[0].out_cache_pages().len(), 2);
    assert_eq!(plan.request_spans()[0].data_parallel_rank(), 2);
    assert_eq!(plan.request_spans()[0].bootstrap_room(), Some(34));
    assert_eq!(
        plan.request_spans()[1].request_id(),
        &RequestId::from("plain-prefill")
    );
    assert_eq!(plan.request_spans()[1].token_range(), 2..4);
    assert!(plan.request_spans()[1].prefix_cache_pages().is_empty());
    assert_eq!(plan.request_spans()[1].out_cache_pages().len(), 2);
    assert_eq!(plan.request_spans()[1].data_parallel_rank(), 0);
    assert_eq!(plan.request_spans()[1].bootstrap_room(), None);

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn deepseek_v4_forward_plan_handles_decode_batches_without_prefill_output_pages() {
    let model_dir = temp_model_dir("deepseek-runtime-decode-forward-plan");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_complete_deepseek_v4_checkpoint(&model_dir);
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("artifacts should load");
    let runtime =
        DeepSeekV4Runtime::from_local_model_artifacts(&artifacts).expect("runtime should build");

    let mut scheduler = Scheduler::new(UnfinishedPrefillWorker);
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("decode-a"),
        vec![1, 2, 3],
        SamplingParams::new(2),
    ));
    scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should produce an unfinished request");

    let batch = scheduler
        .next_decode_batch(1)
        .expect("decode batch should be available");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&batch);
    let plan = runtime.forward_plan(&worker_batch);

    assert_eq!(plan.forward_mode(), ForwardMode::Decode);
    assert_eq!(plan.request_ids(), &[RequestId::from("decode-a")]);
    assert_eq!(plan.input_ids(), &[99]);
    assert_eq!(plan.positions(), &[3]);
    assert!(plan.out_cache_pages().is_empty());
    assert_eq!(plan.request_spans().len(), 1);
    assert_eq!(plan.request_spans()[0].token_range(), 0..1);
    assert!(plan.request_spans()[0].out_cache_pages().is_empty());
    assert!(plan.request_spans()[0].prefix_cache_pages().is_empty());

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn deepseek_v4_runtime_builds_tensor_parallel_placement_plan() {
    let model_dir = temp_model_dir("deepseek-runtime-tp-placement");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_complete_deepseek_v4_checkpoint(&model_dir);
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("artifacts should load");
    let runtime =
        DeepSeekV4Runtime::from_local_model_artifacts(&artifacts).expect("runtime should build");

    let plan = runtime.tensor_parallel_placement_plan(8);

    assert_eq!(plan.tensor_parallel_size(), 8);
    assert_eq!(
        plan.kind_for("model.embed_tokens.weight"),
        Some(DeepSeekV4TensorPlacementKind::VocabParallel { axis: 0 })
    );
    assert_eq!(
        plan.kind_for("lm_head.weight"),
        Some(DeepSeekV4TensorPlacementKind::VocabParallel { axis: 0 })
    );
    assert_eq!(
        plan.kind_for("model.norm.weight"),
        Some(DeepSeekV4TensorPlacementKind::Replicated)
    );
    assert_eq!(
        plan.kind_for("model.hc_head_fn"),
        Some(DeepSeekV4TensorPlacementKind::Replicated)
    );
    assert_eq!(
        plan.kind_for("model.layers.0.self_attn.wq_a.weight"),
        Some(DeepSeekV4TensorPlacementKind::Replicated)
    );
    assert_eq!(
        plan.kind_for("model.layers.0.self_attn.wkv.weight"),
        Some(DeepSeekV4TensorPlacementKind::Replicated)
    );
    assert_eq!(
        plan.kind_for("model.layers.0.self_attn.wq_b.weight"),
        Some(DeepSeekV4TensorPlacementKind::ColumnParallel { axis: 0 })
    );
    assert_eq!(
        plan.kind_for("model.layers.0.self_attn.wo_a.weight"),
        Some(DeepSeekV4TensorPlacementKind::ColumnParallel { axis: 0 })
    );
    assert_eq!(
        plan.kind_for("model.layers.0.self_attn.wo_b.weight"),
        Some(DeepSeekV4TensorPlacementKind::RowParallel { axis: 1 })
    );
    assert_eq!(
        plan.kind_for("model.layers.0.mlp.gate.weight"),
        Some(DeepSeekV4TensorPlacementKind::Replicated)
    );
    assert_eq!(
        plan.kind_for("model.layers.0.ffn.experts.0.w1.weight"),
        Some(DeepSeekV4TensorPlacementKind::ColumnParallel { axis: 0 })
    );
    assert_eq!(
        plan.kind_for("model.layers.0.ffn.experts.0.w3.weight"),
        Some(DeepSeekV4TensorPlacementKind::ColumnParallel { axis: 0 })
    );
    assert_eq!(
        plan.kind_for("model.layers.0.ffn.experts.0.w2.weight"),
        Some(DeepSeekV4TensorPlacementKind::RowParallel { axis: 1 })
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn deepseek_v4_tensor_parallel_rank_plan_computes_tensor_slices() {
    let model_dir = temp_model_dir("deepseek-runtime-tp-rank-shards");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_complete_deepseek_v4_checkpoint_with_shapes(
        &model_dir,
        &[
            ("model.embed_tokens.weight", &[4, 2]),
            ("lm_head.weight", &[4, 2]),
            ("model.layers.0.self_attn.wq_b.weight", &[4, 2]),
            ("model.layers.0.self_attn.wo_a.weight", &[4, 2]),
            ("model.layers.0.self_attn.wo_b.weight", &[2, 4]),
            ("model.layers.0.ffn.experts.0.w1.weight", &[4, 2]),
            ("model.layers.0.ffn.experts.0.w2.weight", &[2, 4]),
            ("model.layers.0.ffn.experts.0.w3.weight", &[4, 2]),
        ],
    );
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("artifacts should load");
    let runtime =
        DeepSeekV4Runtime::from_local_model_artifacts(&artifacts).expect("runtime should build");

    let rank_plan = runtime
        .tensor_parallel_placement_plan(2)
        .rank_shard_plan(1)
        .expect("rank 1 shard plan should build");

    assert_eq!(rank_plan.tensor_parallel_size(), 2);
    assert_eq!(rank_plan.tensor_parallel_rank(), 1);
    assert_eq!(
        rank_plan.selection_for("model.norm.weight"),
        Some(DeepSeekV4TensorShardSelection::Full)
    );
    assert_eq!(
        rank_plan.selection_for("model.embed_tokens.weight"),
        Some(DeepSeekV4TensorShardSelection::Slice {
            axis: 0,
            range: 2..4
        })
    );
    assert_eq!(
        rank_plan.selection_for("model.layers.0.self_attn.wq_b.weight"),
        Some(DeepSeekV4TensorShardSelection::Slice {
            axis: 0,
            range: 2..4
        })
    );
    assert_eq!(
        rank_plan.selection_for("model.layers.0.self_attn.wo_b.weight"),
        Some(DeepSeekV4TensorShardSelection::Slice {
            axis: 1,
            range: 2..4
        })
    );
    assert_eq!(
        rank_plan.selection_for("model.layers.0.ffn.experts.0.w2.weight"),
        Some(DeepSeekV4TensorShardSelection::Slice {
            axis: 1,
            range: 2..4
        })
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn deepseek_v4_tensor_parallel_rank_plan_loads_sharded_tensor_bytes() {
    let model_dir = temp_model_dir("deepseek-runtime-tp-rank-load");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_complete_deepseek_v4_checkpoint_with_shapes(
        &model_dir,
        &[
            ("model.embed_tokens.weight", &[4, 2]),
            ("lm_head.weight", &[4, 2]),
            ("model.layers.0.self_attn.wq_b.weight", &[4, 2]),
            ("model.layers.0.self_attn.wo_a.weight", &[4, 2]),
            ("model.layers.0.self_attn.wo_b.weight", &[2, 4]),
            ("model.layers.0.ffn.experts.0.w1.weight", &[4, 2]),
            ("model.layers.0.ffn.experts.0.w2.weight", &[2, 4]),
            ("model.layers.0.ffn.experts.0.w3.weight", &[4, 2]),
        ],
    );
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("artifacts should load");
    let runtime =
        DeepSeekV4Runtime::from_local_model_artifacts(&artifacts).expect("runtime should build");

    let rank_plan = runtime
        .tensor_parallel_placement_plan(2)
        .rank_shard_plan(1)
        .expect("rank 1 shard plan should build");

    let embeddings = rank_plan
        .load_tensor_shard("model.embed_tokens.weight")
        .expect("embedding shard should load")
        .expect("embedding tensor should be planned");
    assert_eq!(embeddings.tensor_name(), "model.embed_tokens.weight");
    assert_eq!(embeddings.dtype(), "U8");
    assert_eq!(embeddings.shape(), &[2, 2]);
    assert_eq!(embeddings.bytes(), &[4, 5, 6, 7]);

    let final_norm = rank_plan
        .load_tensor_shard("model.norm.weight")
        .expect("replicated tensor should load")
        .expect("replicated tensor should be planned");
    assert_eq!(final_norm.shape(), &[1]);
    assert_eq!(final_norm.bytes(), &[8]);

    let row_parallel = rank_plan
        .load_tensor_shard("model.layers.0.self_attn.wo_b.weight")
        .expect("row-parallel tensor should load")
        .expect("row-parallel tensor should be planned");
    assert_eq!(row_parallel.shape(), &[2, 2]);
    assert_eq!(row_parallel.bytes(), &[42, 43, 46, 47]);
    assert_eq!(
        embeddings
            .decode_f32_values()
            .expect_err("U8 shard should not decode as f32"),
        SafetensorsTensorDecodeError::UnsupportedDtype {
            dtype: "U8".to_string()
        }
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn deepseek_v4_loaded_tensor_shards_decode_f32_values_after_slicing() {
    let model_dir = temp_model_dir("deepseek-runtime-tp-rank-decode-f32");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    let shape_overrides: &[(&str, &[usize])] = &[
        ("model.embed_tokens.weight", &[4, 2]),
        ("lm_head.weight", &[4, 2]),
        ("model.layers.0.self_attn.wq_b.weight", &[4, 2]),
        ("model.layers.0.self_attn.wo_a.weight", &[4, 2]),
        ("model.layers.0.self_attn.wo_b.weight", &[2, 4]),
        ("model.layers.0.ffn.experts.0.w1.weight", &[4, 2]),
        ("model.layers.0.ffn.experts.0.w2.weight", &[2, 4]),
        ("model.layers.0.ffn.experts.0.w3.weight", &[4, 2]),
    ];
    write_complete_deepseek_v4_checkpoint_with_shapes(&model_dir, shape_overrides);
    write_deepseek_v4_safetensors_with_shapes_and_dtype(&model_dir, shape_overrides, "F32");
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("artifacts should load");
    let runtime =
        DeepSeekV4Runtime::from_local_model_artifacts(&artifacts).expect("runtime should build");
    let loaded = runtime
        .load_tensor_parallel_shards(2)
        .expect("all TP rank shards should load");

    let rank1 = loaded.rank(1).expect("rank 1 should load");
    assert_eq!(
        rank1
            .tensor_shard("model.embed_tokens.weight")
            .expect("rank 1 embedding shard should exist")
            .decode_f32_values()
            .expect("F32 embedding shard should decode"),
        vec![4.0, 5.0, 6.0, 7.0]
    );
    assert_eq!(
        rank1
            .tensor_shard("model.layers.0.self_attn.wo_b.weight")
            .expect("rank 1 row-parallel shard should exist")
            .decode_f32_values()
            .expect("F32 row-parallel shard should decode"),
        vec![42.0, 43.0, 46.0, 47.0]
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn deepseek_v4_runtime_loads_all_tensor_parallel_rank_shards() {
    let model_dir = temp_model_dir("deepseek-runtime-load-all-tp-ranks");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_complete_deepseek_v4_checkpoint_with_shapes(
        &model_dir,
        &[
            ("model.embed_tokens.weight", &[4, 2]),
            ("lm_head.weight", &[4, 2]),
            ("model.layers.0.self_attn.wq_b.weight", &[4, 2]),
            ("model.layers.0.self_attn.wo_a.weight", &[4, 2]),
            ("model.layers.0.self_attn.wo_b.weight", &[2, 4]),
            ("model.layers.0.ffn.experts.0.w1.weight", &[4, 2]),
            ("model.layers.0.ffn.experts.0.w2.weight", &[2, 4]),
            ("model.layers.0.ffn.experts.0.w3.weight", &[4, 2]),
        ],
    );
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("artifacts should load");
    let runtime =
        DeepSeekV4Runtime::from_local_model_artifacts(&artifacts).expect("runtime should build");

    let loaded = runtime
        .load_tensor_parallel_shards(2)
        .expect("all TP rank shards should load");

    assert_eq!(loaded.tensor_parallel_size(), 2);
    assert_eq!(loaded.rank_count(), 2);
    assert_eq!(loaded.layer_count(), 1);
    assert_eq!(
        loaded
            .rank(0)
            .expect("rank 0 should load")
            .tensor_shard("model.embed_tokens.weight")
            .expect("rank 0 embedding shard should exist")
            .bytes(),
        &[0, 1, 2, 3]
    );
    assert_eq!(
        loaded
            .rank(1)
            .expect("rank 1 should load")
            .tensor_shard("model.embed_tokens.weight")
            .expect("rank 1 embedding shard should exist")
            .bytes(),
        &[4, 5, 6, 7]
    );
    assert_eq!(
        loaded
            .rank(0)
            .expect("rank 0 should load")
            .tensor_shard("model.layers.0.self_attn.wo_b.weight")
            .expect("rank 0 row-parallel shard should exist")
            .bytes(),
        &[40, 41, 44, 45]
    );
    assert_eq!(
        loaded
            .rank(1)
            .expect("rank 1 should load")
            .tensor_shard("model.layers.0.self_attn.wo_b.weight")
            .expect("rank 1 row-parallel shard should exist")
            .bytes(),
        &[42, 43, 46, 47]
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn deepseek_v4_runtime_rejects_zero_tensor_parallel_size_for_loaded_shards() {
    let model_dir = temp_model_dir("deepseek-runtime-load-zero-tp");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_complete_deepseek_v4_checkpoint(&model_dir);
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("artifacts should load");
    let runtime =
        DeepSeekV4Runtime::from_local_model_artifacts(&artifacts).expect("runtime should build");

    let error = runtime
        .load_tensor_parallel_shards(0)
        .expect_err("zero TP size should be rejected");

    assert_eq!(
        error,
        DeepSeekV4TensorShardLoadError::ShardPlan(
            DeepSeekV4TensorShardPlanError::RankOutOfBounds {
                tensor_parallel_rank: 0,
                tensor_parallel_size: 0,
            }
        )
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

fn write_complete_deepseek_v4_checkpoint(model_dir: &Path) {
    write_complete_deepseek_v4_checkpoint_with_shapes(model_dir, &[]);
}

fn write_complete_deepseek_v4_checkpoint_with_shapes(
    model_dir: &Path,
    shape_overrides: &[(&str, &[usize])],
) {
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
    if !shape_overrides.is_empty() {
        write_deepseek_v4_safetensors_with_shapes(model_dir, shape_overrides);
    }
}

fn write_deepseek_v4_safetensors_with_shapes(model_dir: &Path, overrides: &[(&str, &[usize])]) {
    write_deepseek_v4_safetensors_with_shapes_and_dtype(model_dir, overrides, "U8");
}

fn write_deepseek_v4_safetensors_with_shapes_and_dtype(
    model_dir: &Path,
    overrides: &[(&str, &[usize])],
    dtype: &str,
) {
    let tensor_shapes = complete_deepseek_v4_tensor_shapes(overrides);
    let mut offset = 0;
    let tensors = tensor_shapes
        .iter()
        .map(|(name, shape)| {
            let element_count = shape.iter().product::<usize>();
            let byte_len = match dtype {
                "F32" => element_count * 4,
                "U8" => element_count,
                _ => panic!("test fixture dtype {dtype} is not supported"),
            };
            let tensor = (*name, dtype, shape.as_slice(), [offset, offset + byte_len]);
            offset += byte_len;
            tensor
        })
        .collect::<Vec<_>>();
    let payload = match dtype {
        "F32" => (0..offset / 4)
            .flat_map(|index| (index as f32).to_le_bytes())
            .collect::<Vec<_>>(),
        "U8" => (0..offset)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>(),
        _ => panic!("test fixture dtype {dtype} is not supported"),
    };
    write_safetensors_file(&model_dir.join("model.safetensors"), &tensors, &payload)
        .expect("shardable safetensors should be written");
}

fn complete_deepseek_v4_tensor_shapes(
    overrides: &[(&str, &[usize])],
) -> Vec<(&'static str, Vec<usize>)> {
    let names = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.hc_head_fn",
        "model.hc_head_base",
        "model.hc_head_scale",
        "model.layers.0.self_attn.wq_a.weight",
        "model.layers.0.self_attn.wq_b.weight",
        "model.layers.0.self_attn.wkv.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.kv_norm.weight",
        "model.layers.0.self_attn.wo_a.weight",
        "model.layers.0.self_attn.wo_b.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.hc_attn_fn",
        "model.layers.0.hc_attn_base",
        "model.layers.0.hc_attn_scale",
        "model.layers.0.hc_ffn_fn",
        "model.layers.0.hc_ffn_base",
        "model.layers.0.hc_ffn_scale",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.ffn.experts.0.w1.weight",
        "model.layers.0.ffn.experts.0.w2.weight",
        "model.layers.0.ffn.experts.0.w3.weight",
    ];
    names
        .into_iter()
        .map(|name| {
            let shape = overrides
                .iter()
                .find(|(override_name, _)| *override_name == name)
                .map(|(_, shape)| shape.to_vec())
                .unwrap_or_else(|| {
                    if name.ends_with("_fn") {
                        vec![1, 1]
                    } else {
                        vec![1]
                    }
                });
            (name, shape)
        })
        .collect()
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
