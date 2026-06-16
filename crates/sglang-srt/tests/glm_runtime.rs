use std::fs;
use std::path::{Path, PathBuf};

use sglang_srt::cache::{CachePageAllocator, CachePageId, RadixCache};
use sglang_srt::glm_runtime::{
    GlmMoeDsaF32CachedForwardModel, GlmMoeDsaF32KernelError, GlmMoeDsaF32KvPageStore,
    GlmMoeDsaFeedForwardTensorDescriptors, GlmMoeDsaRuntime, GlmMoeDsaTensorPlacementKind,
    GlmMoeDsaTensorShardSelection,
};
use sglang_srt::model_artifacts::LocalModelArtifacts;
use sglang_srt::model_artifacts::SafetensorsTensorDecodeError;
use sglang_srt::model_executor::{ForwardModel, ModelRunner, ModelWorkerBatch};
use sglang_srt::scheduler::{ScheduledRequest, Scheduler};
use sglang_srt::transfer::{
    DecodeBootstrapRegistry, FakeKvCacheTransferExecutor, KvCacheDtype, KvTransferModelWorker,
    LocalSnapshotTransferPdModelWorkers, MooncakeKvCacheMemoryProvider,
};
use sglang_srt::types::{BootstrapRoom, DisaggregatedParams, RequestId, SamplingParams};
use sglang_srt::worker::{BatchGeneratedTokens, GeneratedToken, ModelWorker};

#[derive(Default)]
struct NoopWorker;

impl ModelWorker for NoopWorker {
    fn generate_batch(
        &mut self,
        batch: &sglang_srt::scheduler::ScheduleBatch,
    ) -> BatchGeneratedTokens {
        BatchGeneratedTokens::from_batch(
            batch,
            batch
                .requests()
                .iter()
                .map(|_| GeneratedToken::finished(vec![0]))
                .collect(),
        )
        .expect("generated tokens should match batch")
    }
}

#[derive(Default)]
struct FirstTokenWorker;

impl ModelWorker for FirstTokenWorker {
    fn generate_batch(
        &mut self,
        batch: &sglang_srt::scheduler::ScheduleBatch,
    ) -> BatchGeneratedTokens {
        BatchGeneratedTokens::from_batch(
            batch,
            batch
                .requests()
                .iter()
                .map(|_| GeneratedToken::unfinished(vec![1]))
                .collect(),
        )
        .expect("generated tokens should match batch")
    }
}

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
fn glm_moe_dsa_f32_runtime_computes_embedding_norm_lm_head_logits_for_batch() {
    let model_dir = temp_model_dir("glm-runtime-f32-forward-logits");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_glm_moe_dsa_forward_fixture(&model_dir);
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let runtime =
        GlmMoeDsaRuntime::from_local_model_artifacts(&artifacts).expect("runtime should build");
    let f32_runtime = runtime
        .load_tensor_parallel_shards(2)
        .expect("all TP rank shards should load")
        .decode_f32_tensor_parallel_shards()
        .expect("F32 cache should decode");
    let mut scheduler = Scheduler::new(NoopWorker);
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("glm-forward-a"),
        vec![0, 1],
        SamplingParams::new(1),
    ));
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("glm-forward-b"),
        vec![0, 3],
        SamplingParams::new(1),
    ));
    let batch = scheduler
        .next_prefill_batch(8)
        .expect("prefill batch should be available");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&batch);

    let logits = f32_runtime
        .embedding_lm_head_logits(&worker_batch)
        .expect("embedding/norm/lm_head logits should compute");

    assert_eq!(
        logits,
        vec![vec![2.0, 4.0, 6.0, 8.0], vec![8.0, 16.0, 24.0, 32.0]]
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn glm_moe_dsa_f32_runtime_computes_tensor_parallel_attention_projection_output() {
    let model_dir = temp_model_dir("glm-runtime-f32-attention-projection");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_glm_moe_dsa_attention_projection_fixture(&model_dir);
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let runtime =
        GlmMoeDsaRuntime::from_local_model_artifacts(&artifacts).expect("runtime should build");
    let f32_runtime = runtime
        .load_tensor_parallel_shards(2)
        .expect("all TP rank shards should load")
        .decode_f32_tensor_parallel_shards()
        .expect("F32 cache should decode");

    let output = f32_runtime
        .attention_projection_output(0, &[1.0, 2.0])
        .expect("attention projection should compute");

    let q_rms = ((1.0_f32 * 1.0 + 4.0 * 4.0) / 2.0).sqrt();
    let q_lora = [1.0 / q_rms, 4.0 / q_rms];
    assert_close(output.q_lora(), &q_lora);
    assert_close(
        output.q(),
        &[q_lora[0], q_lora[1], 2.0 * q_lora[0], 2.0 * q_lora[1]],
    );

    let kv_rms = ((3.0_f32 * 3.0 + 8.0 * 8.0) / 2.0).sqrt();
    let kv_lora = [3.0 / kv_rms, 8.0 / kv_rms];
    assert_close(output.kv_lora(), &kv_lora);
    assert_close(
        output.kv(),
        &[
            kv_lora[0] + kv_lora[1],
            2.0 * kv_lora[0],
            2.0 * kv_lora[1],
            3.0 * kv_lora[0] + kv_lora[1],
        ],
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn glm_moe_dsa_f32_runtime_computes_tensor_parallel_causal_attention_output() {
    let model_dir = temp_model_dir("glm-runtime-f32-attention-output");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_glm_moe_dsa_attention_output_fixture(&model_dir);
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let runtime =
        GlmMoeDsaRuntime::from_local_model_artifacts(&artifacts).expect("runtime should build");
    let f32_runtime = runtime
        .load_tensor_parallel_shards(2)
        .expect("all TP rank shards should load")
        .decode_f32_tensor_parallel_shards()
        .expect("F32 cache should decode");

    let output = f32_runtime
        .attention_output(0, &[vec![1.0, 1.0], vec![1.0, -1.0]])
        .expect("causal attention output should compute");

    assert_eq!(output.len(), 2);
    assert_close(&output[0], &[5.0, 12.0]);
    assert_close(&output[1], &[3.0, 7.0]);

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn glm_moe_dsa_f32_runtime_applies_rope_to_causal_attention_scores() {
    let model_dir = temp_model_dir("glm-runtime-f32-attention-rope");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_glm_moe_dsa_rope_attention_fixture(&model_dir);
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let runtime =
        GlmMoeDsaRuntime::from_local_model_artifacts(&artifacts).expect("runtime should build");
    let f32_runtime = runtime
        .load_tensor_parallel_shards(1)
        .expect("single TP rank shard should load")
        .decode_f32_tensor_parallel_shards()
        .expect("F32 cache should decode");

    let output = f32_runtime
        .attention_output(0, &[vec![1.0, 0.0], vec![0.0, 1.0]])
        .expect("causal attention output should compute");

    assert_eq!(output.len(), 2);
    assert_close(&output[0], &[-10.0, 0.0]);
    assert_close(&output[1], &[expected_rope_attention_value(), 0.0]);

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn glm_moe_dsa_f32_runtime_reuses_kv_page_store_for_decode_attention() {
    let model_dir = temp_model_dir("glm-runtime-f32-attention-kv-cache");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_glm_moe_dsa_attention_output_fixture(&model_dir);
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let runtime =
        GlmMoeDsaRuntime::from_local_model_artifacts(&artifacts).expect("runtime should build");
    let f32_runtime = runtime
        .load_tensor_parallel_shards(2)
        .expect("all TP rank shards should load")
        .decode_f32_tensor_parallel_shards()
        .expect("F32 cache should decode");
    let mut kv_cache = GlmMoeDsaF32KvPageStore::default();

    let full_output = f32_runtime
        .attention_output(0, &[vec![1.0, 1.0], vec![1.0, -1.0], vec![1.0, 0.0]])
        .expect("full attention output should compute");
    f32_runtime
        .attention_output_with_kv_cache(
            0,
            &[vec![1.0, 1.0], vec![1.0, -1.0]],
            &[0, 1],
            &[CachePageId::from(0), CachePageId::from(1)],
            &[CachePageId::from(0), CachePageId::from(1)],
            &mut kv_cache,
        )
        .expect("prefill attention should populate KV cache");

    let decode_output = f32_runtime
        .attention_output_with_kv_cache(
            0,
            &[vec![1.0, 0.0]],
            &[2],
            &[CachePageId::from(2)],
            &[
                CachePageId::from(0),
                CachePageId::from(1),
                CachePageId::from(2),
            ],
            &mut kv_cache,
        )
        .expect("decode attention should reuse cached prefix pages");

    assert_eq!(decode_output.len(), 1);
    assert_close(&decode_output[0], &full_output[2]);
    assert!(kv_cache.contains(0, CachePageId::from(0)));
    assert!(kv_cache.contains(0, CachePageId::from(1)));
    assert!(kv_cache.contains(0, CachePageId::from(2)));

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn glm_moe_dsa_f32_runtime_reports_missing_kv_page_for_cached_attention() {
    let model_dir = temp_model_dir("glm-runtime-f32-attention-kv-cache-missing");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_glm_moe_dsa_attention_output_fixture(&model_dir);
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let runtime =
        GlmMoeDsaRuntime::from_local_model_artifacts(&artifacts).expect("runtime should build");
    let f32_runtime = runtime
        .load_tensor_parallel_shards(2)
        .expect("all TP rank shards should load")
        .decode_f32_tensor_parallel_shards()
        .expect("F32 cache should decode");
    let mut kv_cache = GlmMoeDsaF32KvPageStore::default();

    let error = f32_runtime
        .attention_output_with_kv_cache(
            0,
            &[vec![1.0, 0.0]],
            &[1],
            &[CachePageId::from(1)],
            &[CachePageId::from(0), CachePageId::from(1)],
            &mut kv_cache,
        )
        .expect_err("missing cached prefix page should be reported");

    assert_eq!(
        error,
        GlmMoeDsaF32KernelError::MissingKvCachePage {
            layer_id: 0,
            cache_page: 0
        }
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn glm_moe_dsa_f32_runtime_computes_dense_transformer_layer_output() {
    let model_dir = temp_model_dir("glm-runtime-f32-dense-transformer-layer");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_glm_moe_dsa_attention_output_fixture(&model_dir);
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let runtime =
        GlmMoeDsaRuntime::from_local_model_artifacts(&artifacts).expect("runtime should build");
    let f32_runtime = runtime
        .load_tensor_parallel_shards(2)
        .expect("all TP rank shards should load")
        .decode_f32_tensor_parallel_shards()
        .expect("F32 cache should decode");

    let output = f32_runtime
        .transformer_layer_output(0, &[vec![1.0, 1.0], vec![1.0, -1.0]], None)
        .expect("dense transformer layer should compute");

    assert_eq!(output.len(), 2);

    let residual0 = [6.0_f32, 13.0_f32];
    let residual0_rms = ((residual0[0] * residual0[0] + residual0[1] * residual0[1]) / 2.0).sqrt();
    let normalized0 = [residual0[0] / residual0_rms, residual0[1] / residual0_rms];
    let activation00 = silu(normalized0[0]) * (2.0 * normalized0[0]);
    let activation01 = silu(normalized0[1]) * (3.0 * normalized0[1]);
    assert_close(
        output[0].hidden_states(),
        &[
            5.0 * activation00 + 7.0 * activation01,
            11.0 * activation00 + 13.0 * activation01,
        ],
    );
    assert_close(output[0].residual(), &residual0);

    let residual1 = [4.0_f32, 6.0_f32];
    let residual1_rms = ((residual1[0] * residual1[0] + residual1[1] * residual1[1]) / 2.0).sqrt();
    let normalized1 = [residual1[0] / residual1_rms, residual1[1] / residual1_rms];
    let activation10 = silu(normalized1[0]) * (2.0 * normalized1[0]);
    let activation11 = silu(normalized1[1]) * (3.0 * normalized1[1]);
    assert_close(
        output[1].hidden_states(),
        &[
            5.0 * activation10 + 7.0 * activation11,
            11.0 * activation10 + 13.0 * activation11,
        ],
    );
    assert_close(output[1].residual(), &residual1);

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn glm_moe_dsa_f32_runtime_computes_transformer_lm_head_logits_for_batch() {
    let model_dir = temp_model_dir("glm-runtime-f32-transformer-logits");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_glm_moe_dsa_attention_output_fixture(&model_dir);
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let runtime =
        GlmMoeDsaRuntime::from_local_model_artifacts(&artifacts).expect("runtime should build");
    let f32_runtime = runtime
        .load_tensor_parallel_shards(2)
        .expect("all TP rank shards should load")
        .decode_f32_tensor_parallel_shards()
        .expect("F32 cache should decode");
    let mut scheduler = Scheduler::new(NoopWorker);
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("glm-transformer-a"),
        vec![0],
        SamplingParams::new(1),
    ));
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("glm-transformer-b"),
        vec![0, 1],
        SamplingParams::new(1),
    ));
    let batch = scheduler
        .next_prefill_batch(8)
        .expect("prefill batch should be available");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&batch);

    let logits = f32_runtime
        .transformer_lm_head_logits(&worker_batch)
        .expect("transformer/lm_head logits should compute");

    assert_eq!(logits.len(), 2);
    assert_close(
        &logits[0],
        &expected_dense_transformer_final_logits([6.0, 13.0]),
    );
    assert_close(
        &logits[1],
        &expected_dense_transformer_final_logits([4.0, 6.0]),
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn glm_moe_dsa_f32_runtime_uses_decode_sequence_history_for_transformer_logits() {
    let model_dir = temp_model_dir("glm-runtime-f32-transformer-decode-history");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_glm_moe_dsa_attention_output_fixture(&model_dir);
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let runtime =
        GlmMoeDsaRuntime::from_local_model_artifacts(&artifacts).expect("runtime should build");
    let f32_runtime = runtime
        .load_tensor_parallel_shards(2)
        .expect("all TP rank shards should load")
        .decode_f32_tensor_parallel_shards()
        .expect("F32 cache should decode");
    let mut scheduler = Scheduler::new(FirstTokenWorker);
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("glm-transformer-decode"),
        vec![0],
        SamplingParams::new(2),
    ));
    scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should dispatch first token");
    let decode_batch = scheduler
        .next_decode_batch(1)
        .expect("decode batch should be available");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&decode_batch);

    let logits = f32_runtime
        .transformer_lm_head_logits(&worker_batch)
        .expect("decode transformer/lm_head logits should compute from history");

    assert_eq!(worker_batch.input_ids(), &[1]);
    assert_eq!(worker_batch.positions(), &[1]);
    assert_eq!(worker_batch.sequence_token_ids(), &[0, 1]);
    assert_eq!(logits.len(), 1);
    assert_close(
        &logits[0],
        &expected_dense_transformer_final_logits([4.0, 6.0]),
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn glm_moe_dsa_cached_forward_model_populates_and_reuses_kv_pages_for_decode() {
    let model_dir = temp_model_dir("glm-runtime-f32-cached-forward-decode");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_glm_moe_dsa_attention_output_fixture(&model_dir);
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let runtime =
        GlmMoeDsaRuntime::from_local_model_artifacts(&artifacts).expect("runtime should build");
    let f32_runtime = runtime
        .load_tensor_parallel_shards(2)
        .expect("all TP rank shards should load")
        .decode_f32_tensor_parallel_shards()
        .expect("F32 cache should decode");
    let mut scheduler = Scheduler::with_cache_resources(
        ModelRunner::new(GlmMoeDsaF32CachedForwardModel::new(f32_runtime)),
        RadixCache::default(),
        CachePageAllocator::new(3),
    );
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("glm-cached-forward-decode"),
        vec![0],
        SamplingParams::new(2),
    ));

    scheduler
        .dispatch_prefill_batch(1)
        .expect("cached prefill should dispatch first token");

    assert!(
        scheduler
            .worker()
            .model()
            .kv_cache_contains(0, CachePageId::from(0)),
        "prefill forward should populate the request's prefix cache page"
    );

    let decode_batch = scheduler
        .next_decode_batch(1)
        .expect("decode batch should be available");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&decode_batch);
    let decode_output = scheduler
        .worker_mut()
        .model_mut()
        .forward(&worker_batch)
        .expect("cached decode forward should reuse the prefix page");

    assert_eq!(worker_batch.input_ids(), &[1]);
    assert_eq!(worker_batch.sequence_cache_pages().len(), 2);
    assert!(
        scheduler
            .worker()
            .model()
            .kv_cache_contains(0, CachePageId::from(1)),
        "decode forward should populate the newly allocated cache page"
    );
    assert_eq!(decode_output.logits().len(), 1);
    assert_close(
        &decode_output.logits()[0],
        &expected_dense_transformer_final_logits([4.0, 6.0]),
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn glm_moe_dsa_cached_forward_model_imports_transferred_prefix_pages_for_decode() {
    let model_dir = temp_model_dir("glm-runtime-f32-cached-forward-transfer-import");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_glm_moe_dsa_attention_output_fixture(&model_dir);
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let runtime =
        GlmMoeDsaRuntime::from_local_model_artifacts(&artifacts).expect("runtime should build");
    let f32_runtime = runtime
        .load_tensor_parallel_shards(2)
        .expect("all TP rank shards should load")
        .decode_f32_tensor_parallel_shards()
        .expect("F32 cache should decode");
    let mut decode_model = GlmMoeDsaF32CachedForwardModel::new(f32_runtime.clone());
    let mut prefill_scheduler = Scheduler::with_cache_resources(
        ModelRunner::new(GlmMoeDsaF32CachedForwardModel::new(f32_runtime)),
        RadixCache::default(),
        CachePageAllocator::new(3),
    );
    prefill_scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("glm-transferred-prefix-decode"),
        vec![0],
        SamplingParams::new(2),
    ));
    prefill_scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should populate its local page");

    let transferred_pages = prefill_scheduler
        .worker()
        .model()
        .export_kv_cache_pages(&[CachePageId::from(0)])
        .expect("prefill should export the page selected by the transfer plan");
    decode_model
        .import_kv_cache_pages(transferred_pages)
        .expect("decode worker should import transferred KV pages");

    let decode_batch = prefill_scheduler
        .next_decode_batch(1)
        .expect("decode batch should be available");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&decode_batch);
    let decode_output = decode_model
        .forward(&worker_batch)
        .expect("decode should read the imported prefix page");

    assert!(decode_model.kv_cache_contains(0, CachePageId::from(0)));
    assert!(decode_model.kv_cache_contains(0, CachePageId::from(1)));
    assert_close(
        &decode_output.logits()[0],
        &expected_dense_transformer_final_logits([4.0, 6.0]),
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn glm_moe_dsa_transfer_worker_forwards_kv_page_snapshots_to_inner_model_runner() {
    let model_dir = temp_model_dir("glm-runtime-f32-transfer-worker-snapshot-forwarding");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_glm_moe_dsa_attention_output_fixture(&model_dir);
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let runtime =
        GlmMoeDsaRuntime::from_local_model_artifacts(&artifacts).expect("runtime should build");
    let f32_runtime = runtime
        .load_tensor_parallel_shards(2)
        .expect("all TP rank shards should load")
        .decode_f32_tensor_parallel_shards()
        .expect("F32 cache should decode");
    let mut decode_worker = KvTransferModelWorker::new(
        ModelRunner::new(GlmMoeDsaF32CachedForwardModel::new(f32_runtime.clone())),
        DecodeBootstrapRegistry::default(),
        FakeKvCacheTransferExecutor::default(),
    );
    let mut prefill_scheduler = Scheduler::with_cache_resources(
        KvTransferModelWorker::new(
            ModelRunner::new(GlmMoeDsaF32CachedForwardModel::new(f32_runtime)),
            DecodeBootstrapRegistry::default(),
            FakeKvCacheTransferExecutor::default(),
        ),
        RadixCache::default(),
        CachePageAllocator::new(3),
    );
    prefill_scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("glm-transfer-worker-snapshot-decode"),
        vec![0],
        SamplingParams::new(2),
    ));
    prefill_scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should populate its local page");

    let transferred_pages = prefill_scheduler
        .worker()
        .export_kv_cache_pages(&[CachePageId::from(0)])
        .expect("transfer worker should expose model KV page snapshots");
    decode_worker
        .import_kv_cache_pages(transferred_pages)
        .expect("decode transfer worker should import model KV page snapshots");

    let decode_batch = prefill_scheduler
        .next_decode_batch(1)
        .expect("decode batch should be available");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&decode_batch);
    let decode_output = decode_worker
        .worker_mut()
        .model_mut()
        .forward(&worker_batch)
        .expect("decode should read imported prefix page through worker wrapper");

    assert_close(
        &decode_output.logits()[0],
        &expected_dense_transformer_final_logits([4.0, 6.0]),
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn glm_cached_forward_model_exposes_nonzero_mooncake_kv_memory_before_prefill() {
    let model_dir = temp_model_dir("glm-runtime-initial-mooncake-kv-memory");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_glm_moe_dsa_attention_output_fixture(&model_dir);
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let runtime =
        GlmMoeDsaRuntime::from_local_model_artifacts(&artifacts).expect("runtime should build");
    let f32_runtime = runtime
        .load_tensor_parallel_shards(2)
        .expect("all TP rank shards should load")
        .decode_f32_tensor_parallel_shards()
        .expect("F32 cache should decode");
    let model = GlmMoeDsaF32CachedForwardModel::new(f32_runtime);

    let memory = model
        .mooncake_kv_cache_memory()
        .expect("GLM model should expose initial KV memory");

    assert!(!memory.regions().is_empty());
    assert!(memory.regions()[0].base_addr > 0);
    assert_eq!(memory.regions()[0].byte_len, memory.page_size_bytes());
    assert!(memory.page_size_bytes() > 0);

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn glm_cached_forward_model_exposes_nonzero_mooncake_kv_memory_after_prefill() {
    let model_dir = temp_model_dir("glm-runtime-mooncake-kv-memory");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_glm_moe_dsa_attention_output_fixture(&model_dir);
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let runtime =
        GlmMoeDsaRuntime::from_local_model_artifacts(&artifacts).expect("runtime should build");
    let f32_runtime = runtime
        .load_tensor_parallel_shards(2)
        .expect("all TP rank shards should load")
        .decode_f32_tensor_parallel_shards()
        .expect("F32 cache should decode");
    let mut runner = ModelRunner::new(GlmMoeDsaF32CachedForwardModel::new(f32_runtime));
    let batch = single_request_prefill_worker_batch(vec![0], vec![CachePageId::from(0)]);

    runner
        .model_mut()
        .forward(&batch)
        .expect("prefill should populate KV");
    let memory = runner
        .model()
        .mooncake_kv_cache_memory()
        .expect("GLM model should expose KV memory");

    assert!(!memory.regions().is_empty());
    assert!(memory.regions()[0].base_addr > 0);
    assert!(memory.regions()[0].byte_len >= memory.page_size_bytes());
    assert!(memory.page_size_bytes() > 0);

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn glm_moe_dsa_local_snapshot_pd_workers_transfer_prefill_pages_before_decode() {
    let model_dir = temp_model_dir("glm-runtime-f32-local-snapshot-pd-workers");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_glm_moe_dsa_attention_output_fixture(&model_dir);
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let runtime =
        GlmMoeDsaRuntime::from_local_model_artifacts(&artifacts).expect("runtime should build");
    let f32_runtime = runtime
        .load_tensor_parallel_shards(2)
        .expect("all TP rank shards should load")
        .decode_f32_tensor_parallel_shards()
        .expect("F32 cache should decode");
    let workers = LocalSnapshotTransferPdModelWorkers::new(
        ModelRunner::new(GlmMoeDsaF32CachedForwardModel::new(f32_runtime.clone())),
        ModelRunner::new(GlmMoeDsaF32CachedForwardModel::new(f32_runtime)),
    );
    let mut scheduler =
        Scheduler::with_cache_resources(workers, RadixCache::default(), CachePageAllocator::new(3));
    scheduler.enqueue(
        ScheduledRequest::new(
            RequestId::from("glm-local-snapshot-pd"),
            vec![0],
            SamplingParams::new(2),
        )
        .with_disaggregated_params(Some(test_disaggregated_params(42))),
    );

    let prefill_outputs = scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should compute and transfer KV pages");

    assert_eq!(prefill_outputs[0].token_ids, vec![1]);
    assert_eq!(
        scheduler
            .worker()
            .last_transfer_summary()
            .expect("prefill should record a transfer summary")
            .submitted_spans(),
        1
    );
    assert!(
        scheduler
            .worker()
            .decode()
            .model()
            .kv_cache_contains(0, CachePageId::from(0)),
        "prefill page should be imported into the decode worker before decode dispatch"
    );

    let decode_outputs = scheduler
        .dispatch_decode_batch(1)
        .expect("decode should use transferred prefix page");

    assert_eq!(decode_outputs[0].token_ids, vec![1]);
    assert!(decode_outputs[0].finished);

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn glm_moe_dsa_f32_runtime_computes_tensor_parallel_dense_mlp_output() {
    let model_dir = temp_model_dir("glm-runtime-f32-dense-mlp");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_glm_moe_dsa_dense_mlp_fixture(&model_dir);
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let runtime =
        GlmMoeDsaRuntime::from_local_model_artifacts(&artifacts).expect("runtime should build");
    let f32_runtime = runtime
        .load_tensor_parallel_shards(2)
        .expect("all TP rank shards should load")
        .decode_f32_tensor_parallel_shards()
        .expect("F32 cache should decode");

    let output = f32_runtime
        .dense_mlp_output(0, &[1.0, 2.0])
        .expect("dense MLP should compute");

    let activation0 = silu(1.0) * 2.0;
    let activation1 = silu(2.0) * 6.0;
    assert_close(
        &output,
        &[
            5.0 * activation0 + 7.0 * activation1,
            11.0 * activation0 + 13.0 * activation1,
        ],
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn glm_moe_dsa_f32_runtime_computes_dense_feed_forward_layer_output() {
    let model_dir = temp_model_dir("glm-runtime-f32-dense-feed-forward-layer");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_glm_moe_dsa_dense_mlp_fixture(&model_dir);
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let runtime =
        GlmMoeDsaRuntime::from_local_model_artifacts(&artifacts).expect("runtime should build");
    let f32_runtime = runtime
        .load_tensor_parallel_shards(2)
        .expect("all TP rank shards should load")
        .decode_f32_tensor_parallel_shards()
        .expect("F32 cache should decode");

    let output = f32_runtime
        .feed_forward_layer_output(0, &[1.0, 1.0], Some(&[3.0, 1.0]))
        .expect("dense feed-forward layer should compute");

    let residual = [4.0_f32, 2.0_f32];
    let inv_rms = ((residual[0] * residual[0] + residual[1] * residual[1]) / 2.0_f32)
        .sqrt()
        .recip();
    let normalized = [residual[0] * inv_rms, residual[1] * inv_rms];
    let activation0 = silu(normalized[0]) * (2.0 * normalized[0]);
    let activation1 = silu(normalized[1]) * (3.0 * normalized[1]);
    assert_close(
        output.hidden_states(),
        &[
            5.0 * activation0 + 7.0 * activation1,
            11.0 * activation0 + 13.0 * activation1,
        ],
    );
    assert_close(output.residual(), &residual);

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn glm_moe_dsa_f32_runtime_computes_tensor_parallel_moe_mlp_output() {
    let model_dir = temp_model_dir("glm-runtime-f32-moe-mlp");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_glm_moe_dsa_moe_mlp_fixture(&model_dir);
    let artifacts =
        LocalModelArtifacts::from_model_path(&model_dir).expect("GLM artifacts should load");
    let runtime =
        GlmMoeDsaRuntime::from_local_model_artifacts(&artifacts).expect("runtime should build");
    let f32_runtime = runtime
        .load_tensor_parallel_shards(2)
        .expect("all TP rank shards should load")
        .decode_f32_tensor_parallel_shards()
        .expect("F32 cache should decode");

    let output = f32_runtime
        .moe_mlp_output(0, &[1.0, 2.0])
        .expect("MoE MLP should compute");

    let top1_weight = 4.0_f32.exp() / (1.0_f32.exp() + 4.0_f32.exp());
    let routed_scale = 2.0;
    let activation0 = silu(2.0) * 3.0;
    let activation1 = silu(1.0) * 8.0;
    assert_close(
        &output,
        &[
            routed_scale * top1_weight * (5.0 * activation0 + 7.0 * activation1),
            routed_scale * top1_weight * (11.0 * activation0 + 13.0 * activation1),
        ],
    );

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

fn write_glm_moe_dsa_forward_fixture(model_dir: &Path) {
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "glm_moe_dsa",
  "vocab_size": 4,
  "num_hidden_layers": 1,
  "hidden_size": 4,
  "num_attention_heads": 2,
  "num_key_value_heads": 2,
  "head_dim": 2,
  "rms_norm_eps": 0.0,
  "n_routed_experts": 1,
  "first_k_dense_replace": 0,
  "moe_layer_freq": 1
}"#,
    )
    .expect("config should be written");

    let mut cursor = 0_usize;
    let mut tensors = Vec::new();
    let mut payload = Vec::new();
    for (name, values, shape) in [
        (
            "model.embed_tokens.weight",
            vec![
                0.0, 0.0, 0.0, 0.0, // token 0
                1.0, 0.0, 0.0, 0.0, // token 1
                0.0, 1.0, 0.0, 0.0, // token 2
                0.0, 0.0, 0.0, 2.0, // token 3
            ],
            vec![4, 4],
        ),
        ("model.norm.weight", vec![1.0, 1.0, 1.0, 1.0], vec![4]),
        (
            "lm_head.weight",
            vec![
                1.0, 0.0, 0.0, 4.0, // token 0 logit
                2.0, 0.0, 0.0, 8.0, // token 1 logit
                3.0, 0.0, 0.0, 12.0, // token 2 logit
                4.0, 0.0, 0.0, 16.0, // token 3 logit
            ],
            vec![4, 4],
        ),
    ] {
        let start = cursor;
        for value in values.into_iter().map(|value| value as f32) {
            payload.extend_from_slice(&value.to_le_bytes());
            cursor += 4;
        }
        tensors.push((name, "F32", shape, [start, cursor]));
    }
    for (name, shape) in complete_glm_moe_dsa_tensor_shapes()
        .into_iter()
        .filter(|(name, _)| {
            !matches!(
                *name,
                "model.embed_tokens.weight" | "model.norm.weight" | "lm_head.weight"
            )
        })
    {
        let element_count = shape.iter().product::<usize>();
        let start = cursor;
        payload.resize(payload.len() + element_count * 4, 0);
        cursor += element_count * 4;
        tensors.push((name, "F32", shape, [start, cursor]));
    }

    write_safetensors_file(&model_dir.join("model.safetensors"), &tensors, &payload)
        .expect("safetensors shard should be written");
}

fn write_glm_moe_dsa_attention_projection_fixture(model_dir: &Path) {
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "glm_moe_dsa",
  "vocab_size": 2,
  "num_hidden_layers": 1,
  "hidden_size": 2,
  "intermediate_size": 2,
  "num_attention_heads": 2,
  "num_key_value_heads": 2,
  "head_dim": 2,
  "rms_norm_eps": 0.0,
  "n_routed_experts": 1,
  "first_k_dense_replace": 1,
  "moe_layer_freq": 1
}"#,
    )
    .expect("config should be written");

    let tensors = [
        (
            "model.embed_tokens.weight",
            vec![2, 2],
            vec![0.0, 0.0, 0.0, 0.0],
        ),
        ("model.norm.weight", vec![2], vec![1.0, 1.0]),
        ("lm_head.weight", vec![2, 2], vec![0.0, 0.0, 0.0, 0.0]),
        (
            "model.layers.0.self_attn.q_a_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 2.0],
        ),
        (
            "model.layers.0.self_attn.q_a_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.layers.0.self_attn.q_b_proj.weight",
            vec![4, 2],
            vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
        ),
        (
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            vec![2, 2],
            vec![3.0, 0.0, 0.0, 4.0],
        ),
        (
            "model.layers.0.self_attn.kv_a_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.layers.0.self_attn.kv_b_proj.weight",
            vec![4, 2],
            vec![1.0, 1.0, 2.0, 0.0, 0.0, 2.0, 3.0, 1.0],
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            vec![2, 4],
            vec![0.0; 8],
        ),
        (
            "model.layers.0.input_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
    ];
    let mut cursor = 0_usize;
    let mut metadata = Vec::new();
    let mut payload = Vec::new();
    for (name, shape, values) in tensors {
        let start = cursor;
        for value in values.into_iter().map(|value| value as f32) {
            payload.extend_from_slice(&value.to_le_bytes());
            cursor += 4;
        }
        metadata.push((name, "F32", shape, [start, cursor]));
    }

    write_safetensors_file(&model_dir.join("model.safetensors"), &metadata, &payload)
        .expect("safetensors shard should be written");
}

fn write_glm_moe_dsa_rope_attention_fixture(model_dir: &Path) {
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "glm_moe_dsa",
  "vocab_size": 2,
  "num_hidden_layers": 1,
  "hidden_size": 2,
  "intermediate_size": 2,
  "num_attention_heads": 1,
  "num_key_value_heads": 1,
  "head_dim": 3,
  "qk_nope_head_dim": 1,
  "qk_rope_head_dim": 2,
  "v_head_dim": 1,
  "rms_norm_eps": 0.0,
  "n_routed_experts": 1,
  "first_k_dense_replace": 1,
  "moe_layer_freq": 1
}"#,
    )
    .expect("config should be written");

    let tensors = [
        (
            "model.embed_tokens.weight",
            vec![2, 2],
            vec![0.0, 0.0, 0.0, 0.0],
        ),
        ("model.norm.weight", vec![2], vec![1.0, 1.0]),
        ("lm_head.weight", vec![2, 2], vec![0.0, 0.0, 0.0, 0.0]),
        (
            "model.layers.0.self_attn.q_a_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "model.layers.0.self_attn.q_a_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.layers.0.self_attn.q_b_proj.weight",
            vec![3, 2],
            vec![0.0, 0.0, 0.0, std::f32::consts::FRAC_1_SQRT_2, 0.0, 0.0],
        ),
        (
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            vec![3, 2],
            vec![-1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        ),
        (
            "model.layers.0.self_attn.kv_a_layernorm.weight",
            vec![1],
            vec![1.0],
        ),
        (
            "model.layers.0.self_attn.kv_b_proj.weight",
            vec![2, 1],
            vec![0.0, 10.0],
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            vec![2, 1],
            vec![1.0, 0.0],
        ),
        (
            "model.layers.0.input_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
    ];
    let mut cursor = 0_usize;
    let mut metadata = Vec::new();
    let mut payload = Vec::new();
    for (name, shape, values) in tensors {
        let start = cursor;
        for value in values.into_iter().map(|value| value as f32) {
            payload.extend_from_slice(&value.to_le_bytes());
            cursor += 4;
        }
        metadata.push((name, "F32", shape, [start, cursor]));
    }

    write_safetensors_file(&model_dir.join("model.safetensors"), &metadata, &payload)
        .expect("safetensors shard should be written");
}

fn write_glm_moe_dsa_attention_output_fixture(model_dir: &Path) {
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "glm_moe_dsa",
  "vocab_size": 2,
  "num_hidden_layers": 1,
  "hidden_size": 2,
  "intermediate_size": 2,
  "num_attention_heads": 2,
  "num_key_value_heads": 2,
  "head_dim": 1,
  "qk_nope_head_dim": 1,
  "qk_rope_head_dim": 0,
  "v_head_dim": 1,
  "rms_norm_eps": 0.0,
  "n_routed_experts": 1,
  "first_k_dense_replace": 1,
  "moe_layer_freq": 1
}"#,
    )
    .expect("config should be written");

    let tensors = [
        (
            "model.embed_tokens.weight",
            vec![2, 2],
            vec![1.0, 1.0, 1.0, -1.0],
        ),
        ("model.norm.weight", vec![2], vec![1.0, 1.0]),
        ("lm_head.weight", vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        (
            "model.layers.0.self_attn.q_a_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "model.layers.0.self_attn.q_a_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.layers.0.self_attn.q_b_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "model.layers.0.self_attn.kv_a_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.layers.0.self_attn.kv_b_proj.weight",
            vec![4, 2],
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            vec![2, 2],
            vec![2.0, 3.0, 5.0, 7.0],
        ),
        (
            "model.layers.0.input_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            vec![2, 2],
            vec![2.0, 0.0, 0.0, 3.0],
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            vec![2, 2],
            vec![5.0, 7.0, 11.0, 13.0],
        ),
    ];
    let mut cursor = 0_usize;
    let mut metadata = Vec::new();
    let mut payload = Vec::new();
    for (name, shape, values) in tensors {
        let start = cursor;
        for value in values.into_iter().map(|value| value as f32) {
            payload.extend_from_slice(&value.to_le_bytes());
            cursor += 4;
        }
        metadata.push((name, "F32", shape, [start, cursor]));
    }

    write_safetensors_file(&model_dir.join("model.safetensors"), &metadata, &payload)
        .expect("safetensors shard should be written");
}

fn write_glm_moe_dsa_dense_mlp_fixture(model_dir: &Path) {
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "glm_moe_dsa",
  "vocab_size": 2,
  "num_hidden_layers": 1,
  "hidden_size": 2,
  "intermediate_size": 2,
  "num_attention_heads": 1,
  "num_key_value_heads": 1,
  "head_dim": 2,
  "rms_norm_eps": 0.0,
  "n_routed_experts": 1,
  "first_k_dense_replace": 1,
  "moe_layer_freq": 1
}"#,
    )
    .expect("config should be written");

    let tensors = [
        (
            "model.embed_tokens.weight",
            vec![2, 2],
            vec![0.0, 0.0, 0.0, 0.0],
        ),
        ("model.norm.weight", vec![2], vec![1.0, 1.0]),
        ("lm_head.weight", vec![2, 2], vec![0.0, 0.0, 0.0, 0.0]),
        (
            "model.layers.0.self_attn.q_a_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.self_attn.q_a_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.layers.0.self_attn.q_b_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.self_attn.kv_a_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.layers.0.self_attn.kv_b_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.input_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            vec![2, 2],
            vec![2.0, 0.0, 0.0, 3.0],
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            vec![2, 2],
            vec![5.0, 7.0, 11.0, 13.0],
        ),
    ];
    let mut cursor = 0_usize;
    let mut metadata = Vec::new();
    let mut payload = Vec::new();
    for (name, shape, values) in tensors {
        let start = cursor;
        for value in values.into_iter().map(|value| value as f32) {
            payload.extend_from_slice(&value.to_le_bytes());
            cursor += 4;
        }
        metadata.push((name, "F32", shape, [start, cursor]));
    }

    write_safetensors_file(&model_dir.join("model.safetensors"), &metadata, &payload)
        .expect("safetensors shard should be written");
}

fn write_glm_moe_dsa_moe_mlp_fixture(model_dir: &Path) {
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "glm_moe_dsa",
  "vocab_size": 2,
  "num_hidden_layers": 1,
  "hidden_size": 2,
  "moe_intermediate_size": 2,
  "num_attention_heads": 1,
  "num_key_value_heads": 1,
  "head_dim": 2,
  "rms_norm_eps": 0.0,
  "n_routed_experts": 2,
  "num_experts_per_tok": 1,
  "norm_topk_prob": false,
  "routed_scaling_factor": 2.0,
  "first_k_dense_replace": 0,
  "moe_layer_freq": 1
}"#,
    )
    .expect("config should be written");

    let tensors = [
        (
            "model.embed_tokens.weight",
            vec![2, 2],
            vec![0.0, 0.0, 0.0, 0.0],
        ),
        ("model.norm.weight", vec![2], vec![1.0, 1.0]),
        ("lm_head.weight", vec![2, 2], vec![0.0, 0.0, 0.0, 0.0]),
        (
            "model.layers.0.self_attn.q_a_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.self_attn.q_a_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.layers.0.self_attn.q_b_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.self_attn.kv_a_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.layers.0.self_attn.kv_b_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.input_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![2],
            vec![1.0, 1.0],
        ),
        (
            "model.layers.0.mlp.gate.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 2.0],
        ),
        (
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.mlp.experts.0.up_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.mlp.experts.0.down_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.mlp.experts.1.gate_proj.weight",
            vec![2, 2],
            vec![0.0, 1.0, 1.0, 0.0],
        ),
        (
            "model.layers.0.mlp.experts.1.up_proj.weight",
            vec![2, 2],
            vec![3.0, 0.0, 0.0, 4.0],
        ),
        (
            "model.layers.0.mlp.experts.1.down_proj.weight",
            vec![2, 2],
            vec![5.0, 7.0, 11.0, 13.0],
        ),
    ];
    let mut cursor = 0_usize;
    let mut metadata = Vec::new();
    let mut payload = Vec::new();
    for (name, shape, values) in tensors {
        let start = cursor;
        for value in values.into_iter().map(|value| value as f32) {
            payload.extend_from_slice(&value.to_le_bytes());
            cursor += 4;
        }
        metadata.push((name, "F32", shape, [start, cursor]));
    }

    write_safetensors_file(&model_dir.join("model.safetensors"), &metadata, &payload)
        .expect("safetensors shard should be written");
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn expected_dense_transformer_final_logits(residual: [f32; 2]) -> Vec<f32> {
    let residual_rms = ((residual[0] * residual[0] + residual[1] * residual[1]) / 2.0).sqrt();
    let normalized = [residual[0] / residual_rms, residual[1] / residual_rms];
    let activation0 = silu(normalized[0]) * (2.0 * normalized[0]);
    let activation1 = silu(normalized[1]) * (3.0 * normalized[1]);
    let hidden = [
        5.0 * activation0 + 7.0 * activation1,
        11.0 * activation0 + 13.0 * activation1,
    ];
    let final_hidden = [hidden[0] + residual[0], hidden[1] + residual[1]];
    let final_rms =
        ((final_hidden[0] * final_hidden[0] + final_hidden[1] * final_hidden[1]) / 2.0).sqrt();
    vec![final_hidden[0] / final_rms, final_hidden[1] / final_rms]
}

fn expected_rope_attention_value() -> f32 {
    let scale = (3.0_f32).sqrt().recip();
    let key0_score = 1.0_f32.cos() * scale;
    let key1_score = scale;
    let max_score = key0_score.max(key1_score);
    let key0_exp = (key0_score - max_score).exp();
    let key1_exp = (key1_score - max_score).exp();
    let exp_sum = key0_exp + key1_exp;
    let key0_weight = key0_exp / exp_sum;
    let key1_weight = key1_exp / exp_sum;
    key0_weight * -10.0 + key1_weight * 10.0
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "index {index}: expected {expected}, got {actual}"
        );
    }
}

fn test_disaggregated_params(bootstrap_room: BootstrapRoom) -> DisaggregatedParams {
    DisaggregatedParams {
        bootstrap_host: "127.0.0.1".to_string(),
        bootstrap_port: 8998,
        bootstrap_room,
    }
}

fn single_request_prefill_worker_batch(
    input_ids: Vec<u32>,
    out_cache_pages: Vec<CachePageId>,
) -> ModelWorkerBatch {
    let mut scheduler = Scheduler::with_cache_resources(
        NoopWorker,
        RadixCache::default(),
        CachePageAllocator::new(out_cache_pages.len().max(input_ids.len())),
    );
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("glm-mooncake-memory"),
        input_ids,
        SamplingParams::new(1),
    ));
    let batch = scheduler
        .next_prefill_batch(1)
        .expect("prefill batch should build");
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&batch);
    assert_eq!(worker_batch.out_cache_pages(), out_cache_pages);
    worker_batch
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
