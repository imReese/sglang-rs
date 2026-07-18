use sglang_srt::cache::{CachePageAllocator, RadixCache};
use sglang_srt::cuda_attention::{
    CudaBf16PagedAttentionExecutor, CudaPagedAttentionError, CudaPagedAttentionMetadata,
};
use sglang_srt::kv_cache::{PagedKvCacheLayout, PagedKvCacheLayoutError};
use sglang_srt::model_executor::ModelWorkerBatch;
use sglang_srt::scheduler::{ScheduleBatch, ScheduledRequest, Scheduler};
use sglang_srt::transfer::{KvCacheDtype, KvCacheRuntimeLayout};
use sglang_srt::types::{RequestId, SamplingParams};
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

fn runtime_layout(dtype: KvCacheDtype) -> KvCacheRuntimeLayout {
    let page_size = 4;
    let num_layers = 2;
    let kv_heads = 2;
    let head_dim = 8;
    let kv_tensors_per_token = 2;
    let element_bytes = 2;
    let bytes_per_token = num_layers * kv_heads * head_dim * kv_tensors_per_token * element_bytes;
    KvCacheRuntimeLayout {
        dtype,
        page_size,
        num_layers,
        kv_heads,
        head_dim,
        kv_tensors_per_token,
        bytes_per_token,
        page_size_bytes: page_size * bytes_per_token,
    }
}

fn two_request_prefill_batch() -> ModelWorkerBatch {
    let mut scheduler = Scheduler::with_cache_resources(
        NoopWorker,
        RadixCache::default(),
        CachePageAllocator::new(8),
    );
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("req-a"),
        vec![10, 11, 12],
        SamplingParams::new(1),
    ));
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("req-b"),
        vec![20, 21],
        SamplingParams::new(1),
    ));
    let batch = scheduler
        .next_prefill_batch(2)
        .expect("two-request prefill should schedule");
    ModelWorkerBatch::from_schedule_batch(&batch)
}

#[test]
fn scheduler_batch_builds_causal_attention_metadata_over_physical_slots() {
    let batch = two_request_prefill_batch();
    let pool_layout = PagedKvCacheLayout::new(runtime_layout(KvCacheDtype::Bfloat16), 2)
        .expect("BF16 KV pool layout should be valid");

    let metadata = CudaPagedAttentionMetadata::from_model_worker_batch(&batch, pool_layout)
        .expect("scheduler batch should map to CUDA attention metadata");

    assert_eq!(metadata.query_request_indices(), &[0, 0, 0, 1, 1]);
    assert_eq!(metadata.query_sequence_lengths(), &[1, 2, 3, 1, 2]);
    assert_eq!(metadata.request_slot_offsets(), &[0, 3, 5]);
    assert_eq!(metadata.sequence_slots(), &[0, 1, 2, 3, 4]);
    let plan =
        CudaBf16PagedAttentionExecutor::plan(pool_layout, 1, &metadata, 4, 8.0_f32.sqrt().recip())
            .expect("GQA attention plan should derive from the KV pool");
    assert_eq!(plan.query_count(), 5);
    assert_eq!(plan.request_count(), 2);
    assert_eq!(plan.sequence_slot_count(), 5);
    assert_eq!(plan.query_head_count(), 4);
    assert_eq!(plan.kv_head_count(), 2);
    assert_eq!(plan.head_dim(), 8);
}

#[test]
fn metadata_rejects_scheduler_slots_outside_the_allocated_layout() {
    let batch = two_request_prefill_batch();
    let too_small_pool = PagedKvCacheLayout::new(runtime_layout(KvCacheDtype::Bfloat16), 1)
        .expect("small KV pool layout should still be structurally valid");

    assert_eq!(
        CudaPagedAttentionMetadata::from_model_worker_batch(&batch, too_small_pool),
        Err(CudaPagedAttentionError::KvLayout(
            PagedKvCacheLayoutError::BatchSlotOutOfRange {
                batch_index: 4,
                slot_index: 4,
                slot_count: 4,
            }
        ))
    );
}

#[test]
fn attention_plan_rejects_non_bf16_kv_without_cuda_fallback() {
    let batch = two_request_prefill_batch();
    let bf16_pool = PagedKvCacheLayout::new(runtime_layout(KvCacheDtype::Bfloat16), 2)
        .expect("BF16 KV pool layout should be valid");
    let metadata = CudaPagedAttentionMetadata::from_model_worker_batch(&batch, bf16_pool)
        .expect("metadata should validate");
    let fp8_pool = PagedKvCacheLayout::new(runtime_layout(KvCacheDtype::Fp8E4M3), 2)
        .expect("FP8 layout shape should be valid for capability validation");

    assert_eq!(
        CudaBf16PagedAttentionExecutor::plan(fp8_pool, 0, &metadata, 4, 1.0),
        Err(CudaPagedAttentionError::UnsupportedKvDtype {
            actual: KvCacheDtype::Fp8E4M3,
            required: KvCacheDtype::Bfloat16,
        })
    );
}

#[test]
fn attention_plan_rejects_asymmetric_kv_before_kernel_launch() {
    let batch = two_request_prefill_batch();
    let metadata_pool = PagedKvCacheLayout::new(runtime_layout(KvCacheDtype::Bfloat16), 2)
        .expect("uniform metadata pool should be valid");
    let metadata = CudaPagedAttentionMetadata::from_model_worker_batch(&batch, metadata_pool)
        .expect("metadata should validate");
    let asymmetric_runtime = KvCacheRuntimeLayout {
        dtype: KvCacheDtype::Bfloat16,
        page_size: 4,
        num_layers: 2,
        kv_heads: 2,
        head_dim: 8,
        kv_tensors_per_token: 2,
        bytes_per_token: 96,
        page_size_bytes: 384,
    };
    let asymmetric_pool = PagedKvCacheLayout::new_with_tensor_pair(asymmetric_runtime, 2, 32, 16)
        .expect("asymmetric pool should be representable by the common layout");

    assert_eq!(
        CudaBf16PagedAttentionExecutor::plan(asymmetric_pool, 0, &metadata, 4, 1.0),
        Err(CudaPagedAttentionError::KvLayout(
            PagedKvCacheLayoutError::UnevenKvPairCopy {
                key_token_bytes: 32,
                value_token_bytes: 16,
            }
        ))
    );
}
