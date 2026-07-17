use sglang_kernel::cublas::CudaBlas;
use sglang_kernel::cuda_kernels::{CudaF32Kernels, CudaRmsNormLaunch, CudaSiluMulLaunch};
use sglang_kernel::cuda_kv_kernels::CudaKvPairCopyKernels;
use sglang_srt::backend::{ComputeCapability, CudaBackend};
use sglang_srt::cache::{CachePageAllocator, RadixCache};
use sglang_srt::cuda_attention::{
    CudaBf16PagedAttentionExecutor, CudaPagedAttentionForward, CudaPagedAttentionMetadata,
};
use sglang_srt::cuda_kv_cache::{CudaKvCachePool, CudaKvSlotGatherLaunch, CudaKvSlotScatterLaunch};
use sglang_srt::model_executor::ModelWorkerBatch;
use sglang_srt::scheduler::{ScheduleBatch, ScheduledRequest, Scheduler};
use sglang_srt::transfer::{
    KvCacheDtype, KvCacheMemoryLocation, KvCacheMemoryProvider, KvCacheRuntimeLayout,
    MooncakeKvCacheMemoryExt,
};
#[cfg(feature = "mooncake-link")]
use sglang_srt::transfer::{
    MooncakeTransferEngineConfig, RegisteredMooncakeKvCacheMemory,
    SharedLinkedMooncakeTransferEngine,
};
use sglang_srt::types::{RequestId, SamplingParams};
use sglang_srt::worker::{BatchGeneratedTokens, GeneratedToken, ModelWorker};

fn cuda_test_layout() -> KvCacheRuntimeLayout {
    let page_size = 16;
    let num_layers = 2;
    let kv_heads = 8;
    let head_dim = 128;
    let kv_tensors_per_token = 2;
    let bytes_per_token = num_layers * kv_heads * head_dim * kv_tensors_per_token * 2;
    KvCacheRuntimeLayout {
        dtype: KvCacheDtype::Bfloat16,
        page_size,
        num_layers,
        kv_heads,
        head_dim,
        kv_tensors_per_token,
        bytes_per_token,
        page_size_bytes: page_size * bytes_per_token,
    }
}

fn attention_test_layout() -> KvCacheRuntimeLayout {
    let page_size = 4;
    let num_layers = 1;
    let kv_heads = 2;
    let head_dim = 4;
    let kv_tensors_per_token = 2;
    let bytes_per_token = num_layers * kv_heads * head_dim * kv_tensors_per_token * 2;
    KvCacheRuntimeLayout {
        dtype: KvCacheDtype::Bfloat16,
        page_size,
        num_layers,
        kv_heads,
        head_dim,
        kv_tensors_per_token,
        bytes_per_token,
        page_size_bytes: page_size * bytes_per_token,
    }
}

#[derive(Default)]
struct AttentionTestWorker;

impl ModelWorker for AttentionTestWorker {
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

fn attention_test_batch() -> ModelWorkerBatch {
    let mut scheduler = Scheduler::with_cache_resources(
        AttentionTestWorker,
        RadixCache::default(),
        CachePageAllocator::new(8),
    );
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("attention-a"),
        vec![10, 11, 12],
        SamplingParams::new(1),
    ));
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("attention-b"),
        vec![20, 21],
        SamplingParams::new(1),
    ));
    let batch = scheduler
        .next_prefill_batch(2)
        .expect("attention acceptance batch should schedule");
    ModelWorkerBatch::from_schedule_batch(&batch)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn bytes_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("f32 chunk should be exact")))
        .collect()
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| f32_to_bf16(*value).to_ne_bytes())
        .collect()
}

fn bytes_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(size_of::<u16>())
        .map(|chunk| {
            let bits = u16::from_ne_bytes(chunk.try_into().expect("BF16 chunk should be exact"));
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

fn assert_f32_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index} differs: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

fn strided_byte_rows(
    row_count: usize,
    row_bytes: usize,
    row_stride_bytes: usize,
    seed: usize,
) -> (Vec<u8>, Vec<Vec<u8>>) {
    let byte_len = (row_count - 1) * row_stride_bytes + row_bytes;
    let mut storage = vec![0_u8; byte_len];
    let mut rows = Vec::with_capacity(row_count);
    for row_index in 0..row_count {
        let row = (0..row_bytes)
            .map(|byte_index| ((seed + row_index * 37 + byte_index * 13) % 251) as u8)
            .collect::<Vec<_>>();
        let start = row_index * row_stride_bytes;
        storage[start..start + row_bytes].copy_from_slice(&row);
        rows.push(row);
    }
    (storage, rows)
}

fn assert_strided_byte_rows(storage: &[u8], row_stride_bytes: usize, expected_rows: &[Vec<u8>]) {
    for (row_index, expected) in expected_rows.iter().enumerate() {
        let start = row_index * row_stride_bytes;
        assert_eq!(&storage[start..start + expected.len()], expected);
    }
}

struct ReferenceAttentionShape {
    query_head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    scale: f32,
}

fn reference_paged_attention(
    metadata: &CudaPagedAttentionMetadata,
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    shape: ReferenceAttentionShape,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(queries.len());
    let query_heads_per_kv_head = shape.query_head_count / shape.kv_head_count;
    for query_index in 0..metadata.query_count() {
        let request_index = metadata.query_request_indices()[query_index] as usize;
        let sequence_start = metadata.request_slot_offsets()[request_index] as usize;
        let sequence_length = metadata.query_sequence_lengths()[query_index] as usize;
        for query_head in 0..shape.query_head_count {
            let kv_head = query_head / query_heads_per_kv_head;
            let query_start = (query_index * shape.query_head_count + query_head) * shape.head_dim;
            let query = &queries[query_start..query_start + shape.head_dim];
            let scores = (0..sequence_length)
                .map(|sequence_index| {
                    let kv_start = ((sequence_start + sequence_index) * shape.kv_head_count
                        + kv_head)
                        * shape.head_dim;
                    query
                        .iter()
                        .zip(&keys[kv_start..kv_start + shape.head_dim])
                        .map(|(query, key)| query * key)
                        .sum::<f32>()
                        * shape.scale
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = scores
                .iter()
                .map(|score| (score - maximum).exp())
                .collect::<Vec<_>>();
            let denominator = weights.iter().sum::<f32>();
            for dimension in 0..shape.head_dim {
                output.push(
                    weights
                        .iter()
                        .enumerate()
                        .map(|(sequence_index, weight)| {
                            let value_index =
                                ((sequence_start + sequence_index) * shape.kv_head_count + kv_head)
                                    * shape.head_dim
                                    + dimension;
                            weight * values[value_index]
                        })
                        .sum::<f32>()
                        / denominator,
                );
            }
        }
    }
    output
}

#[test]
#[ignore = "requires a CUDA device and NVIDIA driver"]
fn cuda_backend_round_trips_page_major_device_kv_memory() {
    let backend = CudaBackend::initialize(0).expect("CUDA backend should initialize");
    let capability = backend.capabilities();
    let ComputeCapability::Cuda(_) = capability.compute_capability else {
        panic!("CUDA backend must report CUDA compute capability");
    };

    let memory_before = backend
        .context()
        .memory_info()
        .expect("CUDA memory info should be available");
    let mut kv_cache = CudaKvCachePool::allocate(backend.context(), cuda_test_layout(), 16)
        .expect("CUDA KV cache should allocate a page-major device pool");
    let pattern = (0..cuda_test_layout().page_size_bytes)
        .map(|offset| (offset % 251) as u8)
        .collect::<Vec<_>>();
    kv_cache
        .write_page(3, &pattern)
        .expect("page write should reach CUDA memory");
    let mut round_trip = vec![0; pattern.len()];
    kv_cache
        .read_page(3, &mut round_trip)
        .expect("page read should return CUDA memory");
    assert_eq!(round_trip, pattern);

    let tensor = kv_cache
        .slot_location(1, 1, 63)
        .expect("community-style token slot should map into the pool");
    assert_eq!(tensor.byte_offset, 522_240);
    assert_eq!(tensor.byte_len, 2_048);
    assert_eq!(
        tensor.device_ptr,
        kv_cache.allocation().device_ptr() + tensor.byte_offset as u64
    );
    let transferable = kv_cache
        .transferable_kv_cache_memory()
        .expect("CUDA KV cache should expose Mooncake memory");

    assert_eq!(transferable.regions().len(), 1);
    assert_eq!(transferable.page_size_bytes(), 131_072);
    assert_eq!(
        transferable.location(),
        KvCacheMemoryLocation::Cuda { device_id: 0 }
    );
    assert_eq!(transferable.regions()[0].byte_len, 2 * 1024 * 1024);
    assert_eq!(
        transferable
            .mooncake_decode_remote_layout(&[3])
            .dst_kv_item_len,
        131_072
    );
    let memory_after = backend
        .context()
        .memory_info()
        .expect("CUDA memory info should remain available");
    assert!(memory_after.free_bytes < memory_before.free_bytes);
    drop(kv_cache);
    let memory_released = backend
        .context()
        .memory_info()
        .expect("CUDA memory info should remain available after release");
    assert!(memory_released.free_bytes > memory_after.free_bytes);
}

#[test]
#[ignore = "requires a CUDA device, NVIDIA driver, and BF16-capable cuBLAS"]
fn cuda_bf16_gemm_runs_transformer_linear_projection() {
    let backend = CudaBackend::initialize(0).expect("CUDA backend should initialize");
    let blas = CudaBlas::load(backend.context()).expect("cuBLAS should load");
    let input_values = [1.0_f32, 2.0, 3.0, -1.0, 0.5, 2.0];
    let weight_values = [
        1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.5,
    ];
    let output_element_count = 2 * 4;
    let mut input = backend
        .context()
        .allocate(input_values.len() * size_of::<u16>())
        .expect("input should allocate");
    let mut weight = backend
        .context()
        .allocate(weight_values.len() * size_of::<u16>())
        .expect("weight should allocate");
    let mut output = backend
        .context()
        .allocate(output_element_count * size_of::<u16>())
        .expect("output should allocate");
    input
        .copy_from_host(0, &bf16_bytes(&input_values))
        .expect("input should upload");
    weight
        .copy_from_host(0, &bf16_bytes(&weight_values))
        .expect("weight should upload");

    blas.bf16_gemm_row_major(&input, 2, 3, &weight, 4, &mut output)
        .expect("BF16 projection should execute");

    let mut output_bytes = vec![0_u8; output_element_count * size_of::<u16>()];
    output
        .copy_to_host(0, &mut output_bytes)
        .expect("output should download");
    assert_f32_close(
        &bytes_bf16(&output_bytes),
        &[1.0, 2.0, 3.0, 0.5, -1.0, 0.5, 2.0, -0.5],
        0.01,
    );
}

#[test]
#[ignore = "requires a CUDA device, NVIDIA driver, and NVRTC"]
fn cuda_runtime_kernels_execute_and_write_kv_slots() {
    let backend = CudaBackend::initialize(0).expect("CUDA backend should initialize");
    let capability = backend.capabilities();
    let ComputeCapability::Cuda(compute_capability) = capability.compute_capability else {
        panic!("CUDA backend must report CUDA compute capability");
    };
    let kernels = CudaF32Kernels::compile(backend.context(), compute_capability)
        .expect("NVRTC should compile CUDA runtime kernels");

    let input_values = [1.0_f32, -2.0, 3.0, -4.0, 0.5, 1.5, -2.5, 3.5];
    let weight_values = [1.0_f32, 0.5, 2.0, 1.5];
    let up_values = [0.25_f32, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];
    let mut input = backend
        .context()
        .allocate(input_values.len() * size_of::<f32>())
        .expect("input should allocate");
    let mut weight = backend
        .context()
        .allocate(weight_values.len() * size_of::<f32>())
        .expect("weight should allocate");
    let mut up = backend
        .context()
        .allocate(up_values.len() * size_of::<f32>())
        .expect("up tensor should allocate");
    let mut rms_output = backend
        .context()
        .allocate(input_values.len() * size_of::<f32>())
        .expect("RMSNorm output should allocate");
    let mut silu_output = backend
        .context()
        .allocate(input_values.len() * size_of::<f32>())
        .expect("SiLU output should allocate");
    input
        .copy_from_host(0, &f32_bytes(&input_values))
        .expect("input should upload");
    weight
        .copy_from_host(0, &f32_bytes(&weight_values))
        .expect("weight should upload");
    up.copy_from_host(0, &f32_bytes(&up_values))
        .expect("up tensor should upload");

    kernels
        .rms_norm(CudaRmsNormLaunch {
            input: &input,
            input_offset: 0,
            weight: &weight,
            weight_offset: 0,
            output: &mut rms_output,
            output_offset: 0,
            rows: 2,
            width: 4,
            epsilon: 1.0e-5,
        })
        .expect("RMSNorm should execute on CUDA");
    kernels
        .silu_mul(CudaSiluMulLaunch {
            gate: &input,
            gate_offset: 0,
            up: &up,
            up_offset: 0,
            output: &mut silu_output,
            output_offset: 0,
            element_count: input_values.len(),
        })
        .expect("SiLU-mul should execute on CUDA");

    let mut rms_bytes = vec![0_u8; input_values.len() * size_of::<f32>()];
    rms_output
        .copy_to_host(0, &mut rms_bytes)
        .expect("RMSNorm output should download");
    let expected_rms = input_values
        .chunks_exact(4)
        .flat_map(|row| {
            let inverse_rms =
                (row.iter().map(|value| value * value).sum::<f32>() / 4.0 + 1.0e-5).sqrt();
            row.iter()
                .zip(weight_values)
                .map(move |(value, weight)| value / inverse_rms * weight)
        })
        .collect::<Vec<_>>();
    assert_f32_close(&bytes_f32(&rms_bytes), &expected_rms, 1.0e-4);

    let mut silu_bytes = vec![0_u8; input_values.len() * size_of::<f32>()];
    silu_output
        .copy_to_host(0, &mut silu_bytes)
        .expect("SiLU output should download");
    let expected_silu = input_values
        .iter()
        .zip(up_values)
        .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
        .collect::<Vec<_>>();
    assert_f32_close(&bytes_f32(&silu_bytes), &expected_silu, 1.0e-4);

    let mut kv_cache = CudaKvCachePool::allocate(backend.context(), cuda_test_layout(), 2)
        .expect("CUDA KV cache should allocate");
    let slot_byte_len = kv_cache.layout().bytes_per_token_per_tensor();
    let pattern = (0..slot_byte_len)
        .map(|offset| (offset % 251) as u8)
        .collect::<Vec<_>>();
    let mut slot_source = backend
        .context()
        .allocate(slot_byte_len)
        .expect("KV slot source should allocate");
    let mut slot_destination = backend
        .context()
        .allocate(slot_byte_len)
        .expect("KV slot destination should allocate");
    slot_source
        .copy_from_host(0, &pattern)
        .expect("KV slot should upload");
    kv_cache
        .write_tensor_slot_from_device(1, 0, 7, &slot_source, 0)
        .expect("KV slot write should use device-to-device copy");
    kv_cache
        .read_tensor_slot_to_device(1, 0, 7, &mut slot_destination, 0)
        .expect("KV slot read should use device-to-device copy");
    let mut round_trip = vec![0_u8; slot_byte_len];
    slot_destination
        .copy_to_host(0, &mut round_trip)
        .expect("KV slot should download for verification");
    assert_eq!(round_trip, pattern);
}

#[test]
#[ignore = "requires a CUDA device, NVIDIA driver, and NVRTC"]
fn cuda_kv_kernels_scatter_and_gather_batched_physical_slots() {
    let backend = CudaBackend::initialize(0).expect("CUDA backend should initialize");
    let capability = backend.capabilities();
    let ComputeCapability::Cuda(compute_capability) = capability.compute_capability else {
        panic!("CUDA backend must report CUDA compute capability");
    };
    let mut kernels = CudaKvPairCopyKernels::compile(backend.context(), compute_capability)
        .expect("NVRTC should compile dtype-independent KV copy kernels");
    let mut kv_cache = CudaKvCachePool::allocate(backend.context(), cuda_test_layout(), 2)
        .expect("CUDA KV cache should allocate two physical pages");
    let slots = [1_usize, 15, 16, 31];
    let slot_map = kv_cache
        .upload_slot_map(&slots)
        .expect("scheduler physical slots should upload after validation");
    let row_bytes = kv_cache.layout().bytes_per_token_per_tensor();
    let key_stride = row_bytes + 37;
    let value_stride = row_bytes + 53;
    let (key_bytes, expected_keys) = strided_byte_rows(slots.len(), row_bytes, key_stride, 17);
    let (value_bytes, expected_values) =
        strided_byte_rows(slots.len(), row_bytes, value_stride, 91);
    let mut keys = backend
        .context()
        .allocate(key_bytes.len())
        .expect("strided key rows should allocate");
    let mut values = backend
        .context()
        .allocate(value_bytes.len())
        .expect("strided value rows should allocate");
    keys.copy_from_host(0, &key_bytes)
        .expect("key rows should upload");
    values
        .copy_from_host(0, &value_bytes)
        .expect("value rows should upload");

    let transferable_before = kv_cache
        .transferable_memory()
        .expect("KV pool should expose its Mooncake memory before model writes");
    kv_cache
        .write_kv_slots_from_device(CudaKvSlotScatterLaunch {
            kernels: &mut kernels,
            layer_index: 1,
            slot_map: &slot_map,
            keys: &keys,
            keys_offset: 0,
            key_row_stride_bytes: key_stride,
            values: &values,
            values_offset: 0,
            value_row_stride_bytes: value_stride,
        })
        .expect("one CUDA launch should scatter batched K/V rows into physical slots");
    assert_eq!(
        transferable_before.regions()[0].base_addr,
        kv_cache.allocation().device_ptr() as usize,
        "attention writes and Mooncake registration must use the same allocation"
    );

    let gathered_key_stride = row_bytes + 11;
    let gathered_value_stride = row_bytes + 19;
    let gathered_key_len = (slots.len() - 1) * gathered_key_stride + row_bytes;
    let gathered_value_len = (slots.len() - 1) * gathered_value_stride + row_bytes;
    let mut gathered_keys = backend
        .context()
        .allocate(gathered_key_len)
        .expect("gathered key rows should allocate");
    let mut gathered_values = backend
        .context()
        .allocate(gathered_value_len)
        .expect("gathered value rows should allocate");
    gathered_keys.fill(0).expect("key output should clear");
    gathered_values.fill(0).expect("value output should clear");
    kv_cache
        .read_kv_slots_to_device(CudaKvSlotGatherLaunch {
            kernels: &mut kernels,
            layer_index: 1,
            slot_map: &slot_map,
            keys: &mut gathered_keys,
            keys_offset: 0,
            key_row_stride_bytes: gathered_key_stride,
            values: &mut gathered_values,
            values_offset: 0,
            value_row_stride_bytes: gathered_value_stride,
        })
        .expect("one CUDA launch should gather batched K/V rows from physical slots");
    let mut gathered_key_bytes = vec![0_u8; gathered_key_len];
    let mut gathered_value_bytes = vec![0_u8; gathered_value_len];
    gathered_keys
        .copy_to_host(0, &mut gathered_key_bytes)
        .expect("gathered keys should download");
    gathered_values
        .copy_to_host(0, &mut gathered_value_bytes)
        .expect("gathered values should download");
    assert_strided_byte_rows(&gathered_key_bytes, gathered_key_stride, &expected_keys);
    assert_strided_byte_rows(
        &gathered_value_bytes,
        gathered_value_stride,
        &expected_values,
    );

    let layout = kv_cache.layout();
    for page_index in 0..layout.page_count() {
        let mut page = vec![0_u8; layout.runtime().page_size_bytes];
        kv_cache
            .read_page(page_index, &mut page)
            .expect("physical page should be readable for acceptance verification");
        let page_start = layout
            .page_byte_range(page_index)
            .expect("page range should be valid")
            .start;
        for (row_index, slot) in slots.iter().copied().enumerate() {
            if slot / layout.runtime().page_size != page_index {
                continue;
            }
            for (tensor_index, expected) in [
                (0, &expected_keys[row_index]),
                (1, &expected_values[row_index]),
            ] {
                let range = layout
                    .tensor_slot_byte_range(1, tensor_index, slot)
                    .expect("slot tensor range should be valid");
                let start = range.start - page_start;
                assert_eq!(&page[start..start + row_bytes], expected);
            }
        }
    }
}

#[test]
#[ignore = "requires a BF16-capable CUDA device, NVIDIA driver, and NVRTC"]
fn cuda_bf16_paged_attention_reads_mooncake_registered_physical_kv_slots() {
    let backend = CudaBackend::initialize(0).expect("CUDA backend should initialize");
    let capability = backend.capabilities();
    let ComputeCapability::Cuda(compute_capability) = capability.compute_capability else {
        panic!("CUDA backend must report CUDA compute capability");
    };

    let mut kv_copy_kernels = CudaKvPairCopyKernels::compile(backend.context(), compute_capability)
        .expect("NVRTC should compile KV scatter kernels");
    let mut attention =
        CudaBf16PagedAttentionExecutor::compile(backend.context(), compute_capability)
            .expect("NVRTC should compile BF16 paged attention");
    let mut kv_cache = CudaKvCachePool::allocate(backend.context(), attention_test_layout(), 2)
        .expect("two physical KV pages should allocate");
    let worker_batch = attention_test_batch();
    let metadata =
        CudaPagedAttentionMetadata::from_model_worker_batch(&worker_batch, kv_cache.layout())
            .expect("scheduler batch should produce physical attention metadata");
    let device_metadata = metadata
        .upload(backend.context())
        .expect("attention metadata should upload");
    let physical_slots = metadata
        .sequence_slots()
        .iter()
        .map(|slot| *slot as usize)
        .collect::<Vec<_>>();
    let slot_map = kv_cache
        .upload_slot_map(&physical_slots)
        .expect("physical KV slots should upload");

    let runtime = attention_test_layout();
    let kv_elements = metadata.sequence_slot_count() * runtime.kv_heads * runtime.head_dim;
    let key_source = (0..kv_elements)
        .map(|index| (index as f32 - 11.0) * 0.03125)
        .collect::<Vec<_>>();
    let value_source = (0..kv_elements)
        .map(|index| (17.0 - index as f32) * 0.046875)
        .collect::<Vec<_>>();
    let key_bytes = bf16_bytes(&key_source);
    let value_bytes = bf16_bytes(&value_source);
    let quantized_keys = bytes_bf16(&key_bytes);
    let quantized_values = bytes_bf16(&value_bytes);
    let mut keys = backend
        .context()
        .allocate(key_bytes.len())
        .expect("BF16 keys should allocate");
    let mut values = backend
        .context()
        .allocate(value_bytes.len())
        .expect("BF16 values should allocate");
    keys.copy_from_host(0, &key_bytes)
        .expect("BF16 keys should upload");
    values
        .copy_from_host(0, &value_bytes)
        .expect("BF16 values should upload");
    let row_bytes = kv_cache.layout().bytes_per_token_per_tensor();
    kv_cache
        .write_kv_slots_from_device(CudaKvSlotScatterLaunch {
            kernels: &mut kv_copy_kernels,
            layer_index: 0,
            slot_map: &slot_map,
            keys: &keys,
            keys_offset: 0,
            key_row_stride_bytes: row_bytes,
            values: &values,
            values_offset: 0,
            value_row_stride_bytes: row_bytes,
        })
        .expect("KV scatter should write scheduler slots into the physical pool");

    let query_head_count = 4;
    let query_elements = metadata.query_count() * query_head_count * runtime.head_dim;
    let query_source = (0..query_elements)
        .map(|index| ((index * 7 % 29) as f32 - 14.0) * 0.0625)
        .collect::<Vec<_>>();
    let query_bytes = bf16_bytes(&query_source);
    let quantized_queries = bytes_bf16(&query_bytes);
    let mut queries = backend
        .context()
        .allocate(query_bytes.len())
        .expect("BF16 queries should allocate");
    let mut output = backend
        .context()
        .allocate(query_bytes.len())
        .expect("BF16 attention output should allocate");
    queries
        .copy_from_host(0, &query_bytes)
        .expect("BF16 queries should upload");

    let transferable = kv_cache
        .transferable_memory()
        .expect("attention KV pool should expose Mooncake memory");
    assert_eq!(
        transferable.regions()[0].base_addr,
        kv_cache.allocation().device_ptr() as usize,
        "attention and Mooncake must read and register the same CUDA allocation"
    );
    let scale = (runtime.head_dim as f32).sqrt().recip();
    attention
        .forward(CudaPagedAttentionForward {
            kv_cache: &kv_cache,
            layer_index: 0,
            metadata: &device_metadata,
            queries: &queries,
            queries_offset: 0,
            query_head_count,
            scale,
            output: &mut output,
            output_offset: 0,
        })
        .expect("BF16 paged attention should execute over physical KV slots");
    let mut output_bytes = vec![0_u8; query_bytes.len()];
    output
        .copy_to_host(0, &mut output_bytes)
        .expect("attention output should download");
    let expected = reference_paged_attention(
        &metadata,
        &quantized_queries,
        &quantized_keys,
        &quantized_values,
        ReferenceAttentionShape {
            query_head_count,
            kv_head_count: runtime.kv_heads,
            head_dim: runtime.head_dim,
            scale,
        },
    );
    assert_f32_close(&bytes_bf16(&output_bytes), &expected, 0.02);
}

#[cfg(feature = "mooncake-link")]
#[test]
#[ignore = "requires a CUDA device, NVIDIA driver, and linked native Mooncake"]
fn cuda_mooncake_registers_real_cuda_kv_memory() {
    let backend = CudaBackend::initialize(0).expect("CUDA backend should initialize");
    let kv_cache = CudaKvCachePool::allocate(backend.context(), cuda_test_layout(), 16)
        .expect("CUDA KV cache should allocate a page-major device pool");
    let transferable = kv_cache
        .transferable_kv_cache_memory()
        .expect("CUDA KV cache should expose Mooncake memory");
    let engine = SharedLinkedMooncakeTransferEngine::new(&MooncakeTransferEngineConfig {
        metadata_server: "P2PHANDSHAKE".to_string(),
        session_id: "127.0.0.1:0".to_string(),
        hostname: "127.0.0.1".to_string(),
        rpc_port: 0,
        protocol: "tcp".to_string(),
        device_name: String::new(),
        gpu_id: 0,
    })
    .expect("linked Mooncake engine should initialize");
    let mut registration = RegisteredMooncakeKvCacheMemory::register(engine, transferable)
        .expect("Mooncake should register CUDA KV regions");

    assert_eq!(registration.memory().regions().len(), 1);
    registration
        .unregister()
        .expect("Mooncake should unregister CUDA KV regions");
}
