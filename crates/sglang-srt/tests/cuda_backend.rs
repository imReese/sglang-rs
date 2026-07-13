use sglang_kernel::cuda::CudaComputeCapability;
use sglang_kernel::cuda_kernels::CudaF32Kernels;
use sglang_srt::backend::{ComputeCapability, CudaBackend};
use sglang_srt::cuda_kv_cache::CudaKvCachePool;
use sglang_srt::transfer::{
    KvCacheDtype, KvCacheMemoryLocation, KvCacheRuntimeLayout, MooncakeKvCacheMemoryProvider,
};
#[cfg(feature = "mooncake-link")]
use sglang_srt::transfer::{
    MooncakeTransferEngineConfig, RegisteredMooncakeKvCacheMemory,
    SharedLinkedMooncakeTransferEngine,
};

fn b200_test_layout() -> KvCacheRuntimeLayout {
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

fn assert_f32_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index} differs: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

#[test]
#[ignore = "requires a B200-class CUDA device and NVIDIA driver"]
fn b200_cuda_backend_round_trips_page_major_device_kv_memory() {
    let backend = CudaBackend::initialize(0).expect("CUDA backend should initialize on B200");
    let capability = backend.capabilities();
    let ComputeCapability::Cuda(compute_capability) = capability.compute_capability else {
        panic!("CUDA backend must report CUDA compute capability");
    };
    assert!(
        compute_capability >= CudaComputeCapability::new(10, 0),
        "B200 acceptance requires sm_100 or newer, found {compute_capability}"
    );

    let memory_before = backend
        .context()
        .memory_info()
        .expect("CUDA memory info should be available");
    let mut kv_cache = CudaKvCachePool::allocate(backend.context(), b200_test_layout(), 16)
        .expect("CUDA KV cache should allocate a page-major device pool");
    let pattern = (0..b200_test_layout().page_size_bytes)
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
        .mooncake_kv_cache_memory()
        .expect("CUDA KV cache should expose Mooncake memory");

    assert_eq!(transferable.regions().len(), 1);
    assert_eq!(transferable.page_size_bytes(), 131_072);
    assert_eq!(
        transferable.location(),
        KvCacheMemoryLocation::Cuda { device_id: 0 }
    );
    assert_eq!(transferable.regions()[0].byte_len, 2 * 1024 * 1024);
    assert_eq!(
        transferable.decode_remote_layout(&[3]).dst_kv_item_len,
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
#[ignore = "requires a B200-class CUDA device, NVIDIA driver, and NVRTC"]
fn b200_cuda_runtime_kernels_execute_and_write_kv_slots() {
    let backend = CudaBackend::initialize(0).expect("CUDA backend should initialize on B200");
    let capability = backend.capabilities();
    let ComputeCapability::Cuda(compute_capability) = capability.compute_capability else {
        panic!("CUDA backend must report CUDA compute capability");
    };
    assert!(
        compute_capability >= CudaComputeCapability::new(10, 0),
        "B200 acceptance requires sm_100 or newer, found {compute_capability}"
    );
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
        .rms_norm(&input, 0, &weight, 0, &mut rms_output, 0, 2, 4, 1.0e-5)
        .expect("RMSNorm should execute on CUDA");
    kernels
        .silu_mul(&input, 0, &up, 0, &mut silu_output, 0, input_values.len())
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

    let mut kv_cache = CudaKvCachePool::allocate(backend.context(), b200_test_layout(), 2)
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

#[cfg(feature = "mooncake-link")]
#[test]
#[ignore = "requires B200, NVIDIA driver, and linked native Mooncake"]
fn b200_mooncake_registers_real_cuda_kv_memory() {
    let backend = CudaBackend::initialize(0).expect("CUDA backend should initialize on B200");
    let kv_cache = CudaKvCachePool::allocate(backend.context(), b200_test_layout(), 16)
        .expect("CUDA KV cache should allocate a page-major device pool");
    let transferable = kv_cache
        .mooncake_kv_cache_memory()
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
