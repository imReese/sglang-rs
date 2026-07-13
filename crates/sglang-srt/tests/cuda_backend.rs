use sglang_kernel::cuda::CudaComputeCapability;
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
