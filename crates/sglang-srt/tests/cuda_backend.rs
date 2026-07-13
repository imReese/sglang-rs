use sglang_kernel::cuda::CudaComputeCapability;
use sglang_srt::backend::{ComputeCapability, CudaBackend};
use sglang_srt::transfer::{
    CudaKvCacheMemory, KvCacheMemoryLocation, MooncakeKvCacheMemoryProvider,
};
#[cfg(feature = "mooncake-link")]
use sglang_srt::transfer::{
    MooncakeTransferEngineConfig, RegisteredMooncakeKvCacheMemory,
    SharedLinkedMooncakeTransferEngine,
};

#[test]
#[ignore = "requires a B200-class CUDA device and NVIDIA driver"]
fn b200_cuda_backend_allocates_transferable_device_kv_memory() {
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
    let kv_cache = CudaKvCacheMemory::allocate(backend.context(), 2, 2 * 1024 * 1024, 16 * 1024)
        .expect("CUDA KV cache should allocate device regions");
    let transferable = kv_cache
        .mooncake_kv_cache_memory()
        .expect("CUDA KV cache should expose Mooncake memory");

    assert_eq!(transferable.regions().len(), 2);
    assert_eq!(
        transferable.location(),
        KvCacheMemoryLocation::Cuda { device_id: 0 }
    );
    assert!(
        transferable
            .regions()
            .iter()
            .all(|region| region.base_addr != 0 && region.byte_len == 2 * 1024 * 1024)
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
    let kv_cache = CudaKvCacheMemory::allocate(backend.context(), 2, 2 * 1024 * 1024, 16 * 1024)
        .expect("CUDA KV cache should allocate device regions");
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

    assert_eq!(registration.memory().regions().len(), 2);
    registration
        .unregister()
        .expect("Mooncake should unregister CUDA KV regions");
}
