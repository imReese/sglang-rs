use sglang_kernel::cuda::{CudaComputeCapability, CudaDeviceInfo};
use sglang_srt::backend::{
    CapabilityStatus, ComputeCapability, RuntimeCapability, RuntimeDtype, RuntimeRequirements,
};

#[test]
fn cpu_reference_reports_missing_production_requirements_together() {
    let capability = RuntimeCapability::cpu_reference("cpu-reference", false);
    let error = capability
        .validate_requirements(&RuntimeRequirements {
            dtype: Some(RuntimeDtype::Bf16),
            attention_backend: Some("flashinfer"),
            tensor_parallel_size: 2,
            requires_kv_cache_registration: true,
            requires_mooncake: true,
        })
        .expect_err("CPU reference must not satisfy production requirements");

    assert_eq!(
        error.missing,
        [
            "dtype bfloat16",
            "attention backend flashinfer",
            "tensor parallel size 2",
            "KV cache memory registration",
            "Mooncake transport",
        ]
    );
}

#[test]
fn cuda_capabilities_are_derived_from_compute_capability_not_gpu_product_name() {
    let a100 = RuntimeCapability::cuda_hardware(
        "cuda-device",
        &cuda_device("arbitrary-sm80-device", CudaComputeCapability::new(8, 0)),
    );
    assert_eq!(
        a100.compute_capability,
        ComputeCapability::Cuda(CudaComputeCapability::new(8, 0))
    );
    assert!(a100.supported_dtypes.contains(&RuntimeDtype::Fp16));
    assert!(a100.supported_dtypes.contains(&RuntimeDtype::Bf16));
    assert!(!a100.supported_dtypes.contains(&RuntimeDtype::Fp8E4M3));

    let sm100 = RuntimeCapability::cuda_hardware(
        "cuda-device",
        &cuda_device("arbitrary-sm100-device", CudaComputeCapability::new(10, 0)),
    );
    assert!(sm100.supported_dtypes.contains(&RuntimeDtype::Fp8E4M3));
    assert!(sm100.supported_dtypes.contains(&RuntimeDtype::Fp8E5M2));
    assert_eq!(sm100.tensor_parallel, CapabilityStatus::Supported);
    assert_eq!(
        sm100.kv_cache_memory_registration,
        CapabilityStatus::Supported
    );
    assert_eq!(sm100.rdma, CapabilityStatus::Unknown);
    assert_eq!(sm100.nvlink, CapabilityStatus::Unknown);
    assert!(sm100.attention_backends.is_empty());
    assert!(!sm100.supports_forward);
}

#[test]
fn cuda_without_unified_addressing_does_not_claim_kv_registration() {
    let mut device = cuda_device("legacy", CudaComputeCapability::new(7, 0));
    device.unified_addressing = false;

    let capability = RuntimeCapability::cuda_hardware("cuda-device", &device);

    assert_eq!(
        capability.kv_cache_memory_registration,
        CapabilityStatus::Unsupported
    );
    assert!(!capability.supports_transferable_kv);
}

fn cuda_device(name: &str, compute_capability: CudaComputeCapability) -> CudaDeviceInfo {
    CudaDeviceInfo {
        ordinal: 0,
        name: name.to_string(),
        total_memory_bytes: 80 * 1024 * 1024 * 1024,
        multiprocessor_count: 108,
        unified_addressing: true,
        compute_capability,
        driver_version: 12_080,
    }
}
