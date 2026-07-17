use std::fmt;

use sglang_kernel::cuda::{
    CudaComputeCapability, CudaContext, CudaDeviceInfo, CudaDriver, CudaError,
};

use crate::transfer::TransferBackend;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeBackend {
    Auto,
    Cpu,
    Cuda,
    Metal,
    Rocm,
    Musa,
    Xpu,
    Npu,
    Hpu,
}

impl RuntimeBackend {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "auto" => Some(Self::Auto),
            "cpu" => Some(Self::Cpu),
            "cuda" => Some(Self::Cuda),
            "metal" => Some(Self::Metal),
            "rocm" => Some(Self::Rocm),
            "musa" => Some(Self::Musa),
            "xpu" => Some(Self::Xpu),
            "npu" => Some(Self::Npu),
            "hpu" => Some(Self::Hpu),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::Rocm => "rocm",
            Self::Musa => "musa",
            Self::Xpu => "xpu",
            Self::Npu => "npu",
            Self::Hpu => "hpu",
        }
    }

    pub fn requires_production_runtime(self) -> bool {
        matches!(
            self,
            Self::Cuda | Self::Metal | Self::Rocm | Self::Musa | Self::Xpu | Self::Npu | Self::Hpu
        )
    }
}

impl fmt::Display for RuntimeBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeDtype {
    F32,
    Fp16,
    Bf16,
    Fp8E4M3,
    Fp8E5M2,
}

impl RuntimeDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "float32",
            Self::Fp16 => "float16",
            Self::Bf16 => "bfloat16",
            Self::Fp8E4M3 => "fp8_e4m3",
            Self::Fp8E5M2 => "fp8_e5m2",
        }
    }
}

impl fmt::Display for RuntimeDtype {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapabilityStatus {
    Supported,
    Unsupported,
    Unknown,
}

impl CapabilityStatus {
    pub fn is_supported(self) -> bool {
        self == Self::Supported
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComputeCapability {
    CpuReference,
    Cuda(CudaComputeCapability),
    Unspecified(RuntimeBackend),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeRequirements<'a> {
    pub requires_forward: bool,
    pub dtype: Option<RuntimeDtype>,
    pub attention_backend: Option<&'a str>,
    pub tensor_parallel_size: usize,
    pub requires_kv_cache_registration: bool,
    pub requires_mooncake: bool,
}

impl Default for RuntimeRequirements<'_> {
    fn default() -> Self {
        Self {
            requires_forward: false,
            dtype: None,
            attention_backend: None,
            tensor_parallel_size: 1,
            requires_kv_cache_registration: false,
            requires_mooncake: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeCapabilityMismatch {
    pub runtime_name: &'static str,
    pub missing: Vec<String>,
}

pub struct CudaBackend {
    driver: CudaDriver,
    context: CudaContext,
    device: CudaDeviceInfo,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RuntimeDevicePlacement {
    pub(crate) base_gpu_id: usize,
    pub(crate) gpu_id_step: usize,
    pub(crate) local_rank: usize,
    pub(crate) tensor_parallel_rank: usize,
}

impl RuntimeDevicePlacement {
    pub(crate) fn for_tensor_parallel_rank(
        base_gpu_id: usize,
        gpu_id_step: usize,
        tensor_parallel_rank: usize,
        tensor_parallel_size: usize,
        node_count: usize,
    ) -> Result<Self, String> {
        if gpu_id_step == 0 {
            return Err("gpu_id_step must be positive".to_string());
        }
        if tensor_parallel_size == 0 {
            return Err("tensor parallel size must be positive".to_string());
        }
        if node_count == 0 || !tensor_parallel_size.is_multiple_of(node_count) {
            return Err(format!(
                "tensor parallel size {tensor_parallel_size} must be divisible by node count {node_count}"
            ));
        }
        if tensor_parallel_rank >= tensor_parallel_size {
            return Err(format!(
                "tensor parallel rank {tensor_parallel_rank} must be smaller than tensor parallel size {tensor_parallel_size}"
            ));
        }
        let ranks_per_node = tensor_parallel_size / node_count;
        Ok(Self {
            base_gpu_id,
            gpu_id_step,
            local_rank: tensor_parallel_rank % ranks_per_node,
            tensor_parallel_rank,
        })
    }

    pub(crate) fn device_ordinal(self) -> Result<usize, String> {
        self.local_rank
            .checked_mul(self.gpu_id_step)
            .and_then(|rank_offset| self.base_gpu_id.checked_add(rank_offset))
            .ok_or_else(|| {
                format!(
                    "device ordinal overflowed for base_gpu_id={}, gpu_id_step={}, local_rank={}, tp_rank={}",
                    self.base_gpu_id,
                    self.gpu_id_step,
                    self.local_rank,
                    self.tensor_parallel_rank
                )
            })
    }
}

impl CudaBackend {
    pub fn initialize(device_ordinal: usize) -> Result<Self, CudaError> {
        let driver = CudaDriver::load()?;
        let device = driver.device_info(device_ordinal)?;
        let context = driver.retain_primary_context(device_ordinal)?;
        Ok(Self {
            driver,
            context,
            device,
        })
    }

    pub fn driver(&self) -> &CudaDriver {
        &self.driver
    }

    pub fn context(&self) -> &CudaContext {
        &self.context
    }

    pub fn device(&self) -> &CudaDeviceInfo {
        &self.device
    }

    pub fn capabilities(&self) -> RuntimeCapability {
        RuntimeCapability::cuda_hardware("cuda-driver-backend", &self.device)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimeCapabilityClass {
    CpuReference,
    Production(RuntimeBackend),
    MetadataOnly,
    Unsupported,
}

impl RuntimeCapabilityClass {
    pub fn label(&self) -> &'static str {
        match self {
            Self::CpuReference => "cpu-reference",
            Self::Production(backend) => backend.as_str(),
            Self::MetadataOnly => "metadata-only",
            Self::Unsupported => "unsupported",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeCapability {
    pub runtime_name: &'static str,
    pub class: RuntimeCapabilityClass,
    pub supports_forward: bool,
    pub supports_transferable_kv: bool,
    pub compute_capability: ComputeCapability,
    pub supported_dtypes: Vec<RuntimeDtype>,
    pub attention_backends: Vec<&'static str>,
    pub tensor_parallel: CapabilityStatus,
    pub kv_cache_memory_registration: CapabilityStatus,
    pub mooncake: CapabilityStatus,
    pub rdma: CapabilityStatus,
    pub nvlink: CapabilityStatus,
}

impl RuntimeCapability {
    pub fn cpu_reference(runtime_name: &'static str, supports_transferable_kv: bool) -> Self {
        Self {
            runtime_name,
            class: RuntimeCapabilityClass::CpuReference,
            supports_forward: true,
            supports_transferable_kv,
            compute_capability: ComputeCapability::CpuReference,
            supported_dtypes: vec![RuntimeDtype::F32],
            attention_backends: vec!["reference"],
            tensor_parallel: CapabilityStatus::Unsupported,
            kv_cache_memory_registration: if supports_transferable_kv {
                CapabilityStatus::Supported
            } else {
                CapabilityStatus::Unsupported
            },
            mooncake: if supports_transferable_kv && cfg!(feature = "mooncake-link") {
                CapabilityStatus::Supported
            } else {
                CapabilityStatus::Unsupported
            },
            rdma: CapabilityStatus::Unsupported,
            nvlink: CapabilityStatus::Unsupported,
        }
    }

    pub fn metadata_only(runtime_name: &'static str, supports_transferable_kv: bool) -> Self {
        Self {
            runtime_name,
            class: RuntimeCapabilityClass::MetadataOnly,
            supports_forward: false,
            supports_transferable_kv,
            compute_capability: ComputeCapability::Unspecified(RuntimeBackend::Auto),
            supported_dtypes: Vec::new(),
            attention_backends: Vec::new(),
            tensor_parallel: CapabilityStatus::Unknown,
            kv_cache_memory_registration: CapabilityStatus::Unknown,
            mooncake: CapabilityStatus::Unknown,
            rdma: CapabilityStatus::Unknown,
            nvlink: CapabilityStatus::Unknown,
        }
    }

    pub fn unsupported(runtime_name: &'static str) -> Self {
        Self {
            runtime_name,
            class: RuntimeCapabilityClass::Unsupported,
            supports_forward: false,
            supports_transferable_kv: false,
            compute_capability: ComputeCapability::Unspecified(RuntimeBackend::Auto),
            supported_dtypes: Vec::new(),
            attention_backends: Vec::new(),
            tensor_parallel: CapabilityStatus::Unsupported,
            kv_cache_memory_registration: CapabilityStatus::Unsupported,
            mooncake: CapabilityStatus::Unsupported,
            rdma: CapabilityStatus::Unsupported,
            nvlink: CapabilityStatus::Unsupported,
        }
    }

    pub fn production(
        runtime_name: &'static str,
        backend: RuntimeBackend,
        supports_transferable_kv: bool,
    ) -> Self {
        debug_assert!(backend.requires_production_runtime());
        Self {
            runtime_name,
            class: RuntimeCapabilityClass::Production(backend),
            supports_forward: true,
            supports_transferable_kv,
            compute_capability: ComputeCapability::Unspecified(backend),
            supported_dtypes: Vec::new(),
            attention_backends: Vec::new(),
            tensor_parallel: CapabilityStatus::Unknown,
            kv_cache_memory_registration: if supports_transferable_kv {
                CapabilityStatus::Supported
            } else {
                CapabilityStatus::Unsupported
            },
            mooncake: CapabilityStatus::Unknown,
            rdma: CapabilityStatus::Unknown,
            nvlink: CapabilityStatus::Unknown,
        }
    }

    pub fn cuda_hardware(runtime_name: &'static str, device: &CudaDeviceInfo) -> Self {
        let compute_capability = device.compute_capability;
        let mut supported_dtypes = vec![RuntimeDtype::F32];
        if compute_capability >= CudaComputeCapability::new(5, 3) {
            supported_dtypes.push(RuntimeDtype::Fp16);
        }
        if compute_capability >= CudaComputeCapability::new(8, 0) {
            supported_dtypes.push(RuntimeDtype::Bf16);
        }
        if compute_capability >= CudaComputeCapability::new(8, 9) {
            supported_dtypes.extend([RuntimeDtype::Fp8E4M3, RuntimeDtype::Fp8E5M2]);
        }

        Self {
            runtime_name,
            class: RuntimeCapabilityClass::Production(RuntimeBackend::Cuda),
            supports_forward: false,
            supports_transferable_kv: device.unified_addressing,
            compute_capability: ComputeCapability::Cuda(compute_capability),
            supported_dtypes,
            attention_backends: Vec::new(),
            tensor_parallel: CapabilityStatus::Unsupported,
            kv_cache_memory_registration: if device.unified_addressing {
                CapabilityStatus::Supported
            } else {
                CapabilityStatus::Unsupported
            },
            mooncake: if cfg!(feature = "mooncake-link") {
                CapabilityStatus::Supported
            } else {
                CapabilityStatus::Unsupported
            },
            rdma: CapabilityStatus::Unknown,
            nvlink: CapabilityStatus::Unknown,
        }
    }

    pub fn with_tensor_parallel(mut self, tensor_parallel: CapabilityStatus) -> Self {
        self.tensor_parallel = tensor_parallel;
        self
    }

    pub fn validate_requirements(
        &self,
        requirements: &RuntimeRequirements<'_>,
    ) -> Result<(), RuntimeCapabilityMismatch> {
        let mut missing = Vec::new();
        if requirements.requires_forward && !self.supports_forward {
            missing.push("model forward execution".to_string());
        }
        if let Some(dtype) = requirements.dtype
            && !self.supported_dtypes.contains(&dtype)
        {
            missing.push(format!("dtype {dtype}"));
        }
        if let Some(attention_backend) = requirements.attention_backend
            && !self.attention_backends.contains(&attention_backend)
        {
            missing.push(format!("attention backend {attention_backend}"));
        }
        if requirements.tensor_parallel_size > 1 && !self.tensor_parallel.is_supported() {
            missing.push(format!(
                "tensor parallel size {}",
                requirements.tensor_parallel_size
            ));
        }
        if requirements.requires_kv_cache_registration
            && !self.kv_cache_memory_registration.is_supported()
        {
            missing.push("KV cache memory registration".to_string());
        }
        if requirements.requires_mooncake && !self.mooncake.is_supported() {
            missing.push("Mooncake transport".to_string());
        }

        if missing.is_empty() {
            Ok(())
        } else {
            Err(RuntimeCapabilityMismatch {
                runtime_name: self.runtime_name,
                missing,
            })
        }
    }

    pub fn backend_label(&self) -> &'static str {
        self.class.label()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeBackendMismatch {
    pub requested: RuntimeBackend,
    pub actual: &'static str,
    pub runtime_name: &'static str,
    pub reason: &'static str,
}

pub fn validate_runtime_backend(
    requested: RuntimeBackend,
    capability: &RuntimeCapability,
) -> Result<(), RuntimeBackendMismatch> {
    match requested {
        RuntimeBackend::Auto => Ok(()),
        RuntimeBackend::Cpu => {
            if capability.class == RuntimeCapabilityClass::CpuReference
                && capability.supports_forward
            {
                Ok(())
            } else {
                Err(RuntimeBackendMismatch {
                    requested,
                    actual: capability.backend_label(),
                    runtime_name: capability.runtime_name,
                    reason: "requested CPU device but the loaded runtime is not an executable CPU reference model",
                })
            }
        }
        requested @ (RuntimeBackend::Cuda
        | RuntimeBackend::Metal
        | RuntimeBackend::Rocm
        | RuntimeBackend::Musa
        | RuntimeBackend::Xpu
        | RuntimeBackend::Npu
        | RuntimeBackend::Hpu) => {
            if capability.class == RuntimeCapabilityClass::Production(requested)
                && capability.supports_forward
            {
                Ok(())
            } else {
                Err(RuntimeBackendMismatch {
                    requested,
                    actual: capability.backend_label(),
                    runtime_name: capability.runtime_name,
                    reason: "requested accelerator device but no matching executable production runtime is registered",
                })
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransferBackendClass {
    Production,
    Planned,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransferBackendCapability {
    pub backend: TransferBackend,
    pub class: TransferBackendClass,
    pub linked_transport: bool,
}

impl TransferBackendCapability {
    pub fn from_backend(backend: TransferBackend) -> Self {
        match backend {
            TransferBackend::Mooncake => Self {
                backend,
                class: TransferBackendClass::Production,
                linked_transport: cfg!(feature = "mooncake-link"),
            },
            TransferBackend::Nixl | TransferBackend::Ascend | TransferBackend::Mori => Self {
                backend,
                class: TransferBackendClass::Planned,
                linked_transport: false,
            },
        }
    }

    pub fn is_production(self) -> bool {
        self.class == TransferBackendClass::Production
    }
}

#[cfg(test)]
mod tests {
    use super::RuntimeDevicePlacement;

    #[cfg(target_os = "macos")]
    use super::RuntimeBackend;
    #[cfg(target_os = "macos")]
    use crate::backend_model::BackendProviderRegistry;

    #[test]
    fn device_placement_uses_base_step_and_local_tensor_parallel_rank() {
        let placement = RuntimeDevicePlacement::for_tensor_parallel_rank(2, 3, 5, 8, 2)
            .expect("placement should be valid");

        assert_eq!(placement.local_rank, 1);
        assert_eq!(placement.tensor_parallel_rank, 5);
        assert_eq!(placement.device_ordinal().expect("ordinal"), 5);
    }

    #[test]
    fn device_placement_rejects_invalid_distributed_geometry() {
        assert!(RuntimeDevicePlacement::for_tensor_parallel_rank(0, 0, 0, 1, 1).is_err());
        assert!(RuntimeDevicePlacement::for_tensor_parallel_rank(0, 1, 0, 3, 2).is_err());
        assert!(RuntimeDevicePlacement::for_tensor_parallel_rank(0, 1, 4, 4, 1).is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn auto_does_not_fall_back_to_cpu_reference_on_macos() {
        let placement = RuntimeDevicePlacement::for_tensor_parallel_rank(0, 1, 0, 1, 1)
            .expect("placement should be valid");
        let error = match BackendProviderRegistry::initialize(RuntimeBackend::Auto, placement) {
            Ok(_) => panic!("Mac has no registered production CUDA backend"),
            Err(error) => error,
        };

        assert!(error.message.contains("auto never falls back"));
        assert!(error.message.contains("CUDA device ordinal 0"));
    }
}
