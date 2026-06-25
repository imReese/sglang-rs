use std::fmt;

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
}

impl RuntimeCapability {
    pub fn cpu_reference(runtime_name: &'static str, supports_transferable_kv: bool) -> Self {
        Self {
            runtime_name,
            class: RuntimeCapabilityClass::CpuReference,
            supports_forward: true,
            supports_transferable_kv,
        }
    }

    pub fn metadata_only(runtime_name: &'static str, supports_transferable_kv: bool) -> Self {
        Self {
            runtime_name,
            class: RuntimeCapabilityClass::MetadataOnly,
            supports_forward: false,
            supports_transferable_kv,
        }
    }

    pub fn unsupported(runtime_name: &'static str) -> Self {
        Self {
            runtime_name,
            class: RuntimeCapabilityClass::Unsupported,
            supports_forward: false,
            supports_transferable_kv: false,
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
    Reference,
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
            TransferBackend::Fake => Self {
                backend,
                class: TransferBackendClass::Reference,
                linked_transport: false,
            },
            TransferBackend::Nixl | TransferBackend::Ascend | TransferBackend::Mori => Self {
                backend,
                class: TransferBackendClass::Planned,
                linked_transport: false,
            },
        }
    }

    pub fn is_reference_only(self) -> bool {
        self.class == TransferBackendClass::Reference
    }

    pub fn is_production(self) -> bool {
        self.class == TransferBackendClass::Production
    }
}
