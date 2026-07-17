use std::fmt;

#[cfg(feature = "mooncake-link")]
use crate::backend::RuntimeRequirements;
use crate::model_executor::ModelRunner;
use crate::model_registry::BootstrapForwardModel;
use crate::transfer::{
    KvCacheTransferError, MooncakeError, MooncakeKvCacheLayout, MooncakeKvCacheTransferExecutor,
    PdConfig,
};

#[cfg(not(feature = "mooncake-link"))]
use crate::transfer::UnlinkedMooncakeTransferEngine;
#[cfg(feature = "mooncake-link")]
use crate::transfer::{
    KvCacheMemoryProvider, KvTransferBackend, MooncakeKvCacheMemoryExt,
    MooncakeSessionTargetResolver, MooncakeTransferEngineConfig,
    SharedLinkedMooncakeTransferEngine,
};

#[cfg(feature = "mooncake-link")]
pub(crate) type ProductionMooncakeTransferBackend = MooncakeKvCacheTransferExecutor<
    SharedLinkedMooncakeTransferEngine,
    MooncakeSessionTargetResolver<SharedLinkedMooncakeTransferEngine>,
>;

#[cfg(not(feature = "mooncake-link"))]
pub(crate) type ProductionMooncakeTransferBackend =
    MooncakeKvCacheTransferExecutor<UnlinkedMooncakeTransferEngine>;

pub(crate) struct MooncakeTransferBackendBundle {
    pub(crate) backend: ProductionMooncakeTransferBackend,
    pub(crate) layout: MooncakeKvCacheLayout,
    pub(crate) local_endpoint: String,
}

#[derive(Debug)]
pub(crate) enum TransferBackendFactoryError {
    #[cfg(feature = "mooncake-link")]
    UnsupportedCacheArchitecture,
    #[cfg(feature = "mooncake-link")]
    MissingRuntimeCapabilities {
        runtime_name: String,
        missing: Vec<String>,
    },
    KvCache(KvCacheTransferError),
    Mooncake(MooncakeError),
}

impl fmt::Display for TransferBackendFactoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "mooncake-link")]
            Self::UnsupportedCacheArchitecture => formatter.write_str(
                "model cache architecture includes recurrent state; Mooncake KV-only transfer is insufficient",
            ),
            #[cfg(feature = "mooncake-link")]
            Self::MissingRuntimeCapabilities {
                runtime_name,
                missing,
            } => write!(
                formatter,
                "runtime {runtime_name} is missing required transfer capabilities: {}",
                missing.join(", ")
            ),
            Self::KvCache(error) => write!(formatter, "KV cache transfer error: {error}"),
            Self::Mooncake(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for TransferBackendFactoryError {}

impl From<KvCacheTransferError> for TransferBackendFactoryError {
    fn from(value: KvCacheTransferError) -> Self {
        Self::KvCache(value)
    }
}

impl From<MooncakeError> for TransferBackendFactoryError {
    fn from(value: MooncakeError) -> Self {
        Self::Mooncake(value)
    }
}

#[cfg(feature = "mooncake-link")]
pub(crate) fn build_mooncake_transfer_backend(
    model_runner: &mut ModelRunner<BootstrapForwardModel>,
    pd_config: &PdConfig,
    hostname: impl Into<String>,
    slot_capacity: usize,
    page_size: usize,
) -> Result<MooncakeTransferBackendBundle, TransferBackendFactoryError> {
    if !model_runner
        .model()
        .cache_architecture()
        .supports_kv_only_transfer()
    {
        return Err(TransferBackendFactoryError::UnsupportedCacheArchitecture);
    }

    model_runner
        .model()
        .runtime_capability()
        .validate_requirements(&RuntimeRequirements {
            requires_kv_cache_registration: true,
            requires_mooncake: true,
            ..RuntimeRequirements::default()
        })
        .map_err(
            |mismatch| TransferBackendFactoryError::MissingRuntimeCapabilities {
                runtime_name: mismatch.runtime_name.to_string(),
                missing: mismatch.missing,
            },
        )?;

    model_runner.reserve_transferable_kv_cache_slots(slot_capacity, page_size)?;
    let memory = model_runner
        .transferable_kv_cache_memory()
        .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?;
    if memory.regions().len() != 1 {
        return Err(KvCacheTransferError::Runtime(format!(
            "Mooncake backend requires one contiguous NexusKV region, descriptor has {}",
            memory.regions().len()
        ))
        .into());
    }

    let layout = memory.mooncake_prefill_layout(0);
    let device_ordinal = model_runner
        .model()
        .runtime_device_ordinal()
        .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?;
    let engine_config =
        MooncakeTransferEngineConfig::from_pd_config(hostname, device_ordinal, pd_config);
    let engine = SharedLinkedMooncakeTransferEngine::new(&engine_config)?;
    let local_endpoint = engine.local_endpoint()?;
    let target_resolver = MooncakeSessionTargetResolver::new(engine.clone(), Vec::new());
    let mut backend = MooncakeKvCacheTransferExecutor::with_target_resolver(
        engine.clone(),
        layout,
        target_resolver,
    )
    .with_memory_registrar(engine);
    backend.register(memory)?;

    Ok(MooncakeTransferBackendBundle {
        backend,
        layout,
        local_endpoint,
    })
}

#[cfg(not(feature = "mooncake-link"))]
pub(crate) fn build_mooncake_transfer_backend(
    _model_runner: &mut ModelRunner<BootstrapForwardModel>,
    _pd_config: &PdConfig,
    _hostname: impl Into<String>,
    _slot_capacity: usize,
    _page_size: usize,
) -> Result<MooncakeTransferBackendBundle, TransferBackendFactoryError> {
    Err(MooncakeError::UnavailableWithoutLink.into())
}
