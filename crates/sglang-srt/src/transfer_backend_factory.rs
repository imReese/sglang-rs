use std::fmt;

#[cfg(feature = "mooncake-link")]
use crate::backend::RuntimeRequirements;
use crate::model_executor::ModelRunner;
use crate::model_registry::BootstrapForwardModel;
use crate::pd_bootstrap::{MooncakeBootstrapKvCacheTransferExecutor, PrefillBootstrapService};
use crate::transfer::{
    DisaggregationMode, KvCacheTransferError, KvCacheTransferSpan, KvTransferBackend,
    KvTransferPoll, MooncakeError, MooncakeKvCacheLayout, MooncakeKvCacheTransferExecutor,
    PdConfig, TransferBackend, TransferableKvCacheMemory,
};
use crate::types::BootstrapRoom;

#[cfg(not(feature = "mooncake-link"))]
use crate::transfer::UnlinkedMooncakeTransferEngine;
#[cfg(feature = "mooncake-link")]
use crate::transfer::{
    KvCacheMemoryProvider, MooncakeKvCacheMemoryExt, MooncakeSessionTargetResolver,
    MooncakeTransferEngineConfig, SharedLinkedMooncakeTransferEngine,
};

#[cfg(feature = "mooncake-link")]
pub(crate) type ProductionMooncakeTransferBackend = MooncakeKvCacheTransferExecutor<
    SharedLinkedMooncakeTransferEngine,
    MooncakeSessionTargetResolver<SharedLinkedMooncakeTransferEngine>,
>;

#[cfg(not(feature = "mooncake-link"))]
pub(crate) type ProductionMooncakeTransferBackend =
    MooncakeKvCacheTransferExecutor<UnlinkedMooncakeTransferEngine>;

pub(crate) enum ProductionTransferBackend {
    Prefill(MooncakeBootstrapKvCacheTransferExecutor<ProductionMooncakeTransferBackend>),
    Decode(ProductionMooncakeTransferBackend),
}

impl KvTransferBackend for ProductionTransferBackend {
    fn register(&mut self, memory: TransferableKvCacheMemory) -> Result<(), KvCacheTransferError> {
        match self {
            Self::Prefill(backend) => backend.register(memory),
            Self::Decode(backend) => backend.register(memory),
        }
    }

    fn submit(&mut self, span: &KvCacheTransferSpan) -> Result<(), KvCacheTransferError> {
        match self {
            Self::Prefill(backend) => backend.submit(span),
            Self::Decode(backend) => backend.submit(span),
        }
    }

    fn poll(&mut self) -> Result<KvTransferPoll, KvCacheTransferError> {
        match self {
            Self::Prefill(backend) => backend.poll(),
            Self::Decode(backend) => backend.poll(),
        }
    }

    fn cancel(&mut self, bootstrap_room: BootstrapRoom) -> Result<(), KvCacheTransferError> {
        match self {
            Self::Prefill(backend) => backend.cancel(bootstrap_room),
            Self::Decode(backend) => backend.cancel(bootstrap_room),
        }
    }

    fn shutdown(&mut self) -> Result<(), KvCacheTransferError> {
        match self {
            Self::Prefill(backend) => backend.shutdown(),
            Self::Decode(backend) => backend.shutdown(),
        }
    }
}

pub(crate) struct ProductionTransferBackendBundle {
    pub(crate) backend: ProductionTransferBackend,
    pub(crate) layout: MooncakeKvCacheLayout,
    pub(crate) local_endpoint: String,
}

pub(crate) struct TransferBackendBuildContext<'a> {
    model_runner: &'a mut ModelRunner<BootstrapForwardModel>,
    pd_config: &'a PdConfig,
    hostname: String,
    slot_capacity: usize,
    page_size: usize,
    prefill_bootstrap_service: Option<PrefillBootstrapService>,
}

impl<'a> TransferBackendBuildContext<'a> {
    pub(crate) fn new(
        model_runner: &'a mut ModelRunner<BootstrapForwardModel>,
        pd_config: &'a PdConfig,
        hostname: impl Into<String>,
        slot_capacity: usize,
        page_size: usize,
    ) -> Self {
        Self {
            model_runner,
            pd_config,
            hostname: hostname.into(),
            slot_capacity,
            page_size,
            prefill_bootstrap_service: None,
        }
    }

    pub(crate) fn with_prefill_bootstrap_service(
        mut self,
        service: PrefillBootstrapService,
    ) -> Self {
        self.prefill_bootstrap_service = Some(service);
        self
    }
}

pub(crate) struct TransferBackendFactory;

impl TransferBackendFactory {
    pub(crate) fn build(
        role: DisaggregationMode,
        context: TransferBackendBuildContext<'_>,
    ) -> Result<ProductionTransferBackendBundle, TransferBackendFactoryError> {
        let TransferBackendBuildContext {
            model_runner,
            pd_config,
            hostname,
            slot_capacity,
            page_size,
            prefill_bootstrap_service,
        } = context;

        if role != pd_config.mode {
            return Err(TransferBackendFactoryError::RoleMismatch {
                requested: role,
                configured: pd_config.mode,
            });
        }
        if role == DisaggregationMode::Null {
            return Err(TransferBackendFactoryError::UnsupportedRole(role));
        }
        if pd_config.transfer_backend != TransferBackend::Mooncake {
            return Err(TransferBackendFactoryError::UnsupportedBackend {
                role,
                backend: pd_config.transfer_backend,
            });
        }
        let prefill_bootstrap_service = match role {
            DisaggregationMode::Prefill => Some(
                prefill_bootstrap_service
                    .ok_or(TransferBackendFactoryError::MissingPrefillBootstrapService)?,
            ),
            DisaggregationMode::Decode => None,
            DisaggregationMode::Null => unreachable!("null role is rejected before backend build"),
        };

        let bundle = build_mooncake_transfer_backend(
            model_runner,
            pd_config,
            hostname,
            slot_capacity,
            page_size,
        )?;
        let backend = match (role, prefill_bootstrap_service) {
            (DisaggregationMode::Prefill, Some(service)) => ProductionTransferBackend::Prefill(
                MooncakeBootstrapKvCacheTransferExecutor::new(service, bundle.backend),
            ),
            (DisaggregationMode::Decode, None) => ProductionTransferBackend::Decode(bundle.backend),
            _ => unreachable!("transfer role prerequisites are validated before backend build"),
        };

        Ok(ProductionTransferBackendBundle {
            backend,
            layout: bundle.layout,
            local_endpoint: bundle.local_endpoint,
        })
    }
}

struct MooncakeTransferBackendBundle {
    backend: ProductionMooncakeTransferBackend,
    layout: MooncakeKvCacheLayout,
    local_endpoint: String,
}

#[derive(Debug)]
pub(crate) enum TransferBackendFactoryError {
    RoleMismatch {
        requested: DisaggregationMode,
        configured: DisaggregationMode,
    },
    UnsupportedRole(DisaggregationMode),
    UnsupportedBackend {
        role: DisaggregationMode,
        backend: TransferBackend,
    },
    MissingPrefillBootstrapService,
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
            Self::RoleMismatch {
                requested,
                configured,
            } => write!(
                formatter,
                "transfer backend role {requested:?} does not match configured PD role {configured:?}"
            ),
            Self::UnsupportedRole(role) => {
                write!(formatter, "transfer backend cannot be built for PD role {role:?}")
            }
            Self::UnsupportedBackend { role, backend } => write!(
                formatter,
                "transfer backend {backend:?} is not implemented for PD role {role:?}"
            ),
            Self::MissingPrefillBootstrapService => formatter.write_str(
                "prefill transfer backend requires a bootstrap metadata service",
            ),
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
fn build_mooncake_transfer_backend(
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
fn build_mooncake_transfer_backend(
    _model_runner: &mut ModelRunner<BootstrapForwardModel>,
    _pd_config: &PdConfig,
    _hostname: impl Into<String>,
    _slot_capacity: usize,
    _page_size: usize,
) -> Result<MooncakeTransferBackendBundle, TransferBackendFactoryError> {
    Err(MooncakeError::UnavailableWithoutLink.into())
}
