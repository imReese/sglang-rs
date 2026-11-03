use std::fmt;

use crate::backend::RuntimeRequirements;
use crate::model_executor::ModelRunner;
use crate::model_registry::BootstrapForwardModel;
use crate::pd_bootstrap::{PdBootstrapKvCacheTransferExecutor, PrefillBootstrapService};
use crate::tcp_transfer::SharedTcpTransferEngine;
use crate::transfer::{
    DisaggregationMode, KvCacheMemoryExt, KvCacheMemoryProvider, KvCacheTransferError,
    KvCacheTransferExecutor, KvCacheTransferSpan, KvTransferBackend, KvTransferLayout,
    KvTransferPoll, KvTransportError, PdConfig, SessionKvTargetResolver, TransferBackend,
    TransferableKvCacheMemory,
};
use crate::types::BootstrapRoom;

pub(crate) type ProductionTcpTransferBackend = KvCacheTransferExecutor<
    SharedTcpTransferEngine,
    SessionKvTargetResolver<SharedTcpTransferEngine>,
>;

pub(crate) enum ProductionTransferBackend {
    TcpPrefill(PdBootstrapKvCacheTransferExecutor<ProductionTcpTransferBackend>),
    TcpDecode(ProductionTcpTransferBackend),
}

impl KvTransferBackend for ProductionTransferBackend {
    fn register(&mut self, memory: TransferableKvCacheMemory) -> Result<(), KvCacheTransferError> {
        match self {
            Self::TcpPrefill(backend) => backend.register(memory),
            Self::TcpDecode(backend) => backend.register(memory),
        }
    }

    fn submit(&mut self, span: &KvCacheTransferSpan) -> Result<(), KvCacheTransferError> {
        match self {
            Self::TcpPrefill(backend) => backend.submit(span),
            Self::TcpDecode(backend) => backend.submit(span),
        }
    }

    fn poll(&mut self) -> Result<KvTransferPoll, KvCacheTransferError> {
        match self {
            Self::TcpPrefill(backend) => backend.poll(),
            Self::TcpDecode(backend) => backend.poll(),
        }
    }

    fn cancel(&mut self, bootstrap_room: BootstrapRoom) -> Result<(), KvCacheTransferError> {
        match self {
            Self::TcpPrefill(backend) => backend.cancel(bootstrap_room),
            Self::TcpDecode(backend) => backend.cancel(bootstrap_room),
        }
    }

    fn shutdown(&mut self) -> Result<(), KvCacheTransferError> {
        match self {
            Self::TcpPrefill(backend) => backend.shutdown(),
            Self::TcpDecode(backend) => backend.shutdown(),
        }
    }
}

pub(crate) struct ProductionTransferBackendBundle {
    pub(crate) backend: ProductionTransferBackend,
    pub(crate) layout: KvTransferLayout,
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
        let prefill_bootstrap_service = match role {
            DisaggregationMode::Prefill => Some(
                prefill_bootstrap_service
                    .ok_or(TransferBackendFactoryError::MissingPrefillBootstrapService)?,
            ),
            DisaggregationMode::Decode => None,
            DisaggregationMode::Null => unreachable!("null role is rejected before backend build"),
        };

        match pd_config.transfer_backend {
            TransferBackend::Tcp => {
                let bundle = build_tcp_transfer_backend(
                    model_runner,
                    pd_config,
                    hostname,
                    slot_capacity,
                    page_size,
                )?;
                let backend = match (role, prefill_bootstrap_service) {
                    (DisaggregationMode::Prefill, Some(service)) => {
                        ProductionTransferBackend::TcpPrefill(
                            PdBootstrapKvCacheTransferExecutor::new(service, bundle.backend),
                        )
                    }
                    (DisaggregationMode::Decode, None) => {
                        ProductionTransferBackend::TcpDecode(bundle.backend)
                    }
                    _ => unreachable!(
                        "transfer role prerequisites are validated before backend build"
                    ),
                };

                Ok(ProductionTransferBackendBundle {
                    backend,
                    layout: bundle.layout,
                    local_endpoint: bundle.local_endpoint,
                })
            }
            other => Err(TransferBackendFactoryError::UnsupportedBackend {
                role,
                backend: other,
            }),
        }
    }
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
    UnsupportedCacheArchitecture,
    MissingRuntimeCapabilities {
        runtime_name: String,
        missing: Vec<String>,
    },
    KvCache(KvCacheTransferError),
    Transport(KvTransportError),
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
                write!(
                    formatter,
                    "transfer backend cannot be built for PD role {role:?}"
                )
            }
            Self::UnsupportedBackend { role, backend } => write!(
                formatter,
                "transfer backend {backend:?} is not implemented for PD role {role:?}"
            ),
            Self::MissingPrefillBootstrapService => formatter
                .write_str("prefill transfer backend requires a bootstrap metadata service"),
            Self::UnsupportedCacheArchitecture => formatter.write_str(
                "model cache architecture does not declare a transferable KV cache or hybrid state",
            ),
            Self::MissingRuntimeCapabilities {
                runtime_name,
                missing,
            } => write!(
                formatter,
                "runtime {runtime_name} is missing required transfer capabilities: {}",
                missing.join(", ")
            ),
            Self::KvCache(error) => write!(formatter, "KV cache transfer error: {error}"),
            Self::Transport(error) => write!(formatter, "KV transport error: {error}"),
        }
    }
}

impl std::error::Error for TransferBackendFactoryError {}

impl From<KvCacheTransferError> for TransferBackendFactoryError {
    fn from(value: KvCacheTransferError) -> Self {
        Self::KvCache(value)
    }
}

impl From<KvTransportError> for TransferBackendFactoryError {
    fn from(value: KvTransportError) -> Self {
        Self::Transport(value)
    }
}

struct TcpTransferBackendBundle {
    backend: ProductionTcpTransferBackend,
    layout: KvTransferLayout,
    local_endpoint: String,
}

fn build_tcp_transfer_backend(
    model_runner: &mut ModelRunner<BootstrapForwardModel>,
    pd_config: &PdConfig,
    hostname: impl Into<String>,
    slot_capacity: usize,
    page_size: usize,
) -> Result<TcpTransferBackendBundle, TransferBackendFactoryError> {
    if matches!(
        model_runner.model().cache_architecture(),
        crate::models::ModelCacheArchitecture::None
    ) {
        return Err(TransferBackendFactoryError::UnsupportedCacheArchitecture);
    }

    model_runner
        .model()
        .runtime_capability()
        .validate_requirements(&RuntimeRequirements {
            requires_kv_cache_registration: true,
            requires_kv_transfer: false,
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
            "TCP backend requires one contiguous NexusKV region, descriptor has {}",
            memory.regions().len()
        ))
        .into());
    }

    let layout = memory.kv_prefill_layout(0);
    let ranks_per_node = pd_config
        .tp_size
        .checked_div(pd_config.nnodes.max(1))
        .unwrap_or(1);
    let rank_offset = u16::try_from(pd_config.node_rank * ranks_per_node).unwrap_or(0);
    let port = pd_config
        .transfer_rpc_port
        .map(|p| {
            if p == 0 {
                0
            } else {
                p.saturating_add(rank_offset)
            }
        })
        .unwrap_or(0);
    let hostname = hostname.into();
    let engine =
        SharedTcpTransferEngine::new(hostname, port).map_err(KvCacheTransferError::Runtime)?;
    let local_endpoint = engine.local_endpoint()?;
    let target_resolver = SessionKvTargetResolver::new(engine.clone(), Vec::new());
    let mut backend =
        KvCacheTransferExecutor::with_target_resolver(engine.clone(), layout, target_resolver)
            .with_memory_registrar(engine);
    backend.register(memory)?;

    Ok(TcpTransferBackendBundle {
        backend,
        layout,
        local_endpoint,
    })
}
