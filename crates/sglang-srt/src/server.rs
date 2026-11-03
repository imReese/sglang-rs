use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::Path;

use crate::backend::{RuntimeBackend, validate_runtime_backend};
use crate::cache::{CachePageAllocator, RadixCache};
use crate::cli::ServerArgs;
use crate::dp_coordinator::{DpCoordinator, DpCoordinatorError};
use crate::dp_scheduler::DpScheduler;
use crate::dp_worker_pool::DpWorkerPool;
use crate::engine::Engine;
use crate::engine_info_bootstrap::EngineInfoBootstrapService;
use crate::grpc::{
    GrpcRouterService, GrpcServeError, serve_grpc_router_with_shutdown,
    server_info_attributes_from_args,
};
use crate::grpc_sidecar::serve_grpc_http_sidecar_with_shutdown;
use crate::http::{
    HttpKvCacheInfo, HttpKvEventsInfo, HttpRouterService, HttpServeError, HttpServerInfo,
    serve_http_router_with_shutdown,
};
use crate::model_artifacts::{HfModelConfig, LocalModelArtifacts, ModelArtifactError};
use crate::model_executor::ModelRunner;
pub use crate::model_registry::BootstrapForwardModel;
use crate::model_registry::{ModelRegistry, ModelRegistryError};
use crate::parallel_topology::{Parallel4DConfig, Parallel4DTopology};
use crate::pd_bootstrap::{
    MooncakeBootstrapKvCacheTransferExecutor, MooncakeDecodeBootstrapPublisher,
    PrefillBootstrapServeError, PrefillBootstrapService, PrefillRouteRegistration,
    serve_mooncake_bootstrap_zmq_endpoints_with_shutdown, serve_prefill_bootstrap_with_shutdown,
};
use crate::router::{RouterGetModelInfoResponse, RouterRuntime};
use crate::runtime_kv_cache::RuntimeKvCacheMetadata;
use crate::scheduler::{PrefixCachePolicy, Scheduler};
use crate::tokenizer::{RuntimeTokenizer, Tokenizer, TokenizerError};
#[cfg(any(test, feature = "test-support"))]
use crate::transfer::FakeKvCacheTransferExecutor;
use crate::transfer::{
    DecodeBootstrapPublisher, DecodeBootstrapRegistry, DisaggregationMode, KvTransferBackend,
    KvTransferModelWorker, MooncakeBatchReleaser, MooncakeError, MooncakeKvCacheLayout,
    MooncakeKvCacheTransferExecutor, MooncakeTransferStatusReader, MooncakeTransferSubmitter,
    MooncakeTransferTargetResolver, PdConfig, PdConfigError, TransferBackend,
};
use crate::transfer_backend_factory::{
    ProductionTransferBackend, TransferBackendBuildContext, TransferBackendFactory,
    TransferBackendFactoryError,
};
use crate::worker::WorkerExecutor;

#[doc(hidden)]
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use super::*;
    use crate::model_executor::{
        ForwardModel, ModelForwardError, ModelForwardOutput, ModelWorkerBatch,
    };
    use crate::transfer::NoopDecodeBootstrapPublisher;

    #[derive(Clone, Debug, Default)]
    pub struct CpuReferenceModel;

    impl ForwardModel for CpuReferenceModel {
        fn forward(
            &mut self,
            batch: &ModelWorkerBatch,
        ) -> Result<ModelForwardOutput, ModelForwardError> {
            let logits = batch
                .request_ids()
                .iter()
                .map(|_| {
                    let mut row = vec![0.0; (b' ' as usize) + 1];
                    row[b' ' as usize] = 1.0;
                    row
                })
                .collect();
            ModelForwardOutput::new(logits)
        }
    }

    pub type ReferenceGrpcRouterService =
        GrpcRouterService<RuntimeTokenizer, ModelRunner<CpuReferenceModel>>;
    pub type ReferenceHttpRouterService =
        HttpRouterService<RuntimeTokenizer, ModelRunner<CpuReferenceModel>>;
    pub type ReferencePrefillHttpRouterService = HttpRouterService<
        RuntimeTokenizer,
        KvTransferModelWorker<ModelRunner<CpuReferenceModel>, FakeKvCacheTransferExecutor>,
    >;
    pub type ReferencePdHttpRouterService<E> = HttpRouterService<
        RuntimeTokenizer,
        KvTransferModelWorker<ModelRunner<CpuReferenceModel>, E>,
    >;
    pub type ReferencePdGrpcRouterService<E> = GrpcRouterService<
        RuntimeTokenizer,
        KvTransferModelWorker<ModelRunner<CpuReferenceModel>, E>,
    >;

    pub fn build_reference_grpc_router_service(args: &ServerArgs) -> ReferenceGrpcRouterService {
        try_build_reference_grpc_router_service(args).expect("reference tokenizer should load")
    }

    pub fn try_build_reference_grpc_router_service(
        args: &ServerArgs,
    ) -> Result<ReferenceGrpcRouterService, ServerLaunchError> {
        let scheduler = Scheduler::new(ModelRunner::new(CpuReferenceModel))
            .with_max_running_requests(args.max_running_requests)
            .with_chunked_prefill_size(args.chunked_prefill_size);
        let tokenizer = reference_tokenizer_from_args(args)?;
        let runtime = RouterRuntime::new(Engine::new(tokenizer, scheduler))
            .with_default_stop_token_ids(model_config_eos_token_ids(&args.model_path));
        Ok(GrpcRouterService::with_server_args(runtime, args))
    }

    pub fn build_reference_http_router_service(args: &ServerArgs) -> ReferenceHttpRouterService {
        let scheduler = Scheduler::new(ModelRunner::new(CpuReferenceModel))
            .with_max_running_requests(args.max_running_requests)
            .with_chunked_prefill_size(args.chunked_prefill_size);
        let tokenizer =
            reference_tokenizer_from_args(args).expect("reference tokenizer should load");
        let runtime = RouterRuntime::new(Engine::new(tokenizer, scheduler))
            .with_default_stop_token_ids(model_config_eos_token_ids(&args.model_path));
        HttpRouterService::new(runtime, RouterGetModelInfoResponse::from_server_args(args))
            .with_server_info(http_server_info_from_args(args))
    }

    pub fn build_reference_prefill_http_router_service(
        args: &ServerArgs,
    ) -> ReferencePrefillHttpRouterService {
        try_build_reference_prefill_http_router_service(args)
            .expect("reference prefill service should build")
    }

    pub fn try_build_reference_prefill_http_router_service(
        args: &ServerArgs,
    ) -> Result<ReferencePrefillHttpRouterService, ServerLaunchError> {
        let worker = KvTransferModelWorker::new(
            ModelRunner::new(CpuReferenceModel),
            DecodeBootstrapRegistry::default(),
            FakeKvCacheTransferExecutor::default(),
        )
        .with_kv_page_size(args.page_size);
        let scheduler = Scheduler::with_cache_resources(
            worker,
            RadixCache::default(),
            cache_page_allocator_from_server_args(args)?,
        )
        .with_max_running_requests(args.max_running_requests)
        .with_chunked_prefill_size(args.chunked_prefill_size);
        Ok(reference_http_service(args, scheduler).with_disaggregated_requests())
    }

    pub fn build_reference_pd_http_router_service<E>(
        args: &ServerArgs,
        registry: DecodeBootstrapRegistry,
        transfer_executor: E,
    ) -> ReferencePdHttpRouterService<E>
    where
        E: KvTransferBackend,
    {
        let worker = KvTransferModelWorker::new(
            ModelRunner::new(CpuReferenceModel),
            registry,
            transfer_executor,
        )
        .with_decode_bootstrap_publisher(NoopDecodeBootstrapPublisher)
        .with_kv_page_size(args.page_size);
        let scheduler = Scheduler::with_cache_resources(
            worker,
            RadixCache::default(),
            cache_page_allocator_from_server_args(args)
                .expect("reference KV cache arguments should be valid"),
        )
        .with_max_running_requests(args.max_running_requests)
        .with_chunked_prefill_size(args.chunked_prefill_size);
        reference_http_service(args, scheduler)
            .with_disaggregated_requests()
            .with_max_transfer_polls(args.disaggregation_decode_polling_interval)
    }

    pub fn build_reference_mooncake_prefill_http_router_service<S, R>(
        args: &ServerArgs,
        bootstrap_service: PrefillBootstrapService,
        transfer_executor: MooncakeKvCacheTransferExecutor<S, R>,
    ) -> ReferencePdHttpRouterService<
        MooncakeBootstrapKvCacheTransferExecutor<MooncakeKvCacheTransferExecutor<S, R>>,
    >
    where
        S: MooncakeTransferSubmitter + MooncakeTransferStatusReader + MooncakeBatchReleaser,
        R: MooncakeTransferTargetResolver,
    {
        build_reference_pd_http_router_service(
            args,
            DecodeBootstrapRegistry::default(),
            MooncakeBootstrapKvCacheTransferExecutor::new(bootstrap_service, transfer_executor),
        )
    }

    pub fn build_reference_pd_grpc_router_service<E>(
        args: &ServerArgs,
        registry: DecodeBootstrapRegistry,
        transfer_executor: E,
    ) -> ReferencePdGrpcRouterService<E>
    where
        E: KvTransferBackend,
    {
        let worker = KvTransferModelWorker::new(
            ModelRunner::new(CpuReferenceModel),
            registry,
            transfer_executor,
        )
        .with_decode_bootstrap_publisher(NoopDecodeBootstrapPublisher)
        .with_kv_page_size(args.page_size);
        let scheduler = Scheduler::with_cache_resources(
            worker,
            RadixCache::default(),
            cache_page_allocator_from_server_args(args)
                .expect("reference KV cache arguments should be valid"),
        )
        .with_max_running_requests(args.max_running_requests)
        .with_chunked_prefill_size(args.chunked_prefill_size);
        let tokenizer =
            reference_tokenizer_from_args(args).expect("reference tokenizer should load");
        let runtime = RouterRuntime::new(Engine::new(tokenizer, scheduler))
            .with_default_stop_token_ids(model_config_eos_token_ids(&args.model_path));
        GrpcRouterService::with_server_args(runtime, args)
            .with_max_transfer_polls(args.disaggregation_decode_polling_interval)
    }

    pub fn build_reference_fake_pd_grpc_router_service(
        args: &ServerArgs,
    ) -> ReferencePdGrpcRouterService<FakeKvCacheTransferExecutor> {
        build_reference_pd_grpc_router_service(
            args,
            DecodeBootstrapRegistry::default(),
            FakeKvCacheTransferExecutor::default(),
        )
    }

    fn reference_http_service<W>(
        args: &ServerArgs,
        scheduler: Scheduler<W>,
    ) -> HttpRouterService<RuntimeTokenizer, W>
    where
        W: WorkerExecutor,
    {
        let tokenizer =
            reference_tokenizer_from_args(args).expect("reference tokenizer should load");
        let runtime = RouterRuntime::new(Engine::new(tokenizer, scheduler))
            .with_default_stop_token_ids(model_config_eos_token_ids(&args.model_path));
        HttpRouterService::new(runtime, RouterGetModelInfoResponse::from_server_args(args))
            .with_server_info(http_server_info_from_args(args))
    }

    fn reference_tokenizer_from_args(
        args: &ServerArgs,
    ) -> Result<RuntimeTokenizer, TokenizerError> {
        if args.tokenizer_path.is_some()
            || Path::new(&args.model_path).join("tokenizer.json").is_file()
            || Path::new(&args.model_path)
                .file_name()
                .is_some_and(|name| name == "tokenizer.json")
        {
            return RuntimeTokenizer::from_model_or_tokenizer_path(
                &args.model_path,
                args.tokenizer_path.as_deref(),
            );
        }

        Ok(RuntimeTokenizer::Byte(crate::tokenizer::ByteTokenizer))
    }
}

fn bootstrap_forward_model_from_server_args(
    args: &ServerArgs,
) -> Result<BootstrapForwardModel, ServerLaunchError> {
    let model = BootstrapForwardModel::from_server_args(args)?;
    validate_bootstrap_runtime_backend(args, &model)?;
    Ok(model)
}

fn bootstrap_forward_model_for_launch(
    args: &ServerArgs,
) -> Result<BootstrapForwardModel, ServerLaunchError> {
    let model = BootstrapForwardModel::from_server_args(args)?;
    validate_bootstrap_runtime_backend(args, &model)?;
    validate_bootstrap_runtime_requirements(args, &model)?;
    Ok(model)
}

fn bootstrap_model_runner(
    model: BootstrapForwardModel,
) -> Result<ModelRunner<BootstrapForwardModel>, ServerLaunchError> {
    let kv_cache_layout = model.kv_cache_layout();
    Ok(ModelRunner::new_with_kv_cache_layout(
        model,
        kv_cache_layout,
    ))
}

fn model_prefix_cache_policy(model: &BootstrapForwardModel) -> PrefixCachePolicy {
    if model.cache_architecture().supports_radix_prefix_cache() {
        PrefixCachePolicy::Enabled
    } else {
        PrefixCachePolicy::Disabled
    }
}

fn validate_kv_only_transfer_model(model: &BootstrapForwardModel) -> Result<(), ServerLaunchError> {
    if model.cache_architecture().supports_kv_only_transfer() {
        return Ok(());
    }
    Err(ServerLaunchError::MissingRuntimeCapabilities {
        runtime_name: model.runtime_capability().runtime_name.to_string(),
        missing: vec![
            "hybrid recurrent state memory registration and transfer; KV-only transfer is insufficient"
                .to_string(),
        ],
    })
}

fn validate_bootstrap_runtime_backend(
    args: &ServerArgs,
    model: &BootstrapForwardModel,
) -> Result<(), ServerLaunchError> {
    let requested = RuntimeBackend::parse(&args.device)
        .ok_or_else(|| ServerLaunchError::InvalidDevice(args.device.clone()))?;
    let capability = model.runtime_capability();
    validate_runtime_backend(requested, &capability).map_err(|mismatch| {
        ServerLaunchError::UnsupportedDevice {
            requested: mismatch.requested.to_string(),
            actual: mismatch.actual.to_string(),
            runtime_name: mismatch.runtime_name.to_string(),
            reason: mismatch.reason.to_string(),
        }
    })
}

fn validate_bootstrap_runtime_requirements(
    args: &ServerArgs,
    model: &BootstrapForwardModel,
) -> Result<(), ServerLaunchError> {
    let capability = model.runtime_capability();
    let requirements = model.runtime_requirements(args.tp_size, args.attention_backend.as_deref());
    capability
        .validate_requirements(&requirements)
        .map_err(|mismatch| ServerLaunchError::MissingRuntimeCapabilities {
            runtime_name: mismatch.runtime_name.to_string(),
            missing: mismatch.missing,
        })
}

pub type BootstrapGrpcRouterService =
    GrpcRouterService<RuntimeTokenizer, ModelRunner<BootstrapForwardModel>>;
pub type BootstrapHttpRouterService =
    HttpRouterService<RuntimeTokenizer, ModelRunner<BootstrapForwardModel>>;
#[cfg(any(test, feature = "test-support"))]
pub type BootstrapPrefillHttpRouterService = HttpRouterService<
    RuntimeTokenizer,
    KvTransferModelWorker<ModelRunner<BootstrapForwardModel>, FakeKvCacheTransferExecutor>,
>;
pub type BootstrapPdHttpRouterService<E, P = crate::transfer::NoopDecodeBootstrapPublisher> =
    HttpRouterService<
        RuntimeTokenizer,
        KvTransferModelWorker<ModelRunner<BootstrapForwardModel>, E, P>,
    >;
pub type BootstrapPdGrpcRouterService<E, P = crate::transfer::NoopDecodeBootstrapPublisher> =
    GrpcRouterService<
        RuntimeTokenizer,
        KvTransferModelWorker<ModelRunner<BootstrapForwardModel>, E, P>,
    >;

enum ProductionDecodeBootstrapPublisher {
    Noop(crate::transfer::NoopDecodeBootstrapPublisher),
    Mooncake(Box<MooncakeDecodeBootstrapPublisher>),
}

impl DecodeBootstrapPublisher for ProductionDecodeBootstrapPublisher {
    fn publish_decode_bootstrap_metadata(
        &mut self,
        plan: &crate::transfer::KvCacheTransferPlan,
    ) -> Result<crate::transfer::DecodeBootstrapMetadataPublishSummary, String> {
        match self {
            Self::Noop(publisher) => publisher.publish_decode_bootstrap_metadata(plan),
            Self::Mooncake(publisher) => publisher.publish_decode_bootstrap_metadata(plan),
        }
    }

    fn receive_prefill_token(&mut self, room: crate::types::BootstrapRoom) -> Result<u32, String> {
        match self {
            Self::Mooncake(publisher) => publisher.receive_prefill_token(room),
            Self::Noop(_) => Err("prefill role does not receive remote KV".into()),
        }
    }

    fn cancel_receive(&mut self, room: crate::types::BootstrapRoom) -> Result<(), String> {
        match self {
            Self::Mooncake(publisher) => publisher.cancel_receive(room),
            Self::Noop(_) => Err("prefill role does not own a decode receive".into()),
        }
    }
}

type ProductionPdHttpRouterService =
    BootstrapPdHttpRouterService<ProductionTransferBackend, ProductionDecodeBootstrapPublisher>;
type ProductionPdGrpcRouterService =
    BootstrapPdGrpcRouterService<ProductionTransferBackend, ProductionDecodeBootstrapPublisher>;
#[cfg(any(test, feature = "test-support"))]
pub type BootstrapFakePdGrpcRouterService =
    BootstrapPdGrpcRouterService<FakeKvCacheTransferExecutor>;

#[derive(Debug)]
pub enum ServerLaunchError {
    AddressResolve(std::io::Error),
    ShutdownSignal(std::io::Error),
    NoSocketAddress {
        host: String,
        port: u16,
    },
    InvalidGrpcSidecarPort {
        grpc_port: u16,
    },
    PdConfig(PdConfigError),
    UnsupportedBootstrapPdRuntime {
        mode: DisaggregationMode,
        transfer_backend: TransferBackend,
    },
    ModelArtifact(ModelArtifactError),
    ModelRegistry(ModelRegistryError),
    Tokenizer(TokenizerError),
    Grpc(GrpcServeError),
    Http(HttpServeError),
    PrefillBootstrap(PrefillBootstrapServeError),
    MooncakeTransfer(MooncakeError),
    TransferBackendFactory(String),
    KvCacheTransfer(String),
    UnsupportedMooncakeKvMemory {
        model_path: String,
        model_type: Option<String>,
    },
    InvalidDevice(String),
    UnsupportedDevice {
        requested: String,
        actual: String,
        runtime_name: String,
        reason: String,
    },
    MissingRuntimeCapabilities {
        runtime_name: String,
        missing: Vec<String>,
    },
    ServerTaskJoin(String),
    ZmqRoutePortCountMismatch {
        expected: usize,
        actual: usize,
    },
    DpCoordinator(String),
}

impl PartialEq for ServerLaunchError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::NoSocketAddress {
                    host: left_host,
                    port: left_port,
                },
                Self::NoSocketAddress {
                    host: right_host,
                    port: right_port,
                },
            ) => left_host == right_host && left_port == right_port,
            (
                Self::InvalidGrpcSidecarPort {
                    grpc_port: left_port,
                },
                Self::InvalidGrpcSidecarPort {
                    grpc_port: right_port,
                },
            ) => left_port == right_port,
            (Self::PdConfig(left), Self::PdConfig(right)) => left == right,
            (
                Self::UnsupportedBootstrapPdRuntime {
                    mode: left_mode,
                    transfer_backend: left_backend,
                },
                Self::UnsupportedBootstrapPdRuntime {
                    mode: right_mode,
                    transfer_backend: right_backend,
                },
            ) => left_mode == right_mode && left_backend == right_backend,
            (Self::Tokenizer(left), Self::Tokenizer(right)) => left == right,
            (Self::ModelArtifact(left), Self::ModelArtifact(right)) => left == right,
            (Self::ModelRegistry(left), Self::ModelRegistry(right)) => left == right,
            (Self::TransferBackendFactory(left), Self::TransferBackendFactory(right)) => {
                left == right
            }
            (Self::KvCacheTransfer(left), Self::KvCacheTransfer(right)) => left == right,
            (
                Self::UnsupportedMooncakeKvMemory {
                    model_path: left_model_path,
                    model_type: left_model_type,
                },
                Self::UnsupportedMooncakeKvMemory {
                    model_path: right_model_path,
                    model_type: right_model_type,
                },
            ) => left_model_path == right_model_path && left_model_type == right_model_type,
            (Self::InvalidDevice(left), Self::InvalidDevice(right)) => left == right,
            (
                Self::UnsupportedDevice {
                    requested: left_requested,
                    actual: left_actual,
                    runtime_name: left_runtime_name,
                    reason: left_reason,
                },
                Self::UnsupportedDevice {
                    requested: right_requested,
                    actual: right_actual,
                    runtime_name: right_runtime_name,
                    reason: right_reason,
                },
            ) => {
                left_requested == right_requested
                    && left_actual == right_actual
                    && left_runtime_name == right_runtime_name
                    && left_reason == right_reason
            }
            (
                Self::MissingRuntimeCapabilities {
                    runtime_name: left_runtime_name,
                    missing: left_missing,
                },
                Self::MissingRuntimeCapabilities {
                    runtime_name: right_runtime_name,
                    missing: right_missing,
                },
            ) => left_runtime_name == right_runtime_name && left_missing == right_missing,
            (
                Self::ZmqRoutePortCountMismatch {
                    expected: left_expected,
                    actual: left_actual,
                },
                Self::ZmqRoutePortCountMismatch {
                    expected: right_expected,
                    actual: right_actual,
                },
            ) => left_expected == right_expected && left_actual == right_actual,
            (Self::DpCoordinator(left), Self::DpCoordinator(right)) => left == right,
            _ => false,
        }
    }
}

impl fmt::Display for ServerLaunchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AddressResolve(error) => {
                write!(formatter, "failed to resolve listen address: {error}")
            }
            Self::ShutdownSignal(error) => {
                write!(
                    formatter,
                    "failed to install graceful shutdown signal handler: {error}"
                )
            }
            Self::NoSocketAddress { host, port } => {
                write!(formatter, "listen address {host}:{port} did not resolve")
            }
            Self::InvalidGrpcSidecarPort { grpc_port } => write!(
                formatter,
                "gRPC HTTP sidecar defaults to --port + 1, but gRPC port {grpc_port} has no successor; set --smg-http-sidecar-port explicitly"
            ),
            Self::PdConfig(error) => write!(formatter, "PD config error: {error}"),
            Self::UnsupportedBootstrapPdRuntime {
                mode,
                transfer_backend,
            } => write!(
                formatter,
                "bootstrap server does not support PD runtime mode {mode:?} with backend {transfer_backend:?}"
            ),
            Self::ModelArtifact(error) => write!(formatter, "model artifact error: {error}"),
            Self::ModelRegistry(error) => write!(formatter, "model registry error: {error}"),
            Self::Tokenizer(error) => write!(formatter, "tokenizer error: {error}"),
            Self::Grpc(error) => write!(formatter, "{error}"),
            Self::Http(error) => write!(formatter, "{error}"),
            Self::PrefillBootstrap(error) => write!(formatter, "{error}"),
            Self::MooncakeTransfer(error) => write!(formatter, "{error}"),
            Self::TransferBackendFactory(error) => {
                write!(formatter, "transfer backend initialization error: {error}")
            }
            Self::KvCacheTransfer(error) => write!(formatter, "KV cache transfer error: {error}"),
            Self::UnsupportedMooncakeKvMemory {
                model_path,
                model_type,
            } => write!(
                formatter,
                "model {model_path} type {} does not expose transferable Mooncake KV memory",
                model_type.as_deref().unwrap_or("<unknown>")
            ),
            Self::InvalidDevice(value) => {
                write!(formatter, "invalid --device: {value}")
            }
            Self::UnsupportedDevice {
                requested,
                actual,
                runtime_name,
                reason,
            } => write!(
                formatter,
                "device {requested} is not supported by loaded runtime {runtime_name} ({actual}): {reason}"
            ),
            Self::MissingRuntimeCapabilities {
                runtime_name,
                missing,
            } => write!(
                formatter,
                "runtime {runtime_name} is missing required capabilities: {}",
                missing.join(", ")
            ),
            Self::ServerTaskJoin(error) => write!(formatter, "server task failed to join: {error}"),
            Self::ZmqRoutePortCountMismatch { expected, actual } => write!(
                formatter,
                "prefill Mooncake ZMQ route port count mismatch: expected {expected}, got {actual}"
            ),
            Self::DpCoordinator(error) => {
                write!(formatter, "DP coordinator launch error: {error}")
            }
        }
    }
}

impl std::error::Error for ServerLaunchError {}

impl From<GrpcServeError> for ServerLaunchError {
    fn from(value: GrpcServeError) -> Self {
        Self::Grpc(value)
    }
}

impl From<HttpServeError> for ServerLaunchError {
    fn from(value: HttpServeError) -> Self {
        Self::Http(value)
    }
}

impl From<PrefillBootstrapServeError> for ServerLaunchError {
    fn from(value: PrefillBootstrapServeError) -> Self {
        Self::PrefillBootstrap(value)
    }
}

impl From<MooncakeError> for ServerLaunchError {
    fn from(value: MooncakeError) -> Self {
        Self::MooncakeTransfer(value)
    }
}

impl From<TransferBackendFactoryError> for ServerLaunchError {
    fn from(value: TransferBackendFactoryError) -> Self {
        match value {
            error
                @ (TransferBackendFactoryError::RoleMismatch { .. }
                | TransferBackendFactoryError::UnsupportedRole(_)
                | TransferBackendFactoryError::UnsupportedBackend { .. }
                | TransferBackendFactoryError::MissingPrefillBootstrapService) => {
                    Self::TransferBackendFactory(error.to_string())
                }
            TransferBackendFactoryError::UnsupportedCacheArchitecture => {
                Self::MissingRuntimeCapabilities {
                    runtime_name: "model-cache-architecture".to_string(),
                    missing: vec![
                        "hybrid recurrent state memory registration and transfer; KV-only transfer is insufficient"
                            .to_string(),
                    ],
                }
            }
            TransferBackendFactoryError::MissingRuntimeCapabilities {
                runtime_name,
                missing,
            } => Self::MissingRuntimeCapabilities {
                runtime_name,
                missing,
            },
            TransferBackendFactoryError::KvCache(error) => {
                Self::KvCacheTransfer(error.to_string())
            }
            TransferBackendFactoryError::Mooncake(error) => Self::MooncakeTransfer(error),
        }
    }
}

impl From<PdConfigError> for ServerLaunchError {
    fn from(value: PdConfigError) -> Self {
        Self::PdConfig(value)
    }
}

impl From<ModelArtifactError> for ServerLaunchError {
    fn from(value: ModelArtifactError) -> Self {
        Self::ModelArtifact(value)
    }
}

impl From<ModelRegistryError> for ServerLaunchError {
    fn from(value: ModelRegistryError) -> Self {
        Self::ModelRegistry(value)
    }
}

impl From<TokenizerError> for ServerLaunchError {
    fn from(value: TokenizerError) -> Self {
        Self::Tokenizer(value)
    }
}

impl From<DpCoordinatorError> for ServerLaunchError {
    fn from(value: DpCoordinatorError) -> Self {
        Self::DpCoordinator(value.to_string())
    }
}

pub fn grpc_listen_addr(args: &ServerArgs) -> Result<SocketAddr, ServerLaunchError> {
    let mut addresses = (args.host.as_str(), args.port)
        .to_socket_addrs()
        .map_err(ServerLaunchError::AddressResolve)?;

    addresses
        .next()
        .ok_or_else(|| ServerLaunchError::NoSocketAddress {
            host: args.host.clone(),
            port: args.port,
        })
}

pub fn http_listen_addr(args: &ServerArgs) -> Result<SocketAddr, ServerLaunchError> {
    grpc_listen_addr(args)
}

pub fn grpc_http_sidecar_listen_addr(args: &ServerArgs) -> Result<SocketAddr, ServerLaunchError> {
    let port = match args.smg_http_sidecar_port {
        Some(port) => port,
        None => args
            .port
            .checked_add(1)
            .ok_or(ServerLaunchError::InvalidGrpcSidecarPort {
                grpc_port: args.port,
            })?,
    };
    let mut addresses = (args.host.as_str(), port)
        .to_socket_addrs()
        .map_err(ServerLaunchError::AddressResolve)?;
    addresses
        .next()
        .ok_or_else(|| ServerLaunchError::NoSocketAddress {
            host: args.host.clone(),
            port,
        })
}

pub fn prefill_bootstrap_listen_addr(args: &ServerArgs) -> Result<SocketAddr, ServerLaunchError> {
    let mut addresses = (args.host.as_str(), args.disaggregation_bootstrap_port)
        .to_socket_addrs()
        .map_err(ServerLaunchError::AddressResolve)?;

    addresses
        .next()
        .ok_or_else(|| ServerLaunchError::NoSocketAddress {
            host: args.host.clone(),
            port: args.disaggregation_bootstrap_port,
        })
}

pub fn prefill_mooncake_zmq_endpoints(args: &ServerArgs) -> Vec<String> {
    let Some(ports) = args.disaggregation_zmq_ports else {
        return Vec::new();
    };

    (ports.start..=ports.end)
        .map(|port| format!("tcp://{}:{port}", args.host))
        .collect()
}

fn http_server_info_from_args(args: &ServerArgs) -> HttpServerInfo {
    let mut server_info = HttpServerInfo {
        tp_size: args.tp_size,
        dp_size: args.dp_size,
        load_balance_method: args.load_balance_method.clone(),
        max_running_requests: args.max_running_requests,
        max_prefill_tokens: args.max_prefill_tokens,
        max_total_tokens: args.max_total_tokens,
        disaggregation_mode: args.disaggregation_mode.clone(),
        disaggregation_bootstrap_port: if args.disaggregation_mode == "prefill" {
            Some(args.disaggregation_bootstrap_port)
        } else {
            None
        },
        enable_metrics: args.enable_metrics,
        kv_events: None,
        kv_cache: None,
    };

    if let Some(ports) = args.disaggregation_zmq_ports
        && let (Ok(block_size), Ok(dp_size)) =
            (u32::try_from(args.page_size), u32::try_from(args.dp_size))
    {
        server_info.kv_events = Some(HttpKvEventsInfo {
            publisher: "zmq".to_string(),
            endpoint_host: prefill_mooncake_route_rank_ip(args),
            endpoint_port_base: ports.start,
            topic: String::new(),
            block_size,
            dp_size,
        });
    }

    server_info
}

fn active_kv_cache_info<M, S>(model_runner: &ModelRunner<M, S>) -> Option<HttpKvCacheInfo>
where
    M: RuntimeKvCacheMetadata,
{
    let runtime = model_runner.active_kv_cache_layout()?.runtime();
    Some(HttpKvCacheInfo {
        dtype: runtime.dtype.as_str().to_string(),
        page_size: u64::try_from(runtime.page_size).ok()?,
        num_layers: u64::try_from(runtime.num_layers).ok()?,
        kv_heads: u64::try_from(runtime.kv_heads).ok()?,
        head_dim: u64::try_from(runtime.head_dim).ok()?,
        kv_tensors_per_token: u64::try_from(runtime.kv_tensors_per_token).ok()?,
        bytes_per_token: u64::try_from(runtime.bytes_per_token).ok()?,
        page_size_bytes: u64::try_from(runtime.page_size_bytes).ok()?,
    })
}

fn http_server_info_from_model_runner<M, S>(
    args: &ServerArgs,
    model_runner: &ModelRunner<M, S>,
) -> HttpServerInfo
where
    M: RuntimeKvCacheMetadata,
{
    let mut server_info = http_server_info_from_args(args);
    server_info.kv_cache = active_kv_cache_info(model_runner);
    server_info
}

fn grpc_server_info_from_model_runner<M, S>(
    args: &ServerArgs,
    model_runner: &ModelRunner<M, S>,
) -> HashMap<String, String>
where
    M: RuntimeKvCacheMetadata,
{
    let mut attributes = server_info_attributes_from_args(args);
    if let Some(kv_cache) = active_kv_cache_info(model_runner) {
        attributes.extend([
            ("kv_cache.dtype".to_string(), kv_cache.dtype),
            (
                "kv_cache.page_size".to_string(),
                kv_cache.page_size.to_string(),
            ),
            (
                "kv_cache.num_layers".to_string(),
                kv_cache.num_layers.to_string(),
            ),
            (
                "kv_cache.kv_heads".to_string(),
                kv_cache.kv_heads.to_string(),
            ),
            (
                "kv_cache.head_dim".to_string(),
                kv_cache.head_dim.to_string(),
            ),
            (
                "kv_cache.kv_tensors_per_token".to_string(),
                kv_cache.kv_tensors_per_token.to_string(),
            ),
            (
                "kv_cache.bytes_per_token".to_string(),
                kv_cache.bytes_per_token.to_string(),
            ),
            (
                "kv_cache.page_size_bytes".to_string(),
                kv_cache.page_size_bytes.to_string(),
            ),
        ]);
    }
    attributes
}

pub fn register_prefill_mooncake_routes_from_args(
    service: &PrefillBootstrapService,
    args: &ServerArgs,
) -> Result<(), ServerLaunchError> {
    let Some(ports) = args.disaggregation_zmq_ports else {
        return Ok(());
    };

    let port_count = usize::from(ports.end - ports.start + 1);
    let expected = args.dp_size * args.tp_size;
    if port_count != expected {
        return Err(ServerLaunchError::ZmqRoutePortCountMismatch {
            expected,
            actual: port_count,
        });
    }

    let system_dp_size = if args.enable_dp_attention {
        1
    } else {
        args.dp_size
    };
    let rank_ip = prefill_mooncake_route_rank_ip(args);
    let mut state = service
        .state()
        .lock()
        .expect("prefill bootstrap state lock should be held");
    for dp_rank in 0..args.dp_size {
        for tp_rank in 0..args.tp_size {
            let port = ports.start + (dp_rank * args.tp_size + tp_rank) as u16;
            state.register_route(PrefillRouteRegistration {
                attn_tp_size: args.tp_size,
                attn_tp_rank: tp_rank,
                attn_cp_size: 1,
                attn_cp_rank: 0,
                attn_dp_size: args.dp_size,
                attn_dp_rank: dp_rank,
                pp_size: 1,
                pp_rank: 0,
                system_dp_size,
                system_dp_rank: if args.enable_dp_attention { 0 } else { dp_rank },
                rank_ip: rank_ip.clone(),
                rank_port: port,
                page_size: Some(args.page_size),
                kv_cache_dtype: Some(args.kv_cache_dtype.clone()),
                load_balance_method: None,
            });
        }
    }
    Ok(())
}

fn prefill_mooncake_route_rank_ip(args: &ServerArgs) -> String {
    if is_wildcard_host(&args.host)
        && let Some(dist_init_host) = args.dist_init_addr.as_deref().and_then(host_from_addr)
    {
        return dist_init_host.to_string();
    }
    args.host.clone()
}

fn is_wildcard_host(host: &str) -> bool {
    matches!(host, "0.0.0.0" | "::" | "[::]")
}

fn host_from_addr(addr: &str) -> Option<&str> {
    if let Some(rest) = addr.strip_prefix('[') {
        let (host, _) = rest.split_once(']')?;
        return Some(host);
    }
    addr.rsplit_once(':')
        .map(|(host, _)| host)
        .filter(|host| !host.is_empty())
}

pub fn build_bootstrap_grpc_router_service(args: &ServerArgs) -> BootstrapGrpcRouterService {
    try_build_bootstrap_grpc_router_service(args).expect("bootstrap tokenizer should load")
}

pub fn try_build_bootstrap_dp_coordinator(
    args: &ServerArgs,
) -> Result<DpCoordinator<RuntimeTokenizer>, ServerLaunchError> {
    let dp_size = args.dp_size;
    let tp_size = args.tp_size.max(1);
    let pp_size = 1;
    let config = Parallel4DConfig::new(pp_size, dp_size, tp_size, dp_size, tp_size)
        .map_err(|e| ServerLaunchError::DpCoordinator(format!("invalid 4D topology: {e}")))?;
    let topology = Parallel4DTopology::new(config).map_err(|e| {
        ServerLaunchError::DpCoordinator(format!("failed to initialize 4D topology: {e}"))
    })?;

    let mut allocators = Vec::with_capacity(dp_size);
    for _ in 0..dp_size {
        allocators.push(cache_page_allocator_from_server_args(args)?);
    }

    let worker_args = args.clone();
    let worker_pool = DpWorkerPool::launch(dp_size, move |_rank| {
        let model = bootstrap_forward_model_from_server_args(&worker_args)?;
        let model_runner = bootstrap_model_runner(model)?;
        Ok::<_, ServerLaunchError>(model_runner)
    })
    .map_err(|e| {
        ServerLaunchError::DpCoordinator(format!("failed to launch DP worker pool: {e}"))
    })?;

    let max_batch_size = args.max_running_requests.unwrap_or(32);
    let scheduler = DpScheduler::new(worker_pool, allocators, max_batch_size, args.page_size)
        .map_err(|e| {
            ServerLaunchError::DpCoordinator(format!("failed to initialize DP scheduler: {e}"))
        })?
        .with_chunked_prefill_size(args.chunked_prefill_size);

    let tokenizer = RuntimeTokenizer::from_model_or_tokenizer_path(
        &args.model_path,
        args.tokenizer_path.as_deref(),
    )?;

    Ok(DpCoordinator::new(tokenizer, scheduler, topology))
}

pub fn try_build_bootstrap_grpc_router_service(
    args: &ServerArgs,
) -> Result<BootstrapGrpcRouterService, ServerLaunchError> {
    validate_local_model_artifacts_if_present(args)?;
    if args.dp_size > 1 {
        let coordinator = try_build_bootstrap_dp_coordinator(args)?;
        let server_info_attributes = server_info_attributes_from_args(args);
        let runtime = RouterRuntime::from_dp_coordinator(coordinator)
            .with_default_stop_token_ids(model_config_eos_token_ids(&args.model_path));
        Ok(GrpcRouterService::with_server_args(runtime, args)
            .with_server_info_attributes(server_info_attributes))
    } else {
        let model = bootstrap_forward_model_from_server_args(args)?;
        try_build_bootstrap_grpc_router_service_from_model(args, model)
    }
}

fn try_build_bootstrap_grpc_router_service_from_model(
    args: &ServerArgs,
    model: BootstrapForwardModel,
) -> Result<BootstrapGrpcRouterService, ServerLaunchError> {
    let prefix_cache_policy = model_prefix_cache_policy(&model);
    let model_runner = bootstrap_model_runner(model)?;
    let server_info_attributes = grpc_server_info_from_model_runner(args, &model_runner);
    let scheduler = Scheduler::with_cache_resources_and_policy(
        model_runner,
        RadixCache::default(),
        cache_page_allocator_from_server_args(args)?,
        prefix_cache_policy,
    )
    .with_max_running_requests(args.max_running_requests)
    .with_chunked_prefill_size(args.chunked_prefill_size);
    let tokenizer = RuntimeTokenizer::from_model_or_tokenizer_path(
        &args.model_path,
        args.tokenizer_path.as_deref(),
    )?;
    let engine = Engine::new(tokenizer, scheduler);
    let runtime = RouterRuntime::new(engine)
        .with_default_stop_token_ids(model_config_eos_token_ids(&args.model_path));
    Ok(GrpcRouterService::with_server_args(runtime, args)
        .with_server_info_attributes(server_info_attributes))
}

pub fn build_bootstrap_http_router_service(args: &ServerArgs) -> BootstrapHttpRouterService {
    try_build_bootstrap_http_router_service(args).expect("bootstrap tokenizer should load")
}

pub fn try_build_bootstrap_http_router_service(
    args: &ServerArgs,
) -> Result<BootstrapHttpRouterService, ServerLaunchError> {
    validate_local_model_artifacts_if_present(args)?;
    if args.dp_size > 1 {
        let coordinator = try_build_bootstrap_dp_coordinator(args)?;
        let server_info = http_server_info_from_args(args);
        let runtime = RouterRuntime::from_dp_coordinator(coordinator)
            .with_default_stop_token_ids(model_config_eos_token_ids(&args.model_path));
        Ok(
            HttpRouterService::new(runtime, RouterGetModelInfoResponse::from_server_args(args))
                .with_server_info(server_info),
        )
    } else {
        let model = bootstrap_forward_model_from_server_args(args)?;
        try_build_bootstrap_http_router_service_from_model(args, model)
    }
}

fn try_build_bootstrap_http_router_service_from_model(
    args: &ServerArgs,
    model: BootstrapForwardModel,
) -> Result<BootstrapHttpRouterService, ServerLaunchError> {
    let prefix_cache_policy = model_prefix_cache_policy(&model);
    let model_runner = bootstrap_model_runner(model)?;
    let server_info = http_server_info_from_model_runner(args, &model_runner);
    let scheduler = Scheduler::with_cache_resources_and_policy(
        model_runner,
        RadixCache::default(),
        cache_page_allocator_from_server_args(args)?,
        prefix_cache_policy,
    )
    .with_max_running_requests(args.max_running_requests)
    .with_chunked_prefill_size(args.chunked_prefill_size);
    let tokenizer = RuntimeTokenizer::from_model_or_tokenizer_path(
        &args.model_path,
        args.tokenizer_path.as_deref(),
    )?;
    let engine = Engine::new(tokenizer, scheduler);
    let runtime = RouterRuntime::new(engine)
        .with_default_stop_token_ids(model_config_eos_token_ids(&args.model_path));
    Ok(
        HttpRouterService::new(runtime, RouterGetModelInfoResponse::from_server_args(args))
            .with_server_info(server_info),
    )
}

#[cfg(any(test, feature = "test-support"))]
pub fn build_bootstrap_prefill_http_router_service(
    args: &ServerArgs,
) -> BootstrapPrefillHttpRouterService {
    try_build_bootstrap_prefill_http_router_service(args).expect("bootstrap tokenizer should load")
}

fn cache_page_allocator_from_server_args(
    args: &ServerArgs,
) -> Result<CachePageAllocator, ServerLaunchError> {
    CachePageAllocator::with_page_size(crate::cli::resolve_max_total_tokens(args), args.page_size)
        .map_err(|error| ServerLaunchError::KvCacheTransfer(error.to_string()))
}

#[cfg(any(test, feature = "test-support"))]
pub fn try_build_bootstrap_prefill_http_router_service(
    args: &ServerArgs,
) -> Result<BootstrapPrefillHttpRouterService, ServerLaunchError> {
    validate_local_model_artifacts_if_present(args)?;
    let model = bootstrap_forward_model_from_server_args(args)?;
    validate_kv_only_transfer_model(&model)?;
    let worker = KvTransferModelWorker::new(
        ModelRunner::new(model),
        DecodeBootstrapRegistry::default(),
        FakeKvCacheTransferExecutor::default(),
    )
    .with_kv_page_size(args.page_size);
    let scheduler = Scheduler::with_cache_resources(
        worker,
        RadixCache::default(),
        cache_page_allocator_from_server_args(args)?,
    )
    .with_max_running_requests(args.max_running_requests)
    .with_chunked_prefill_size(args.chunked_prefill_size);
    let tokenizer = RuntimeTokenizer::from_model_or_tokenizer_path(
        &args.model_path,
        args.tokenizer_path.as_deref(),
    )?;
    let engine = Engine::new(tokenizer, scheduler);
    let runtime = RouterRuntime::new(engine)
        .with_default_stop_token_ids(model_config_eos_token_ids(&args.model_path));
    Ok(
        HttpRouterService::new(runtime, RouterGetModelInfoResponse::from_server_args(args))
            .with_server_info(http_server_info_from_args(args))
            .with_disaggregated_requests(),
    )
}

pub fn build_bootstrap_pd_http_router_service<E>(
    args: &ServerArgs,
    registry: DecodeBootstrapRegistry,
    transfer_executor: E,
) -> BootstrapPdHttpRouterService<E>
where
    E: KvTransferBackend,
{
    try_build_bootstrap_pd_http_router_service(args, registry, transfer_executor)
        .expect("bootstrap tokenizer should load")
}

pub fn try_build_bootstrap_pd_http_router_service<E>(
    args: &ServerArgs,
    registry: DecodeBootstrapRegistry,
    transfer_executor: E,
) -> Result<BootstrapPdHttpRouterService<E>, ServerLaunchError>
where
    E: KvTransferBackend,
{
    try_build_bootstrap_pd_http_router_service_with_decode_publisher(
        args,
        registry,
        transfer_executor,
        crate::transfer::NoopDecodeBootstrapPublisher,
        false,
    )
}

pub fn try_build_bootstrap_pd_http_router_service_with_decode_publisher<E, P>(
    args: &ServerArgs,
    registry: DecodeBootstrapRegistry,
    transfer_executor: E,
    decode_bootstrap_publisher: P,
    decode_side_bootstrap_only: bool,
) -> Result<BootstrapPdHttpRouterService<E, P>, ServerLaunchError>
where
    E: KvTransferBackend,
    P: DecodeBootstrapPublisher,
{
    validate_local_model_artifacts_if_present(args)?;
    let model = bootstrap_forward_model_from_server_args(args)?;
    try_build_bootstrap_pd_http_router_service_from_model_with_decode_publisher(
        args,
        model,
        registry,
        transfer_executor,
        decode_bootstrap_publisher,
        decode_side_bootstrap_only,
    )
}

fn try_build_bootstrap_pd_http_router_service_from_model_with_decode_publisher<E, P>(
    args: &ServerArgs,
    model: BootstrapForwardModel,
    registry: DecodeBootstrapRegistry,
    transfer_executor: E,
    decode_bootstrap_publisher: P,
    decode_side_bootstrap_only: bool,
) -> Result<BootstrapPdHttpRouterService<E, P>, ServerLaunchError>
where
    E: KvTransferBackend,
    P: DecodeBootstrapPublisher,
{
    validate_kv_only_transfer_model(&model)?;
    try_build_bootstrap_pd_http_router_service_from_runner_with_decode_publisher(
        args,
        bootstrap_model_runner(model)?,
        registry,
        transfer_executor,
        decode_bootstrap_publisher,
        decode_side_bootstrap_only,
    )
}

fn try_build_bootstrap_pd_http_router_service_from_runner_with_decode_publisher<E, P>(
    args: &ServerArgs,
    model_runner: ModelRunner<BootstrapForwardModel>,
    registry: DecodeBootstrapRegistry,
    transfer_executor: E,
    decode_bootstrap_publisher: P,
    decode_side_bootstrap_only: bool,
) -> Result<BootstrapPdHttpRouterService<E, P>, ServerLaunchError>
where
    E: KvTransferBackend,
    P: DecodeBootstrapPublisher,
{
    let server_info = http_server_info_from_model_runner(args, &model_runner);
    let mut worker = KvTransferModelWorker::new(model_runner, registry, transfer_executor)
        .with_decode_bootstrap_publisher(decode_bootstrap_publisher)
        .with_kv_page_size(args.page_size);
    if decode_side_bootstrap_only {
        worker = worker.with_decode_side_bootstrap_only();
    } else if args.disaggregation_mode == "prefill" {
        worker = worker.with_prefill_only();
    }
    let scheduler = Scheduler::with_cache_resources_and_policy(
        worker,
        RadixCache::default(),
        cache_page_allocator_from_server_args(args)?,
        PrefixCachePolicy::Disabled,
    )
    .with_max_running_requests(args.max_running_requests)
    .with_chunked_prefill_size(args.chunked_prefill_size);
    let tokenizer = RuntimeTokenizer::from_model_or_tokenizer_path(
        &args.model_path,
        args.tokenizer_path.as_deref(),
    )?;
    let engine = Engine::new(tokenizer, scheduler);
    let runtime = RouterRuntime::new(engine)
        .with_default_stop_token_ids(model_config_eos_token_ids(&args.model_path));
    Ok(
        HttpRouterService::new(runtime, RouterGetModelInfoResponse::from_server_args(args))
            .with_server_info(server_info)
            .with_disaggregated_requests()
            .with_max_transfer_polls(args.disaggregation_decode_polling_interval),
    )
}

fn launch_mooncake_decode_bootstrap_publisher(
    args: &ServerArgs,
    kv_cache_layout: MooncakeKvCacheLayout,
    mooncake_session_id: impl Into<String>,
) -> Result<MooncakeDecodeBootstrapPublisher, ServerLaunchError> {
    let endpoint = prefill_mooncake_route_rank_ip(args);
    let ranks_per_node = args.tp_size.checked_div(args.nnodes.max(1)).unwrap_or(1);
    let target_tp_rank = i32::try_from(args.node_rank * ranks_per_node).unwrap_or(0);
    let dst_port = args
        .disaggregation_zmq_ports
        .map(|ports| {
            let offset = target_tp_rank as u16;
            if ports
                .start
                .checked_add(offset)
                .is_some_and(|p| p <= ports.end)
            {
                ports.start + offset
            } else {
                ports.start
            }
        })
        .unwrap_or(args.port);
    MooncakeDecodeBootstrapPublisher::new(endpoint, dst_port, mooncake_session_id)
        .with_kv_cache_layout(kv_cache_layout)
        .with_target_ranks(target_tp_rank, 0)
        .with_receiver()
        .map_err(ServerLaunchError::KvCacheTransfer)
}

pub fn build_bootstrap_mooncake_prefill_http_router_service<S, R>(
    args: &ServerArgs,
    bootstrap_service: PrefillBootstrapService,
    transfer_executor: MooncakeKvCacheTransferExecutor<S, R>,
) -> BootstrapPdHttpRouterService<
    MooncakeBootstrapKvCacheTransferExecutor<MooncakeKvCacheTransferExecutor<S, R>>,
>
where
    S: MooncakeTransferSubmitter + MooncakeTransferStatusReader + MooncakeBatchReleaser,
    R: MooncakeTransferTargetResolver,
{
    try_build_bootstrap_mooncake_prefill_http_router_service(
        args,
        bootstrap_service,
        transfer_executor,
    )
    .expect("bootstrap tokenizer should load")
}

pub fn try_build_bootstrap_mooncake_prefill_http_router_service<S, R>(
    args: &ServerArgs,
    bootstrap_service: PrefillBootstrapService,
    transfer_executor: MooncakeKvCacheTransferExecutor<S, R>,
) -> Result<
    BootstrapPdHttpRouterService<
        MooncakeBootstrapKvCacheTransferExecutor<MooncakeKvCacheTransferExecutor<S, R>>,
    >,
    ServerLaunchError,
>
where
    S: MooncakeTransferSubmitter + MooncakeTransferStatusReader + MooncakeBatchReleaser,
    R: MooncakeTransferTargetResolver,
{
    try_build_bootstrap_pd_http_router_service(
        args,
        DecodeBootstrapRegistry::default(),
        MooncakeBootstrapKvCacheTransferExecutor::new(bootstrap_service, transfer_executor),
    )
}

pub fn try_build_bootstrap_mooncake_prefill_grpc_router_service<S, R>(
    args: &ServerArgs,
    bootstrap_service: PrefillBootstrapService,
    transfer_executor: MooncakeKvCacheTransferExecutor<S, R>,
) -> Result<
    BootstrapPdGrpcRouterService<
        MooncakeBootstrapKvCacheTransferExecutor<MooncakeKvCacheTransferExecutor<S, R>>,
    >,
    ServerLaunchError,
>
where
    S: MooncakeTransferSubmitter + MooncakeTransferStatusReader + MooncakeBatchReleaser,
    R: MooncakeTransferTargetResolver,
{
    try_build_bootstrap_pd_grpc_router_service(
        args,
        DecodeBootstrapRegistry::default(),
        MooncakeBootstrapKvCacheTransferExecutor::new(bootstrap_service, transfer_executor),
    )
}

struct ProductionPdRuntimeBundle {
    model_runner: ModelRunner<BootstrapForwardModel>,
    transfer_backend: ProductionTransferBackend,
    decode_bootstrap_publisher: ProductionDecodeBootstrapPublisher,
    decode_side_bootstrap_only: bool,
}

fn try_build_launch_pd_runtime_bundle(
    args: &ServerArgs,
    pd_config: &PdConfig,
    prefill_bootstrap_service: Option<PrefillBootstrapService>,
) -> Result<ProductionPdRuntimeBundle, ServerLaunchError> {
    let mut model_runner = bootstrap_model_runner(bootstrap_forward_model_for_launch(args)?)?;
    let mut context = TransferBackendBuildContext::new(
        &mut model_runner,
        pd_config,
        prefill_mooncake_route_rank_ip(args),
        crate::cli::resolve_max_total_tokens(args),
        args.page_size,
    );
    if let Some(service) = prefill_bootstrap_service {
        context = context.with_prefill_bootstrap_service(service);
    }
    let transfer_backend = TransferBackendFactory::build(pd_config.mode, context)?;
    let (decode_bootstrap_publisher, decode_side_bootstrap_only) = match pd_config.mode {
        DisaggregationMode::Prefill => (
            ProductionDecodeBootstrapPublisher::Noop(crate::transfer::NoopDecodeBootstrapPublisher),
            false,
        ),
        DisaggregationMode::Decode => (
            ProductionDecodeBootstrapPublisher::Mooncake(Box::new(
                launch_mooncake_decode_bootstrap_publisher(
                    args,
                    transfer_backend.layout,
                    transfer_backend.local_endpoint,
                )?,
            )),
            true,
        ),
        DisaggregationMode::Null => {
            return Err(ServerLaunchError::UnsupportedBootstrapPdRuntime {
                mode: pd_config.mode,
                transfer_backend: pd_config.transfer_backend,
            });
        }
    };

    Ok(ProductionPdRuntimeBundle {
        model_runner,
        transfer_backend: transfer_backend.backend,
        decode_bootstrap_publisher,
        decode_side_bootstrap_only,
    })
}

fn try_build_launch_pd_http_router_service(
    args: &ServerArgs,
    bundle: ProductionPdRuntimeBundle,
) -> Result<ProductionPdHttpRouterService, ServerLaunchError> {
    try_build_bootstrap_pd_http_router_service_from_runner_with_decode_publisher(
        args,
        bundle.model_runner,
        DecodeBootstrapRegistry::default(),
        bundle.transfer_backend,
        bundle.decode_bootstrap_publisher,
        bundle.decode_side_bootstrap_only,
    )
}

fn try_build_launch_pd_grpc_router_service(
    args: &ServerArgs,
    bundle: ProductionPdRuntimeBundle,
) -> Result<ProductionPdGrpcRouterService, ServerLaunchError> {
    try_build_bootstrap_pd_grpc_router_service_from_runner_with_decode_publisher(
        args,
        bundle.model_runner,
        DecodeBootstrapRegistry::default(),
        bundle.transfer_backend,
        bundle.decode_bootstrap_publisher,
        bundle.decode_side_bootstrap_only,
    )
}

pub fn build_bootstrap_pd_grpc_router_service<E>(
    args: &ServerArgs,
    registry: DecodeBootstrapRegistry,
    transfer_executor: E,
) -> BootstrapPdGrpcRouterService<E>
where
    E: KvTransferBackend,
{
    try_build_bootstrap_pd_grpc_router_service(args, registry, transfer_executor)
        .expect("bootstrap tokenizer should load")
}

pub fn try_build_bootstrap_pd_grpc_router_service<E>(
    args: &ServerArgs,
    registry: DecodeBootstrapRegistry,
    transfer_executor: E,
) -> Result<BootstrapPdGrpcRouterService<E>, ServerLaunchError>
where
    E: KvTransferBackend,
{
    try_build_bootstrap_pd_grpc_router_service_with_decode_publisher(
        args,
        registry,
        transfer_executor,
        crate::transfer::NoopDecodeBootstrapPublisher,
        false,
    )
}

pub fn try_build_bootstrap_pd_grpc_router_service_with_decode_publisher<E, P>(
    args: &ServerArgs,
    registry: DecodeBootstrapRegistry,
    transfer_executor: E,
    decode_bootstrap_publisher: P,
    decode_side_bootstrap_only: bool,
) -> Result<BootstrapPdGrpcRouterService<E, P>, ServerLaunchError>
where
    E: KvTransferBackend,
    P: DecodeBootstrapPublisher,
{
    validate_local_model_artifacts_if_present(args)?;
    let model = bootstrap_forward_model_from_server_args(args)?;
    try_build_bootstrap_pd_grpc_router_service_from_model_with_decode_publisher(
        args,
        model,
        registry,
        transfer_executor,
        decode_bootstrap_publisher,
        decode_side_bootstrap_only,
    )
}

fn try_build_bootstrap_pd_grpc_router_service_from_model_with_decode_publisher<E, P>(
    args: &ServerArgs,
    model: BootstrapForwardModel,
    registry: DecodeBootstrapRegistry,
    transfer_executor: E,
    decode_bootstrap_publisher: P,
    decode_side_bootstrap_only: bool,
) -> Result<BootstrapPdGrpcRouterService<E, P>, ServerLaunchError>
where
    E: KvTransferBackend,
    P: DecodeBootstrapPublisher,
{
    validate_kv_only_transfer_model(&model)?;
    try_build_bootstrap_pd_grpc_router_service_from_runner_with_decode_publisher(
        args,
        bootstrap_model_runner(model)?,
        registry,
        transfer_executor,
        decode_bootstrap_publisher,
        decode_side_bootstrap_only,
    )
}

fn try_build_bootstrap_pd_grpc_router_service_from_runner_with_decode_publisher<E, P>(
    args: &ServerArgs,
    model_runner: ModelRunner<BootstrapForwardModel>,
    registry: DecodeBootstrapRegistry,
    transfer_executor: E,
    decode_bootstrap_publisher: P,
    decode_side_bootstrap_only: bool,
) -> Result<BootstrapPdGrpcRouterService<E, P>, ServerLaunchError>
where
    E: KvTransferBackend,
    P: DecodeBootstrapPublisher,
{
    let server_info_attributes = grpc_server_info_from_model_runner(args, &model_runner);
    let mut worker = KvTransferModelWorker::new(model_runner, registry, transfer_executor)
        .with_decode_bootstrap_publisher(decode_bootstrap_publisher)
        .with_kv_page_size(args.page_size);
    if decode_side_bootstrap_only {
        worker = worker.with_decode_side_bootstrap_only();
    } else if args.disaggregation_mode == "prefill" {
        worker = worker.with_prefill_only();
    }
    let scheduler = Scheduler::with_cache_resources_and_policy(
        worker,
        RadixCache::default(),
        cache_page_allocator_from_server_args(args)?,
        PrefixCachePolicy::Disabled,
    )
    .with_max_running_requests(args.max_running_requests)
    .with_chunked_prefill_size(args.chunked_prefill_size);
    let tokenizer = RuntimeTokenizer::from_model_or_tokenizer_path(
        &args.model_path,
        args.tokenizer_path.as_deref(),
    )?;
    let engine = Engine::new(tokenizer, scheduler);
    let runtime = RouterRuntime::new(engine)
        .with_default_stop_token_ids(model_config_eos_token_ids(&args.model_path));
    Ok(GrpcRouterService::with_server_args(runtime, args)
        .with_server_info_attributes(server_info_attributes)
        .with_max_transfer_polls(args.disaggregation_decode_polling_interval))
}

fn model_config_eos_token_ids(model_path: &str) -> Vec<u32> {
    HfModelConfig::from_model_path(model_path)
        .map(|config| config.eos_token_ids)
        .unwrap_or_default()
}

fn validate_local_model_artifacts_if_present(args: &ServerArgs) -> Result<(), ServerLaunchError> {
    let model_path = Path::new(&args.model_path);
    if !model_path.is_dir() || !model_path.join("config.json").is_file() {
        return Ok(());
    }

    let artifacts = match LocalModelArtifacts::from_model_path(model_path) {
        Ok(artifacts) => artifacts,
        Err(ModelArtifactError::NoSafetensorsWeights { .. }) => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    ModelRegistry.resolve(artifacts.model_path(), artifacts.config())?;
    Ok(())
}

#[cfg(any(test, feature = "test-support"))]
pub fn build_bootstrap_fake_pd_grpc_router_service(
    args: &ServerArgs,
) -> BootstrapFakePdGrpcRouterService {
    build_bootstrap_pd_grpc_router_service(
        args,
        DecodeBootstrapRegistry::default(),
        FakeKvCacheTransferExecutor::default(),
    )
}

fn validate_launch_transfer_backend(pd_config: &PdConfig) -> Result<(), ServerLaunchError> {
    if pd_config.mode == DisaggregationMode::Null {
        return Ok(());
    }
    if pd_config.nnodes == 0 {
        return Err(ServerLaunchError::KvCacheTransfer(
            "P/D disaggregation requires nnodes >= 1".into(),
        ));
    }
    if pd_config.node_rank >= pd_config.nnodes {
        return Err(ServerLaunchError::KvCacheTransfer(format!(
            "P/D node_rank {} must be less than nnodes {}",
            pd_config.node_rank, pd_config.nnodes
        )));
    }
    if pd_config.tp_size == 0 || pd_config.dp_size == 0 {
        return Err(ServerLaunchError::KvCacheTransfer(
            "P/D disaggregation requires tp_size >= 1 and dp_size >= 1".into(),
        ));
    }
    if !pd_config.tp_size.is_multiple_of(pd_config.nnodes) {
        return Err(ServerLaunchError::KvCacheTransfer(format!(
            "P/D tensor parallel size (tp_size={}) must be divisible by node count (nnodes={})",
            pd_config.tp_size, pd_config.nnodes
        )));
    }
    if pd_config.decode_enable_radix_cache || pd_config.decode_enable_offload_kvcache {
        return Err(ServerLaunchError::KvCacheTransfer(
            "P/D prefix reuse and decode KV offload are not implemented; disable --disaggregation-decode-enable-radix-cache and --disaggregation-decode-enable-offload-kvcache".into(),
        ));
    }

    match pd_config.transfer_backend {
        TransferBackend::Mooncake if cfg!(feature = "mooncake-link") => Ok(()),
        TransferBackend::Mooncake => Err(MooncakeError::UnavailableWithoutLink.into()),
        TransferBackend::Tcp => Ok(()),
        backend @ (TransferBackend::Nixl | TransferBackend::Ascend | TransferBackend::Mori) => {
            Err(ServerLaunchError::UnsupportedBootstrapPdRuntime {
                mode: pd_config.mode,
                transfer_backend: backend,
            })
        }
    }
}

fn try_build_launch_grpc_router_service(
    args: &ServerArgs,
) -> Result<BootstrapGrpcRouterService, ServerLaunchError> {
    validate_local_model_artifacts_if_present(args)?;
    let model = bootstrap_forward_model_for_launch(args)?;
    try_build_bootstrap_grpc_router_service_from_model(args, model)
}

fn try_build_launch_http_router_service(
    args: &ServerArgs,
) -> Result<BootstrapHttpRouterService, ServerLaunchError> {
    validate_local_model_artifacts_if_present(args)?;
    let model = bootstrap_forward_model_for_launch(args)?;
    try_build_bootstrap_http_router_service_from_model(args, model)
}

pub async fn launch_grpc_server(args: ServerArgs) -> Result<(), ServerLaunchError> {
    launch_grpc_server_with_shutdown(args, process_shutdown_signal()?).await
}

pub async fn launch_grpc_server_with_shutdown<F>(
    args: ServerArgs,
    shutdown: F,
) -> Result<(), ServerLaunchError>
where
    F: Future<Output = ()> + Send + 'static,
{
    let pd_config = PdConfig::from_server_args(&args)?;
    validate_launch_transfer_backend(&pd_config)?;
    let addr = grpc_listen_addr(&args)?;
    let sidecar_addr = grpc_http_sidecar_listen_addr(&args)?;
    match pd_config.mode {
        DisaggregationMode::Null => {
            let service = try_build_launch_grpc_router_service(&args)?;
            serve_grpc_and_sidecar(addr, service, sidecar_addr, args.enable_metrics, shutdown)
                .await?;
        }
        DisaggregationMode::Prefill
            if pd_config.transfer_backend == TransferBackend::Mooncake
                || pd_config.transfer_backend == TransferBackend::Tcp =>
        {
            let bootstrap_addr = prefill_bootstrap_listen_addr(&args)?;
            let zmq_endpoints = prefill_mooncake_zmq_endpoints(&args);
            let bootstrap_service = PrefillBootstrapService::default();
            register_prefill_mooncake_routes_from_args(&bootstrap_service, &args)?;
            let runtime_bundle = try_build_launch_pd_runtime_bundle(
                &args,
                &pd_config,
                Some(bootstrap_service.clone()),
            )?;
            let service = try_build_launch_pd_grpc_router_service(&args, runtime_bundle)?;
            serve_prefill_grpc_and_bootstrap(
                addr,
                service,
                PrefillGrpcBootstrapServices {
                    sidecar_addr,
                    enable_metrics: args.enable_metrics,
                    prefill_addr: bootstrap_addr,
                    prefill_service: bootstrap_service,
                    zmq_endpoints,
                },
                shutdown,
            )
            .await?;
        }
        DisaggregationMode::Decode
            if pd_config.transfer_backend == TransferBackend::Mooncake
                || pd_config.transfer_backend == TransferBackend::Tcp =>
        {
            let runtime_bundle = try_build_launch_pd_runtime_bundle(&args, &pd_config, None)?;
            let service = try_build_launch_pd_grpc_router_service(&args, runtime_bundle)?;
            serve_grpc_and_sidecar(addr, service, sidecar_addr, args.enable_metrics, shutdown)
                .await?;
        }
        _ => {
            return Err(ServerLaunchError::UnsupportedBootstrapPdRuntime {
                mode: pd_config.mode,
                transfer_backend: pd_config.transfer_backend,
            });
        }
    }
    Ok(())
}

pub async fn launch_http_server(args: ServerArgs) -> Result<(), ServerLaunchError> {
    launch_http_server_with_shutdown(args, process_shutdown_signal()?).await
}

fn process_shutdown_signal() -> Result<impl Future<Output = ()> + Send, ServerLaunchError> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut interrupt =
            signal(SignalKind::interrupt()).map_err(ServerLaunchError::ShutdownSignal)?;
        let mut terminate =
            signal(SignalKind::terminate()).map_err(ServerLaunchError::ShutdownSignal)?;
        Ok(async move {
            tokio::select! {
                _ = interrupt.recv() => {},
                _ = terminate.recv() => {},
            }
        })
    }
    #[cfg(not(unix))]
    {
        let mut interrupt =
            tokio::signal::windows::ctrl_c().map_err(ServerLaunchError::ShutdownSignal)?;
        Ok(async move {
            interrupt.recv().await;
        })
    }
}

pub async fn launch_http_server_with_shutdown<F>(
    args: ServerArgs,
    shutdown: F,
) -> Result<(), ServerLaunchError>
where
    F: Future<Output = ()> + Send + 'static,
{
    let pd_config = PdConfig::from_server_args(&args)?;
    validate_launch_transfer_backend(&pd_config)?;
    let addr = http_listen_addr(&args)?;
    let engine_info_service = EngineInfoBootstrapService::default();
    match pd_config.mode {
        DisaggregationMode::Null => {
            let service = try_build_launch_http_router_service(&args)?
                .with_engine_info_bootstrap_service(engine_info_service);
            serve_http_router_with_shutdown(addr, service, shutdown).await?;
        }
        DisaggregationMode::Prefill
            if pd_config.transfer_backend == TransferBackend::Mooncake
                || pd_config.transfer_backend == TransferBackend::Tcp =>
        {
            let bootstrap_addr = prefill_bootstrap_listen_addr(&args)?;
            let zmq_endpoints = prefill_mooncake_zmq_endpoints(&args);
            let bootstrap_service = PrefillBootstrapService::default();
            register_prefill_mooncake_routes_from_args(&bootstrap_service, &args)?;
            let runtime_bundle = try_build_launch_pd_runtime_bundle(
                &args,
                &pd_config,
                Some(bootstrap_service.clone()),
            )?;
            let service = try_build_launch_pd_http_router_service(&args, runtime_bundle)?
                .with_engine_info_bootstrap_service(engine_info_service);
            serve_prefill_http_and_bootstrap(
                addr,
                service,
                PrefillHttpBootstrapServices {
                    prefill_addr: bootstrap_addr,
                    prefill_service: bootstrap_service,
                    zmq_endpoints,
                },
                shutdown,
            )
            .await?;
        }
        DisaggregationMode::Decode
            if pd_config.transfer_backend == TransferBackend::Mooncake
                || pd_config.transfer_backend == TransferBackend::Tcp =>
        {
            let runtime_bundle = try_build_launch_pd_runtime_bundle(&args, &pd_config, None)?;
            let service = try_build_launch_pd_http_router_service(&args, runtime_bundle)?
                .with_engine_info_bootstrap_service(engine_info_service);
            serve_http_router_with_shutdown(addr, service, shutdown).await?;
        }
        _ => {
            return Err(ServerLaunchError::UnsupportedBootstrapPdRuntime {
                mode: pd_config.mode,
                transfer_backend: pd_config.transfer_backend,
            });
        }
    }
    Ok(())
}

struct PrefillHttpBootstrapServices {
    prefill_addr: SocketAddr,
    prefill_service: PrefillBootstrapService,
    zmq_endpoints: Vec<String>,
}

async fn serve_prefill_http_and_bootstrap<T, W, F>(
    http_addr: SocketAddr,
    http_service: HttpRouterService<T, W>,
    bootstrap: PrefillHttpBootstrapServices,
    shutdown: F,
) -> Result<(), ServerLaunchError>
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
    F: Future<Output = ()> + Send + 'static,
{
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let (bootstrap_shutdown_tx, bootstrap_shutdown_rx) = tokio::sync::watch::channel(false);
    let mut http_task = tokio::spawn(serve_http_router_with_shutdown(
        http_addr,
        http_service,
        watch_shutdown(shutdown_rx.clone()),
    ));
    let mut bootstrap_tasks = tokio::task::JoinSet::new();
    let prefill_shutdown_rx = bootstrap_shutdown_rx.clone();
    let prefill_bootstrap_service = bootstrap.prefill_service.clone();
    bootstrap_tasks.spawn(async move {
        serve_prefill_bootstrap_with_shutdown(
            bootstrap.prefill_addr,
            prefill_bootstrap_service,
            watch_shutdown(prefill_shutdown_rx),
        )
        .await
        .map_err(ServerLaunchError::from)
    });
    if !bootstrap.zmq_endpoints.is_empty() {
        bootstrap_tasks.spawn(async move {
            serve_mooncake_bootstrap_zmq_endpoints_with_shutdown(
                bootstrap.zmq_endpoints,
                bootstrap.prefill_service,
                watch_shutdown(bootstrap_shutdown_rx),
            )
            .await
            .map_err(ServerLaunchError::from)
        });
    }

    tokio::select! {
        _ = shutdown => {
            let _ = shutdown_tx.send(true);
            // Keep bootstrap/ABORT handling alive until request DMA is drained.
            let http_result = join_http_task(http_task).await;
            let _ = bootstrap_shutdown_tx.send(true);
            join_launch_tasks(bootstrap_tasks).await?;
            http_result?;
            Ok(())
        }
        result = &mut http_task => {
            let _ = shutdown_tx.send(true);
            let _ = bootstrap_shutdown_tx.send(true);
            join_launch_tasks(bootstrap_tasks).await?;
            result.map_err(|error| ServerLaunchError::ServerTaskJoin(error.to_string()))??;
            Ok(())
        }
        result = bootstrap_tasks.join_next() => {
            let _ = shutdown_tx.send(true);
            let _ = bootstrap_shutdown_tx.send(true);
            join_http_task(http_task).await?;
            match result {
                Some(Ok(Ok(()))) => {
                    join_launch_tasks(bootstrap_tasks).await?;
                    Ok(())
                }
                Some(Ok(Err(error))) => Err(error),
                Some(Err(error)) => Err(ServerLaunchError::ServerTaskJoin(error.to_string())),
                None => Ok(()),
            }
        }
    }
}

async fn serve_grpc_and_sidecar<T, W, F>(
    grpc_addr: SocketAddr,
    grpc_service: GrpcRouterService<T, W>,
    sidecar_addr: SocketAddr,
    enable_metrics: bool,
    shutdown: F,
) -> Result<(), ServerLaunchError>
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
    F: Future<Output = ()> + Send + 'static,
{
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let sidecar_service = grpc_service.clone();
    let mut grpc_task = tokio::spawn(serve_grpc_router_with_shutdown(
        grpc_addr,
        grpc_service,
        true,
        watch_shutdown(shutdown_rx.clone()),
    ));
    let mut sidecar_task = tokio::spawn(serve_grpc_http_sidecar_with_shutdown(
        sidecar_addr,
        sidecar_service,
        enable_metrics,
        watch_shutdown(shutdown_rx),
    ));

    tokio::select! {
        _ = shutdown => {
            let _ = shutdown_tx.send(true);
            join_grpc_task(grpc_task).await?;
            join_http_task(sidecar_task).await?;
            Ok(())
        }
        result = &mut grpc_task => {
            let _ = shutdown_tx.send(true);
            join_http_task(sidecar_task).await?;
            result.map_err(|error| ServerLaunchError::ServerTaskJoin(error.to_string()))??;
            Ok(())
        }
        result = &mut sidecar_task => {
            let _ = shutdown_tx.send(true);
            join_grpc_task(grpc_task).await?;
            result.map_err(|error| ServerLaunchError::ServerTaskJoin(error.to_string()))??;
            Ok(())
        }
    }
}

struct PrefillGrpcBootstrapServices {
    sidecar_addr: SocketAddr,
    enable_metrics: bool,
    prefill_addr: SocketAddr,
    prefill_service: PrefillBootstrapService,
    zmq_endpoints: Vec<String>,
}

async fn serve_prefill_grpc_and_bootstrap<T, W, F>(
    grpc_addr: SocketAddr,
    grpc_service: GrpcRouterService<T, W>,
    bootstrap: PrefillGrpcBootstrapServices,
    shutdown: F,
) -> Result<(), ServerLaunchError>
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
    F: Future<Output = ()> + Send + 'static,
{
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let (bootstrap_shutdown_tx, bootstrap_shutdown_rx) = tokio::sync::watch::channel(false);
    let sidecar_service = grpc_service.clone();
    let mut grpc_task = tokio::spawn(serve_grpc_router_with_shutdown(
        grpc_addr,
        grpc_service,
        true,
        watch_shutdown(shutdown_rx.clone()),
    ));
    let mut sidecar_task = tokio::spawn(serve_grpc_http_sidecar_with_shutdown(
        bootstrap.sidecar_addr,
        sidecar_service,
        bootstrap.enable_metrics,
        watch_shutdown(shutdown_rx.clone()),
    ));
    let mut bootstrap_tasks = tokio::task::JoinSet::new();
    bootstrap_tasks.spawn(serve_prefill_bootstrap_with_shutdown(
        bootstrap.prefill_addr,
        bootstrap.prefill_service.clone(),
        watch_shutdown(bootstrap_shutdown_rx.clone()),
    ));
    if !bootstrap.zmq_endpoints.is_empty() {
        bootstrap_tasks.spawn(serve_mooncake_bootstrap_zmq_endpoints_with_shutdown(
            bootstrap.zmq_endpoints,
            bootstrap.prefill_service,
            watch_shutdown(bootstrap_shutdown_rx),
        ));
    }

    tokio::select! {
        _ = shutdown => {
            let _ = shutdown_tx.send(true);
            let grpc_result = join_grpc_task(grpc_task).await;
            let _ = bootstrap_shutdown_tx.send(true);
            join_http_task(sidecar_task).await?;
            join_bootstrap_tasks(bootstrap_tasks).await?;
            grpc_result?;
            Ok(())
        }
        result = &mut grpc_task => {
            let _ = shutdown_tx.send(true);
            let _ = bootstrap_shutdown_tx.send(true);
            join_http_task(sidecar_task).await?;
            join_bootstrap_tasks(bootstrap_tasks).await?;
            result.map_err(|error| ServerLaunchError::ServerTaskJoin(error.to_string()))??;
            Ok(())
        }
        result = &mut sidecar_task => {
            let _ = shutdown_tx.send(true);
            let grpc_result = join_grpc_task(grpc_task).await;
            let _ = bootstrap_shutdown_tx.send(true);
            join_bootstrap_tasks(bootstrap_tasks).await?;
            grpc_result?;
            result.map_err(|error| ServerLaunchError::ServerTaskJoin(error.to_string()))??;
            Ok(())
        }
        result = bootstrap_tasks.join_next() => {
            let _ = shutdown_tx.send(true);
            let _ = bootstrap_shutdown_tx.send(true);
            join_grpc_task(grpc_task).await?;
            join_http_task(sidecar_task).await?;
            match result {
                Some(Ok(Ok(()))) => {
                    join_bootstrap_tasks(bootstrap_tasks).await?;
                    Ok(())
                }
                Some(Ok(Err(error))) => Err(error.into()),
                Some(Err(error)) => Err(ServerLaunchError::ServerTaskJoin(error.to_string())),
                None => Ok(()),
            }
        }
    }
}

async fn join_http_task(
    task: tokio::task::JoinHandle<Result<(), HttpServeError>>,
) -> Result<(), ServerLaunchError> {
    task.await
        .map_err(|error| ServerLaunchError::ServerTaskJoin(error.to_string()))??;
    Ok(())
}

async fn join_grpc_task(
    task: tokio::task::JoinHandle<Result<(), GrpcServeError>>,
) -> Result<(), ServerLaunchError> {
    task.await
        .map_err(|error| ServerLaunchError::ServerTaskJoin(error.to_string()))??;
    Ok(())
}

async fn join_launch_tasks(
    mut tasks: tokio::task::JoinSet<Result<(), ServerLaunchError>>,
) -> Result<(), ServerLaunchError> {
    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => return Err(error),
            Err(error) => return Err(ServerLaunchError::ServerTaskJoin(error.to_string())),
        }
    }
    Ok(())
}

async fn join_bootstrap_tasks(
    mut tasks: tokio::task::JoinSet<Result<(), PrefillBootstrapServeError>>,
) -> Result<(), ServerLaunchError> {
    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => return Err(error.into()),
            Err(error) => return Err(ServerLaunchError::ServerTaskJoin(error.to_string())),
        }
    }
    Ok(())
}

async fn watch_shutdown(mut shutdown: tokio::sync::watch::Receiver<bool>) {
    while !*shutdown.borrow() {
        if shutdown.changed().await.is_err() {
            break;
        }
    }
}

#[cfg(test)]
mod pd_shutdown_tests {
    use super::*;
    use crate::mooncake_control::send_frames;
    use crate::scheduler::ScheduleBatch;
    use crate::tokenizer::ByteTokenizer;
    use crate::worker::{BatchGeneratedTokens, FallibleModelWorker, WorkerExecutionError};
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use zeromq::{PullSocket, Socket, SocketRecv};

    struct ShutdownGate {
        entered: Option<tokio::sync::oneshot::Sender<()>>,
        release: std::sync::mpsc::Receiver<()>,
    }

    impl FallibleModelWorker for ShutdownGate {
        fn try_generate_batch(
            &mut self,
            _: &ScheduleBatch,
        ) -> Result<BatchGeneratedTokens, WorkerExecutionError> {
            unreachable!("this test sends only bootstrap control messages")
        }

        fn shutdown(&mut self) -> Result<(), WorkerExecutionError> {
            self.entered
                .take()
                .expect("shutdown once")
                .send(())
                .expect("observer");
            self.release
                .recv_timeout(Duration::from_secs(10))
                .map_err(|e| WorkerExecutionError::Runtime(e.to_string()))
        }
    }

    fn address() -> SocketAddr {
        std::net::TcpListener::bind("127.0.0.1:0")
            .expect("port")
            .local_addr()
            .expect("address")
    }

    async fn bootstrap_health(addr: SocketAddr) {
        for _ in 0..100 {
            if let Ok(mut socket) = tokio::net::TcpStream::connect(addr).await {
                socket
                    .write_all(
                        b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
                    )
                    .await
                    .expect("health request");
                let mut response = String::new();
                socket
                    .read_to_string(&mut response)
                    .await
                    .expect("health response");
                assert!(response.starts_with("HTTP/1.1 200"), "{response}");
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("bootstrap unavailable at {addr}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn http_and_grpc_keep_abort_control_alive_until_runtime_shutdown_finishes() {
        for grpc in [false, true] {
            let service_addr = address();
            let bootstrap_addr = address();
            let zmq_addr = address();
            let (entered, draining) = tokio::sync::oneshot::channel();
            let (release, released) = std::sync::mpsc::channel();
            let worker = ShutdownGate {
                entered: Some(entered),
                release: released,
            };
            let runtime = RouterRuntime::new(Engine::new(ByteTokenizer, Scheduler::new(worker)));
            let args =
                ServerArgs::parse_from(["serve", "--model-path", "unused-control-only-test"])
                    .expect("args");
            let bootstrap = PrefillBootstrapService::default();
            let endpoints = vec![format!("tcp://{zmq_addr}")];
            let (stop, stopped) = tokio::sync::oneshot::channel();
            let server = tokio::spawn(async move {
                let shutdown = async move {
                    let _ = stopped.await;
                };
                if grpc {
                    serve_prefill_grpc_and_bootstrap(
                        service_addr,
                        GrpcRouterService::with_server_args(runtime, &args),
                        PrefillGrpcBootstrapServices {
                            sidecar_addr: address(),
                            enable_metrics: false,
                            prefill_addr: bootstrap_addr,
                            prefill_service: bootstrap,
                            zmq_endpoints: endpoints,
                        },
                        shutdown,
                    )
                    .await
                } else {
                    serve_prefill_http_and_bootstrap(
                        service_addr,
                        HttpRouterService::new(
                            runtime,
                            RouterGetModelInfoResponse::from_server_args(&args),
                        ),
                        PrefillHttpBootstrapServices {
                            prefill_addr: bootstrap_addr,
                            prefill_service: bootstrap,
                            zmq_endpoints: endpoints,
                        },
                        shutdown,
                    )
                    .await
                }
            });
            bootstrap_health(bootstrap_addr).await;
            stop.send(()).expect("start graceful shutdown");
            tokio::time::timeout(Duration::from_secs(5), draining)
                .await
                .expect("runtime drain starts")
                .expect("worker alive");
            bootstrap_health(bootstrap_addr).await;
            let mut acknowledgements = PullSocket::new();
            let endpoint = acknowledgements
                .bind("tcp://127.0.0.1:0")
                .await
                .expect("ACK receiver")
                .to_string();
            let port = endpoint.rsplit(':').next().expect("port");
            send_frames(
                format!("tcp://{zmq_addr}"),
                vec![vec![
                    b"ABORT".to_vec(),
                    b"17".to_vec(),
                    b"127.0.0.1".to_vec(),
                    port.as_bytes().to_vec(),
                ]],
            )
            .expect("ABORT during shutdown");
            let ack = tokio::time::timeout(Duration::from_secs(5), acknowledgements.recv())
                .await
                .expect("control still serves ACK")
                .expect("ACK")
                .into_vec();
            assert_eq!(ack[0].as_ref(), b"ABORT_ACK");
            assert!(!server.is_finished());
            release.send(()).expect("allow runtime teardown");
            server
                .await
                .expect("server joins")
                .expect("shutdown completes");
        }
    }
}
