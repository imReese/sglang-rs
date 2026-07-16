use std::fmt;
use std::future::Future;
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::Path;

#[cfg(feature = "mooncake-link")]
use crate::backend::RuntimeRequirements;
use crate::backend::{RuntimeBackend, validate_runtime_backend};
use crate::cache::{CachePageAllocator, RadixCache};
use crate::cli::ServerArgs;
use crate::engine::Engine;
use crate::engine_info_bootstrap::{
    EngineInfoBootstrapServeError, EngineInfoBootstrapService,
    serve_engine_info_bootstrap_with_shutdown,
};
use crate::grpc::{GrpcRouterService, GrpcServeError, serve_grpc_router_with_shutdown};
use crate::grpc_sidecar::serve_grpc_http_sidecar_with_shutdown;
use crate::http::{
    HttpKvCacheInfo, HttpKvEventsInfo, HttpRouterService, HttpServeError, HttpServerInfo,
    serve_http_router_with_shutdown,
};
use crate::model_artifacts::{HfModelConfig, LocalModelArtifacts, ModelArtifactError};
use crate::model_executor::ModelRunner;
pub use crate::model_registry::BootstrapForwardModel;
use crate::model_registry::{ModelRegistry, ModelRegistryError};
use crate::pd_bootstrap::{
    MooncakeBootstrapKvCacheTransferExecutor, MooncakeDecodeBootstrapPublisher,
    PrefillBootstrapServeError, PrefillBootstrapService, PrefillRouteRegistration,
    serve_mooncake_bootstrap_zmq_endpoints_with_shutdown, serve_prefill_bootstrap_with_shutdown,
};
use crate::router::{RouterGetModelInfoResponse, RouterRuntime};
use crate::scheduler::{PrefixCachePolicy, Scheduler};
use crate::tokenizer::{RuntimeTokenizer, Tokenizer, TokenizerError};
#[cfg(not(feature = "mooncake-link"))]
use crate::transfer::MooncakeTransferTarget;
#[cfg(not(feature = "mooncake-link"))]
use crate::transfer::UnlinkedMooncakeTransferEngine;
use crate::transfer::{
    DecodeBootstrapPublisher, DecodeBootstrapRegistry, DisaggregationMode,
    FakeKvCacheTransferExecutor, KvCacheTransferExecutor, KvTransferModelWorker,
    MooncakeBatchReleaser, MooncakeError, MooncakeKvCacheLayout, MooncakeKvCacheTransferExecutor,
    MooncakeTransferStatusReader, MooncakeTransferSubmitter, MooncakeTransferTargetResolver,
    PdConfig, PdConfigError, TransferBackend, TransferableKvCacheMemory,
};
#[cfg(feature = "mooncake-link")]
use crate::transfer::{
    MooncakeSessionTargetResolver, MooncakeTransferEngineConfig, RegisteredMooncakeKvCacheMemory,
    SharedLinkedMooncakeTransferEngine,
};
use crate::worker::WorkerExecutor;

#[doc(hidden)]
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
            .with_max_running_requests(args.max_running_requests);
        let tokenizer = reference_tokenizer_from_args(args)?;
        let runtime = RouterRuntime::new(Engine::new(tokenizer, scheduler))
            .with_default_stop_token_ids(model_config_eos_token_ids(&args.model_path));
        Ok(GrpcRouterService::with_server_args(runtime, args))
    }

    pub fn build_reference_http_router_service(args: &ServerArgs) -> ReferenceHttpRouterService {
        let scheduler = Scheduler::new(ModelRunner::new(CpuReferenceModel))
            .with_max_running_requests(args.max_running_requests);
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
        .with_max_running_requests(args.max_running_requests);
        Ok(reference_http_service(args, scheduler).with_disaggregated_requests())
    }

    pub fn build_reference_pd_http_router_service<E>(
        args: &ServerArgs,
        registry: DecodeBootstrapRegistry,
        transfer_executor: E,
    ) -> ReferencePdHttpRouterService<E>
    where
        E: KvCacheTransferExecutor,
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
        .with_max_running_requests(args.max_running_requests);
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
        E: KvCacheTransferExecutor,
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
        .with_max_running_requests(args.max_running_requests);
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

fn bootstrap_model_runner(model: BootstrapForwardModel) -> ModelRunner<BootstrapForwardModel> {
    let kv_cache_layout = model.kv_cache_layout();
    ModelRunner::new_with_kv_cache_layout(model, kv_cache_layout)
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
pub type BootstrapFakePdGrpcRouterService =
    BootstrapPdGrpcRouterService<FakeKvCacheTransferExecutor>;

#[derive(Debug)]
pub enum ServerLaunchError {
    AddressResolve(std::io::Error),
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
    EngineInfoBootstrap(EngineInfoBootstrapServeError),
    PrefillBootstrap(PrefillBootstrapServeError),
    MooncakeTransfer(MooncakeError),
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
            Self::EngineInfoBootstrap(error) => write!(formatter, "{error}"),
            Self::PrefillBootstrap(error) => write!(formatter, "{error}"),
            Self::MooncakeTransfer(error) => write!(formatter, "{error}"),
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

impl From<EngineInfoBootstrapServeError> for ServerLaunchError {
    fn from(value: EngineInfoBootstrapServeError) -> Self {
        Self::EngineInfoBootstrap(value)
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

pub fn engine_info_bootstrap_listen_addr(
    args: &ServerArgs,
) -> Result<SocketAddr, ServerLaunchError> {
    let mut addresses = (args.host.as_str(), args.engine_info_bootstrap_port)
        .to_socket_addrs()
        .map_err(ServerLaunchError::AddressResolve)?;

    addresses
        .next()
        .ok_or_else(|| ServerLaunchError::NoSocketAddress {
            host: args.host.clone(),
            port: args.engine_info_bootstrap_port,
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

    server_info.kv_cache = http_kv_cache_info_from_args(args);
    server_info
}

fn http_kv_cache_info_from_args(args: &ServerArgs) -> Option<HttpKvCacheInfo> {
    let pd_config = PdConfig::from_server_args(args).ok()?;
    let layout = pd_config.kv_cache_runtime_layout().ok()??;

    Some(HttpKvCacheInfo {
        dtype: layout.dtype.as_str().to_string(),
        page_size: u64::try_from(layout.page_size).ok()?,
        num_layers: u64::try_from(layout.num_layers).ok()?,
        kv_heads: u64::try_from(layout.kv_heads).ok()?,
        head_dim: u64::try_from(layout.head_dim).ok()?,
        kv_tensors_per_token: u64::try_from(layout.kv_tensors_per_token).ok()?,
        bytes_per_token: u64::try_from(layout.bytes_per_token).ok()?,
        page_size_bytes: u64::try_from(layout.page_size_bytes).ok()?,
    })
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

pub fn try_build_bootstrap_grpc_router_service(
    args: &ServerArgs,
) -> Result<BootstrapGrpcRouterService, ServerLaunchError> {
    validate_local_model_artifacts_if_present(args)?;
    let model = bootstrap_forward_model_from_server_args(args)?;
    try_build_bootstrap_grpc_router_service_from_model(args, model)
}

fn try_build_bootstrap_grpc_router_service_from_model(
    args: &ServerArgs,
    model: BootstrapForwardModel,
) -> Result<BootstrapGrpcRouterService, ServerLaunchError> {
    let prefix_cache_policy = model_prefix_cache_policy(&model);
    let scheduler = Scheduler::with_cache_resources_and_policy(
        bootstrap_model_runner(model),
        RadixCache::default(),
        cache_page_allocator_from_server_args(args)?,
        prefix_cache_policy,
    )
    .with_max_running_requests(args.max_running_requests);
    let tokenizer = RuntimeTokenizer::from_model_or_tokenizer_path(
        &args.model_path,
        args.tokenizer_path.as_deref(),
    )?;
    let engine = Engine::new(tokenizer, scheduler);
    let runtime = RouterRuntime::new(engine)
        .with_default_stop_token_ids(model_config_eos_token_ids(&args.model_path));
    Ok(GrpcRouterService::with_server_args(runtime, args))
}

pub fn build_bootstrap_http_router_service(args: &ServerArgs) -> BootstrapHttpRouterService {
    try_build_bootstrap_http_router_service(args).expect("bootstrap tokenizer should load")
}

pub fn try_build_bootstrap_http_router_service(
    args: &ServerArgs,
) -> Result<BootstrapHttpRouterService, ServerLaunchError> {
    validate_local_model_artifacts_if_present(args)?;
    let model = bootstrap_forward_model_from_server_args(args)?;
    try_build_bootstrap_http_router_service_from_model(args, model)
}

fn try_build_bootstrap_http_router_service_from_model(
    args: &ServerArgs,
    model: BootstrapForwardModel,
) -> Result<BootstrapHttpRouterService, ServerLaunchError> {
    let prefix_cache_policy = model_prefix_cache_policy(&model);
    let scheduler = Scheduler::with_cache_resources_and_policy(
        bootstrap_model_runner(model),
        RadixCache::default(),
        cache_page_allocator_from_server_args(args)?,
        prefix_cache_policy,
    )
    .with_max_running_requests(args.max_running_requests);
    let tokenizer = RuntimeTokenizer::from_model_or_tokenizer_path(
        &args.model_path,
        args.tokenizer_path.as_deref(),
    )?;
    let engine = Engine::new(tokenizer, scheduler);
    let runtime = RouterRuntime::new(engine)
        .with_default_stop_token_ids(model_config_eos_token_ids(&args.model_path));
    Ok(
        HttpRouterService::new(runtime, RouterGetModelInfoResponse::from_server_args(args))
            .with_server_info(http_server_info_from_args(args)),
    )
}

pub fn build_bootstrap_prefill_http_router_service(
    args: &ServerArgs,
) -> BootstrapPrefillHttpRouterService {
    try_build_bootstrap_prefill_http_router_service(args).expect("bootstrap tokenizer should load")
}

fn cache_page_allocator_from_server_args(
    args: &ServerArgs,
) -> Result<CachePageAllocator, ServerLaunchError> {
    CachePageAllocator::with_page_size(args.num_reserved_decode_tokens, args.page_size)
        .map_err(|error| ServerLaunchError::KvCacheTransfer(error.to_string()))
}

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
    .with_max_running_requests(args.max_running_requests);
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
    E: KvCacheTransferExecutor,
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
    E: KvCacheTransferExecutor,
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
    E: KvCacheTransferExecutor,
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
    E: KvCacheTransferExecutor,
    P: DecodeBootstrapPublisher,
{
    validate_kv_only_transfer_model(&model)?;
    try_build_bootstrap_pd_http_router_service_from_runner_with_decode_publisher(
        args,
        bootstrap_model_runner(model),
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
    E: KvCacheTransferExecutor,
    P: DecodeBootstrapPublisher,
{
    let mut worker = KvTransferModelWorker::new(model_runner, registry, transfer_executor)
        .with_decode_bootstrap_publisher(decode_bootstrap_publisher)
        .with_kv_page_size(args.page_size);
    if decode_side_bootstrap_only {
        worker = worker.with_decode_side_bootstrap_only();
    }
    let scheduler = Scheduler::with_cache_resources(
        worker,
        RadixCache::default(),
        cache_page_allocator_from_server_args(args)?,
    )
    .with_max_running_requests(args.max_running_requests);
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
            .with_disaggregated_requests()
            .with_max_transfer_polls(args.disaggregation_decode_polling_interval),
    )
}

fn launch_mooncake_decode_bootstrap_publisher(
    args: &ServerArgs,
    kv_cache_layout: MooncakeKvCacheLayout,
    mooncake_session_id: impl Into<String>,
) -> MooncakeDecodeBootstrapPublisher {
    let endpoint = prefill_mooncake_route_rank_ip(args);
    let dst_port = args
        .disaggregation_zmq_ports
        .map(|ports| ports.start)
        .unwrap_or(args.port);
    MooncakeDecodeBootstrapPublisher::new(endpoint, dst_port, mooncake_session_id)
        .with_kv_cache_layout(kv_cache_layout)
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

#[cfg(not(feature = "mooncake-link"))]
fn launch_mooncake_prefill_kv_layout(
    model_runner: &ModelRunner<BootstrapForwardModel>,
) -> Result<MooncakeKvCacheLayout, ServerLaunchError> {
    validate_kv_only_transfer_model(model_runner.model())?;
    Ok(mooncake_kv_memory_from_model_runner(model_runner)?.prefill_layout(0))
}

#[cfg(not(feature = "mooncake-link"))]
fn launch_mooncake_decode_kv_layout(
    model_runner: &ModelRunner<BootstrapForwardModel>,
) -> Result<MooncakeKvCacheLayout, ServerLaunchError> {
    validate_kv_only_transfer_model(model_runner.model())?;
    Ok(mooncake_kv_memory_from_model_runner(model_runner)?.prefill_layout(0))
}

fn mooncake_kv_memory_from_model_runner(
    model_runner: &ModelRunner<BootstrapForwardModel>,
) -> Result<TransferableKvCacheMemory, ServerLaunchError> {
    model_runner
        .mooncake_kv_cache_memory()
        .map_err(|error| ServerLaunchError::KvCacheTransfer(error.to_string()))
}

#[cfg(feature = "mooncake-link")]
fn prepare_linked_mooncake_kv_memory(
    model_runner: &mut ModelRunner<BootstrapForwardModel>,
    engine: &SharedLinkedMooncakeTransferEngine,
    slot_capacity: usize,
    page_size: usize,
) -> Result<RegisteredMooncakeKvCacheMemory<SharedLinkedMooncakeTransferEngine>, ServerLaunchError>
{
    validate_kv_only_transfer_model(model_runner.model())?;
    model_runner
        .model()
        .runtime_capability()
        .validate_requirements(&RuntimeRequirements {
            requires_kv_cache_registration: true,
            requires_mooncake: true,
            ..RuntimeRequirements::default()
        })
        .map_err(|mismatch| ServerLaunchError::MissingRuntimeCapabilities {
            runtime_name: mismatch.runtime_name.to_string(),
            missing: mismatch.missing,
        })?;
    model_runner
        .reserve_mooncake_kv_cache_slots(slot_capacity, page_size)
        .map_err(|error| ServerLaunchError::KvCacheTransfer(error.to_string()))?;
    let memory = mooncake_kv_memory_from_model_runner(model_runner)?;
    RegisteredMooncakeKvCacheMemory::register(engine.clone(), memory).map_err(Into::into)
}

#[cfg(not(feature = "mooncake-link"))]
fn try_build_launch_mooncake_prefill_http_router_service(
    args: &ServerArgs,
    _pd_config: &PdConfig,
    bootstrap_service: PrefillBootstrapService,
) -> Result<
    BootstrapPdHttpRouterService<
        MooncakeBootstrapKvCacheTransferExecutor<
            MooncakeKvCacheTransferExecutor<UnlinkedMooncakeTransferEngine>,
        >,
    >,
    ServerLaunchError,
> {
    let model_runner = bootstrap_model_runner(bootstrap_forward_model_for_launch(args)?);
    let transfer_executor = MooncakeKvCacheTransferExecutor::new(
        UnlinkedMooncakeTransferEngine,
        launch_mooncake_prefill_kv_layout(&model_runner)?,
        MooncakeTransferTarget { target_id: 0 },
    );
    try_build_bootstrap_pd_http_router_service_from_runner_with_decode_publisher(
        args,
        model_runner,
        DecodeBootstrapRegistry::default(),
        MooncakeBootstrapKvCacheTransferExecutor::new(bootstrap_service, transfer_executor),
        crate::transfer::NoopDecodeBootstrapPublisher,
        false,
    )
}

#[cfg(not(feature = "mooncake-link"))]
fn try_build_launch_mooncake_decode_http_router_service(
    args: &ServerArgs,
    _pd_config: &PdConfig,
) -> Result<
    BootstrapPdHttpRouterService<
        MooncakeKvCacheTransferExecutor<UnlinkedMooncakeTransferEngine>,
        MooncakeDecodeBootstrapPublisher,
    >,
    ServerLaunchError,
> {
    let model_runner = bootstrap_model_runner(bootstrap_forward_model_for_launch(args)?);
    let kv_cache_layout = launch_mooncake_decode_kv_layout(&model_runner)?;
    let transfer_executor = MooncakeKvCacheTransferExecutor::new(
        UnlinkedMooncakeTransferEngine,
        kv_cache_layout,
        MooncakeTransferTarget { target_id: 0 },
    );
    try_build_bootstrap_pd_http_router_service_from_runner_with_decode_publisher(
        args,
        model_runner,
        DecodeBootstrapRegistry::default(),
        transfer_executor,
        launch_mooncake_decode_bootstrap_publisher(
            args,
            kv_cache_layout,
            format!("{}:{}", prefill_mooncake_route_rank_ip(args), args.port),
        ),
        true,
    )
}

#[cfg(not(feature = "mooncake-link"))]
fn try_build_launch_mooncake_prefill_grpc_router_service(
    args: &ServerArgs,
    _pd_config: &PdConfig,
    bootstrap_service: PrefillBootstrapService,
) -> Result<
    BootstrapPdGrpcRouterService<
        MooncakeBootstrapKvCacheTransferExecutor<
            MooncakeKvCacheTransferExecutor<UnlinkedMooncakeTransferEngine>,
        >,
    >,
    ServerLaunchError,
> {
    let model_runner = bootstrap_model_runner(bootstrap_forward_model_for_launch(args)?);
    let transfer_executor = MooncakeKvCacheTransferExecutor::new(
        UnlinkedMooncakeTransferEngine,
        launch_mooncake_prefill_kv_layout(&model_runner)?,
        MooncakeTransferTarget { target_id: 0 },
    );
    try_build_bootstrap_pd_grpc_router_service_from_runner_with_decode_publisher(
        args,
        model_runner,
        DecodeBootstrapRegistry::default(),
        MooncakeBootstrapKvCacheTransferExecutor::new(bootstrap_service, transfer_executor),
        crate::transfer::NoopDecodeBootstrapPublisher,
        false,
    )
}

#[cfg(not(feature = "mooncake-link"))]
fn try_build_launch_mooncake_decode_grpc_router_service(
    args: &ServerArgs,
    _pd_config: &PdConfig,
) -> Result<
    BootstrapPdGrpcRouterService<
        MooncakeKvCacheTransferExecutor<UnlinkedMooncakeTransferEngine>,
        MooncakeDecodeBootstrapPublisher,
    >,
    ServerLaunchError,
> {
    let model_runner = bootstrap_model_runner(bootstrap_forward_model_for_launch(args)?);
    let kv_cache_layout = launch_mooncake_decode_kv_layout(&model_runner)?;
    let transfer_executor = MooncakeKvCacheTransferExecutor::new(
        UnlinkedMooncakeTransferEngine,
        kv_cache_layout,
        MooncakeTransferTarget { target_id: 0 },
    );
    try_build_bootstrap_pd_grpc_router_service_from_runner_with_decode_publisher(
        args,
        model_runner,
        DecodeBootstrapRegistry::default(),
        transfer_executor,
        launch_mooncake_decode_bootstrap_publisher(
            args,
            kv_cache_layout,
            format!("{}:{}", prefill_mooncake_route_rank_ip(args), args.port),
        ),
        true,
    )
}

#[cfg(feature = "mooncake-link")]
fn try_build_launch_mooncake_prefill_http_router_service(
    args: &ServerArgs,
    pd_config: &PdConfig,
    bootstrap_service: PrefillBootstrapService,
) -> Result<
    BootstrapPdHttpRouterService<
        MooncakeBootstrapKvCacheTransferExecutor<
            MooncakeKvCacheTransferExecutor<
                SharedLinkedMooncakeTransferEngine,
                MooncakeSessionTargetResolver<SharedLinkedMooncakeTransferEngine>,
            >,
        >,
    >,
    ServerLaunchError,
> {
    let mut model_runner = bootstrap_model_runner(bootstrap_forward_model_for_launch(args)?);
    let engine_config = MooncakeTransferEngineConfig::from_pd_config_for_rank(
        prefill_mooncake_route_rank_ip(args),
        0,
        pd_config,
    );
    let engine = SharedLinkedMooncakeTransferEngine::new(&engine_config)?;
    let kv_registration = prepare_linked_mooncake_kv_memory(
        &mut model_runner,
        &engine,
        args.num_reserved_decode_tokens,
        args.page_size,
    )?;
    let kv_cache_layout = kv_registration.memory().prefill_layout(0);
    let target_resolver = MooncakeSessionTargetResolver::new(engine.clone(), Vec::new());
    let transfer_executor = MooncakeKvCacheTransferExecutor::with_target_resolver(
        engine,
        kv_cache_layout,
        target_resolver,
    )
    .with_local_memory_registration(kv_registration)
    .map_err(|error| ServerLaunchError::KvCacheTransfer(error.to_string()))?;
    try_build_bootstrap_pd_http_router_service_from_runner_with_decode_publisher(
        args,
        model_runner,
        DecodeBootstrapRegistry::default(),
        MooncakeBootstrapKvCacheTransferExecutor::new(bootstrap_service, transfer_executor),
        crate::transfer::NoopDecodeBootstrapPublisher,
        false,
    )
}

#[cfg(feature = "mooncake-link")]
fn try_build_launch_mooncake_decode_http_router_service(
    args: &ServerArgs,
    pd_config: &PdConfig,
) -> Result<
    BootstrapPdHttpRouterService<
        MooncakeKvCacheTransferExecutor<
            SharedLinkedMooncakeTransferEngine,
            MooncakeSessionTargetResolver<SharedLinkedMooncakeTransferEngine>,
        >,
        MooncakeDecodeBootstrapPublisher,
    >,
    ServerLaunchError,
> {
    let mut model_runner = bootstrap_model_runner(bootstrap_forward_model_for_launch(args)?);
    let engine_config = MooncakeTransferEngineConfig::from_pd_config_for_rank(
        prefill_mooncake_route_rank_ip(args),
        0,
        pd_config,
    );
    let engine = SharedLinkedMooncakeTransferEngine::new(&engine_config)?;
    let mooncake_session_id = engine.local_endpoint()?;
    let kv_registration = prepare_linked_mooncake_kv_memory(
        &mut model_runner,
        &engine,
        args.num_reserved_decode_tokens,
        args.page_size,
    )?;
    let target_resolver = MooncakeSessionTargetResolver::new(engine.clone(), Vec::new());
    let kv_cache_layout = kv_registration.memory().prefill_layout(0);
    let transfer_executor = MooncakeKvCacheTransferExecutor::with_target_resolver(
        engine,
        kv_cache_layout,
        target_resolver,
    )
    .with_local_memory_registration(kv_registration)
    .map_err(|error| ServerLaunchError::KvCacheTransfer(error.to_string()))?;
    try_build_bootstrap_pd_http_router_service_from_runner_with_decode_publisher(
        args,
        model_runner,
        DecodeBootstrapRegistry::default(),
        transfer_executor,
        launch_mooncake_decode_bootstrap_publisher(args, kv_cache_layout, mooncake_session_id),
        true,
    )
}

#[cfg(feature = "mooncake-link")]
fn try_build_launch_mooncake_prefill_grpc_router_service(
    args: &ServerArgs,
    pd_config: &PdConfig,
    bootstrap_service: PrefillBootstrapService,
) -> Result<
    BootstrapPdGrpcRouterService<
        MooncakeBootstrapKvCacheTransferExecutor<
            MooncakeKvCacheTransferExecutor<
                SharedLinkedMooncakeTransferEngine,
                MooncakeSessionTargetResolver<SharedLinkedMooncakeTransferEngine>,
            >,
        >,
    >,
    ServerLaunchError,
> {
    let mut model_runner = bootstrap_model_runner(bootstrap_forward_model_for_launch(args)?);
    let engine_config = MooncakeTransferEngineConfig::from_pd_config_for_rank(
        prefill_mooncake_route_rank_ip(args),
        0,
        pd_config,
    );
    let engine = SharedLinkedMooncakeTransferEngine::new(&engine_config)?;
    let kv_registration = prepare_linked_mooncake_kv_memory(
        &mut model_runner,
        &engine,
        args.num_reserved_decode_tokens,
        args.page_size,
    )?;
    let kv_cache_layout = kv_registration.memory().prefill_layout(0);
    let target_resolver = MooncakeSessionTargetResolver::new(engine.clone(), Vec::new());
    let transfer_executor = MooncakeKvCacheTransferExecutor::with_target_resolver(
        engine,
        kv_cache_layout,
        target_resolver,
    )
    .with_local_memory_registration(kv_registration)
    .map_err(|error| ServerLaunchError::KvCacheTransfer(error.to_string()))?;
    try_build_bootstrap_pd_grpc_router_service_from_runner_with_decode_publisher(
        args,
        model_runner,
        DecodeBootstrapRegistry::default(),
        MooncakeBootstrapKvCacheTransferExecutor::new(bootstrap_service, transfer_executor),
        crate::transfer::NoopDecodeBootstrapPublisher,
        false,
    )
}

#[cfg(feature = "mooncake-link")]
fn try_build_launch_mooncake_decode_grpc_router_service(
    args: &ServerArgs,
    pd_config: &PdConfig,
) -> Result<
    BootstrapPdGrpcRouterService<
        MooncakeKvCacheTransferExecutor<
            SharedLinkedMooncakeTransferEngine,
            MooncakeSessionTargetResolver<SharedLinkedMooncakeTransferEngine>,
        >,
        MooncakeDecodeBootstrapPublisher,
    >,
    ServerLaunchError,
> {
    let mut model_runner = bootstrap_model_runner(bootstrap_forward_model_for_launch(args)?);
    let engine_config = MooncakeTransferEngineConfig::from_pd_config_for_rank(
        prefill_mooncake_route_rank_ip(args),
        0,
        pd_config,
    );
    let engine = SharedLinkedMooncakeTransferEngine::new(&engine_config)?;
    let mooncake_session_id = engine.local_endpoint()?;
    let kv_registration = prepare_linked_mooncake_kv_memory(
        &mut model_runner,
        &engine,
        args.num_reserved_decode_tokens,
        args.page_size,
    )?;
    let target_resolver = MooncakeSessionTargetResolver::new(engine.clone(), Vec::new());
    let kv_cache_layout = kv_registration.memory().prefill_layout(0);
    let transfer_executor = MooncakeKvCacheTransferExecutor::with_target_resolver(
        engine,
        kv_cache_layout,
        target_resolver,
    )
    .with_local_memory_registration(kv_registration)
    .map_err(|error| ServerLaunchError::KvCacheTransfer(error.to_string()))?;
    try_build_bootstrap_pd_grpc_router_service_from_runner_with_decode_publisher(
        args,
        model_runner,
        DecodeBootstrapRegistry::default(),
        transfer_executor,
        launch_mooncake_decode_bootstrap_publisher(args, kv_cache_layout, mooncake_session_id),
        true,
    )
}

pub fn build_bootstrap_pd_grpc_router_service<E>(
    args: &ServerArgs,
    registry: DecodeBootstrapRegistry,
    transfer_executor: E,
) -> BootstrapPdGrpcRouterService<E>
where
    E: KvCacheTransferExecutor,
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
    E: KvCacheTransferExecutor,
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
    E: KvCacheTransferExecutor,
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
    E: KvCacheTransferExecutor,
    P: DecodeBootstrapPublisher,
{
    validate_kv_only_transfer_model(&model)?;
    try_build_bootstrap_pd_grpc_router_service_from_runner_with_decode_publisher(
        args,
        bootstrap_model_runner(model),
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
    E: KvCacheTransferExecutor,
    P: DecodeBootstrapPublisher,
{
    let mut worker = KvTransferModelWorker::new(model_runner, registry, transfer_executor)
        .with_decode_bootstrap_publisher(decode_bootstrap_publisher)
        .with_kv_page_size(args.page_size);
    if decode_side_bootstrap_only {
        worker = worker.with_decode_side_bootstrap_only();
    }
    let scheduler = Scheduler::with_cache_resources(
        worker,
        RadixCache::default(),
        cache_page_allocator_from_server_args(args)?,
    )
    .with_max_running_requests(args.max_running_requests);
    let tokenizer = RuntimeTokenizer::from_model_or_tokenizer_path(
        &args.model_path,
        args.tokenizer_path.as_deref(),
    )?;
    let engine = Engine::new(tokenizer, scheduler);
    let runtime = RouterRuntime::new(engine)
        .with_default_stop_token_ids(model_config_eos_token_ids(&args.model_path));
    Ok(GrpcRouterService::with_server_args(runtime, args)
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

    match pd_config.transfer_backend {
        TransferBackend::Fake => Ok(()),
        TransferBackend::Mooncake if cfg!(feature = "mooncake-link") => Ok(()),
        TransferBackend::Mooncake => Err(MooncakeError::UnavailableWithoutLink.into()),
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

fn try_build_launch_fake_pd_grpc_router_service(
    args: &ServerArgs,
    decode_side_bootstrap_only: bool,
) -> Result<BootstrapFakePdGrpcRouterService, ServerLaunchError> {
    validate_local_model_artifacts_if_present(args)?;
    let model = bootstrap_forward_model_for_launch(args)?;
    try_build_bootstrap_pd_grpc_router_service_from_model_with_decode_publisher(
        args,
        model,
        DecodeBootstrapRegistry::default(),
        FakeKvCacheTransferExecutor::default(),
        crate::transfer::NoopDecodeBootstrapPublisher,
        decode_side_bootstrap_only,
    )
}

fn try_build_launch_fake_pd_http_router_service(
    args: &ServerArgs,
    decode_side_bootstrap_only: bool,
) -> Result<BootstrapPdHttpRouterService<FakeKvCacheTransferExecutor>, ServerLaunchError> {
    validate_local_model_artifacts_if_present(args)?;
    let model = bootstrap_forward_model_for_launch(args)?;
    try_build_bootstrap_pd_http_router_service_from_model_with_decode_publisher(
        args,
        model,
        DecodeBootstrapRegistry::default(),
        FakeKvCacheTransferExecutor::default(),
        crate::transfer::NoopDecodeBootstrapPublisher,
        decode_side_bootstrap_only,
    )
}

pub async fn launch_grpc_server(args: ServerArgs) -> Result<(), ServerLaunchError> {
    launch_grpc_server_with_shutdown(args, std::future::pending::<()>()).await
}

pub async fn launch_grpc_server_with_shutdown<F>(
    args: ServerArgs,
    shutdown: F,
) -> Result<(), ServerLaunchError>
where
    F: Future<Output = ()> + Send + 'static,
{
    let pd_config = PdConfig::from_server_args(&args)?;
    if pd_config.mode == DisaggregationMode::Decode
        && pd_config.transfer_backend == TransferBackend::Mooncake
        && pd_config.kv_cache_model_layout.is_none()
    {
        return Err(ServerLaunchError::PdConfig(
            PdConfigError::MissingMooncakeKvCacheModelLayout,
        ));
    }
    validate_launch_transfer_backend(&pd_config)?;
    let addr = grpc_listen_addr(&args)?;
    let sidecar_addr = grpc_http_sidecar_listen_addr(&args)?;
    match pd_config.mode {
        DisaggregationMode::Null => {
            let service = try_build_launch_grpc_router_service(&args)?;
            serve_grpc_and_sidecar(addr, service, sidecar_addr, args.enable_metrics, shutdown)
                .await?;
        }
        DisaggregationMode::Prefill if pd_config.transfer_backend == TransferBackend::Fake => {
            let service = try_build_launch_fake_pd_grpc_router_service(&args, false)?;
            serve_grpc_and_sidecar(addr, service, sidecar_addr, args.enable_metrics, shutdown)
                .await?;
        }
        DisaggregationMode::Decode if pd_config.transfer_backend == TransferBackend::Fake => {
            let service = try_build_launch_fake_pd_grpc_router_service(&args, true)?;
            serve_grpc_and_sidecar(addr, service, sidecar_addr, args.enable_metrics, shutdown)
                .await?;
        }
        DisaggregationMode::Prefill if pd_config.transfer_backend == TransferBackend::Mooncake => {
            let bootstrap_addr = prefill_bootstrap_listen_addr(&args)?;
            let zmq_endpoints = prefill_mooncake_zmq_endpoints(&args);
            let bootstrap_service = PrefillBootstrapService::default();
            register_prefill_mooncake_routes_from_args(&bootstrap_service, &args)?;
            let service = try_build_launch_mooncake_prefill_grpc_router_service(
                &args,
                &pd_config,
                bootstrap_service.clone(),
            )?;
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
        DisaggregationMode::Decode if pd_config.transfer_backend == TransferBackend::Mooncake => {
            let service = try_build_launch_mooncake_decode_grpc_router_service(&args, &pd_config)?;
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
    launch_http_server_with_shutdown(args, std::future::pending::<()>()).await
}

pub async fn launch_http_server_with_shutdown<F>(
    args: ServerArgs,
    shutdown: F,
) -> Result<(), ServerLaunchError>
where
    F: Future<Output = ()> + Send + 'static,
{
    let pd_config = PdConfig::from_server_args(&args)?;
    if pd_config.mode == DisaggregationMode::Decode
        && pd_config.transfer_backend == TransferBackend::Mooncake
        && pd_config.kv_cache_model_layout.is_none()
    {
        return Err(ServerLaunchError::PdConfig(
            PdConfigError::MissingMooncakeKvCacheModelLayout,
        ));
    }
    validate_launch_transfer_backend(&pd_config)?;
    let addr = http_listen_addr(&args)?;
    let engine_info_addr = engine_info_bootstrap_listen_addr(&args)?;
    let engine_info_service = EngineInfoBootstrapService::default();
    match pd_config.mode {
        DisaggregationMode::Null => {
            let service = try_build_launch_http_router_service(&args)?
                .with_engine_info_bootstrap_service(engine_info_service.clone());
            serve_http_and_engine_info_bootstrap(
                addr,
                service,
                engine_info_addr,
                engine_info_service,
                shutdown,
            )
            .await?;
        }
        DisaggregationMode::Prefill if pd_config.transfer_backend == TransferBackend::Fake => {
            let service = try_build_launch_fake_pd_http_router_service(&args, false)?
                .with_engine_info_bootstrap_service(engine_info_service.clone());
            serve_http_and_engine_info_bootstrap(
                addr,
                service,
                engine_info_addr,
                engine_info_service,
                shutdown,
            )
            .await?;
        }
        DisaggregationMode::Decode if pd_config.transfer_backend == TransferBackend::Fake => {
            let service = try_build_launch_fake_pd_http_router_service(&args, true)?
                .with_engine_info_bootstrap_service(engine_info_service.clone());
            serve_http_and_engine_info_bootstrap(
                addr,
                service,
                engine_info_addr,
                engine_info_service,
                shutdown,
            )
            .await?;
        }
        DisaggregationMode::Prefill if pd_config.transfer_backend == TransferBackend::Mooncake => {
            let bootstrap_addr = prefill_bootstrap_listen_addr(&args)?;
            let zmq_endpoints = prefill_mooncake_zmq_endpoints(&args);
            let bootstrap_service = PrefillBootstrapService::default();
            register_prefill_mooncake_routes_from_args(&bootstrap_service, &args)?;
            let service = try_build_launch_mooncake_prefill_http_router_service(
                &args,
                &pd_config,
                bootstrap_service.clone(),
            )?
            .with_engine_info_bootstrap_service(engine_info_service.clone());
            serve_prefill_http_and_bootstrap(
                addr,
                service,
                PrefillHttpBootstrapServices {
                    prefill_addr: bootstrap_addr,
                    prefill_service: bootstrap_service,
                    engine_info_addr,
                    engine_info_service,
                    zmq_endpoints,
                },
                shutdown,
            )
            .await?;
        }
        DisaggregationMode::Decode if pd_config.transfer_backend == TransferBackend::Mooncake => {
            let service = try_build_launch_mooncake_decode_http_router_service(&args, &pd_config)?
                .with_engine_info_bootstrap_service(engine_info_service.clone());
            serve_http_and_engine_info_bootstrap(
                addr,
                service,
                engine_info_addr,
                engine_info_service,
                shutdown,
            )
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

async fn serve_http_and_engine_info_bootstrap<T, W, F>(
    http_addr: SocketAddr,
    http_service: HttpRouterService<T, W>,
    engine_info_addr: SocketAddr,
    engine_info_service: EngineInfoBootstrapService,
    shutdown: F,
) -> Result<(), ServerLaunchError>
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
    F: Future<Output = ()> + Send + 'static,
{
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let mut http_task = tokio::spawn(serve_http_router_with_shutdown(
        http_addr,
        http_service,
        watch_shutdown(shutdown_rx.clone()),
    ));
    let mut bootstrap_task = tokio::spawn(serve_engine_info_bootstrap_with_shutdown(
        engine_info_addr,
        engine_info_service,
        watch_shutdown(shutdown_rx.clone()),
    ));

    tokio::select! {
        _ = shutdown => {
            let _ = shutdown_tx.send(true);
            join_http_task(http_task).await?;
            join_engine_info_bootstrap_task(bootstrap_task).await?;
            Ok(())
        }
        result = &mut http_task => {
            let _ = shutdown_tx.send(true);
            join_engine_info_bootstrap_task(bootstrap_task).await?;
            result.map_err(|error| ServerLaunchError::ServerTaskJoin(error.to_string()))??;
            Ok(())
        }
        result = &mut bootstrap_task => {
            let _ = shutdown_tx.send(true);
            join_http_task(http_task).await?;
            result.map_err(|error| ServerLaunchError::ServerTaskJoin(error.to_string()))??;
            Ok(())
        }
    }
}

struct PrefillHttpBootstrapServices {
    prefill_addr: SocketAddr,
    prefill_service: PrefillBootstrapService,
    engine_info_addr: SocketAddr,
    engine_info_service: EngineInfoBootstrapService,
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
    let mut http_task = tokio::spawn(serve_http_router_with_shutdown(
        http_addr,
        http_service,
        watch_shutdown(shutdown_rx.clone()),
    ));
    let mut bootstrap_tasks = tokio::task::JoinSet::new();
    let prefill_shutdown_rx = shutdown_rx.clone();
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
    let engine_info_shutdown_rx = shutdown_rx.clone();
    bootstrap_tasks.spawn(async move {
        serve_engine_info_bootstrap_with_shutdown(
            bootstrap.engine_info_addr,
            bootstrap.engine_info_service,
            watch_shutdown(engine_info_shutdown_rx),
        )
        .await
        .map_err(ServerLaunchError::from)
    });
    if !bootstrap.zmq_endpoints.is_empty() {
        bootstrap_tasks.spawn(async move {
            serve_mooncake_bootstrap_zmq_endpoints_with_shutdown(
                bootstrap.zmq_endpoints,
                bootstrap.prefill_service,
                watch_shutdown(shutdown_rx),
            )
            .await
            .map_err(ServerLaunchError::from)
        });
    }

    tokio::select! {
        _ = shutdown => {
            let _ = shutdown_tx.send(true);
            join_http_task(http_task).await?;
            join_launch_tasks(bootstrap_tasks).await?;
            Ok(())
        }
        result = &mut http_task => {
            let _ = shutdown_tx.send(true);
            join_launch_tasks(bootstrap_tasks).await?;
            result.map_err(|error| ServerLaunchError::ServerTaskJoin(error.to_string()))??;
            Ok(())
        }
        result = bootstrap_tasks.join_next() => {
            let _ = shutdown_tx.send(true);
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
        watch_shutdown(shutdown_rx.clone()),
    ));
    if !bootstrap.zmq_endpoints.is_empty() {
        bootstrap_tasks.spawn(serve_mooncake_bootstrap_zmq_endpoints_with_shutdown(
            bootstrap.zmq_endpoints,
            bootstrap.prefill_service,
            watch_shutdown(shutdown_rx),
        ));
    }

    tokio::select! {
        _ = shutdown => {
            let _ = shutdown_tx.send(true);
            join_grpc_task(grpc_task).await?;
            join_http_task(sidecar_task).await?;
            join_bootstrap_tasks(bootstrap_tasks).await?;
            Ok(())
        }
        result = &mut grpc_task => {
            let _ = shutdown_tx.send(true);
            join_http_task(sidecar_task).await?;
            join_bootstrap_tasks(bootstrap_tasks).await?;
            result.map_err(|error| ServerLaunchError::ServerTaskJoin(error.to_string()))??;
            Ok(())
        }
        result = &mut sidecar_task => {
            let _ = shutdown_tx.send(true);
            join_grpc_task(grpc_task).await?;
            join_bootstrap_tasks(bootstrap_tasks).await?;
            result.map_err(|error| ServerLaunchError::ServerTaskJoin(error.to_string()))??;
            Ok(())
        }
        result = bootstrap_tasks.join_next() => {
            let _ = shutdown_tx.send(true);
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

async fn join_engine_info_bootstrap_task(
    task: tokio::task::JoinHandle<Result<(), EngineInfoBootstrapServeError>>,
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
