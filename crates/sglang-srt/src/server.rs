use std::fmt;
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::Path;

use crate::cache::{CachePageAllocator, RadixCache};
use crate::cli::ServerArgs;
use crate::engine::Engine;
use crate::grpc::{GrpcRouterService, GrpcServeError, serve_grpc_router};
use crate::model_artifacts::{LocalModelArtifacts, ModelArtifactError};
use crate::model_executor::{ForwardModel, ModelForwardOutput, ModelRunner, ModelWorkerBatch};
use crate::router::RouterRuntime;
use crate::scheduler::Scheduler;
use crate::tokenizer::{RuntimeTokenizer, TokenizerError};
use crate::transfer::{
    DecodeBootstrapRegistry, DisaggregationMode, FakeKvCacheTransferExecutor,
    KvCacheTransferExecutor, KvTransferModelWorker, PdConfig, PdConfigError, TransferBackend,
};

#[derive(Clone, Debug, Default)]
pub struct BootstrapForwardModel;

impl ForwardModel for BootstrapForwardModel {
    fn forward(&mut self, batch: &ModelWorkerBatch) -> ModelForwardOutput {
        let mut logits = Vec::with_capacity(batch.request_ids().len());
        for _ in batch.request_ids() {
            let mut row = vec![0.0; (b' ' as usize) + 1];
            row[b' ' as usize] = 1.0;
            logits.push(row);
        }

        ModelForwardOutput::new(logits).expect("bootstrap logits should be rectangular")
    }
}

pub type BootstrapGrpcRouterService =
    GrpcRouterService<RuntimeTokenizer, ModelRunner<BootstrapForwardModel>>;
pub type BootstrapPdGrpcRouterService<E> = GrpcRouterService<
    RuntimeTokenizer,
    KvTransferModelWorker<ModelRunner<BootstrapForwardModel>, E>,
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
    PdConfig(PdConfigError),
    UnsupportedBootstrapPdRuntime {
        mode: DisaggregationMode,
        transfer_backend: TransferBackend,
    },
    ModelArtifact(ModelArtifactError),
    Tokenizer(TokenizerError),
    Grpc(GrpcServeError),
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
            Self::PdConfig(error) => write!(formatter, "PD config error: {error}"),
            Self::UnsupportedBootstrapPdRuntime {
                mode,
                transfer_backend,
            } => write!(
                formatter,
                "bootstrap server does not support PD runtime mode {mode:?} with backend {transfer_backend:?}"
            ),
            Self::ModelArtifact(error) => write!(formatter, "model artifact error: {error}"),
            Self::Tokenizer(error) => write!(formatter, "tokenizer error: {error}"),
            Self::Grpc(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for ServerLaunchError {}

impl From<GrpcServeError> for ServerLaunchError {
    fn from(value: GrpcServeError) -> Self {
        Self::Grpc(value)
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

pub fn build_bootstrap_grpc_router_service(args: &ServerArgs) -> BootstrapGrpcRouterService {
    try_build_bootstrap_grpc_router_service(args).expect("bootstrap tokenizer should load")
}

pub fn try_build_bootstrap_grpc_router_service(
    args: &ServerArgs,
) -> Result<BootstrapGrpcRouterService, ServerLaunchError> {
    validate_local_model_artifacts_if_present(args)?;
    let scheduler = Scheduler::new(ModelRunner::new(BootstrapForwardModel))
        .with_max_running_requests(args.max_running_requests);
    let tokenizer = RuntimeTokenizer::from_model_or_tokenizer_path(
        &args.model_path,
        args.tokenizer_path.as_deref(),
    )?;
    let engine = Engine::new(tokenizer, scheduler);
    let runtime = RouterRuntime::new(engine);
    Ok(GrpcRouterService::with_server_args(runtime, args))
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
    validate_local_model_artifacts_if_present(args)?;
    let worker = KvTransferModelWorker::new(
        ModelRunner::new(BootstrapForwardModel),
        registry,
        transfer_executor,
    );
    let scheduler = Scheduler::with_cache_resources(
        worker,
        RadixCache::default(),
        CachePageAllocator::new(args.num_reserved_decode_tokens),
    )
    .with_max_running_requests(args.max_running_requests);
    let tokenizer = RuntimeTokenizer::from_model_or_tokenizer_path(
        &args.model_path,
        args.tokenizer_path.as_deref(),
    )?;
    let engine = Engine::new(tokenizer, scheduler);
    let runtime = RouterRuntime::new(engine);
    Ok(GrpcRouterService::with_server_args(runtime, args)
        .with_max_transfer_polls(args.disaggregation_decode_polling_interval))
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
    artifacts.validate_routed_expert_checkpoint_coverage()?;
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

pub async fn launch_grpc_server(args: ServerArgs) -> Result<(), ServerLaunchError> {
    let pd_config = PdConfig::from_server_args(&args)?;
    if pd_config.mode == DisaggregationMode::Decode
        && pd_config.transfer_backend == TransferBackend::Mooncake
        && pd_config.kv_cache_model_layout.is_none()
    {
        return Err(ServerLaunchError::PdConfig(
            PdConfigError::MissingMooncakeKvCacheModelLayout,
        ));
    }
    if pd_config.mode != DisaggregationMode::Null
        && (pd_config.mode != DisaggregationMode::Decode
            || pd_config.transfer_backend != TransferBackend::Fake)
    {
        return Err(ServerLaunchError::UnsupportedBootstrapPdRuntime {
            mode: pd_config.mode,
            transfer_backend: pd_config.transfer_backend,
        });
    }

    let addr = grpc_listen_addr(&args)?;
    if pd_config.mode == DisaggregationMode::Decode {
        let service = try_build_bootstrap_pd_grpc_router_service(
            &args,
            DecodeBootstrapRegistry::default(),
            FakeKvCacheTransferExecutor::default(),
        )?;
        serve_grpc_router(addr, service, true).await?;
    } else {
        let service = try_build_bootstrap_grpc_router_service(&args)?;
        serve_grpc_router(addr, service, true).await?;
    }
    Ok(())
}
