use std::fmt;
use std::net::{SocketAddr, ToSocketAddrs};

use crate::cache::{CachePageAllocator, RadixCache};
use crate::cli::ServerArgs;
use crate::engine::Engine;
use crate::grpc::{GrpcRouterService, GrpcServeError, serve_grpc_router};
use crate::model_executor::{ForwardModel, ModelForwardOutput, ModelRunner, ModelWorkerBatch};
use crate::router::RouterRuntime;
use crate::scheduler::Scheduler;
use crate::tokenizer::ByteTokenizer;
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
    GrpcRouterService<ByteTokenizer, ModelRunner<BootstrapForwardModel>>;
pub type BootstrapPdGrpcRouterService<E> =
    GrpcRouterService<ByteTokenizer, KvTransferModelWorker<ModelRunner<BootstrapForwardModel>, E>>;
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
    let scheduler = Scheduler::new(ModelRunner::new(BootstrapForwardModel))
        .with_max_running_requests(args.max_running_requests);
    let engine = Engine::new(ByteTokenizer, scheduler);
    let runtime = RouterRuntime::new(engine);
    GrpcRouterService::with_server_args(runtime, args)
}

pub fn build_bootstrap_pd_grpc_router_service<E>(
    args: &ServerArgs,
    registry: DecodeBootstrapRegistry,
    transfer_executor: E,
) -> BootstrapPdGrpcRouterService<E>
where
    E: KvCacheTransferExecutor,
{
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
    let engine = Engine::new(ByteTokenizer, scheduler);
    let runtime = RouterRuntime::new(engine);
    GrpcRouterService::with_server_args(runtime, args)
        .with_max_transfer_polls(args.disaggregation_decode_polling_interval)
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
        let service = build_bootstrap_fake_pd_grpc_router_service(&args);
        serve_grpc_router(addr, service, true).await?;
    } else {
        let service = build_bootstrap_grpc_router_service(&args);
        serve_grpc_router(addr, service, true).await?;
    }
    Ok(())
}
