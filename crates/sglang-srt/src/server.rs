use std::fmt;
use std::net::{SocketAddr, ToSocketAddrs};

use crate::cli::ServerArgs;
use crate::engine::Engine;
use crate::grpc::{GrpcRouterService, GrpcServeError, serve_grpc_router};
use crate::router::RouterRuntime;
use crate::scheduler::{ScheduleBatch, Scheduler};
use crate::tokenizer::ByteTokenizer;
use crate::worker::{BatchGeneratedTokens, GeneratedToken, ModelWorker};

#[derive(Clone, Debug, Default)]
pub struct BootstrapModelWorker;

impl ModelWorker for BootstrapModelWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        let token = GeneratedToken::finished(vec![b' ' as u32]);
        BatchGeneratedTokens::from_batch(batch, vec![token; batch.batch_size()])
            .expect("bootstrap worker output shape should match batch")
    }
}

pub type BootstrapGrpcRouterService = GrpcRouterService<ByteTokenizer, BootstrapModelWorker>;

#[derive(Debug)]
pub enum ServerLaunchError {
    AddressResolve(std::io::Error),
    NoSocketAddress { host: String, port: u16 },
    Grpc(GrpcServeError),
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
    let scheduler = Scheduler::new(BootstrapModelWorker);
    let engine = Engine::new(ByteTokenizer, scheduler);
    let runtime = RouterRuntime::new(engine);
    GrpcRouterService::with_server_args(runtime, args)
}

pub async fn launch_grpc_server(args: ServerArgs) -> Result<(), ServerLaunchError> {
    let addr = grpc_listen_addr(&args)?;
    let service = build_bootstrap_grpc_router_service(&args);
    serve_grpc_router(addr, service, true).await?;
    Ok(())
}
