#[cfg(feature = "mooncake-link")]
use std::ffi::c_void;
use std::fmt;
use std::future::Future;
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};

use crate::cache::{CachePageAllocator, RadixCache};
use crate::cli::ServerArgs;
use crate::deepseek_runtime::{
    DeepSeekRuntimeError, DeepSeekV4LoadedTensorParallelRuntime, DeepSeekV4Runtime,
    DeepSeekV4TensorShardLoadError,
};
use crate::engine::Engine;
use crate::glm_runtime::{
    GlmMoeDsaF32CachedForwardModel, GlmMoeDsaRuntime, GlmMoeDsaRuntimeError,
    GlmMoeDsaTensorShardLoadError,
};
use crate::grpc::{GrpcRouterService, GrpcServeError, serve_grpc_router_with_shutdown};
use crate::http::{
    HttpKvCacheInfo, HttpKvEventsInfo, HttpRouterService, HttpServeError, HttpServerInfo,
    serve_http_router_with_shutdown,
};
use crate::model_artifacts::{
    HfModelConfig, LocalModelArtifacts, ModelArtifactError, SafetensorsTensorDecodeError,
};
use crate::model_executor::{
    CpuEmbeddingLmModel, ForwardModel, ModelForwardError, ModelForwardOutput, ModelRunner,
    ModelWorkerBatch,
};
use crate::pd_bootstrap::{
    MooncakeBootstrapKvCacheTransferExecutor, MooncakeDecodeBootstrapPublisher,
    PrefillBootstrapServeError, PrefillBootstrapService, PrefillRouteRegistration,
    serve_mooncake_bootstrap_zmq_endpoints_with_shutdown, serve_prefill_bootstrap_with_shutdown,
};
use crate::router::{RouterGetModelInfoResponse, RouterRuntime};
use crate::scheduler::Scheduler;
use crate::tokenizer::{RuntimeTokenizer, Tokenizer, TokenizerError};
#[cfg(not(feature = "mooncake-link"))]
use crate::transfer::MooncakeTransferTarget;
#[cfg(not(feature = "mooncake-link"))]
use crate::transfer::UnlinkedMooncakeTransferEngine;
use crate::transfer::{
    DecodeBootstrapPublisher, DecodeBootstrapRegistry, DisaggregationMode,
    FakeKvCacheTransferExecutor, KvCacheTransferExecutor, KvTransferModelWorker,
    MooncakeBatchReleaser, MooncakeError, MooncakeKvCacheLayout, MooncakeKvCacheMemoryProvider,
    MooncakeKvCacheTransferExecutor, MooncakeTransferStatusReader, MooncakeTransferSubmitter,
    MooncakeTransferTargetResolver, PdConfig, PdConfigError, TransferBackend,
    TransferableKvCacheMemory,
};
#[cfg(feature = "mooncake-link")]
use crate::transfer::{
    MooncakeBufferEntry, MooncakeSessionTargetResolver, MooncakeTransferEngineConfig,
    SharedLinkedMooncakeTransferEngine,
};
use crate::worker::{WorkerExecutor, WorkerWeightUpdateRequest};

#[derive(Clone, Debug, Default)]
pub enum BootstrapForwardModel {
    #[default]
    Space,
    CpuEmbeddingLm(CpuEmbeddingLmModel),
    DeepSeekV4(DeepSeekV4LoadedTensorParallelRuntime),
    GlmMoeDsa(GlmMoeDsaF32CachedForwardModel),
    UnsupportedLocalModelRuntime {
        model_path: PathBuf,
        model_type: Option<String>,
    },
}

impl BootstrapForwardModel {
    fn from_server_args(args: &ServerArgs) -> Result<Self, ServerLaunchError> {
        let model_path = Path::new(&args.model_path);
        if model_path.is_dir() && !model_path.join("config.json").is_file() {
            return Ok(Self::Space);
        }

        let artifacts = match LocalModelArtifacts::from_model_path(&args.model_path) {
            Ok(artifacts) => artifacts,
            Err(ModelArtifactError::ModelPathNotLocalDirectory { .. })
            | Err(ModelArtifactError::NoSafetensorsWeights { .. }) => return Ok(Self::Space),
            Err(error) => return Err(error.into()),
        };

        Self::from_local_model_artifacts(&artifacts, args.tp_size)
    }

    fn from_local_model_artifacts(
        artifacts: &LocalModelArtifacts,
        tp_size: usize,
    ) -> Result<Self, ServerLaunchError> {
        Ok(
            match CpuEmbeddingLmModel::from_local_model_artifacts(artifacts)? {
                Some(model) => Self::CpuEmbeddingLm(model),
                None if artifacts.config().model_type.as_deref() == Some("deepseek_v4") => {
                    let runtime = DeepSeekV4Runtime::from_local_model_artifacts(artifacts)?;
                    Self::DeepSeekV4(runtime.load_tensor_parallel_shards(tp_size)?)
                }
                None if artifacts.config().model_type.as_deref() == Some("glm_moe_dsa") => {
                    let runtime = GlmMoeDsaRuntime::from_local_model_artifacts(artifacts)?;
                    Self::GlmMoeDsa(GlmMoeDsaF32CachedForwardModel::new(
                        runtime
                            .load_tensor_parallel_shards(tp_size)?
                            .decode_f32_tensor_parallel_shards()?,
                    ))
                }
                None => Self::UnsupportedLocalModelRuntime {
                    model_path: artifacts.model_path().to_path_buf(),
                    model_type: artifacts.config().model_type.clone(),
                },
            },
        )
    }

    fn reload_tp_size(&self) -> usize {
        match self {
            Self::DeepSeekV4(runtime) => runtime.rank_count(),
            Self::GlmMoeDsa(model) => model.rank_count(),
            Self::Space | Self::CpuEmbeddingLm(_) | Self::UnsupportedLocalModelRuntime { .. } => 1,
        }
    }

    #[cfg(feature = "mooncake-link")]
    fn reserve_mooncake_kv_cache_pages(
        &mut self,
        page_count: usize,
    ) -> Result<(), ServerLaunchError> {
        match self {
            Self::GlmMoeDsa(model) => model
                .reserve_transfer_pages(page_count)
                .map_err(|error| ServerLaunchError::KvCacheTransfer(error.to_string())),
            Self::UnsupportedLocalModelRuntime {
                model_path,
                model_type,
            } => Err(ServerLaunchError::UnsupportedMooncakeKvMemory {
                model_path: model_path.display().to_string(),
                model_type: model_type.clone(),
            }),
            Self::Space => Err(ServerLaunchError::UnsupportedMooncakeKvMemory {
                model_path: "<bootstrap>".to_string(),
                model_type: Some("space".to_string()),
            }),
            Self::CpuEmbeddingLm(_) => Err(ServerLaunchError::UnsupportedMooncakeKvMemory {
                model_path: "<bootstrap>".to_string(),
                model_type: Some("cpu_embedding_lm".to_string()),
            }),
            Self::DeepSeekV4(_) => Err(ServerLaunchError::UnsupportedMooncakeKvMemory {
                model_path: "<bootstrap>".to_string(),
                model_type: Some("deepseek_v4".to_string()),
            }),
        }
    }
}

impl ForwardModel for BootstrapForwardModel {
    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        match self {
            Self::Space => {
                let mut logits = Vec::with_capacity(batch.request_ids().len());
                for _ in batch.request_ids() {
                    let mut row = vec![0.0; (b' ' as usize) + 1];
                    row[b' ' as usize] = 1.0;
                    logits.push(row);
                }

                ModelForwardOutput::new(logits)
            }
            Self::CpuEmbeddingLm(model) => model.forward(batch),
            Self::DeepSeekV4(runtime) => {
                let plan = runtime.forward_plan(batch);
                Err(ModelForwardError::Runtime(format!(
                    "DeepSeek V4 Rust forward kernels are not implemented; loaded {} tensor-parallel rank(s), {} tensor shard(s), and {} byte(s) across {} layer(s), planning {} request(s) and {} token(s)",
                    runtime.rank_count(),
                    runtime.loaded_shard_count(),
                    runtime.loaded_byte_len(),
                    runtime.layer_count(),
                    plan.request_ids().len(),
                    plan.input_ids().len()
                )))
            }
            Self::GlmMoeDsa(model) => model.forward(batch),
            Self::UnsupportedLocalModelRuntime {
                model_path,
                model_type,
            } => Err(ModelForwardError::Runtime(format!(
                "local model type {} has checkpoint metadata but no Rust forward runtime: {}",
                model_type.as_deref().unwrap_or("<unknown>"),
                model_path.display()
            ))),
        }
    }

    fn update_weights_from_disk(
        &mut self,
        request: &WorkerWeightUpdateRequest,
    ) -> Result<(), ModelForwardError> {
        let artifacts = LocalModelArtifacts::from_model_path(&request.model_path)
            .map_err(|error| ModelForwardError::Runtime(error.to_string()))?;
        let next = Self::from_local_model_artifacts(&artifacts, self.reload_tp_size())
            .map_err(|error| ModelForwardError::Runtime(error.to_string()))?;
        *self = next;
        Ok(())
    }
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
    PdConfig(PdConfigError),
    UnsupportedBootstrapPdRuntime {
        mode: DisaggregationMode,
        transfer_backend: TransferBackend,
    },
    ModelArtifact(ModelArtifactError),
    Tokenizer(TokenizerError),
    Grpc(GrpcServeError),
    Http(HttpServeError),
    PrefillBootstrap(PrefillBootstrapServeError),
    MooncakeTransfer(MooncakeError),
    KvCacheTransfer(String),
    UnsupportedMooncakeKvMemory {
        model_path: String,
        model_type: Option<String>,
    },
    DeepSeekRuntime(DeepSeekRuntimeError),
    DeepSeekTensorShardLoad(DeepSeekV4TensorShardLoadError),
    GlmRuntime(GlmMoeDsaRuntimeError),
    GlmTensorShardLoad(GlmMoeDsaTensorShardLoadError),
    GlmTensorDecode(SafetensorsTensorDecodeError),
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
            (Self::DeepSeekRuntime(left), Self::DeepSeekRuntime(right)) => left == right,
            (Self::DeepSeekTensorShardLoad(left), Self::DeepSeekTensorShardLoad(right)) => {
                left == right
            }
            (Self::GlmRuntime(left), Self::GlmRuntime(right)) => left == right,
            (Self::GlmTensorShardLoad(left), Self::GlmTensorShardLoad(right)) => left == right,
            (Self::GlmTensorDecode(left), Self::GlmTensorDecode(right)) => left == right,
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
            Self::Http(error) => write!(formatter, "{error}"),
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
            Self::DeepSeekRuntime(error) => write!(formatter, "DeepSeek runtime error: {error}"),
            Self::DeepSeekTensorShardLoad(error) => {
                write!(formatter, "DeepSeek tensor shard load error: {error}")
            }
            Self::GlmRuntime(error) => write!(formatter, "GLM runtime error: {error}"),
            Self::GlmTensorShardLoad(error) => {
                write!(formatter, "GLM tensor shard load error: {error}")
            }
            Self::GlmTensorDecode(error) => {
                write!(formatter, "GLM tensor decode error: {error}")
            }
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

impl From<TokenizerError> for ServerLaunchError {
    fn from(value: TokenizerError) -> Self {
        Self::Tokenizer(value)
    }
}

impl From<DeepSeekRuntimeError> for ServerLaunchError {
    fn from(value: DeepSeekRuntimeError) -> Self {
        Self::DeepSeekRuntime(value)
    }
}

impl From<DeepSeekV4TensorShardLoadError> for ServerLaunchError {
    fn from(value: DeepSeekV4TensorShardLoadError) -> Self {
        Self::DeepSeekTensorShardLoad(value)
    }
}

impl From<GlmMoeDsaRuntimeError> for ServerLaunchError {
    fn from(value: GlmMoeDsaRuntimeError) -> Self {
        Self::GlmRuntime(value)
    }
}

impl From<GlmMoeDsaTensorShardLoadError> for ServerLaunchError {
    fn from(value: GlmMoeDsaTensorShardLoadError) -> Self {
        Self::GlmTensorShardLoad(value)
    }
}

impl From<SafetensorsTensorDecodeError> for ServerLaunchError {
    fn from(value: SafetensorsTensorDecodeError) -> Self {
        Self::GlmTensorDecode(value)
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
        disaggregation_mode: args.disaggregation_mode.clone(),
        disaggregation_bootstrap_port: if args.disaggregation_mode == "prefill" {
            Some(args.disaggregation_bootstrap_port)
        } else {
            None
        },
        kv_events: None,
        kv_cache: None,
    };

    if let Some(ports) = args.disaggregation_zmq_ports {
        if let (Ok(block_size), Ok(dp_size)) =
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
    if is_wildcard_host(&args.host) {
        if let Some(dist_init_host) = args.dist_init_addr.as_deref().and_then(host_from_addr) {
            return dist_init_host.to_string();
        }
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
    let scheduler = Scheduler::new(ModelRunner::new(BootstrapForwardModel::from_server_args(
        args,
    )?))
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
    let scheduler = Scheduler::new(ModelRunner::new(BootstrapForwardModel::from_server_args(
        args,
    )?))
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

pub fn try_build_bootstrap_prefill_http_router_service(
    args: &ServerArgs,
) -> Result<BootstrapPrefillHttpRouterService, ServerLaunchError> {
    validate_local_model_artifacts_if_present(args)?;
    let worker = KvTransferModelWorker::new(
        ModelRunner::new(BootstrapForwardModel::from_server_args(args)?),
        DecodeBootstrapRegistry::default(),
        FakeKvCacheTransferExecutor::default(),
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
    let model = BootstrapForwardModel::from_server_args(args)?;
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
    let mut worker =
        KvTransferModelWorker::new(ModelRunner::new(model), registry, transfer_executor)
            .with_decode_bootstrap_publisher(decode_bootstrap_publisher);
    if decode_side_bootstrap_only {
        worker = worker.with_decode_side_bootstrap_only();
    }
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
    model: &BootstrapForwardModel,
) -> Result<MooncakeKvCacheLayout, ServerLaunchError> {
    Ok(mooncake_kv_memory_from_bootstrap_model(model)?.prefill_layout(0))
}

#[cfg(not(feature = "mooncake-link"))]
fn launch_mooncake_decode_kv_layout(
    model: &BootstrapForwardModel,
) -> Result<MooncakeKvCacheLayout, ServerLaunchError> {
    Ok(mooncake_kv_memory_from_bootstrap_model(model)?.prefill_layout(0))
}

fn mooncake_kv_memory_from_bootstrap_model(
    model: &BootstrapForwardModel,
) -> Result<TransferableKvCacheMemory, ServerLaunchError> {
    match model {
        BootstrapForwardModel::GlmMoeDsa(model) => model
            .mooncake_kv_cache_memory()
            .map_err(|error| ServerLaunchError::KvCacheTransfer(error.to_string())),
        BootstrapForwardModel::UnsupportedLocalModelRuntime {
            model_path,
            model_type,
        } => Err(ServerLaunchError::UnsupportedMooncakeKvMemory {
            model_path: model_path.display().to_string(),
            model_type: model_type.clone(),
        }),
        BootstrapForwardModel::Space => Err(ServerLaunchError::UnsupportedMooncakeKvMemory {
            model_path: "<bootstrap>".to_string(),
            model_type: Some("space".to_string()),
        }),
        BootstrapForwardModel::CpuEmbeddingLm(_) => {
            Err(ServerLaunchError::UnsupportedMooncakeKvMemory {
                model_path: "<bootstrap>".to_string(),
                model_type: Some("cpu_embedding_lm".to_string()),
            })
        }
        BootstrapForwardModel::DeepSeekV4(_) => {
            Err(ServerLaunchError::UnsupportedMooncakeKvMemory {
                model_path: "<bootstrap>".to_string(),
                model_type: Some("deepseek_v4".to_string()),
            })
        }
    }
}

#[cfg(feature = "mooncake-link")]
fn prepare_linked_mooncake_kv_memory(
    model: &mut BootstrapForwardModel,
    engine: &SharedLinkedMooncakeTransferEngine,
    page_count: usize,
) -> Result<TransferableKvCacheMemory, ServerLaunchError> {
    model.reserve_mooncake_kv_cache_pages(page_count)?;
    let memory = mooncake_kv_memory_from_bootstrap_model(model)?;
    let mut buffers = memory
        .regions()
        .iter()
        .map(|region| MooncakeBufferEntry {
            addr: region.base_addr as *mut c_void,
            length: region.byte_len,
        })
        .collect::<Vec<_>>();
    engine.register_memory_batch(&mut buffers, "cpu:0")?;
    Ok(memory)
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
    let model = BootstrapForwardModel::from_server_args(args)?;
    let transfer_executor = MooncakeKvCacheTransferExecutor::new(
        UnlinkedMooncakeTransferEngine,
        launch_mooncake_prefill_kv_layout(&model)?,
        MooncakeTransferTarget { target_id: 0 },
    );
    try_build_bootstrap_mooncake_prefill_http_router_service(
        args,
        bootstrap_service,
        transfer_executor,
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
    let model = BootstrapForwardModel::from_server_args(args)?;
    let kv_cache_layout = launch_mooncake_decode_kv_layout(&model)?;
    let transfer_executor = MooncakeKvCacheTransferExecutor::new(
        UnlinkedMooncakeTransferEngine,
        kv_cache_layout,
        MooncakeTransferTarget { target_id: 0 },
    );
    try_build_bootstrap_pd_http_router_service_with_decode_publisher(
        args,
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
#[doc(hidden)]
pub fn try_build_launch_mooncake_decode_http_router_service_for_test(
    args: &ServerArgs,
    pd_config: &PdConfig,
) -> Result<
    BootstrapPdHttpRouterService<
        MooncakeKvCacheTransferExecutor<UnlinkedMooncakeTransferEngine>,
        MooncakeDecodeBootstrapPublisher,
    >,
    ServerLaunchError,
> {
    try_build_launch_mooncake_decode_http_router_service(args, pd_config)
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
    let model = BootstrapForwardModel::from_server_args(args)?;
    let transfer_executor = MooncakeKvCacheTransferExecutor::new(
        UnlinkedMooncakeTransferEngine,
        launch_mooncake_prefill_kv_layout(&model)?,
        MooncakeTransferTarget { target_id: 0 },
    );
    try_build_bootstrap_mooncake_prefill_grpc_router_service(
        args,
        bootstrap_service,
        transfer_executor,
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
    let model = BootstrapForwardModel::from_server_args(args)?;
    let kv_cache_layout = launch_mooncake_decode_kv_layout(&model)?;
    let transfer_executor = MooncakeKvCacheTransferExecutor::new(
        UnlinkedMooncakeTransferEngine,
        kv_cache_layout,
        MooncakeTransferTarget { target_id: 0 },
    );
    try_build_bootstrap_pd_grpc_router_service_with_decode_publisher(
        args,
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
    let mut model = BootstrapForwardModel::from_server_args(args)?;
    let engine_config = MooncakeTransferEngineConfig::from_pd_config_for_rank(
        prefill_mooncake_route_rank_ip(args),
        0,
        pd_config,
    );
    let engine = SharedLinkedMooncakeTransferEngine::new(&engine_config)?;
    let kv_memory =
        prepare_linked_mooncake_kv_memory(&mut model, &engine, args.num_reserved_decode_tokens)?;
    let target_resolver = MooncakeSessionTargetResolver::new(engine.clone(), Vec::new());
    let transfer_executor = MooncakeKvCacheTransferExecutor::with_target_resolver(
        engine,
        kv_memory.prefill_layout(0),
        target_resolver,
    );
    try_build_bootstrap_pd_http_router_service_from_model_with_decode_publisher(
        args,
        model,
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
    let mut model = BootstrapForwardModel::from_server_args(args)?;
    let engine_config = MooncakeTransferEngineConfig::from_pd_config_for_rank(
        prefill_mooncake_route_rank_ip(args),
        0,
        pd_config,
    );
    let engine = SharedLinkedMooncakeTransferEngine::new(&engine_config)?;
    let mooncake_session_id = engine.local_endpoint()?;
    let kv_memory =
        prepare_linked_mooncake_kv_memory(&mut model, &engine, args.num_reserved_decode_tokens)?;
    let target_resolver = MooncakeSessionTargetResolver::new(engine.clone(), Vec::new());
    let kv_cache_layout = kv_memory.prefill_layout(0);
    let transfer_executor = MooncakeKvCacheTransferExecutor::with_target_resolver(
        engine,
        kv_cache_layout,
        target_resolver,
    );
    try_build_bootstrap_pd_http_router_service_from_model_with_decode_publisher(
        args,
        model,
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
    let mut model = BootstrapForwardModel::from_server_args(args)?;
    let engine_config = MooncakeTransferEngineConfig::from_pd_config_for_rank(
        prefill_mooncake_route_rank_ip(args),
        0,
        pd_config,
    );
    let engine = SharedLinkedMooncakeTransferEngine::new(&engine_config)?;
    let kv_memory =
        prepare_linked_mooncake_kv_memory(&mut model, &engine, args.num_reserved_decode_tokens)?;
    let target_resolver = MooncakeSessionTargetResolver::new(engine.clone(), Vec::new());
    let transfer_executor = MooncakeKvCacheTransferExecutor::with_target_resolver(
        engine,
        kv_memory.prefill_layout(0),
        target_resolver,
    );
    try_build_bootstrap_pd_grpc_router_service_from_model_with_decode_publisher(
        args,
        model,
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
    let mut model = BootstrapForwardModel::from_server_args(args)?;
    let engine_config = MooncakeTransferEngineConfig::from_pd_config_for_rank(
        prefill_mooncake_route_rank_ip(args),
        0,
        pd_config,
    );
    let engine = SharedLinkedMooncakeTransferEngine::new(&engine_config)?;
    let mooncake_session_id = engine.local_endpoint()?;
    let kv_memory =
        prepare_linked_mooncake_kv_memory(&mut model, &engine, args.num_reserved_decode_tokens)?;
    let target_resolver = MooncakeSessionTargetResolver::new(engine.clone(), Vec::new());
    let kv_cache_layout = kv_memory.prefill_layout(0);
    let transfer_executor = MooncakeKvCacheTransferExecutor::with_target_resolver(
        engine,
        kv_cache_layout,
        target_resolver,
    );
    try_build_bootstrap_pd_grpc_router_service_from_model_with_decode_publisher(
        args,
        model,
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
    let model = BootstrapForwardModel::from_server_args(args)?;
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
    let mut worker =
        KvTransferModelWorker::new(ModelRunner::new(model), registry, transfer_executor)
            .with_decode_bootstrap_publisher(decode_bootstrap_publisher);
    if decode_side_bootstrap_only {
        worker = worker.with_decode_side_bootstrap_only();
    }
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
    artifacts.validate_checkpoint_for_supported_model()?;
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
    let addr = grpc_listen_addr(&args)?;
    match pd_config.mode {
        DisaggregationMode::Null => {
            let service = try_build_bootstrap_grpc_router_service(&args)?;
            serve_grpc_router_with_shutdown(addr, service, true, shutdown).await?;
        }
        DisaggregationMode::Decode if pd_config.transfer_backend == TransferBackend::Fake => {
            let service = try_build_bootstrap_pd_grpc_router_service(
                &args,
                DecodeBootstrapRegistry::default(),
                FakeKvCacheTransferExecutor::default(),
            )?;
            serve_grpc_router_with_shutdown(addr, service, true, shutdown).await?;
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
                bootstrap_addr,
                bootstrap_service,
                zmq_endpoints,
                shutdown,
            )
            .await?;
        }
        DisaggregationMode::Decode if pd_config.transfer_backend == TransferBackend::Mooncake => {
            let service = try_build_launch_mooncake_decode_grpc_router_service(&args, &pd_config)?;
            serve_grpc_router_with_shutdown(addr, service, true, shutdown).await?;
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
    let addr = http_listen_addr(&args)?;
    match pd_config.mode {
        DisaggregationMode::Null => {
            let service = try_build_bootstrap_http_router_service(&args)?;
            serve_http_router_with_shutdown(addr, service, shutdown).await?;
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
            )?;
            serve_prefill_http_and_bootstrap(
                addr,
                service,
                bootstrap_addr,
                bootstrap_service,
                zmq_endpoints,
                shutdown,
            )
            .await?;
        }
        DisaggregationMode::Decode if pd_config.transfer_backend == TransferBackend::Mooncake => {
            let service = try_build_launch_mooncake_decode_http_router_service(&args, &pd_config)?;
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

async fn serve_prefill_http_and_bootstrap<T, W, F>(
    http_addr: SocketAddr,
    http_service: HttpRouterService<T, W>,
    bootstrap_addr: SocketAddr,
    bootstrap_service: PrefillBootstrapService,
    zmq_endpoints: Vec<String>,
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
    bootstrap_tasks.spawn(serve_prefill_bootstrap_with_shutdown(
        bootstrap_addr,
        bootstrap_service.clone(),
        watch_shutdown(shutdown_rx.clone()),
    ));
    if !zmq_endpoints.is_empty() {
        bootstrap_tasks.spawn(serve_mooncake_bootstrap_zmq_endpoints_with_shutdown(
            zmq_endpoints,
            bootstrap_service,
            watch_shutdown(shutdown_rx),
        ));
    }

    tokio::select! {
        _ = shutdown => {
            let _ = shutdown_tx.send(true);
            join_http_task(http_task).await?;
            join_bootstrap_tasks(bootstrap_tasks).await?;
            Ok(())
        }
        result = &mut http_task => {
            let _ = shutdown_tx.send(true);
            join_bootstrap_tasks(bootstrap_tasks).await?;
            result.map_err(|error| ServerLaunchError::ServerTaskJoin(error.to_string()))??;
            Ok(())
        }
        result = bootstrap_tasks.join_next() => {
            let _ = shutdown_tx.send(true);
            join_http_task(http_task).await?;
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

async fn serve_prefill_grpc_and_bootstrap<T, W, F>(
    grpc_addr: SocketAddr,
    grpc_service: GrpcRouterService<T, W>,
    bootstrap_addr: SocketAddr,
    bootstrap_service: PrefillBootstrapService,
    zmq_endpoints: Vec<String>,
    shutdown: F,
) -> Result<(), ServerLaunchError>
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
    F: Future<Output = ()> + Send + 'static,
{
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let mut grpc_task = tokio::spawn(serve_grpc_router_with_shutdown(
        grpc_addr,
        grpc_service,
        true,
        watch_shutdown(shutdown_rx.clone()),
    ));
    let mut bootstrap_tasks = tokio::task::JoinSet::new();
    bootstrap_tasks.spawn(serve_prefill_bootstrap_with_shutdown(
        bootstrap_addr,
        bootstrap_service.clone(),
        watch_shutdown(shutdown_rx.clone()),
    ));
    if !zmq_endpoints.is_empty() {
        bootstrap_tasks.spawn(serve_mooncake_bootstrap_zmq_endpoints_with_shutdown(
            zmq_endpoints,
            bootstrap_service,
            watch_shutdown(shutdown_rx),
        ));
    }

    tokio::select! {
        _ = shutdown => {
            let _ = shutdown_tx.send(true);
            join_grpc_task(grpc_task).await?;
            join_bootstrap_tasks(bootstrap_tasks).await?;
            Ok(())
        }
        result = &mut grpc_task => {
            let _ = shutdown_tx.send(true);
            join_bootstrap_tasks(bootstrap_tasks).await?;
            result.map_err(|error| ServerLaunchError::ServerTaskJoin(error.to_string()))??;
            Ok(())
        }
        result = bootstrap_tasks.join_next() => {
            let _ = shutdown_tx.send(true);
            join_grpc_task(grpc_task).await?;
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
