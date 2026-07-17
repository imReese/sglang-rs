use std::collections::BTreeMap;
#[cfg(feature = "mooncake-link")]
use std::ffi::{CStr, CString};
use std::ffi::{NulError, c_char, c_int, c_void};
use std::fmt;
use std::fs;
use std::path::Path;
#[cfg(feature = "mooncake-link")]
use std::sync::{Arc, Mutex};

use sha2::{Digest, Sha256};

pub use nexus_transfer::{
    KvCacheMemoryLocation, KvCacheMemoryProvider, TransferableKvCacheMemory,
    TransferableKvCacheMemoryError, TransferableKvCacheRegion,
};

use crate::cache::CachePageId;
use crate::cli::{ServerArgs, ZmqPortRange};
use crate::model_artifacts::{HfModelConfig, resolve_model_path};
use crate::model_executor::{ModelRunner, ModelWorkerBatch};
use crate::scheduler::{ForwardMode, ScheduleBatch, ScheduledRequest};
use crate::types::{BootstrapRoom, DisaggregatedParams, RequestId};
use crate::worker::{
    BatchGeneratedTokens, DecodeRequestState, FallibleModelWorker, WorkerExecutionError,
    WorkerWeightUpdateRequest,
};

#[cfg(feature = "mooncake-link")]
#[link(name = "transfer_engine", kind = "static")]
unsafe extern "C" {}

#[cfg(feature = "mooncake-link")]
#[link(name = "mooncake_common", kind = "static")]
unsafe extern "C" {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DisaggregationMode {
    Null,
    Prefill,
    Decode,
}

impl DisaggregationMode {
    fn parse(value: &str) -> Result<Self, PdConfigError> {
        match value {
            "null" => Ok(Self::Null),
            "prefill" => Ok(Self::Prefill),
            "decode" => Ok(Self::Decode),
            other => Err(PdConfigError::InvalidDisaggregationMode(other.to_string())),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransferBackend {
    Mooncake,
    Nixl,
    Ascend,
    Fake,
    Mori,
}

impl TransferBackend {
    fn parse(value: &str) -> Result<ParsedTransferBackend, PdConfigError> {
        match value {
            "mooncake" => Ok(ParsedTransferBackend {
                backend: Self::Mooncake,
                force_tcp_transport: false,
            }),
            "mooncake_tcp" => Ok(ParsedTransferBackend {
                backend: Self::Mooncake,
                force_tcp_transport: true,
            }),
            "nixl" => Ok(ParsedTransferBackend {
                backend: Self::Nixl,
                force_tcp_transport: false,
            }),
            "ascend" => Ok(ParsedTransferBackend {
                backend: Self::Ascend,
                force_tcp_transport: false,
            }),
            "fake" => Ok(ParsedTransferBackend {
                backend: Self::Fake,
                force_tcp_transport: false,
            }),
            "mori" => Ok(ParsedTransferBackend {
                backend: Self::Mori,
                force_tcp_transport: false,
            }),
            other => Err(PdConfigError::InvalidTransferBackend(other.to_string())),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ParsedTransferBackend {
    backend: TransferBackend,
    force_tcp_transport: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvCacheDtype {
    Auto,
    Bfloat16,
    Fp8E4M3,
    Fp8E5M2,
    Fp4E2M1,
}

impl KvCacheDtype {
    fn parse(value: &str) -> Result<Self, PdConfigError> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "bf16" | "bfloat16" => Ok(Self::Bfloat16),
            "fp8_e4m3" => Ok(Self::Fp8E4M3),
            "fp8_e5m2" => Ok(Self::Fp8E5M2),
            "fp4_e2m1" => Ok(Self::Fp4E2M1),
            _ => Err(PdConfigError::InvalidKvCacheDtype(value.to_string())),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Bfloat16 => "bfloat16",
            Self::Fp8E4M3 => "fp8_e4m3",
            Self::Fp8E5M2 => "fp8_e5m2",
            Self::Fp4E2M1 => "fp4_e2m1",
        }
    }

    pub fn bytes_per_element(&self) -> Option<usize> {
        match self {
            Self::Auto | Self::Bfloat16 => Some(2),
            Self::Fp8E4M3 | Self::Fp8E5M2 => Some(1),
            Self::Fp4E2M1 => None,
        }
    }

    pub fn runtime_storage_dtype(self) -> Self {
        match self {
            Self::Auto => Self::Bfloat16,
            dtype => dtype,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvCacheRuntimeLayout {
    pub dtype: KvCacheDtype,
    pub page_size: usize,
    pub num_layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub kv_tensors_per_token: usize,
    pub bytes_per_token: usize,
    pub page_size_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvCacheModelLayout {
    pub num_layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub kv_tensors_per_token: usize,
    pub bytes_per_token_per_layer: Option<usize>,
}

impl KvCacheModelLayout {
    pub fn multi_tensor(
        num_layers: usize,
        kv_heads: usize,
        head_dim: usize,
        kv_tensors_per_token: usize,
    ) -> Self {
        Self {
            num_layers,
            kv_heads,
            head_dim,
            kv_tensors_per_token,
            bytes_per_token_per_layer: None,
        }
    }

    pub fn packed_bytes_per_layer(num_layers: usize, bytes_per_token_per_layer: usize) -> Self {
        Self {
            num_layers,
            kv_heads: 1,
            head_dim: bytes_per_token_per_layer,
            kv_tensors_per_token: 1,
            bytes_per_token_per_layer: Some(bytes_per_token_per_layer),
        }
    }

    pub fn elements_per_token(&self) -> Option<usize> {
        self.num_layers
            .checked_mul(self.kv_tensors_per_token)
            .and_then(|value| value.checked_mul(self.kv_heads))
            .and_then(|value| value.checked_mul(self.head_dim))
    }

    pub fn token_size_bytes(&self, dtype: KvCacheDtype) -> Result<usize, PdConfigError> {
        if let Some(bytes_per_token_per_layer) = self.bytes_per_token_per_layer {
            return self
                .num_layers
                .checked_mul(bytes_per_token_per_layer)
                .ok_or(PdConfigError::KvCacheLayoutOverflow);
        }

        let elements_per_token = self
            .elements_per_token()
            .ok_or(PdConfigError::KvCacheLayoutOverflow)?;
        let bytes_per_element = dtype
            .bytes_per_element()
            .ok_or(PdConfigError::KvCacheDtypeRequiresModelMetadata(dtype))?;

        elements_per_token
            .checked_mul(bytes_per_element)
            .ok_or(PdConfigError::KvCacheLayoutOverflow)
    }

    fn from_model_path(model_path: &str) -> Result<Option<Self>, PdConfigError> {
        let resolved_model_path = resolve_model_path(Path::new(model_path));
        if let Some(layout) = Self::from_resolved_model_path(&resolved_model_path)? {
            return Ok(Some(layout));
        }

        if !looks_like_hf_model_id(model_path) {
            return Ok(None);
        }

        let config = HfModelConfig::from_model_path(model_path).map_err(|error| {
            PdConfigError::InvalidModelConfig(format!(
                "failed to load Hugging Face config for {model_path}: {error}"
            ))
        })?;
        Self::from_hf_config(&config)
    }

    pub fn from_model_path_with_hf_cache(
        model_path: &str,
        hub_cache: impl AsRef<Path>,
    ) -> Result<Option<Self>, PdConfigError> {
        let model_path =
            crate::model_artifacts::resolve_model_path_from_hf_cache(model_path, hub_cache)
                .unwrap_or_else(|| Path::new(model_path).to_path_buf());
        Self::from_resolved_model_path(&model_path)
    }

    pub fn from_hf_config(config: &HfModelConfig) -> Result<Option<Self>, PdConfigError> {
        let Some(num_layers) = config.num_hidden_layers else {
            return Ok(None);
        };

        if config.qk_nope_head_dim.is_some() || config.qk_rope_head_dim.is_some() {
            let qk_nope_head_dim =
                required_hf_config_usize(config.qk_nope_head_dim, "qk_nope_head_dim")?;
            let qk_rope_head_dim =
                required_hf_config_usize(config.qk_rope_head_dim, "qk_rope_head_dim")?;
            return Self::packed_mla(num_layers, qk_nope_head_dim, qk_rope_head_dim).map(Some);
        }

        let num_attention_heads =
            required_hf_config_usize(config.num_attention_heads, "num_attention_heads")?;
        let kv_heads = config.num_key_value_heads.unwrap_or(num_attention_heads);
        let head_dim = match config.head_dim {
            Some(head_dim) => head_dim,
            None => {
                let hidden_size = required_hf_config_usize(config.hidden_size, "hidden_size")?;
                if num_attention_heads == 0 || hidden_size % num_attention_heads != 0 {
                    return Err(PdConfigError::InvalidModelConfig(format!(
                        "hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
                    )));
                }
                hidden_size / num_attention_heads
            }
        };

        Ok(Some(Self::multi_tensor(num_layers, kv_heads, head_dim, 2)))
    }

    fn from_resolved_model_path(model_path: &Path) -> Result<Option<Self>, PdConfigError> {
        if !model_path.is_dir() {
            return Ok(None);
        }

        let config_path = model_path.join("config.json");
        if !config_path.is_file() {
            return Ok(None);
        }

        let config = fs::read_to_string(&config_path).map_err(|error| {
            PdConfigError::InvalidModelConfig(format!(
                "failed to read {}: {error}",
                config_path.display()
            ))
        })?;
        let config: serde_json::Value = serde_json::from_str(&config).map_err(|error| {
            PdConfigError::InvalidModelConfig(format!(
                "failed to parse {}: {error}",
                config_path.display()
            ))
        })?;

        Self::from_hf_config_value(&config)
    }

    fn from_hf_config_value(config: &serde_json::Value) -> Result<Option<Self>, PdConfigError> {
        let Some(num_layers) = read_usize_field(config, "num_hidden_layers")? else {
            return Ok(None);
        };

        if config.get("qk_nope_head_dim").is_some() || config.get("qk_rope_head_dim").is_some() {
            let qk_nope_head_dim = required_usize_field(config, "qk_nope_head_dim")?;
            let qk_rope_head_dim = required_usize_field(config, "qk_rope_head_dim")?;
            return Self::packed_mla(num_layers, qk_nope_head_dim, qk_rope_head_dim).map(Some);
        }

        let num_attention_heads = required_usize_field(config, "num_attention_heads")?;
        let kv_heads =
            read_usize_field(config, "num_key_value_heads")?.unwrap_or(num_attention_heads);
        let head_dim = match read_usize_field(config, "head_dim")? {
            Some(head_dim) => head_dim,
            None => {
                let hidden_size = required_usize_field(config, "hidden_size")?;
                if num_attention_heads == 0 || hidden_size % num_attention_heads != 0 {
                    return Err(PdConfigError::InvalidModelConfig(format!(
                        "hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
                    )));
                }
                hidden_size / num_attention_heads
            }
        };

        Ok(Some(Self::multi_tensor(num_layers, kv_heads, head_dim, 2)))
    }

    pub fn packed_mla(
        num_layers: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
    ) -> Result<Self, PdConfigError> {
        if !qk_nope_head_dim.is_multiple_of(64) {
            return Err(PdConfigError::InvalidModelConfig(format!(
                "qk_nope_head_dim must be divisible by 64 for packed MLA KV layout: {qk_nope_head_dim}"
            )));
        }

        let rope_bytes = qk_rope_head_dim
            .checked_mul(2)
            .ok_or(PdConfigError::KvCacheLayoutOverflow)?;
        let scale_bytes = qk_nope_head_dim / 64;
        let bytes_per_token_per_layer = qk_nope_head_dim
            .checked_add(rope_bytes)
            .and_then(|value| value.checked_add(scale_bytes))
            .and_then(|value| value.checked_add(1))
            .ok_or(PdConfigError::KvCacheLayoutOverflow)?;

        Ok(Self::packed_bytes_per_layer(
            num_layers,
            bytes_per_token_per_layer,
        ))
    }
}

fn required_hf_config_usize(
    value: Option<usize>,
    field: &'static str,
) -> Result<usize, PdConfigError> {
    value
        .ok_or_else(|| PdConfigError::InvalidModelConfig(format!("missing required field {field}")))
}

fn read_usize_field(
    config: &serde_json::Value,
    field: &'static str,
) -> Result<Option<usize>, PdConfigError> {
    let Some(value) = config.get(field) else {
        return Ok(None);
    };

    let Some(value) = value.as_u64() else {
        return Err(PdConfigError::InvalidModelConfig(format!(
            "{field} must be an unsigned integer"
        )));
    };

    usize::try_from(value).map(Some).map_err(|_| {
        PdConfigError::InvalidModelConfig(format!(
            "{field} is too large for this platform: {value}"
        ))
    })
}

fn looks_like_hf_model_id(model_path: &str) -> bool {
    model_path.contains('/')
        && !model_path.starts_with('/')
        && !model_path.starts_with('-')
        && !model_path.contains('\\')
}

fn required_usize_field(
    config: &serde_json::Value,
    field: &'static str,
) -> Result<usize, PdConfigError> {
    read_usize_field(config, field)?
        .ok_or_else(|| PdConfigError::InvalidModelConfig(format!("missing required field {field}")))
}

#[derive(Clone, Debug, PartialEq)]
pub struct PdConfig {
    pub mode: DisaggregationMode,
    pub transfer_backend: TransferBackend,
    pub force_tcp_transport: bool,
    pub bootstrap_port: u16,
    pub mooncake_rpc_port: Option<u16>,
    pub ib_device: Option<String>,
    pub decode_enable_radix_cache: bool,
    pub decode_enable_offload_kvcache: bool,
    pub num_reserved_decode_tokens: usize,
    pub decode_polling_interval: usize,
    pub kv_cache_dtype: KvCacheDtype,
    pub kv_cache_model_layout: Option<KvCacheModelLayout>,
    pub page_size: usize,
    pub base_gpu_id: usize,
    pub gpu_id_step: usize,
    pub tp_size: usize,
    pub dp_size: usize,
    pub nnodes: usize,
    pub node_rank: usize,
    pub dist_init_addr: Option<String>,
    pub trust_remote_code: bool,
    pub enable_dp_attention: bool,
    pub enable_dp_lm_head: bool,
    pub disable_cuda_graph: bool,
    pub moe_a2a_backend: Option<String>,
    pub moe_dense_tp_size: Option<usize>,
    pub mem_fraction_static: Option<f32>,
    pub max_running_requests: Option<usize>,
    pub max_prefill_tokens: Option<usize>,
    pub max_total_tokens: Option<usize>,
    pub disaggregation_zmq_ports: Option<ZmqPortRange>,
    pub deepep_config: Option<serde_json::Value>,
    pub deepep_mode: Option<String>,
    pub attention_backend: Option<String>,
    pub enable_nsa_prefill_context_parallel: bool,
    pub nsa_prefill_backend: Option<String>,
    pub nsa_prefill_cp_mode: Option<String>,
    pub speculative_algorithm: Option<String>,
    pub speculative_eagle_topk: Option<usize>,
    pub speculative_num_draft_tokens: Option<usize>,
    pub speculative_num_steps: Option<usize>,
    pub chunked_prefill_size: Option<usize>,
    pub decode_log_interval: Option<usize>,
    pub disable_overlap_schedule: bool,
    pub model_loader_extra_config: Option<serde_json::Value>,
    pub tokenizer_worker_num: Option<usize>,
    pub allow_auto_truncate: bool,
    pub collect_tokens_histogram: bool,
    pub enable_cache_report: bool,
    pub enable_metrics: bool,
    pub disable_radix_cache: bool,
    pub tool_call_parser: Option<String>,
}

impl PdConfig {
    pub fn from_server_args(args: &ServerArgs) -> Result<Self, PdConfigError> {
        let kv_cache_model_layout = Self::model_layout_from_server_args(args)?;
        Self::from_server_args_with_model_layout(args, kv_cache_model_layout)
    }

    pub fn from_server_args_with_hf_cache(
        args: &ServerArgs,
        hub_cache: impl AsRef<Path>,
    ) -> Result<Self, PdConfigError> {
        let kv_cache_model_layout = match (
            args.kv_cache_num_layers,
            args.kv_cache_kv_heads,
            args.kv_cache_head_dim,
        ) {
            (None, None, None) => {
                KvCacheModelLayout::from_model_path_with_hf_cache(&args.model_path, hub_cache)?
            }
            (Some(num_layers), Some(kv_heads), Some(head_dim)) => Some(
                KvCacheModelLayout::multi_tensor(num_layers, kv_heads, head_dim, 2),
            ),
            _ => return Err(PdConfigError::IncompleteKvCacheModelLayout),
        };

        Self::from_server_args_with_model_layout(args, kv_cache_model_layout)
    }

    fn model_layout_from_server_args(
        args: &ServerArgs,
    ) -> Result<Option<KvCacheModelLayout>, PdConfigError> {
        match (
            args.kv_cache_num_layers,
            args.kv_cache_kv_heads,
            args.kv_cache_head_dim,
        ) {
            (None, None, None) => KvCacheModelLayout::from_model_path(&args.model_path),
            (Some(num_layers), Some(kv_heads), Some(head_dim)) => Ok(Some(
                KvCacheModelLayout::multi_tensor(num_layers, kv_heads, head_dim, 2),
            )),
            _ => Err(PdConfigError::IncompleteKvCacheModelLayout),
        }
    }

    fn from_server_args_with_model_layout(
        args: &ServerArgs,
        kv_cache_model_layout: Option<KvCacheModelLayout>,
    ) -> Result<Self, PdConfigError> {
        let mode = DisaggregationMode::parse(&args.disaggregation_mode)?;
        let backend = TransferBackend::parse(&args.disaggregation_transfer_backend)?;
        let kv_cache_dtype = KvCacheDtype::parse(&args.kv_cache_dtype)?;

        Ok(Self {
            mode,
            transfer_backend: backend.backend,
            force_tcp_transport: backend.force_tcp_transport,
            bootstrap_port: args.disaggregation_bootstrap_port,
            mooncake_rpc_port: args.disaggregation_mooncake_rpc_port,
            ib_device: if backend.force_tcp_transport {
                None
            } else {
                args.disaggregation_ib_device.clone()
            },
            decode_enable_radix_cache: args.disaggregation_decode_enable_radix_cache,
            decode_enable_offload_kvcache: args.disaggregation_decode_enable_offload_kvcache,
            num_reserved_decode_tokens: args.num_reserved_decode_tokens,
            decode_polling_interval: args.disaggregation_decode_polling_interval,
            kv_cache_dtype,
            kv_cache_model_layout,
            page_size: args.page_size,
            base_gpu_id: args.base_gpu_id,
            gpu_id_step: args.gpu_id_step,
            tp_size: args.tp_size,
            dp_size: args.dp_size,
            nnodes: args.nnodes,
            node_rank: args.node_rank,
            dist_init_addr: args.dist_init_addr.clone(),
            trust_remote_code: args.trust_remote_code,
            enable_dp_attention: args.enable_dp_attention,
            enable_dp_lm_head: args.enable_dp_lm_head,
            disable_cuda_graph: args.disable_cuda_graph,
            moe_a2a_backend: args.moe_a2a_backend.clone(),
            moe_dense_tp_size: args.moe_dense_tp_size,
            mem_fraction_static: args.mem_fraction_static,
            max_running_requests: args.max_running_requests,
            max_prefill_tokens: args.max_prefill_tokens,
            max_total_tokens: args.max_total_tokens,
            disaggregation_zmq_ports: args.disaggregation_zmq_ports,
            deepep_config: args.deepep_config.clone(),
            deepep_mode: args.deepep_mode.clone(),
            attention_backend: args.attention_backend.clone(),
            enable_nsa_prefill_context_parallel: args.enable_nsa_prefill_context_parallel,
            nsa_prefill_backend: args.nsa_prefill_backend.clone(),
            nsa_prefill_cp_mode: args.nsa_prefill_cp_mode.clone(),
            speculative_algorithm: args.speculative_algorithm.clone(),
            speculative_eagle_topk: args.speculative_eagle_topk,
            speculative_num_draft_tokens: args.speculative_num_draft_tokens,
            speculative_num_steps: args.speculative_num_steps,
            chunked_prefill_size: args.chunked_prefill_size,
            decode_log_interval: args.decode_log_interval,
            disable_overlap_schedule: args.disable_overlap_schedule,
            model_loader_extra_config: args.model_loader_extra_config.clone(),
            tokenizer_worker_num: args.tokenizer_worker_num,
            allow_auto_truncate: args.allow_auto_truncate,
            collect_tokens_histogram: args.collect_tokens_histogram,
            enable_cache_report: args.enable_cache_report,
            enable_metrics: args.enable_metrics,
            disable_radix_cache: args.disable_radix_cache,
            tool_call_parser: args.tool_call_parser.clone(),
        })
    }

    pub fn kv_cache_runtime_layout(&self) -> Result<Option<KvCacheRuntimeLayout>, PdConfigError> {
        let Some(model_layout) = self.kv_cache_model_layout else {
            return Ok(None);
        };

        let runtime_dtype = self.kv_cache_dtype.runtime_storage_dtype();
        let bytes_per_token = model_layout.token_size_bytes(runtime_dtype)?;
        let page_size_bytes = self
            .page_size
            .checked_mul(bytes_per_token)
            .ok_or(PdConfigError::KvCacheLayoutOverflow)?;

        Ok(Some(KvCacheRuntimeLayout {
            dtype: runtime_dtype,
            page_size: self.page_size,
            num_layers: model_layout.num_layers,
            kv_heads: model_layout.kv_heads,
            head_dim: model_layout.head_dim,
            kv_tensors_per_token: model_layout.kv_tensors_per_token,
            bytes_per_token,
            page_size_bytes,
        }))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PdConfigError {
    InvalidDisaggregationMode(String),
    InvalidTransferBackend(String),
    InvalidKvCacheDtype(String),
    InvalidModelConfig(String),
    IncompleteKvCacheModelLayout,
    MissingMooncakeKvCacheModelLayout,
    KvCacheDtypeRequiresModelMetadata(KvCacheDtype),
    KvCacheLayoutOverflow,
}

impl fmt::Display for PdConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDisaggregationMode(mode) => {
                write!(formatter, "invalid disaggregation mode: {mode}")
            }
            Self::InvalidTransferBackend(backend) => {
                write!(
                    formatter,
                    "invalid disaggregation transfer backend: {backend}"
                )
            }
            Self::InvalidKvCacheDtype(dtype) => {
                write!(formatter, "invalid kv cache dtype: {dtype}")
            }
            Self::InvalidModelConfig(message) => {
                write!(formatter, "invalid model config: {message}")
            }
            Self::IncompleteKvCacheModelLayout => formatter
                .write_str("kv cache model layout requires num layers, KV heads, and head dim"),
            Self::MissingMooncakeKvCacheModelLayout => formatter.write_str(
                "mooncake decode requires kv cache model layout; provide \
                     --kv-cache-num-layers, --kv-cache-kv-heads, and --kv-cache-head-dim, \
                     or use a local model path / Hugging Face cache snapshot with config.json metadata",
            ),
            Self::KvCacheDtypeRequiresModelMetadata(dtype) => {
                write!(
                    formatter,
                    "kv cache dtype requires model metadata for byte width: {dtype:?}"
                )
            }
            Self::KvCacheLayoutOverflow => {
                formatter.write_str("kv cache layout byte size overflow")
            }
        }
    }
}

impl std::error::Error for PdConfigError {}

#[derive(Debug, Eq, PartialEq)]
pub enum MooncakeError {
    InteriorNul,
    UnavailableWithoutLink,
    UnsupportedMemoryLocation(KvCacheMemoryLocation),
    EngineCreateFailed,
    LocalEndpointQueryFailed(i32),
    LocalEndpointUtf8,
    TransportInstallFailed(String),
    RegisterMemoryFailed(i32),
    UnregisterMemoryFailed(i32),
    OpenSegmentFailed(String),
    SubmitTransferFailed(i32),
    StatusQueryFailed(i32),
    FreeBatchFailed(i32),
}

impl From<NulError> for MooncakeError {
    fn from(_: NulError) -> Self {
        Self::InteriorNul
    }
}

impl fmt::Display for MooncakeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InteriorNul => formatter.write_str("mooncake string contains interior nul byte"),
            Self::UnavailableWithoutLink => formatter.write_str(
                "mooncake transfer engine requires building sglang-srt with the mooncake-link feature",
            ),
            Self::UnsupportedMemoryLocation(location) => write!(
                formatter,
                "mooncake transfer engine does not support KV memory location {location:?}"
            ),
            Self::EngineCreateFailed => {
                formatter.write_str("mooncake transfer engine create failed")
            }
            Self::LocalEndpointQueryFailed(code) => {
                write!(formatter, "mooncake local endpoint query failed: {code}")
            }
            Self::LocalEndpointUtf8 => {
                formatter.write_str("mooncake local endpoint is not valid UTF-8")
            }
            Self::TransportInstallFailed(protocol) => {
                write!(formatter, "mooncake transport install failed: {protocol}")
            }
            Self::RegisterMemoryFailed(code) => {
                write!(formatter, "mooncake memory register failed: {code}")
            }
            Self::UnregisterMemoryFailed(code) => {
                write!(formatter, "mooncake memory unregister failed: {code}")
            }
            Self::OpenSegmentFailed(segment) => {
                write!(formatter, "mooncake open segment failed: {segment}")
            }
            Self::SubmitTransferFailed(code) => {
                write!(formatter, "mooncake transfer submit failed: {code}")
            }
            Self::StatusQueryFailed(code) => {
                write!(formatter, "mooncake status query failed: {code}")
            }
            Self::FreeBatchFailed(code) => write!(formatter, "mooncake free batch failed: {code}"),
        }
    }
}

impl std::error::Error for MooncakeError {}

pub const MOONCAKE_P2P_HANDSHAKE_METADATA: &str = "P2PHANDSHAKE";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MooncakeTransferEngineConfig {
    pub hostname: String,
    pub gpu_id: usize,
    pub rpc_port: u16,
    pub session_id: String,
    pub metadata_server: String,
    pub protocol: String,
    pub device_name: String,
}

impl MooncakeTransferEngineConfig {
    pub fn from_pd_config(hostname: impl Into<String>, gpu_id: usize, config: &PdConfig) -> Self {
        let hostname = hostname.into();
        let rpc_port = config.mooncake_rpc_port.unwrap_or(0);
        let session_id = format!("{hostname}:{rpc_port}");
        Self {
            hostname,
            gpu_id,
            rpc_port,
            session_id,
            metadata_server: MOONCAKE_P2P_HANDSHAKE_METADATA.to_string(),
            protocol: if config.force_tcp_transport {
                "tcp".to_string()
            } else {
                "rdma".to_string()
            },
            device_name: config.ib_device.clone().unwrap_or_default(),
        }
    }

    pub fn from_pd_config_for_rank(
        hostname: impl Into<String>,
        local_rank: usize,
        config: &PdConfig,
    ) -> Self {
        Self::from_pd_config(
            hostname,
            config.base_gpu_id + local_rank * config.gpu_id_step,
            config,
        )
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum KvPoll {
    Failed = 0,
    Bootstrapping = 1,
    WaitingForInput = 2,
    Transferring = 3,
    Success = 4,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodeBootstrapSession {
    request_id: RequestId,
    disaggregated_params: DisaggregatedParams,
    data_parallel_rank: i32,
    status: KvPoll,
}

impl DecodeBootstrapSession {
    pub fn new(
        request_id: RequestId,
        disaggregated_params: DisaggregatedParams,
        data_parallel_rank: i32,
    ) -> Self {
        Self {
            request_id,
            disaggregated_params,
            data_parallel_rank,
            status: KvPoll::Bootstrapping,
        }
    }

    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    pub fn disaggregated_params(&self) -> &DisaggregatedParams {
        &self.disaggregated_params
    }

    pub fn data_parallel_rank(&self) -> i32 {
        self.data_parallel_rank
    }

    pub fn status(&self) -> KvPoll {
        self.status
    }

    fn set_status(&mut self, status: KvPoll) {
        self.status = status;
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecodeBootstrapRegistryError {
    DuplicateBootstrapRoom(BootstrapRoom),
    MissingBootstrapRoom(BootstrapRoom),
}

impl fmt::Display for DecodeBootstrapRegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateBootstrapRoom(room) => {
                write!(formatter, "duplicate decode bootstrap room: {room}")
            }
            Self::MissingBootstrapRoom(room) => {
                write!(formatter, "missing decode bootstrap room: {room}")
            }
        }
    }
}

impl std::error::Error for DecodeBootstrapRegistryError {}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DecodeBootstrapRegistry {
    sessions_by_room: BTreeMap<BootstrapRoom, DecodeBootstrapSession>,
}

impl DecodeBootstrapRegistry {
    pub fn register(
        &mut self,
        session: DecodeBootstrapSession,
    ) -> Result<(), DecodeBootstrapRegistryError> {
        let room = session.disaggregated_params.bootstrap_room;
        if self.sessions_by_room.contains_key(&room) {
            return Err(DecodeBootstrapRegistryError::DuplicateBootstrapRoom(room));
        }

        self.sessions_by_room.insert(room, session);
        Ok(())
    }

    pub fn get(&self, bootstrap_room: BootstrapRoom) -> Option<&DecodeBootstrapSession> {
        self.sessions_by_room.get(&bootstrap_room)
    }

    pub fn query_data_parallel_rank(&self, bootstrap_room: BootstrapRoom) -> Option<i32> {
        self.get(bootstrap_room)
            .map(DecodeBootstrapSession::data_parallel_rank)
    }

    pub fn update_status(
        &mut self,
        bootstrap_room: BootstrapRoom,
        status: KvPoll,
    ) -> Result<(), DecodeBootstrapRegistryError> {
        let session = self.sessions_by_room.get_mut(&bootstrap_room).ok_or(
            DecodeBootstrapRegistryError::MissingBootstrapRoom(bootstrap_room),
        )?;
        session.set_status(status);
        Ok(())
    }

    pub fn remove(&mut self, bootstrap_room: BootstrapRoom) -> Option<DecodeBootstrapSession> {
        self.sessions_by_room.remove(&bootstrap_room)
    }

    pub fn len(&self) -> usize {
        self.sessions_by_room.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sessions_by_room.is_empty()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvCachePageIndex(usize);

impl KvCachePageIndex {
    pub fn as_usize(&self) -> usize {
        self.0
    }
}

impl From<usize> for KvCachePageIndex {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvCacheTransferSpan {
    request_id: RequestId,
    disaggregated_params: DisaggregatedParams,
    data_parallel_rank: i32,
    token_offset: usize,
    token_count: usize,
    cache_pages: Vec<CachePageId>,
    page_size: usize,
    page_indices: Vec<KvCachePageIndex>,
}

impl KvCacheTransferSpan {
    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    pub fn disaggregated_params(&self) -> &DisaggregatedParams {
        &self.disaggregated_params
    }

    pub fn data_parallel_rank(&self) -> i32 {
        self.data_parallel_rank
    }

    pub fn bootstrap_room(&self) -> BootstrapRoom {
        self.disaggregated_params.bootstrap_room
    }

    pub fn token_offset(&self) -> usize {
        self.token_offset
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn cache_pages(&self) -> &[CachePageId] {
        &self.cache_pages
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn page_offset(&self) -> usize {
        self.token_offset / self.page_size
    }

    pub fn page_indices(&self) -> &[KvCachePageIndex] {
        &self.page_indices
    }

    pub fn descriptor_checksum(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"sglang-rs.kv-transfer-span.v2\0");
        update_len_prefixed_str(&mut hasher, self.request_id.as_str());
        update_len_prefixed_str(&mut hasher, &self.disaggregated_params.bootstrap_host);
        hasher.update(self.disaggregated_params.bootstrap_port.to_le_bytes());
        hasher.update(self.disaggregated_params.bootstrap_room.to_le_bytes());
        hasher.update(self.data_parallel_rank.to_le_bytes());
        hasher.update((self.token_offset as u64).to_le_bytes());
        hasher.update((self.token_count as u64).to_le_bytes());
        hasher.update((self.page_size as u64).to_le_bytes());
        hasher.update((self.cache_pages.len() as u64).to_le_bytes());
        for cache_page in &self.cache_pages {
            hasher.update((cache_page.as_usize() as u64).to_le_bytes());
        }
        hasher.update((self.page_indices.len() as u64).to_le_bytes());
        for page_index in &self.page_indices {
            hasher.update((page_index.as_usize() as u64).to_le_bytes());
        }

        let digest = hasher.finalize();
        hex_encode(&digest)
    }

    pub fn is_noop(&self) -> bool {
        self.token_count == 0
    }
}

fn update_len_prefixed_str(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KvCacheTransferPlan {
    spans: Vec<KvCacheTransferSpan>,
}

impl KvCacheTransferPlan {
    pub fn from_prefill_worker_batch(
        batch: &ModelWorkerBatch,
    ) -> Result<Self, KvCacheTransferPlanError> {
        Self::from_prefill_worker_batch_with_page_size(batch, 1)
    }

    pub fn from_prefill_worker_batch_with_page_size(
        batch: &ModelWorkerBatch,
        page_size: usize,
    ) -> Result<Self, KvCacheTransferPlanError> {
        if batch.forward_mode() != ForwardMode::Prefill {
            return Err(KvCacheTransferPlanError::NonPrefillBatch);
        }
        if page_size == 0 {
            return Err(KvCacheTransferPlanError::ZeroPageSize);
        }

        let mut spans = Vec::new();
        let mut cache_page_offset = 0;

        for request_index in 0..batch.request_ids().len() {
            let input_token_count = batch.input_token_counts()[request_index];
            let cache_page_end = cache_page_offset + input_token_count;
            if cache_page_end > batch.out_cache_pages().len() {
                return Err(KvCacheTransferPlanError::CachePageCountMismatch {
                    request_id: batch.request_ids()[request_index].clone(),
                    input_token_count,
                    cache_page_count: batch
                        .out_cache_pages()
                        .len()
                        .saturating_sub(cache_page_offset),
                });
            }

            let cache_pages = batch.out_cache_pages()[cache_page_offset..cache_page_end].to_vec();
            cache_page_offset = cache_page_end;

            let Some(disaggregated_params) = batch.disaggregated_params()[request_index].clone()
            else {
                continue;
            };

            let request_id = batch.request_ids()[request_index].clone();
            let token_offset = batch.cached_token_counts()[request_index];
            let page_indices =
                transfer_page_indices(&request_id, token_offset, &cache_pages, page_size)?;
            spans.push(KvCacheTransferSpan {
                request_id,
                disaggregated_params,
                data_parallel_rank: batch.data_parallel_ranks()[request_index],
                token_offset,
                token_count: input_token_count,
                cache_pages,
                page_size,
                page_indices,
            });
        }

        if cache_page_offset != batch.out_cache_pages().len() {
            return Err(KvCacheTransferPlanError::TrailingCachePages {
                consumed_cache_page_count: cache_page_offset,
                cache_page_count: batch.out_cache_pages().len(),
            });
        }

        Ok(Self { spans })
    }

    pub fn spans(&self) -> &[KvCacheTransferSpan] {
        &self.spans
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    pub fn len(&self) -> usize {
        self.spans.len()
    }
}

fn transfer_page_indices(
    request_id: &RequestId,
    token_offset: usize,
    cache_slots: &[CachePageId],
    page_size: usize,
) -> Result<Vec<KvCachePageIndex>, KvCacheTransferPlanError> {
    if cache_slots.is_empty() {
        return Ok(Vec::new());
    }
    if !token_offset.is_multiple_of(page_size) {
        return Err(KvCacheTransferPlanError::UnalignedTokenOffset {
            request_id: request_id.clone(),
            token_offset,
            page_size,
        });
    }

    let mut page_indices = Vec::with_capacity(cache_slots.len().div_ceil(page_size));
    for slots in cache_slots.chunks(page_size) {
        let first_slot = slots[0].as_usize();
        if !first_slot.is_multiple_of(page_size) {
            return Err(KvCacheTransferPlanError::UnalignedCacheSlot {
                request_id: request_id.clone(),
                cache_slot: first_slot,
                page_size,
            });
        }
        for (slot_offset, slot) in slots.iter().enumerate() {
            let expected = first_slot + slot_offset;
            let actual = slot.as_usize();
            if actual != expected {
                return Err(KvCacheTransferPlanError::NonContiguousCacheSlots {
                    request_id: request_id.clone(),
                    expected,
                    actual,
                });
            }
        }
        page_indices.push(KvCachePageIndex::from(first_slot / page_size));
    }
    Ok(page_indices)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvCacheTransferPlanError {
    NonPrefillBatch,
    ZeroPageSize,
    CachePageCountMismatch {
        request_id: RequestId,
        input_token_count: usize,
        cache_page_count: usize,
    },
    TrailingCachePages {
        consumed_cache_page_count: usize,
        cache_page_count: usize,
    },
    UnalignedTokenOffset {
        request_id: RequestId,
        token_offset: usize,
        page_size: usize,
    },
    UnalignedCacheSlot {
        request_id: RequestId,
        cache_slot: usize,
        page_size: usize,
    },
    NonContiguousCacheSlots {
        request_id: RequestId,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for KvCacheTransferPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPrefillBatch => formatter.write_str("KV transfer plan requires prefill batch"),
            Self::ZeroPageSize => formatter.write_str("KV transfer page size must be non-zero"),
            Self::CachePageCountMismatch {
                request_id,
                input_token_count,
                cache_page_count,
            } => write!(
                formatter,
                "request {} has {input_token_count} input tokens but {cache_page_count} cache pages available",
                request_id.as_str()
            ),
            Self::TrailingCachePages {
                consumed_cache_page_count,
                cache_page_count,
            } => write!(
                formatter,
                "KV transfer plan consumed {consumed_cache_page_count} cache pages but batch has {cache_page_count}"
            ),
            Self::UnalignedTokenOffset {
                request_id,
                token_offset,
                page_size,
            } => write!(
                formatter,
                "request {} KV delta starts at token {token_offset}, which is not aligned to page size {page_size}",
                request_id.as_str()
            ),
            Self::UnalignedCacheSlot {
                request_id,
                cache_slot,
                page_size,
            } => write!(
                formatter,
                "request {} KV cache slot {cache_slot} does not start a page of size {page_size}",
                request_id.as_str()
            ),
            Self::NonContiguousCacheSlots {
                request_id,
                expected,
                actual,
            } => write!(
                formatter,
                "request {} KV cache page expected slot {expected} but found {actual}",
                request_id.as_str()
            ),
        }
    }
}

impl std::error::Error for KvCacheTransferPlanError {}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KvCacheTransferSummary {
    submitted_spans: usize,
    noop_spans: usize,
    snapshot_content_checksums: Vec<String>,
}

impl KvCacheTransferSummary {
    pub fn submitted_spans(&self) -> usize {
        self.submitted_spans
    }

    pub fn noop_spans(&self) -> usize {
        self.noop_spans
    }

    pub fn snapshot_content_checksums(&self) -> &[String] {
        &self.snapshot_content_checksums
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DecodeBootstrapMetadataPublishSummary {
    pub published_spans: usize,
}

pub trait DecodeBootstrapPublisher {
    fn publish_decode_bootstrap_metadata(
        &mut self,
        plan: &KvCacheTransferPlan,
    ) -> Result<DecodeBootstrapMetadataPublishSummary, String>;
}

pub trait KvCachePageSnapshotProvider {
    type Snapshot;

    fn export_kv_cache_pages(
        &self,
        cache_pages: &[CachePageId],
    ) -> Result<Vec<Self::Snapshot>, KvCacheTransferError>;
}

pub trait KvCachePageSnapshotChecksum {
    fn update_content_checksum(&self, hasher: &mut Sha256);

    fn content_checksum(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"sglang-rs.kv-page-snapshot.v1\0");
        self.update_content_checksum(&mut hasher);
        let digest = hasher.finalize();
        hex_encode(&digest)
    }
}

impl KvCachePageSnapshotChecksum for CachePageId {
    fn update_content_checksum(&self, hasher: &mut Sha256) {
        hasher.update((self.as_usize() as u64).to_le_bytes());
    }
}

pub fn snapshot_content_checksum<S>(snapshots: &[S]) -> String
where
    S: KvCachePageSnapshotChecksum,
{
    let mut hasher = Sha256::new();
    hasher.update(b"sglang-rs.kv-page-snapshot-batch.v1\0");
    hasher.update((snapshots.len() as u64).to_le_bytes());
    for snapshot in snapshots {
        snapshot.update_content_checksum(&mut hasher);
    }
    let digest = hasher.finalize();
    hex_encode(&digest)
}

pub trait KvCachePageSnapshotImporter {
    type Snapshot;

    fn import_kv_cache_pages(
        &mut self,
        snapshots: Vec<Self::Snapshot>,
    ) -> Result<(), KvCacheTransferError>;
}

impl<M, S> KvCachePageSnapshotProvider for ModelRunner<M, S>
where
    M: KvCachePageSnapshotProvider,
{
    type Snapshot = M::Snapshot;

    fn export_kv_cache_pages(
        &self,
        cache_pages: &[CachePageId],
    ) -> Result<Vec<Self::Snapshot>, KvCacheTransferError> {
        self.model().export_kv_cache_pages(cache_pages)
    }
}

impl<M, S> KvCachePageSnapshotImporter for ModelRunner<M, S>
where
    M: KvCachePageSnapshotImporter,
{
    type Snapshot = M::Snapshot;

    fn import_kv_cache_pages(
        &mut self,
        snapshots: Vec<Self::Snapshot>,
    ) -> Result<(), KvCacheTransferError> {
        self.model_mut().import_kv_cache_pages(snapshots)
    }
}

pub struct LocalSnapshotTransferPdModelWorkers<P, D> {
    prefill: P,
    decode: D,
    last_transfer_summary: Option<KvCacheTransferSummary>,
}

impl<P, D> LocalSnapshotTransferPdModelWorkers<P, D> {
    pub fn new(prefill: P, decode: D) -> Self {
        Self {
            prefill,
            decode,
            last_transfer_summary: None,
        }
    }

    pub fn prefill(&self) -> &P {
        &self.prefill
    }

    pub fn prefill_mut(&mut self) -> &mut P {
        &mut self.prefill
    }

    pub fn decode(&self) -> &D {
        &self.decode
    }

    pub fn decode_mut(&mut self) -> &mut D {
        &mut self.decode
    }

    pub fn last_transfer_summary(&self) -> Option<&KvCacheTransferSummary> {
        self.last_transfer_summary.as_ref()
    }
}

impl<P, D> FallibleModelWorker for LocalSnapshotTransferPdModelWorkers<P, D>
where
    P: FallibleModelWorker + KvCachePageSnapshotProvider,
    P::Snapshot: KvCachePageSnapshotChecksum,
    D: FallibleModelWorker + KvCachePageSnapshotImporter<Snapshot = P::Snapshot>,
{
    fn try_generate_batch(
        &mut self,
        batch: &ScheduleBatch,
    ) -> Result<BatchGeneratedTokens, WorkerExecutionError> {
        match batch.forward_mode() {
            ForwardMode::Prefill => self.try_generate_prefill_batch(batch),
            ForwardMode::Decode => self.decode.try_generate_batch(batch),
        }
    }

    fn decode_request_state(
        &self,
        request: &ScheduledRequest,
    ) -> Result<DecodeRequestState, WorkerExecutionError> {
        self.decode.decode_request_state(request)
    }

    fn poll_transfers(&mut self) -> Result<MooncakeTransferPollSummary, KvCacheTransferError> {
        Ok(MooncakeTransferPollSummary::default())
    }

    fn complete_request(&mut self, request: &ScheduledRequest) {
        self.prefill.complete_request(request);
        self.decode.complete_request(request);
    }

    fn update_weights_from_disk(
        &mut self,
        request: &WorkerWeightUpdateRequest,
    ) -> Result<(), WorkerExecutionError> {
        self.prefill.update_weights_from_disk(request)?;
        self.decode.update_weights_from_disk(request)
    }
}

impl<P, D> LocalSnapshotTransferPdModelWorkers<P, D>
where
    P: FallibleModelWorker + KvCachePageSnapshotProvider,
    P::Snapshot: KvCachePageSnapshotChecksum,
    D: FallibleModelWorker + KvCachePageSnapshotImporter<Snapshot = P::Snapshot>,
{
    fn try_generate_prefill_batch(
        &mut self,
        batch: &ScheduleBatch,
    ) -> Result<BatchGeneratedTokens, WorkerExecutionError> {
        let worker_batch = ModelWorkerBatch::from_schedule_batch(batch);
        let transfer_plan =
            KvCacheTransferPlan::from_prefill_worker_batch(&worker_batch).map_err(|error| {
                WorkerExecutionError::Runtime(format!("KV transfer planning failed: {error}"))
            })?;

        let output = self.prefill.try_generate_batch(batch)?;
        let mut summary = KvCacheTransferSummary::default();
        for span in transfer_plan.spans() {
            if span.is_noop() {
                summary.noop_spans += 1;
                continue;
            }

            let snapshots = self
                .prefill
                .export_kv_cache_pages(span.cache_pages())
                .map_err(|error| {
                    WorkerExecutionError::Runtime(format!(
                        "local KV snapshot export failed: {error}"
                    ))
                })?;
            let content_checksum = snapshot_content_checksum(&snapshots);
            self.decode
                .import_kv_cache_pages(snapshots)
                .map_err(|error| {
                    WorkerExecutionError::Runtime(format!(
                        "local KV snapshot import failed: {error}"
                    ))
                })?;
            summary.snapshot_content_checksums.push(content_checksum);
            summary.submitted_spans += 1;
        }
        self.last_transfer_summary = Some(summary);

        Ok(output)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NoopDecodeBootstrapPublisher;

impl DecodeBootstrapPublisher for NoopDecodeBootstrapPublisher {
    fn publish_decode_bootstrap_metadata(
        &mut self,
        _plan: &KvCacheTransferPlan,
    ) -> Result<DecodeBootstrapMetadataPublishSummary, String> {
        Ok(DecodeBootstrapMetadataPublishSummary::default())
    }
}

pub trait KvCacheTransferExecutor {
    fn transfer_span(&mut self, span: &KvCacheTransferSpan) -> Result<(), KvCacheTransferError>;

    fn completes_inline(&self) -> bool {
        true
    }

    fn poll_transfers(
        &mut self,
        _registry: &mut DecodeBootstrapRegistry,
    ) -> Result<MooncakeTransferPollSummary, KvCacheTransferError> {
        Ok(MooncakeTransferPollSummary::default())
    }

    fn cancel_transfer_room(
        &mut self,
        _bootstrap_room: BootstrapRoom,
    ) -> Result<(), KvCacheTransferError> {
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FakeKvCacheTransferExecutor {
    transferred_rooms: Vec<BootstrapRoom>,
}

impl FakeKvCacheTransferExecutor {
    pub fn transferred_rooms(&self) -> &[BootstrapRoom] {
        &self.transferred_rooms
    }
}

impl KvCacheTransferExecutor for FakeKvCacheTransferExecutor {
    fn transfer_span(&mut self, span: &KvCacheTransferSpan) -> Result<(), KvCacheTransferError> {
        self.transferred_rooms.push(span.bootstrap_room());
        Ok(())
    }
}

pub fn execute_kv_cache_transfer_plan<E>(
    registry: &mut DecodeBootstrapRegistry,
    executor: &mut E,
    plan: &KvCacheTransferPlan,
) -> Result<KvCacheTransferSummary, KvCacheTransferError>
where
    E: KvCacheTransferExecutor,
{
    let mut summary = KvCacheTransferSummary::default();

    for span in plan.spans() {
        if span.is_noop() {
            registry.update_status(span.bootstrap_room(), KvPoll::Success)?;
            summary.noop_spans += 1;
            continue;
        }

        registry.update_status(span.bootstrap_room(), KvPoll::Transferring)?;
        if let Err(error) = executor.transfer_span(span) {
            registry.update_status(span.bootstrap_room(), KvPoll::Failed)?;
            return Err(error);
        }
        if executor.completes_inline() {
            registry.update_status(span.bootstrap_room(), KvPoll::Success)?;
        }
        summary.submitted_spans += 1;
    }

    Ok(summary)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvCacheTransferError {
    Registry(DecodeBootstrapRegistryError),
    Runtime(String),
}

impl From<DecodeBootstrapRegistryError> for KvCacheTransferError {
    fn from(value: DecodeBootstrapRegistryError) -> Self {
        Self::Registry(value)
    }
}

impl From<TransferableKvCacheMemoryError> for KvCacheTransferError {
    fn from(value: TransferableKvCacheMemoryError) -> Self {
        Self::Runtime(value.to_string())
    }
}

impl fmt::Display for KvCacheTransferError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Registry(error) => write!(formatter, "decode bootstrap registry error: {error}"),
            Self::Runtime(error) => write!(formatter, "KV cache transfer runtime error: {error}"),
        }
    }
}

impl std::error::Error for KvCacheTransferError {}

pub fn is_decode_request_kv_ready(
    request: &ScheduledRequest,
    registry: &DecodeBootstrapRegistry,
) -> Result<bool, KvCacheTransferError> {
    let Some(disaggregated_params) = request.disaggregated_params() else {
        return Ok(true);
    };

    let session = registry.get(disaggregated_params.bootstrap_room).ok_or(
        DecodeBootstrapRegistryError::MissingBootstrapRoom(disaggregated_params.bootstrap_room),
    )?;

    Ok(session.status() == KvPoll::Success)
}

pub struct KvTransferModelWorker<W, E, P = NoopDecodeBootstrapPublisher> {
    // Registration leases belong to the transfer executor, so it must drop before
    // the model releases the backing KV allocations.
    transfer_executor: E,
    worker: W,
    registry: DecodeBootstrapRegistry,
    decode_bootstrap_publisher: P,
    kv_page_size: usize,
    submit_prefill_transfers: bool,
    last_transfer_summary: Option<KvCacheTransferSummary>,
}

impl<W, E> KvTransferModelWorker<W, E> {
    pub fn new(worker: W, registry: DecodeBootstrapRegistry, transfer_executor: E) -> Self {
        Self {
            transfer_executor,
            worker,
            registry,
            decode_bootstrap_publisher: NoopDecodeBootstrapPublisher,
            kv_page_size: 1,
            submit_prefill_transfers: true,
            last_transfer_summary: None,
        }
    }
}

impl<W, E, P> KvTransferModelWorker<W, E, P> {
    pub fn worker(&self) -> &W {
        &self.worker
    }

    pub fn worker_mut(&mut self) -> &mut W {
        &mut self.worker
    }

    pub fn registry(&self) -> &DecodeBootstrapRegistry {
        &self.registry
    }

    pub fn registry_mut(&mut self) -> &mut DecodeBootstrapRegistry {
        &mut self.registry
    }

    pub fn transfer_executor(&self) -> &E {
        &self.transfer_executor
    }

    pub fn transfer_executor_mut(&mut self) -> &mut E {
        &mut self.transfer_executor
    }

    pub fn decode_bootstrap_publisher(&self) -> &P {
        &self.decode_bootstrap_publisher
    }

    pub fn decode_bootstrap_publisher_mut(&mut self) -> &mut P {
        &mut self.decode_bootstrap_publisher
    }

    pub fn with_decode_bootstrap_publisher<NextP>(
        self,
        decode_bootstrap_publisher: NextP,
    ) -> KvTransferModelWorker<W, E, NextP> {
        KvTransferModelWorker {
            transfer_executor: self.transfer_executor,
            worker: self.worker,
            registry: self.registry,
            decode_bootstrap_publisher,
            kv_page_size: self.kv_page_size,
            submit_prefill_transfers: self.submit_prefill_transfers,
            last_transfer_summary: self.last_transfer_summary,
        }
    }

    pub fn with_decode_side_bootstrap_only(mut self) -> Self {
        self.submit_prefill_transfers = false;
        self
    }

    pub fn with_kv_page_size(mut self, kv_page_size: usize) -> Self {
        self.kv_page_size = kv_page_size;
        self
    }

    pub fn last_transfer_summary(&self) -> Option<&KvCacheTransferSummary> {
        self.last_transfer_summary.as_ref()
    }

    pub fn export_kv_cache_pages(
        &self,
        cache_pages: &[CachePageId],
    ) -> Result<Vec<W::Snapshot>, KvCacheTransferError>
    where
        W: KvCachePageSnapshotProvider,
    {
        self.worker.export_kv_cache_pages(cache_pages)
    }

    pub fn import_kv_cache_pages(
        &mut self,
        snapshots: Vec<W::Snapshot>,
    ) -> Result<(), KvCacheTransferError>
    where
        W: KvCachePageSnapshotImporter,
    {
        self.worker.import_kv_cache_pages(snapshots)
    }
}

impl<W, E, P> FallibleModelWorker for KvTransferModelWorker<W, E, P>
where
    W: FallibleModelWorker,
    E: KvCacheTransferExecutor,
    P: DecodeBootstrapPublisher,
{
    fn try_generate_batch(
        &mut self,
        batch: &ScheduleBatch,
    ) -> Result<BatchGeneratedTokens, WorkerExecutionError> {
        let transfer_plan = if batch.forward_mode() == ForwardMode::Prefill {
            let worker_batch = ModelWorkerBatch::from_schedule_batch(batch);
            let transfer_plan = KvCacheTransferPlan::from_prefill_worker_batch_with_page_size(
                &worker_batch,
                self.kv_page_size,
            )
            .map_err(|error| {
                WorkerExecutionError::Runtime(format!("KV transfer planning failed: {error}"))
            })?;
            self.decode_bootstrap_publisher
                .publish_decode_bootstrap_metadata(&transfer_plan)
                .map_err(|error| {
                    WorkerExecutionError::Runtime(format!(
                        "decode bootstrap metadata publish failed: {error}"
                    ))
                })?;
            Some(transfer_plan)
        } else {
            None
        };

        let output = self.worker.try_generate_batch(batch)?;

        if let Some(transfer_plan) = transfer_plan {
            self.register_prefill_bootstrap_sessions(batch)
                .map_err(|error| {
                    WorkerExecutionError::Runtime(format!(
                        "KV transfer bootstrap registration failed: {error}"
                    ))
                })?;
            if !self.submit_prefill_transfers {
                let transfer_summary = self
                    .mark_prefill_bootstrap_sessions_ready_without_submit(&transfer_plan)
                    .map_err(|error| {
                        WorkerExecutionError::Runtime(format!(
                            "KV transfer bootstrap registration failed: {error}"
                        ))
                    })?;
                self.last_transfer_summary = Some(transfer_summary);
                return Ok(output);
            }
            let transfer_summary = execute_kv_cache_transfer_plan(
                &mut self.registry,
                &mut self.transfer_executor,
                &transfer_plan,
            )
            .map_err(|error| {
                WorkerExecutionError::Runtime(format!("KV transfer execution failed: {error}"))
            })?;
            self.last_transfer_summary = Some(transfer_summary);
        }

        Ok(output)
    }

    fn decode_request_state(
        &self,
        request: &ScheduledRequest,
    ) -> Result<DecodeRequestState, WorkerExecutionError> {
        let Some(disaggregated_params) = request.disaggregated_params() else {
            return self.worker.decode_request_state(request);
        };

        match self.registry.get(disaggregated_params.bootstrap_room) {
            Some(session) => match session.status() {
                KvPoll::Success => self.worker.decode_request_state(request),
                KvPoll::Failed => Ok(DecodeRequestState::Failed(format!(
                    "KV transfer failed for bootstrap room {}",
                    disaggregated_params.bootstrap_room
                ))),
                KvPoll::Bootstrapping | KvPoll::WaitingForInput | KvPoll::Transferring => {
                    Ok(DecodeRequestState::Pending)
                }
            },
            None => Ok(DecodeRequestState::Pending),
        }
    }

    fn poll_transfers(&mut self) -> Result<MooncakeTransferPollSummary, KvCacheTransferError> {
        self.transfer_executor.poll_transfers(&mut self.registry)
    }

    fn complete_request(&mut self, request: &ScheduledRequest) {
        if let Some(disaggregated_params) = request.disaggregated_params() {
            self.registry.remove(disaggregated_params.bootstrap_room);
            let _ = self
                .transfer_executor
                .cancel_transfer_room(disaggregated_params.bootstrap_room);
        }

        self.worker.complete_request(request)
    }

    fn fail_request(&mut self, request: &ScheduledRequest) {
        if let Some(disaggregated_params) = request.disaggregated_params() {
            let transfer_failed = self
                .registry
                .get(disaggregated_params.bootstrap_room)
                .is_some_and(|session| session.status() == KvPoll::Failed);
            if !transfer_failed {
                self.registry.remove(disaggregated_params.bootstrap_room);
                let _ = self
                    .transfer_executor
                    .cancel_transfer_room(disaggregated_params.bootstrap_room);
            }
        }

        self.worker.fail_request(request)
    }

    fn update_weights_from_disk(
        &mut self,
        request: &WorkerWeightUpdateRequest,
    ) -> Result<(), WorkerExecutionError> {
        self.worker.update_weights_from_disk(request)
    }
}

impl<W, E, P> KvTransferModelWorker<W, E, P> {
    fn register_prefill_bootstrap_sessions(
        &mut self,
        batch: &ScheduleBatch,
    ) -> Result<(), DecodeBootstrapRegistryError> {
        for request in batch.requests() {
            let Some(disaggregated_params) = request.disaggregated_params().cloned() else {
                continue;
            };
            if self
                .registry
                .get(disaggregated_params.bootstrap_room)
                .is_some()
            {
                continue;
            }

            self.registry.register(DecodeBootstrapSession::new(
                request.request_id().clone(),
                disaggregated_params,
                request.data_parallel_rank(),
            ))?;
        }

        Ok(())
    }

    fn mark_prefill_bootstrap_sessions_ready_without_submit(
        &mut self,
        transfer_plan: &KvCacheTransferPlan,
    ) -> Result<KvCacheTransferSummary, DecodeBootstrapRegistryError> {
        let mut summary = KvCacheTransferSummary::default();
        for span in transfer_plan.spans() {
            self.registry
                .update_status(span.bootstrap_room(), KvPoll::Success)?;
            if span.is_noop() {
                summary.noop_spans += 1;
            }
        }
        Ok(summary)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum MooncakeOpcode {
    Read = 0,
    Write = 1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum MooncakeTransferStatusCode {
    Waiting = 0,
    Pending = 1,
    Invalid = 2,
    Canceled = 3,
    Completed = 4,
    Timeout = 5,
    Failed = 6,
}

impl TryFrom<c_int> for MooncakeTransferStatusCode {
    type Error = KvCacheTransferError;

    fn try_from(value: c_int) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Waiting),
            1 => Ok(Self::Pending),
            2 => Ok(Self::Invalid),
            3 => Ok(Self::Canceled),
            4 => Ok(Self::Completed),
            5 => Ok(Self::Timeout),
            6 => Ok(Self::Failed),
            other => Err(KvCacheTransferError::Runtime(format!(
                "unknown Mooncake transfer status: {other}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct MooncakeTransferStatus {
    pub status: c_int,
    pub transferred_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct MooncakeTransferRequest {
    pub opcode: c_int,
    pub source: *mut c_void,
    pub target_id: i32,
    pub target_offset: u64,
    pub length: u64,
}

// The pointer is an opaque registered device/host address passed to Mooncake; moving
// the request between Rust threads does not grant dereference access in Rust.
unsafe impl Send for MooncakeTransferRequest {}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct MooncakeBufferEntry {
    pub addr: *mut c_void,
    pub length: usize,
}

pub type MooncakeTransferEngineHandle = *mut c_void;
pub type MooncakeTransportHandle = *mut c_void;
pub type MooncakeSegmentId = i32;
pub type MooncakeBatchId = u64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MooncakeKvCacheLayout {
    pub source_base_addr: usize,
    pub page_size_bytes: usize,
    pub target_base_offset: u64,
}

impl MooncakeKvCacheLayout {
    pub fn from_pd_config(
        source_base_addr: usize,
        token_size_bytes: usize,
        target_base_offset: u64,
        config: &PdConfig,
    ) -> Self {
        Self {
            source_base_addr,
            page_size_bytes: config.page_size * token_size_bytes,
            target_base_offset,
        }
    }

    pub fn from_pd_config_kv_elements(
        source_base_addr: usize,
        kv_elements_per_token: usize,
        target_base_offset: u64,
        config: &PdConfig,
    ) -> Result<Self, PdConfigError> {
        let bytes_per_element = config.kv_cache_dtype.bytes_per_element().ok_or(
            PdConfigError::KvCacheDtypeRequiresModelMetadata(config.kv_cache_dtype),
        )?;
        let token_size_bytes = kv_elements_per_token
            .checked_mul(bytes_per_element)
            .ok_or(PdConfigError::KvCacheLayoutOverflow)?;
        let page_size_bytes = config
            .page_size
            .checked_mul(token_size_bytes)
            .ok_or(PdConfigError::KvCacheLayoutOverflow)?;

        Ok(Self {
            source_base_addr,
            page_size_bytes,
            target_base_offset,
        })
    }

    pub fn from_pd_config_model_layout(
        source_base_addr: usize,
        target_base_offset: u64,
        config: &PdConfig,
        model_layout: &KvCacheModelLayout,
    ) -> Result<Self, PdConfigError> {
        let token_size_bytes = model_layout.token_size_bytes(config.kv_cache_dtype)?;
        let page_size_bytes = config
            .page_size
            .checked_mul(token_size_bytes)
            .ok_or(PdConfigError::KvCacheLayoutOverflow)?;

        Ok(Self {
            source_base_addr,
            page_size_bytes,
            target_base_offset,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MooncakeTransferTarget {
    pub target_id: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MooncakeRemoteKvLayout {
    pub dst_kv_ptrs: Vec<u64>,
    pub dst_kv_indices: Vec<i32>,
    pub dst_kv_item_len: usize,
}

pub trait MooncakeMemoryLocationExt {
    fn mooncake_memory_location_label(self) -> Result<String, MooncakeError>;
}

impl MooncakeMemoryLocationExt for KvCacheMemoryLocation {
    fn mooncake_memory_location_label(self) -> Result<String, MooncakeError> {
        let label = match self {
            Self::Cpu { numa_node } => format!("cpu:{numa_node}"),
            Self::Cuda { device_id } => format!("cuda:{device_id}"),
            Self::Rocm { device_id } => format!("hip:{device_id}"),
            Self::Musa { device_id } => format!("musa:{device_id}"),
            Self::Npu { device_id } => format!("npu:{device_id}"),
            location @ (Self::Metal { .. } | Self::Xpu { .. } | Self::Hpu { .. }) => {
                return Err(MooncakeError::UnsupportedMemoryLocation(location));
            }
        };
        Ok(label)
    }
}

pub trait MooncakeKvCacheMemoryExt {
    fn mooncake_prefill_layout(&self, target_base_offset: u64) -> MooncakeKvCacheLayout;

    fn mooncake_decode_remote_layout(&self, dst_kv_indices: &[i32]) -> MooncakeRemoteKvLayout;
}

impl MooncakeKvCacheMemoryExt for TransferableKvCacheMemory {
    fn mooncake_prefill_layout(&self, target_base_offset: u64) -> MooncakeKvCacheLayout {
        MooncakeKvCacheLayout {
            source_base_addr: self.regions()[0].base_addr,
            page_size_bytes: self.page_size_bytes(),
            target_base_offset,
        }
    }

    fn mooncake_decode_remote_layout(&self, dst_kv_indices: &[i32]) -> MooncakeRemoteKvLayout {
        MooncakeRemoteKvLayout {
            dst_kv_ptrs: self
                .regions()
                .iter()
                .map(|region| region.base_addr as u64)
                .collect(),
            dst_kv_indices: dst_kv_indices.to_vec(),
            dst_kv_item_len: self.page_size_bytes() / self.regions().len(),
        }
    }
}

pub trait MooncakeMemoryRegistrar: Send + 'static {
    fn register_memory_batch(
        &mut self,
        buffers: &mut [MooncakeBufferEntry],
        location: &str,
    ) -> Result<(), MooncakeError>;

    fn unregister_memory_batch(&mut self, addrs: &mut [*mut c_void]) -> Result<(), MooncakeError>;
}

pub trait MooncakeMemoryRegistrationLease: Send {
    fn memory(&self) -> &TransferableKvCacheMemory;
    fn unregister(&mut self) -> Result<(), MooncakeError>;
}

pub struct RegisteredMooncakeKvCacheMemory<R>
where
    R: MooncakeMemoryRegistrar,
{
    registrar: R,
    memory: TransferableKvCacheMemory,
    registered_addrs: Vec<usize>,
    active: bool,
}

impl<R> RegisteredMooncakeKvCacheMemory<R>
where
    R: MooncakeMemoryRegistrar,
{
    pub fn register(
        mut registrar: R,
        memory: TransferableKvCacheMemory,
    ) -> Result<Self, MooncakeError> {
        let mut buffers = memory
            .regions()
            .iter()
            .map(|region| MooncakeBufferEntry {
                addr: region.base_addr as *mut c_void,
                length: region.byte_len,
            })
            .collect::<Vec<_>>();
        let location = memory.location().mooncake_memory_location_label()?;
        registrar.register_memory_batch(&mut buffers, &location)?;

        Ok(Self {
            registrar,
            registered_addrs: memory
                .regions()
                .iter()
                .map(|region| region.base_addr)
                .collect(),
            memory,
            active: true,
        })
    }

    pub fn memory(&self) -> &TransferableKvCacheMemory {
        &self.memory
    }

    pub fn unregister(&mut self) -> Result<(), MooncakeError> {
        if !self.active {
            return Ok(());
        }

        let mut addrs = self
            .registered_addrs
            .iter()
            .map(|addr| *addr as *mut c_void)
            .collect::<Vec<_>>();
        self.registrar.unregister_memory_batch(&mut addrs)?;
        self.active = false;
        Ok(())
    }
}

impl<R> MooncakeMemoryRegistrationLease for RegisteredMooncakeKvCacheMemory<R>
where
    R: MooncakeMemoryRegistrar,
{
    fn memory(&self) -> &TransferableKvCacheMemory {
        self.memory()
    }

    fn unregister(&mut self) -> Result<(), MooncakeError> {
        self.unregister()
    }
}

impl<R> Drop for RegisteredMooncakeKvCacheMemory<R>
where
    R: MooncakeMemoryRegistrar,
{
    fn drop(&mut self) {
        if let Err(error) = self.unregister() {
            eprintln!("failed to unregister Mooncake KV cache memory: {error}");
        }
    }
}

pub trait MooncakeSegmentOpener {
    fn open_segment(&mut self, segment: &str) -> Result<i32, MooncakeError>;
}

pub trait MooncakeTransferTargetResolver {
    fn resolve_target(
        &mut self,
        span: &KvCacheTransferSpan,
    ) -> Result<MooncakeTransferTarget, KvCacheTransferError>;

    fn resolve_session_target(
        &mut self,
        span: &KvCacheTransferSpan,
        _session_id: &str,
    ) -> Result<MooncakeTransferTarget, KvCacheTransferError> {
        self.resolve_target(span)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedMooncakeTransferTargetResolver {
    target: MooncakeTransferTarget,
}

impl FixedMooncakeTransferTargetResolver {
    pub fn new(target: MooncakeTransferTarget) -> Self {
        Self { target }
    }
}

impl MooncakeTransferTargetResolver for FixedMooncakeTransferTargetResolver {
    fn resolve_target(
        &mut self,
        _span: &KvCacheTransferSpan,
    ) -> Result<MooncakeTransferTarget, KvCacheTransferError> {
        Ok(self.target)
    }
}

#[derive(Clone, Debug)]
pub struct MooncakeSessionTargetResolver<O> {
    opener: O,
    sessions_by_room: BTreeMap<BootstrapRoom, String>,
    targets_by_session: BTreeMap<String, MooncakeTransferTarget>,
}

impl<O> MooncakeSessionTargetResolver<O> {
    pub fn new(opener: O, sessions_by_room: Vec<(BootstrapRoom, String)>) -> Self {
        Self {
            opener,
            sessions_by_room: sessions_by_room.into_iter().collect(),
            targets_by_session: BTreeMap::new(),
        }
    }

    pub fn opener(&self) -> &O {
        &self.opener
    }

    pub fn opener_mut(&mut self) -> &mut O {
        &mut self.opener
    }

    pub fn insert_room_session(
        &mut self,
        bootstrap_room: BootstrapRoom,
        session_id: impl Into<String>,
    ) {
        self.sessions_by_room
            .insert(bootstrap_room, session_id.into());
    }
}

impl<O> MooncakeTransferTargetResolver for MooncakeSessionTargetResolver<O>
where
    O: MooncakeSegmentOpener,
{
    fn resolve_target(
        &mut self,
        span: &KvCacheTransferSpan,
    ) -> Result<MooncakeTransferTarget, KvCacheTransferError> {
        let session_id = self
            .sessions_by_room
            .get(&span.bootstrap_room())
            .ok_or_else(|| {
                KvCacheTransferError::Runtime(format!(
                    "missing Mooncake session for bootstrap room {}",
                    span.bootstrap_room()
                ))
            })?
            .clone();

        if let Some(target) = self.targets_by_session.get(&session_id) {
            return Ok(*target);
        }

        let target_id = self
            .opener
            .open_segment(&session_id)
            .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?;
        let target = MooncakeTransferTarget { target_id };
        self.targets_by_session.insert(session_id, target);
        Ok(target)
    }

    fn resolve_session_target(
        &mut self,
        _span: &KvCacheTransferSpan,
        session_id: &str,
    ) -> Result<MooncakeTransferTarget, KvCacheTransferError> {
        if let Some(target) = self.targets_by_session.get(session_id) {
            return Ok(*target);
        }

        let target_id = self
            .opener
            .open_segment(session_id)
            .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?;
        let target = MooncakeTransferTarget { target_id };
        self.targets_by_session
            .insert(session_id.to_string(), target);
        Ok(target)
    }
}

pub fn build_mooncake_kv_transfer_requests(
    span: &KvCacheTransferSpan,
    layout: MooncakeKvCacheLayout,
    target: MooncakeTransferTarget,
) -> Result<Vec<MooncakeTransferRequest>, MooncakeRequestBuildError> {
    if layout.page_size_bytes == 0 {
        return Err(MooncakeRequestBuildError::ZeroPageSize);
    }

    let expected_page_count = span.token_count().div_ceil(span.page_size());
    if expected_page_count != span.page_indices().len() {
        return Err(MooncakeRequestBuildError::SpanPageCountMismatch {
            expected_page_count,
            source_page_index_count: span.page_indices().len(),
        });
    }

    let page_size_bytes = layout.page_size_bytes as u64;
    let mut requests = Vec::with_capacity(span.page_indices().len());

    for (page_index, source_page_index) in span.page_indices().iter().enumerate() {
        let source_offset = source_page_index
            .as_usize()
            .checked_mul(layout.page_size_bytes)
            .ok_or(MooncakeRequestBuildError::AddressOverflow)?;
        let source_addr = layout
            .source_base_addr
            .checked_add(source_offset)
            .ok_or(MooncakeRequestBuildError::AddressOverflow)?;
        let target_page_index = span
            .page_offset()
            .checked_add(page_index)
            .ok_or(MooncakeRequestBuildError::OffsetOverflow)?;
        let target_offset = layout
            .target_base_offset
            .checked_add(
                (target_page_index as u64)
                    .checked_mul(page_size_bytes)
                    .ok_or(MooncakeRequestBuildError::OffsetOverflow)?,
            )
            .ok_or(MooncakeRequestBuildError::OffsetOverflow)?;

        requests.push(MooncakeTransferRequest {
            opcode: MooncakeOpcode::Write as c_int,
            source: source_addr as *mut c_void,
            target_id: target.target_id,
            target_offset,
            length: page_size_bytes,
        });
    }

    Ok(requests)
}

pub fn build_mooncake_remote_kv_transfer_requests(
    span: &KvCacheTransferSpan,
    layout: MooncakeKvCacheLayout,
    target: MooncakeTransferTarget,
    remote_layout: &MooncakeRemoteKvLayout,
) -> Result<Vec<MooncakeTransferRequest>, MooncakeRequestBuildError> {
    if layout.page_size_bytes == 0 {
        return Err(MooncakeRequestBuildError::ZeroPageSize);
    }

    if remote_layout.dst_kv_item_len == 0 {
        return Err(MooncakeRequestBuildError::ZeroRemoteKvItemSize);
    }

    if remote_layout.dst_kv_ptrs.is_empty() {
        return Err(MooncakeRequestBuildError::MissingRemoteKvPointers);
    }

    let expected_page_count = span.token_count().div_ceil(span.page_size());
    if expected_page_count != span.page_indices().len() {
        return Err(MooncakeRequestBuildError::SpanPageCountMismatch {
            expected_page_count,
            source_page_index_count: span.page_indices().len(),
        });
    }

    if span.page_indices().len() != remote_layout.dst_kv_indices.len() {
        return Err(MooncakeRequestBuildError::RemoteKvIndexCountMismatch {
            source_page_count: span.page_indices().len(),
            dst_kv_index_count: remote_layout.dst_kv_indices.len(),
        });
    }

    let item_len = remote_layout.dst_kv_item_len;
    let split_page_size = item_len
        .checked_mul(remote_layout.dst_kv_ptrs.len())
        .ok_or(MooncakeRequestBuildError::AddressOverflow)?;
    if split_page_size != layout.page_size_bytes {
        return Err(MooncakeRequestBuildError::RemoteKvItemLayoutMismatch {
            page_size_bytes: layout.page_size_bytes,
            split_page_size,
        });
    }

    let mut requests =
        Vec::with_capacity(span.page_indices().len() * remote_layout.dst_kv_ptrs.len());

    for (page_index, source_page_index) in span.page_indices().iter().enumerate() {
        let dst_kv_index = remote_layout.dst_kv_indices[page_index];
        if dst_kv_index < 0 {
            return Err(MooncakeRequestBuildError::NegativeRemoteKvIndex(
                dst_kv_index,
            ));
        }

        let source_offset = source_page_index
            .as_usize()
            .checked_mul(layout.page_size_bytes)
            .ok_or(MooncakeRequestBuildError::AddressOverflow)?;
        let source_page_addr = layout
            .source_base_addr
            .checked_add(source_offset)
            .ok_or(MooncakeRequestBuildError::AddressOverflow)?;

        for (ptr_index, dst_kv_ptr) in remote_layout.dst_kv_ptrs.iter().enumerate() {
            let source_addr = source_page_addr
                .checked_add(
                    ptr_index
                        .checked_mul(item_len)
                        .ok_or(MooncakeRequestBuildError::AddressOverflow)?,
                )
                .ok_or(MooncakeRequestBuildError::AddressOverflow)?;
            let target_offset = dst_kv_ptr
                .checked_add(
                    (dst_kv_index as u64)
                        .checked_mul(item_len as u64)
                        .ok_or(MooncakeRequestBuildError::OffsetOverflow)?,
                )
                .ok_or(MooncakeRequestBuildError::OffsetOverflow)?;

            requests.push(MooncakeTransferRequest {
                opcode: MooncakeOpcode::Write as c_int,
                source: source_addr as *mut c_void,
                target_id: target.target_id,
                target_offset,
                length: item_len as u64,
            });
        }
    }

    Ok(requests)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MooncakeRequestBuildError {
    ZeroPageSize,
    ZeroRemoteKvItemSize,
    MissingRemoteKvPointers,
    SpanPageCountMismatch {
        expected_page_count: usize,
        source_page_index_count: usize,
    },
    RemoteKvIndexCountMismatch {
        source_page_count: usize,
        dst_kv_index_count: usize,
    },
    RemoteKvItemLayoutMismatch {
        page_size_bytes: usize,
        split_page_size: usize,
    },
    NegativeRemoteKvIndex(i32),
    AddressOverflow,
    OffsetOverflow,
}

impl fmt::Display for MooncakeRequestBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroPageSize => formatter.write_str("Mooncake KV page size must be non-zero"),
            Self::ZeroRemoteKvItemSize => {
                formatter.write_str("Mooncake remote KV item size must be non-zero")
            }
            Self::MissingRemoteKvPointers => {
                formatter.write_str("Mooncake remote KV pointers must not be empty")
            }
            Self::SpanPageCountMismatch {
                expected_page_count,
                source_page_index_count,
            } => write!(
                formatter,
                "KV transfer span requires {expected_page_count} physical pages but has {source_page_index_count} source page indices"
            ),
            Self::RemoteKvIndexCountMismatch {
                source_page_count,
                dst_kv_index_count,
            } => write!(
                formatter,
                "KV transfer span has {source_page_count} source pages but {dst_kv_index_count} remote KV indices"
            ),
            Self::RemoteKvItemLayoutMismatch {
                page_size_bytes,
                split_page_size,
            } => write!(
                formatter,
                "Mooncake remote KV item layout requires exactly {page_size_bytes} bytes but has {split_page_size} bytes"
            ),
            Self::NegativeRemoteKvIndex(index) => {
                write!(
                    formatter,
                    "Mooncake remote KV index must be non-negative: {index}"
                )
            }
            Self::AddressOverflow => formatter.write_str("Mooncake source address overflow"),
            Self::OffsetOverflow => formatter.write_str("Mooncake target offset overflow"),
        }
    }
}

impl std::error::Error for MooncakeRequestBuildError {}

pub trait MooncakeTransferSubmitter {
    fn submit_transfer(
        &mut self,
        requests: &mut [MooncakeTransferRequest],
    ) -> Result<MooncakeBatchId, MooncakeError>;
}

pub trait MooncakeTransferStatusReader {
    fn transfer_status(
        &mut self,
        batch_id: MooncakeBatchId,
        task_id: usize,
    ) -> Result<MooncakeTransferStatus, MooncakeError>;
}

pub trait MooncakeBatchReleaser {
    fn free_batch(&mut self, batch_id: MooncakeBatchId) -> Result<(), MooncakeError>;
}

#[cfg(not(feature = "mooncake-link"))]
#[derive(Clone, Debug, Default)]
pub struct UnlinkedMooncakeTransferEngine;

#[cfg(not(feature = "mooncake-link"))]
impl MooncakeTransferSubmitter for UnlinkedMooncakeTransferEngine {
    fn submit_transfer(
        &mut self,
        _requests: &mut [MooncakeTransferRequest],
    ) -> Result<MooncakeBatchId, MooncakeError> {
        Err(MooncakeError::UnavailableWithoutLink)
    }
}

#[cfg(not(feature = "mooncake-link"))]
impl MooncakeTransferStatusReader for UnlinkedMooncakeTransferEngine {
    fn transfer_status(
        &mut self,
        _batch_id: MooncakeBatchId,
        _task_id: usize,
    ) -> Result<MooncakeTransferStatus, MooncakeError> {
        Err(MooncakeError::UnavailableWithoutLink)
    }
}

#[cfg(not(feature = "mooncake-link"))]
impl MooncakeBatchReleaser for UnlinkedMooncakeTransferEngine {
    fn free_batch(&mut self, _batch_id: MooncakeBatchId) -> Result<(), MooncakeError> {
        Err(MooncakeError::UnavailableWithoutLink)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MooncakeSubmittedBatch {
    bootstrap_room: BootstrapRoom,
    batch_id: MooncakeBatchId,
    task_count: usize,
    descriptor_checksum: String,
}

impl MooncakeSubmittedBatch {
    pub fn new(
        bootstrap_room: BootstrapRoom,
        batch_id: MooncakeBatchId,
        task_count: usize,
    ) -> Self {
        Self {
            bootstrap_room,
            batch_id,
            task_count,
            descriptor_checksum: String::new(),
        }
    }

    pub fn from_span(
        span: &KvCacheTransferSpan,
        batch_id: MooncakeBatchId,
        task_count: usize,
    ) -> Self {
        Self {
            bootstrap_room: span.bootstrap_room(),
            batch_id,
            task_count,
            descriptor_checksum: span.descriptor_checksum(),
        }
    }

    pub fn bootstrap_room(&self) -> BootstrapRoom {
        self.bootstrap_room
    }

    pub fn batch_id(&self) -> MooncakeBatchId {
        self.batch_id
    }

    pub fn task_count(&self) -> usize {
        self.task_count
    }

    pub fn descriptor_checksum(&self) -> &str {
        &self.descriptor_checksum
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MooncakeTransferPollSummary {
    completed_batches: usize,
    pending_batches: usize,
    completed_descriptor_checksums: Vec<String>,
    pending_descriptor_checksums: Vec<String>,
}

impl MooncakeTransferPollSummary {
    pub fn completed_batches(&self) -> usize {
        self.completed_batches
    }

    pub fn pending_batches(&self) -> usize {
        self.pending_batches
    }

    pub fn completed_descriptor_checksums(&self) -> &[String] {
        &self.completed_descriptor_checksums
    }

    pub fn pending_descriptor_checksums(&self) -> &[String] {
        &self.pending_descriptor_checksums
    }
}

pub fn poll_mooncake_transfer_batches<R>(
    registry: &mut DecodeBootstrapRegistry,
    reader: &mut R,
    submitted_batches: &[MooncakeSubmittedBatch],
) -> Result<MooncakeTransferPollSummary, KvCacheTransferError>
where
    R: MooncakeTransferStatusReader,
{
    let mut summary = MooncakeTransferPollSummary::default();

    for batch in submitted_batches {
        let mut completed_tasks = 0;

        for task_id in 0..batch.task_count() {
            let status = reader
                .transfer_status(batch.batch_id(), task_id)
                .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?;
            match MooncakeTransferStatusCode::try_from(status.status)? {
                MooncakeTransferStatusCode::Completed => {
                    completed_tasks += 1;
                }
                MooncakeTransferStatusCode::Waiting | MooncakeTransferStatusCode::Pending => {}
                MooncakeTransferStatusCode::Invalid
                | MooncakeTransferStatusCode::Canceled
                | MooncakeTransferStatusCode::Timeout
                | MooncakeTransferStatusCode::Failed => {
                    registry.update_status(batch.bootstrap_room(), KvPoll::Failed)?;
                    return Err(KvCacheTransferError::Runtime(format!(
                        "Mooncake transfer batch {} task {task_id} failed with status {}",
                        batch.batch_id(),
                        status.status
                    )));
                }
            }
        }

        if completed_tasks == batch.task_count() {
            registry.update_status(batch.bootstrap_room(), KvPoll::Success)?;
            summary.completed_batches += 1;
            if !batch.descriptor_checksum().is_empty() {
                summary
                    .completed_descriptor_checksums
                    .push(batch.descriptor_checksum().to_string());
            }
        } else {
            summary.pending_batches += 1;
            if !batch.descriptor_checksum().is_empty() {
                summary
                    .pending_descriptor_checksums
                    .push(batch.descriptor_checksum().to_string());
            }
        }
    }

    Ok(summary)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MooncakePolledBatchState {
    Completed,
    Pending,
}

fn poll_mooncake_submitted_batch<R>(
    registry: &mut DecodeBootstrapRegistry,
    reader: &mut R,
    batch: &MooncakeSubmittedBatch,
) -> Result<MooncakePolledBatchState, KvCacheTransferError>
where
    R: MooncakeTransferStatusReader,
{
    let mut completed_tasks = 0;

    for task_id in 0..batch.task_count() {
        let status = reader
            .transfer_status(batch.batch_id(), task_id)
            .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?;
        match MooncakeTransferStatusCode::try_from(status.status)? {
            MooncakeTransferStatusCode::Completed => {
                completed_tasks += 1;
            }
            MooncakeTransferStatusCode::Waiting | MooncakeTransferStatusCode::Pending => {}
            MooncakeTransferStatusCode::Invalid
            | MooncakeTransferStatusCode::Canceled
            | MooncakeTransferStatusCode::Timeout
            | MooncakeTransferStatusCode::Failed => {
                registry.update_status(batch.bootstrap_room(), KvPoll::Failed)?;
                return Err(KvCacheTransferError::Runtime(format!(
                    "Mooncake transfer batch {} task {task_id} failed with status {}",
                    batch.batch_id(),
                    status.status
                )));
            }
        }
    }

    if completed_tasks == batch.task_count() {
        registry.update_status(batch.bootstrap_room(), KvPoll::Success)?;
        Ok(MooncakePolledBatchState::Completed)
    } else {
        Ok(MooncakePolledBatchState::Pending)
    }
}

pub struct MooncakeKvCacheTransferExecutor<S, R = FixedMooncakeTransferTargetResolver> {
    local_memory_registration: Option<Box<dyn MooncakeMemoryRegistrationLease>>,
    submitter: S,
    layout: MooncakeKvCacheLayout,
    target_resolver: R,
    remote_kv_layouts_by_room: BTreeMap<BootstrapRoom, BTreeMap<String, MooncakeRemoteKvLayout>>,
    submitted_batches: Vec<MooncakeBatchId>,
    submitted_transfers: Vec<MooncakeSubmittedBatch>,
}

impl<S> MooncakeKvCacheTransferExecutor<S> {
    pub fn new(
        submitter: S,
        layout: MooncakeKvCacheLayout,
        target: MooncakeTransferTarget,
    ) -> Self {
        Self::with_target_resolver(
            submitter,
            layout,
            FixedMooncakeTransferTargetResolver::new(target),
        )
    }

    pub fn with_remote_kv_layouts(
        submitter: S,
        layout: MooncakeKvCacheLayout,
        target: MooncakeTransferTarget,
        remote_kv_layouts: Vec<(BootstrapRoom, MooncakeRemoteKvLayout)>,
    ) -> Self {
        Self::with_target_resolver_and_remote_kv_layouts(
            submitter,
            layout,
            FixedMooncakeTransferTargetResolver::new(target),
            remote_kv_layouts,
        )
    }
}

impl<S, R> MooncakeKvCacheTransferExecutor<S, R> {
    pub fn with_target_resolver(
        submitter: S,
        layout: MooncakeKvCacheLayout,
        target_resolver: R,
    ) -> Self {
        Self {
            local_memory_registration: None,
            submitter,
            layout,
            target_resolver,
            remote_kv_layouts_by_room: BTreeMap::new(),
            submitted_batches: Vec::new(),
            submitted_transfers: Vec::new(),
        }
    }

    pub fn with_target_resolver_and_remote_kv_layouts(
        submitter: S,
        layout: MooncakeKvCacheLayout,
        target_resolver: R,
        remote_kv_layouts: Vec<(BootstrapRoom, MooncakeRemoteKvLayout)>,
    ) -> Self {
        let mut remote_kv_layouts_by_room = BTreeMap::new();
        for (room, layout) in remote_kv_layouts {
            remote_kv_layouts_by_room
                .entry(room)
                .or_insert_with(BTreeMap::new)
                .insert(String::new(), layout);
        }

        Self {
            local_memory_registration: None,
            submitter,
            layout,
            target_resolver,
            remote_kv_layouts_by_room,
            submitted_batches: Vec::new(),
            submitted_transfers: Vec::new(),
        }
    }

    pub fn with_target_resolver_and_remote_kv_session_layouts(
        submitter: S,
        layout: MooncakeKvCacheLayout,
        target_resolver: R,
        remote_kv_layouts: Vec<(BootstrapRoom, String, MooncakeRemoteKvLayout)>,
    ) -> Self {
        let mut remote_kv_layouts_by_room = BTreeMap::new();
        for (room, session_id, layout) in remote_kv_layouts {
            remote_kv_layouts_by_room
                .entry(room)
                .or_insert_with(BTreeMap::new)
                .insert(session_id, layout);
        }

        Self {
            local_memory_registration: None,
            submitter,
            layout,
            target_resolver,
            remote_kv_layouts_by_room,
            submitted_batches: Vec::new(),
            submitted_transfers: Vec::new(),
        }
    }

    pub fn submitter(&self) -> &S {
        &self.submitter
    }

    pub fn with_local_memory_registration<L>(
        mut self,
        registration: L,
    ) -> Result<Self, KvCacheTransferError>
    where
        L: MooncakeMemoryRegistrationLease + 'static,
    {
        if self.local_memory_registration.is_some() {
            return Err(KvCacheTransferError::Runtime(
                "Mooncake executor already owns a local memory registration".to_string(),
            ));
        }
        let memory = registration.memory();
        if memory.page_size_bytes() != self.layout.page_size_bytes
            || memory.regions()[0].base_addr != self.layout.source_base_addr
        {
            return Err(KvCacheTransferError::Runtime(
                "Mooncake memory registration does not match the executor KV layout".to_string(),
            ));
        }

        self.local_memory_registration = Some(Box::new(registration));
        Ok(self)
    }

    pub fn has_local_memory_registration(&self) -> bool {
        self.local_memory_registration.is_some()
    }

    pub fn submitter_mut(&mut self) -> &mut S {
        &mut self.submitter
    }

    pub fn target_resolver(&self) -> &R {
        &self.target_resolver
    }

    pub fn target_resolver_mut(&mut self) -> &mut R {
        &mut self.target_resolver
    }

    pub fn submitted_batches(&self) -> &[MooncakeBatchId] {
        &self.submitted_batches
    }

    pub fn submitted_transfers(&self) -> &[MooncakeSubmittedBatch] {
        &self.submitted_transfers
    }

    pub fn remote_kv_layouts(
        &self,
    ) -> &BTreeMap<BootstrapRoom, BTreeMap<String, MooncakeRemoteKvLayout>> {
        &self.remote_kv_layouts_by_room
    }

    pub fn insert_remote_kv_layout(
        &mut self,
        bootstrap_room: BootstrapRoom,
        layout: MooncakeRemoteKvLayout,
    ) {
        self.insert_remote_kv_session_layout(bootstrap_room, String::new(), layout);
    }

    pub fn insert_remote_kv_session_layout(
        &mut self,
        bootstrap_room: BootstrapRoom,
        session_id: impl Into<String>,
        layout: MooncakeRemoteKvLayout,
    ) {
        self.remote_kv_layouts_by_room
            .entry(bootstrap_room)
            .or_default()
            .insert(session_id.into(), layout);
    }
}

impl<S, R> MooncakeKvCacheTransferExecutor<S, R>
where
    S: MooncakeTransferStatusReader + MooncakeBatchReleaser,
{
    pub fn cancel_submitted_transfers_for_room(
        &mut self,
        bootstrap_room: BootstrapRoom,
    ) -> Result<(), KvCacheTransferError> {
        let submitted_transfers = self.submitted_transfers.clone();
        let mut pending_transfers = Vec::new();

        for transfer in submitted_transfers {
            if transfer.bootstrap_room() == bootstrap_room {
                self.submitter
                    .free_batch(transfer.batch_id())
                    .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?;
            } else {
                pending_transfers.push(transfer);
            }
        }

        self.submitted_transfers = pending_transfers;
        self.submitted_batches = self
            .submitted_transfers
            .iter()
            .map(MooncakeSubmittedBatch::batch_id)
            .collect();

        Ok(())
    }

    pub fn poll_submitted_transfers(
        &mut self,
        registry: &mut DecodeBootstrapRegistry,
    ) -> Result<MooncakeTransferPollSummary, KvCacheTransferError> {
        let submitted_transfers = self.submitted_transfers.clone();
        let mut summary = MooncakeTransferPollSummary::default();
        let mut pending_transfers = Vec::new();
        let mut first_error = None;

        for transfer in submitted_transfers {
            match poll_mooncake_submitted_batch(registry, &mut self.submitter, &transfer) {
                Err(error) => {
                    let release_error = self
                        .submitter
                        .free_batch(transfer.batch_id())
                        .map_err(|error| KvCacheTransferError::Runtime(error.to_string()));
                    first_error = Some(match release_error {
                        Ok(()) => error,
                        Err(release_error) => release_error,
                    });
                }
                Ok(MooncakePolledBatchState::Completed) => {
                    self.submitter
                        .free_batch(transfer.batch_id())
                        .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?;
                    summary.completed_batches += 1;
                    if !transfer.descriptor_checksum().is_empty() {
                        summary
                            .completed_descriptor_checksums
                            .push(transfer.descriptor_checksum().to_string());
                    }
                }
                Ok(MooncakePolledBatchState::Pending) => {
                    if !transfer.descriptor_checksum().is_empty() {
                        summary
                            .pending_descriptor_checksums
                            .push(transfer.descriptor_checksum().to_string());
                    }
                    pending_transfers.push(transfer);
                    summary.pending_batches += 1;
                }
            }
        }

        self.submitted_transfers = pending_transfers;
        self.submitted_batches = self
            .submitted_transfers
            .iter()
            .map(MooncakeSubmittedBatch::batch_id)
            .collect();

        if let Some(error) = first_error {
            return Err(error);
        }

        Ok(summary)
    }
}

impl<S, R> KvCacheTransferExecutor for MooncakeKvCacheTransferExecutor<S, R>
where
    S: MooncakeTransferSubmitter + MooncakeTransferStatusReader + MooncakeBatchReleaser,
    R: MooncakeTransferTargetResolver,
{
    fn transfer_span(&mut self, span: &KvCacheTransferSpan) -> Result<(), KvCacheTransferError> {
        let mut requests = if let Some(remote_layouts) =
            self.remote_kv_layouts_by_room.get(&span.bootstrap_room())
        {
            let mut requests = Vec::new();
            for (session_id, remote_layout) in remote_layouts {
                let target = self
                    .target_resolver
                    .resolve_session_target(span, session_id)?;
                requests.extend(
                    build_mooncake_remote_kv_transfer_requests(
                        span,
                        self.layout,
                        target,
                        remote_layout,
                    )
                    .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?,
                );
            }
            requests
        } else {
            let target = self.target_resolver.resolve_target(span)?;
            build_mooncake_kv_transfer_requests(span, self.layout, target)
                .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?
        };
        if requests.is_empty() {
            return Ok(());
        }
        let task_count = requests.len();

        let batch_id = self
            .submitter
            .submit_transfer(&mut requests)
            .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?;
        self.submitted_batches.push(batch_id);
        self.submitted_transfers
            .push(MooncakeSubmittedBatch::from_span(
                span, batch_id, task_count,
            ));
        Ok(())
    }

    fn completes_inline(&self) -> bool {
        false
    }

    fn poll_transfers(
        &mut self,
        registry: &mut DecodeBootstrapRegistry,
    ) -> Result<MooncakeTransferPollSummary, KvCacheTransferError> {
        self.poll_submitted_transfers(registry)
    }

    fn cancel_transfer_room(
        &mut self,
        bootstrap_room: BootstrapRoom,
    ) -> Result<(), KvCacheTransferError> {
        self.cancel_submitted_transfers_for_room(bootstrap_room)
    }
}
#[cfg(feature = "mooncake-link")]
pub struct LinkedMooncakeTransferEngine {
    handle: MooncakeTransferEngineHandle,
}

// MooncakeTransferEngineHandle is an opaque C++ engine owner. Rust never dereferences
// the pointer directly, and shared use goes through SharedLinkedMooncakeTransferEngine's Mutex.
#[cfg(feature = "mooncake-link")]
unsafe impl Send for LinkedMooncakeTransferEngine {}

#[cfg(feature = "mooncake-link")]
impl LinkedMooncakeTransferEngine {
    pub fn new(config: &MooncakeTransferEngineConfig) -> Result<Self, MooncakeError> {
        let metadata = CString::new(config.metadata_server.as_str())?;
        let local_server = CString::new(config.session_id.as_str())?;
        let host = CString::new(config.hostname.as_str())?;

        let auto_discover = if config.protocol == "tcp" { 0 } else { 1 };
        let handle = unsafe {
            createTransferEngine(
                metadata.as_ptr(),
                local_server.as_ptr(),
                host.as_ptr(),
                u64::from(config.rpc_port),
                auto_discover,
            )
        };
        if handle.is_null() {
            return Err(MooncakeError::EngineCreateFailed);
        }

        let engine = Self { handle };
        engine.install_transport(config.protocol.as_str())?;
        Ok(engine)
    }

    pub fn handle(&self) -> MooncakeTransferEngineHandle {
        self.handle
    }

    pub fn local_endpoint(&self) -> Result<String, MooncakeError> {
        let mut buffer = vec![0_u8; 256];
        let code = unsafe {
            getLocalIpAndPort(
                self.handle,
                buffer.as_mut_ptr().cast::<c_char>(),
                buffer.len(),
            )
        };
        if code != 0 {
            return Err(MooncakeError::LocalEndpointQueryFailed(code));
        }
        let endpoint = CStr::from_bytes_until_nul(&buffer)
            .map_err(|_| MooncakeError::LocalEndpointUtf8)?
            .to_str()
            .map_err(|_| MooncakeError::LocalEndpointUtf8)?;
        Ok(endpoint.to_string())
    }

    pub fn install_transport(&self, protocol: &str) -> Result<(), MooncakeError> {
        let protocol_c = CString::new(protocol)?;
        let transport =
            unsafe { installTransport(self.handle, protocol_c.as_ptr(), std::ptr::null_mut()) };
        if transport.is_null() {
            return Err(MooncakeError::TransportInstallFailed(protocol.to_string()));
        }
        Ok(())
    }

    pub fn register_memory_batch(
        &self,
        buffers: &mut [MooncakeBufferEntry],
        location: &str,
    ) -> Result<(), MooncakeError> {
        let location_c = CString::new(location)?;
        let code = unsafe {
            registerLocalMemoryBatch(
                self.handle,
                buffers.as_mut_ptr(),
                buffers.len(),
                location_c.as_ptr(),
            )
        };
        if code != 0 {
            return Err(MooncakeError::RegisterMemoryFailed(code));
        }
        Ok(())
    }

    pub fn unregister_memory_batch(&self, addrs: &mut [*mut c_void]) -> Result<(), MooncakeError> {
        let code =
            unsafe { unregisterLocalMemoryBatch(self.handle, addrs.as_mut_ptr(), addrs.len()) };
        if code != 0 {
            return Err(MooncakeError::UnregisterMemoryFailed(code));
        }
        Ok(())
    }

    pub fn open_segment(&self, segment: &str) -> Result<MooncakeSegmentId, MooncakeError> {
        let segment_c = CString::new(segment)?;
        let segment_id = unsafe { openSegment(self.handle, segment_c.as_ptr()) };
        if segment_id < 0 {
            return Err(MooncakeError::OpenSegmentFailed(segment.to_string()));
        }
        Ok(segment_id)
    }

    pub fn submit_transfer(
        &self,
        requests: &mut [MooncakeTransferRequest],
    ) -> Result<MooncakeBatchId, MooncakeError> {
        let batch_id = unsafe { allocateBatchID(self.handle, requests.len()) };
        let code =
            unsafe { submitTransfer(self.handle, batch_id, requests.as_mut_ptr(), requests.len()) };
        if code != 0 {
            let _ = unsafe { freeBatchID(self.handle, batch_id) };
            return Err(MooncakeError::SubmitTransferFailed(code));
        }
        Ok(batch_id)
    }

    pub fn transfer_status(
        &self,
        batch_id: MooncakeBatchId,
        task_id: usize,
    ) -> Result<MooncakeTransferStatus, MooncakeError> {
        let mut status = MooncakeTransferStatus {
            status: MooncakeTransferStatusCode::Waiting as c_int,
            transferred_bytes: 0,
        };
        let code = unsafe { getTransferStatus(self.handle, batch_id, task_id, &mut status) };
        if code != 0 {
            return Err(MooncakeError::StatusQueryFailed(code));
        }
        Ok(status)
    }

    pub fn free_batch(&self, batch_id: MooncakeBatchId) -> Result<(), MooncakeError> {
        let code = unsafe { freeBatchID(self.handle, batch_id) };
        if code != 0 {
            return Err(MooncakeError::FreeBatchFailed(code));
        }
        Ok(())
    }
}

#[cfg(feature = "mooncake-link")]
impl Drop for LinkedMooncakeTransferEngine {
    fn drop(&mut self) {
        unsafe { destroyTransferEngine(self.handle) };
    }
}

#[cfg(feature = "mooncake-link")]
impl MooncakeTransferSubmitter for LinkedMooncakeTransferEngine {
    fn submit_transfer(
        &mut self,
        requests: &mut [MooncakeTransferRequest],
    ) -> Result<MooncakeBatchId, MooncakeError> {
        LinkedMooncakeTransferEngine::submit_transfer(self, requests)
    }
}

#[cfg(feature = "mooncake-link")]
impl MooncakeSegmentOpener for LinkedMooncakeTransferEngine {
    fn open_segment(&mut self, segment: &str) -> Result<i32, MooncakeError> {
        LinkedMooncakeTransferEngine::open_segment(self, segment)
    }
}

#[cfg(feature = "mooncake-link")]
impl MooncakeTransferStatusReader for LinkedMooncakeTransferEngine {
    fn transfer_status(
        &mut self,
        batch_id: MooncakeBatchId,
        task_id: usize,
    ) -> Result<MooncakeTransferStatus, MooncakeError> {
        LinkedMooncakeTransferEngine::transfer_status(self, batch_id, task_id)
    }
}

#[cfg(feature = "mooncake-link")]
impl MooncakeBatchReleaser for LinkedMooncakeTransferEngine {
    fn free_batch(&mut self, batch_id: MooncakeBatchId) -> Result<(), MooncakeError> {
        LinkedMooncakeTransferEngine::free_batch(self, batch_id)
    }
}

#[cfg(feature = "mooncake-link")]
#[derive(Clone)]
pub struct SharedLinkedMooncakeTransferEngine {
    inner: Arc<Mutex<LinkedMooncakeTransferEngine>>,
}

#[cfg(feature = "mooncake-link")]
impl SharedLinkedMooncakeTransferEngine {
    pub fn new(config: &MooncakeTransferEngineConfig) -> Result<Self, MooncakeError> {
        Ok(Self {
            inner: Arc::new(Mutex::new(LinkedMooncakeTransferEngine::new(config)?)),
        })
    }

    pub fn local_endpoint(&self) -> Result<String, MooncakeError> {
        self.inner
            .lock()
            .expect("linked Mooncake engine lock should be held")
            .local_endpoint()
    }

    pub fn register_memory_batch(
        &self,
        buffers: &mut [MooncakeBufferEntry],
        location: &str,
    ) -> Result<(), MooncakeError> {
        self.inner
            .lock()
            .expect("linked Mooncake engine lock should be held")
            .register_memory_batch(buffers, location)
    }

    pub fn unregister_memory_batch(&self, addrs: &mut [*mut c_void]) -> Result<(), MooncakeError> {
        self.inner
            .lock()
            .expect("linked Mooncake engine lock should be held")
            .unregister_memory_batch(addrs)
    }
}

#[cfg(feature = "mooncake-link")]
impl MooncakeTransferSubmitter for SharedLinkedMooncakeTransferEngine {
    fn submit_transfer(
        &mut self,
        requests: &mut [MooncakeTransferRequest],
    ) -> Result<MooncakeBatchId, MooncakeError> {
        self.inner
            .lock()
            .expect("linked Mooncake engine lock should be held")
            .submit_transfer(requests)
    }
}

#[cfg(feature = "mooncake-link")]
impl MooncakeSegmentOpener for SharedLinkedMooncakeTransferEngine {
    fn open_segment(&mut self, segment: &str) -> Result<i32, MooncakeError> {
        self.inner
            .lock()
            .expect("linked Mooncake engine lock should be held")
            .open_segment(segment)
    }
}

#[cfg(feature = "mooncake-link")]
impl MooncakeTransferStatusReader for SharedLinkedMooncakeTransferEngine {
    fn transfer_status(
        &mut self,
        batch_id: MooncakeBatchId,
        task_id: usize,
    ) -> Result<MooncakeTransferStatus, MooncakeError> {
        self.inner
            .lock()
            .expect("linked Mooncake engine lock should be held")
            .transfer_status(batch_id, task_id)
    }
}

#[cfg(feature = "mooncake-link")]
impl MooncakeBatchReleaser for SharedLinkedMooncakeTransferEngine {
    fn free_batch(&mut self, batch_id: MooncakeBatchId) -> Result<(), MooncakeError> {
        self.inner
            .lock()
            .expect("linked Mooncake engine lock should be held")
            .free_batch(batch_id)
    }
}

#[cfg(feature = "mooncake-link")]
impl MooncakeMemoryRegistrar for SharedLinkedMooncakeTransferEngine {
    fn register_memory_batch(
        &mut self,
        buffers: &mut [MooncakeBufferEntry],
        location: &str,
    ) -> Result<(), MooncakeError> {
        SharedLinkedMooncakeTransferEngine::register_memory_batch(self, buffers, location)
    }

    fn unregister_memory_batch(&mut self, addrs: &mut [*mut c_void]) -> Result<(), MooncakeError> {
        SharedLinkedMooncakeTransferEngine::unregister_memory_batch(self, addrs)
    }
}

unsafe extern "C" {
    pub fn createTransferEngine(
        metadata_conn_string: *const c_char,
        local_server_name: *const c_char,
        ip_or_host_name: *const c_char,
        rpc_port: u64,
        auto_discover: c_int,
    ) -> MooncakeTransferEngineHandle;

    pub fn destroyTransferEngine(engine: MooncakeTransferEngineHandle);

    pub fn getLocalIpAndPort(
        engine: MooncakeTransferEngineHandle,
        buf_out: *mut c_char,
        buf_len: usize,
    ) -> c_int;

    pub fn installTransport(
        engine: MooncakeTransferEngineHandle,
        proto: *const c_char,
        args: *mut *mut c_void,
    ) -> MooncakeTransportHandle;

    pub fn registerLocalMemoryBatch(
        engine: MooncakeTransferEngineHandle,
        buffer_list: *mut MooncakeBufferEntry,
        buffer_len: usize,
        location: *const c_char,
    ) -> c_int;

    pub fn unregisterLocalMemoryBatch(
        engine: MooncakeTransferEngineHandle,
        addr_list: *mut *mut c_void,
        addr_len: usize,
    ) -> c_int;

    pub fn openSegment(
        engine: MooncakeTransferEngineHandle,
        segment_name: *const c_char,
    ) -> MooncakeSegmentId;

    pub fn closeSegment(
        engine: MooncakeTransferEngineHandle,
        segment_id: MooncakeSegmentId,
    ) -> c_int;

    pub fn allocateBatchID(
        engine: MooncakeTransferEngineHandle,
        batch_size: usize,
    ) -> MooncakeBatchId;

    pub fn submitTransfer(
        engine: MooncakeTransferEngineHandle,
        batch_id: MooncakeBatchId,
        entries: *mut MooncakeTransferRequest,
        count: usize,
    ) -> c_int;

    pub fn getTransferStatus(
        engine: MooncakeTransferEngineHandle,
        batch_id: MooncakeBatchId,
        task_id: usize,
        status: *mut MooncakeTransferStatus,
    ) -> c_int;

    pub fn freeBatchID(engine: MooncakeTransferEngineHandle, batch_id: MooncakeBatchId) -> c_int;
}
