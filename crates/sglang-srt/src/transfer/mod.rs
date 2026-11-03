pub mod backend_factory;
pub mod engine_info;
pub mod pd_bootstrap;
pub mod pd_control;
pub mod state_events;
pub mod tcp;

pub use state_events::{
    BlockRemovedEvent, BlockStoredEvent, BroadcastKvStateEventPublisher,
    CompositeKvStateEventPublisher, KvStateEventPublisher, NoopKvStateEventPublisher,
    RecordingKvStateEventPublisher, ReusableBoundary, ZmqKvStateEventPublisher,
};

use std::collections::BTreeMap;
use std::ffi::{c_int, c_void};
use std::fmt;

use sha2::{Digest, Sha256};

pub use nexus_transfer::{
    KvCacheMemoryLocation, KvCacheMemoryProvider, TransferableKvCacheMemory,
    TransferableKvCacheMemoryError, TransferableKvCacheRegion,
};

use crate::cache::CachePageId;
use crate::cli::{ServerArgs, ZmqPortRange};
pub use crate::kv_cache::{KvCacheDtype, KvCacheModelLayout, KvCacheRuntimeLayout};
#[cfg(any(test, feature = "test-support"))]
use crate::model_executor::ModelRunner;
use crate::model_executor::ModelWorkerBatch;
use crate::scheduler::{ForwardMode, ScheduleBatch, ScheduledRequest};
use crate::types::{BootstrapRoom, DisaggregatedParams, RequestId};
use crate::worker::{
    BatchGeneratedTokens, DecodeRequestState, FallibleModelWorker, GeneratedToken,
    WorkerExecutionError, WorkerWeightUpdateRequest,
};

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
    Tcp,
    Nixl,
    Ascend,
    Mori,
}

impl TransferBackend {
    fn parse(value: &str) -> Result<ParsedTransferBackend, PdConfigError> {
        match value {
            "tcp" => Ok(ParsedTransferBackend {
                backend: Self::Tcp,
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

#[derive(Clone, Debug, PartialEq)]
pub struct PdConfig {
    pub mode: DisaggregationMode,
    pub transfer_backend: TransferBackend,
    pub force_tcp_transport: bool,
    pub bootstrap_port: u16,
    pub transfer_rpc_port: Option<u16>,
    pub ib_device: Option<String>,
    pub decode_enable_radix_cache: bool,
    pub decode_enable_offload_kvcache: bool,
    pub decode_polling_interval: usize,
    pub kv_cache_dtype: KvCacheDtype,
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
    pub kv_events_port: Option<u16>,
    pub enable_metrics: bool,
    pub disable_radix_cache: bool,
    pub tool_call_parser: Option<String>,
}

impl PdConfig {
    pub fn from_server_args(args: &ServerArgs) -> Result<Self, PdConfigError> {
        let mode = DisaggregationMode::parse(&args.disaggregation_mode)?;
        let backend = TransferBackend::parse(&args.disaggregation_transfer_backend)?;
        let kv_cache_dtype = KvCacheDtype::parse(&args.kv_cache_dtype)
            .ok_or_else(|| PdConfigError::InvalidKvCacheDtype(args.kv_cache_dtype.clone()))?;

        Ok(Self {
            mode,
            transfer_backend: backend.backend,
            force_tcp_transport: backend.force_tcp_transport,
            bootstrap_port: args.disaggregation_bootstrap_port,
            transfer_rpc_port: args.disaggregation_transfer_rpc_port,
            ib_device: if backend.force_tcp_transport {
                None
            } else {
                args.disaggregation_ib_device.clone()
            },
            decode_enable_radix_cache: args.disaggregation_decode_enable_radix_cache,
            decode_enable_offload_kvcache: args.disaggregation_decode_enable_offload_kvcache,
            decode_polling_interval: args.disaggregation_decode_polling_interval,
            kv_cache_dtype,
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
            kv_events_port: args.kv_events_port,
            enable_metrics: args.enable_metrics,
            disable_radix_cache: args.disable_radix_cache,
            tool_call_parser: args.tool_call_parser.clone(),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PdConfigError {
    InvalidDisaggregationMode(String),
    InvalidTransferBackend(String),
    InvalidKvCacheDtype(String),
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
        }
    }
}

impl std::error::Error for PdConfigError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvTransportError {
    Internal(String),
    ConnectionFailed(String),
    UnsupportedMemoryLocation(KvCacheMemoryLocation),
    RegisterMemoryFailed(String),
    UnregisterMemoryFailed(String),
    OpenSegmentFailed(String),
    SubmitTransferFailed(String),
    StatusQueryFailed(String),
    FreeBatchFailed(String),
}

impl fmt::Display for KvTransportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Internal(msg) => write!(formatter, "KV transport internal error: {msg}"),
            Self::ConnectionFailed(msg) => {
                write!(formatter, "KV transport connection failed: {msg}")
            }
            Self::UnsupportedMemoryLocation(loc) => {
                write!(formatter, "unsupported KV memory location: {loc:?}")
            }
            Self::RegisterMemoryFailed(msg) => {
                write!(formatter, "KV memory register failed: {msg}")
            }
            Self::UnregisterMemoryFailed(msg) => {
                write!(formatter, "KV memory unregister failed: {msg}")
            }
            Self::OpenSegmentFailed(msg) => write!(formatter, "KV open segment failed: {msg}"),
            Self::SubmitTransferFailed(msg) => {
                write!(formatter, "KV transfer submit failed: {msg}")
            }
            Self::StatusQueryFailed(msg) => write!(formatter, "KV status query failed: {msg}"),
            Self::FreeBatchFailed(msg) => write!(formatter, "KV free batch failed: {msg}"),
        }
    }
}

impl std::error::Error for KvTransportError {}

pub const PD_P2P_HANDSHAKE_METADATA: &str = "P2PHANDSHAKE";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PdTransferEngineConfig {
    pub hostname: String,
    pub gpu_id: usize,
    pub rpc_port: u16,
    pub session_id: String,
    pub metadata_server: String,
    pub protocol: String,
    pub device_name: String,
}

impl PdTransferEngineConfig {
    pub fn from_pd_config(hostname: impl Into<String>, gpu_id: usize, config: &PdConfig) -> Self {
        let hostname = hostname.into();
        let ranks_per_node = config
            .tp_size
            .checked_div(config.nnodes.max(1))
            .unwrap_or(1);
        let rank_offset = u16::try_from(config.node_rank * ranks_per_node).unwrap_or(0);
        let rpc_port = config
            .transfer_rpc_port
            .map(|p| {
                if p == 0 {
                    0
                } else {
                    p.saturating_add(rank_offset)
                }
            })
            .unwrap_or(0);
        let session_id = format!("{hostname}:{rpc_port}");
        Self {
            hostname,
            gpu_id,
            rpc_port,
            session_id,
            metadata_server: PD_P2P_HANDSHAKE_METADATA.to_string(),
            protocol: if config.force_tcp_transport {
                "tcp".to_string()
            } else {
                "rdma".to_string()
            },
            device_name: config.ib_device.clone().unwrap_or_default(),
        }
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
    first_token: Option<u32>,
    recurrent_state: Option<Vec<u8>>,
}

impl KvCacheTransferSpan {
    pub fn first_token(&self) -> Option<u32> {
        self.first_token
    }

    pub fn recurrent_state(&self) -> Option<&[u8]> {
        self.recurrent_state.as_deref()
    }

    pub fn with_recurrent_state(mut self, recurrent_state: Option<Vec<u8>>) -> Self {
        self.recurrent_state = recurrent_state;
        self
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
                first_token: None,
                recurrent_state: None,
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

    pub fn spans_mut(&mut self) -> &mut [KvCacheTransferSpan] {
        &mut self.spans
    }

    fn with_prefill_output(
        mut self,
        batch: &ScheduleBatch,
        output: &BatchGeneratedTokens,
    ) -> Result<Self, WorkerExecutionError> {
        for span in &mut self.spans {
            let index = batch
                .requests()
                .iter()
                .position(|request| request.request_id() == span.request_id())
                .ok_or_else(|| {
                    WorkerExecutionError::Runtime(
                        "transfer span has no matching prefill request".to_string(),
                    )
                })?;
            let token_ids = output.tokens()[index].token_ids();
            if token_ids.len() != 1 {
                return Err(WorkerExecutionError::Runtime(
                    "single-rank PD requires exactly one prefill output token".to_string(),
                ));
            }
            span.first_token = Some(token_ids[0]);
        }
        Ok(self)
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

    fn receive_prefill_token(&mut self, room: BootstrapRoom) -> Result<u32, String> {
        self.receive_prefill_token_and_state(room)
            .map(|(token, _)| token)
    }

    fn receive_prefill_token_and_state(
        &mut self,
        _room: BootstrapRoom,
    ) -> Result<(u32, Option<Vec<u8>>), String> {
        Err("decode bootstrap publisher has no receive/completion implementation".to_string())
    }

    fn cancel_receive(&mut self, _room: BootstrapRoom) -> Result<(), String> {
        Err(
            "decode bootstrap publisher has no safe receive cancellation implementation"
                .to_string(),
        )
    }
}

#[cfg(any(test, feature = "test-support"))]
pub trait KvCachePageSnapshotProvider {
    type Snapshot;

    fn export_kv_cache_pages(
        &self,
        cache_pages: &[CachePageId],
    ) -> Result<Vec<Self::Snapshot>, KvCacheTransferError>;
}

#[cfg(any(test, feature = "test-support"))]
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

#[cfg(any(test, feature = "test-support"))]
impl KvCachePageSnapshotChecksum for CachePageId {
    fn update_content_checksum(&self, hasher: &mut Sha256) {
        hasher.update((self.as_usize() as u64).to_le_bytes());
    }
}

#[cfg(any(test, feature = "test-support"))]
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

#[cfg(any(test, feature = "test-support"))]
pub trait KvCachePageSnapshotImporter {
    type Snapshot;

    fn import_kv_cache_pages(
        &mut self,
        snapshots: Vec<Self::Snapshot>,
    ) -> Result<(), KvCacheTransferError>;
}

#[cfg(any(test, feature = "test-support"))]
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

#[cfg(any(test, feature = "test-support"))]
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

#[cfg(any(test, feature = "test-support"))]
pub struct LocalSnapshotTransferPdModelWorkers<P, D> {
    prefill: P,
    decode: D,
    last_transfer_summary: Option<KvCacheTransferSummary>,
}

#[cfg(any(test, feature = "test-support"))]
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

#[cfg(any(test, feature = "test-support"))]
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
            ForwardMode::Mixed => self.try_generate_prefill_batch(batch),
        }
    }

    fn decode_request_state(
        &self,
        request: &ScheduledRequest,
    ) -> Result<DecodeRequestState, WorkerExecutionError> {
        self.decode.decode_request_state(request)
    }

    fn poll_transfers(&mut self) -> Result<KvTransferPollSummary, KvCacheTransferError> {
        Ok(KvTransferPollSummary::default())
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

    fn export_recurrent_state(
        &self,
        request_id: &RequestId,
    ) -> Result<Option<Vec<u8>>, WorkerExecutionError> {
        self.decode.export_recurrent_state(request_id)
    }

    fn restore_recurrent_state(
        &mut self,
        request_id: &RequestId,
        state: &[u8],
    ) -> Result<(), WorkerExecutionError> {
        self.decode.restore_recurrent_state(request_id, state)
    }
}

#[cfg(any(test, feature = "test-support"))]
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
            if let Some(state) = self.prefill.export_recurrent_state(span.request_id())? {
                self.decode
                    .restore_recurrent_state(span.request_id(), &state)?;
            }
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvTransferEventStatus {
    Completed,
    Failed(String),
    Canceled,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvTransferCompletionEvent {
    bootstrap_room: BootstrapRoom,
    descriptor_checksum: String,
    status: KvTransferEventStatus,
}

impl KvTransferCompletionEvent {
    pub fn completed(bootstrap_room: BootstrapRoom, descriptor_checksum: String) -> Self {
        Self {
            bootstrap_room,
            descriptor_checksum,
            status: KvTransferEventStatus::Completed,
        }
    }

    pub fn failed(
        bootstrap_room: BootstrapRoom,
        descriptor_checksum: String,
        message: String,
    ) -> Self {
        Self {
            bootstrap_room,
            descriptor_checksum,
            status: KvTransferEventStatus::Failed(message),
        }
    }

    pub fn canceled(bootstrap_room: BootstrapRoom, descriptor_checksum: String) -> Self {
        Self {
            bootstrap_room,
            descriptor_checksum,
            status: KvTransferEventStatus::Canceled,
        }
    }

    pub fn bootstrap_room(&self) -> BootstrapRoom {
        self.bootstrap_room
    }

    pub fn descriptor_checksum(&self) -> &str {
        &self.descriptor_checksum
    }

    pub fn status(&self) -> &KvTransferEventStatus {
        &self.status
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KvTransferPoll {
    events: Vec<KvTransferCompletionEvent>,
    pending_transfers: usize,
}

impl KvTransferPoll {
    pub fn new(events: Vec<KvTransferCompletionEvent>, pending_transfers: usize) -> Self {
        Self {
            events,
            pending_transfers,
        }
    }

    pub fn events(&self) -> &[KvTransferCompletionEvent] {
        &self.events
    }

    pub fn pending_transfers(&self) -> usize {
        self.pending_transfers
    }
}

pub trait KvTransferBackend {
    fn register(&mut self, memory: TransferableKvCacheMemory) -> Result<(), KvCacheTransferError>;
    fn submit(&mut self, span: &KvCacheTransferSpan) -> Result<(), KvCacheTransferError>;
    fn poll(&mut self) -> Result<KvTransferPoll, KvCacheTransferError>;
    fn cancel(&mut self, bootstrap_room: BootstrapRoom) -> Result<(), KvCacheTransferError>;
    fn shutdown(&mut self) -> Result<(), KvCacheTransferError>;
}

#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FakeKvCacheTransferExecutor {
    registered_memory: Option<TransferableKvCacheMemory>,
    transferred_rooms: Vec<BootstrapRoom>,
    pending: Vec<(BootstrapRoom, String)>,
}

#[cfg(any(test, feature = "test-support"))]
impl FakeKvCacheTransferExecutor {
    pub fn transferred_rooms(&self) -> &[BootstrapRoom] {
        &self.transferred_rooms
    }
}

#[cfg(any(test, feature = "test-support"))]
impl KvTransferBackend for FakeKvCacheTransferExecutor {
    fn register(&mut self, memory: TransferableKvCacheMemory) -> Result<(), KvCacheTransferError> {
        self.registered_memory = Some(memory);
        Ok(())
    }

    fn submit(&mut self, span: &KvCacheTransferSpan) -> Result<(), KvCacheTransferError> {
        self.transferred_rooms.push(span.bootstrap_room());
        self.pending
            .push((span.bootstrap_room(), span.descriptor_checksum()));
        Ok(())
    }

    fn poll(&mut self) -> Result<KvTransferPoll, KvCacheTransferError> {
        Ok(KvTransferPoll {
            events: self
                .pending
                .drain(..)
                .map(|(room, checksum)| KvTransferCompletionEvent::completed(room, checksum))
                .collect(),
            pending_transfers: 0,
        })
    }

    fn cancel(&mut self, bootstrap_room: BootstrapRoom) -> Result<(), KvCacheTransferError> {
        self.pending.retain(|(room, _)| *room != bootstrap_room);
        Ok(())
    }

    fn shutdown(&mut self) -> Result<(), KvCacheTransferError> {
        self.pending.clear();
        self.registered_memory = None;
        Ok(())
    }
}

pub fn execute_kv_cache_transfer_plan<E>(
    registry: &mut DecodeBootstrapRegistry,
    executor: &mut E,
    plan: &KvCacheTransferPlan,
) -> Result<KvCacheTransferSummary, KvCacheTransferError>
where
    E: KvTransferBackend,
{
    let mut summary = KvCacheTransferSummary::default();

    for span in plan.spans() {
        if span.is_noop() && span.first_token().is_none() {
            registry.update_status(span.bootstrap_room(), KvPoll::Success)?;
            summary.noop_spans += 1;
            continue;
        }

        registry.update_status(span.bootstrap_room(), KvPoll::Transferring)?;
        if let Err(error) = executor.submit(span) {
            registry.update_status(span.bootstrap_room(), KvPoll::Failed)?;
            return Err(error);
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
    decode_bootstrap_publisher: P,
    transfer_executor: E,
    worker: W,
    registry: DecodeBootstrapRegistry,
    kv_page_size: usize,
    submit_prefill_transfers: bool,
    prefill_only: bool,
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
            prefill_only: false,
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
            prefill_only: self.prefill_only,
            last_transfer_summary: self.last_transfer_summary,
        }
    }

    pub fn with_decode_side_bootstrap_only(mut self) -> Self {
        self.submit_prefill_transfers = false;
        self
    }

    pub fn with_prefill_only(mut self) -> Self {
        self.prefill_only = true;
        self
    }

    pub fn with_kv_page_size(mut self, kv_page_size: usize) -> Self {
        self.kv_page_size = kv_page_size;
        self
    }

    pub fn last_transfer_summary(&self) -> Option<&KvCacheTransferSummary> {
        self.last_transfer_summary.as_ref()
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn export_kv_cache_pages(
        &self,
        cache_pages: &[CachePageId],
    ) -> Result<Vec<W::Snapshot>, KvCacheTransferError>
    where
        W: KvCachePageSnapshotProvider,
    {
        self.worker.export_kv_cache_pages(cache_pages)
    }

    #[cfg(any(test, feature = "test-support"))]
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
    E: KvTransferBackend,
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

        if !self.submit_prefill_transfers
            && let Some(plan) = transfer_plan.as_ref()
        {
            self.register_prefill_bootstrap_sessions(batch)
                .map_err(|error| WorkerExecutionError::Runtime(error.to_string()))?;
            let mut tokens = Vec::with_capacity(batch.batch_size());
            for request in batch.requests() {
                let room = request
                    .disaggregated_params()
                    .ok_or_else(|| {
                        WorkerExecutionError::Runtime(
                            "decode PD request requires bootstrap metadata".to_string(),
                        )
                    })?
                    .bootstrap_room;
                let (token, recurrent_state) = self
                    .decode_bootstrap_publisher
                    .receive_prefill_token_and_state(room)
                    .map_err(WorkerExecutionError::Runtime)?;
                if let Some(state_bytes) = recurrent_state {
                    self.worker
                        .restore_recurrent_state(request.request_id(), &state_bytes)?;
                }
                self.registry
                    .update_status(room, KvPoll::Success)
                    .map_err(|error| WorkerExecutionError::Runtime(error.to_string()))?;
                let stop = !request.sampling().ignore_eos
                    && request.sampling().stop_token_ids.contains(&token);
                tokens.push(if stop {
                    GeneratedToken::finished(vec![token])
                } else {
                    GeneratedToken::unfinished(vec![token])
                });
            }
            self.last_transfer_summary = Some(KvCacheTransferSummary {
                noop_spans: plan.spans().iter().filter(|span| span.is_noop()).count(),
                ..KvCacheTransferSummary::default()
            });
            return BatchGeneratedTokens::from_batch(batch, tokens).map_err(Into::into);
        }

        let output = self.worker.try_generate_batch(batch)?;

        if let Some(transfer_plan) = transfer_plan {
            self.register_prefill_bootstrap_sessions(batch)
                .map_err(|error| {
                    WorkerExecutionError::Runtime(format!(
                        "KV transfer bootstrap registration failed: {error}"
                    ))
                })?;
            let mut transfer_plan = transfer_plan.with_prefill_output(batch, &output)?;
            for span in transfer_plan.spans_mut() {
                span.recurrent_state = self.worker.export_recurrent_state(span.request_id())?;
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
            if self.prefill_only {
                while batch.requests().iter().any(|request| {
                    request.disaggregated_params().is_some_and(|params| {
                        self.registry
                            .get(params.bootstrap_room)
                            .is_some_and(|session| session.status() != KvPoll::Success)
                    })
                }) {
                    self.poll_transfers()
                        .map_err(|error| WorkerExecutionError::Runtime(error.to_string()))?;
                    for request in batch.requests() {
                        if matches!(
                            self.decode_request_state(request)?,
                            DecodeRequestState::Failed(_)
                        ) {
                            return Err(WorkerExecutionError::Runtime(
                                "prefill KV transfer failed".to_string(),
                            ));
                        }
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                return BatchGeneratedTokens::from_batch(
                    batch,
                    output
                        .tokens()
                        .iter()
                        .map(|token| GeneratedToken::finished(token.token_ids().to_vec()))
                        .collect(),
                )
                .map_err(Into::into);
            }
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

    fn poll_transfers(&mut self) -> Result<KvTransferPollSummary, KvCacheTransferError> {
        let poll = self.transfer_executor.poll()?;
        apply_transfer_completion_events(&mut self.registry, poll)
    }

    fn complete_request(&mut self, request: &ScheduledRequest) {
        if let Err(error) = self.finish_request(request) {
            eprintln!("PD request cleanup deferred: {error}");
        }
    }

    fn finish_request(&mut self, request: &ScheduledRequest) -> Result<(), WorkerExecutionError> {
        if let Some(params) = request.disaggregated_params() {
            if !self.submit_prefill_transfers {
                self.decode_bootstrap_publisher
                    .cancel_receive(params.bootstrap_room)
                    .map_err(WorkerExecutionError::Runtime)?;
            }
            self.transfer_executor
                .cancel(params.bootstrap_room)
                .map_err(|error| WorkerExecutionError::Runtime(error.to_string()))?;
            self.registry.remove(params.bootstrap_room);
        }
        self.worker.finish_request(request)
    }

    fn fail_request(&mut self, request: &ScheduledRequest) {
        if let Err(error) = self.finish_request(request) {
            eprintln!("PD failure cleanup deferred; KV memory remains reserved: {error}");
            return;
        }
        self.worker.fail_request(request)
    }

    fn shutdown(&mut self) -> Result<(), WorkerExecutionError> {
        self.transfer_executor.shutdown().map_err(|error| {
            WorkerExecutionError::Runtime(format!("KV transfer backend shutdown failed: {error}"))
        })?;
        self.registry = DecodeBootstrapRegistry::default();
        self.worker.shutdown()
    }

    fn update_weights_from_disk(
        &mut self,
        request: &WorkerWeightUpdateRequest,
    ) -> Result<(), WorkerExecutionError> {
        self.worker.update_weights_from_disk(request)
    }

    fn export_recurrent_state(
        &self,
        request_id: &RequestId,
    ) -> Result<Option<Vec<u8>>, WorkerExecutionError> {
        self.worker.export_recurrent_state(request_id)
    }

    fn restore_recurrent_state(
        &mut self,
        request_id: &RequestId,
        state: &[u8],
    ) -> Result<(), WorkerExecutionError> {
        self.worker.restore_recurrent_state(request_id, state)
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum KvOpcode {
    Read = 0,
    Write = 1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum KvTransferStatusCode {
    Waiting = 0,
    Pending = 1,
    Invalid = 2,
    Canceled = 3,
    Completed = 4,
    Timeout = 5,
    Failed = 6,
}

impl TryFrom<c_int> for KvTransferStatusCode {
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
                "unknown KV transfer status: {other}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct KvTransferStatus {
    pub status: c_int,
    pub transferred_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct KvTransferRequest {
    pub opcode: c_int,
    pub source: *mut c_void,
    pub target_id: i32,
    pub target_offset: u64,
    pub length: u64,
}

// The pointer is an opaque registered device/host address passed to the transport engine; moving
// the request between Rust threads does not grant dereference access in Rust.
unsafe impl Send for KvTransferRequest {}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct KvBufferEntry {
    pub addr: *mut c_void,
    pub length: usize,
}

pub type KvTransferEngineHandle = *mut c_void;
pub type KvTransportHandle = *mut c_void;
pub type KvSegmentId = i32;
pub type KvBatchId = u64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvTransferLayout {
    pub source_base_addr: usize,
    pub page_size_bytes: usize,
    pub target_base_offset: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvTransferTarget {
    pub target_id: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvRemoteLayout {
    pub dst_kv_ptrs: Vec<u64>,
    pub dst_kv_indices: Vec<i32>,
    pub dst_kv_item_len: usize,
}

pub trait KvMemoryLocationExt {
    fn kv_memory_location_label(self) -> Result<String, KvTransportError>;

    fn mooncake_memory_location_label(self) -> Result<String, KvTransportError>
    where
        Self: Sized,
    {
        self.kv_memory_location_label()
    }
}

impl KvMemoryLocationExt for KvCacheMemoryLocation {
    fn kv_memory_location_label(self) -> Result<String, KvTransportError> {
        let label = match self {
            Self::Cpu { numa_node } => format!("cpu:{numa_node}"),
            Self::Cuda { device_id } => format!("cuda:{device_id}"),
            Self::Rocm { device_id } => format!("hip:{device_id}"),
            Self::Musa { device_id } => format!("musa:{device_id}"),
            Self::Npu { device_id } => format!("npu:{device_id}"),
            location @ (Self::Metal { .. } | Self::Xpu { .. } | Self::Hpu { .. }) => {
                return Err(KvTransportError::UnsupportedMemoryLocation(location));
            }
        };
        Ok(label)
    }
}

pub trait KvCacheMemoryExt {
    fn kv_prefill_layout(&self, target_base_offset: u64) -> KvTransferLayout;

    fn kv_decode_remote_layout(&self, dst_kv_indices: &[i32]) -> KvRemoteLayout;

    fn mooncake_prefill_layout(&self, target_base_offset: u64) -> KvTransferLayout {
        self.kv_prefill_layout(target_base_offset)
    }

    fn mooncake_decode_remote_layout(&self, dst_kv_indices: &[i32]) -> KvRemoteLayout {
        self.kv_decode_remote_layout(dst_kv_indices)
    }
}

impl KvCacheMemoryExt for TransferableKvCacheMemory {
    fn kv_prefill_layout(&self, target_base_offset: u64) -> KvTransferLayout {
        KvTransferLayout {
            source_base_addr: self.regions()[0].base_addr,
            page_size_bytes: self.page_size_bytes(),
            target_base_offset,
        }
    }

    fn kv_decode_remote_layout(&self, dst_kv_indices: &[i32]) -> KvRemoteLayout {
        KvRemoteLayout {
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

pub trait KvMemoryRegistrar: Send + 'static {
    fn register_memory_batch(
        &mut self,
        buffers: &mut [KvBufferEntry],
        location: &str,
    ) -> Result<(), KvTransportError>;

    fn unregister_memory_batch(
        &mut self,
        addrs: &mut [*mut c_void],
    ) -> Result<(), KvTransportError>;
}

impl KvMemoryRegistrar for Box<dyn KvMemoryRegistrar> {
    fn register_memory_batch(
        &mut self,
        buffers: &mut [KvBufferEntry],
        location: &str,
    ) -> Result<(), KvTransportError> {
        self.as_mut().register_memory_batch(buffers, location)
    }

    fn unregister_memory_batch(
        &mut self,
        addrs: &mut [*mut c_void],
    ) -> Result<(), KvTransportError> {
        self.as_mut().unregister_memory_batch(addrs)
    }
}

pub trait KvMemoryRegistrationLease: Send {
    fn memory(&self) -> &TransferableKvCacheMemory;
    fn unregister(&mut self) -> Result<(), KvTransportError>;
}

pub struct RegisteredKvCacheMemory<R>
where
    R: KvMemoryRegistrar,
{
    registrar: R,
    memory: TransferableKvCacheMemory,
    registered_addrs: Vec<usize>,
    active: bool,
}

impl<R> RegisteredKvCacheMemory<R>
where
    R: KvMemoryRegistrar,
{
    pub fn register(
        mut registrar: R,
        memory: TransferableKvCacheMemory,
    ) -> Result<Self, KvTransportError> {
        let mut buffers = memory
            .regions()
            .iter()
            .map(|region| KvBufferEntry {
                addr: region.base_addr as *mut c_void,
                length: region.byte_len,
            })
            .collect::<Vec<_>>();
        let location = memory.location().kv_memory_location_label()?;
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

    pub fn unregister(&mut self) -> Result<(), KvTransportError> {
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

impl<R> KvMemoryRegistrationLease for RegisteredKvCacheMemory<R>
where
    R: KvMemoryRegistrar,
{
    fn memory(&self) -> &TransferableKvCacheMemory {
        self.memory()
    }

    fn unregister(&mut self) -> Result<(), KvTransportError> {
        self.unregister()
    }
}

impl<R> Drop for RegisteredKvCacheMemory<R>
where
    R: KvMemoryRegistrar,
{
    fn drop(&mut self) {
        if let Err(error) = self.unregister() {
            eprintln!("fatal: cannot release KV memory with an active registration: {error}");
            std::process::abort();
        }
    }
}

pub trait KvSegmentOpener {
    fn open_segment(&mut self, segment: &str) -> Result<i32, KvTransportError>;
}

pub trait KvTransferTargetResolver {
    fn resolve_target(
        &mut self,
        span: &KvCacheTransferSpan,
    ) -> Result<KvTransferTarget, KvCacheTransferError>;

    fn resolve_session_target(
        &mut self,
        span: &KvCacheTransferSpan,
        _session_id: &str,
    ) -> Result<KvTransferTarget, KvCacheTransferError> {
        self.resolve_target(span)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedKvTransferTargetResolver {
    target: KvTransferTarget,
}

impl FixedKvTransferTargetResolver {
    pub fn new(target: KvTransferTarget) -> Self {
        Self { target }
    }
}

impl KvTransferTargetResolver for FixedKvTransferTargetResolver {
    fn resolve_target(
        &mut self,
        _span: &KvCacheTransferSpan,
    ) -> Result<KvTransferTarget, KvCacheTransferError> {
        Ok(self.target)
    }
}

#[derive(Clone, Debug)]
pub struct SessionKvTargetResolver<O> {
    opener: O,
    sessions_by_room: BTreeMap<BootstrapRoom, String>,
    targets_by_session: BTreeMap<String, KvTransferTarget>,
}

impl<O> SessionKvTargetResolver<O> {
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

impl<O> KvTransferTargetResolver for SessionKvTargetResolver<O>
where
    O: KvSegmentOpener,
{
    fn resolve_target(
        &mut self,
        span: &KvCacheTransferSpan,
    ) -> Result<KvTransferTarget, KvCacheTransferError> {
        let session_id = self
            .sessions_by_room
            .get(&span.bootstrap_room())
            .ok_or_else(|| {
                KvCacheTransferError::Runtime(format!(
                    "missing session for bootstrap room {}",
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
        let target = KvTransferTarget { target_id };
        self.targets_by_session.insert(session_id, target);
        Ok(target)
    }

    fn resolve_session_target(
        &mut self,
        _span: &KvCacheTransferSpan,
        session_id: &str,
    ) -> Result<KvTransferTarget, KvCacheTransferError> {
        if let Some(target) = self.targets_by_session.get(session_id) {
            return Ok(*target);
        }

        let target_id = self
            .opener
            .open_segment(session_id)
            .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?;
        let target = KvTransferTarget { target_id };
        self.targets_by_session
            .insert(session_id.to_string(), target);
        Ok(target)
    }
}

pub fn build_kv_transfer_requests(
    span: &KvCacheTransferSpan,
    layout: KvTransferLayout,
    target: KvTransferTarget,
) -> Result<Vec<KvTransferRequest>, KvRequestBuildError> {
    if layout.page_size_bytes == 0 {
        return Err(KvRequestBuildError::ZeroPageSize);
    }

    let expected_page_count = span.token_count().div_ceil(span.page_size());
    if expected_page_count != span.page_indices().len() {
        return Err(KvRequestBuildError::SpanPageCountMismatch {
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
            .ok_or(KvRequestBuildError::AddressOverflow)?;
        let source_addr = layout
            .source_base_addr
            .checked_add(source_offset)
            .ok_or(KvRequestBuildError::AddressOverflow)?;
        let target_page_index = span
            .page_offset()
            .checked_add(page_index)
            .ok_or(KvRequestBuildError::OffsetOverflow)?;
        let target_offset = layout
            .target_base_offset
            .checked_add(
                (target_page_index as u64)
                    .checked_mul(page_size_bytes)
                    .ok_or(KvRequestBuildError::OffsetOverflow)?,
            )
            .ok_or(KvRequestBuildError::OffsetOverflow)?;

        requests.push(KvTransferRequest {
            opcode: KvOpcode::Write as c_int,
            source: source_addr as *mut c_void,
            target_id: target.target_id,
            target_offset,
            length: page_size_bytes,
        });
    }

    Ok(requests)
}

pub fn build_remote_kv_transfer_requests(
    span: &KvCacheTransferSpan,
    layout: KvTransferLayout,
    target: KvTransferTarget,
    remote_layout: &KvRemoteLayout,
) -> Result<Vec<KvTransferRequest>, KvRequestBuildError> {
    if layout.page_size_bytes == 0 {
        return Err(KvRequestBuildError::ZeroPageSize);
    }

    if remote_layout.dst_kv_item_len == 0 {
        return Err(KvRequestBuildError::ZeroRemoteKvItemSize);
    }

    if remote_layout.dst_kv_ptrs.is_empty() {
        return Err(KvRequestBuildError::MissingRemoteKvPointers);
    }

    let expected_page_count = span.token_count().div_ceil(span.page_size());
    if expected_page_count != span.page_indices().len() {
        return Err(KvRequestBuildError::SpanPageCountMismatch {
            expected_page_count,
            source_page_index_count: span.page_indices().len(),
        });
    }

    if span.page_indices().len() != remote_layout.dst_kv_indices.len() {
        return Err(KvRequestBuildError::RemoteKvIndexCountMismatch {
            source_page_count: span.page_indices().len(),
            dst_kv_index_count: remote_layout.dst_kv_indices.len(),
        });
    }

    let item_len = remote_layout.dst_kv_item_len;
    let split_page_size = item_len
        .checked_mul(remote_layout.dst_kv_ptrs.len())
        .ok_or(KvRequestBuildError::AddressOverflow)?;
    if split_page_size != layout.page_size_bytes {
        return Err(KvRequestBuildError::RemoteKvItemLayoutMismatch {
            page_size_bytes: layout.page_size_bytes,
            split_page_size,
        });
    }

    let mut requests =
        Vec::with_capacity(span.page_indices().len() * remote_layout.dst_kv_ptrs.len());

    for (page_index, source_page_index) in span.page_indices().iter().enumerate() {
        let dst_kv_index = remote_layout.dst_kv_indices[page_index];
        if dst_kv_index < 0 {
            return Err(KvRequestBuildError::NegativeRemoteKvIndex(dst_kv_index));
        }

        let source_offset = source_page_index
            .as_usize()
            .checked_mul(layout.page_size_bytes)
            .ok_or(KvRequestBuildError::AddressOverflow)?;
        let source_page_addr = layout
            .source_base_addr
            .checked_add(source_offset)
            .ok_or(KvRequestBuildError::AddressOverflow)?;

        for (ptr_index, dst_kv_ptr) in remote_layout.dst_kv_ptrs.iter().enumerate() {
            let source_addr = source_page_addr
                .checked_add(
                    ptr_index
                        .checked_mul(item_len)
                        .ok_or(KvRequestBuildError::AddressOverflow)?,
                )
                .ok_or(KvRequestBuildError::AddressOverflow)?;
            let target_offset = dst_kv_ptr
                .checked_add(
                    (dst_kv_index as u64)
                        .checked_mul(item_len as u64)
                        .ok_or(KvRequestBuildError::OffsetOverflow)?,
                )
                .ok_or(KvRequestBuildError::OffsetOverflow)?;

            requests.push(KvTransferRequest {
                opcode: KvOpcode::Write as c_int,
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
pub enum KvRequestBuildError {
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

impl fmt::Display for KvRequestBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroPageSize => formatter.write_str("KV page size must be non-zero"),
            Self::ZeroRemoteKvItemSize => {
                formatter.write_str("remote KV item size must be non-zero")
            }
            Self::MissingRemoteKvPointers => {
                formatter.write_str("remote KV pointers must not be empty")
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
                "remote KV item layout requires exactly {page_size_bytes} bytes but has {split_page_size} bytes"
            ),
            Self::NegativeRemoteKvIndex(index) => {
                write!(formatter, "remote KV index must be non-negative: {index}")
            }
            Self::AddressOverflow => formatter.write_str("source address overflow"),
            Self::OffsetOverflow => formatter.write_str("target offset overflow"),
        }
    }
}

impl std::error::Error for KvRequestBuildError {}

pub trait KvTransferSubmitter {
    fn submit_transfer(
        &mut self,
        requests: &mut [KvTransferRequest],
    ) -> Result<KvBatchId, KvTransportError>;
}

pub trait KvTransferStatusReader {
    fn transfer_status(
        &mut self,
        batch_id: KvBatchId,
        task_id: usize,
    ) -> Result<KvTransferStatus, KvTransportError>;
}

pub trait KvBatchReleaser {
    fn free_batch(&mut self, batch_id: KvBatchId) -> Result<(), KvTransportError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvSubmittedBatch {
    bootstrap_room: BootstrapRoom,
    batch_id: KvBatchId,
    task_count: usize,
    descriptor_checksum: String,
}

impl KvSubmittedBatch {
    pub fn new(bootstrap_room: BootstrapRoom, batch_id: KvBatchId, task_count: usize) -> Self {
        Self {
            bootstrap_room,
            batch_id,
            task_count,
            descriptor_checksum: String::new(),
        }
    }

    pub fn from_span(span: &KvCacheTransferSpan, batch_id: KvBatchId, task_count: usize) -> Self {
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

    pub fn batch_id(&self) -> KvBatchId {
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
pub struct KvTransferPollSummary {
    completed_batches: usize,
    failed_batches: usize,
    pending_batches: usize,
    completed_descriptor_checksums: Vec<String>,
    failed_descriptor_checksums: Vec<String>,
    pending_descriptor_checksums: Vec<String>,
}

impl KvTransferPollSummary {
    pub fn completed_batches(&self) -> usize {
        self.completed_batches
    }

    pub fn pending_batches(&self) -> usize {
        self.pending_batches
    }

    pub fn failed_batches(&self) -> usize {
        self.failed_batches
    }

    pub fn completed_descriptor_checksums(&self) -> &[String] {
        &self.completed_descriptor_checksums
    }

    pub fn pending_descriptor_checksums(&self) -> &[String] {
        &self.pending_descriptor_checksums
    }

    pub fn failed_descriptor_checksums(&self) -> &[String] {
        &self.failed_descriptor_checksums
    }
}

pub fn apply_transfer_completion_events(
    registry: &mut DecodeBootstrapRegistry,
    poll: KvTransferPoll,
) -> Result<KvTransferPollSummary, KvCacheTransferError> {
    let mut summary = KvTransferPollSummary {
        pending_batches: poll.pending_transfers,
        ..KvTransferPollSummary::default()
    };
    for event in poll.events {
        match event.status {
            KvTransferEventStatus::Completed => {
                registry.update_status(event.bootstrap_room, KvPoll::Success)?;
                summary.completed_batches += 1;
                if !event.descriptor_checksum.is_empty() {
                    summary
                        .completed_descriptor_checksums
                        .push(event.descriptor_checksum);
                }
            }
            KvTransferEventStatus::Failed(_) | KvTransferEventStatus::Canceled => {
                registry.update_status(event.bootstrap_room, KvPoll::Failed)?;
                summary.failed_batches += 1;
                if !event.descriptor_checksum.is_empty() {
                    summary
                        .failed_descriptor_checksums
                        .push(event.descriptor_checksum);
                }
            }
        }
    }
    Ok(summary)
}

pub fn poll_kv_transfer_batches<R>(
    reader: &mut R,
    submitted_batches: &[KvSubmittedBatch],
) -> Result<KvTransferPoll, KvCacheTransferError>
where
    R: KvTransferStatusReader,
{
    let mut poll = KvTransferPoll::default();
    for batch in submitted_batches {
        match poll_kv_submitted_batch(reader, batch)? {
            KvPolledBatchState::Completed => {
                poll.events.push(KvTransferCompletionEvent::completed(
                    batch.bootstrap_room(),
                    batch.descriptor_checksum().to_string(),
                ))
            }
            KvPolledBatchState::Pending => poll.pending_transfers += 1,
            KvPolledBatchState::Failed(message) => {
                poll.events.push(KvTransferCompletionEvent::failed(
                    batch.bootstrap_room(),
                    batch.descriptor_checksum().to_string(),
                    message,
                ))
            }
        }
    }
    Ok(poll)
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum KvPolledBatchState {
    Completed,
    Pending,
    Failed(String),
}

fn poll_kv_submitted_batch<R>(
    reader: &mut R,
    batch: &KvSubmittedBatch,
) -> Result<KvPolledBatchState, KvCacheTransferError>
where
    R: KvTransferStatusReader,
{
    let mut completed_tasks = 0;
    let mut failure = None;
    let mut pending = false;

    for task_id in 0..batch.task_count() {
        let status = reader
            .transfer_status(batch.batch_id(), task_id)
            .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?;
        match KvTransferStatusCode::try_from(status.status)? {
            KvTransferStatusCode::Completed => {
                completed_tasks += 1;
            }
            KvTransferStatusCode::Waiting | KvTransferStatusCode::Pending => pending = true,
            KvTransferStatusCode::Invalid
            | KvTransferStatusCode::Canceled
            | KvTransferStatusCode::Timeout
            | KvTransferStatusCode::Failed => {
                failure.get_or_insert_with(|| {
                    format!(
                        "KV transfer batch {} task {task_id} failed with status {}",
                        batch.batch_id(),
                        status.status
                    )
                });
            }
        }
    }

    if pending {
        Ok(KvPolledBatchState::Pending)
    } else if let Some(message) = failure {
        Ok(KvPolledBatchState::Failed(message))
    } else if completed_tasks == batch.task_count() {
        Ok(KvPolledBatchState::Completed)
    } else {
        Ok(KvPolledBatchState::Pending)
    }
}

pub struct KvCacheTransferExecutor<
    S: KvTransferStatusReader + KvBatchReleaser,
    R = FixedKvTransferTargetResolver,
> {
    local_memory_registration: Option<Box<dyn KvMemoryRegistrationLease>>,
    memory_registrar: Option<Box<dyn KvMemoryRegistrar>>,
    submitter: S,
    layout: KvTransferLayout,
    target_resolver: R,
    remote_kv_layouts_by_room: BTreeMap<BootstrapRoom, BTreeMap<String, KvRemoteLayout>>,
    submitted_batches: Vec<KvBatchId>,
    submitted_transfers: Vec<KvSubmittedBatch>,
    completion_events: Vec<KvTransferCompletionEvent>,
}

impl<S: KvTransferStatusReader + KvBatchReleaser> KvCacheTransferExecutor<S> {
    pub fn new(submitter: S, layout: KvTransferLayout, target: KvTransferTarget) -> Self {
        Self::with_target_resolver(
            submitter,
            layout,
            FixedKvTransferTargetResolver::new(target),
        )
    }

    pub fn with_remote_kv_layouts(
        submitter: S,
        layout: KvTransferLayout,
        target: KvTransferTarget,
        remote_kv_layouts: Vec<(BootstrapRoom, KvRemoteLayout)>,
    ) -> Self {
        Self::with_target_resolver_and_remote_kv_layouts(
            submitter,
            layout,
            FixedKvTransferTargetResolver::new(target),
            remote_kv_layouts,
        )
    }
}

impl<S: KvTransferStatusReader + KvBatchReleaser, R> KvCacheTransferExecutor<S, R> {
    pub fn with_target_resolver(
        submitter: S,
        layout: KvTransferLayout,
        target_resolver: R,
    ) -> Self {
        Self {
            local_memory_registration: None,
            memory_registrar: None,
            submitter,
            layout,
            target_resolver,
            remote_kv_layouts_by_room: BTreeMap::new(),
            submitted_batches: Vec::new(),
            submitted_transfers: Vec::new(),
            completion_events: Vec::new(),
        }
    }

    pub fn with_target_resolver_and_remote_kv_layouts(
        submitter: S,
        layout: KvTransferLayout,
        target_resolver: R,
        remote_kv_layouts: Vec<(BootstrapRoom, KvRemoteLayout)>,
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
            memory_registrar: None,
            submitter,
            layout,
            target_resolver,
            remote_kv_layouts_by_room,
            submitted_batches: Vec::new(),
            submitted_transfers: Vec::new(),
            completion_events: Vec::new(),
        }
    }

    pub fn with_target_resolver_and_remote_kv_session_layouts(
        submitter: S,
        layout: KvTransferLayout,
        target_resolver: R,
        remote_kv_layouts: Vec<(BootstrapRoom, String, KvRemoteLayout)>,
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
            memory_registrar: None,
            submitter,
            layout,
            target_resolver,
            remote_kv_layouts_by_room,
            submitted_batches: Vec::new(),
            submitted_transfers: Vec::new(),
            completion_events: Vec::new(),
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
        L: KvMemoryRegistrationLease + 'static,
    {
        if self.local_memory_registration.is_some() {
            return Err(KvCacheTransferError::Runtime(
                "KV executor already owns a local memory registration".to_string(),
            ));
        }
        let memory = registration.memory();
        if memory.page_size_bytes() != self.layout.page_size_bytes
            || memory.regions()[0].base_addr != self.layout.source_base_addr
        {
            return Err(KvCacheTransferError::Runtime(
                "KV memory registration does not match the executor KV layout".to_string(),
            ));
        }

        self.local_memory_registration = Some(Box::new(registration));
        Ok(self)
    }

    pub fn with_memory_registrar<M>(mut self, registrar: M) -> Self
    where
        M: KvMemoryRegistrar,
    {
        self.memory_registrar = Some(Box::new(registrar));
        self
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

    pub fn submitted_batches(&self) -> &[KvBatchId] {
        &self.submitted_batches
    }

    pub fn submitted_transfers(&self) -> &[KvSubmittedBatch] {
        &self.submitted_transfers
    }

    pub fn remote_kv_layouts(&self) -> &BTreeMap<BootstrapRoom, BTreeMap<String, KvRemoteLayout>> {
        &self.remote_kv_layouts_by_room
    }

    pub fn insert_remote_kv_layout(
        &mut self,
        bootstrap_room: BootstrapRoom,
        layout: KvRemoteLayout,
    ) {
        self.insert_remote_kv_session_layout(bootstrap_room, String::new(), layout);
    }

    pub fn insert_remote_kv_session_layout(
        &mut self,
        bootstrap_room: BootstrapRoom,
        session_id: impl Into<String>,
        layout: KvRemoteLayout,
    ) {
        self.remote_kv_layouts_by_room
            .entry(bootstrap_room)
            .or_default()
            .insert(session_id.into(), layout);
    }
}

impl<S, R> KvCacheTransferExecutor<S, R>
where
    S: KvTransferStatusReader + KvBatchReleaser,
{
    pub fn cancel_submitted_transfers_for_room(
        &mut self,
        bootstrap_room: BootstrapRoom,
    ) -> Result<(), KvCacheTransferError> {
        while let Some(index) = self
            .submitted_transfers
            .iter()
            .position(|transfer| transfer.bootstrap_room() == bootstrap_room)
        {
            let transfer = &self.submitted_transfers[index];
            while matches!(
                poll_kv_submitted_batch(&mut self.submitter, transfer)?,
                KvPolledBatchState::Pending
            ) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            self.submitter
                .free_batch(transfer.batch_id())
                .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?;
            self.submitted_transfers.remove(index);
            self.submitted_batches.remove(index);
        }
        self.completion_events
            .retain(|event| event.bootstrap_room != bootstrap_room);
        self.remote_kv_layouts_by_room.remove(&bootstrap_room);
        Ok(())
    }

    pub fn poll_submitted_transfers(&mut self) -> Result<KvTransferPoll, KvCacheTransferError> {
        let mut index = 0;
        while index < self.submitted_transfers.len() {
            let transfer = &self.submitted_transfers[index];
            match poll_kv_submitted_batch(&mut self.submitter, transfer)? {
                KvPolledBatchState::Completed => {
                    self.submitter
                        .free_batch(transfer.batch_id())
                        .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?;
                    self.completion_events
                        .push(KvTransferCompletionEvent::completed(
                            transfer.bootstrap_room(),
                            transfer.descriptor_checksum().to_string(),
                        ));
                }
                KvPolledBatchState::Pending => {
                    index += 1;
                    continue;
                }
                KvPolledBatchState::Failed(message) => {
                    self.submitter
                        .free_batch(transfer.batch_id())
                        .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?;
                    self.completion_events
                        .push(KvTransferCompletionEvent::failed(
                            transfer.bootstrap_room(),
                            transfer.descriptor_checksum().to_string(),
                            message,
                        ));
                }
            }
            self.submitted_transfers.remove(index);
            self.submitted_batches.remove(index);
        }
        Ok(KvTransferPoll::new(
            std::mem::take(&mut self.completion_events),
            self.submitted_transfers.len(),
        ))
    }
}

impl<S, R> KvTransferBackend for KvCacheTransferExecutor<S, R>
where
    S: KvTransferSubmitter + KvTransferStatusReader + KvBatchReleaser,
    R: KvTransferTargetResolver,
{
    fn register(&mut self, memory: TransferableKvCacheMemory) -> Result<(), KvCacheTransferError> {
        if self.local_memory_registration.is_some() {
            return Err(KvCacheTransferError::Runtime(
                "KV memory is already registered".to_string(),
            ));
        }
        if memory.regions().len() != 1
            || memory.regions()[0].base_addr != self.layout.source_base_addr
            || memory.page_size_bytes() != self.layout.page_size_bytes
        {
            return Err(KvCacheTransferError::Runtime(
                "NexusKV descriptor does not match the KV executor source layout".to_string(),
            ));
        }
        let registrar = self.memory_registrar.take().ok_or_else(|| {
            KvCacheTransferError::Runtime(
                "KV backend has no memory registrar; construct it through the production transfer backend factory"
                    .to_string(),
            )
        })?;
        let registration = RegisteredKvCacheMemory::register(registrar, memory)
            .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?;
        self.local_memory_registration = Some(Box::new(registration));
        Ok(())
    }

    fn submit(&mut self, span: &KvCacheTransferSpan) -> Result<(), KvCacheTransferError> {
        if self.local_memory_registration.is_none() {
            return Err(KvCacheTransferError::Runtime(
                "KV backend must register ModelRunner-owned NexusKV memory before submit"
                    .to_string(),
            ));
        }
        let mut requests = if let Some(remote_layouts) =
            self.remote_kv_layouts_by_room.get(&span.bootstrap_room())
        {
            let mut requests = Vec::new();
            for (session_id, remote_layout) in remote_layouts {
                let target = self
                    .target_resolver
                    .resolve_session_target(span, session_id)?;
                requests.extend(
                    build_remote_kv_transfer_requests(span, self.layout, target, remote_layout)
                        .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?,
                );
            }
            requests
        } else {
            let target = self.target_resolver.resolve_target(span)?;
            build_kv_transfer_requests(span, self.layout, target)
                .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?
        };
        if requests.is_empty() {
            self.completion_events
                .push(KvTransferCompletionEvent::completed(
                    span.bootstrap_room(),
                    span.descriptor_checksum(),
                ));
            return Ok(());
        }
        let task_count = requests.len();

        let batch_id = self
            .submitter
            .submit_transfer(&mut requests)
            .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?;
        self.submitted_batches.push(batch_id);
        self.submitted_transfers
            .push(KvSubmittedBatch::from_span(span, batch_id, task_count));
        Ok(())
    }

    fn poll(&mut self) -> Result<KvTransferPoll, KvCacheTransferError> {
        self.poll_submitted_transfers()
    }

    fn cancel(&mut self, bootstrap_room: BootstrapRoom) -> Result<(), KvCacheTransferError> {
        self.cancel_submitted_transfers_for_room(bootstrap_room)
    }

    fn shutdown(&mut self) -> Result<(), KvCacheTransferError> {
        let rooms = self
            .submitted_transfers
            .iter()
            .map(KvSubmittedBatch::bootstrap_room)
            .collect::<Vec<_>>();
        for room in rooms {
            self.cancel_submitted_transfers_for_room(room)?;
        }
        if let Some(registration) = self.local_memory_registration.as_mut() {
            registration
                .unregister()
                .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?;
        }
        self.local_memory_registration = None;
        Ok(())
    }
}

impl<S: KvTransferStatusReader + KvBatchReleaser, R> Drop for KvCacheTransferExecutor<S, R> {
    fn drop(&mut self) {
        while let Some(transfer) = self.submitted_transfers.first() {
            let room = transfer.bootstrap_room();
            if let Err(error) = self.cancel_submitted_transfers_for_room(room) {
                // Releasing registered memory without proving DMA quiescence is unsafe.
                eprintln!("fatal: cannot drain KV memory before destruction: {error}");
                std::process::abort();
            }
        }
    }
}

// Backward compatibility re-exports
pub use FixedKvTransferTargetResolver as FixedMooncakeTransferTargetResolver;
pub use KvBatchId as MooncakeBatchId;
pub use KvBatchReleaser as MooncakeBatchReleaser;
pub use KvBufferEntry as MooncakeBufferEntry;
pub use KvCacheMemoryExt as MooncakeKvCacheMemoryExt;
pub use KvCacheTransferExecutor as MooncakeKvCacheTransferExecutor;
pub use KvMemoryLocationExt as MooncakeMemoryLocationExt;
pub use KvMemoryRegistrar as MooncakeMemoryRegistrar;
pub use KvMemoryRegistrationLease as MooncakeMemoryRegistrationLease;
pub use KvOpcode as MooncakeOpcode;
pub use KvRemoteLayout as MooncakeRemoteKvLayout;
pub use KvRequestBuildError as MooncakeRequestBuildError;
pub use KvSegmentId as MooncakeSegmentId;
pub use KvSegmentOpener as MooncakeSegmentOpener;
pub use KvSubmittedBatch as MooncakeSubmittedBatch;
pub use KvTransferEngineHandle as MooncakeTransferEngineHandle;
pub use KvTransferLayout as MooncakeKvCacheLayout;
pub use KvTransferRequest as MooncakeTransferRequest;
pub use KvTransferStatus as MooncakeTransferStatus;
pub use KvTransferStatusCode as MooncakeTransferStatusCode;
pub use KvTransferStatusReader as MooncakeTransferStatusReader;
pub use KvTransferSubmitter as MooncakeTransferSubmitter;
pub use KvTransferTarget as MooncakeTransferTarget;
pub use KvTransferTargetResolver as MooncakeTransferTargetResolver;
pub use KvTransportHandle as MooncakeTransportHandle;
pub use RegisteredKvCacheMemory as RegisteredMooncakeKvCacheMemory;
pub use SessionKvTargetResolver as MooncakeSessionTargetResolver;
pub use build_kv_transfer_requests as build_mooncake_kv_transfer_requests;
pub use build_remote_kv_transfer_requests as build_mooncake_remote_kv_transfer_requests;
pub use poll_kv_transfer_batches as poll_mooncake_transfer_batches;
