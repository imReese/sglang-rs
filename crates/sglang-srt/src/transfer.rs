use std::collections::BTreeMap;
#[cfg(feature = "mooncake-link")]
use std::ffi::CString;
use std::ffi::{NulError, c_char, c_int, c_void};
use std::fmt;

use crate::cache::CachePageId;
use crate::cli::ServerArgs;
use crate::model_executor::ModelWorkerBatch;
use crate::scheduler::{ForwardMode, ScheduleBatch, ScheduledRequest};
use crate::types::{DisaggregatedParams, RequestId};
use crate::worker::{
    BatchGeneratedTokens, DecodeRequestState, FallibleModelWorker, WorkerExecutionError,
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PdConfig {
    pub mode: DisaggregationMode,
    pub transfer_backend: TransferBackend,
    pub force_tcp_transport: bool,
    pub bootstrap_port: u16,
    pub ib_device: Option<String>,
    pub decode_enable_radix_cache: bool,
    pub decode_enable_offload_kvcache: bool,
    pub num_reserved_decode_tokens: usize,
    pub decode_polling_interval: usize,
}

impl PdConfig {
    pub fn from_server_args(args: &ServerArgs) -> Result<Self, PdConfigError> {
        let mode = DisaggregationMode::parse(&args.disaggregation_mode)?;
        let backend = TransferBackend::parse(&args.disaggregation_transfer_backend)?;

        if mode == DisaggregationMode::Prefill && backend.backend == TransferBackend::Fake {
            return Err(PdConfigError::FakePrefillUnsupported);
        }

        Ok(Self {
            mode,
            transfer_backend: backend.backend,
            force_tcp_transport: backend.force_tcp_transport,
            bootstrap_port: args.disaggregation_bootstrap_port,
            ib_device: if backend.force_tcp_transport {
                None
            } else {
                args.disaggregation_ib_device.clone()
            },
            decode_enable_radix_cache: args.disaggregation_decode_enable_radix_cache,
            decode_enable_offload_kvcache: args.disaggregation_decode_enable_offload_kvcache,
            num_reserved_decode_tokens: args.num_reserved_decode_tokens,
            decode_polling_interval: args.disaggregation_decode_polling_interval,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PdConfigError {
    InvalidDisaggregationMode(String),
    InvalidTransferBackend(String),
    FakePrefillUnsupported,
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
            Self::FakePrefillUnsupported => {
                formatter.write_str("prefill server does not support fake transfer backend")
            }
        }
    }
}

impl std::error::Error for PdConfigError {}

#[derive(Debug, Eq, PartialEq)]
pub enum MooncakeError {
    InteriorNul,
    EngineCreateFailed,
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
            Self::EngineCreateFailed => {
                formatter.write_str("mooncake transfer engine create failed")
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
    pub metadata_server: String,
    pub protocol: String,
    pub device_name: String,
}

impl MooncakeTransferEngineConfig {
    pub fn from_pd_config(hostname: impl Into<String>, gpu_id: usize, config: &PdConfig) -> Self {
        Self {
            hostname: hostname.into(),
            gpu_id,
            metadata_server: MOONCAKE_P2P_HANDSHAKE_METADATA.to_string(),
            protocol: if config.force_tcp_transport {
                "tcp".to_string()
            } else {
                "rdma".to_string()
            },
            device_name: config.ib_device.clone().unwrap_or_default(),
        }
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
    DuplicateBootstrapRoom(i32),
    MissingBootstrapRoom(i32),
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
    sessions_by_room: BTreeMap<i32, DecodeBootstrapSession>,
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

    pub fn get(&self, bootstrap_room: i32) -> Option<&DecodeBootstrapSession> {
        self.sessions_by_room.get(&bootstrap_room)
    }

    pub fn query_data_parallel_rank(&self, bootstrap_room: i32) -> Option<i32> {
        self.get(bootstrap_room)
            .map(DecodeBootstrapSession::data_parallel_rank)
    }

    pub fn update_status(
        &mut self,
        bootstrap_room: i32,
        status: KvPoll,
    ) -> Result<(), DecodeBootstrapRegistryError> {
        let session = self.sessions_by_room.get_mut(&bootstrap_room).ok_or(
            DecodeBootstrapRegistryError::MissingBootstrapRoom(bootstrap_room),
        )?;
        session.set_status(status);
        Ok(())
    }

    pub fn remove(&mut self, bootstrap_room: i32) -> Option<DecodeBootstrapSession> {
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
pub struct KvCacheTransferSpan {
    request_id: RequestId,
    disaggregated_params: DisaggregatedParams,
    data_parallel_rank: i32,
    token_offset: usize,
    token_count: usize,
    cache_pages: Vec<CachePageId>,
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

    pub fn bootstrap_room(&self) -> i32 {
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

    pub fn is_noop(&self) -> bool {
        self.token_count == 0
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KvCacheTransferPlan {
    spans: Vec<KvCacheTransferSpan>,
}

impl KvCacheTransferPlan {
    pub fn from_prefill_worker_batch(
        batch: &ModelWorkerBatch,
    ) -> Result<Self, KvCacheTransferPlanError> {
        if batch.forward_mode() != ForwardMode::Prefill {
            return Err(KvCacheTransferPlanError::NonPrefillBatch);
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

            spans.push(KvCacheTransferSpan {
                request_id: batch.request_ids()[request_index].clone(),
                disaggregated_params,
                data_parallel_rank: batch.data_parallel_ranks()[request_index],
                token_offset: batch.cached_token_counts()[request_index],
                token_count: input_token_count,
                cache_pages,
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvCacheTransferPlanError {
    NonPrefillBatch,
    CachePageCountMismatch {
        request_id: RequestId,
        input_token_count: usize,
        cache_page_count: usize,
    },
    TrailingCachePages {
        consumed_cache_page_count: usize,
        cache_page_count: usize,
    },
}

impl fmt::Display for KvCacheTransferPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPrefillBatch => formatter.write_str("KV transfer plan requires prefill batch"),
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
        }
    }
}

impl std::error::Error for KvCacheTransferPlanError {}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KvCacheTransferSummary {
    submitted_spans: usize,
    noop_spans: usize,
}

impl KvCacheTransferSummary {
    pub fn submitted_spans(&self) -> usize {
        self.submitted_spans
    }

    pub fn noop_spans(&self) -> usize {
        self.noop_spans
    }
}

pub trait KvCacheTransferExecutor {
    fn transfer_span(&mut self, span: &KvCacheTransferSpan) -> Result<(), KvCacheTransferError>;

    fn completes_inline(&self) -> bool {
        true
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

pub struct KvTransferModelWorker<W, E> {
    worker: W,
    registry: DecodeBootstrapRegistry,
    transfer_executor: E,
    last_transfer_summary: Option<KvCacheTransferSummary>,
}

impl<W, E> KvTransferModelWorker<W, E> {
    pub fn new(worker: W, registry: DecodeBootstrapRegistry, transfer_executor: E) -> Self {
        Self {
            worker,
            registry,
            transfer_executor,
            last_transfer_summary: None,
        }
    }

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

    pub fn last_transfer_summary(&self) -> Option<&KvCacheTransferSummary> {
        self.last_transfer_summary.as_ref()
    }
}

impl<W, E> FallibleModelWorker for KvTransferModelWorker<W, E>
where
    W: FallibleModelWorker,
    E: KvCacheTransferExecutor,
{
    fn try_generate_batch(
        &mut self,
        batch: &ScheduleBatch,
    ) -> Result<BatchGeneratedTokens, WorkerExecutionError> {
        let output = self.worker.try_generate_batch(batch)?;

        if batch.forward_mode() == ForwardMode::Prefill {
            let worker_batch = ModelWorkerBatch::from_schedule_batch(batch);
            let transfer_plan = KvCacheTransferPlan::from_prefill_worker_batch(&worker_batch)
                .map_err(|error| {
                    WorkerExecutionError::Runtime(format!("KV transfer planning failed: {error}"))
                })?;
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MooncakeTransferTarget {
    pub target_id: i32,
}

pub trait MooncakeTransferTargetResolver {
    fn resolve_target(
        &mut self,
        span: &KvCacheTransferSpan,
    ) -> Result<MooncakeTransferTarget, KvCacheTransferError>;
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

pub fn build_mooncake_kv_transfer_requests(
    span: &KvCacheTransferSpan,
    layout: MooncakeKvCacheLayout,
    target: MooncakeTransferTarget,
) -> Result<Vec<MooncakeTransferRequest>, MooncakeRequestBuildError> {
    if layout.page_size_bytes == 0 {
        return Err(MooncakeRequestBuildError::ZeroPageSize);
    }

    if span.token_count() != span.cache_pages().len() {
        return Err(MooncakeRequestBuildError::SpanPageCountMismatch {
            token_count: span.token_count(),
            cache_page_count: span.cache_pages().len(),
        });
    }

    let page_size_bytes = layout.page_size_bytes as u64;
    let mut requests = Vec::with_capacity(span.cache_pages().len());

    for (page_index, cache_page) in span.cache_pages().iter().enumerate() {
        let source_offset = cache_page
            .as_usize()
            .checked_mul(layout.page_size_bytes)
            .ok_or(MooncakeRequestBuildError::AddressOverflow)?;
        let source_addr = layout
            .source_base_addr
            .checked_add(source_offset)
            .ok_or(MooncakeRequestBuildError::AddressOverflow)?;
        let target_token_index = span
            .token_offset()
            .checked_add(page_index)
            .ok_or(MooncakeRequestBuildError::OffsetOverflow)?;
        let target_offset = layout
            .target_base_offset
            .checked_add(
                (target_token_index as u64)
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MooncakeRequestBuildError {
    ZeroPageSize,
    SpanPageCountMismatch {
        token_count: usize,
        cache_page_count: usize,
    },
    AddressOverflow,
    OffsetOverflow,
}

impl fmt::Display for MooncakeRequestBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroPageSize => formatter.write_str("Mooncake KV page size must be non-zero"),
            Self::SpanPageCountMismatch {
                token_count,
                cache_page_count,
            } => write!(
                formatter,
                "KV transfer span has {token_count} tokens but {cache_page_count} cache pages"
            ),
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MooncakeSubmittedBatch {
    bootstrap_room: i32,
    batch_id: MooncakeBatchId,
    task_count: usize,
}

impl MooncakeSubmittedBatch {
    pub fn new(bootstrap_room: i32, batch_id: MooncakeBatchId, task_count: usize) -> Self {
        Self {
            bootstrap_room,
            batch_id,
            task_count,
        }
    }

    pub fn bootstrap_room(&self) -> i32 {
        self.bootstrap_room
    }

    pub fn batch_id(&self) -> MooncakeBatchId {
        self.batch_id
    }

    pub fn task_count(&self) -> usize {
        self.task_count
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MooncakeTransferPollSummary {
    completed_batches: usize,
    pending_batches: usize,
}

impl MooncakeTransferPollSummary {
    pub fn completed_batches(&self) -> usize {
        self.completed_batches
    }

    pub fn pending_batches(&self) -> usize {
        self.pending_batches
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
        } else {
            summary.pending_batches += 1;
        }
    }

    Ok(summary)
}

pub trait KvTransferPoller {
    fn poll_transfers(&mut self) -> Result<MooncakeTransferPollSummary, KvCacheTransferError>;
}

pub struct MooncakeKvCacheTransferExecutor<S, R = FixedMooncakeTransferTargetResolver> {
    submitter: S,
    layout: MooncakeKvCacheLayout,
    target_resolver: R,
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
}

impl<S, R> MooncakeKvCacheTransferExecutor<S, R> {
    pub fn with_target_resolver(
        submitter: S,
        layout: MooncakeKvCacheLayout,
        target_resolver: R,
    ) -> Self {
        Self {
            submitter,
            layout,
            target_resolver,
            submitted_batches: Vec::new(),
            submitted_transfers: Vec::new(),
        }
    }

    pub fn submitter(&self) -> &S {
        &self.submitter
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
}

impl<S, R> MooncakeKvCacheTransferExecutor<S, R>
where
    S: MooncakeTransferStatusReader,
{
    pub fn poll_submitted_transfers(
        &mut self,
        registry: &mut DecodeBootstrapRegistry,
    ) -> Result<MooncakeTransferPollSummary, KvCacheTransferError> {
        let submitted_transfers = self.submitted_transfers.clone();
        poll_mooncake_transfer_batches(registry, &mut self.submitter, &submitted_transfers)
    }
}

impl<W, S, R> KvTransferPoller for KvTransferModelWorker<W, MooncakeKvCacheTransferExecutor<S, R>>
where
    S: MooncakeTransferStatusReader,
{
    fn poll_transfers(&mut self) -> Result<MooncakeTransferPollSummary, KvCacheTransferError> {
        self.transfer_executor
            .poll_submitted_transfers(&mut self.registry)
    }
}

impl<S, R> KvCacheTransferExecutor for MooncakeKvCacheTransferExecutor<S, R>
where
    S: MooncakeTransferSubmitter,
    R: MooncakeTransferTargetResolver,
{
    fn transfer_span(&mut self, span: &KvCacheTransferSpan) -> Result<(), KvCacheTransferError> {
        let target = self.target_resolver.resolve_target(span)?;
        let mut requests = build_mooncake_kv_transfer_requests(span, self.layout, target)
            .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?;
        if requests.is_empty() {
            return Ok(());
        }
        let task_count = requests.len();

        let batch_id = self
            .submitter
            .submit_transfer(&mut requests)
            .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?;
        self.submitted_batches.push(batch_id);
        self.submitted_transfers.push(MooncakeSubmittedBatch::new(
            span.bootstrap_room(),
            batch_id,
            task_count,
        ));
        Ok(())
    }

    fn completes_inline(&self) -> bool {
        false
    }
}

#[cfg(feature = "mooncake-link")]
pub struct LinkedMooncakeTransferEngine {
    handle: MooncakeTransferEngineHandle,
}

#[cfg(feature = "mooncake-link")]
impl LinkedMooncakeTransferEngine {
    pub fn new(config: &MooncakeTransferEngineConfig) -> Result<Self, MooncakeError> {
        let metadata = CString::new(config.metadata_server.as_str())?;
        let local_server = CString::new(config.hostname.as_str())?;
        let host = CString::new(config.hostname.as_str())?;

        let handle = unsafe {
            createTransferEngine(
                metadata.as_ptr(),
                local_server.as_ptr(),
                host.as_ptr(),
                0,
                1,
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
impl MooncakeTransferStatusReader for LinkedMooncakeTransferEngine {
    fn transfer_status(
        &mut self,
        batch_id: MooncakeBatchId,
        task_id: usize,
    ) -> Result<MooncakeTransferStatus, MooncakeError> {
        LinkedMooncakeTransferEngine::transfer_status(self, batch_id, task_id)
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
