use std::collections::BTreeMap;
#[cfg(feature = "mooncake-link")]
use std::ffi::CString;
use std::ffi::{NulError, c_char, c_int, c_void};
use std::fmt;

use crate::cli::ServerArgs;
use crate::types::{DisaggregatedParams, RequestId};

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

#[derive(Debug, Eq, PartialEq)]
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
