use std::ffi::{c_char, c_int, c_void};
use std::fmt;

use crate::cli::ServerArgs;

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
