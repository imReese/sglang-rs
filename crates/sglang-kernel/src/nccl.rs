use std::ffi::{CStr, c_char, c_int, c_void};
use std::fmt;

use libloading::Library;

use crate::cuda::{CudaContext, CudaDeviceAllocation, CudaError};

const NCCL_SUCCESS: c_int = 0;
const NCCL_BFLOAT16: c_int = 9;
const NCCL_SUM: c_int = 0;
const MINIMUM_BFLOAT16_VERSION: i32 = 21_003;
const BF16_BYTES: usize = std::mem::size_of::<u16>();

type NcclCommHandle = *mut c_void;
type CudaStreamHandle = *mut c_void;
type NcclGetVersion = unsafe extern "C" fn(*mut c_int) -> c_int;
type NcclGetUniqueId = unsafe extern "C" fn(*mut NcclUniqueId) -> c_int;
type NcclGetErrorString = unsafe extern "C" fn(c_int) -> *const c_char;
type NcclCommInitRank =
    unsafe extern "C" fn(*mut NcclCommHandle, c_int, NcclUniqueId, c_int) -> c_int;
type NcclCommDestroy = unsafe extern "C" fn(NcclCommHandle) -> c_int;
type NcclAllReduce = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    usize,
    c_int,
    c_int,
    NcclCommHandle,
    CudaStreamHandle,
) -> c_int;

#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct NcclUniqueId {
    bytes: [u8; Self::BYTE_LEN],
}

impl NcclUniqueId {
    pub const BYTE_LEN: usize = 128;

    pub const fn from_bytes(bytes: [u8; Self::BYTE_LEN]) -> Self {
        Self { bytes }
    }

    pub const fn as_bytes(&self) -> &[u8; Self::BYTE_LEN] {
        &self.bytes
    }
}

impl fmt::Debug for NcclUniqueId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("NcclUniqueId(<opaque>)")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NcclRank {
    world_size: c_int,
    rank: c_int,
}

impl NcclRank {
    pub fn new(world_size: usize, rank: usize) -> Result<Self, NcclError> {
        if world_size == 0 {
            return Err(NcclError::ZeroWorldSize);
        }
        if rank >= world_size {
            return Err(NcclError::InvalidRank { rank, world_size });
        }
        let world_size = c_int::try_from(world_size)
            .map_err(|_| NcclError::WorldSizeExceedsCInt { world_size })?;
        let rank = c_int::try_from(rank).map_err(|_| NcclError::RankExceedsCInt { rank })?;
        Ok(Self { world_size, rank })
    }

    pub fn world_size(self) -> usize {
        self.world_size as usize
    }

    pub fn rank(self) -> usize {
        self.rank as usize
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum NcclError {
    LibraryUnavailable {
        attempts: Vec<String>,
    },
    MissingLibrarySymbol {
        symbol: &'static str,
        detail: String,
    },
    Call {
        operation: &'static str,
        status: i32,
        description: Option<String>,
    },
    UnsupportedVersion {
        actual: i32,
        minimum: i32,
    },
    ZeroWorldSize,
    InvalidRank {
        rank: usize,
        world_size: usize,
    },
    WorldSizeExceedsCInt {
        world_size: usize,
    },
    RankExceedsCInt {
        rank: usize,
    },
    NullCommunicator,
    ZeroElementCount,
    CollectiveSizeOverflow,
    AllocationDeviceMismatch {
        communicator_ordinal: usize,
        allocation_ordinal: usize,
    },
    Cuda(CudaError),
}

impl fmt::Display for NcclError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LibraryUnavailable { attempts } => write!(
                formatter,
                "NCCL library is unavailable; tried {}",
                attempts.join(", ")
            ),
            Self::MissingLibrarySymbol { symbol, detail } => {
                write!(
                    formatter,
                    "NCCL library is missing symbol {symbol}: {detail}"
                )
            }
            Self::Call {
                operation,
                status,
                description,
            } => {
                write!(
                    formatter,
                    "NCCL call {operation} failed with status {status}"
                )?;
                if let Some(description) = description {
                    write!(formatter, ": {description}")?;
                }
                Ok(())
            }
            Self::UnsupportedVersion { actual, minimum } => write!(
                formatter,
                "NCCL version code {actual} does not support the required BF16 collectives; minimum is {minimum}"
            ),
            Self::ZeroWorldSize => formatter.write_str("NCCL world size must be positive"),
            Self::InvalidRank { rank, world_size } => write!(
                formatter,
                "NCCL rank {rank} must be smaller than world size {world_size}"
            ),
            Self::WorldSizeExceedsCInt { world_size } => write!(
                formatter,
                "NCCL world size {world_size} exceeds the c_int API limit"
            ),
            Self::RankExceedsCInt { rank } => {
                write!(formatter, "NCCL rank {rank} exceeds the c_int API limit")
            }
            Self::NullCommunicator => {
                formatter.write_str("ncclCommInitRank returned a null communicator")
            }
            Self::ZeroElementCount => {
                formatter.write_str("NCCL collective element count must be positive")
            }
            Self::CollectiveSizeOverflow => {
                formatter.write_str("NCCL collective byte size overflowed")
            }
            Self::AllocationDeviceMismatch {
                communicator_ordinal,
                allocation_ordinal,
            } => write!(
                formatter,
                "NCCL communicator belongs to CUDA device {communicator_ordinal}, but the allocation belongs to device {allocation_ordinal}"
            ),
            Self::Cuda(error) => write!(formatter, "CUDA operation for NCCL failed: {error}"),
        }
    }
}

impl std::error::Error for NcclError {}

impl From<CudaError> for NcclError {
    fn from(value: CudaError) -> Self {
        Self::Cuda(value)
    }
}

pub struct NcclLibrary {
    api: NcclApi,
    version: i32,
    library: Library,
}

impl fmt::Debug for NcclLibrary {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NcclLibrary")
            .field("version", &self.version)
            .finish_non_exhaustive()
    }
}

impl NcclLibrary {
    pub fn load() -> Result<Self, NcclError> {
        Self::load_from_candidates(nccl_library_candidates())
    }

    pub fn load_from_candidates<I, S>(candidates: I) -> Result<Self, NcclError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut attempts = Vec::new();
        for candidate in candidates {
            let candidate = candidate.as_ref();
            let library = match unsafe { Library::new(candidate) } {
                Ok(library) => library,
                Err(error) => {
                    attempts.push(format!("{candidate} ({error})"));
                    continue;
                }
            };
            let api = unsafe { NcclApi::load(&library) }?;
            let mut version = 0;
            check_status(
                api,
                unsafe { (api.get_version)(&mut version) },
                "ncclGetVersion",
            )?;
            if version < MINIMUM_BFLOAT16_VERSION {
                return Err(NcclError::UnsupportedVersion {
                    actual: version,
                    minimum: MINIMUM_BFLOAT16_VERSION,
                });
            }
            return Ok(Self {
                api,
                version,
                library,
            });
        }
        Err(NcclError::LibraryUnavailable { attempts })
    }

    pub fn version(&self) -> i32 {
        self.version
    }

    pub fn unique_id(&self) -> Result<NcclUniqueId, NcclError> {
        let mut unique_id = NcclUniqueId::from_bytes([0; NcclUniqueId::BYTE_LEN]);
        check_status(
            self.api,
            unsafe { (self.api.get_unique_id)(&mut unique_id) },
            "ncclGetUniqueId",
        )?;
        Ok(unique_id)
    }
}

pub struct NcclCommunicator {
    context: CudaContext,
    rank: NcclRank,
    handle: usize,
    api: NcclApi,
    version: i32,
    _library: Library,
}

impl fmt::Debug for NcclCommunicator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NcclCommunicator")
            .field("device_ordinal", &self.context.device_ordinal())
            .field("world_size", &self.rank.world_size())
            .field("rank", &self.rank.rank())
            .field("version", &self.version)
            .finish_non_exhaustive()
    }
}

impl NcclCommunicator {
    pub fn initialize(
        library: NcclLibrary,
        context: &CudaContext,
        unique_id: NcclUniqueId,
        rank: NcclRank,
    ) -> Result<Self, NcclError> {
        let mut handle = std::ptr::null_mut();
        context.with_current(|| {
            check_status(
                library.api,
                unsafe {
                    (library.api.comm_init_rank)(&mut handle, rank.world_size, unique_id, rank.rank)
                },
                "ncclCommInitRank",
            )
        })?;
        if handle.is_null() {
            return Err(NcclError::NullCommunicator);
        }
        Ok(Self {
            context: context.clone(),
            rank,
            handle: handle as usize,
            api: library.api,
            version: library.version,
            _library: library.library,
        })
    }

    pub fn device_ordinal(&self) -> usize {
        self.context.device_ordinal()
    }

    pub fn rank(&self) -> NcclRank {
        self.rank
    }

    pub fn all_reduce_bf16_sum_in_place(
        &self,
        allocation: &mut CudaDeviceAllocation,
        element_count: usize,
    ) -> Result<(), NcclError> {
        let byte_len = bf16_collective_byte_len(element_count)?;
        if allocation.device_ordinal() != self.device_ordinal() {
            return Err(NcclError::AllocationDeviceMismatch {
                communicator_ordinal: self.device_ordinal(),
                allocation_ordinal: allocation.device_ordinal(),
            });
        }
        let device_ptr = allocation.device_ptr_at(0, byte_len)?;
        self.context.with_current(|| {
            check_status(
                self.api,
                unsafe {
                    (self.api.all_reduce)(
                        device_ptr as *const c_void,
                        device_ptr as *mut c_void,
                        element_count,
                        NCCL_BFLOAT16,
                        NCCL_SUM,
                        self.handle as NcclCommHandle,
                        std::ptr::null_mut(),
                    )
                },
                "ncclAllReduce",
            )
        })?;
        self.context.synchronize()?;
        Ok(())
    }
}

impl Drop for NcclCommunicator {
    fn drop(&mut self) {
        let result = self.context.with_current(|| {
            check_status(
                self.api,
                unsafe { (self.api.comm_destroy)(self.handle as NcclCommHandle) },
                "ncclCommDestroy",
            )
        });
        if let Err(error) = result {
            eprintln!("failed to destroy NCCL communicator: {error}");
        }
    }
}

#[derive(Clone, Copy)]
struct NcclApi {
    get_version: NcclGetVersion,
    get_unique_id: NcclGetUniqueId,
    get_error_string: NcclGetErrorString,
    comm_init_rank: NcclCommInitRank,
    comm_destroy: NcclCommDestroy,
    all_reduce: NcclAllReduce,
}

impl NcclApi {
    unsafe fn load(library: &Library) -> Result<Self, NcclError> {
        Ok(Self {
            get_version: unsafe { load_symbol(library, b"ncclGetVersion\0", "ncclGetVersion")? },
            get_unique_id: unsafe {
                load_symbol(library, b"ncclGetUniqueId\0", "ncclGetUniqueId")?
            },
            get_error_string: unsafe {
                load_symbol(library, b"ncclGetErrorString\0", "ncclGetErrorString")?
            },
            comm_init_rank: unsafe {
                load_symbol(library, b"ncclCommInitRank\0", "ncclCommInitRank")?
            },
            comm_destroy: unsafe { load_symbol(library, b"ncclCommDestroy\0", "ncclCommDestroy")? },
            all_reduce: unsafe { load_symbol(library, b"ncclAllReduce\0", "ncclAllReduce")? },
        })
    }
}

unsafe fn load_symbol<T: Copy>(
    library: &Library,
    symbol: &'static [u8],
    symbol_name: &'static str,
) -> Result<T, NcclError> {
    unsafe { library.get::<T>(symbol) }
        .map(|loaded| *loaded)
        .map_err(|error| NcclError::MissingLibrarySymbol {
            symbol: symbol_name,
            detail: error.to_string(),
        })
}

fn check_status(api: NcclApi, status: c_int, operation: &'static str) -> Result<(), NcclError> {
    if status == NCCL_SUCCESS {
        return Ok(());
    }
    let description = unsafe { (api.get_error_string)(status) };
    let description = (!description.is_null()).then(|| {
        unsafe { CStr::from_ptr(description) }
            .to_string_lossy()
            .into_owned()
    });
    Err(NcclError::Call {
        operation,
        status,
        description,
    })
}

fn bf16_collective_byte_len(element_count: usize) -> Result<usize, NcclError> {
    if element_count == 0 {
        return Err(NcclError::ZeroElementCount);
    }
    element_count
        .checked_mul(BF16_BYTES)
        .ok_or(NcclError::CollectiveSizeOverflow)
}

fn nccl_library_candidates() -> &'static [&'static str] {
    #[cfg(target_os = "windows")]
    {
        &["nccl.dll"]
    }
    #[cfg(target_os = "linux")]
    {
        &["libnccl.so.2", "libnccl.so"]
    }
    #[cfg(target_os = "macos")]
    {
        &["libnccl.dylib"]
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        &["libnccl.so.2", "libnccl.so"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unique_id_matches_the_nccl_c_abi_and_round_trips_bytes() {
        let bytes = std::array::from_fn(|index| index as u8);
        let unique_id = NcclUniqueId::from_bytes(bytes);

        assert_eq!(std::mem::size_of::<NcclUniqueId>(), 128);
        assert_eq!(unique_id.as_bytes(), &bytes);
        assert_eq!(format!("{unique_id:?}"), "NcclUniqueId(<opaque>)");
    }

    #[test]
    fn rank_geometry_fails_before_loading_cuda_or_nccl() {
        assert_eq!(NcclRank::new(0, 0), Err(NcclError::ZeroWorldSize));
        assert_eq!(
            NcclRank::new(2, 2),
            Err(NcclError::InvalidRank {
                rank: 2,
                world_size: 2,
            })
        );
        assert_eq!(
            NcclRank::new(8, 3).expect("rank geometry should be valid"),
            NcclRank {
                world_size: 8,
                rank: 3,
            }
        );
        assert_eq!(
            NcclRank::new(usize::MAX, 0),
            Err(NcclError::WorldSizeExceedsCInt {
                world_size: usize::MAX,
            })
        );
    }

    #[test]
    fn bf16_collective_geometry_rejects_empty_and_overflowing_buffers() {
        assert_eq!(
            bf16_collective_byte_len(0),
            Err(NcclError::ZeroElementCount)
        );
        assert_eq!(
            bf16_collective_byte_len(usize::MAX),
            Err(NcclError::CollectiveSizeOverflow)
        );
        assert_eq!(bf16_collective_byte_len(4096), Ok(8192));
    }

    #[test]
    fn missing_nccl_library_reports_every_candidate() {
        let error = NcclLibrary::load_from_candidates([
            "/definitely/missing/libnccl-a.so",
            "/definitely/missing/libnccl-b.so",
        ])
        .expect_err("missing NCCL must fail fast");
        let NcclError::LibraryUnavailable { attempts } = error else {
            panic!("unexpected NCCL load error")
        };

        assert_eq!(attempts.len(), 2);
        assert!(attempts[0].contains("libnccl-a.so"));
        assert!(attempts[1].contains("libnccl-b.so"));
    }
}
