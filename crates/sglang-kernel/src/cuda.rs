use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::fmt;
use std::sync::Arc;

use libloading::Library;

const CUDA_SUCCESS: c_int = 0;
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
const CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING: c_int = 41;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

type CuDevice = c_int;
type CuContextHandle = *mut c_void;
type CuDevicePtr = u64;

type CuInit = unsafe extern "C" fn(c_uint) -> c_int;
type CuDriverGetVersion = unsafe extern "C" fn(*mut c_int) -> c_int;
type CuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> c_int;
type CuDeviceGet = unsafe extern "C" fn(*mut CuDevice, c_int) -> c_int;
type CuDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> c_int;
type CuDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> c_int;
type CuDeviceTotalMem = unsafe extern "C" fn(*mut usize, CuDevice) -> c_int;
type CuDevicePrimaryCtxRetain = unsafe extern "C" fn(*mut CuContextHandle, CuDevice) -> c_int;
type CuDevicePrimaryCtxRelease = unsafe extern "C" fn(CuDevice) -> c_int;
type CuCtxPushCurrent = unsafe extern "C" fn(CuContextHandle) -> c_int;
type CuCtxPopCurrent = unsafe extern "C" fn(*mut CuContextHandle) -> c_int;
type CuCtxSynchronize = unsafe extern "C" fn() -> c_int;
type CuMemAlloc = unsafe extern "C" fn(*mut CuDevicePtr, usize) -> c_int;
type CuMemFree = unsafe extern "C" fn(CuDevicePtr) -> c_int;
type CuMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> c_int;
type CuMemcpyHtoD = unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> c_int;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> c_int;
type CuMemsetD8 = unsafe extern "C" fn(CuDevicePtr, u8, usize) -> c_int;
type CuGetErrorName = unsafe extern "C" fn(c_int, *mut *const c_char) -> c_int;
type CuGetErrorString = unsafe extern "C" fn(c_int, *mut *const c_char) -> c_int;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct CudaComputeCapability {
    pub major: u32,
    pub minor: u32,
}

impl CudaComputeCapability {
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    pub const fn sm(self) -> u32 {
        self.major * 10 + self.minor
    }
}

impl fmt::Display for CudaComputeCapability {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}.{} (sm_{})",
            self.major,
            self.minor,
            self.sm()
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaDeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub total_memory_bytes: usize,
    pub multiprocessor_count: u32,
    pub unified_addressing: bool,
    pub compute_capability: CudaComputeCapability,
    pub driver_version: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CudaError {
    DriverUnavailable {
        attempts: Vec<String>,
    },
    MissingDriverSymbol {
        symbol: &'static str,
        detail: String,
    },
    DriverCall {
        operation: &'static str,
        code: i32,
        name: Option<String>,
        description: Option<String>,
    },
    InvalidDeviceOrdinal {
        ordinal: usize,
        device_count: usize,
    },
    InvalidDeviceAttribute {
        attribute: &'static str,
        value: i32,
    },
    ZeroAllocation,
    AllocationOutOfBounds {
        offset: usize,
        byte_len: usize,
        allocation_byte_len: usize,
    },
    DeviceAddressOverflow,
    FreedAllocation,
}

impl fmt::Display for CudaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DriverUnavailable { attempts } => write!(
                formatter,
                "CUDA driver library is unavailable; tried {}",
                attempts.join(", ")
            ),
            Self::MissingDriverSymbol { symbol, detail } => {
                write!(
                    formatter,
                    "CUDA driver is missing symbol {symbol}: {detail}"
                )
            }
            Self::DriverCall {
                operation,
                code,
                name,
                description,
            } => {
                write!(
                    formatter,
                    "CUDA driver call {operation} failed with code {code}"
                )?;
                if let Some(name) = name {
                    write!(formatter, " ({name})")?;
                }
                if let Some(description) = description {
                    write!(formatter, ": {description}")?;
                }
                Ok(())
            }
            Self::InvalidDeviceOrdinal {
                ordinal,
                device_count,
            } => write!(
                formatter,
                "CUDA device ordinal {ordinal} is out of range for {device_count} visible devices"
            ),
            Self::InvalidDeviceAttribute { attribute, value } => {
                write!(
                    formatter,
                    "CUDA device attribute {attribute} has invalid value {value}"
                )
            }
            Self::ZeroAllocation => formatter.write_str("CUDA allocation size must be non-zero"),
            Self::AllocationOutOfBounds {
                offset,
                byte_len,
                allocation_byte_len,
            } => write!(
                formatter,
                "CUDA allocation access [{offset}, {}) exceeds allocation size {allocation_byte_len}",
                offset.saturating_add(*byte_len)
            ),
            Self::DeviceAddressOverflow => {
                formatter.write_str("CUDA device pointer arithmetic overflowed")
            }
            Self::FreedAllocation => formatter.write_str("CUDA allocation has already been freed"),
        }
    }
}

impl std::error::Error for CudaError {}

impl CudaError {
    pub fn is_unavailable_for_auto_selection(&self) -> bool {
        matches!(self, Self::DriverUnavailable { .. })
            || matches!(
                self,
                Self::DriverCall {
                    operation: "cuInit",
                    code: 100,
                    ..
                }
            )
    }
}

#[derive(Clone)]
pub struct CudaDriver {
    inner: Arc<CudaDriverInner>,
}

impl CudaDriver {
    pub fn load() -> Result<Self, CudaError> {
        Self::load_from_candidates(driver_library_candidates())
    }

    pub fn load_from_candidates<I, S>(candidates: I) -> Result<Self, CudaError>
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

            let api = unsafe { CudaApi::load(&library) }?;
            let driver = Self {
                inner: Arc::new(CudaDriverInner {
                    _library: Some(library),
                    api,
                }),
            };
            driver.check(unsafe { (driver.inner.api.init)(0) }, "cuInit")?;
            return Ok(driver);
        }

        Err(CudaError::DriverUnavailable { attempts })
    }

    pub fn driver_version(&self) -> Result<i32, CudaError> {
        let mut version = 0;
        self.check(
            unsafe { (self.inner.api.driver_get_version)(&mut version) },
            "cuDriverGetVersion",
        )?;
        Ok(version)
    }

    pub fn device_count(&self) -> Result<usize, CudaError> {
        let mut count = 0;
        self.check(
            unsafe { (self.inner.api.device_get_count)(&mut count) },
            "cuDeviceGetCount",
        )?;
        usize::try_from(count).map_err(|_| CudaError::InvalidDeviceAttribute {
            attribute: "device count",
            value: count,
        })
    }

    pub fn devices(&self) -> Result<Vec<CudaDeviceInfo>, CudaError> {
        (0..self.device_count()?)
            .map(|ordinal| self.device_info(ordinal))
            .collect()
    }

    pub fn device_info(&self, ordinal: usize) -> Result<CudaDeviceInfo, CudaError> {
        let device = self.device_handle(ordinal)?;
        let mut name = [0 as c_char; 256];
        self.check(
            unsafe {
                (self.inner.api.device_get_name)(name.as_mut_ptr(), name.len() as c_int, device)
            },
            "cuDeviceGetName",
        )?;
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let mut total_memory_bytes = 0;
        self.check(
            unsafe { (self.inner.api.device_total_mem)(&mut total_memory_bytes, device) },
            "cuDeviceTotalMem_v2",
        )?;

        let major = self.device_attribute(
            device,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            "compute capability major",
        )?;
        let minor = self.device_attribute(
            device,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            "compute capability minor",
        )?;
        let multiprocessor_count = self.device_attribute(
            device,
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            "multiprocessor count",
        )?;
        let unified_addressing = self.device_attribute(
            device,
            CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
            "unified addressing",
        )? != 0;

        Ok(CudaDeviceInfo {
            ordinal,
            name,
            total_memory_bytes,
            multiprocessor_count,
            unified_addressing,
            compute_capability: CudaComputeCapability::new(major, minor),
            driver_version: self.driver_version()?,
        })
    }

    pub fn retain_primary_context(&self, ordinal: usize) -> Result<CudaContext, CudaError> {
        let device = self.device_handle(ordinal)?;
        let mut context = std::ptr::null_mut();
        self.check(
            unsafe { (self.inner.api.device_primary_ctx_retain)(&mut context, device) },
            "cuDevicePrimaryCtxRetain",
        )?;
        if context.is_null() {
            return Err(CudaError::DriverCall {
                operation: "cuDevicePrimaryCtxRetain",
                code: CUDA_SUCCESS,
                name: None,
                description: Some("driver returned a null primary context".to_string()),
            });
        }

        Ok(CudaContext {
            inner: Arc::new(CudaContextInner {
                driver: self.clone(),
                device,
                ordinal,
                handle: context as usize,
            }),
        })
    }

    fn device_handle(&self, ordinal: usize) -> Result<CuDevice, CudaError> {
        let device_count = self.device_count()?;
        if ordinal >= device_count {
            return Err(CudaError::InvalidDeviceOrdinal {
                ordinal,
                device_count,
            });
        }
        let ordinal_i32 =
            c_int::try_from(ordinal).map_err(|_| CudaError::InvalidDeviceOrdinal {
                ordinal,
                device_count,
            })?;
        let mut device = 0;
        self.check(
            unsafe { (self.inner.api.device_get)(&mut device, ordinal_i32) },
            "cuDeviceGet",
        )?;
        Ok(device)
    }

    fn device_attribute(
        &self,
        device: CuDevice,
        attribute: c_int,
        attribute_name: &'static str,
    ) -> Result<u32, CudaError> {
        let mut value = 0;
        self.check(
            unsafe { (self.inner.api.device_get_attribute)(&mut value, attribute, device) },
            "cuDeviceGetAttribute",
        )?;
        u32::try_from(value).map_err(|_| CudaError::InvalidDeviceAttribute {
            attribute: attribute_name,
            value,
        })
    }

    fn check(&self, code: c_int, operation: &'static str) -> Result<(), CudaError> {
        if code == CUDA_SUCCESS {
            return Ok(());
        }
        Err(self.call_error(operation, code))
    }

    fn call_error(&self, operation: &'static str, code: c_int) -> CudaError {
        CudaError::DriverCall {
            operation,
            code,
            name: self.error_text(code, self.inner.api.get_error_name),
            description: self.error_text(code, self.inner.api.get_error_string),
        }
    }

    fn error_text(&self, code: c_int, get_text: CuGetErrorName) -> Option<String> {
        let mut text = std::ptr::null();
        if unsafe { get_text(code, &mut text) } != CUDA_SUCCESS || text.is_null() {
            return None;
        }
        Some(
            unsafe { CStr::from_ptr(text) }
                .to_string_lossy()
                .into_owned(),
        )
    }
}

pub struct CudaContext {
    inner: Arc<CudaContextInner>,
}

impl Clone for CudaContext {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl CudaContext {
    pub fn device_ordinal(&self) -> usize {
        self.inner.ordinal
    }

    pub fn allocate(&self, byte_len: usize) -> Result<CudaDeviceAllocation, CudaError> {
        if byte_len == 0 {
            return Err(CudaError::ZeroAllocation);
        }
        let mut device_ptr = 0;
        self.with_current(|| {
            self.inner.driver.check(
                unsafe { (self.inner.driver.inner.api.mem_alloc)(&mut device_ptr, byte_len) },
                "cuMemAlloc_v2",
            )
        })?;
        Ok(CudaDeviceAllocation {
            context: self.clone(),
            device_ptr,
            byte_len,
            active: true,
        })
    }

    pub fn memory_info(&self) -> Result<CudaMemoryInfo, CudaError> {
        let mut free_bytes = 0;
        let mut total_bytes = 0;
        self.with_current(|| {
            self.inner.driver.check(
                unsafe {
                    (self.inner.driver.inner.api.mem_get_info)(&mut free_bytes, &mut total_bytes)
                },
                "cuMemGetInfo_v2",
            )
        })?;
        Ok(CudaMemoryInfo {
            free_bytes,
            total_bytes,
        })
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.with_current(|| {
            self.inner.driver.check(
                unsafe { (self.inner.driver.inner.api.ctx_synchronize)() },
                "cuCtxSynchronize",
            )
        })
    }

    pub(crate) fn with_current<T, E>(
        &self,
        operation: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E>
    where
        E: From<CudaError>,
    {
        let driver = &self.inner.driver;
        driver
            .check(
                unsafe {
                    (driver.inner.api.ctx_push_current)(self.inner.handle as CuContextHandle)
                },
                "cuCtxPushCurrent_v2",
            )
            .map_err(E::from)?;
        let result = operation();
        let mut popped = std::ptr::null_mut();
        let pop_result = driver.check(
            unsafe { (driver.inner.api.ctx_pop_current)(&mut popped) },
            "cuCtxPopCurrent_v2",
        );
        match (result, pop_result) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(E::from(error)),
            (Ok(value), Ok(())) => Ok(value),
        }
    }
}

struct CudaContextInner {
    driver: CudaDriver,
    device: CuDevice,
    ordinal: usize,
    handle: usize,
}

impl Drop for CudaContextInner {
    fn drop(&mut self) {
        let code = unsafe { (self.driver.inner.api.device_primary_ctx_release)(self.device) };
        if let Err(error) = self.driver.check(code, "cuDevicePrimaryCtxRelease_v2") {
            eprintln!("failed to release CUDA primary context: {error}");
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaMemoryInfo {
    pub free_bytes: usize,
    pub total_bytes: usize,
}

pub struct CudaDeviceAllocation {
    context: CudaContext,
    device_ptr: CuDevicePtr,
    byte_len: usize,
    active: bool,
}

impl CudaDeviceAllocation {
    pub fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub fn device_ordinal(&self) -> usize {
        self.context.device_ordinal()
    }

    pub fn copy_from_host(&mut self, offset: usize, bytes: &[u8]) -> Result<(), CudaError> {
        if bytes.is_empty() {
            self.device_ptr_at(offset, 0)?;
            return Ok(());
        }
        let device_ptr = self.device_ptr_at(offset, bytes.len())?;
        self.context.with_current(|| {
            self.context.inner.driver.check(
                unsafe {
                    (self.context.inner.driver.inner.api.memcpy_htod)(
                        device_ptr,
                        bytes.as_ptr().cast(),
                        bytes.len(),
                    )
                },
                "cuMemcpyHtoD_v2",
            )
        })
    }

    pub fn copy_to_host(&self, offset: usize, bytes: &mut [u8]) -> Result<(), CudaError> {
        if bytes.is_empty() {
            self.device_ptr_at(offset, 0)?;
            return Ok(());
        }
        let device_ptr = self.device_ptr_at(offset, bytes.len())?;
        self.context.with_current(|| {
            self.context.inner.driver.check(
                unsafe {
                    (self.context.inner.driver.inner.api.memcpy_dtoh)(
                        bytes.as_mut_ptr().cast(),
                        device_ptr,
                        bytes.len(),
                    )
                },
                "cuMemcpyDtoH_v2",
            )
        })
    }

    pub fn fill(&mut self, value: u8) -> Result<(), CudaError> {
        let device_ptr = self.device_ptr_at(0, self.byte_len)?;
        self.context.with_current(|| {
            self.context.inner.driver.check(
                unsafe {
                    (self.context.inner.driver.inner.api.memset_d8)(
                        device_ptr,
                        value,
                        self.byte_len,
                    )
                },
                "cuMemsetD8_v2",
            )
        })
    }

    pub fn device_ptr_at(&self, offset: usize, byte_len: usize) -> Result<CuDevicePtr, CudaError> {
        if !self.active {
            return Err(CudaError::FreedAllocation);
        }
        let end = offset
            .checked_add(byte_len)
            .ok_or(CudaError::AllocationOutOfBounds {
                offset,
                byte_len,
                allocation_byte_len: self.byte_len,
            })?;
        if end > self.byte_len {
            return Err(CudaError::AllocationOutOfBounds {
                offset,
                byte_len,
                allocation_byte_len: self.byte_len,
            });
        }
        self.device_ptr
            .checked_add(u64::try_from(offset).map_err(|_| CudaError::DeviceAddressOverflow)?)
            .ok_or(CudaError::DeviceAddressOverflow)
    }

    pub fn free(&mut self) -> Result<(), CudaError> {
        if !self.active {
            return Ok(());
        }
        self.context.with_current(|| {
            self.context.inner.driver.check(
                unsafe { (self.context.inner.driver.inner.api.mem_free)(self.device_ptr) },
                "cuMemFree_v2",
            )
        })?;
        self.active = false;
        Ok(())
    }
}

impl Drop for CudaDeviceAllocation {
    fn drop(&mut self) {
        if let Err(error) = self.free() {
            eprintln!("failed to free CUDA device allocation: {error}");
        }
    }
}

struct CudaDriverInner {
    _library: Option<Library>,
    api: CudaApi,
}

#[derive(Clone, Copy)]
struct CudaApi {
    init: CuInit,
    driver_get_version: CuDriverGetVersion,
    device_get_count: CuDeviceGetCount,
    device_get: CuDeviceGet,
    device_get_name: CuDeviceGetName,
    device_get_attribute: CuDeviceGetAttribute,
    device_total_mem: CuDeviceTotalMem,
    device_primary_ctx_retain: CuDevicePrimaryCtxRetain,
    device_primary_ctx_release: CuDevicePrimaryCtxRelease,
    ctx_push_current: CuCtxPushCurrent,
    ctx_pop_current: CuCtxPopCurrent,
    ctx_synchronize: CuCtxSynchronize,
    mem_alloc: CuMemAlloc,
    mem_free: CuMemFree,
    mem_get_info: CuMemGetInfo,
    memcpy_htod: CuMemcpyHtoD,
    memcpy_dtoh: CuMemcpyDtoH,
    memset_d8: CuMemsetD8,
    get_error_name: CuGetErrorName,
    get_error_string: CuGetErrorString,
}

impl CudaApi {
    unsafe fn load(library: &Library) -> Result<Self, CudaError> {
        Ok(Self {
            init: unsafe { load_symbol(library, b"cuInit\0", "cuInit")? },
            driver_get_version: unsafe {
                load_symbol(library, b"cuDriverGetVersion\0", "cuDriverGetVersion")?
            },
            device_get_count: unsafe {
                load_symbol(library, b"cuDeviceGetCount\0", "cuDeviceGetCount")?
            },
            device_get: unsafe { load_symbol(library, b"cuDeviceGet\0", "cuDeviceGet")? },
            device_get_name: unsafe {
                load_symbol(library, b"cuDeviceGetName\0", "cuDeviceGetName")?
            },
            device_get_attribute: unsafe {
                load_symbol(library, b"cuDeviceGetAttribute\0", "cuDeviceGetAttribute")?
            },
            device_total_mem: unsafe {
                load_symbol(library, b"cuDeviceTotalMem_v2\0", "cuDeviceTotalMem_v2")?
            },
            device_primary_ctx_retain: unsafe {
                load_symbol(
                    library,
                    b"cuDevicePrimaryCtxRetain\0",
                    "cuDevicePrimaryCtxRetain",
                )?
            },
            device_primary_ctx_release: unsafe {
                load_symbol(
                    library,
                    b"cuDevicePrimaryCtxRelease_v2\0",
                    "cuDevicePrimaryCtxRelease_v2",
                )?
            },
            ctx_push_current: unsafe {
                load_symbol(library, b"cuCtxPushCurrent_v2\0", "cuCtxPushCurrent_v2")?
            },
            ctx_pop_current: unsafe {
                load_symbol(library, b"cuCtxPopCurrent_v2\0", "cuCtxPopCurrent_v2")?
            },
            ctx_synchronize: unsafe {
                load_symbol(library, b"cuCtxSynchronize\0", "cuCtxSynchronize")?
            },
            mem_alloc: unsafe { load_symbol(library, b"cuMemAlloc_v2\0", "cuMemAlloc_v2")? },
            mem_free: unsafe { load_symbol(library, b"cuMemFree_v2\0", "cuMemFree_v2")? },
            mem_get_info: unsafe { load_symbol(library, b"cuMemGetInfo_v2\0", "cuMemGetInfo_v2")? },
            memcpy_htod: unsafe { load_symbol(library, b"cuMemcpyHtoD_v2\0", "cuMemcpyHtoD_v2")? },
            memcpy_dtoh: unsafe { load_symbol(library, b"cuMemcpyDtoH_v2\0", "cuMemcpyDtoH_v2")? },
            memset_d8: unsafe { load_symbol(library, b"cuMemsetD8_v2\0", "cuMemsetD8_v2")? },
            get_error_name: unsafe { load_symbol(library, b"cuGetErrorName\0", "cuGetErrorName")? },
            get_error_string: unsafe {
                load_symbol(library, b"cuGetErrorString\0", "cuGetErrorString")?
            },
        })
    }
}

unsafe fn load_symbol<T: Copy>(
    library: &Library,
    symbol: &'static [u8],
    symbol_name: &'static str,
) -> Result<T, CudaError> {
    unsafe { library.get::<T>(symbol) }
        .map(|loaded| *loaded)
        .map_err(|error| CudaError::MissingDriverSymbol {
            symbol: symbol_name,
            detail: error.to_string(),
        })
}

fn driver_library_candidates() -> &'static [&'static str] {
    #[cfg(target_os = "windows")]
    {
        &["nvcuda.dll"]
    }
    #[cfg(target_os = "linux")]
    {
        &["libcuda.so.1", "libcuda.so"]
    }
    #[cfg(target_os = "macos")]
    {
        &["libcuda.dylib"]
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        &["libcuda.so.1", "libcuda.so"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static EVENTS: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn missing_driver_reports_every_candidate() {
        let error = match CudaDriver::load_from_candidates([
            "/definitely/missing/libcuda-one.so",
            "/definitely/missing/libcuda-two.so",
        ]) {
            Ok(_) => panic!("missing CUDA driver candidates must fail"),
            Err(error) => error,
        };
        let CudaError::DriverUnavailable { attempts } = error else {
            panic!("unexpected error: {error:?}");
        };
        assert_eq!(attempts.len(), 2);
        assert!(attempts[0].contains("libcuda-one.so"));
        assert!(attempts[1].contains("libcuda-two.so"));
    }

    #[test]
    fn auto_selection_only_treats_missing_driver_or_no_device_as_cpu_eligible() {
        assert!(
            CudaError::DriverUnavailable {
                attempts: vec!["missing".to_string()]
            }
            .is_unavailable_for_auto_selection()
        );
        assert!(
            CudaError::DriverCall {
                operation: "cuInit",
                code: 100,
                name: Some("CUDA_ERROR_NO_DEVICE".to_string()),
                description: None,
            }
            .is_unavailable_for_auto_selection()
        );
        assert!(
            !CudaError::DriverCall {
                operation: "cuInit",
                code: 35,
                name: Some("CUDA_ERROR_INSUFFICIENT_DRIVER".to_string()),
                description: None,
            }
            .is_unavailable_for_auto_selection()
        );
    }

    #[test]
    fn fake_driver_probes_b200_and_manages_allocation_lifetime() {
        let _test_guard = CUDA_TEST_LOCK
            .lock()
            .expect("CUDA test lock should be held");
        EVENTS.lock().expect("events lock should be held").clear();
        let driver = fake_driver();

        let info = driver.device_info(0).expect("fake device should probe");
        assert_eq!(info.name, "NVIDIA B200");
        assert_eq!(info.compute_capability, CudaComputeCapability::new(10, 0));
        assert_eq!(info.total_memory_bytes, 192 * 1024 * 1024 * 1024);
        assert!(info.unified_addressing);

        let context = driver
            .retain_primary_context(0)
            .expect("fake context should retain");
        assert_eq!(
            context.memory_info().expect("memory info should query"),
            CudaMemoryInfo {
                free_bytes: 180 * 1024 * 1024 * 1024,
                total_bytes: 192 * 1024 * 1024 * 1024,
            }
        );
        let allocation = context.allocate(4096).expect("allocation should succeed");
        assert_eq!(allocation.device_ptr(), 0x4000);
        assert_eq!(allocation.byte_len(), 4096);
        drop(allocation);
        drop(context);

        assert_eq!(
            *EVENTS.lock().expect("events lock should be held"),
            [
                "retain",
                "push",
                "memory-info",
                "pop",
                "push",
                "alloc",
                "pop",
                "push",
                "free",
                "pop",
                "release"
            ]
        );
    }

    #[test]
    fn allocation_copies_and_fills_stay_within_owned_device_memory() {
        let _test_guard = CUDA_TEST_LOCK
            .lock()
            .expect("CUDA test lock should be held");
        EVENTS.lock().expect("events lock should be held").clear();
        let driver = fake_driver();
        let context = driver
            .retain_primary_context(0)
            .expect("fake context should retain");
        let mut allocation = context.allocate(16).expect("allocation should succeed");
        assert_eq!(
            allocation
                .device_ptr_at(4, 4)
                .expect("checked pointer should stay within the allocation"),
            allocation.device_ptr() + 4
        );

        allocation
            .copy_from_host(4, &[1, 2, 3, 4])
            .expect("host-to-device copy should succeed");
        let mut output = [0_u8; 4];
        allocation
            .copy_to_host(8, &mut output)
            .expect("device-to-host copy should succeed");
        assert_eq!(output, [0xa5; 4]);
        allocation.fill(0).expect("device fill should succeed");
        context.synchronize().expect("context should synchronize");

        assert_eq!(
            allocation
                .copy_from_host(14, &[1, 2, 3])
                .expect_err("out-of-bounds copy must fail"),
            CudaError::AllocationOutOfBounds {
                offset: 14,
                byte_len: 3,
                allocation_byte_len: 16,
            }
        );
        assert_eq!(
            allocation
                .device_ptr_at(15, 2)
                .expect_err("out-of-bounds pointer must fail"),
            CudaError::AllocationOutOfBounds {
                offset: 15,
                byte_len: 2,
                allocation_byte_len: 16,
            }
        );

        allocation.free().expect("allocation should free");
        assert_eq!(
            allocation
                .device_ptr_at(0, 1)
                .expect_err("freed allocation must reject pointers"),
            CudaError::FreedAllocation
        );
        assert_eq!(
            allocation
                .copy_to_host(0, &mut output)
                .expect_err("freed allocation must reject access"),
            CudaError::FreedAllocation
        );
    }

    fn fake_driver() -> CudaDriver {
        CudaDriver {
            inner: Arc::new(CudaDriverInner {
                _library: None,
                api: CudaApi {
                    init: fake_init,
                    driver_get_version: fake_driver_get_version,
                    device_get_count: fake_device_get_count,
                    device_get: fake_device_get,
                    device_get_name: fake_device_get_name,
                    device_get_attribute: fake_device_get_attribute,
                    device_total_mem: fake_device_total_mem,
                    device_primary_ctx_retain: fake_primary_ctx_retain,
                    device_primary_ctx_release: fake_primary_ctx_release,
                    ctx_push_current: fake_ctx_push_current,
                    ctx_pop_current: fake_ctx_pop_current,
                    ctx_synchronize: fake_ctx_synchronize,
                    mem_alloc: fake_mem_alloc,
                    mem_free: fake_mem_free,
                    mem_get_info: fake_mem_get_info,
                    memcpy_htod: fake_memcpy_htod,
                    memcpy_dtoh: fake_memcpy_dtoh,
                    memset_d8: fake_memset_d8,
                    get_error_name: fake_get_error,
                    get_error_string: fake_get_error,
                },
            }),
        }
    }

    unsafe extern "C" fn fake_init(_flags: c_uint) -> c_int {
        CUDA_SUCCESS
    }

    unsafe extern "C" fn fake_driver_get_version(version: *mut c_int) -> c_int {
        unsafe { *version = 12_080 };
        CUDA_SUCCESS
    }

    unsafe extern "C" fn fake_device_get_count(count: *mut c_int) -> c_int {
        unsafe { *count = 1 };
        CUDA_SUCCESS
    }

    unsafe extern "C" fn fake_device_get(device: *mut CuDevice, ordinal: c_int) -> c_int {
        unsafe { *device = ordinal };
        CUDA_SUCCESS
    }

    unsafe extern "C" fn fake_device_get_name(
        name: *mut c_char,
        len: c_int,
        _device: CuDevice,
    ) -> c_int {
        let value = b"NVIDIA B200\0";
        assert!(len as usize >= value.len());
        unsafe { std::ptr::copy_nonoverlapping(value.as_ptr().cast(), name, value.len()) };
        CUDA_SUCCESS
    }

    unsafe extern "C" fn fake_device_get_attribute(
        value: *mut c_int,
        attribute: c_int,
        _device: CuDevice,
    ) -> c_int {
        let result = match attribute {
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR => 10,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR => 0,
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT => 148,
            CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING => 1,
            _ => 0,
        };
        unsafe { *value = result };
        CUDA_SUCCESS
    }

    unsafe extern "C" fn fake_device_total_mem(total: *mut usize, _device: CuDevice) -> c_int {
        unsafe { *total = 192 * 1024 * 1024 * 1024 };
        CUDA_SUCCESS
    }

    unsafe extern "C" fn fake_primary_ctx_retain(
        context: *mut CuContextHandle,
        _device: CuDevice,
    ) -> c_int {
        EVENTS
            .lock()
            .expect("events lock should be held")
            .push("retain");
        unsafe { *context = 0x1000usize as CuContextHandle };
        CUDA_SUCCESS
    }

    unsafe extern "C" fn fake_primary_ctx_release(_device: CuDevice) -> c_int {
        EVENTS
            .lock()
            .expect("events lock should be held")
            .push("release");
        CUDA_SUCCESS
    }

    unsafe extern "C" fn fake_ctx_push_current(_context: CuContextHandle) -> c_int {
        EVENTS
            .lock()
            .expect("events lock should be held")
            .push("push");
        CUDA_SUCCESS
    }

    unsafe extern "C" fn fake_ctx_pop_current(context: *mut CuContextHandle) -> c_int {
        EVENTS
            .lock()
            .expect("events lock should be held")
            .push("pop");
        unsafe { *context = 0x1000usize as CuContextHandle };
        CUDA_SUCCESS
    }

    unsafe extern "C" fn fake_ctx_synchronize() -> c_int {
        EVENTS
            .lock()
            .expect("events lock should be held")
            .push("synchronize");
        CUDA_SUCCESS
    }

    unsafe extern "C" fn fake_mem_alloc(device_ptr: *mut CuDevicePtr, _len: usize) -> c_int {
        EVENTS
            .lock()
            .expect("events lock should be held")
            .push("alloc");
        unsafe { *device_ptr = 0x4000 };
        CUDA_SUCCESS
    }

    unsafe extern "C" fn fake_mem_free(_device_ptr: CuDevicePtr) -> c_int {
        EVENTS
            .lock()
            .expect("events lock should be held")
            .push("free");
        CUDA_SUCCESS
    }

    unsafe extern "C" fn fake_mem_get_info(free: *mut usize, total: *mut usize) -> c_int {
        EVENTS
            .lock()
            .expect("events lock should be held")
            .push("memory-info");
        unsafe {
            *free = 180 * 1024 * 1024 * 1024;
            *total = 192 * 1024 * 1024 * 1024;
        }
        CUDA_SUCCESS
    }

    unsafe extern "C" fn fake_memcpy_htod(
        device_ptr: CuDevicePtr,
        source: *const c_void,
        byte_len: usize,
    ) -> c_int {
        assert_eq!(device_ptr, 0x4004);
        assert_eq!(
            unsafe { std::slice::from_raw_parts(source.cast::<u8>(), byte_len) },
            [1, 2, 3, 4]
        );
        EVENTS
            .lock()
            .expect("events lock should be held")
            .push("copy-htod");
        CUDA_SUCCESS
    }

    unsafe extern "C" fn fake_memcpy_dtoh(
        destination: *mut c_void,
        device_ptr: CuDevicePtr,
        byte_len: usize,
    ) -> c_int {
        assert_eq!(device_ptr, 0x4008);
        unsafe { std::ptr::write_bytes(destination, 0xa5, byte_len) };
        EVENTS
            .lock()
            .expect("events lock should be held")
            .push("copy-dtoh");
        CUDA_SUCCESS
    }

    unsafe extern "C" fn fake_memset_d8(
        device_ptr: CuDevicePtr,
        value: u8,
        byte_len: usize,
    ) -> c_int {
        assert_eq!(device_ptr, 0x4000);
        assert_eq!(value, 0);
        assert_eq!(byte_len, 16);
        EVENTS
            .lock()
            .expect("events lock should be held")
            .push("memset");
        CUDA_SUCCESS
    }

    unsafe extern "C" fn fake_get_error(_code: c_int, text: *mut *const c_char) -> c_int {
        unsafe { *text = c"CUDA_ERROR_UNKNOWN".as_ptr() };
        CUDA_SUCCESS
    }
}
