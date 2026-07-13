use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fmt;

use libloading::Library;

use crate::cuda::CudaComputeCapability;

const NVRTC_SUCCESS: c_int = 0;

type NvrtcProgram = *mut c_void;
type NvrtcCreateProgram = unsafe extern "C" fn(
    *mut NvrtcProgram,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> c_int;
type NvrtcDestroyProgram = unsafe extern "C" fn(*mut NvrtcProgram) -> c_int;
type NvrtcCompileProgram = unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> c_int;
type NvrtcGetPtxSize = unsafe extern "C" fn(NvrtcProgram, *mut usize) -> c_int;
type NvrtcGetPtx = unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> c_int;
type NvrtcGetProgramLogSize = unsafe extern "C" fn(NvrtcProgram, *mut usize) -> c_int;
type NvrtcGetProgramLog = unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> c_int;
type NvrtcGetErrorString = unsafe extern "C" fn(c_int) -> *const c_char;
type NvrtcVersion = unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NvrtcError {
    LibraryUnavailable {
        attempts: Vec<String>,
    },
    MissingSymbol {
        symbol: &'static str,
        detail: String,
    },
    InvalidSource,
    InvalidProgramName(String),
    CreateReturnedNull,
    Call {
        operation: &'static str,
        code: i32,
        description: Option<String>,
    },
    Compilation {
        code: i32,
        architecture: String,
        log: String,
    },
    EmptyPtx,
}

impl fmt::Display for NvrtcError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LibraryUnavailable { attempts } => write!(
                formatter,
                "NVRTC library is unavailable; install the CUDA toolkit and expose one of: {}",
                attempts.join(", ")
            ),
            Self::MissingSymbol { symbol, detail } => {
                write!(
                    formatter,
                    "NVRTC library is missing symbol {symbol}: {detail}"
                )
            }
            Self::InvalidSource => {
                formatter.write_str("CUDA kernel source contains an interior NUL byte")
            }
            Self::InvalidProgramName(name) => {
                write!(
                    formatter,
                    "NVRTC program name contains an interior NUL: {name:?}"
                )
            }
            Self::CreateReturnedNull => {
                formatter.write_str("nvrtcCreateProgram returned a null program")
            }
            Self::Call {
                operation,
                code,
                description,
            } => {
                write!(formatter, "NVRTC call {operation} failed with code {code}")?;
                if let Some(description) = description {
                    write!(formatter, ": {description}")?;
                }
                Ok(())
            }
            Self::Compilation {
                code,
                architecture,
                log,
            } => write!(
                formatter,
                "NVRTC compilation for {architecture} failed with code {code}: {log}"
            ),
            Self::EmptyPtx => formatter.write_str("NVRTC returned an empty PTX image"),
        }
    }
}

impl std::error::Error for NvrtcError {}

pub struct NvrtcCompiler {
    _library: Option<Library>,
    api: NvrtcApi,
}

impl NvrtcCompiler {
    pub fn load() -> Result<Self, NvrtcError> {
        Self::load_from_candidates(nvrtc_library_candidates())
    }

    pub fn load_from_candidates<I, S>(candidates: I) -> Result<Self, NvrtcError>
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
            let api = unsafe { NvrtcApi::load(&library) }?;
            return Ok(Self {
                _library: Some(library),
                api,
            });
        }
        Err(NvrtcError::LibraryUnavailable { attempts })
    }

    pub fn version(&self) -> Result<(i32, i32), NvrtcError> {
        let mut major = 0;
        let mut minor = 0;
        self.check(
            unsafe { (self.api.version)(&mut major, &mut minor) },
            "nvrtcVersion",
        )?;
        Ok((major, minor))
    }

    pub fn compile_ptx(
        &self,
        source: &str,
        program_name: &str,
        compute_capability: CudaComputeCapability,
    ) -> Result<Vec<u8>, NvrtcError> {
        let source = CString::new(source).map_err(|_| NvrtcError::InvalidSource)?;
        let program_name = CString::new(program_name)
            .map_err(|_| NvrtcError::InvalidProgramName(program_name.to_string()))?;
        let mut program = std::ptr::null_mut();
        self.check(
            unsafe {
                (self.api.create_program)(
                    &mut program,
                    source.as_ptr(),
                    program_name.as_ptr(),
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                )
            },
            "nvrtcCreateProgram",
        )?;
        if program.is_null() {
            return Err(NvrtcError::CreateReturnedNull);
        }
        let program = NvrtcProgramGuard {
            api: self.api,
            handle: program,
        };

        let architecture = format!(
            "compute_{}{}",
            compute_capability.major, compute_capability.minor
        );
        let options = [
            CString::new("--std=c++17").expect("static NVRTC option is valid"),
            CString::new(format!("--gpu-architecture={architecture}"))
                .expect("generated NVRTC architecture option is valid"),
        ];
        let option_ptrs = options
            .iter()
            .map(|option| option.as_ptr())
            .collect::<Vec<_>>();
        let compile_status = unsafe {
            (self.api.compile_program)(
                program.handle,
                option_ptrs.len() as c_int,
                option_ptrs.as_ptr(),
            )
        };
        if compile_status != NVRTC_SUCCESS {
            return Err(NvrtcError::Compilation {
                code: compile_status,
                architecture,
                log: self.program_log(program.handle),
            });
        }

        let mut ptx_size = 0;
        self.check(
            unsafe { (self.api.get_ptx_size)(program.handle, &mut ptx_size) },
            "nvrtcGetPTXSize",
        )?;
        if ptx_size == 0 {
            return Err(NvrtcError::EmptyPtx);
        }
        let mut ptx = vec![0_u8; ptx_size];
        self.check(
            unsafe { (self.api.get_ptx)(program.handle, ptx.as_mut_ptr().cast()) },
            "nvrtcGetPTX",
        )?;
        Ok(ptx)
    }

    fn check(&self, code: c_int, operation: &'static str) -> Result<(), NvrtcError> {
        if code == NVRTC_SUCCESS {
            Ok(())
        } else {
            Err(NvrtcError::Call {
                operation,
                code,
                description: self.error_description(code),
            })
        }
    }

    fn error_description(&self, code: c_int) -> Option<String> {
        let description = unsafe { (self.api.get_error_string)(code) };
        if description.is_null() {
            None
        } else {
            Some(
                unsafe { CStr::from_ptr(description) }
                    .to_string_lossy()
                    .into_owned(),
            )
        }
    }

    fn program_log(&self, program: NvrtcProgram) -> String {
        let mut log_size = 0;
        let status = unsafe { (self.api.get_program_log_size)(program, &mut log_size) };
        if status != NVRTC_SUCCESS {
            return format!(
                "unable to read compiler log: {}",
                self.error_description(status)
                    .unwrap_or_else(|| format!("NVRTC code {status}"))
            );
        }
        if log_size == 0 {
            return "compiler returned no diagnostic log".to_string();
        }
        let mut log = vec![0_u8; log_size];
        let status = unsafe { (self.api.get_program_log)(program, log.as_mut_ptr().cast()) };
        if status != NVRTC_SUCCESS {
            return format!(
                "unable to read compiler log: {}",
                self.error_description(status)
                    .unwrap_or_else(|| format!("NVRTC code {status}"))
            );
        }
        String::from_utf8_lossy(&log)
            .trim_end_matches('\0')
            .to_string()
    }
}

struct NvrtcProgramGuard {
    api: NvrtcApi,
    handle: NvrtcProgram,
}

impl Drop for NvrtcProgramGuard {
    fn drop(&mut self) {
        unsafe { (self.api.destroy_program)(&mut self.handle) };
    }
}

#[derive(Clone, Copy)]
struct NvrtcApi {
    create_program: NvrtcCreateProgram,
    destroy_program: NvrtcDestroyProgram,
    compile_program: NvrtcCompileProgram,
    get_ptx_size: NvrtcGetPtxSize,
    get_ptx: NvrtcGetPtx,
    get_program_log_size: NvrtcGetProgramLogSize,
    get_program_log: NvrtcGetProgramLog,
    get_error_string: NvrtcGetErrorString,
    version: NvrtcVersion,
}

impl NvrtcApi {
    unsafe fn load(library: &Library) -> Result<Self, NvrtcError> {
        Ok(Self {
            create_program: unsafe {
                load_symbol(library, b"nvrtcCreateProgram\0", "nvrtcCreateProgram")?
            },
            destroy_program: unsafe {
                load_symbol(library, b"nvrtcDestroyProgram\0", "nvrtcDestroyProgram")?
            },
            compile_program: unsafe {
                load_symbol(library, b"nvrtcCompileProgram\0", "nvrtcCompileProgram")?
            },
            get_ptx_size: unsafe { load_symbol(library, b"nvrtcGetPTXSize\0", "nvrtcGetPTXSize")? },
            get_ptx: unsafe { load_symbol(library, b"nvrtcGetPTX\0", "nvrtcGetPTX")? },
            get_program_log_size: unsafe {
                load_symbol(
                    library,
                    b"nvrtcGetProgramLogSize\0",
                    "nvrtcGetProgramLogSize",
                )?
            },
            get_program_log: unsafe {
                load_symbol(library, b"nvrtcGetProgramLog\0", "nvrtcGetProgramLog")?
            },
            get_error_string: unsafe {
                load_symbol(library, b"nvrtcGetErrorString\0", "nvrtcGetErrorString")?
            },
            version: unsafe { load_symbol(library, b"nvrtcVersion\0", "nvrtcVersion")? },
        })
    }
}

unsafe fn load_symbol<T: Copy>(
    library: &Library,
    symbol: &'static [u8],
    symbol_name: &'static str,
) -> Result<T, NvrtcError> {
    unsafe { library.get::<T>(symbol) }
        .map(|loaded| *loaded)
        .map_err(|error| NvrtcError::MissingSymbol {
            symbol: symbol_name,
            detail: error.to_string(),
        })
}

fn nvrtc_library_candidates() -> &'static [&'static str] {
    #[cfg(target_os = "windows")]
    {
        &["nvrtc64_130_0.dll", "nvrtc64_120_0.dll"]
    }
    #[cfg(target_os = "linux")]
    {
        &[
            "libnvrtc.so.13",
            "libnvrtc.so.12",
            "libnvrtc.so.11.2",
            "libnvrtc.so",
        ]
    }
    #[cfg(target_os = "macos")]
    {
        &["libnvrtc.dylib"]
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        &["libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static COMPILE_OPTIONS: Mutex<Vec<String>> = Mutex::new(Vec::new());

    #[test]
    fn missing_nvrtc_reports_every_candidate() {
        let error = match NvrtcCompiler::load_from_candidates([
            "/definitely/missing/libnvrtc-one.so",
            "/definitely/missing/libnvrtc-two.so",
        ]) {
            Ok(_) => panic!("missing NVRTC candidates must fail"),
            Err(error) => error,
        };
        let NvrtcError::LibraryUnavailable { attempts } = error else {
            panic!("unexpected error: {error:?}");
        };
        assert_eq!(attempts.len(), 2);
        assert!(attempts[0].contains("libnvrtc-one.so"));
        assert!(attempts[1].contains("libnvrtc-two.so"));
    }

    #[test]
    fn compiler_targets_detected_compute_capability() {
        COMPILE_OPTIONS
            .lock()
            .expect("compile options lock should be held")
            .clear();
        let compiler = fake_compiler();
        assert_eq!(compiler.version().expect("version should load"), (12, 8));
        let ptx = compiler
            .compile_ptx(
                "extern \"C\" __global__ void test() {}",
                "test.cu",
                CudaComputeCapability::new(10, 0),
            )
            .expect("source should compile");

        assert_eq!(ptx, b".version 8.0\0");
        assert_eq!(
            *COMPILE_OPTIONS
                .lock()
                .expect("compile options lock should be held"),
            ["--std=c++17", "--gpu-architecture=compute_100"]
        );
    }

    fn fake_compiler() -> NvrtcCompiler {
        NvrtcCompiler {
            _library: None,
            api: NvrtcApi {
                create_program: fake_create_program,
                destroy_program: fake_destroy_program,
                compile_program: fake_compile_program,
                get_ptx_size: fake_get_ptx_size,
                get_ptx: fake_get_ptx,
                get_program_log_size: fake_get_program_log_size,
                get_program_log: fake_get_program_log,
                get_error_string: fake_get_error_string,
                version: fake_version,
            },
        }
    }

    unsafe extern "C" fn fake_create_program(
        program: *mut NvrtcProgram,
        source: *const c_char,
        name: *const c_char,
        header_count: c_int,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> c_int {
        assert_eq!(
            unsafe { CStr::from_ptr(source) }.to_bytes(),
            b"extern \"C\" __global__ void test() {}"
        );
        assert_eq!(unsafe { CStr::from_ptr(name) }.to_bytes(), b"test.cu");
        assert_eq!(header_count, 0);
        assert!(headers.is_null());
        assert!(include_names.is_null());
        unsafe { *program = 0x7000usize as NvrtcProgram };
        NVRTC_SUCCESS
    }

    unsafe extern "C" fn fake_destroy_program(program: *mut NvrtcProgram) -> c_int {
        assert_eq!(unsafe { *program } as usize, 0x7000);
        unsafe { *program = std::ptr::null_mut() };
        NVRTC_SUCCESS
    }

    unsafe extern "C" fn fake_compile_program(
        program: NvrtcProgram,
        option_count: c_int,
        options: *const *const c_char,
    ) -> c_int {
        assert_eq!(program as usize, 0x7000);
        let options = unsafe { std::slice::from_raw_parts(options, option_count as usize) };
        let options = options
            .iter()
            .map(|option| {
                unsafe { CStr::from_ptr(*option) }
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        *COMPILE_OPTIONS
            .lock()
            .expect("compile options lock should be held") = options;
        NVRTC_SUCCESS
    }

    unsafe extern "C" fn fake_get_ptx_size(program: NvrtcProgram, size: *mut usize) -> c_int {
        assert_eq!(program as usize, 0x7000);
        unsafe { *size = b".version 8.0\0".len() };
        NVRTC_SUCCESS
    }

    unsafe extern "C" fn fake_get_ptx(program: NvrtcProgram, ptx: *mut c_char) -> c_int {
        assert_eq!(program as usize, 0x7000);
        let output = b".version 8.0\0";
        unsafe { std::ptr::copy_nonoverlapping(output.as_ptr().cast(), ptx, output.len()) };
        NVRTC_SUCCESS
    }

    unsafe extern "C" fn fake_get_program_log_size(
        _program: NvrtcProgram,
        size: *mut usize,
    ) -> c_int {
        unsafe { *size = 0 };
        NVRTC_SUCCESS
    }

    unsafe extern "C" fn fake_get_program_log(_program: NvrtcProgram, _log: *mut c_char) -> c_int {
        NVRTC_SUCCESS
    }

    unsafe extern "C" fn fake_get_error_string(_code: c_int) -> *const c_char {
        c"NVRTC_ERROR_UNKNOWN".as_ptr()
    }

    unsafe extern "C" fn fake_version(major: *mut c_int, minor: *mut c_int) -> c_int {
        unsafe {
            *major = 12;
            *minor = 8;
        }
        NVRTC_SUCCESS
    }
}
