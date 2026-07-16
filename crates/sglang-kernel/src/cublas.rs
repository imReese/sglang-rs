use std::ffi::{c_int, c_void};
use std::fmt;

use libloading::Library;

use crate::cuda::{CudaContext, CudaDeviceAllocation, CudaError};

const CUBLAS_STATUS_SUCCESS: c_int = 0;
const CUBLAS_OP_T: c_int = 1;
const F32_BYTES: usize = std::mem::size_of::<f32>();

type CublasHandle = *mut c_void;
type CublasCreate = unsafe extern "C" fn(*mut CublasHandle) -> c_int;
type CublasDestroy = unsafe extern "C" fn(CublasHandle) -> c_int;
type CublasSgemv = unsafe extern "C" fn(
    CublasHandle,
    c_int,
    c_int,
    c_int,
    *const f32,
    *const f32,
    c_int,
    *const f32,
    c_int,
    *const f32,
    *mut f32,
    c_int,
) -> c_int;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CudaBlasError {
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
    },
    NullHandle,
    InvalidMatrixDimensions {
        rows: usize,
        columns: usize,
    },
    MatrixSizeOverflow,
    DimensionExceedsCublasInt {
        dimension: &'static str,
        value: usize,
    },
    Cuda(CudaError),
}

impl fmt::Display for CudaBlasError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LibraryUnavailable { attempts } => write!(
                formatter,
                "cuBLAS library is unavailable; tried {}",
                attempts.join(", ")
            ),
            Self::MissingLibrarySymbol { symbol, detail } => {
                write!(
                    formatter,
                    "cuBLAS library is missing symbol {symbol}: {detail}"
                )
            }
            Self::Call { operation, status } => {
                write!(
                    formatter,
                    "cuBLAS call {operation} failed with status {status}"
                )
            }
            Self::NullHandle => formatter.write_str("cublasCreate_v2 returned a null handle"),
            Self::InvalidMatrixDimensions { rows, columns } => write!(
                formatter,
                "cuBLAS matrix dimensions must be non-zero, got [{rows}, {columns}]"
            ),
            Self::MatrixSizeOverflow => formatter.write_str("cuBLAS matrix byte size overflowed"),
            Self::DimensionExceedsCublasInt { dimension, value } => write!(
                formatter,
                "cuBLAS {dimension} dimension {value} exceeds the c_int API limit"
            ),
            Self::Cuda(error) => write!(formatter, "CUDA operation for cuBLAS failed: {error}"),
        }
    }
}

impl std::error::Error for CudaBlasError {}

impl From<CudaError> for CudaBlasError {
    fn from(value: CudaError) -> Self {
        Self::Cuda(value)
    }
}

pub struct CudaBlas {
    context: CudaContext,
    handle: usize,
    api: CudaBlasApi,
    _library: Library,
}

impl CudaBlas {
    pub fn load(context: &CudaContext) -> Result<Self, CudaBlasError> {
        Self::load_from_candidates(context, cublas_library_candidates())
    }

    pub fn load_from_candidates<I, S>(
        context: &CudaContext,
        candidates: I,
    ) -> Result<Self, CudaBlasError>
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
            let api = unsafe { CudaBlasApi::load(&library) }?;
            let mut handle = std::ptr::null_mut();
            context.with_current(|| {
                check_status(unsafe { (api.create)(&mut handle) }, "cublasCreate_v2")
            })?;
            if handle.is_null() {
                return Err(CudaBlasError::NullHandle);
            }
            return Ok(Self {
                context: context.clone(),
                handle: handle as usize,
                api,
                _library: library,
            });
        }
        Err(CudaBlasError::LibraryUnavailable { attempts })
    }

    pub fn sgemv_row_major(
        &self,
        matrix: &CudaDeviceAllocation,
        rows: usize,
        columns: usize,
        vector: &CudaDeviceAllocation,
        vector_offset_bytes: usize,
        output: &mut CudaDeviceAllocation,
    ) -> Result<(), CudaBlasError> {
        let shape = SgemvShape::new(rows, columns)?;
        let matrix_ptr = matrix.device_ptr_at(0, shape.matrix_byte_len)? as *const f32;
        let vector_ptr =
            vector.device_ptr_at(vector_offset_bytes, shape.vector_byte_len)? as *const f32;
        let output_ptr = output.device_ptr_at(0, shape.output_byte_len)? as *mut f32;
        self.context.with_current(|| {
            launch_sgemv(
                self.api,
                SgemvLaunch {
                    handle: self.handle as CublasHandle,
                    shape,
                    matrix: matrix_ptr,
                    vector: vector_ptr,
                    output: output_ptr,
                    alpha: 1.0,
                    beta: 0.0,
                },
            )
        })?;
        self.context.synchronize()?;
        Ok(())
    }
}

impl Drop for CudaBlas {
    fn drop(&mut self) {
        let result = self.context.with_current(|| {
            check_status(
                unsafe { (self.api.destroy)(self.handle as CublasHandle) },
                "cublasDestroy_v2",
            )
        });
        if let Err(error) = result {
            eprintln!("failed to destroy cuBLAS handle: {error}");
        }
    }
}

#[derive(Clone, Copy)]
struct CudaBlasApi {
    create: CublasCreate,
    destroy: CublasDestroy,
    sgemv: CublasSgemv,
}

impl CudaBlasApi {
    unsafe fn load(library: &Library) -> Result<Self, CudaBlasError> {
        Ok(Self {
            create: unsafe { load_symbol(library, b"cublasCreate_v2\0", "cublasCreate_v2")? },
            destroy: unsafe { load_symbol(library, b"cublasDestroy_v2\0", "cublasDestroy_v2")? },
            sgemv: unsafe { load_symbol(library, b"cublasSgemv_v2\0", "cublasSgemv_v2")? },
        })
    }
}

unsafe fn load_symbol<T: Copy>(
    library: &Library,
    symbol: &'static [u8],
    symbol_name: &'static str,
) -> Result<T, CudaBlasError> {
    unsafe { library.get::<T>(symbol) }
        .map(|loaded| *loaded)
        .map_err(|error| CudaBlasError::MissingLibrarySymbol {
            symbol: symbol_name,
            detail: error.to_string(),
        })
}

fn check_status(status: c_int, operation: &'static str) -> Result<(), CudaBlasError> {
    if status == CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(CudaBlasError::Call { operation, status })
    }
}

#[derive(Clone, Copy)]
struct SgemvLaunch {
    handle: CublasHandle,
    shape: SgemvShape,
    matrix: *const f32,
    vector: *const f32,
    output: *mut f32,
    alpha: f32,
    beta: f32,
}

fn launch_sgemv(api: CudaBlasApi, launch: SgemvLaunch) -> Result<(), CudaBlasError> {
    check_status(
        unsafe {
            (api.sgemv)(
                launch.handle,
                CUBLAS_OP_T,
                launch.shape.columns,
                launch.shape.rows,
                &launch.alpha,
                launch.matrix,
                launch.shape.columns,
                launch.vector,
                1,
                &launch.beta,
                launch.output,
                1,
            )
        },
        "cublasSgemv_v2",
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SgemvShape {
    rows: c_int,
    columns: c_int,
    matrix_byte_len: usize,
    vector_byte_len: usize,
    output_byte_len: usize,
}

impl SgemvShape {
    fn new(rows: usize, columns: usize) -> Result<Self, CudaBlasError> {
        if rows == 0 || columns == 0 {
            return Err(CudaBlasError::InvalidMatrixDimensions { rows, columns });
        }
        let matrix_byte_len = rows
            .checked_mul(columns)
            .and_then(|elements| elements.checked_mul(F32_BYTES))
            .ok_or(CudaBlasError::MatrixSizeOverflow)?;
        let vector_byte_len = columns
            .checked_mul(F32_BYTES)
            .ok_or(CudaBlasError::MatrixSizeOverflow)?;
        let output_byte_len = rows
            .checked_mul(F32_BYTES)
            .ok_or(CudaBlasError::MatrixSizeOverflow)?;
        Ok(Self {
            rows: c_int::try_from(rows).map_err(|_| CudaBlasError::DimensionExceedsCublasInt {
                dimension: "row",
                value: rows,
            })?,
            columns: c_int::try_from(columns).map_err(|_| {
                CudaBlasError::DimensionExceedsCublasInt {
                    dimension: "column",
                    value: columns,
                }
            })?,
            matrix_byte_len,
            vector_byte_len,
            output_byte_len,
        })
    }
}

fn cublas_library_candidates() -> &'static [&'static str] {
    #[cfg(target_os = "windows")]
    {
        &["cublas64_13.dll", "cublas64_12.dll", "cublas64_11.dll"]
    }
    #[cfg(target_os = "linux")]
    {
        &[
            "libcublas.so.13",
            "libcublas.so.12",
            "libcublas.so.11",
            "libcublas.so",
        ]
    }
    #[cfg(target_os = "macos")]
    {
        &["libcublas.dylib"]
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        &["libcublas.so.13", "libcublas.so.12", "libcublas.so"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static SGEMV_CALL: Mutex<Option<(c_int, c_int, c_int, c_int)>> = Mutex::new(None);

    #[test]
    fn row_major_sgemv_shape_maps_to_transposed_cublas_view() {
        let shape = SgemvShape::new(32_000, 4_096).expect("shape should be valid");

        assert_eq!(shape.rows, 32_000);
        assert_eq!(shape.columns, 4_096);
        assert_eq!(shape.matrix_byte_len, 32_000 * 4_096 * 4);
        assert_eq!(shape.vector_byte_len, 4_096 * 4);
        assert_eq!(shape.output_byte_len, 32_000 * 4);
    }

    #[test]
    fn row_major_sgemv_rejects_empty_or_overflowing_shapes() {
        assert_eq!(
            SgemvShape::new(0, 4).expect_err("empty rows must fail"),
            CudaBlasError::InvalidMatrixDimensions {
                rows: 0,
                columns: 4,
            }
        );
        assert_eq!(
            SgemvShape::new(usize::MAX, 2).expect_err("overflow must fail"),
            CudaBlasError::MatrixSizeOverflow
        );
    }

    #[test]
    fn row_major_sgemv_calls_cublas_with_transposed_column_major_view() {
        *SGEMV_CALL.lock().expect("sgemv call lock should be held") = None;
        let api = CudaBlasApi {
            create: fake_create,
            destroy: fake_destroy,
            sgemv: fake_sgemv,
        };
        let shape = SgemvShape::new(32_000, 4_096).expect("shape should be valid");
        launch_sgemv(
            api,
            SgemvLaunch {
                handle: 0x1000usize as CublasHandle,
                shape,
                matrix: 0x2000usize as *const f32,
                vector: 0x3000usize as *const f32,
                output: 0x4000usize as *mut f32,
                alpha: 1.0,
                beta: 0.0,
            },
        )
        .expect("fake sgemv should succeed");

        assert_eq!(
            *SGEMV_CALL.lock().expect("sgemv call lock should be held"),
            Some((CUBLAS_OP_T, 4_096, 32_000, 4_096))
        );
    }

    unsafe extern "C" fn fake_create(_handle: *mut CublasHandle) -> c_int {
        CUBLAS_STATUS_SUCCESS
    }

    unsafe extern "C" fn fake_destroy(_handle: CublasHandle) -> c_int {
        CUBLAS_STATUS_SUCCESS
    }

    unsafe extern "C" fn fake_sgemv(
        handle: CublasHandle,
        operation: c_int,
        rows: c_int,
        columns: c_int,
        alpha: *const f32,
        matrix: *const f32,
        leading_dimension: c_int,
        vector: *const f32,
        vector_stride: c_int,
        beta: *const f32,
        output: *mut f32,
        output_stride: c_int,
    ) -> c_int {
        assert_eq!(handle as usize, 0x1000);
        assert_eq!(unsafe { *alpha }, 1.0);
        assert_eq!(unsafe { *beta }, 0.0);
        assert_eq!(matrix as usize, 0x2000);
        assert_eq!(vector as usize, 0x3000);
        assert_eq!(output as usize, 0x4000);
        assert_eq!(vector_stride, 1);
        assert_eq!(output_stride, 1);
        *SGEMV_CALL.lock().expect("sgemv call lock should be held") =
            Some((operation, rows, columns, leading_dimension));
        CUBLAS_STATUS_SUCCESS
    }
}
