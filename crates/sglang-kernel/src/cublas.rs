use std::ffi::{c_int, c_void};
use std::fmt;

use libloading::Library;

use crate::cuda::{CudaContext, CudaDeviceAllocation, CudaError};

const CUBLAS_STATUS_SUCCESS: c_int = 0;
const CUBLAS_OP_N: c_int = 0;
const CUBLAS_OP_T: c_int = 1;
const CUDA_R_16BF: c_int = 14;
const CUBLAS_COMPUTE_32F: c_int = 68;
const CUBLAS_GEMM_DEFAULT: c_int = -1;
const BF16_BYTES: usize = std::mem::size_of::<u16>();
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
type CublasGemmEx = unsafe extern "C" fn(
    CublasHandle,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const c_void,
    *const c_void,
    c_int,
    c_int,
    *const c_void,
    c_int,
    c_int,
    *const c_void,
    *mut c_void,
    c_int,
    c_int,
    c_int,
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
    InvalidGemmDimensions {
        rows: usize,
        input_columns: usize,
        output_columns: usize,
    },
    MatrixSizeOverflow,
    DimensionExceedsCublasInt {
        dimension: &'static str,
        value: usize,
    },
    AllocationDeviceMismatch {
        allocation: &'static str,
        expected_ordinal: usize,
        actual_ordinal: usize,
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
            Self::InvalidGemmDimensions {
                rows,
                input_columns,
                output_columns,
            } => write!(
                formatter,
                "cuBLAS GEMM dimensions must be non-zero, got input [{rows}, {input_columns}] and output [{rows}, {output_columns}]"
            ),
            Self::MatrixSizeOverflow => formatter.write_str("cuBLAS matrix byte size overflowed"),
            Self::DimensionExceedsCublasInt { dimension, value } => write!(
                formatter,
                "cuBLAS {dimension} dimension {value} exceeds the c_int API limit"
            ),
            Self::AllocationDeviceMismatch {
                allocation,
                expected_ordinal,
                actual_ordinal,
            } => write!(
                formatter,
                "cuBLAS {allocation} allocation belongs to CUDA device {actual_ordinal}, but the handle belongs to device {expected_ordinal}"
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
        self.validate_allocation_device(matrix, "matrix")?;
        self.validate_allocation_device(vector, "vector")?;
        self.validate_allocation_device(output, "output")?;
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

    pub fn bf16_gemm_row_major(
        &self,
        input: &CudaDeviceAllocation,
        rows: usize,
        input_columns: usize,
        weight: &CudaDeviceAllocation,
        output_columns: usize,
        output: &mut CudaDeviceAllocation,
    ) -> Result<(), CudaBlasError> {
        self.validate_allocation_device(input, "input")?;
        self.validate_allocation_device(weight, "weight")?;
        self.validate_allocation_device(output, "output")?;
        let shape = Bf16GemmShape::new(rows, input_columns, output_columns)?;
        let input_ptr = input.device_ptr_at(0, shape.input_byte_len)? as *const c_void;
        let weight_ptr = weight.device_ptr_at(0, shape.weight_byte_len)? as *const c_void;
        let output_ptr = output.device_ptr_at(0, shape.output_byte_len)? as *mut c_void;
        self.context.with_current(|| {
            launch_bf16_gemm(
                self.api,
                Bf16GemmLaunch {
                    handle: self.handle as CublasHandle,
                    shape,
                    input: input_ptr,
                    weight: weight_ptr,
                    output: output_ptr,
                    alpha: 1.0,
                    beta: 0.0,
                },
            )
        })?;
        Ok(())
    }

    fn validate_allocation_device(
        &self,
        allocation: &CudaDeviceAllocation,
        allocation_name: &'static str,
    ) -> Result<(), CudaBlasError> {
        let expected_ordinal = self.context.device_ordinal();
        let actual_ordinal = allocation.device_ordinal();
        if actual_ordinal != expected_ordinal {
            return Err(CudaBlasError::AllocationDeviceMismatch {
                allocation: allocation_name,
                expected_ordinal,
                actual_ordinal,
            });
        }
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
    gemm_ex: CublasGemmEx,
}

impl CudaBlasApi {
    unsafe fn load(library: &Library) -> Result<Self, CudaBlasError> {
        Ok(Self {
            create: unsafe { load_symbol(library, b"cublasCreate_v2\0", "cublasCreate_v2")? },
            destroy: unsafe { load_symbol(library, b"cublasDestroy_v2\0", "cublasDestroy_v2")? },
            sgemv: unsafe { load_symbol(library, b"cublasSgemv_v2\0", "cublasSgemv_v2")? },
            gemm_ex: unsafe { load_symbol(library, b"cublasGemmEx\0", "cublasGemmEx")? },
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

#[derive(Clone, Copy)]
struct Bf16GemmLaunch {
    handle: CublasHandle,
    shape: Bf16GemmShape,
    input: *const c_void,
    weight: *const c_void,
    output: *mut c_void,
    alpha: f32,
    beta: f32,
}

fn launch_bf16_gemm(api: CudaBlasApi, launch: Bf16GemmLaunch) -> Result<(), CudaBlasError> {
    check_status(
        unsafe {
            (api.gemm_ex)(
                launch.handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                launch.shape.output_columns,
                launch.shape.rows,
                launch.shape.input_columns,
                (&launch.alpha as *const f32).cast(),
                launch.weight,
                CUDA_R_16BF,
                launch.shape.input_columns,
                launch.input,
                CUDA_R_16BF,
                launch.shape.input_columns,
                (&launch.beta as *const f32).cast(),
                launch.output,
                CUDA_R_16BF,
                launch.shape.output_columns,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT,
            )
        },
        "cublasGemmEx",
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Bf16GemmShape {
    rows: c_int,
    input_columns: c_int,
    output_columns: c_int,
    input_byte_len: usize,
    weight_byte_len: usize,
    output_byte_len: usize,
}

impl Bf16GemmShape {
    fn new(
        rows: usize,
        input_columns: usize,
        output_columns: usize,
    ) -> Result<Self, CudaBlasError> {
        if rows == 0 || input_columns == 0 || output_columns == 0 {
            return Err(CudaBlasError::InvalidGemmDimensions {
                rows,
                input_columns,
                output_columns,
            });
        }
        let input_byte_len = matrix_byte_len(rows, input_columns, BF16_BYTES)?;
        let weight_byte_len = matrix_byte_len(output_columns, input_columns, BF16_BYTES)?;
        let output_byte_len = matrix_byte_len(rows, output_columns, BF16_BYTES)?;
        Ok(Self {
            rows: cublas_int_dimension("row", rows)?,
            input_columns: cublas_int_dimension("input column", input_columns)?,
            output_columns: cublas_int_dimension("output column", output_columns)?,
            input_byte_len,
            weight_byte_len,
            output_byte_len,
        })
    }
}

fn matrix_byte_len(
    rows: usize,
    columns: usize,
    element_bytes: usize,
) -> Result<usize, CudaBlasError> {
    rows.checked_mul(columns)
        .and_then(|elements| elements.checked_mul(element_bytes))
        .ok_or(CudaBlasError::MatrixSizeOverflow)
}

fn cublas_int_dimension(dimension: &'static str, value: usize) -> Result<c_int, CudaBlasError> {
    c_int::try_from(value)
        .map_err(|_| CudaBlasError::DimensionExceedsCublasInt { dimension, value })
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
    static GEMM_CALL: Mutex<Option<GemmCall>> = Mutex::new(None);

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct GemmCall {
        operation_a: c_int,
        operation_b: c_int,
        rows: c_int,
        columns: c_int,
        inner: c_int,
        leading_a: c_int,
        leading_b: c_int,
        leading_output: c_int,
        input_type: c_int,
        output_type: c_int,
        compute_type: c_int,
        algorithm: c_int,
    }

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
            gemm_ex: fake_gemm_ex,
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

    #[test]
    fn row_major_bf16_gemm_maps_input_weight_and_output_shapes() {
        let shape = Bf16GemmShape::new(8, 4_096, 16_384).expect("shape should be valid");

        assert_eq!(shape.rows, 8);
        assert_eq!(shape.input_columns, 4_096);
        assert_eq!(shape.output_columns, 16_384);
        assert_eq!(shape.input_byte_len, 8 * 4_096 * 2);
        assert_eq!(shape.weight_byte_len, 16_384 * 4_096 * 2);
        assert_eq!(shape.output_byte_len, 8 * 16_384 * 2);
        assert!(matches!(
            Bf16GemmShape::new(0, 4_096, 16_384),
            Err(CudaBlasError::InvalidGemmDimensions { .. })
        ));
    }

    #[test]
    fn row_major_bf16_gemm_calls_cublas_with_transposed_weight() {
        *GEMM_CALL.lock().expect("gemm call lock should be held") = None;
        let api = CudaBlasApi {
            create: fake_create,
            destroy: fake_destroy,
            sgemv: fake_sgemv,
            gemm_ex: fake_gemm_ex,
        };
        let shape = Bf16GemmShape::new(8, 4_096, 16_384).expect("shape should be valid");
        launch_bf16_gemm(
            api,
            Bf16GemmLaunch {
                handle: 0x1000usize as CublasHandle,
                shape,
                input: 0x2000usize as *const c_void,
                weight: 0x3000usize as *const c_void,
                output: 0x4000usize as *mut c_void,
                alpha: 1.0,
                beta: 0.0,
            },
        )
        .expect("fake gemm should succeed");

        assert_eq!(
            *GEMM_CALL.lock().expect("gemm call lock should be held"),
            Some(GemmCall {
                operation_a: CUBLAS_OP_T,
                operation_b: CUBLAS_OP_N,
                rows: 16_384,
                columns: 8,
                inner: 4_096,
                leading_a: 4_096,
                leading_b: 4_096,
                leading_output: 16_384,
                input_type: CUDA_R_16BF,
                output_type: CUDA_R_16BF,
                compute_type: CUBLAS_COMPUTE_32F,
                algorithm: CUBLAS_GEMM_DEFAULT,
            })
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

    unsafe extern "C" fn fake_gemm_ex(
        handle: CublasHandle,
        operation_a: c_int,
        operation_b: c_int,
        rows: c_int,
        columns: c_int,
        inner: c_int,
        alpha: *const c_void,
        a: *const c_void,
        a_type: c_int,
        leading_a: c_int,
        b: *const c_void,
        b_type: c_int,
        leading_b: c_int,
        beta: *const c_void,
        output: *mut c_void,
        output_type: c_int,
        leading_output: c_int,
        compute_type: c_int,
        algorithm: c_int,
    ) -> c_int {
        assert_eq!(handle as usize, 0x1000);
        assert_eq!(unsafe { *alpha.cast::<f32>() }, 1.0);
        assert_eq!(unsafe { *beta.cast::<f32>() }, 0.0);
        assert_eq!(a as usize, 0x3000);
        assert_eq!(b as usize, 0x2000);
        assert_eq!(output as usize, 0x4000);
        assert_eq!(a_type, b_type);
        *GEMM_CALL.lock().expect("gemm call lock should be held") = Some(GemmCall {
            operation_a,
            operation_b,
            rows,
            columns,
            inner,
            leading_a,
            leading_b,
            leading_output,
            input_type: a_type,
            output_type,
            compute_type,
            algorithm,
        });
        CUBLAS_STATUS_SUCCESS
    }
}
