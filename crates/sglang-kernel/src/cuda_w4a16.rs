use std::ffi::c_void;
use std::fmt;

use crate::cuda::{
    CudaComputeCapability, CudaContext, CudaDeviceAllocation, CudaError, CudaFunction,
    CudaLaunchDimensions, CudaModule,
};
use crate::nvrtc::{NvrtcCompiler, NvrtcError};

const CUDA_W4A16_SOURCE: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void sglang_w4a16_gemv_bf16(
    const __nv_bfloat16* input,
    const unsigned int* packed_weight,
    const __nv_bfloat16* scales,
    __nv_bfloat16* output,
    unsigned int columns,
    unsigned int group_size) {
  const unsigned int row = blockIdx.x;
  const unsigned int packed_columns = columns / 8;
  const unsigned int groups_per_row = columns / group_size;
  float sum = 0.0f;
  for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
    const unsigned int word = packed_weight[row * packed_columns + column / 8];
    const unsigned int nibble = (word >> ((column % 8) * 4)) & 0xf;
    const int quantized = static_cast<int>(nibble) - 8;
    const float scale = __bfloat162float(
        scales[row * groups_per_row + column / group_size]);
    sum += __bfloat162float(input[column]) * static_cast<float>(quantized) * scale;
  }
  extern __shared__ float partial[];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[row] = __float2bfloat16_rn(partial[0]);
  }
}
"#;

const CUDA_BLOCK_SIZE: u32 = 256;
const BF16_BYTES: usize = 2;
const PACKED_I32_BYTES: usize = 4;
const PACKED_FACTOR: usize = 8;
const MINIMUM_W4A16_COMPUTE_CAPABILITY: CudaComputeCapability = CudaComputeCapability::new(8, 0);

#[derive(Debug)]
pub enum CudaW4A16KernelError {
    Cuda(CudaError),
    Nvrtc(NvrtcError),
    UnsupportedComputeCapability {
        actual: CudaComputeCapability,
        minimum: CudaComputeCapability,
    },
    InvalidShape(String),
    ShapeOverflow,
    DimensionTooLarge {
        dimension: &'static str,
        value: usize,
        maximum: usize,
    },
    DeviceMismatch {
        allocation: &'static str,
        expected_ordinal: usize,
        actual_ordinal: usize,
    },
}

impl fmt::Display for CudaW4A16KernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(error) => write!(formatter, "CUDA W4A16 kernel operation failed: {error}"),
            Self::Nvrtc(error) => {
                write!(formatter, "CUDA W4A16 kernel compilation failed: {error}")
            }
            Self::UnsupportedComputeCapability { actual, minimum } => write!(
                formatter,
                "CUDA W4A16 kernels require compute capability {minimum} or newer, found {actual}"
            ),
            Self::InvalidShape(message) => write!(formatter, "invalid W4A16 shape: {message}"),
            Self::ShapeOverflow => formatter.write_str("CUDA W4A16 tensor shape overflowed"),
            Self::DimensionTooLarge {
                dimension,
                value,
                maximum,
            } => write!(
                formatter,
                "CUDA W4A16 dimension {dimension}={value} exceeds maximum {maximum}"
            ),
            Self::DeviceMismatch {
                allocation,
                expected_ordinal,
                actual_ordinal,
            } => write!(
                formatter,
                "CUDA W4A16 allocation {allocation} is on device {actual_ordinal}, expected device {expected_ordinal}"
            ),
        }
    }
}

impl std::error::Error for CudaW4A16KernelError {}

impl From<CudaError> for CudaW4A16KernelError {
    fn from(value: CudaError) -> Self {
        Self::Cuda(value)
    }
}

impl From<NvrtcError> for CudaW4A16KernelError {
    fn from(value: NvrtcError) -> Self {
        Self::Nvrtc(value)
    }
}

pub struct CudaW4A16Kernels {
    context: CudaContext,
    _module: CudaModule,
    gemv: CudaFunction,
}

pub struct CudaW4A16GemvLaunch<'a> {
    pub input: &'a CudaDeviceAllocation,
    pub packed_weight: &'a CudaDeviceAllocation,
    pub scales: &'a CudaDeviceAllocation,
    pub output: &'a mut CudaDeviceAllocation,
    pub rows: usize,
    pub columns: usize,
    pub group_size: usize,
}

impl CudaW4A16Kernels {
    pub fn compile(
        context: &CudaContext,
        compute_capability: CudaComputeCapability,
    ) -> Result<Self, CudaW4A16KernelError> {
        if compute_capability < MINIMUM_W4A16_COMPUTE_CAPABILITY {
            return Err(CudaW4A16KernelError::UnsupportedComputeCapability {
                actual: compute_capability,
                minimum: MINIMUM_W4A16_COMPUTE_CAPABILITY,
            });
        }
        let compiler = NvrtcCompiler::load()?;
        let ptx = compiler.compile_ptx(
            CUDA_W4A16_SOURCE,
            "sglang_w4a16_kernels.cu",
            compute_capability,
        )?;
        let module = context.load_module(&ptx)?;
        Ok(Self {
            context: context.clone(),
            gemv: module.get_function("sglang_w4a16_gemv_bf16")?,
            _module: module,
        })
    }

    pub fn gemv(&self, launch: CudaW4A16GemvLaunch<'_>) -> Result<(), CudaW4A16KernelError> {
        let CudaW4A16GemvLaunch {
            input,
            packed_weight,
            scales,
            output,
            rows,
            columns,
            group_size,
        } = launch;
        validate_shape(rows, columns, group_size)?;
        self.validate_devices(&[
            ("input", input),
            ("packed_weight", packed_weight),
            ("scales", scales),
            ("output", output),
        ])?;
        let packed_count = checked_product(rows, columns / PACKED_FACTOR)?;
        let scale_count = checked_product(rows, columns / group_size)?;
        let mut input_ptr = input.device_ptr_at(0, checked_bytes(columns, BF16_BYTES)?)?;
        let mut packed_ptr =
            packed_weight.device_ptr_at(0, checked_bytes(packed_count, PACKED_I32_BYTES)?)?;
        let mut scales_ptr = scales.device_ptr_at(0, checked_bytes(scale_count, BF16_BYTES)?)?;
        let mut output_ptr = output.device_ptr_at(0, checked_bytes(rows, BF16_BYTES)?)?;
        let mut columns_u32 = dimension_u32("columns", columns)?;
        let mut group_size_u32 = dimension_u32("group_size", group_size)?;
        let mut arguments = [
            argument_pointer(&mut input_ptr),
            argument_pointer(&mut packed_ptr),
            argument_pointer(&mut scales_ptr),
            argument_pointer(&mut output_ptr),
            argument_pointer(&mut columns_u32),
            argument_pointer(&mut group_size_u32),
        ];
        unsafe {
            self.gemv.launch(
                CudaLaunchDimensions::new(dimension_u32("rows", rows)?, 1, 1),
                CudaLaunchDimensions::new(CUDA_BLOCK_SIZE, 1, 1),
                CUDA_BLOCK_SIZE * size_of::<f32>() as u32,
                &mut arguments,
            )?;
        }
        self.context.synchronize()?;
        Ok(())
    }

    fn validate_devices(
        &self,
        allocations: &[(&'static str, &CudaDeviceAllocation)],
    ) -> Result<(), CudaW4A16KernelError> {
        let expected_ordinal = self.context.device_ordinal();
        for (name, allocation) in allocations {
            let actual_ordinal = allocation.device_ordinal();
            if actual_ordinal != expected_ordinal {
                return Err(CudaW4A16KernelError::DeviceMismatch {
                    allocation: name,
                    expected_ordinal,
                    actual_ordinal,
                });
            }
        }
        Ok(())
    }
}

fn validate_shape(
    rows: usize,
    columns: usize,
    group_size: usize,
) -> Result<(), CudaW4A16KernelError> {
    if rows == 0 || columns == 0 || group_size == 0 {
        return Err(CudaW4A16KernelError::InvalidShape(
            "rows, columns, and group_size must be non-zero".to_string(),
        ));
    }
    if !columns.is_multiple_of(PACKED_FACTOR) || !columns.is_multiple_of(group_size) {
        return Err(CudaW4A16KernelError::InvalidShape(format!(
            "columns {columns} must be divisible by packed factor {PACKED_FACTOR} and group size {group_size}"
        )));
    }
    Ok(())
}

fn checked_product(left: usize, right: usize) -> Result<usize, CudaW4A16KernelError> {
    left.checked_mul(right)
        .ok_or(CudaW4A16KernelError::ShapeOverflow)
}

fn checked_bytes(
    element_count: usize,
    element_bytes: usize,
) -> Result<usize, CudaW4A16KernelError> {
    checked_product(element_count, element_bytes)
}

fn dimension_u32(dimension: &'static str, value: usize) -> Result<u32, CudaW4A16KernelError> {
    u32::try_from(value).map_err(|_| CudaW4A16KernelError::DimensionTooLarge {
        dimension,
        value,
        maximum: u32::MAX as usize,
    })
}

fn argument_pointer<T>(value: &mut T) -> *mut c_void {
    (value as *mut T).cast()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_floor_is_architecture_based() {
        assert_eq!(
            MINIMUM_W4A16_COMPUTE_CAPABILITY,
            CudaComputeCapability::new(8, 0)
        );
        assert!(CudaComputeCapability::new(10, 0) >= MINIMUM_W4A16_COMPUTE_CAPABILITY);
    }

    #[test]
    fn shape_contract_rejects_unpacked_or_partial_groups() {
        assert!(validate_shape(1, 64, 32).is_ok());
        assert!(validate_shape(1, 36, 32).is_err());
        assert!(validate_shape(0, 64, 32).is_err());
    }

    #[test]
    fn embedded_source_decodes_unsigned_offset_int4() {
        assert!(CUDA_W4A16_SOURCE.contains("sglang_w4a16_gemv_bf16"));
        assert!(CUDA_W4A16_SOURCE.contains("static_cast<int>(nibble) - 8"));
    }
}
