use std::ffi::c_void;
use std::fmt;

use crate::cuda::{
    CudaComputeCapability, CudaContext, CudaDeviceAllocation, CudaError, CudaFunction,
    CudaLaunchDimensions, CudaModule,
};
use crate::nvrtc::{NvrtcCompiler, NvrtcError};

const CUDA_F32_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void sglang_rms_norm_f32(
    const float* input,
    const float* weight,
    float* output,
    unsigned long long rows,
    unsigned int width,
    float epsilon) {
  const unsigned long long row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float partial[];
  float sum_squares = 0.0f;
  const unsigned long long row_offset = row * width;
  for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
    const float value = input[row_offset + column];
    sum_squares += value * value;
  }
  partial[threadIdx.x] = sum_squares;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inverse_rms = rsqrtf(partial[0] / static_cast<float>(width) + epsilon);
  for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
    output[row_offset + column] = input[row_offset + column] * inverse_rms * weight[column];
  }
}

extern "C" __global__ void sglang_silu_mul_f32(
    const float* gate,
    const float* up,
    float* output,
    unsigned long long element_count) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  const float gate_value = gate[index];
  const float silu = gate_value / (1.0f + expf(-gate_value));
  output[index] = silu * up[index];
}
"#;

const CUDA_BLOCK_SIZE: u32 = 256;

#[derive(Debug)]
pub enum CudaF32KernelError {
    Cuda(CudaError),
    Nvrtc(NvrtcError),
    ZeroDimension(&'static str),
    ShapeOverflow,
    DimensionTooLarge {
        dimension: &'static str,
        value: usize,
        maximum: usize,
    },
    InvalidEpsilon(f32),
    DeviceMismatch {
        allocation: &'static str,
        expected_ordinal: usize,
        actual_ordinal: usize,
    },
}

impl fmt::Display for CudaF32KernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(error) => write!(formatter, "CUDA kernel operation failed: {error}"),
            Self::Nvrtc(error) => write!(formatter, "CUDA kernel compilation failed: {error}"),
            Self::ZeroDimension(dimension) => {
                write!(
                    formatter,
                    "CUDA kernel dimension {dimension} must be non-zero"
                )
            }
            Self::ShapeOverflow => formatter.write_str("CUDA kernel tensor shape overflowed"),
            Self::DimensionTooLarge {
                dimension,
                value,
                maximum,
            } => write!(
                formatter,
                "CUDA kernel dimension {dimension}={value} exceeds maximum {maximum}"
            ),
            Self::InvalidEpsilon(epsilon) => write!(
                formatter,
                "CUDA RMSNorm epsilon must be finite and positive, got {epsilon}"
            ),
            Self::DeviceMismatch {
                allocation,
                expected_ordinal,
                actual_ordinal,
            } => write!(
                formatter,
                "CUDA kernel allocation {allocation} is on device {actual_ordinal}, expected device {expected_ordinal}"
            ),
        }
    }
}

impl std::error::Error for CudaF32KernelError {}

impl From<CudaError> for CudaF32KernelError {
    fn from(value: CudaError) -> Self {
        Self::Cuda(value)
    }
}

impl From<NvrtcError> for CudaF32KernelError {
    fn from(value: NvrtcError) -> Self {
        Self::Nvrtc(value)
    }
}

pub struct CudaF32Kernels {
    context: CudaContext,
    _module: CudaModule,
    rms_norm: CudaFunction,
    silu_mul: CudaFunction,
}

impl CudaF32Kernels {
    pub fn compile(
        context: &CudaContext,
        compute_capability: CudaComputeCapability,
    ) -> Result<Self, CudaF32KernelError> {
        let compiler = NvrtcCompiler::load()?;
        let ptx = compiler.compile_ptx(
            CUDA_F32_KERNEL_SOURCE,
            "sglang_f32_kernels.cu",
            compute_capability,
        )?;
        let module = context.load_module(&ptx)?;
        let rms_norm = module.get_function("sglang_rms_norm_f32")?;
        let silu_mul = module.get_function("sglang_silu_mul_f32")?;
        Ok(Self {
            context: context.clone(),
            _module: module,
            rms_norm,
            silu_mul,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm(
        &self,
        input: &CudaDeviceAllocation,
        input_offset: usize,
        weight: &CudaDeviceAllocation,
        weight_offset: usize,
        output: &mut CudaDeviceAllocation,
        output_offset: usize,
        rows: usize,
        width: usize,
        epsilon: f32,
    ) -> Result<(), CudaF32KernelError> {
        if rows == 0 {
            return Err(CudaF32KernelError::ZeroDimension("rows"));
        }
        if width == 0 {
            return Err(CudaF32KernelError::ZeroDimension("width"));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(CudaF32KernelError::InvalidEpsilon(epsilon));
        }
        self.validate_device("input", input)?;
        self.validate_device("weight", weight)?;
        self.validate_device("output", output)?;

        let element_count = rows
            .checked_mul(width)
            .ok_or(CudaF32KernelError::ShapeOverflow)?;
        let tensor_byte_len = f32_byte_len(element_count)?;
        let weight_byte_len = f32_byte_len(width)?;
        let mut input_ptr = input.device_ptr_at(input_offset, tensor_byte_len)?;
        let mut weight_ptr = weight.device_ptr_at(weight_offset, weight_byte_len)?;
        let mut output_ptr = output.device_ptr_at(output_offset, tensor_byte_len)?;
        let mut rows_u64 =
            u64::try_from(rows).map_err(|_| CudaF32KernelError::DimensionTooLarge {
                dimension: "rows",
                value: rows,
                maximum: u64::MAX as usize,
            })?;
        let mut width_u32 =
            u32::try_from(width).map_err(|_| CudaF32KernelError::DimensionTooLarge {
                dimension: "width",
                value: width,
                maximum: u32::MAX as usize,
            })?;
        let grid_x = u32::try_from(rows).map_err(|_| CudaF32KernelError::DimensionTooLarge {
            dimension: "rows",
            value: rows,
            maximum: u32::MAX as usize,
        })?;
        let mut epsilon = epsilon;
        let mut arguments = [
            argument_pointer(&mut input_ptr),
            argument_pointer(&mut weight_ptr),
            argument_pointer(&mut output_ptr),
            argument_pointer(&mut rows_u64),
            argument_pointer(&mut width_u32),
            argument_pointer(&mut epsilon),
        ];
        unsafe {
            self.rms_norm.launch(
                CudaLaunchDimensions::new(grid_x, 1, 1),
                CudaLaunchDimensions::new(CUDA_BLOCK_SIZE, 1, 1),
                CUDA_BLOCK_SIZE * size_of::<f32>() as u32,
                &mut arguments,
            )?;
        }
        self.context.synchronize()?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn silu_mul(
        &self,
        gate: &CudaDeviceAllocation,
        gate_offset: usize,
        up: &CudaDeviceAllocation,
        up_offset: usize,
        output: &mut CudaDeviceAllocation,
        output_offset: usize,
        element_count: usize,
    ) -> Result<(), CudaF32KernelError> {
        if element_count == 0 {
            return Err(CudaF32KernelError::ZeroDimension("element_count"));
        }
        self.validate_device("gate", gate)?;
        self.validate_device("up", up)?;
        self.validate_device("output", output)?;

        let tensor_byte_len = f32_byte_len(element_count)?;
        let mut gate_ptr = gate.device_ptr_at(gate_offset, tensor_byte_len)?;
        let mut up_ptr = up.device_ptr_at(up_offset, tensor_byte_len)?;
        let mut output_ptr = output.device_ptr_at(output_offset, tensor_byte_len)?;
        let mut element_count_u64 =
            u64::try_from(element_count).map_err(|_| CudaF32KernelError::DimensionTooLarge {
                dimension: "element_count",
                value: element_count,
                maximum: u64::MAX as usize,
            })?;
        let grid = elementwise_grid(element_count)?;
        let mut arguments = [
            argument_pointer(&mut gate_ptr),
            argument_pointer(&mut up_ptr),
            argument_pointer(&mut output_ptr),
            argument_pointer(&mut element_count_u64),
        ];
        unsafe {
            self.silu_mul.launch(
                grid,
                CudaLaunchDimensions::new(CUDA_BLOCK_SIZE, 1, 1),
                0,
                &mut arguments,
            )?;
        }
        self.context.synchronize()?;
        Ok(())
    }

    fn validate_device(
        &self,
        allocation_name: &'static str,
        allocation: &CudaDeviceAllocation,
    ) -> Result<(), CudaF32KernelError> {
        let expected_ordinal = self.context.device_ordinal();
        let actual_ordinal = allocation.device_ordinal();
        if actual_ordinal == expected_ordinal {
            Ok(())
        } else {
            Err(CudaF32KernelError::DeviceMismatch {
                allocation: allocation_name,
                expected_ordinal,
                actual_ordinal,
            })
        }
    }
}

fn f32_byte_len(element_count: usize) -> Result<usize, CudaF32KernelError> {
    element_count
        .checked_mul(size_of::<f32>())
        .ok_or(CudaF32KernelError::ShapeOverflow)
}

fn elementwise_grid(element_count: usize) -> Result<CudaLaunchDimensions, CudaF32KernelError> {
    let block_size = CUDA_BLOCK_SIZE as usize;
    let block_count = element_count
        .checked_add(block_size - 1)
        .ok_or(CudaF32KernelError::ShapeOverflow)?
        / block_size;
    let grid_x = u32::try_from(block_count).map_err(|_| CudaF32KernelError::DimensionTooLarge {
        dimension: "elementwise grid blocks",
        value: block_count,
        maximum: u32::MAX as usize,
    })?;
    Ok(CudaLaunchDimensions::new(grid_x, 1, 1))
}

fn argument_pointer<T>(value: &mut T) -> *mut c_void {
    (value as *mut T).cast()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elementwise_grid_rounds_up_without_device_assumptions() {
        assert_eq!(
            elementwise_grid(1).expect("one element should launch"),
            CudaLaunchDimensions::new(1, 1, 1)
        );
        assert_eq!(
            elementwise_grid(256).expect("one block should launch"),
            CudaLaunchDimensions::new(1, 1, 1)
        );
        assert_eq!(
            elementwise_grid(257).expect("partial block should launch"),
            CudaLaunchDimensions::new(2, 1, 1)
        );
    }

    #[test]
    fn embedded_source_exports_production_kernel_entry_points() {
        assert!(CUDA_F32_KERNEL_SOURCE.contains("sglang_rms_norm_f32"));
        assert!(CUDA_F32_KERNEL_SOURCE.contains("sglang_silu_mul_f32"));
        assert!(CUDA_F32_KERNEL_SOURCE.contains("rsqrtf"));
        assert!(CUDA_F32_KERNEL_SOURCE.contains("expf"));
    }
}
