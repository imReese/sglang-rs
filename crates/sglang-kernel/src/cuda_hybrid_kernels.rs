use std::ffi::c_void;
use std::fmt;

use crate::cuda::{
    CudaComputeCapability, CudaContext, CudaDeviceAllocation, CudaError, CudaFunction,
    CudaLaunchDimensions, CudaModule,
};
use crate::nvrtc::{NvrtcCompiler, NvrtcError};

const CUDA_HYBRID_SOURCE: &str = r#"
#include <cuda_bf16.h>
#include <math.h>

extern "C" __global__ void sglang_silu_inplace_bf16(
    __nv_bfloat16* values,
    unsigned long long element_count) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  const float value = __bfloat162float(values[index]);
  values[index] = __float2bfloat16_rn(value / (1.0f + expf(-value)));
}

extern "C" __global__ void sglang_l2_normalize_heads_inplace_bf16(
    __nv_bfloat16* values,
    unsigned long long row_count,
    unsigned int width,
    float output_scale,
    float epsilon) {
  const unsigned long long row = blockIdx.x;
  if (row >= row_count) {
    return;
  }
  extern __shared__ float sums[];
  const unsigned long long row_offset = row * width;
  float sum = 0.0f;
  for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
    const float value = __bfloat162float(values[row_offset + column]);
    sum += value * value;
  }
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      sums[threadIdx.x] += sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float multiplier = output_scale * rsqrtf(sums[0] + epsilon);
  for (unsigned int column = threadIdx.x; column < width; column += blockDim.x) {
    const float value = __bfloat162float(values[row_offset + column]);
    values[row_offset + column] = __float2bfloat16_rn(value * multiplier);
  }
}

extern "C" __global__ void sglang_kda_decay_bf16_to_f32(
    const __nv_bfloat16* raw_forget,
    const float* dt_bias,
    const float* a_log,
    float* decay,
    unsigned long long batch_size,
    unsigned int head_count,
    unsigned int key_head_dim) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long key_size =
      static_cast<unsigned long long>(head_count) * key_head_dim;
  const unsigned long long element_count = batch_size * key_size;
  if (index >= element_count) {
    return;
  }
  const unsigned long long column = index % key_size;
  const unsigned int head = column / key_head_dim;
  const float raw = __bfloat162float(raw_forget[index]) + dt_bias[column];
  const float softplus = raw > 20.0f ? raw : log1pf(expf(raw));
  decay[index] = expf(-expf(a_log[head]) * softplus);
}

extern "C" __global__ void sglang_sigmoid_bf16_to_f32(
    const __nv_bfloat16* input,
    float* output,
    unsigned long long element_count) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  const float value = __bfloat162float(input[index]);
  output[index] = 1.0f / (1.0f + expf(-value));
}

extern "C" __global__ void sglang_sigmoid_mul_bf16(
    __nv_bfloat16* values,
    const __nv_bfloat16* gate,
    unsigned long long element_count) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  const float gate_value = __bfloat162float(gate[index]);
  const float multiplier = 1.0f / (1.0f + expf(-gate_value));
  values[index] = __float2bfloat16_rn(
      __bfloat162float(values[index]) * multiplier);
}
"#;

const CUDA_BLOCK_SIZE: u32 = 256;
const BF16_BYTES: usize = 2;
const F32_BYTES: usize = 4;
const MINIMUM_BF16_COMPUTE_CAPABILITY: CudaComputeCapability = CudaComputeCapability::new(8, 0);

#[derive(Debug)]
pub enum CudaHybridKernelError {
    Cuda(CudaError),
    Nvrtc(NvrtcError),
    UnsupportedComputeCapability {
        actual: CudaComputeCapability,
        minimum: CudaComputeCapability,
    },
    ZeroDimension(&'static str),
    InvalidScalar {
        name: &'static str,
        value: f32,
    },
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
    MisalignedOffset {
        allocation: &'static str,
        offset: usize,
        alignment: usize,
    },
}

impl fmt::Display for CudaHybridKernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(error) => write!(formatter, "CUDA hybrid kernel failed: {error}"),
            Self::Nvrtc(error) => {
                write!(formatter, "CUDA hybrid kernel compilation failed: {error}")
            }
            Self::UnsupportedComputeCapability { actual, minimum } => write!(
                formatter,
                "CUDA BF16 hybrid kernels require compute capability {minimum} or newer; device reports {actual}"
            ),
            Self::ZeroDimension(dimension) => {
                write!(
                    formatter,
                    "CUDA hybrid kernel dimension {dimension} must be non-zero"
                )
            }
            Self::InvalidScalar { name, value } => write!(
                formatter,
                "CUDA hybrid kernel scalar {name} must be finite and positive, got {value}"
            ),
            Self::ShapeOverflow => formatter.write_str("CUDA hybrid kernel shape overflowed"),
            Self::DimensionTooLarge {
                dimension,
                value,
                maximum,
            } => write!(
                formatter,
                "CUDA hybrid kernel dimension {dimension}={value} exceeds maximum {maximum}"
            ),
            Self::DeviceMismatch {
                allocation,
                expected_ordinal,
                actual_ordinal,
            } => write!(
                formatter,
                "CUDA hybrid kernel allocation {allocation} is on device {actual_ordinal}, expected device {expected_ordinal}"
            ),
            Self::MisalignedOffset {
                allocation,
                offset,
                alignment,
            } => write!(
                formatter,
                "CUDA hybrid kernel allocation {allocation} offset {offset} is not aligned to {alignment} bytes"
            ),
        }
    }
}

impl std::error::Error for CudaHybridKernelError {}

impl From<CudaError> for CudaHybridKernelError {
    fn from(value: CudaError) -> Self {
        Self::Cuda(value)
    }
}

impl From<NvrtcError> for CudaHybridKernelError {
    fn from(value: NvrtcError) -> Self {
        Self::Nvrtc(value)
    }
}

pub struct CudaBf16HybridKernels {
    context: CudaContext,
    _module: CudaModule,
    silu_inplace: CudaFunction,
    l2_normalize_heads: CudaFunction,
    kda_decay: CudaFunction,
    sigmoid_to_f32: CudaFunction,
    sigmoid_mul: CudaFunction,
}

pub struct CudaKdaDecayLaunch<'a> {
    pub raw_forget: &'a CudaDeviceAllocation,
    pub dt_bias: &'a CudaDeviceAllocation,
    pub a_log: &'a CudaDeviceAllocation,
    pub decay: &'a mut CudaDeviceAllocation,
    pub batch_size: usize,
    pub head_count: usize,
    pub key_head_dim: usize,
}

impl CudaBf16HybridKernels {
    pub fn compile(
        context: &CudaContext,
        compute_capability: CudaComputeCapability,
    ) -> Result<Self, CudaHybridKernelError> {
        if compute_capability < MINIMUM_BF16_COMPUTE_CAPABILITY {
            return Err(CudaHybridKernelError::UnsupportedComputeCapability {
                actual: compute_capability,
                minimum: MINIMUM_BF16_COMPUTE_CAPABILITY,
            });
        }
        let compiler = NvrtcCompiler::load()?;
        let ptx = compiler.compile_ptx(
            CUDA_HYBRID_SOURCE,
            "sglang_hybrid_kernels.cu",
            compute_capability,
        )?;
        let module = context.load_module(&ptx)?;
        Ok(Self {
            context: context.clone(),
            silu_inplace: module.get_function("sglang_silu_inplace_bf16")?,
            l2_normalize_heads: module.get_function("sglang_l2_normalize_heads_inplace_bf16")?,
            kda_decay: module.get_function("sglang_kda_decay_bf16_to_f32")?,
            sigmoid_to_f32: module.get_function("sglang_sigmoid_bf16_to_f32")?,
            sigmoid_mul: module.get_function("sglang_sigmoid_mul_bf16")?,
            _module: module,
        })
    }

    pub fn silu_inplace(
        &self,
        values: &mut CudaDeviceAllocation,
        element_count: usize,
    ) -> Result<(), CudaHybridKernelError> {
        validate_nonzero(&[("element_count", element_count)])?;
        self.validate_devices(&[("values", values)])?;
        let mut values_ptr = values.device_ptr_at(0, checked_bytes(element_count, BF16_BYTES)?)?;
        let mut element_count_u64 = dimension_u64("element_count", element_count)?;
        let mut arguments = [
            argument_pointer(&mut values_ptr),
            argument_pointer(&mut element_count_u64),
        ];
        self.launch_elementwise(&self.silu_inplace, element_count, &mut arguments)
    }

    pub fn l2_normalize_heads_inplace(
        &self,
        values: &mut CudaDeviceAllocation,
        values_offset: usize,
        row_count: usize,
        width: usize,
        output_scale: f32,
        epsilon: f32,
    ) -> Result<(), CudaHybridKernelError> {
        validate_nonzero(&[("row_count", row_count), ("width", width)])?;
        validate_positive_scalar("output_scale", output_scale)?;
        validate_positive_scalar("epsilon", epsilon)?;
        validate_alignment("values", values_offset, BF16_BYTES)?;
        self.validate_devices(&[("values", values)])?;
        let element_count = checked_product(row_count, width)?;
        let mut values_ptr =
            values.device_ptr_at(values_offset, checked_bytes(element_count, BF16_BYTES)?)?;
        let mut row_count_u64 = dimension_u64("row_count", row_count)?;
        let mut width_u32 = dimension_u32("width", width)?;
        let mut output_scale_value = output_scale;
        let mut epsilon_value = epsilon;
        let mut arguments = [
            argument_pointer(&mut values_ptr),
            argument_pointer(&mut row_count_u64),
            argument_pointer(&mut width_u32),
            argument_pointer(&mut output_scale_value),
            argument_pointer(&mut epsilon_value),
        ];
        unsafe {
            self.l2_normalize_heads.launch(
                CudaLaunchDimensions::new(dimension_u32("row_count", row_count)?, 1, 1),
                CudaLaunchDimensions::new(CUDA_BLOCK_SIZE, 1, 1),
                CUDA_BLOCK_SIZE * F32_BYTES as u32,
                &mut arguments,
            )?;
        }
        self.context.synchronize()?;
        Ok(())
    }

    pub fn kda_decay(&self, launch: CudaKdaDecayLaunch<'_>) -> Result<(), CudaHybridKernelError> {
        let CudaKdaDecayLaunch {
            raw_forget,
            dt_bias,
            a_log,
            decay,
            batch_size,
            head_count,
            key_head_dim,
        } = launch;
        validate_nonzero(&[
            ("batch_size", batch_size),
            ("head_count", head_count),
            ("key_head_dim", key_head_dim),
        ])?;
        self.validate_devices(&[
            ("raw_forget", raw_forget),
            ("dt_bias", dt_bias),
            ("a_log", a_log),
            ("decay", decay),
        ])?;
        let key_size = checked_product(head_count, key_head_dim)?;
        let element_count = checked_product(batch_size, key_size)?;
        let mut raw_forget_ptr =
            raw_forget.device_ptr_at(0, checked_bytes(element_count, BF16_BYTES)?)?;
        let mut dt_bias_ptr = dt_bias.device_ptr_at(0, checked_bytes(key_size, F32_BYTES)?)?;
        let mut a_log_ptr = a_log.device_ptr_at(0, checked_bytes(head_count, F32_BYTES)?)?;
        let mut decay_ptr = decay.device_ptr_at(0, checked_bytes(element_count, F32_BYTES)?)?;
        let mut batch_size_u64 = dimension_u64("batch_size", batch_size)?;
        let mut head_count_u32 = dimension_u32("head_count", head_count)?;
        let mut key_head_dim_u32 = dimension_u32("key_head_dim", key_head_dim)?;
        let mut arguments = [
            argument_pointer(&mut raw_forget_ptr),
            argument_pointer(&mut dt_bias_ptr),
            argument_pointer(&mut a_log_ptr),
            argument_pointer(&mut decay_ptr),
            argument_pointer(&mut batch_size_u64),
            argument_pointer(&mut head_count_u32),
            argument_pointer(&mut key_head_dim_u32),
        ];
        self.launch_elementwise(&self.kda_decay, element_count, &mut arguments)
    }

    pub fn sigmoid_to_f32(
        &self,
        input: &CudaDeviceAllocation,
        output: &mut CudaDeviceAllocation,
        element_count: usize,
    ) -> Result<(), CudaHybridKernelError> {
        validate_nonzero(&[("element_count", element_count)])?;
        self.validate_devices(&[("input", input), ("output", output)])?;
        let mut input_ptr = input.device_ptr_at(0, checked_bytes(element_count, BF16_BYTES)?)?;
        let mut output_ptr = output.device_ptr_at(0, checked_bytes(element_count, F32_BYTES)?)?;
        let mut element_count_u64 = dimension_u64("element_count", element_count)?;
        let mut arguments = [
            argument_pointer(&mut input_ptr),
            argument_pointer(&mut output_ptr),
            argument_pointer(&mut element_count_u64),
        ];
        self.launch_elementwise(&self.sigmoid_to_f32, element_count, &mut arguments)
    }

    pub fn sigmoid_mul_inplace(
        &self,
        values: &mut CudaDeviceAllocation,
        gate: &CudaDeviceAllocation,
        element_count: usize,
    ) -> Result<(), CudaHybridKernelError> {
        validate_nonzero(&[("element_count", element_count)])?;
        self.validate_devices(&[("values", values), ("gate", gate)])?;
        let byte_len = checked_bytes(element_count, BF16_BYTES)?;
        let mut values_ptr = values.device_ptr_at(0, byte_len)?;
        let mut gate_ptr = gate.device_ptr_at(0, byte_len)?;
        let mut element_count_u64 = dimension_u64("element_count", element_count)?;
        let mut arguments = [
            argument_pointer(&mut values_ptr),
            argument_pointer(&mut gate_ptr),
            argument_pointer(&mut element_count_u64),
        ];
        self.launch_elementwise(&self.sigmoid_mul, element_count, &mut arguments)
    }

    fn launch_elementwise(
        &self,
        function: &CudaFunction,
        element_count: usize,
        arguments: &mut [*mut c_void],
    ) -> Result<(), CudaHybridKernelError> {
        let block_count = element_count
            .checked_add(CUDA_BLOCK_SIZE as usize - 1)
            .ok_or(CudaHybridKernelError::ShapeOverflow)?
            / CUDA_BLOCK_SIZE as usize;
        unsafe {
            function.launch(
                CudaLaunchDimensions::new(
                    dimension_u32("elementwise grid blocks", block_count)?,
                    1,
                    1,
                ),
                CudaLaunchDimensions::new(CUDA_BLOCK_SIZE, 1, 1),
                0,
                arguments,
            )?;
        }
        self.context.synchronize()?;
        Ok(())
    }

    fn validate_devices(
        &self,
        allocations: &[(&'static str, &CudaDeviceAllocation)],
    ) -> Result<(), CudaHybridKernelError> {
        let expected_ordinal = self.context.device_ordinal();
        for (allocation, buffer) in allocations {
            let actual_ordinal = buffer.device_ordinal();
            if actual_ordinal != expected_ordinal {
                return Err(CudaHybridKernelError::DeviceMismatch {
                    allocation,
                    expected_ordinal,
                    actual_ordinal,
                });
            }
        }
        Ok(())
    }
}

fn validate_nonzero(dimensions: &[(&'static str, usize)]) -> Result<(), CudaHybridKernelError> {
    if let Some((dimension, _)) = dimensions.iter().find(|(_, value)| *value == 0) {
        return Err(CudaHybridKernelError::ZeroDimension(dimension));
    }
    Ok(())
}

fn validate_positive_scalar(name: &'static str, value: f32) -> Result<(), CudaHybridKernelError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(CudaHybridKernelError::InvalidScalar { name, value });
    }
    Ok(())
}

fn validate_alignment(
    allocation: &'static str,
    offset: usize,
    alignment: usize,
) -> Result<(), CudaHybridKernelError> {
    if !offset.is_multiple_of(alignment) {
        return Err(CudaHybridKernelError::MisalignedOffset {
            allocation,
            offset,
            alignment,
        });
    }
    Ok(())
}

fn checked_product(left: usize, right: usize) -> Result<usize, CudaHybridKernelError> {
    left.checked_mul(right)
        .ok_or(CudaHybridKernelError::ShapeOverflow)
}

fn checked_bytes(
    element_count: usize,
    element_size: usize,
) -> Result<usize, CudaHybridKernelError> {
    checked_product(element_count, element_size)
}

fn dimension_u32(dimension: &'static str, value: usize) -> Result<u32, CudaHybridKernelError> {
    u32::try_from(value).map_err(|_| CudaHybridKernelError::DimensionTooLarge {
        dimension,
        value,
        maximum: u32::MAX as usize,
    })
}

fn dimension_u64(dimension: &'static str, value: usize) -> Result<u64, CudaHybridKernelError> {
    u64::try_from(value).map_err(|_| CudaHybridKernelError::DimensionTooLarge {
        dimension,
        value,
        maximum: u64::MAX as usize,
    })
}

fn argument_pointer<T>(value: &mut T) -> *mut c_void {
    (value as *mut T).cast()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_and_shape_validation_fails_before_cuda() {
        assert!(matches!(
            validate_nonzero(&[("rows", 0)]),
            Err(CudaHybridKernelError::ZeroDimension("rows"))
        ));
        assert!(matches!(
            validate_positive_scalar("epsilon", f32::NAN),
            Err(CudaHybridKernelError::InvalidScalar {
                name: "epsilon",
                ..
            })
        ));
        assert!(matches!(
            checked_product(usize::MAX, 2),
            Err(CudaHybridKernelError::ShapeOverflow)
        ));
    }

    #[test]
    fn embedded_source_exports_complete_kda_activation_set() {
        for entry_point in [
            "sglang_silu_inplace_bf16",
            "sglang_l2_normalize_heads_inplace_bf16",
            "sglang_kda_decay_bf16_to_f32",
            "sglang_sigmoid_bf16_to_f32",
            "sglang_sigmoid_mul_bf16",
        ] {
            assert!(CUDA_HYBRID_SOURCE.contains(entry_point));
        }
    }
}
