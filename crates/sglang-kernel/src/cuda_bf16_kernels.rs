use std::ffi::c_void;
use std::fmt;

use crate::cuda::{
    CudaComputeCapability, CudaContext, CudaDeviceAllocation, CudaError, CudaFunction,
    CudaLaunchDimensions, CudaModule,
};
use crate::nvrtc::{NvrtcCompiler, NvrtcError};

const CUDA_BF16_DENSE_SOURCE: &str = r#"
#include <cuda_bf16.h>
#include <math.h>

extern "C" __global__ void sglang_embedding_lookup_bf16(
    const unsigned int* token_ids,
    const __nv_bfloat16* table,
    __nv_bfloat16* output,
    unsigned int* error_flag,
    unsigned long long row_count,
    unsigned long long vocabulary_size,
    unsigned int width) {
  const unsigned long long element =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long element_count = row_count * width;
  if (element >= element_count) {
    return;
  }
  const unsigned long long row = element / width;
  const unsigned int column = element % width;
  const unsigned int token_id = token_ids[row];
  if (token_id >= vocabulary_size) {
    atomicOr(error_flag, 1u);
    output[element] = __float2bfloat16_rn(0.0f);
    return;
  }
  output[element] = table[static_cast<unsigned long long>(token_id) * width + column];
}

extern "C" __global__ void sglang_rms_norm_bf16(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
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
    const float value = __bfloat162float(input[row_offset + column]);
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
    const float value = __bfloat162float(input[row_offset + column]);
    const float scale = __bfloat162float(weight[column]);
    output[row_offset + column] = __float2bfloat16_rn(value * inverse_rms * scale);
  }
}

extern "C" __global__ void sglang_add_bias_bf16(
    __nv_bfloat16* values,
    const __nv_bfloat16* bias,
    unsigned long long element_count,
    unsigned int width) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  values[index] = __float2bfloat16_rn(
      __bfloat162float(values[index]) + __bfloat162float(bias[index % width]));
}

extern "C" __global__ void sglang_neox_rope_bf16(
    __nv_bfloat16* values,
    const unsigned long long* positions,
    unsigned long long rows,
    unsigned int head_count,
    unsigned int head_dim,
    float theta) {
  const unsigned long long pair_index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned int half_dim = head_dim / 2;
  const unsigned long long pair_count = rows * head_count * half_dim;
  if (pair_index >= pair_count) {
    return;
  }
  const unsigned int index = pair_index % half_dim;
  const unsigned long long vector_index = pair_index / half_dim;
  const unsigned long long row = vector_index / head_count;
  const unsigned long long vector_offset = vector_index * head_dim;
  const float exponent = -static_cast<float>(2 * index) / static_cast<float>(head_dim);
  const float angle = static_cast<float>(positions[row]) * powf(theta, exponent);
  float sine;
  float cosine;
  sincosf(angle, &sine, &cosine);
  const float first = __bfloat162float(values[vector_offset + index]);
  const float second = __bfloat162float(values[vector_offset + half_dim + index]);
  values[vector_offset + index] = __float2bfloat16_rn(first * cosine - second * sine);
  values[vector_offset + half_dim + index] =
      __float2bfloat16_rn(second * cosine + first * sine);
}

extern "C" __global__ void sglang_add_bf16(
    const __nv_bfloat16* left,
    const __nv_bfloat16* right,
    __nv_bfloat16* output,
    unsigned long long element_count) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  output[index] = __float2bfloat16_rn(
      __bfloat162float(left[index]) + __bfloat162float(right[index]));
}

extern "C" __global__ void sglang_weighted_accumulate_bf16(
    __nv_bfloat16* accumulator,
    const __nv_bfloat16* source,
    unsigned long long element_count,
    float weight) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  accumulator[index] = __float2bfloat16_rn(
      __bfloat162float(accumulator[index]) +
      weight * __bfloat162float(source[index]));
}

extern "C" __global__ void sglang_silu_mul_bf16(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    __nv_bfloat16* output,
    unsigned long long element_count) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  const float gate_value = __bfloat162float(gate[index]);
  const float silu = gate_value / (1.0f + expf(-gate_value));
  output[index] = __float2bfloat16_rn(silu * __bfloat162float(up[index]));
}

extern "C" __global__ void sglang_gather_rows_bf16(
    const __nv_bfloat16* input,
    const unsigned long long* row_indices,
    __nv_bfloat16* output,
    unsigned int* error_flag,
    unsigned long long input_rows,
    unsigned long long output_rows,
    unsigned int width) {
  const unsigned long long element =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long element_count = output_rows * width;
  if (element >= element_count) {
    return;
  }
  const unsigned long long output_row = element / width;
  const unsigned int column = element % width;
  const unsigned long long input_row = row_indices[output_row];
  if (input_row >= input_rows) {
    atomicOr(error_flag, 2u);
    output[element] = __float2bfloat16_rn(0.0f);
    return;
  }
  output[element] = input[input_row * width + column];
}
"#;

const CUDA_BLOCK_SIZE: u32 = 256;
const BF16_BYTES: usize = 2;
const MINIMUM_BF16_COMPUTE_CAPABILITY: CudaComputeCapability = CudaComputeCapability::new(8, 0);

#[derive(Debug)]
pub enum CudaBf16KernelError {
    Cuda(CudaError),
    Nvrtc(NvrtcError),
    UnsupportedComputeCapability {
        actual: CudaComputeCapability,
        minimum: CudaComputeCapability,
    },
    ZeroDimension(&'static str),
    InvalidEpsilon(f32),
    InvalidTheta(f32),
    InvalidWeight(f32),
    OddHeadDimension(usize),
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
    DeviceInputInvalid {
        flags: u32,
    },
}

impl fmt::Display for CudaBf16KernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(error) => write!(formatter, "CUDA BF16 kernel operation failed: {error}"),
            Self::Nvrtc(error) => write!(formatter, "CUDA BF16 kernel compilation failed: {error}"),
            Self::UnsupportedComputeCapability { actual, minimum } => write!(
                formatter,
                "CUDA BF16 dense kernels require compute capability {minimum} or newer, found {actual}"
            ),
            Self::ZeroDimension(dimension) => {
                write!(
                    formatter,
                    "CUDA BF16 kernel dimension {dimension} must be non-zero"
                )
            }
            Self::InvalidEpsilon(epsilon) => write!(
                formatter,
                "CUDA BF16 RMSNorm epsilon must be finite and non-negative, got {epsilon}"
            ),
            Self::InvalidTheta(theta) => write!(
                formatter,
                "CUDA BF16 RoPE theta must be finite and positive, got {theta}"
            ),
            Self::InvalidWeight(weight) => write!(
                formatter,
                "CUDA BF16 accumulation weight must be finite and non-negative, got {weight}"
            ),
            Self::OddHeadDimension(head_dim) => write!(
                formatter,
                "CUDA BF16 NeoX RoPE head dimension must be even, found {head_dim}"
            ),
            Self::ShapeOverflow => formatter.write_str("CUDA BF16 kernel tensor shape overflowed"),
            Self::DimensionTooLarge {
                dimension,
                value,
                maximum,
            } => write!(
                formatter,
                "CUDA BF16 kernel dimension {dimension}={value} exceeds maximum {maximum}"
            ),
            Self::DeviceMismatch {
                allocation,
                expected_ordinal,
                actual_ordinal,
            } => write!(
                formatter,
                "CUDA BF16 kernel allocation {allocation} is on device {actual_ordinal}, expected device {expected_ordinal}"
            ),
            Self::DeviceInputInvalid { flags } => write!(
                formatter,
                "CUDA BF16 kernel rejected device input with flags 0x{flags:x} (token=0x1, row=0x2)"
            ),
        }
    }
}

impl std::error::Error for CudaBf16KernelError {}

impl From<CudaError> for CudaBf16KernelError {
    fn from(value: CudaError) -> Self {
        Self::Cuda(value)
    }
}

impl From<NvrtcError> for CudaBf16KernelError {
    fn from(value: NvrtcError) -> Self {
        Self::Nvrtc(value)
    }
}

pub struct CudaBf16DenseKernels {
    context: CudaContext,
    _module: CudaModule,
    embedding_lookup: CudaFunction,
    rms_norm: CudaFunction,
    add_bias: CudaFunction,
    neox_rope: CudaFunction,
    add: CudaFunction,
    weighted_accumulate: CudaFunction,
    silu_mul: CudaFunction,
    gather_rows: CudaFunction,
    error_flag: CudaDeviceAllocation,
}

impl CudaBf16DenseKernels {
    pub fn compile(
        context: &CudaContext,
        compute_capability: CudaComputeCapability,
    ) -> Result<Self, CudaBf16KernelError> {
        if compute_capability < MINIMUM_BF16_COMPUTE_CAPABILITY {
            return Err(CudaBf16KernelError::UnsupportedComputeCapability {
                actual: compute_capability,
                minimum: MINIMUM_BF16_COMPUTE_CAPABILITY,
            });
        }
        let compiler = NvrtcCompiler::load()?;
        let ptx = compiler.compile_ptx(
            CUDA_BF16_DENSE_SOURCE,
            "sglang_bf16_dense_kernels.cu",
            compute_capability,
        )?;
        let module = context.load_module(&ptx)?;
        let mut error_flag = context.allocate(size_of::<u32>())?;
        error_flag.fill(0)?;
        Ok(Self {
            context: context.clone(),
            embedding_lookup: module.get_function("sglang_embedding_lookup_bf16")?,
            rms_norm: module.get_function("sglang_rms_norm_bf16")?,
            add_bias: module.get_function("sglang_add_bias_bf16")?,
            neox_rope: module.get_function("sglang_neox_rope_bf16")?,
            add: module.get_function("sglang_add_bf16")?,
            weighted_accumulate: module.get_function("sglang_weighted_accumulate_bf16")?,
            silu_mul: module.get_function("sglang_silu_mul_bf16")?,
            gather_rows: module.get_function("sglang_gather_rows_bf16")?,
            _module: module,
            error_flag,
        })
    }

    pub fn embedding_lookup(
        &mut self,
        token_ids: &CudaDeviceAllocation,
        table: &CudaDeviceAllocation,
        output: &mut CudaDeviceAllocation,
        row_count: usize,
        vocabulary_size: usize,
        width: usize,
    ) -> Result<(), CudaBf16KernelError> {
        validate_nonzero(&[
            ("row_count", row_count),
            ("vocabulary_size", vocabulary_size),
            ("width", width),
        ])?;
        self.validate_devices(&[
            ("token_ids", token_ids),
            ("table", table),
            ("output", output),
        ])?;
        let element_count = checked_product(row_count, width)?;
        let mut token_ids_ptr = token_ids.device_ptr_at(0, checked_bytes(row_count, 4)?)?;
        let mut table_ptr = table.device_ptr_at(
            0,
            checked_bytes(checked_product(vocabulary_size, width)?, BF16_BYTES)?,
        )?;
        let mut output_ptr = output.device_ptr_at(0, checked_bytes(element_count, BF16_BYTES)?)?;
        self.error_flag.fill(0)?;
        let mut error_ptr = self.error_flag.device_ptr_at(0, size_of::<u32>())?;
        let mut rows = dimension_u64("row_count", row_count)?;
        let mut vocabulary = dimension_u64("vocabulary_size", vocabulary_size)?;
        let mut width = dimension_u32("width", width)?;
        let mut arguments = [
            argument_pointer(&mut token_ids_ptr),
            argument_pointer(&mut table_ptr),
            argument_pointer(&mut output_ptr),
            argument_pointer(&mut error_ptr),
            argument_pointer(&mut rows),
            argument_pointer(&mut vocabulary),
            argument_pointer(&mut width),
        ];
        self.launch_elementwise(&self.embedding_lookup, element_count, &mut arguments)?;
        self.check_device_input()
    }

    pub fn rms_norm(
        &self,
        input: &CudaDeviceAllocation,
        weight: &CudaDeviceAllocation,
        output: &mut CudaDeviceAllocation,
        rows: usize,
        width: usize,
        epsilon: f32,
    ) -> Result<(), CudaBf16KernelError> {
        validate_nonzero(&[("rows", rows), ("width", width)])?;
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err(CudaBf16KernelError::InvalidEpsilon(epsilon));
        }
        self.validate_devices(&[("input", input), ("weight", weight), ("output", output)])?;
        let tensor_bytes = checked_bytes(checked_product(rows, width)?, BF16_BYTES)?;
        let mut input_ptr = input.device_ptr_at(0, tensor_bytes)?;
        let mut weight_ptr = weight.device_ptr_at(0, checked_bytes(width, BF16_BYTES)?)?;
        let mut output_ptr = output.device_ptr_at(0, tensor_bytes)?;
        let grid_x = dimension_u32("rows", rows)?;
        let mut rows = dimension_u64("rows", rows)?;
        let mut width = dimension_u32("width", width)?;
        let mut epsilon = epsilon;
        let mut arguments = [
            argument_pointer(&mut input_ptr),
            argument_pointer(&mut weight_ptr),
            argument_pointer(&mut output_ptr),
            argument_pointer(&mut rows),
            argument_pointer(&mut width),
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

    pub fn add_bias(
        &self,
        values: &mut CudaDeviceAllocation,
        bias: &CudaDeviceAllocation,
        rows: usize,
        width: usize,
    ) -> Result<(), CudaBf16KernelError> {
        validate_nonzero(&[("rows", rows), ("width", width)])?;
        self.validate_devices(&[("values", values), ("bias", bias)])?;
        let element_count = checked_product(rows, width)?;
        let mut values_ptr = values.device_ptr_at(0, checked_bytes(element_count, BF16_BYTES)?)?;
        let mut bias_ptr = bias.device_ptr_at(0, checked_bytes(width, BF16_BYTES)?)?;
        let mut element_count_u64 = dimension_u64("element_count", element_count)?;
        let mut width_u32 = dimension_u32("width", width)?;
        let mut arguments = [
            argument_pointer(&mut values_ptr),
            argument_pointer(&mut bias_ptr),
            argument_pointer(&mut element_count_u64),
            argument_pointer(&mut width_u32),
        ];
        self.launch_elementwise(&self.add_bias, element_count, &mut arguments)
    }

    pub fn neox_rope(
        &self,
        values: &mut CudaDeviceAllocation,
        positions: &CudaDeviceAllocation,
        rows: usize,
        head_count: usize,
        head_dim: usize,
        theta: f32,
    ) -> Result<(), CudaBf16KernelError> {
        validate_nonzero(&[
            ("rows", rows),
            ("head_count", head_count),
            ("head_dim", head_dim),
        ])?;
        if !head_dim.is_multiple_of(2) {
            return Err(CudaBf16KernelError::OddHeadDimension(head_dim));
        }
        if !theta.is_finite() || theta <= 0.0 {
            return Err(CudaBf16KernelError::InvalidTheta(theta));
        }
        self.validate_devices(&[("values", values), ("positions", positions)])?;
        let value_count = checked_product(checked_product(rows, head_count)?, head_dim)?;
        let pair_count = value_count / 2;
        let mut values_ptr = values.device_ptr_at(0, checked_bytes(value_count, BF16_BYTES)?)?;
        let mut positions_ptr = positions.device_ptr_at(0, checked_bytes(rows, 8)?)?;
        let mut rows_u64 = dimension_u64("rows", rows)?;
        let mut head_count_u32 = dimension_u32("head_count", head_count)?;
        let mut head_dim_u32 = dimension_u32("head_dim", head_dim)?;
        let mut theta = theta;
        let mut arguments = [
            argument_pointer(&mut values_ptr),
            argument_pointer(&mut positions_ptr),
            argument_pointer(&mut rows_u64),
            argument_pointer(&mut head_count_u32),
            argument_pointer(&mut head_dim_u32),
            argument_pointer(&mut theta),
        ];
        self.launch_elementwise(&self.neox_rope, pair_count, &mut arguments)
    }

    pub fn add(
        &self,
        left: &CudaDeviceAllocation,
        right: &CudaDeviceAllocation,
        output: &mut CudaDeviceAllocation,
        element_count: usize,
    ) -> Result<(), CudaBf16KernelError> {
        validate_nonzero(&[("element_count", element_count)])?;
        self.validate_devices(&[("left", left), ("right", right), ("output", output)])?;
        let tensor_bytes = checked_bytes(element_count, BF16_BYTES)?;
        let mut left_ptr = left.device_ptr_at(0, tensor_bytes)?;
        let mut right_ptr = right.device_ptr_at(0, tensor_bytes)?;
        let mut output_ptr = output.device_ptr_at(0, tensor_bytes)?;
        let mut element_count_u64 = dimension_u64("element_count", element_count)?;
        let mut arguments = [
            argument_pointer(&mut left_ptr),
            argument_pointer(&mut right_ptr),
            argument_pointer(&mut output_ptr),
            argument_pointer(&mut element_count_u64),
        ];
        self.launch_elementwise(&self.add, element_count, &mut arguments)
    }

    pub fn silu_mul(
        &self,
        gate: &CudaDeviceAllocation,
        up: &CudaDeviceAllocation,
        output: &mut CudaDeviceAllocation,
        element_count: usize,
    ) -> Result<(), CudaBf16KernelError> {
        validate_nonzero(&[("element_count", element_count)])?;
        self.validate_devices(&[("gate", gate), ("up", up), ("output", output)])?;
        let tensor_bytes = checked_bytes(element_count, BF16_BYTES)?;
        let mut gate_ptr = gate.device_ptr_at(0, tensor_bytes)?;
        let mut up_ptr = up.device_ptr_at(0, tensor_bytes)?;
        let mut output_ptr = output.device_ptr_at(0, tensor_bytes)?;
        let mut element_count_u64 = dimension_u64("element_count", element_count)?;
        let mut arguments = [
            argument_pointer(&mut gate_ptr),
            argument_pointer(&mut up_ptr),
            argument_pointer(&mut output_ptr),
            argument_pointer(&mut element_count_u64),
        ];
        self.launch_elementwise(&self.silu_mul, element_count, &mut arguments)
    }

    pub fn weighted_accumulate(
        &self,
        accumulator: &mut CudaDeviceAllocation,
        source: &CudaDeviceAllocation,
        element_count: usize,
        weight: f32,
    ) -> Result<(), CudaBf16KernelError> {
        validate_nonzero(&[("element_count", element_count)])?;
        if !weight.is_finite() || weight < 0.0 {
            return Err(CudaBf16KernelError::InvalidWeight(weight));
        }
        self.validate_devices(&[("accumulator", accumulator), ("source", source)])?;
        let tensor_bytes = checked_bytes(element_count, BF16_BYTES)?;
        let mut accumulator_ptr = accumulator.device_ptr_at(0, tensor_bytes)?;
        let mut source_ptr = source.device_ptr_at(0, tensor_bytes)?;
        let mut element_count_u64 = dimension_u64("element_count", element_count)?;
        let mut weight = weight;
        let mut arguments = [
            argument_pointer(&mut accumulator_ptr),
            argument_pointer(&mut source_ptr),
            argument_pointer(&mut element_count_u64),
            argument_pointer(&mut weight),
        ];
        self.launch_elementwise(&self.weighted_accumulate, element_count, &mut arguments)
    }

    pub fn gather_rows(
        &mut self,
        input: &CudaDeviceAllocation,
        row_indices: &CudaDeviceAllocation,
        output: &mut CudaDeviceAllocation,
        input_rows: usize,
        output_rows: usize,
        width: usize,
    ) -> Result<(), CudaBf16KernelError> {
        validate_nonzero(&[
            ("input_rows", input_rows),
            ("output_rows", output_rows),
            ("width", width),
        ])?;
        self.validate_devices(&[
            ("input", input),
            ("row_indices", row_indices),
            ("output", output),
        ])?;
        let input_elements = checked_product(input_rows, width)?;
        let output_elements = checked_product(output_rows, width)?;
        let mut input_ptr = input.device_ptr_at(0, checked_bytes(input_elements, BF16_BYTES)?)?;
        let mut indices_ptr = row_indices.device_ptr_at(0, checked_bytes(output_rows, 8)?)?;
        let mut output_ptr =
            output.device_ptr_at(0, checked_bytes(output_elements, BF16_BYTES)?)?;
        self.error_flag.fill(0)?;
        let mut error_ptr = self.error_flag.device_ptr_at(0, size_of::<u32>())?;
        let mut input_rows_u64 = dimension_u64("input_rows", input_rows)?;
        let mut output_rows_u64 = dimension_u64("output_rows", output_rows)?;
        let mut width_u32 = dimension_u32("width", width)?;
        let mut arguments = [
            argument_pointer(&mut input_ptr),
            argument_pointer(&mut indices_ptr),
            argument_pointer(&mut output_ptr),
            argument_pointer(&mut error_ptr),
            argument_pointer(&mut input_rows_u64),
            argument_pointer(&mut output_rows_u64),
            argument_pointer(&mut width_u32),
        ];
        self.launch_elementwise(&self.gather_rows, output_elements, &mut arguments)?;
        self.check_device_input()
    }

    fn launch_elementwise(
        &self,
        function: &CudaFunction,
        element_count: usize,
        arguments: &mut [*mut c_void],
    ) -> Result<(), CudaBf16KernelError> {
        let block_count = element_count
            .checked_add(CUDA_BLOCK_SIZE as usize - 1)
            .ok_or(CudaBf16KernelError::ShapeOverflow)?
            / CUDA_BLOCK_SIZE as usize;
        let grid_x = dimension_u32("elementwise grid blocks", block_count)?;
        unsafe {
            function.launch(
                CudaLaunchDimensions::new(grid_x, 1, 1),
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
    ) -> Result<(), CudaBf16KernelError> {
        let expected_ordinal = self.context.device_ordinal();
        for (name, allocation) in allocations {
            let actual_ordinal = allocation.device_ordinal();
            if actual_ordinal != expected_ordinal {
                return Err(CudaBf16KernelError::DeviceMismatch {
                    allocation: name,
                    expected_ordinal,
                    actual_ordinal,
                });
            }
        }
        Ok(())
    }

    fn check_device_input(&self) -> Result<(), CudaBf16KernelError> {
        let mut bytes = [0_u8; size_of::<u32>()];
        self.error_flag.copy_to_host(0, &mut bytes)?;
        let flags = u32::from_ne_bytes(bytes);
        if flags == 0 {
            Ok(())
        } else {
            Err(CudaBf16KernelError::DeviceInputInvalid { flags })
        }
    }
}

fn validate_nonzero(values: &[(&'static str, usize)]) -> Result<(), CudaBf16KernelError> {
    for (name, value) in values {
        if *value == 0 {
            return Err(CudaBf16KernelError::ZeroDimension(name));
        }
    }
    Ok(())
}

fn checked_product(left: usize, right: usize) -> Result<usize, CudaBf16KernelError> {
    left.checked_mul(right)
        .ok_or(CudaBf16KernelError::ShapeOverflow)
}

fn checked_bytes(element_count: usize, element_bytes: usize) -> Result<usize, CudaBf16KernelError> {
    checked_product(element_count, element_bytes)
}

fn dimension_u64(dimension: &'static str, value: usize) -> Result<u64, CudaBf16KernelError> {
    u64::try_from(value).map_err(|_| CudaBf16KernelError::DimensionTooLarge {
        dimension,
        value,
        maximum: u64::MAX as usize,
    })
}

fn dimension_u32(dimension: &'static str, value: usize) -> Result<u32, CudaBf16KernelError> {
    u32::try_from(value).map_err(|_| CudaBf16KernelError::DimensionTooLarge {
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
    fn bf16_dense_kernel_capability_floor_is_architecture_based() {
        assert_eq!(
            MINIMUM_BF16_COMPUTE_CAPABILITY,
            CudaComputeCapability::new(8, 0)
        );
        assert!(CudaComputeCapability::new(9, 0) >= MINIMUM_BF16_COMPUTE_CAPABILITY);
        assert!(CudaComputeCapability::new(10, 0) >= MINIMUM_BF16_COMPUTE_CAPABILITY);
    }

    #[test]
    fn embedded_source_contains_the_complete_dense_decoder_primitive_set() {
        for kernel in [
            "sglang_embedding_lookup_bf16",
            "sglang_rms_norm_bf16",
            "sglang_add_bias_bf16",
            "sglang_neox_rope_bf16",
            "sglang_add_bf16",
            "sglang_weighted_accumulate_bf16",
            "sglang_silu_mul_bf16",
            "sglang_gather_rows_bf16",
        ] {
            assert!(CUDA_BF16_DENSE_SOURCE.contains(kernel));
        }
    }
}
