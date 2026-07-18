use std::ffi::c_void;
use std::fmt;

use crate::cuda::{
    CudaComputeCapability, CudaContext, CudaDeviceAllocation, CudaError, CudaFunction,
    CudaLaunchDimensions, CudaModule,
};
use crate::nvrtc::{NvrtcCompiler, NvrtcError};

const CUDA_LINEAR_ATTENTION_SOURCE: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void sglang_validate_linear_state_indices(
    const unsigned int* state_indices,
    unsigned int* error_flag,
    unsigned long long batch_size,
    unsigned int state_slot_count) {
  const unsigned long long row =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= batch_size) {
    return;
  }
  const unsigned int slot = state_indices[row];
  if (slot >= state_slot_count) {
    atomicOr(error_flag, 1u);
    return;
  }
  for (unsigned long long previous = 0; previous < row; ++previous) {
    if (state_indices[previous] == slot) {
      atomicOr(error_flag, 2u);
      return;
    }
  }
}

extern "C" __global__ void sglang_causal_conv1d_update_bf16(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* state,
    const unsigned int* state_indices,
    __nv_bfloat16* output,
    unsigned long long batch_size,
    unsigned int channels,
    unsigned int kernel_size) {
  const unsigned long long element =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long element_count = batch_size * channels;
  if (element >= element_count) {
    return;
  }

  const unsigned long long batch = element / channels;
  const unsigned int channel = element % channels;
  const unsigned int history = kernel_size - 1;
  const unsigned long long slot = state_indices[batch];
  const unsigned long long state_base = slot * history * channels + channel;
  const unsigned long long weight_base =
      static_cast<unsigned long long>(channel) * kernel_size;

  float value = __bfloat162float(weight[weight_base + history]) *
      __bfloat162float(input[element]);
  for (unsigned int step = 0; step < history; ++step) {
    value += __bfloat162float(weight[weight_base + step]) *
        __bfloat162float(state[state_base +
            static_cast<unsigned long long>(step) * channels]);
  }
  output[element] = __float2bfloat16_rn(value);

  for (unsigned int step = 0; step + 1 < history; ++step) {
    state[state_base + static_cast<unsigned long long>(step) * channels] =
        state[state_base + static_cast<unsigned long long>(step + 1) * channels];
  }
  state[state_base + static_cast<unsigned long long>(history - 1) * channels] =
      input[element];
}

extern "C" __global__ void sglang_key_gated_delta_decode_bf16_f32_state(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    const float* decay,
    const float* beta,
    float* state,
    const unsigned int* state_indices,
    __nv_bfloat16* output,
    unsigned long long batch_size,
    unsigned int head_count,
    unsigned int key_head_dim,
    unsigned int value_head_dim) {
  const unsigned long long row = blockIdx.x;
  const unsigned long long batch = row / head_count;
  const unsigned int head = row % head_count;
  if (batch >= batch_size || head >= head_count) {
    return;
  }

  const unsigned long long slot = state_indices[batch];
  const unsigned long long key_vector =
      (batch * head_count + head) * key_head_dim;
  const unsigned long long value_vector =
      (batch * head_count + head) * value_head_dim;
  const unsigned long long state_head =
      (slot * head_count + head) * key_head_dim * value_head_dim;
  const float beta_value = beta[batch * head_count + head];

  for (unsigned int value_index = threadIdx.x;
       value_index < value_head_dim;
       value_index += blockDim.x) {
    float previous = 0.0f;
    for (unsigned int key_index = 0; key_index < key_head_dim; ++key_index) {
      const unsigned long long state_index = state_head +
          static_cast<unsigned long long>(value_index) * key_head_dim + key_index;
      const float decayed = state[state_index] * decay[key_vector + key_index];
      state[state_index] = decayed;
      previous += __bfloat162float(key[key_vector + key_index]) * decayed;
    }

    const float delta =
        (__bfloat162float(value[value_vector + value_index]) - previous) * beta_value;
    float result = 0.0f;
    for (unsigned int key_index = 0; key_index < key_head_dim; ++key_index) {
      const unsigned long long state_index = state_head +
          static_cast<unsigned long long>(value_index) * key_head_dim + key_index;
      const float updated = state[state_index] +
          __bfloat162float(key[key_vector + key_index]) * delta;
      state[state_index] = updated;
      result += __bfloat162float(query[key_vector + key_index]) * updated;
    }
    output[value_vector + value_index] = __float2bfloat16_rn(result);
  }
}
"#;

const CUDA_BLOCK_SIZE: u32 = 256;
const BF16_BYTES: usize = 2;
const F32_BYTES: usize = 4;
const U32_BYTES: usize = 4;
const INVALID_STATE_INDEX: u32 = 1;
const DUPLICATE_STATE_INDEX: u32 = 2;
const MINIMUM_BF16_COMPUTE_CAPABILITY: CudaComputeCapability = CudaComputeCapability::new(8, 0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Causal-convolution decode geometry with KDA state laid out as
/// `[state_slot, kernel_size - 1, channels]`.
pub struct CudaCausalConv1dShape {
    pub batch_size: usize,
    pub state_slot_count: usize,
    pub channels: usize,
    pub kernel_size: usize,
}

impl CudaCausalConv1dShape {
    fn validate(self) -> Result<(), CudaLinearAttentionError> {
        validate_nonzero(&[
            ("batch_size", self.batch_size),
            ("state_slot_count", self.state_slot_count),
            ("channels", self.channels),
        ])?;
        if self.kernel_size < 2 {
            return Err(CudaLinearAttentionError::InvalidKernelSize(
                self.kernel_size,
            ));
        }
        dimension_u32("state_slot_count", self.state_slot_count)?;
        dimension_u32("channels", self.channels)?;
        dimension_u32("kernel_size", self.kernel_size)?;
        self.input_elements()?;
        self.weight_elements()?;
        self.state_elements()?;
        Ok(())
    }

    fn input_elements(self) -> Result<usize, CudaLinearAttentionError> {
        checked_product(self.batch_size, self.channels)
    }

    fn weight_elements(self) -> Result<usize, CudaLinearAttentionError> {
        checked_product(self.channels, self.kernel_size)
    }

    fn state_elements(self) -> Result<usize, CudaLinearAttentionError> {
        checked_product(
            checked_product(self.state_slot_count, self.kernel_size - 1)?,
            self.channels,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// KDA decode geometry with temporal state laid out as
/// `[state_slot, head, value_head_dim, key_head_dim]`.
pub struct CudaKeyGatedDeltaShape {
    pub batch_size: usize,
    pub state_slot_count: usize,
    pub head_count: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
}

impl CudaKeyGatedDeltaShape {
    fn validate(self) -> Result<(), CudaLinearAttentionError> {
        validate_nonzero(&[
            ("batch_size", self.batch_size),
            ("state_slot_count", self.state_slot_count),
            ("head_count", self.head_count),
            ("key_head_dim", self.key_head_dim),
            ("value_head_dim", self.value_head_dim),
        ])?;
        dimension_u32("state_slot_count", self.state_slot_count)?;
        dimension_u32("head_count", self.head_count)?;
        dimension_u32("key_head_dim", self.key_head_dim)?;
        dimension_u32("value_head_dim", self.value_head_dim)?;
        self.key_elements()?;
        self.value_elements()?;
        self.state_elements()?;
        Ok(())
    }

    fn key_elements(self) -> Result<usize, CudaLinearAttentionError> {
        checked_product(
            checked_product(self.batch_size, self.head_count)?,
            self.key_head_dim,
        )
    }

    fn value_elements(self) -> Result<usize, CudaLinearAttentionError> {
        checked_product(
            checked_product(self.batch_size, self.head_count)?,
            self.value_head_dim,
        )
    }

    fn state_elements(self) -> Result<usize, CudaLinearAttentionError> {
        checked_product(
            checked_product(
                checked_product(self.state_slot_count, self.head_count)?,
                self.key_head_dim,
            )?,
            self.value_head_dim,
        )
    }
}

pub struct CudaCausalConv1dLaunch<'a> {
    pub input: &'a CudaDeviceAllocation,
    pub input_offset: usize,
    pub weight: &'a CudaDeviceAllocation,
    pub weight_offset: usize,
    pub state: &'a mut CudaDeviceAllocation,
    pub state_offset: usize,
    pub state_indices: &'a CudaDeviceAllocation,
    pub state_indices_offset: usize,
    pub output: &'a mut CudaDeviceAllocation,
    pub output_offset: usize,
    pub shape: CudaCausalConv1dShape,
}

pub struct CudaKeyGatedDeltaLaunch<'a> {
    pub query: &'a CudaDeviceAllocation,
    pub query_offset: usize,
    pub key: &'a CudaDeviceAllocation,
    pub key_offset: usize,
    pub value: &'a CudaDeviceAllocation,
    pub value_offset: usize,
    /// Non-negative per-key multiplicative decay after gate activation.
    pub decay: &'a CudaDeviceAllocation,
    pub decay_offset: usize,
    /// Per-head update rate after sigmoid activation.
    pub beta: &'a CudaDeviceAllocation,
    pub beta_offset: usize,
    /// Persistent FP32 temporal state in the layout declared by `shape`.
    pub state: &'a mut CudaDeviceAllocation,
    pub state_offset: usize,
    pub state_indices: &'a CudaDeviceAllocation,
    pub state_indices_offset: usize,
    pub output: &'a mut CudaDeviceAllocation,
    pub output_offset: usize,
    pub shape: CudaKeyGatedDeltaShape,
}

#[derive(Debug)]
pub enum CudaLinearAttentionError {
    Cuda(CudaError),
    Nvrtc(NvrtcError),
    UnsupportedComputeCapability {
        actual: CudaComputeCapability,
        minimum: CudaComputeCapability,
    },
    ZeroDimension(&'static str),
    InvalidKernelSize(usize),
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
    StateIndexOutOfRange,
    DuplicateStateIndex,
}

impl fmt::Display for CudaLinearAttentionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(error) => {
                write!(formatter, "CUDA linear-attention operation failed: {error}")
            }
            Self::Nvrtc(error) => {
                write!(
                    formatter,
                    "CUDA linear-attention kernel compilation failed: {error}"
                )
            }
            Self::UnsupportedComputeCapability { actual, minimum } => write!(
                formatter,
                "CUDA BF16 linear attention requires compute capability {minimum} or newer; device reports {actual}"
            ),
            Self::ZeroDimension(dimension) => {
                write!(
                    formatter,
                    "CUDA linear-attention dimension {dimension} must be non-zero"
                )
            }
            Self::InvalidKernelSize(kernel_size) => write!(
                formatter,
                "CUDA causal convolution kernel_size must be at least 2, got {kernel_size}"
            ),
            Self::ShapeOverflow => {
                formatter.write_str("CUDA linear-attention tensor shape overflowed")
            }
            Self::DimensionTooLarge {
                dimension,
                value,
                maximum,
            } => write!(
                formatter,
                "CUDA linear-attention dimension {dimension}={value} exceeds maximum {maximum}"
            ),
            Self::DeviceMismatch {
                allocation,
                expected_ordinal,
                actual_ordinal,
            } => write!(
                formatter,
                "CUDA linear-attention allocation {allocation} is on device {actual_ordinal}, expected device {expected_ordinal}"
            ),
            Self::MisalignedOffset {
                allocation,
                offset,
                alignment,
            } => write!(
                formatter,
                "CUDA linear-attention allocation {allocation} offset {offset} is not aligned to {alignment} bytes"
            ),
            Self::StateIndexOutOfRange => formatter.write_str(
                "CUDA linear-attention state index exceeds the allocated state slot count",
            ),
            Self::DuplicateStateIndex => formatter
                .write_str("CUDA linear-attention batch contains duplicate writable state indices"),
        }
    }
}

impl std::error::Error for CudaLinearAttentionError {}

impl From<CudaError> for CudaLinearAttentionError {
    fn from(value: CudaError) -> Self {
        Self::Cuda(value)
    }
}

impl From<NvrtcError> for CudaLinearAttentionError {
    fn from(value: NvrtcError) -> Self {
        Self::Nvrtc(value)
    }
}

pub struct CudaBf16LinearAttentionKernels {
    context: CudaContext,
    _module: CudaModule,
    validate_state_indices: CudaFunction,
    causal_conv1d_update: CudaFunction,
    key_gated_delta_decode: CudaFunction,
    error_flag: CudaDeviceAllocation,
}

impl CudaBf16LinearAttentionKernels {
    pub fn compile(
        context: &CudaContext,
        compute_capability: CudaComputeCapability,
    ) -> Result<Self, CudaLinearAttentionError> {
        if compute_capability < MINIMUM_BF16_COMPUTE_CAPABILITY {
            return Err(CudaLinearAttentionError::UnsupportedComputeCapability {
                actual: compute_capability,
                minimum: MINIMUM_BF16_COMPUTE_CAPABILITY,
            });
        }
        let compiler = NvrtcCompiler::load()?;
        let ptx = compiler.compile_ptx(
            CUDA_LINEAR_ATTENTION_SOURCE,
            "sglang_linear_attention_kernels.cu",
            compute_capability,
        )?;
        let module = context.load_module(&ptx)?;
        let mut error_flag = context.allocate(size_of::<u32>())?;
        error_flag.fill(0)?;
        Ok(Self {
            context: context.clone(),
            validate_state_indices: module.get_function("sglang_validate_linear_state_indices")?,
            causal_conv1d_update: module.get_function("sglang_causal_conv1d_update_bf16")?,
            key_gated_delta_decode: module
                .get_function("sglang_key_gated_delta_decode_bf16_f32_state")?,
            _module: module,
            error_flag,
        })
    }

    pub fn causal_conv1d_update(
        &mut self,
        launch: CudaCausalConv1dLaunch<'_>,
    ) -> Result<(), CudaLinearAttentionError> {
        let CudaCausalConv1dLaunch {
            input,
            input_offset,
            weight,
            weight_offset,
            state,
            state_offset,
            state_indices,
            state_indices_offset,
            output,
            output_offset,
            shape,
        } = launch;
        shape.validate()?;
        self.validate_devices(&[
            ("input", input),
            ("weight", weight),
            ("state", state),
            ("state_indices", state_indices),
            ("output", output),
        ])?;
        validate_alignment(&[
            ("input", input_offset, BF16_BYTES),
            ("weight", weight_offset, BF16_BYTES),
            ("state", state_offset, BF16_BYTES),
            ("state_indices", state_indices_offset, U32_BYTES),
            ("output", output_offset, BF16_BYTES),
        ])?;
        self.validate_state_indices(
            state_indices,
            state_indices_offset,
            shape.batch_size,
            shape.state_slot_count,
        )?;

        let input_bytes = checked_bytes(shape.input_elements()?, BF16_BYTES)?;
        let mut input_ptr = input.device_ptr_at(input_offset, input_bytes)?;
        let mut weight_ptr = weight.device_ptr_at(
            weight_offset,
            checked_bytes(shape.weight_elements()?, BF16_BYTES)?,
        )?;
        let mut state_ptr = state.device_ptr_at(
            state_offset,
            checked_bytes(shape.state_elements()?, BF16_BYTES)?,
        )?;
        let mut state_indices_ptr = state_indices.device_ptr_at(
            state_indices_offset,
            checked_bytes(shape.batch_size, U32_BYTES)?,
        )?;
        let mut output_ptr = output.device_ptr_at(output_offset, input_bytes)?;
        let mut batch_size = dimension_u64("batch_size", shape.batch_size)?;
        let mut channels = dimension_u32("channels", shape.channels)?;
        let mut kernel_size = dimension_u32("kernel_size", shape.kernel_size)?;
        let mut arguments = [
            argument_pointer(&mut input_ptr),
            argument_pointer(&mut weight_ptr),
            argument_pointer(&mut state_ptr),
            argument_pointer(&mut state_indices_ptr),
            argument_pointer(&mut output_ptr),
            argument_pointer(&mut batch_size),
            argument_pointer(&mut channels),
            argument_pointer(&mut kernel_size),
        ];
        self.launch_elementwise(
            &self.causal_conv1d_update,
            shape.input_elements()?,
            &mut arguments,
        )
    }

    pub fn key_gated_delta_decode(
        &mut self,
        launch: CudaKeyGatedDeltaLaunch<'_>,
    ) -> Result<(), CudaLinearAttentionError> {
        let CudaKeyGatedDeltaLaunch {
            query,
            query_offset,
            key,
            key_offset,
            value,
            value_offset,
            decay,
            decay_offset,
            beta,
            beta_offset,
            state,
            state_offset,
            state_indices,
            state_indices_offset,
            output,
            output_offset,
            shape,
        } = launch;
        shape.validate()?;
        self.validate_devices(&[
            ("query", query),
            ("key", key),
            ("value", value),
            ("decay", decay),
            ("beta", beta),
            ("state", state),
            ("state_indices", state_indices),
            ("output", output),
        ])?;
        validate_alignment(&[
            ("query", query_offset, BF16_BYTES),
            ("key", key_offset, BF16_BYTES),
            ("value", value_offset, BF16_BYTES),
            ("decay", decay_offset, F32_BYTES),
            ("beta", beta_offset, F32_BYTES),
            ("state", state_offset, F32_BYTES),
            ("state_indices", state_indices_offset, U32_BYTES),
            ("output", output_offset, BF16_BYTES),
        ])?;
        self.validate_state_indices(
            state_indices,
            state_indices_offset,
            shape.batch_size,
            shape.state_slot_count,
        )?;

        let key_elements = shape.key_elements()?;
        let value_elements = shape.value_elements()?;
        let mut query_ptr =
            query.device_ptr_at(query_offset, checked_bytes(key_elements, BF16_BYTES)?)?;
        let mut key_ptr =
            key.device_ptr_at(key_offset, checked_bytes(key_elements, BF16_BYTES)?)?;
        let mut value_ptr =
            value.device_ptr_at(value_offset, checked_bytes(value_elements, BF16_BYTES)?)?;
        let mut decay_ptr =
            decay.device_ptr_at(decay_offset, checked_bytes(key_elements, F32_BYTES)?)?;
        let beta_elements = checked_product(shape.batch_size, shape.head_count)?;
        let mut beta_ptr =
            beta.device_ptr_at(beta_offset, checked_bytes(beta_elements, F32_BYTES)?)?;
        let mut state_ptr = state.device_ptr_at(
            state_offset,
            checked_bytes(shape.state_elements()?, F32_BYTES)?,
        )?;
        let mut state_indices_ptr = state_indices.device_ptr_at(
            state_indices_offset,
            checked_bytes(shape.batch_size, U32_BYTES)?,
        )?;
        let mut output_ptr =
            output.device_ptr_at(output_offset, checked_bytes(value_elements, BF16_BYTES)?)?;
        let mut batch_size = dimension_u64("batch_size", shape.batch_size)?;
        let mut head_count = dimension_u32("head_count", shape.head_count)?;
        let mut key_head_dim = dimension_u32("key_head_dim", shape.key_head_dim)?;
        let mut value_head_dim = dimension_u32("value_head_dim", shape.value_head_dim)?;
        let mut arguments = [
            argument_pointer(&mut query_ptr),
            argument_pointer(&mut key_ptr),
            argument_pointer(&mut value_ptr),
            argument_pointer(&mut decay_ptr),
            argument_pointer(&mut beta_ptr),
            argument_pointer(&mut state_ptr),
            argument_pointer(&mut state_indices_ptr),
            argument_pointer(&mut output_ptr),
            argument_pointer(&mut batch_size),
            argument_pointer(&mut head_count),
            argument_pointer(&mut key_head_dim),
            argument_pointer(&mut value_head_dim),
        ];
        let row_count = checked_product(shape.batch_size, shape.head_count)?;
        unsafe {
            self.key_gated_delta_decode.launch(
                CudaLaunchDimensions::new(dimension_u32("batch head rows", row_count)?, 1, 1),
                CudaLaunchDimensions::new(CUDA_BLOCK_SIZE, 1, 1),
                0,
                &mut arguments,
            )?;
        }
        self.context.synchronize()?;
        Ok(())
    }

    fn validate_state_indices(
        &mut self,
        state_indices: &CudaDeviceAllocation,
        state_indices_offset: usize,
        batch_size: usize,
        state_slot_count: usize,
    ) -> Result<(), CudaLinearAttentionError> {
        self.error_flag.fill(0)?;
        let mut state_indices_ptr = state_indices
            .device_ptr_at(state_indices_offset, checked_bytes(batch_size, U32_BYTES)?)?;
        let mut error_flag_ptr = self.error_flag.device_ptr_at(0, size_of::<u32>())?;
        let mut batch_size_u64 = dimension_u64("batch_size", batch_size)?;
        let mut state_slot_count_u32 = dimension_u32("state_slot_count", state_slot_count)?;
        let mut arguments = [
            argument_pointer(&mut state_indices_ptr),
            argument_pointer(&mut error_flag_ptr),
            argument_pointer(&mut batch_size_u64),
            argument_pointer(&mut state_slot_count_u32),
        ];
        self.launch_elementwise(&self.validate_state_indices, batch_size, &mut arguments)?;
        let mut flag = [0_u8; size_of::<u32>()];
        self.error_flag.copy_to_host(0, &mut flag)?;
        let flag = u32::from_ne_bytes(flag);
        if flag & INVALID_STATE_INDEX != 0 {
            return Err(CudaLinearAttentionError::StateIndexOutOfRange);
        }
        if flag & DUPLICATE_STATE_INDEX != 0 {
            return Err(CudaLinearAttentionError::DuplicateStateIndex);
        }
        Ok(())
    }

    fn launch_elementwise(
        &self,
        function: &CudaFunction,
        element_count: usize,
        arguments: &mut [*mut c_void],
    ) -> Result<(), CudaLinearAttentionError> {
        let block_count = element_count
            .checked_add(CUDA_BLOCK_SIZE as usize - 1)
            .ok_or(CudaLinearAttentionError::ShapeOverflow)?
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
    ) -> Result<(), CudaLinearAttentionError> {
        let expected_ordinal = self.context.device_ordinal();
        for (allocation, buffer) in allocations {
            let actual_ordinal = buffer.device_ordinal();
            if actual_ordinal != expected_ordinal {
                return Err(CudaLinearAttentionError::DeviceMismatch {
                    allocation,
                    expected_ordinal,
                    actual_ordinal,
                });
            }
        }
        Ok(())
    }
}

fn validate_nonzero(dimensions: &[(&'static str, usize)]) -> Result<(), CudaLinearAttentionError> {
    if let Some((dimension, _)) = dimensions.iter().find(|(_, value)| *value == 0) {
        return Err(CudaLinearAttentionError::ZeroDimension(dimension));
    }
    Ok(())
}

fn validate_alignment(
    allocations: &[(&'static str, usize, usize)],
) -> Result<(), CudaLinearAttentionError> {
    if let Some((allocation, offset, alignment)) = allocations
        .iter()
        .find(|(_, offset, alignment)| !offset.is_multiple_of(*alignment))
    {
        return Err(CudaLinearAttentionError::MisalignedOffset {
            allocation,
            offset: *offset,
            alignment: *alignment,
        });
    }
    Ok(())
}

fn checked_product(left: usize, right: usize) -> Result<usize, CudaLinearAttentionError> {
    left.checked_mul(right)
        .ok_or(CudaLinearAttentionError::ShapeOverflow)
}

fn checked_bytes(
    element_count: usize,
    element_size: usize,
) -> Result<usize, CudaLinearAttentionError> {
    checked_product(element_count, element_size)
}

fn dimension_u32(dimension: &'static str, value: usize) -> Result<u32, CudaLinearAttentionError> {
    u32::try_from(value).map_err(|_| CudaLinearAttentionError::DimensionTooLarge {
        dimension,
        value,
        maximum: u32::MAX as usize,
    })
}

fn dimension_u64(dimension: &'static str, value: usize) -> Result<u64, CudaLinearAttentionError> {
    u64::try_from(value).map_err(|_| CudaLinearAttentionError::DimensionTooLarge {
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
    fn causal_conv_shape_rejects_missing_history_and_overflow() {
        let missing_history = CudaCausalConv1dShape {
            batch_size: 1,
            state_slot_count: 1,
            channels: 4,
            kernel_size: 1,
        };
        assert!(matches!(
            missing_history.validate(),
            Err(CudaLinearAttentionError::InvalidKernelSize(1))
        ));

        let overflow = CudaCausalConv1dShape {
            batch_size: usize::MAX,
            state_slot_count: 1,
            channels: 2,
            kernel_size: 2,
        };
        assert!(matches!(
            overflow.validate(),
            Err(CudaLinearAttentionError::ShapeOverflow)
        ));
    }

    #[test]
    fn key_gated_delta_shape_accounts_for_slot_owned_f32_state() {
        let shape = CudaKeyGatedDeltaShape {
            batch_size: 3,
            state_slot_count: 7,
            head_count: 2,
            key_head_dim: 4,
            value_head_dim: 8,
        };
        shape.validate().expect("shape should be valid");
        assert_eq!(shape.key_elements().expect("key elements"), 24);
        assert_eq!(shape.value_elements().expect("value elements"), 48);
        assert_eq!(shape.state_elements().expect("state elements"), 448);
    }
}
