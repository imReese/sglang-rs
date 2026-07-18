use std::ffi::c_void;
use std::fmt;

use crate::cuda::{
    CudaComputeCapability, CudaContext, CudaDeviceAllocation, CudaError, CudaFunction,
    CudaLaunchDimensions, CudaModule,
};
use crate::nvrtc::{NvrtcCompiler, NvrtcError};

const CUDA_MLA_SOURCE: &str = r#"
#include <cuda_bf16.h>
#include <math.h>

__device__ float sglang_rope_frequency(
    unsigned int pair_index,
    unsigned int rotary_dim,
    float theta) {
  return powf(theta, -2.0f * static_cast<float>(pair_index) /
                          static_cast<float>(rotary_dim));
}

extern "C" __global__ void sglang_mla_prepare_query_bf16(
    const __nv_bfloat16* query,
    const __nv_bfloat16* kv_b_weight,
    const unsigned long long* positions,
    __nv_bfloat16* prepared_query,
    unsigned long long row_count,
    unsigned int head_count,
    unsigned int kv_lora_rank,
    unsigned int qk_nope_head_dim,
    unsigned int qk_rope_head_dim,
    unsigned int value_head_dim,
    float rope_theta,
    unsigned int skip_rope) {
  const unsigned long long row = blockIdx.x;
  const unsigned int head = blockIdx.y;
  if (row >= row_count || head >= head_count) {
    return;
  }
  const unsigned int query_head_dim = qk_nope_head_dim + qk_rope_head_dim;
  const unsigned int expanded_head_dim = qk_nope_head_dim + value_head_dim;
  const unsigned int prepared_head_dim = kv_lora_rank + qk_rope_head_dim;
  const unsigned long long query_offset =
      (row * head_count + head) * query_head_dim;
  const unsigned long long prepared_offset =
      (row * head_count + head) * prepared_head_dim;
  const unsigned long long weight_head_offset =
      static_cast<unsigned long long>(head) * expanded_head_dim * kv_lora_rank;

  for (unsigned int latent = threadIdx.x; latent < kv_lora_rank;
       latent += blockDim.x) {
    float sum = 0.0f;
    for (unsigned int dimension = 0; dimension < qk_nope_head_dim; ++dimension) {
      const float q = __bfloat162float(query[query_offset + dimension]);
      const float weight = __bfloat162float(
          kv_b_weight[weight_head_offset + dimension * kv_lora_rank + latent]);
      sum += q * weight;
    }
    prepared_query[prepared_offset + latent] = __float2bfloat16_rn(sum);
  }

  const unsigned int half_rope = qk_rope_head_dim / 2;
  for (unsigned int pair = threadIdx.x; pair < half_rope; pair += blockDim.x) {
    const float first = __bfloat162float(
        query[query_offset + qk_nope_head_dim + pair]);
    const float second = __bfloat162float(
        query[query_offset + qk_nope_head_dim + half_rope + pair]);
    float output_first = first;
    float output_second = second;
    if (!skip_rope) {
      const float angle = static_cast<float>(positions[row]) *
                          sglang_rope_frequency(pair, qk_rope_head_dim, rope_theta);
      const float cosine = cosf(angle);
      const float sine = sinf(angle);
      output_first = first * cosine - second * sine;
      output_second = second * cosine + first * sine;
    }
    prepared_query[prepared_offset + kv_lora_rank + pair] =
        __float2bfloat16_rn(output_first);
    prepared_query[prepared_offset + kv_lora_rank + half_rope + pair] =
        __float2bfloat16_rn(output_second);
  }
}

extern "C" __global__ void sglang_mla_prepare_cache_bf16(
    const __nv_bfloat16* compressed_kv,
    const __nv_bfloat16* kv_norm_weight,
    const unsigned long long* positions,
    __nv_bfloat16* cache_key,
    __nv_bfloat16* cache_value,
    unsigned long long row_count,
    unsigned int kv_lora_rank,
    unsigned int qk_rope_head_dim,
    float rms_norm_epsilon,
    float rope_theta,
    unsigned int skip_rope) {
  const unsigned long long row = blockIdx.x;
  if (row >= row_count) {
    return;
  }
  extern __shared__ float partial[];
  const unsigned int compressed_width = kv_lora_rank + qk_rope_head_dim;
  const unsigned long long input_offset = row * compressed_width;
  const unsigned long long value_offset = row * kv_lora_rank;
  float sum = 0.0f;
  for (unsigned int dimension = threadIdx.x; dimension < kv_lora_rank;
       dimension += blockDim.x) {
    const float value = __bfloat162float(compressed_kv[input_offset + dimension]);
    sum += value * value;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inverse_rms =
      rsqrtf(partial[0] / static_cast<float>(kv_lora_rank) + rms_norm_epsilon);
  for (unsigned int dimension = threadIdx.x; dimension < kv_lora_rank;
       dimension += blockDim.x) {
    const float normalized =
        __bfloat162float(compressed_kv[input_offset + dimension]) * inverse_rms *
        __bfloat162float(kv_norm_weight[dimension]);
    const __nv_bfloat16 value = __float2bfloat16_rn(normalized);
    cache_key[input_offset + dimension] = value;
    cache_value[value_offset + dimension] = value;
  }

  const unsigned int half_rope = qk_rope_head_dim / 2;
  for (unsigned int pair = threadIdx.x; pair < half_rope; pair += blockDim.x) {
    const float first = __bfloat162float(
        compressed_kv[input_offset + kv_lora_rank + pair]);
    const float second = __bfloat162float(
        compressed_kv[input_offset + kv_lora_rank + half_rope + pair]);
    float output_first = first;
    float output_second = second;
    if (!skip_rope) {
      const float angle = static_cast<float>(positions[row]) *
                          sglang_rope_frequency(pair, qk_rope_head_dim, rope_theta);
      const float cosine = cosf(angle);
      const float sine = sinf(angle);
      output_first = first * cosine - second * sine;
      output_second = second * cosine + first * sine;
    }
    cache_key[input_offset + kv_lora_rank + pair] =
        __float2bfloat16_rn(output_first);
    cache_key[input_offset + kv_lora_rank + half_rope + pair] =
        __float2bfloat16_rn(output_second);
  }
}

extern "C" __global__ void sglang_mla_expand_output_bf16(
    const __nv_bfloat16* latent_attention,
    const __nv_bfloat16* kv_b_weight,
    __nv_bfloat16* expanded_output,
    unsigned long long row_count,
    unsigned int head_count,
    unsigned int kv_lora_rank,
    unsigned int qk_nope_head_dim,
    unsigned int value_head_dim) {
  const unsigned long long row = blockIdx.x;
  const unsigned int head = blockIdx.y;
  if (row >= row_count || head >= head_count) {
    return;
  }
  const unsigned int expanded_head_dim = qk_nope_head_dim + value_head_dim;
  const unsigned long long input_offset =
      (row * head_count + head) * kv_lora_rank;
  const unsigned long long output_offset =
      (row * head_count + head) * value_head_dim;
  const unsigned long long weight_offset =
      (static_cast<unsigned long long>(head) * expanded_head_dim +
       qk_nope_head_dim) * kv_lora_rank;
  for (unsigned int dimension = threadIdx.x; dimension < value_head_dim;
       dimension += blockDim.x) {
    float sum = 0.0f;
    const unsigned long long weight_row =
        weight_offset + static_cast<unsigned long long>(dimension) * kv_lora_rank;
    for (unsigned int latent = 0; latent < kv_lora_rank; ++latent) {
      sum += __bfloat162float(latent_attention[input_offset + latent]) *
             __bfloat162float(kv_b_weight[weight_row + latent]);
    }
    expanded_output[output_offset + dimension] = __float2bfloat16_rn(sum);
  }
}
"#;

const CUDA_BLOCK_SIZE: u32 = 256;
const BF16_BYTES: usize = 2;
const U64_BYTES: usize = 8;
const MINIMUM_BF16_COMPUTE_CAPABILITY: CudaComputeCapability = CudaComputeCapability::new(8, 0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaBf16MlaShape {
    pub row_count: usize,
    pub head_count: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub value_head_dim: usize,
}

impl CudaBf16MlaShape {
    pub fn validate(self) -> Result<Self, CudaMlaKernelError> {
        for (dimension, value) in [
            ("row_count", self.row_count),
            ("head_count", self.head_count),
            ("kv_lora_rank", self.kv_lora_rank),
            ("qk_nope_head_dim", self.qk_nope_head_dim),
            ("qk_rope_head_dim", self.qk_rope_head_dim),
            ("value_head_dim", self.value_head_dim),
        ] {
            if value == 0 {
                return Err(CudaMlaKernelError::ZeroDimension(dimension));
            }
        }
        if !self.qk_rope_head_dim.is_multiple_of(2) {
            return Err(CudaMlaKernelError::OddRotaryDimension(
                self.qk_rope_head_dim,
            ));
        }
        self.query_head_dim()?;
        self.expanded_head_dim()?;
        self.prepared_head_dim()?;
        Ok(self)
    }

    pub fn query_head_dim(self) -> Result<usize, CudaMlaKernelError> {
        checked_add(self.qk_nope_head_dim, self.qk_rope_head_dim)
    }

    pub fn expanded_head_dim(self) -> Result<usize, CudaMlaKernelError> {
        checked_add(self.qk_nope_head_dim, self.value_head_dim)
    }

    pub fn prepared_head_dim(self) -> Result<usize, CudaMlaKernelError> {
        checked_add(self.kv_lora_rank, self.qk_rope_head_dim)
    }
}

#[derive(Debug)]
pub enum CudaMlaKernelError {
    Cuda(CudaError),
    Nvrtc(NvrtcError),
    UnsupportedComputeCapability {
        actual: CudaComputeCapability,
        minimum: CudaComputeCapability,
    },
    ZeroDimension(&'static str),
    OddRotaryDimension(usize),
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
}

impl fmt::Display for CudaMlaKernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(error) => write!(formatter, "CUDA MLA kernel failed: {error}"),
            Self::Nvrtc(error) => write!(formatter, "CUDA MLA compilation failed: {error}"),
            Self::UnsupportedComputeCapability { actual, minimum } => write!(
                formatter,
                "CUDA BF16 MLA kernels require compute capability {minimum} or newer; device reports {actual}"
            ),
            Self::ZeroDimension(dimension) => {
                write!(formatter, "CUDA MLA dimension {dimension} must be non-zero")
            }
            Self::OddRotaryDimension(dimension) => write!(
                formatter,
                "CUDA MLA rotary dimension must be even, got {dimension}"
            ),
            Self::InvalidScalar { name, value } => write!(
                formatter,
                "CUDA MLA scalar {name} must be finite and positive, got {value}"
            ),
            Self::ShapeOverflow => formatter.write_str("CUDA MLA shape overflowed"),
            Self::DimensionTooLarge {
                dimension,
                value,
                maximum,
            } => write!(
                formatter,
                "CUDA MLA dimension {dimension}={value} exceeds maximum {maximum}"
            ),
            Self::DeviceMismatch {
                allocation,
                expected_ordinal,
                actual_ordinal,
            } => write!(
                formatter,
                "CUDA MLA allocation {allocation} is on device {actual_ordinal}, expected device {expected_ordinal}"
            ),
        }
    }
}

impl std::error::Error for CudaMlaKernelError {}

impl From<CudaError> for CudaMlaKernelError {
    fn from(value: CudaError) -> Self {
        Self::Cuda(value)
    }
}

impl From<NvrtcError> for CudaMlaKernelError {
    fn from(value: NvrtcError) -> Self {
        Self::Nvrtc(value)
    }
}

pub struct CudaBf16MlaKernels {
    context: CudaContext,
    _module: CudaModule,
    prepare_query: CudaFunction,
    prepare_cache: CudaFunction,
    expand_output: CudaFunction,
}

pub struct CudaMlaPrepareQuery<'a> {
    pub query: &'a CudaDeviceAllocation,
    pub kv_b_weight: &'a CudaDeviceAllocation,
    pub positions: &'a CudaDeviceAllocation,
    pub output: &'a mut CudaDeviceAllocation,
    pub shape: CudaBf16MlaShape,
    pub rope_theta: f32,
    pub skip_rope: bool,
}

pub struct CudaMlaPrepareCache<'a> {
    pub compressed_kv: &'a CudaDeviceAllocation,
    pub kv_norm_weight: &'a CudaDeviceAllocation,
    pub positions: &'a CudaDeviceAllocation,
    pub cache_key: &'a mut CudaDeviceAllocation,
    pub cache_value: &'a mut CudaDeviceAllocation,
    pub shape: CudaBf16MlaShape,
    pub rms_norm_epsilon: f32,
    pub rope_theta: f32,
    pub skip_rope: bool,
}

pub struct CudaMlaExpandOutput<'a> {
    pub latent_attention: &'a CudaDeviceAllocation,
    pub kv_b_weight: &'a CudaDeviceAllocation,
    pub output: &'a mut CudaDeviceAllocation,
    pub shape: CudaBf16MlaShape,
}

impl CudaBf16MlaKernels {
    pub fn compile(
        context: &CudaContext,
        compute_capability: CudaComputeCapability,
    ) -> Result<Self, CudaMlaKernelError> {
        if compute_capability < MINIMUM_BF16_COMPUTE_CAPABILITY {
            return Err(CudaMlaKernelError::UnsupportedComputeCapability {
                actual: compute_capability,
                minimum: MINIMUM_BF16_COMPUTE_CAPABILITY,
            });
        }
        let ptx = NvrtcCompiler::load()?.compile_ptx(
            CUDA_MLA_SOURCE,
            "sglang_mla_kernels.cu",
            compute_capability,
        )?;
        let module = context.load_module(&ptx)?;
        Ok(Self {
            context: context.clone(),
            prepare_query: module.get_function("sglang_mla_prepare_query_bf16")?,
            prepare_cache: module.get_function("sglang_mla_prepare_cache_bf16")?,
            expand_output: module.get_function("sglang_mla_expand_output_bf16")?,
            _module: module,
        })
    }

    pub fn prepare_query(&self, launch: CudaMlaPrepareQuery<'_>) -> Result<(), CudaMlaKernelError> {
        let CudaMlaPrepareQuery {
            query,
            kv_b_weight,
            positions,
            output,
            shape,
            rope_theta,
            skip_rope,
        } = launch;
        let shape = shape.validate()?;
        validate_positive("rope_theta", rope_theta)?;
        self.validate_devices(&[
            ("query", query),
            ("kv_b_weight", kv_b_weight),
            ("positions", positions),
            ("output", output),
        ])?;
        let mut query_ptr = query.device_ptr_at(
            0,
            bf16_bytes(product3(
                shape.row_count,
                shape.head_count,
                shape.query_head_dim()?,
            )?)?,
        )?;
        let mut weight_ptr = kv_b_weight.device_ptr_at(
            0,
            bf16_bytes(product3(
                shape.head_count,
                shape.expanded_head_dim()?,
                shape.kv_lora_rank,
            )?)?,
        )?;
        let mut positions_ptr =
            positions.device_ptr_at(0, checked_product(shape.row_count, U64_BYTES)?)?;
        let mut output_ptr = output.device_ptr_at(
            0,
            bf16_bytes(product3(
                shape.row_count,
                shape.head_count,
                shape.prepared_head_dim()?,
            )?)?,
        )?;
        let mut values = KernelShapeValues::new(shape)?;
        let mut rope_theta_value = rope_theta;
        let mut skip_rope_value = u32::from(skip_rope);
        let mut arguments = [
            pointer(&mut query_ptr),
            pointer(&mut weight_ptr),
            pointer(&mut positions_ptr),
            pointer(&mut output_ptr),
            pointer(&mut values.row_count),
            pointer(&mut values.head_count),
            pointer(&mut values.kv_lora_rank),
            pointer(&mut values.qk_nope_head_dim),
            pointer(&mut values.qk_rope_head_dim),
            pointer(&mut values.value_head_dim),
            pointer(&mut rope_theta_value),
            pointer(&mut skip_rope_value),
        ];
        self.launch_2d(&self.prepare_query, shape, 0, &mut arguments)
    }

    pub fn prepare_cache(&self, launch: CudaMlaPrepareCache<'_>) -> Result<(), CudaMlaKernelError> {
        let CudaMlaPrepareCache {
            compressed_kv,
            kv_norm_weight,
            positions,
            cache_key,
            cache_value,
            shape,
            rms_norm_epsilon,
            rope_theta,
            skip_rope,
        } = launch;
        let shape = shape.validate()?;
        validate_positive("rms_norm_epsilon", rms_norm_epsilon)?;
        validate_positive("rope_theta", rope_theta)?;
        self.validate_devices(&[
            ("compressed_kv", compressed_kv),
            ("kv_norm_weight", kv_norm_weight),
            ("positions", positions),
            ("cache_key", cache_key),
            ("cache_value", cache_value),
        ])?;
        let compressed_elements = checked_product(shape.row_count, shape.prepared_head_dim()?)?;
        let latent_elements = checked_product(shape.row_count, shape.kv_lora_rank)?;
        let mut compressed_ptr =
            compressed_kv.device_ptr_at(0, bf16_bytes(compressed_elements)?)?;
        let mut norm_ptr = kv_norm_weight.device_ptr_at(0, bf16_bytes(shape.kv_lora_rank)?)?;
        let mut positions_ptr =
            positions.device_ptr_at(0, checked_product(shape.row_count, U64_BYTES)?)?;
        let mut cache_key_ptr = cache_key.device_ptr_at(0, bf16_bytes(compressed_elements)?)?;
        let mut cache_value_ptr = cache_value.device_ptr_at(0, bf16_bytes(latent_elements)?)?;
        let mut row_count = dimension_u64("row_count", shape.row_count)?;
        let mut kv_lora_rank = dimension_u32("kv_lora_rank", shape.kv_lora_rank)?;
        let mut qk_rope_head_dim = dimension_u32("qk_rope_head_dim", shape.qk_rope_head_dim)?;
        let mut epsilon_value = rms_norm_epsilon;
        let mut rope_theta_value = rope_theta;
        let mut skip_rope_value = u32::from(skip_rope);
        let mut arguments = [
            pointer(&mut compressed_ptr),
            pointer(&mut norm_ptr),
            pointer(&mut positions_ptr),
            pointer(&mut cache_key_ptr),
            pointer(&mut cache_value_ptr),
            pointer(&mut row_count),
            pointer(&mut kv_lora_rank),
            pointer(&mut qk_rope_head_dim),
            pointer(&mut epsilon_value),
            pointer(&mut rope_theta_value),
            pointer(&mut skip_rope_value),
        ];
        unsafe {
            self.prepare_cache.launch(
                CudaLaunchDimensions::new(dimension_u32("row_count", shape.row_count)?, 1, 1),
                CudaLaunchDimensions::new(CUDA_BLOCK_SIZE, 1, 1),
                CUDA_BLOCK_SIZE * size_of::<f32>() as u32,
                &mut arguments,
            )?;
        }
        self.context.synchronize()?;
        Ok(())
    }

    pub fn expand_output(&self, launch: CudaMlaExpandOutput<'_>) -> Result<(), CudaMlaKernelError> {
        let CudaMlaExpandOutput {
            latent_attention,
            kv_b_weight,
            output,
            shape,
        } = launch;
        let shape = shape.validate()?;
        self.validate_devices(&[
            ("latent_attention", latent_attention),
            ("kv_b_weight", kv_b_weight),
            ("output", output),
        ])?;
        let mut input_ptr = latent_attention.device_ptr_at(
            0,
            bf16_bytes(product3(
                shape.row_count,
                shape.head_count,
                shape.kv_lora_rank,
            )?)?,
        )?;
        let mut weight_ptr = kv_b_weight.device_ptr_at(
            0,
            bf16_bytes(product3(
                shape.head_count,
                shape.expanded_head_dim()?,
                shape.kv_lora_rank,
            )?)?,
        )?;
        let mut output_ptr = output.device_ptr_at(
            0,
            bf16_bytes(product3(
                shape.row_count,
                shape.head_count,
                shape.value_head_dim,
            )?)?,
        )?;
        let mut values = KernelShapeValues::new(shape)?;
        let mut arguments = [
            pointer(&mut input_ptr),
            pointer(&mut weight_ptr),
            pointer(&mut output_ptr),
            pointer(&mut values.row_count),
            pointer(&mut values.head_count),
            pointer(&mut values.kv_lora_rank),
            pointer(&mut values.qk_nope_head_dim),
            pointer(&mut values.value_head_dim),
        ];
        self.launch_2d(&self.expand_output, shape, 0, &mut arguments)
    }

    fn launch_2d(
        &self,
        function: &CudaFunction,
        shape: CudaBf16MlaShape,
        shared_memory_bytes: u32,
        arguments: &mut [*mut c_void],
    ) -> Result<(), CudaMlaKernelError> {
        unsafe {
            function.launch(
                CudaLaunchDimensions::new(
                    dimension_u32("row_count", shape.row_count)?,
                    dimension_u32("head_count", shape.head_count)?,
                    1,
                ),
                CudaLaunchDimensions::new(CUDA_BLOCK_SIZE, 1, 1),
                shared_memory_bytes,
                arguments,
            )?;
        }
        self.context.synchronize()?;
        Ok(())
    }

    fn validate_devices(
        &self,
        allocations: &[(&'static str, &CudaDeviceAllocation)],
    ) -> Result<(), CudaMlaKernelError> {
        let expected_ordinal = self.context.device_ordinal();
        for (allocation, buffer) in allocations {
            let actual_ordinal = buffer.device_ordinal();
            if actual_ordinal != expected_ordinal {
                return Err(CudaMlaKernelError::DeviceMismatch {
                    allocation,
                    expected_ordinal,
                    actual_ordinal,
                });
            }
        }
        Ok(())
    }
}

struct KernelShapeValues {
    row_count: u64,
    head_count: u32,
    kv_lora_rank: u32,
    qk_nope_head_dim: u32,
    qk_rope_head_dim: u32,
    value_head_dim: u32,
}

impl KernelShapeValues {
    fn new(shape: CudaBf16MlaShape) -> Result<Self, CudaMlaKernelError> {
        Ok(Self {
            row_count: dimension_u64("row_count", shape.row_count)?,
            head_count: dimension_u32("head_count", shape.head_count)?,
            kv_lora_rank: dimension_u32("kv_lora_rank", shape.kv_lora_rank)?,
            qk_nope_head_dim: dimension_u32("qk_nope_head_dim", shape.qk_nope_head_dim)?,
            qk_rope_head_dim: dimension_u32("qk_rope_head_dim", shape.qk_rope_head_dim)?,
            value_head_dim: dimension_u32("value_head_dim", shape.value_head_dim)?,
        })
    }
}

fn validate_positive(name: &'static str, value: f32) -> Result<(), CudaMlaKernelError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(CudaMlaKernelError::InvalidScalar { name, value });
    }
    Ok(())
}

fn checked_add(left: usize, right: usize) -> Result<usize, CudaMlaKernelError> {
    left.checked_add(right)
        .ok_or(CudaMlaKernelError::ShapeOverflow)
}

fn checked_product(left: usize, right: usize) -> Result<usize, CudaMlaKernelError> {
    left.checked_mul(right)
        .ok_or(CudaMlaKernelError::ShapeOverflow)
}

fn product3(first: usize, second: usize, third: usize) -> Result<usize, CudaMlaKernelError> {
    checked_product(checked_product(first, second)?, third)
}

fn bf16_bytes(element_count: usize) -> Result<usize, CudaMlaKernelError> {
    checked_product(element_count, BF16_BYTES)
}

fn dimension_u32(dimension: &'static str, value: usize) -> Result<u32, CudaMlaKernelError> {
    u32::try_from(value).map_err(|_| CudaMlaKernelError::DimensionTooLarge {
        dimension,
        value,
        maximum: u32::MAX as usize,
    })
}

fn dimension_u64(dimension: &'static str, value: usize) -> Result<u64, CudaMlaKernelError> {
    u64::try_from(value).map_err(|_| CudaMlaKernelError::DimensionTooLarge {
        dimension,
        value,
        maximum: u64::MAX as usize,
    })
}

fn pointer<T>(value: &mut T) -> *mut c_void {
    (value as *mut T).cast()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_shape() -> CudaBf16MlaShape {
        CudaBf16MlaShape {
            row_count: 2,
            head_count: 4,
            kv_lora_rank: 8,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 4,
            value_head_dim: 6,
        }
    }

    #[test]
    fn shape_derives_compressed_and_expanded_geometry() {
        let shape = test_shape().validate().expect("shape should be valid");
        assert_eq!(shape.query_head_dim().expect("query width"), 8);
        assert_eq!(shape.prepared_head_dim().expect("prepared width"), 12);
        assert_eq!(shape.expanded_head_dim().expect("expanded width"), 10);
    }

    #[test]
    fn shape_rejects_invalid_rotary_geometry() {
        assert!(matches!(
            CudaBf16MlaShape {
                qk_rope_head_dim: 3,
                ..test_shape()
            }
            .validate(),
            Err(CudaMlaKernelError::OddRotaryDimension(3))
        ));
    }

    #[test]
    fn embedded_source_contains_complete_absorbed_mla_path() {
        for entry_point in [
            "sglang_mla_prepare_query_bf16",
            "sglang_mla_prepare_cache_bf16",
            "sglang_mla_expand_output_bf16",
        ] {
            assert!(CUDA_MLA_SOURCE.contains(entry_point));
        }
        assert!(CUDA_MLA_SOURCE.contains("kv_lora_rank"));
        assert!(CUDA_MLA_SOURCE.contains("rsqrtf"));
    }
}
