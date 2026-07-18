use std::ffi::c_void;
use std::fmt;

use crate::cuda::{
    CudaComputeCapability, CudaContext, CudaDeviceAllocation, CudaError, CudaFunction,
    CudaLaunchDimensions, CudaModule,
};
use crate::nvrtc::{NvrtcCompiler, NvrtcError};

const CUDA_BF16_PAGED_ATTENTION_SOURCE: &str = r#"
#include <cuda_bf16.h>
#include <math.h>

extern "C" __global__ void sglang_paged_attention_bf16(
    const __nv_bfloat16* queries,
    const unsigned int* query_request_indices,
    const unsigned long long* query_sequence_lengths,
    const unsigned long long* request_slot_offsets,
    const unsigned long long* sequence_slots,
    const unsigned char* pool,
    __nv_bfloat16* output,
    unsigned int* error_flag,
    unsigned long long query_count,
    unsigned long long request_count,
    unsigned long long sequence_slot_count,
    unsigned long long slot_count,
    unsigned long long page_size,
    unsigned long long page_stride_bytes,
    unsigned long long key_in_page_offset,
    unsigned long long value_in_page_offset,
    unsigned long long key_row_bytes,
    unsigned long long value_row_bytes,
    unsigned int query_head_count,
    unsigned int kv_head_count,
    unsigned int query_key_head_dim,
    unsigned int value_head_dim,
    float scale) {
  const unsigned long long query_index = blockIdx.x;
  const unsigned int query_head = blockIdx.y;
  if (query_index >= query_count || query_head >= query_head_count) {
    return;
  }

  extern __shared__ float scratch[];
  float* partial = scratch;
  float* accumulator = scratch + blockDim.x;
  __shared__ unsigned int metadata_valid;
  __shared__ unsigned long long sequence_start;
  __shared__ unsigned long long sequence_length;
  __shared__ unsigned long long current_slot;
  __shared__ float running_max;
  __shared__ float denominator;
  __shared__ float accumulator_rescale;
  __shared__ float current_weight;

  if (threadIdx.x == 0) {
    metadata_valid = 1;
    const unsigned int request_index = query_request_indices[query_index];
    if (request_index >= request_count) {
      atomicOr(error_flag, 1u);
      metadata_valid = 0;
    } else {
      const unsigned long long start = request_slot_offsets[request_index];
      const unsigned long long end = request_slot_offsets[request_index + 1];
      const unsigned long long length = query_sequence_lengths[query_index];
      if (start > end || end > sequence_slot_count || length == 0 ||
          length > end - start) {
        atomicOr(error_flag, 2u);
        metadata_valid = 0;
      } else {
        sequence_start = start;
        sequence_length = length;
      }
    }
    running_max = -3.402823466e+38F;
    denominator = 0.0f;
  }
  for (unsigned int dimension = threadIdx.x; dimension < value_head_dim;
       dimension += blockDim.x) {
    accumulator[dimension] = 0.0f;
  }
  __syncthreads();
  if (!metadata_valid) {
    return;
  }

  const unsigned int query_heads_per_kv_head = query_head_count / kv_head_count;
  const unsigned int kv_head = query_head / query_heads_per_kv_head;
  const unsigned long long query_vector_offset =
      (query_index * query_head_count + query_head) * query_key_head_dim;
  const __nv_bfloat16* query = queries + query_vector_offset;

  for (unsigned long long sequence_index = 0; sequence_index < sequence_length;
       ++sequence_index) {
    if (threadIdx.x == 0) {
      current_slot = sequence_slots[sequence_start + sequence_index];
      if (current_slot >= slot_count) {
        atomicOr(error_flag, 4u);
        metadata_valid = 0;
      }
    }
    __syncthreads();
    if (!metadata_valid) {
      return;
    }

    const unsigned long long page = current_slot / page_size;
    const unsigned long long token = current_slot % page_size;
    const unsigned long long key_head_byte_offset =
        static_cast<unsigned long long>(kv_head) * query_key_head_dim *
        sizeof(__nv_bfloat16);
    const unsigned long long value_head_byte_offset =
        static_cast<unsigned long long>(kv_head) * value_head_dim *
        sizeof(__nv_bfloat16);
    const unsigned long long page_byte_offset = page * page_stride_bytes;
    const __nv_bfloat16* key = reinterpret_cast<const __nv_bfloat16*>(
        pool + page_byte_offset + key_in_page_offset + token * key_row_bytes +
        key_head_byte_offset);
    const __nv_bfloat16* value = reinterpret_cast<const __nv_bfloat16*>(
        pool + page_byte_offset + value_in_page_offset + token * value_row_bytes +
        value_head_byte_offset);

    float dot = 0.0f;
    for (unsigned int dimension = threadIdx.x; dimension < query_key_head_dim;
         dimension += blockDim.x) {
      dot += __bfloat162float(query[dimension]) * __bfloat162float(key[dimension]);
    }
    partial[threadIdx.x] = dot;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        partial[threadIdx.x] += partial[threadIdx.x + stride];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      const float score = partial[0] * scale;
      const float next_max = fmaxf(running_max, score);
      accumulator_rescale = expf(running_max - next_max);
      current_weight = expf(score - next_max);
      denominator = denominator * accumulator_rescale + current_weight;
      running_max = next_max;
    }
    __syncthreads();
    for (unsigned int dimension = threadIdx.x; dimension < value_head_dim;
         dimension += blockDim.x) {
      accumulator[dimension] = accumulator[dimension] * accumulator_rescale +
                               current_weight * __bfloat162float(value[dimension]);
    }
    __syncthreads();
  }

  const unsigned long long output_vector_offset =
      (query_index * query_head_count + query_head) * value_head_dim;
  for (unsigned int dimension = threadIdx.x; dimension < value_head_dim;
       dimension += blockDim.x) {
    output[output_vector_offset + dimension] =
        __float2bfloat16_rn(accumulator[dimension] / denominator);
  }
}
"#;

const CUDA_BLOCK_SIZE: u32 = 256;
const BF16_BYTES: usize = 2;
const MINIMUM_BF16_COMPUTE_CAPABILITY: CudaComputeCapability = CudaComputeCapability::new(8, 0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaBf16PagedAttentionLayout {
    page_size: usize,
    page_stride_bytes: usize,
    key_in_page_offset: usize,
    value_in_page_offset: usize,
    key_row_bytes: usize,
    value_row_bytes: usize,
}

impl CudaBf16PagedAttentionLayout {
    pub const fn new(
        page_size: usize,
        page_stride_bytes: usize,
        key_in_page_offset: usize,
        value_in_page_offset: usize,
        kv_row_bytes: usize,
    ) -> Self {
        Self::new_tensor_pair(
            page_size,
            page_stride_bytes,
            key_in_page_offset,
            value_in_page_offset,
            kv_row_bytes,
            kv_row_bytes,
        )
    }

    pub const fn new_tensor_pair(
        page_size: usize,
        page_stride_bytes: usize,
        key_in_page_offset: usize,
        value_in_page_offset: usize,
        key_row_bytes: usize,
        value_row_bytes: usize,
    ) -> Self {
        Self {
            page_size,
            page_stride_bytes,
            key_in_page_offset,
            value_in_page_offset,
            key_row_bytes,
            value_row_bytes,
        }
    }

    pub const fn page_size(self) -> usize {
        self.page_size
    }

    pub const fn page_stride_bytes(self) -> usize {
        self.page_stride_bytes
    }

    pub const fn key_in_page_offset(self) -> usize {
        self.key_in_page_offset
    }

    pub const fn value_in_page_offset(self) -> usize {
        self.value_in_page_offset
    }

    pub const fn key_row_bytes(self) -> usize {
        self.key_row_bytes
    }

    pub const fn value_row_bytes(self) -> usize {
        self.value_row_bytes
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaBf16PagedAttentionPlan {
    query_count: usize,
    request_count: usize,
    sequence_slot_count: usize,
    slot_count: usize,
    query_head_count: usize,
    kv_head_count: usize,
    query_key_head_dim: usize,
    value_head_dim: usize,
    scale: f32,
    layout: CudaBf16PagedAttentionLayout,
    query_required_bytes: usize,
    query_request_indices_required_bytes: usize,
    query_sequence_lengths_required_bytes: usize,
    request_slot_offsets_required_bytes: usize,
    sequence_slots_required_bytes: usize,
    pool_required_bytes: usize,
    output_required_bytes: usize,
    grid: CudaLaunchDimensions,
    shared_memory_bytes: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaBf16PagedAttentionPlanConfig {
    pub query_count: usize,
    pub request_count: usize,
    pub sequence_slot_count: usize,
    pub slot_count: usize,
    pub query_head_count: usize,
    pub kv_head_count: usize,
    pub query_key_head_dim: usize,
    pub value_head_dim: usize,
    pub scale: f32,
    pub layout: CudaBf16PagedAttentionLayout,
}

impl CudaBf16PagedAttentionPlan {
    pub fn new(
        config: CudaBf16PagedAttentionPlanConfig,
    ) -> Result<Self, CudaBf16PagedAttentionError> {
        let CudaBf16PagedAttentionPlanConfig {
            query_count,
            request_count,
            sequence_slot_count,
            slot_count,
            query_head_count,
            kv_head_count,
            query_key_head_dim,
            value_head_dim,
            scale,
            layout,
        } = config;
        for (dimension, value) in [
            ("query_count", query_count),
            ("request_count", request_count),
            ("sequence_slot_count", sequence_slot_count),
            ("slot_count", slot_count),
            ("query_head_count", query_head_count),
            ("kv_head_count", kv_head_count),
            ("query_key_head_dim", query_key_head_dim),
            ("value_head_dim", value_head_dim),
            ("page_size", layout.page_size),
            ("page_stride_bytes", layout.page_stride_bytes),
            ("key_row_bytes", layout.key_row_bytes),
            ("value_row_bytes", layout.value_row_bytes),
        ] {
            if value == 0 {
                return Err(CudaBf16PagedAttentionError::ZeroDimension(dimension));
            }
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(CudaBf16PagedAttentionError::InvalidScale(scale));
        }
        if !query_head_count.is_multiple_of(kv_head_count) {
            return Err(
                CudaBf16PagedAttentionError::QueryHeadsNotDivisibleByKvHeads {
                    query_head_count,
                    kv_head_count,
                },
            );
        }
        let expected_key_row_bytes = kv_head_count
            .checked_mul(query_key_head_dim)
            .and_then(|value| value.checked_mul(BF16_BYTES))
            .ok_or(CudaBf16PagedAttentionError::ShapeOverflow)?;
        if layout.key_row_bytes != expected_key_row_bytes {
            return Err(CudaBf16PagedAttentionError::KvRowByteSizeMismatch {
                expected: expected_key_row_bytes,
                actual: layout.key_row_bytes,
            });
        }
        let expected_value_row_bytes = kv_head_count
            .checked_mul(value_head_dim)
            .and_then(|value| value.checked_mul(BF16_BYTES))
            .ok_or(CudaBf16PagedAttentionError::ShapeOverflow)?;
        if layout.value_row_bytes != expected_value_row_bytes {
            return Err(CudaBf16PagedAttentionError::ValueRowByteSizeMismatch {
                expected: expected_value_row_bytes,
                actual: layout.value_row_bytes,
            });
        }

        let key_page_bytes = layout
            .page_size
            .checked_mul(layout.key_row_bytes)
            .ok_or(CudaBf16PagedAttentionError::ShapeOverflow)?;
        let value_page_bytes = layout
            .page_size
            .checked_mul(layout.value_row_bytes)
            .ok_or(CudaBf16PagedAttentionError::ShapeOverflow)?;
        let key_page_end = validate_tensor_page_region(
            "key",
            layout.key_in_page_offset,
            key_page_bytes,
            layout.page_stride_bytes,
        )?;
        let value_page_end = validate_tensor_page_region(
            "value",
            layout.value_in_page_offset,
            value_page_bytes,
            layout.page_stride_bytes,
        )?;
        if layout.key_in_page_offset < value_page_end && layout.value_in_page_offset < key_page_end
        {
            return Err(CudaBf16PagedAttentionError::TensorPageRegionsOverlap {
                key_offset: layout.key_in_page_offset,
                value_offset: layout.value_in_page_offset,
                key_page_bytes,
                value_page_bytes,
            });
        }

        let query_elements = query_count
            .checked_mul(query_head_count)
            .and_then(|value| value.checked_mul(query_key_head_dim))
            .ok_or(CudaBf16PagedAttentionError::ShapeOverflow)?;
        let query_required_bytes = query_elements
            .checked_mul(BF16_BYTES)
            .ok_or(CudaBf16PagedAttentionError::ShapeOverflow)?;
        let query_request_indices_required_bytes = query_count
            .checked_mul(size_of::<u32>())
            .ok_or(CudaBf16PagedAttentionError::ShapeOverflow)?;
        let query_sequence_lengths_required_bytes = query_count
            .checked_mul(size_of::<u64>())
            .ok_or(CudaBf16PagedAttentionError::ShapeOverflow)?;
        let request_slot_offsets_required_bytes = request_count
            .checked_add(1)
            .and_then(|value| value.checked_mul(size_of::<u64>()))
            .ok_or(CudaBf16PagedAttentionError::ShapeOverflow)?;
        let sequence_slots_required_bytes = sequence_slot_count
            .checked_mul(size_of::<u64>())
            .ok_or(CudaBf16PagedAttentionError::ShapeOverflow)?;
        let last_slot = slot_count - 1;
        let last_page = last_slot / layout.page_size;
        let last_token = last_slot % layout.page_size;
        let key_end = layout
            .key_in_page_offset
            .checked_add(
                last_token
                    .checked_mul(layout.key_row_bytes)
                    .ok_or(CudaBf16PagedAttentionError::ShapeOverflow)?,
            )
            .and_then(|offset| offset.checked_add(layout.key_row_bytes))
            .ok_or(CudaBf16PagedAttentionError::ShapeOverflow)?;
        let value_end = layout
            .value_in_page_offset
            .checked_add(
                last_token
                    .checked_mul(layout.value_row_bytes)
                    .ok_or(CudaBf16PagedAttentionError::ShapeOverflow)?,
            )
            .and_then(|offset| offset.checked_add(layout.value_row_bytes))
            .ok_or(CudaBf16PagedAttentionError::ShapeOverflow)?;
        let pool_required_bytes = last_page
            .checked_mul(layout.page_stride_bytes)
            .and_then(|offset| offset.checked_add(key_end.max(value_end)))
            .ok_or(CudaBf16PagedAttentionError::ShapeOverflow)?;
        let output_required_bytes = query_count
            .checked_mul(query_head_count)
            .and_then(|value| value.checked_mul(value_head_dim))
            .and_then(|value| value.checked_mul(BF16_BYTES))
            .ok_or(CudaBf16PagedAttentionError::ShapeOverflow)?;
        let shared_memory_bytes = (CUDA_BLOCK_SIZE as usize)
            .checked_add(value_head_dim)
            .and_then(|value| value.checked_mul(size_of::<f32>()))
            .and_then(|value| u32::try_from(value).ok())
            .ok_or(CudaBf16PagedAttentionError::ShapeOverflow)?;
        let grid_x = u32::try_from(query_count).map_err(|_| {
            CudaBf16PagedAttentionError::DimensionTooLarge {
                dimension: "query_count",
                value: query_count,
                maximum: u32::MAX as usize,
            }
        })?;
        let grid_y = u32::try_from(query_head_count).map_err(|_| {
            CudaBf16PagedAttentionError::DimensionTooLarge {
                dimension: "query_head_count",
                value: query_head_count,
                maximum: u32::MAX as usize,
            }
        })?;

        Ok(Self {
            query_count,
            request_count,
            sequence_slot_count,
            slot_count,
            query_head_count,
            kv_head_count,
            query_key_head_dim,
            value_head_dim,
            scale,
            layout,
            query_required_bytes,
            query_request_indices_required_bytes,
            query_sequence_lengths_required_bytes,
            request_slot_offsets_required_bytes,
            sequence_slots_required_bytes,
            pool_required_bytes,
            output_required_bytes,
            grid: CudaLaunchDimensions::new(grid_x, grid_y, 1),
            shared_memory_bytes,
        })
    }

    pub const fn query_count(self) -> usize {
        self.query_count
    }

    pub const fn request_count(self) -> usize {
        self.request_count
    }

    pub const fn sequence_slot_count(self) -> usize {
        self.sequence_slot_count
    }

    pub const fn slot_count(self) -> usize {
        self.slot_count
    }

    pub const fn query_head_count(self) -> usize {
        self.query_head_count
    }

    pub const fn kv_head_count(self) -> usize {
        self.kv_head_count
    }

    pub const fn query_key_head_dim(self) -> usize {
        self.query_key_head_dim
    }

    pub const fn value_head_dim(self) -> usize {
        self.value_head_dim
    }

    pub const fn scale(self) -> f32 {
        self.scale
    }

    pub const fn layout(self) -> CudaBf16PagedAttentionLayout {
        self.layout
    }

    pub const fn query_required_bytes(self) -> usize {
        self.query_required_bytes
    }

    pub const fn output_required_bytes(self) -> usize {
        self.output_required_bytes
    }

    pub const fn pool_required_bytes(self) -> usize {
        self.pool_required_bytes
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CudaBf16PagedAttentionError {
    Cuda(CudaError),
    Nvrtc(NvrtcError),
    UnsupportedComputeCapability {
        actual: CudaComputeCapability,
        minimum: CudaComputeCapability,
    },
    ZeroDimension(&'static str),
    ShapeOverflow,
    DimensionTooLarge {
        dimension: &'static str,
        value: usize,
        maximum: usize,
    },
    InvalidScale(f32),
    QueryHeadsNotDivisibleByKvHeads {
        query_head_count: usize,
        kv_head_count: usize,
    },
    KvRowByteSizeMismatch {
        expected: usize,
        actual: usize,
    },
    ValueRowByteSizeMismatch {
        expected: usize,
        actual: usize,
    },
    TensorPageRegionOutOfBounds {
        tensor: &'static str,
        offset: usize,
        tensor_page_bytes: usize,
        page_stride_bytes: usize,
    },
    TensorPageRegionsOverlap {
        key_offset: usize,
        value_offset: usize,
        key_page_bytes: usize,
        value_page_bytes: usize,
    },
    DeviceMismatch {
        allocation: &'static str,
        expected_ordinal: usize,
        actual_ordinal: usize,
    },
    DeviceMetadataInvalid {
        flags: u32,
    },
}

impl fmt::Display for CudaBf16PagedAttentionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(error) => write!(formatter, "CUDA paged attention failed: {error}"),
            Self::Nvrtc(error) => {
                write!(
                    formatter,
                    "CUDA paged attention compilation failed: {error}"
                )
            }
            Self::UnsupportedComputeCapability { actual, minimum } => write!(
                formatter,
                "CUDA BF16 paged attention requires compute capability {minimum} or newer, found {actual}"
            ),
            Self::ZeroDimension(dimension) => write!(
                formatter,
                "CUDA paged attention dimension {dimension} must be non-zero"
            ),
            Self::ShapeOverflow => formatter.write_str("CUDA paged attention shape overflowed"),
            Self::DimensionTooLarge {
                dimension,
                value,
                maximum,
            } => write!(
                formatter,
                "CUDA paged attention dimension {dimension}={value} exceeds maximum {maximum}"
            ),
            Self::InvalidScale(scale) => write!(
                formatter,
                "CUDA paged attention scale must be finite and positive, got {scale}"
            ),
            Self::QueryHeadsNotDivisibleByKvHeads {
                query_head_count,
                kv_head_count,
            } => write!(
                formatter,
                "CUDA paged attention query head count {query_head_count} is not divisible by KV head count {kv_head_count}"
            ),
            Self::KvRowByteSizeMismatch { expected, actual } => write!(
                formatter,
                "CUDA BF16 paged attention requires {expected} bytes per KV row, layout has {actual}"
            ),
            Self::ValueRowByteSizeMismatch { expected, actual } => write!(
                formatter,
                "CUDA BF16 paged attention requires {expected} bytes per value row, layout has {actual}"
            ),
            Self::TensorPageRegionOutOfBounds {
                tensor,
                offset,
                tensor_page_bytes,
                page_stride_bytes,
            } => write!(
                formatter,
                "CUDA paged attention {tensor} page region [{offset}, {}) exceeds page stride {page_stride_bytes}",
                offset.saturating_add(*tensor_page_bytes)
            ),
            Self::TensorPageRegionsOverlap {
                key_offset,
                value_offset,
                key_page_bytes,
                value_page_bytes,
            } => write!(
                formatter,
                "CUDA paged attention key/value page regions overlap: key offset {key_offset} ({key_page_bytes} bytes), value offset {value_offset} ({value_page_bytes} bytes)"
            ),
            Self::DeviceMismatch {
                allocation,
                expected_ordinal,
                actual_ordinal,
            } => write!(
                formatter,
                "CUDA paged attention allocation {allocation} is on device {actual_ordinal}, expected device {expected_ordinal}"
            ),
            Self::DeviceMetadataInvalid { flags } => write!(
                formatter,
                "CUDA paged attention rejected device metadata with error flags 0x{flags:x} (request=0x1, sequence=0x2, slot=0x4)"
            ),
        }
    }
}

impl std::error::Error for CudaBf16PagedAttentionError {}

impl From<CudaError> for CudaBf16PagedAttentionError {
    fn from(value: CudaError) -> Self {
        Self::Cuda(value)
    }
}

impl From<NvrtcError> for CudaBf16PagedAttentionError {
    fn from(value: NvrtcError) -> Self {
        Self::Nvrtc(value)
    }
}

pub struct CudaBf16PagedAttentionKernels {
    context: CudaContext,
    _module: CudaModule,
    forward: CudaFunction,
    error_flag: CudaDeviceAllocation,
}

pub struct CudaBf16PagedAttentionLaunch<'a> {
    pub queries: &'a CudaDeviceAllocation,
    pub queries_offset: usize,
    pub query_request_indices: &'a CudaDeviceAllocation,
    pub query_request_indices_offset: usize,
    pub query_sequence_lengths: &'a CudaDeviceAllocation,
    pub query_sequence_lengths_offset: usize,
    pub request_slot_offsets: &'a CudaDeviceAllocation,
    pub request_slot_offsets_offset: usize,
    pub sequence_slots: &'a CudaDeviceAllocation,
    pub sequence_slots_offset: usize,
    pub pool: &'a CudaDeviceAllocation,
    pub pool_offset: usize,
    pub output: &'a mut CudaDeviceAllocation,
    pub output_offset: usize,
}

impl CudaBf16PagedAttentionKernels {
    pub fn compile(
        context: &CudaContext,
        compute_capability: CudaComputeCapability,
    ) -> Result<Self, CudaBf16PagedAttentionError> {
        if compute_capability < MINIMUM_BF16_COMPUTE_CAPABILITY {
            return Err(CudaBf16PagedAttentionError::UnsupportedComputeCapability {
                actual: compute_capability,
                minimum: MINIMUM_BF16_COMPUTE_CAPABILITY,
            });
        }
        let compiler = NvrtcCompiler::load()?;
        let ptx = compiler.compile_ptx(
            CUDA_BF16_PAGED_ATTENTION_SOURCE,
            "sglang_bf16_paged_attention.cu",
            compute_capability,
        )?;
        let module = context.load_module(&ptx)?;
        let forward = module.get_function("sglang_paged_attention_bf16")?;
        let mut error_flag = context.allocate(size_of::<u32>())?;
        error_flag.fill(0)?;
        Ok(Self {
            context: context.clone(),
            _module: module,
            forward,
            error_flag,
        })
    }

    pub fn forward(
        &mut self,
        plan: CudaBf16PagedAttentionPlan,
        launch: CudaBf16PagedAttentionLaunch<'_>,
    ) -> Result<(), CudaBf16PagedAttentionError> {
        let CudaBf16PagedAttentionLaunch {
            queries,
            queries_offset,
            query_request_indices,
            query_request_indices_offset,
            query_sequence_lengths,
            query_sequence_lengths_offset,
            request_slot_offsets,
            request_slot_offsets_offset,
            sequence_slots,
            sequence_slots_offset,
            pool,
            pool_offset,
            output,
            output_offset,
        } = launch;
        for (name, allocation) in [
            ("queries", queries),
            ("query_request_indices", query_request_indices),
            ("query_sequence_lengths", query_sequence_lengths),
            ("request_slot_offsets", request_slot_offsets),
            ("sequence_slots", sequence_slots),
            ("pool", pool),
            ("output", &*output),
        ] {
            self.validate_device(name, allocation)?;
        }

        let mut queries_ptr = queries.device_ptr_at(queries_offset, plan.query_required_bytes)?;
        let mut query_request_indices_ptr = query_request_indices.device_ptr_at(
            query_request_indices_offset,
            plan.query_request_indices_required_bytes,
        )?;
        let mut query_sequence_lengths_ptr = query_sequence_lengths.device_ptr_at(
            query_sequence_lengths_offset,
            plan.query_sequence_lengths_required_bytes,
        )?;
        let mut request_slot_offsets_ptr = request_slot_offsets.device_ptr_at(
            request_slot_offsets_offset,
            plan.request_slot_offsets_required_bytes,
        )?;
        let mut sequence_slots_ptr = sequence_slots
            .device_ptr_at(sequence_slots_offset, plan.sequence_slots_required_bytes)?;
        let mut pool_ptr = pool.device_ptr_at(pool_offset, plan.pool_required_bytes)?;
        let mut output_ptr = output.device_ptr_at(output_offset, plan.output_required_bytes)?;
        self.error_flag.fill(0)?;
        let mut error_ptr = self.error_flag.device_ptr_at(0, size_of::<u32>())?;

        let mut query_count = dimension_u64("query_count", plan.query_count)?;
        let mut request_count = dimension_u64("request_count", plan.request_count)?;
        let mut sequence_slot_count =
            dimension_u64("sequence_slot_count", plan.sequence_slot_count)?;
        let mut slot_count = dimension_u64("slot_count", plan.slot_count)?;
        let mut page_size = dimension_u64("page_size", plan.layout.page_size)?;
        let mut page_stride_bytes =
            dimension_u64("page_stride_bytes", plan.layout.page_stride_bytes)?;
        let mut key_in_page_offset =
            dimension_u64("key_in_page_offset", plan.layout.key_in_page_offset)?;
        let mut value_in_page_offset =
            dimension_u64("value_in_page_offset", plan.layout.value_in_page_offset)?;
        let mut key_row_bytes = dimension_u64("key_row_bytes", plan.layout.key_row_bytes)?;
        let mut value_row_bytes = dimension_u64("value_row_bytes", plan.layout.value_row_bytes)?;
        let mut query_head_count = dimension_u32("query_head_count", plan.query_head_count)?;
        let mut kv_head_count = dimension_u32("kv_head_count", plan.kv_head_count)?;
        let mut query_key_head_dim = dimension_u32("query_key_head_dim", plan.query_key_head_dim)?;
        let mut value_head_dim = dimension_u32("value_head_dim", plan.value_head_dim)?;
        let mut scale = plan.scale;
        let mut arguments = [
            argument_pointer(&mut queries_ptr),
            argument_pointer(&mut query_request_indices_ptr),
            argument_pointer(&mut query_sequence_lengths_ptr),
            argument_pointer(&mut request_slot_offsets_ptr),
            argument_pointer(&mut sequence_slots_ptr),
            argument_pointer(&mut pool_ptr),
            argument_pointer(&mut output_ptr),
            argument_pointer(&mut error_ptr),
            argument_pointer(&mut query_count),
            argument_pointer(&mut request_count),
            argument_pointer(&mut sequence_slot_count),
            argument_pointer(&mut slot_count),
            argument_pointer(&mut page_size),
            argument_pointer(&mut page_stride_bytes),
            argument_pointer(&mut key_in_page_offset),
            argument_pointer(&mut value_in_page_offset),
            argument_pointer(&mut key_row_bytes),
            argument_pointer(&mut value_row_bytes),
            argument_pointer(&mut query_head_count),
            argument_pointer(&mut kv_head_count),
            argument_pointer(&mut query_key_head_dim),
            argument_pointer(&mut value_head_dim),
            argument_pointer(&mut scale),
        ];
        unsafe {
            self.forward.launch(
                plan.grid,
                CudaLaunchDimensions::new(CUDA_BLOCK_SIZE, 1, 1),
                plan.shared_memory_bytes,
                &mut arguments,
            )?;
        }
        self.context.synchronize()?;

        let mut error_flag = [0_u8; size_of::<u32>()];
        self.error_flag.copy_to_host(0, &mut error_flag)?;
        let flags = u32::from_ne_bytes(error_flag);
        if flags != 0 {
            return Err(CudaBf16PagedAttentionError::DeviceMetadataInvalid { flags });
        }
        Ok(())
    }

    fn validate_device(
        &self,
        allocation_name: &'static str,
        allocation: &CudaDeviceAllocation,
    ) -> Result<(), CudaBf16PagedAttentionError> {
        let expected_ordinal = self.context.device_ordinal();
        let actual_ordinal = allocation.device_ordinal();
        if actual_ordinal == expected_ordinal {
            Ok(())
        } else {
            Err(CudaBf16PagedAttentionError::DeviceMismatch {
                allocation: allocation_name,
                expected_ordinal,
                actual_ordinal,
            })
        }
    }
}

fn validate_tensor_page_region(
    tensor: &'static str,
    offset: usize,
    tensor_page_bytes: usize,
    page_stride_bytes: usize,
) -> Result<usize, CudaBf16PagedAttentionError> {
    let end = offset
        .checked_add(tensor_page_bytes)
        .ok_or(CudaBf16PagedAttentionError::ShapeOverflow)?;
    if end > page_stride_bytes {
        Err(CudaBf16PagedAttentionError::TensorPageRegionOutOfBounds {
            tensor,
            offset,
            tensor_page_bytes,
            page_stride_bytes,
        })
    } else {
        Ok(end)
    }
}

fn dimension_u64(
    dimension: &'static str,
    value: usize,
) -> Result<u64, CudaBf16PagedAttentionError> {
    u64::try_from(value).map_err(|_| CudaBf16PagedAttentionError::DimensionTooLarge {
        dimension,
        value,
        maximum: u64::MAX as usize,
    })
}

fn dimension_u32(
    dimension: &'static str,
    value: usize,
) -> Result<u32, CudaBf16PagedAttentionError> {
    u32::try_from(value).map_err(|_| CudaBf16PagedAttentionError::DimensionTooLarge {
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

    fn test_layout() -> CudaBf16PagedAttentionLayout {
        CudaBf16PagedAttentionLayout::new(4, 512, 256, 384, 32)
    }

    fn test_plan_config() -> CudaBf16PagedAttentionPlanConfig {
        CudaBf16PagedAttentionPlanConfig {
            query_count: 1,
            request_count: 1,
            sequence_slot_count: 1,
            slot_count: 4,
            query_head_count: 4,
            kv_head_count: 2,
            query_key_head_dim: 8,
            value_head_dim: 8,
            scale: 1.0,
            layout: test_layout(),
        }
    }

    #[test]
    fn plan_maps_gqa_queries_into_page_major_bf16_kv() {
        let plan = CudaBf16PagedAttentionPlan::new(CudaBf16PagedAttentionPlanConfig {
            query_count: 3,
            request_count: 2,
            sequence_slot_count: 6,
            slot_count: 12,
            scale: 8.0_f32.sqrt().recip(),
            ..test_plan_config()
        })
        .expect("valid BF16 paged attention plan should build");

        assert_eq!(plan.query_count(), 3);
        assert_eq!(plan.request_count(), 2);
        assert_eq!(plan.sequence_slot_count(), 6);
        assert_eq!(plan.slot_count(), 12);
        assert_eq!(plan.query_head_count(), 4);
        assert_eq!(plan.kv_head_count(), 2);
        assert_eq!(plan.query_key_head_dim(), 8);
        assert_eq!(plan.value_head_dim(), 8);
        assert_eq!(plan.layout(), test_layout());
        assert_eq!(plan.query_required_bytes(), 3 * 4 * 8 * 2);
        assert_eq!(plan.output_required_bytes(), 3 * 4 * 8 * 2);
        assert_eq!(plan.pool_required_bytes(), 1_536);
        assert_eq!(plan.grid, CudaLaunchDimensions::new(3, 4, 1));
        assert_eq!(plan.shared_memory_bytes, (256 + 8) * 4);
    }

    #[test]
    fn plan_fails_fast_on_head_and_bf16_layout_mismatches() {
        assert_eq!(
            CudaBf16PagedAttentionPlan::new(CudaBf16PagedAttentionPlanConfig {
                query_head_count: 3,
                ..test_plan_config()
            }),
            Err(
                CudaBf16PagedAttentionError::QueryHeadsNotDivisibleByKvHeads {
                    query_head_count: 3,
                    kv_head_count: 2,
                }
            )
        );
        assert_eq!(
            CudaBf16PagedAttentionPlan::new(CudaBf16PagedAttentionPlanConfig {
                layout: CudaBf16PagedAttentionLayout::new(4, 512, 256, 384, 31),
                ..test_plan_config()
            }),
            Err(CudaBf16PagedAttentionError::KvRowByteSizeMismatch {
                expected: 32,
                actual: 31,
            })
        );
        assert_eq!(
            CudaBf16PagedAttentionPlan::new(CudaBf16PagedAttentionPlanConfig {
                value_head_dim: 4,
                ..test_plan_config()
            }),
            Err(CudaBf16PagedAttentionError::ValueRowByteSizeMismatch {
                expected: 16,
                actual: 32,
            })
        );
    }

    #[test]
    fn plan_supports_unequal_mla_key_and_value_widths() {
        let plan = CudaBf16PagedAttentionPlan::new(CudaBf16PagedAttentionPlanConfig {
            kv_head_count: 1,
            query_key_head_dim: 12,
            value_head_dim: 8,
            layout: CudaBf16PagedAttentionLayout::new_tensor_pair(4, 512, 128, 224, 24, 16),
            ..test_plan_config()
        })
        .expect("valid unequal-width MLA plan should build");

        assert_eq!(plan.query_required_bytes(), 4 * 12 * 2);
        assert_eq!(plan.output_required_bytes(), 4 * 8 * 2);
        assert_eq!(plan.layout().key_row_bytes(), 24);
        assert_eq!(plan.layout().value_row_bytes(), 16);
    }

    #[test]
    fn plan_rejects_empty_or_invalid_scaled_attention() {
        assert_eq!(
            CudaBf16PagedAttentionPlan::new(CudaBf16PagedAttentionPlanConfig {
                query_count: 0,
                ..test_plan_config()
            }),
            Err(CudaBf16PagedAttentionError::ZeroDimension("query_count"))
        );
        assert_eq!(
            CudaBf16PagedAttentionPlan::new(CudaBf16PagedAttentionPlanConfig {
                scale: 0.0,
                ..test_plan_config()
            }),
            Err(CudaBf16PagedAttentionError::InvalidScale(0.0))
        );
    }

    #[test]
    fn embedded_source_uses_bf16_with_fp32_online_softmax() {
        assert!(CUDA_BF16_PAGED_ATTENTION_SOURCE.contains("sglang_paged_attention_bf16"));
        assert!(CUDA_BF16_PAGED_ATTENTION_SOURCE.contains("__bfloat162float"));
        assert!(CUDA_BF16_PAGED_ATTENTION_SOURCE.contains("__float2bfloat16_rn"));
        assert!(CUDA_BF16_PAGED_ATTENTION_SOURCE.contains("running_max"));
        assert!(CUDA_BF16_PAGED_ATTENTION_SOURCE.contains("atomicOr"));
    }

    #[test]
    fn bf16_kernel_capability_floor_is_architecture_not_product_specific() {
        assert_eq!(
            MINIMUM_BF16_COMPUTE_CAPABILITY,
            CudaComputeCapability::new(8, 0)
        );
        assert!(CudaComputeCapability::new(8, 0) >= MINIMUM_BF16_COMPUTE_CAPABILITY);
        assert!(CudaComputeCapability::new(9, 0) >= MINIMUM_BF16_COMPUTE_CAPABILITY);
        assert!(CudaComputeCapability::new(10, 0) >= MINIMUM_BF16_COMPUTE_CAPABILITY);
    }
}
