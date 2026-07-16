use std::fmt;

use sglang_kernel::cuda::{CudaComputeCapability, CudaContext, CudaDeviceAllocation, CudaError};
use sglang_kernel::cuda_attention::{
    CudaBf16PagedAttentionError, CudaBf16PagedAttentionKernels, CudaBf16PagedAttentionLayout,
    CudaBf16PagedAttentionPlan,
};

use crate::cuda_kv_cache::{CudaKvCachePool, CudaKvCachePoolError, CudaKvCachePoolLayout};
use crate::model_executor::ModelWorkerBatch;
use crate::transfer::KvCacheDtype;

#[derive(Clone, Debug, PartialEq)]
pub enum CudaPagedAttentionError {
    EmptyBatch,
    EmptyRequestInput {
        request_index: usize,
    },
    BatchFieldLengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    FlattenedRangeNotContiguous {
        field: &'static str,
        request_index: usize,
        expected_offset: usize,
        actual_offset: usize,
    },
    FlattenedRangeOutOfBounds {
        field: &'static str,
        request_index: usize,
        offset: usize,
        count: usize,
        flattened_len: usize,
    },
    SequenceLengthMismatch {
        request_index: usize,
        sequence_length: usize,
        sequence_token_count: usize,
    },
    QueryPositionOutOfRange {
        request_index: usize,
        query_index: usize,
        position: usize,
        sequence_token_count: usize,
    },
    IntegerConversion {
        field: &'static str,
        value: usize,
        target: &'static str,
    },
    UnsupportedKvDtype {
        actual: KvCacheDtype,
        required: KvCacheDtype,
    },
    LayerOutOfRange {
        layer_index: usize,
        layer_count: usize,
    },
    KvPairRequiresTwoTensors {
        tensor_count: usize,
    },
    SizeOverflow,
    Cuda(CudaError),
    KvCache(CudaKvCachePoolError),
    Kernel(CudaBf16PagedAttentionError),
}

impl fmt::Display for CudaPagedAttentionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBatch => {
                formatter.write_str("CUDA paged attention requires a non-empty batch")
            }
            Self::EmptyRequestInput { request_index } => write!(
                formatter,
                "CUDA paged attention request {request_index} has no query tokens"
            ),
            Self::BatchFieldLengthMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "CUDA paged attention batch field {field} has length {actual}, expected {expected}"
            ),
            Self::FlattenedRangeNotContiguous {
                field,
                request_index,
                expected_offset,
                actual_offset,
            } => write!(
                formatter,
                "CUDA paged attention batch field {field} for request {request_index} starts at {actual_offset}, expected contiguous offset {expected_offset}"
            ),
            Self::FlattenedRangeOutOfBounds {
                field,
                request_index,
                offset,
                count,
                flattened_len,
            } => write!(
                formatter,
                "CUDA paged attention batch field {field} range [{offset}, {}) for request {request_index} exceeds flattened length {flattened_len}",
                offset.saturating_add(*count)
            ),
            Self::SequenceLengthMismatch {
                request_index,
                sequence_length,
                sequence_token_count,
            } => write!(
                formatter,
                "CUDA paged attention request {request_index} reports sequence length {sequence_length} but contains {sequence_token_count} sequence tokens"
            ),
            Self::QueryPositionOutOfRange {
                request_index,
                query_index,
                position,
                sequence_token_count,
            } => write!(
                formatter,
                "CUDA paged attention query {query_index} for request {request_index} has position {position}, outside {sequence_token_count} sequence tokens"
            ),
            Self::IntegerConversion {
                field,
                value,
                target,
            } => write!(
                formatter,
                "CUDA paged attention field {field} value {value} cannot be represented as {target}"
            ),
            Self::UnsupportedKvDtype { actual, required } => write!(
                formatter,
                "CUDA paged attention requires KV cache dtype {required:?}, found {actual:?}"
            ),
            Self::LayerOutOfRange {
                layer_index,
                layer_count,
            } => write!(
                formatter,
                "CUDA paged attention layer {layer_index} is outside {layer_count} KV cache layers"
            ),
            Self::KvPairRequiresTwoTensors { tensor_count } => write!(
                formatter,
                "CUDA paged attention requires key and value tensors, KV layout has {tensor_count} tensor(s) per token"
            ),
            Self::SizeOverflow => formatter.write_str("CUDA paged attention size overflowed"),
            Self::Cuda(error) => write!(formatter, "CUDA paged attention metadata failed: {error}"),
            Self::KvCache(error) => {
                write!(formatter, "CUDA paged attention KV cache failed: {error}")
            }
            Self::Kernel(error) => write!(formatter, "CUDA paged attention kernel failed: {error}"),
        }
    }
}

impl std::error::Error for CudaPagedAttentionError {}

impl From<CudaError> for CudaPagedAttentionError {
    fn from(value: CudaError) -> Self {
        Self::Cuda(value)
    }
}

impl From<CudaKvCachePoolError> for CudaPagedAttentionError {
    fn from(value: CudaKvCachePoolError) -> Self {
        Self::KvCache(value)
    }
}

impl From<CudaBf16PagedAttentionError> for CudaPagedAttentionError {
    fn from(value: CudaBf16PagedAttentionError) -> Self {
        Self::Kernel(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaPagedAttentionMetadata {
    query_request_indices: Vec<u32>,
    query_sequence_lengths: Vec<u64>,
    request_slot_offsets: Vec<u64>,
    sequence_slots: Vec<u64>,
}

impl CudaPagedAttentionMetadata {
    pub fn from_model_worker_batch(
        batch: &ModelWorkerBatch,
        pool_layout: CudaKvCachePoolLayout,
    ) -> Result<Self, CudaPagedAttentionError> {
        let request_count = batch.request_ids().len();
        if request_count == 0 {
            return Err(CudaPagedAttentionError::EmptyBatch);
        }
        validate_batch_field_len(
            "positions",
            batch.input_ids().len(),
            batch.positions().len(),
        )?;
        validate_batch_field_len(
            "sequence_cache_pages",
            batch.sequence_token_ids().len(),
            batch.sequence_cache_pages().len(),
        )?;
        for (field, actual) in [
            ("sequence_lengths", batch.sequence_lengths().len()),
            ("sequence_offsets", batch.sequence_offsets().len()),
            ("sequence_token_counts", batch.sequence_token_counts().len()),
            ("request_offsets", batch.request_offsets().len()),
            ("input_token_counts", batch.input_token_counts().len()),
        ] {
            validate_batch_field_len(field, request_count, actual)?;
        }

        let mut query_request_indices = Vec::with_capacity(batch.input_ids().len());
        let mut query_sequence_lengths = Vec::with_capacity(batch.input_ids().len());
        let mut request_slot_offsets = Vec::with_capacity(request_count + 1);
        let mut sequence_slots = Vec::with_capacity(batch.sequence_cache_pages().len());
        let mut expected_sequence_offset = 0;
        let mut expected_request_offset = 0;
        request_slot_offsets.push(0);

        for request_index in 0..request_count {
            let sequence_offset = batch.sequence_offsets()[request_index];
            let sequence_count = batch.sequence_token_counts()[request_index];
            validate_flattened_range(
                "sequence_token_ids",
                request_index,
                sequence_offset,
                sequence_count,
                expected_sequence_offset,
                batch.sequence_token_ids().len(),
            )?;
            if batch.sequence_lengths()[request_index] != sequence_count {
                return Err(CudaPagedAttentionError::SequenceLengthMismatch {
                    request_index,
                    sequence_length: batch.sequence_lengths()[request_index],
                    sequence_token_count: sequence_count,
                });
            }

            let request_offset = batch.request_offsets()[request_index];
            let input_count = batch.input_token_counts()[request_index];
            if input_count == 0 {
                return Err(CudaPagedAttentionError::EmptyRequestInput { request_index });
            }
            validate_flattened_range(
                "input_ids",
                request_index,
                request_offset,
                input_count,
                expected_request_offset,
                batch.input_ids().len(),
            )?;

            let request_index_u32 = convert_u32("request_index", request_index)?;
            for query_index in request_offset..request_offset + input_count {
                let position = batch.positions()[query_index];
                if position >= sequence_count {
                    return Err(CudaPagedAttentionError::QueryPositionOutOfRange {
                        request_index,
                        query_index,
                        position,
                        sequence_token_count: sequence_count,
                    });
                }
                query_request_indices.push(request_index_u32);
                query_sequence_lengths.push(convert_u64("query_sequence_length", position + 1)?);
            }

            for slot in
                &batch.sequence_cache_pages()[sequence_offset..sequence_offset + sequence_count]
            {
                sequence_slots.push(convert_u64("sequence_slot", slot.as_usize())?);
            }
            expected_sequence_offset += sequence_count;
            expected_request_offset += input_count;
            request_slot_offsets.push(convert_u64(
                "request_slot_offset",
                expected_sequence_offset,
            )?);
        }

        validate_batch_field_len(
            "flattened sequence ranges",
            batch.sequence_token_ids().len(),
            expected_sequence_offset,
        )?;
        validate_batch_field_len(
            "flattened input ranges",
            batch.input_ids().len(),
            expected_request_offset,
        )?;
        let host_slots = sequence_slots
            .iter()
            .copied()
            .map(|slot| {
                usize::try_from(slot).map_err(|_| CudaPagedAttentionError::IntegerConversion {
                    field: "sequence_slot",
                    value: usize::MAX,
                    target: "usize",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        pool_layout.validate_slot_indices(&host_slots)?;

        Ok(Self {
            query_request_indices,
            query_sequence_lengths,
            request_slot_offsets,
            sequence_slots,
        })
    }

    pub fn query_count(&self) -> usize {
        self.query_request_indices.len()
    }

    pub fn request_count(&self) -> usize {
        self.request_slot_offsets.len() - 1
    }

    pub fn sequence_slot_count(&self) -> usize {
        self.sequence_slots.len()
    }

    pub fn query_request_indices(&self) -> &[u32] {
        &self.query_request_indices
    }

    pub fn query_sequence_lengths(&self) -> &[u64] {
        &self.query_sequence_lengths
    }

    pub fn request_slot_offsets(&self) -> &[u64] {
        &self.request_slot_offsets
    }

    pub fn sequence_slots(&self) -> &[u64] {
        &self.sequence_slots
    }

    pub fn upload(
        &self,
        context: &CudaContext,
    ) -> Result<CudaPagedAttentionDeviceMetadata, CudaPagedAttentionError> {
        Ok(CudaPagedAttentionDeviceMetadata {
            query_count: self.query_count(),
            request_count: self.request_count(),
            sequence_slot_count: self.sequence_slot_count(),
            query_request_indices: upload_u32(context, &self.query_request_indices)?,
            query_sequence_lengths: upload_u64(context, &self.query_sequence_lengths)?,
            request_slot_offsets: upload_u64(context, &self.request_slot_offsets)?,
            sequence_slots: upload_u64(context, &self.sequence_slots)?,
        })
    }
}

pub struct CudaPagedAttentionDeviceMetadata {
    query_count: usize,
    request_count: usize,
    sequence_slot_count: usize,
    query_request_indices: CudaDeviceAllocation,
    query_sequence_lengths: CudaDeviceAllocation,
    request_slot_offsets: CudaDeviceAllocation,
    sequence_slots: CudaDeviceAllocation,
}

impl CudaPagedAttentionDeviceMetadata {
    pub fn query_count(&self) -> usize {
        self.query_count
    }

    pub fn request_count(&self) -> usize {
        self.request_count
    }

    pub fn sequence_slot_count(&self) -> usize {
        self.sequence_slot_count
    }
}

pub struct CudaBf16PagedAttentionExecutor {
    kernels: CudaBf16PagedAttentionKernels,
}

impl CudaBf16PagedAttentionExecutor {
    pub fn compile(
        context: &CudaContext,
        compute_capability: CudaComputeCapability,
    ) -> Result<Self, CudaPagedAttentionError> {
        Ok(Self {
            kernels: CudaBf16PagedAttentionKernels::compile(context, compute_capability)?,
        })
    }

    pub fn plan(
        pool_layout: CudaKvCachePoolLayout,
        layer_index: usize,
        metadata: &CudaPagedAttentionMetadata,
        query_head_count: usize,
        scale: f32,
    ) -> Result<CudaBf16PagedAttentionPlan, CudaPagedAttentionError> {
        build_plan(
            pool_layout,
            layer_index,
            metadata.query_count(),
            metadata.request_count(),
            metadata.sequence_slot_count(),
            query_head_count,
            scale,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        kv_cache: &CudaKvCachePool,
        layer_index: usize,
        metadata: &CudaPagedAttentionDeviceMetadata,
        queries: &CudaDeviceAllocation,
        queries_offset: usize,
        query_head_count: usize,
        scale: f32,
        output: &mut CudaDeviceAllocation,
        output_offset: usize,
    ) -> Result<(), CudaPagedAttentionError> {
        let plan = build_plan(
            kv_cache.layout(),
            layer_index,
            metadata.query_count,
            metadata.request_count,
            metadata.sequence_slot_count,
            query_head_count,
            scale,
        )?;
        self.kernels.forward(
            plan,
            queries,
            queries_offset,
            &metadata.query_request_indices,
            0,
            &metadata.query_sequence_lengths,
            0,
            &metadata.request_slot_offsets,
            0,
            &metadata.sequence_slots,
            0,
            kv_cache.allocation(),
            0,
            output,
            output_offset,
        )?;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn build_plan(
    pool_layout: CudaKvCachePoolLayout,
    layer_index: usize,
    query_count: usize,
    request_count: usize,
    sequence_slot_count: usize,
    query_head_count: usize,
    scale: f32,
) -> Result<CudaBf16PagedAttentionPlan, CudaPagedAttentionError> {
    let runtime = pool_layout.runtime();
    if runtime.dtype != KvCacheDtype::Bfloat16 {
        return Err(CudaPagedAttentionError::UnsupportedKvDtype {
            actual: runtime.dtype,
            required: KvCacheDtype::Bfloat16,
        });
    }
    if layer_index >= runtime.num_layers {
        return Err(CudaPagedAttentionError::LayerOutOfRange {
            layer_index,
            layer_count: runtime.num_layers,
        });
    }
    if runtime.kv_tensors_per_token < 2 {
        return Err(CudaPagedAttentionError::KvPairRequiresTwoTensors {
            tensor_count: runtime.kv_tensors_per_token,
        });
    }
    let key_in_page_offset = layer_index
        .checked_mul(pool_layout.bytes_per_layer_page())
        .ok_or(CudaPagedAttentionError::SizeOverflow)?;
    let value_in_page_offset = key_in_page_offset
        .checked_add(pool_layout.bytes_per_tensor_page())
        .ok_or(CudaPagedAttentionError::SizeOverflow)?;
    let attention_layout = CudaBf16PagedAttentionLayout::new(
        runtime.page_size,
        runtime.page_size_bytes,
        key_in_page_offset,
        value_in_page_offset,
        pool_layout.bytes_per_token_per_tensor(),
    );
    Ok(CudaBf16PagedAttentionPlan::new(
        query_count,
        request_count,
        sequence_slot_count,
        pool_layout.slot_count(),
        query_head_count,
        runtime.kv_heads,
        runtime.head_dim,
        scale,
        attention_layout,
    )?)
}

fn validate_batch_field_len(
    field: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), CudaPagedAttentionError> {
    if actual == expected {
        Ok(())
    } else {
        Err(CudaPagedAttentionError::BatchFieldLengthMismatch {
            field,
            expected,
            actual,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_flattened_range(
    field: &'static str,
    request_index: usize,
    offset: usize,
    count: usize,
    expected_offset: usize,
    flattened_len: usize,
) -> Result<(), CudaPagedAttentionError> {
    if offset != expected_offset {
        return Err(CudaPagedAttentionError::FlattenedRangeNotContiguous {
            field,
            request_index,
            expected_offset,
            actual_offset: offset,
        });
    }
    if offset
        .checked_add(count)
        .is_none_or(|end| end > flattened_len)
    {
        return Err(CudaPagedAttentionError::FlattenedRangeOutOfBounds {
            field,
            request_index,
            offset,
            count,
            flattened_len,
        });
    }
    Ok(())
}

fn convert_u32(field: &'static str, value: usize) -> Result<u32, CudaPagedAttentionError> {
    u32::try_from(value).map_err(|_| CudaPagedAttentionError::IntegerConversion {
        field,
        value,
        target: "u32",
    })
}

fn convert_u64(field: &'static str, value: usize) -> Result<u64, CudaPagedAttentionError> {
    u64::try_from(value).map_err(|_| CudaPagedAttentionError::IntegerConversion {
        field,
        value,
        target: "u64",
    })
}

fn upload_u32(
    context: &CudaContext,
    values: &[u32],
) -> Result<CudaDeviceAllocation, CudaPagedAttentionError> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    upload_bytes(context, &bytes)
}

fn upload_u64(
    context: &CudaContext,
    values: &[u64],
) -> Result<CudaDeviceAllocation, CudaPagedAttentionError> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    upload_bytes(context, &bytes)
}

fn upload_bytes(
    context: &CudaContext,
    bytes: &[u8],
) -> Result<CudaDeviceAllocation, CudaPagedAttentionError> {
    let mut allocation = context.allocate(bytes.len())?;
    allocation.copy_from_host(0, bytes)?;
    Ok(allocation)
}
