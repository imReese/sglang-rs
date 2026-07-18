use std::fmt;

use sglang_kernel::cuda::{CudaComputeCapability, CudaContext, CudaDeviceAllocation, CudaError};
use sglang_kernel::cuda_attention::{
    CudaBf16PagedAttentionError, CudaBf16PagedAttentionKernels, CudaBf16PagedAttentionLaunch,
    CudaBf16PagedAttentionLayout, CudaBf16PagedAttentionPlan, CudaBf16PagedAttentionPlanConfig,
};

use crate::cuda_kv_cache::{CudaKvStorage, CudaKvStorageError};
use crate::kv_cache::KvCacheDtype;
use crate::kv_cache::{PagedKvCacheLayout, PagedKvCacheLayoutError};
use crate::model_executor::ModelWorkerBatch;

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
    TensorRowGeometryMismatch {
        tensor: &'static str,
        row_bytes: usize,
        kv_head_count: usize,
        element_bytes: usize,
    },
    Cuda(CudaError),
    KvLayout(PagedKvCacheLayoutError),
    KvStorage(CudaKvStorageError),
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
            Self::TensorRowGeometryMismatch {
                tensor,
                row_bytes,
                kv_head_count,
                element_bytes,
            } => write!(
                formatter,
                "CUDA paged attention {tensor} row has {row_bytes} bytes, which cannot be divided across {kv_head_count} heads of {element_bytes}-byte elements"
            ),
            Self::Cuda(error) => write!(formatter, "CUDA paged attention metadata failed: {error}"),
            Self::KvLayout(error) => {
                write!(formatter, "CUDA paged attention KV layout failed: {error}")
            }
            Self::KvStorage(error) => {
                write!(formatter, "CUDA paged attention KV storage failed: {error}")
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

impl From<CudaKvStorageError> for CudaPagedAttentionError {
    fn from(value: CudaKvStorageError) -> Self {
        Self::KvStorage(value)
    }
}

impl From<PagedKvCacheLayoutError> for CudaPagedAttentionError {
    fn from(value: PagedKvCacheLayoutError) -> Self {
        Self::KvLayout(value)
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
    pub fn for_single_query(
        sequence_slots: &[crate::cache::CachePageId],
        pool_layout: PagedKvCacheLayout,
    ) -> Result<Self, CudaPagedAttentionError> {
        let host_slots = sequence_slots
            .iter()
            .map(|slot| slot.as_usize())
            .collect::<Vec<_>>();
        pool_layout.validate_slot_indices(&host_slots)?;
        let sequence_length = convert_u64("single query sequence length", host_slots.len())?;
        let sequence_slots = host_slots
            .into_iter()
            .map(|slot| convert_u64("single query sequence slot", slot))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            query_request_indices: vec![0],
            query_sequence_lengths: vec![sequence_length],
            request_slot_offsets: vec![0, sequence_length],
            sequence_slots,
        })
    }

    pub fn from_model_worker_batch(
        batch: &ModelWorkerBatch,
        pool_layout: PagedKvCacheLayout,
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

pub struct CudaPagedAttentionForward<'a> {
    pub kv_layout: PagedKvCacheLayout,
    pub kv_storage: &'a CudaKvStorage,
    pub layer_index: usize,
    pub metadata: &'a CudaPagedAttentionDeviceMetadata,
    pub queries: &'a CudaDeviceAllocation,
    pub queries_offset: usize,
    pub query_head_count: usize,
    pub scale: f32,
    pub output: &'a mut CudaDeviceAllocation,
    pub output_offset: usize,
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
        pool_layout: PagedKvCacheLayout,
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

    pub fn forward(
        &mut self,
        forward: CudaPagedAttentionForward<'_>,
    ) -> Result<(), CudaPagedAttentionError> {
        let CudaPagedAttentionForward {
            kv_layout,
            kv_storage,
            layer_index,
            metadata,
            queries,
            queries_offset,
            query_head_count,
            scale,
            output,
            output_offset,
        } = forward;
        kv_storage.validate_layout(kv_layout)?;
        let plan = build_plan(
            kv_layout,
            layer_index,
            metadata.query_count,
            metadata.request_count,
            metadata.sequence_slot_count,
            query_head_count,
            scale,
        )?;
        self.kernels.forward(
            plan,
            CudaBf16PagedAttentionLaunch {
                queries,
                queries_offset,
                query_request_indices: &metadata.query_request_indices,
                query_request_indices_offset: 0,
                query_sequence_lengths: &metadata.query_sequence_lengths,
                query_sequence_lengths_offset: 0,
                request_slot_offsets: &metadata.request_slot_offsets,
                request_slot_offsets_offset: 0,
                sequence_slots: &metadata.sequence_slots,
                sequence_slots_offset: 0,
                pool: kv_storage.allocation(),
                pool_offset: 0,
                output,
                output_offset,
            },
        )?;
        Ok(())
    }
}

fn build_plan(
    pool_layout: PagedKvCacheLayout,
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
    let element_bytes = std::mem::size_of::<u16>();
    let key_row_bytes = pool_layout.tensor_token_size_bytes(0)?;
    let value_row_bytes = pool_layout.tensor_token_size_bytes(1)?;
    let key_denominator = runtime.kv_heads.checked_mul(element_bytes).ok_or(
        CudaPagedAttentionError::TensorRowGeometryMismatch {
            tensor: "key",
            row_bytes: key_row_bytes,
            kv_head_count: runtime.kv_heads,
            element_bytes,
        },
    )?;
    let value_denominator = key_denominator;
    if !key_row_bytes.is_multiple_of(key_denominator) {
        return Err(CudaPagedAttentionError::TensorRowGeometryMismatch {
            tensor: "key",
            row_bytes: key_row_bytes,
            kv_head_count: runtime.kv_heads,
            element_bytes,
        });
    }
    if !value_row_bytes.is_multiple_of(value_denominator) {
        return Err(CudaPagedAttentionError::TensorRowGeometryMismatch {
            tensor: "value",
            row_bytes: value_row_bytes,
            kv_head_count: runtime.kv_heads,
            element_bytes,
        });
    }
    let key_offset_bytes = pool_layout
        .tensor_token_byte_range(0, layer_index, 0, 0)?
        .start;
    let value_offset_bytes = pool_layout
        .tensor_token_byte_range(0, layer_index, 1, 0)?
        .start;
    let attention_layout = CudaBf16PagedAttentionLayout::new_tensor_pair(
        runtime.page_size,
        runtime.page_size_bytes,
        key_offset_bytes,
        value_offset_bytes,
        key_row_bytes,
        value_row_bytes,
    );
    Ok(CudaBf16PagedAttentionPlan::new(
        CudaBf16PagedAttentionPlanConfig {
            query_count,
            request_count,
            sequence_slot_count,
            slot_count: pool_layout.slot_count(),
            query_head_count,
            kv_head_count: runtime.kv_heads,
            query_key_head_dim: key_row_bytes / key_denominator,
            value_head_dim: value_row_bytes / value_denominator,
            scale,
            layout: attention_layout,
        },
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
