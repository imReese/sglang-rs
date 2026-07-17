use std::fmt;
use std::ops::Range;

use sglang_kernel::cuda::{CudaContext, CudaDeviceAllocation, CudaError};
use sglang_kernel::cuda_kv_kernels::{
    CudaKvPairCopyError, CudaKvPairCopyKernels, CudaKvPairCopyLayout, CudaKvPairCopyPlan,
    CudaKvPairGather, CudaKvPairScatter,
};

use crate::transfer::{
    KvCacheMemoryLocation, KvCacheMemoryProvider, KvCacheRuntimeLayout, KvCacheTransferError,
    TransferableKvCacheMemory, TransferableKvCacheRegion,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CudaKvCachePoolError {
    ZeroPageCount,
    ZeroLayoutField(&'static str),
    RuntimePageSizeMismatch {
        expected: usize,
        actual: usize,
    },
    UnevenLayerLayout {
        bytes_per_token: usize,
        num_layers: usize,
    },
    UnevenTensorLayout {
        bytes_per_token_per_layer: usize,
        kv_tensors_per_token: usize,
    },
    SizeOverflow,
    PageOutOfRange {
        page_index: usize,
        page_count: usize,
    },
    LayerOutOfRange {
        layer_index: usize,
        layer_count: usize,
    },
    TensorOutOfRange {
        tensor_index: usize,
        tensor_count: usize,
    },
    TokenOutOfRange {
        token_index: usize,
        page_size: usize,
    },
    SlotOutOfRange {
        slot_index: usize,
        slot_count: usize,
    },
    EmptySlotMap,
    BatchSlotOutOfRange {
        batch_index: usize,
        slot_index: usize,
        slot_count: usize,
    },
    SlotMapCapacityMismatch {
        map_slot_count: usize,
        pool_slot_count: usize,
    },
    KvPairRequiresTwoTensors {
        tensor_count: usize,
    },
    PageBufferSizeMismatch {
        expected: usize,
        actual: usize,
    },
    Cuda(CudaError),
    CudaKvCopy(CudaKvPairCopyError),
    Transfer(KvCacheTransferError),
}

impl fmt::Display for CudaKvCachePoolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroPageCount => formatter.write_str("CUDA KV cache page count must be non-zero"),
            Self::ZeroLayoutField(field) => {
                write!(
                    formatter,
                    "CUDA KV cache layout field {field} must be non-zero"
                )
            }
            Self::RuntimePageSizeMismatch { expected, actual } => write!(
                formatter,
                "CUDA KV cache runtime page size is {actual} bytes but page_size * bytes_per_token requires {expected} bytes"
            ),
            Self::UnevenLayerLayout {
                bytes_per_token,
                num_layers,
            } => write!(
                formatter,
                "CUDA KV cache token size {bytes_per_token} bytes is not divisible by {num_layers} layers"
            ),
            Self::UnevenTensorLayout {
                bytes_per_token_per_layer,
                kv_tensors_per_token,
            } => write!(
                formatter,
                "CUDA KV cache per-layer token size {bytes_per_token_per_layer} bytes is not divisible by {kv_tensors_per_token} KV tensors"
            ),
            Self::SizeOverflow => formatter.write_str("CUDA KV cache layout size overflowed"),
            Self::PageOutOfRange {
                page_index,
                page_count,
            } => write!(
                formatter,
                "CUDA KV cache page index {page_index} is outside {page_count} pages"
            ),
            Self::LayerOutOfRange {
                layer_index,
                layer_count,
            } => write!(
                formatter,
                "CUDA KV cache layer index {layer_index} is outside {layer_count} layers"
            ),
            Self::TensorOutOfRange {
                tensor_index,
                tensor_count,
            } => write!(
                formatter,
                "CUDA KV cache tensor index {tensor_index} is outside {tensor_count} tensors per token"
            ),
            Self::TokenOutOfRange {
                token_index,
                page_size,
            } => write!(
                formatter,
                "CUDA KV cache token index {token_index} is outside page size {page_size}"
            ),
            Self::SlotOutOfRange {
                slot_index,
                slot_count,
            } => write!(
                formatter,
                "CUDA KV cache slot index {slot_index} is outside {slot_count} token slots"
            ),
            Self::EmptySlotMap => formatter.write_str("CUDA KV cache slot map must not be empty"),
            Self::BatchSlotOutOfRange {
                batch_index,
                slot_index,
                slot_count,
            } => write!(
                formatter,
                "CUDA KV cache batch slot {batch_index} references physical slot {slot_index}, outside {slot_count} token slots"
            ),
            Self::SlotMapCapacityMismatch {
                map_slot_count,
                pool_slot_count,
            } => write!(
                formatter,
                "CUDA KV cache slot map was validated for {map_slot_count} slots but the pool contains {pool_slot_count} slots"
            ),
            Self::KvPairRequiresTwoTensors { tensor_count } => write!(
                formatter,
                "CUDA KV cache K/V pair copy requires at least two tensors per token, layout has {tensor_count}"
            ),
            Self::PageBufferSizeMismatch { expected, actual } => write!(
                formatter,
                "CUDA KV cache page buffer has {actual} bytes but requires exactly {expected} bytes"
            ),
            Self::Cuda(error) => write!(formatter, "CUDA KV cache operation failed: {error}"),
            Self::CudaKvCopy(error) => {
                write!(formatter, "CUDA KV cache device copy failed: {error}")
            }
            Self::Transfer(error) => {
                write!(formatter, "CUDA KV cache transfer layout failed: {error}")
            }
        }
    }
}

impl std::error::Error for CudaKvCachePoolError {}

impl From<CudaError> for CudaKvCachePoolError {
    fn from(value: CudaError) -> Self {
        Self::Cuda(value)
    }
}

impl From<CudaKvPairCopyError> for CudaKvCachePoolError {
    fn from(value: CudaKvPairCopyError) -> Self {
        Self::CudaKvCopy(value)
    }
}

impl From<KvCacheTransferError> for CudaKvCachePoolError {
    fn from(value: KvCacheTransferError) -> Self {
        Self::Transfer(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaKvCacheTensorLocation {
    pub device_ptr: u64,
    pub byte_offset: usize,
    pub byte_len: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaKvCachePoolLayout {
    runtime: KvCacheRuntimeLayout,
    page_count: usize,
    bytes_per_token_per_layer: usize,
    bytes_per_token_per_tensor: usize,
    bytes_per_layer_page: usize,
    bytes_per_tensor_page: usize,
    total_byte_len: usize,
    slot_count: usize,
}

impl CudaKvCachePoolLayout {
    pub fn new(
        runtime: KvCacheRuntimeLayout,
        page_count: usize,
    ) -> Result<Self, CudaKvCachePoolError> {
        if page_count == 0 {
            return Err(CudaKvCachePoolError::ZeroPageCount);
        }
        for (field, value) in [
            ("page_size", runtime.page_size),
            ("num_layers", runtime.num_layers),
            ("kv_heads", runtime.kv_heads),
            ("head_dim", runtime.head_dim),
            ("kv_tensors_per_token", runtime.kv_tensors_per_token),
            ("bytes_per_token", runtime.bytes_per_token),
            ("page_size_bytes", runtime.page_size_bytes),
        ] {
            if value == 0 {
                return Err(CudaKvCachePoolError::ZeroLayoutField(field));
            }
        }

        let expected_page_size_bytes = runtime
            .page_size
            .checked_mul(runtime.bytes_per_token)
            .ok_or(CudaKvCachePoolError::SizeOverflow)?;
        if runtime.page_size_bytes != expected_page_size_bytes {
            return Err(CudaKvCachePoolError::RuntimePageSizeMismatch {
                expected: expected_page_size_bytes,
                actual: runtime.page_size_bytes,
            });
        }
        if !runtime.bytes_per_token.is_multiple_of(runtime.num_layers) {
            return Err(CudaKvCachePoolError::UnevenLayerLayout {
                bytes_per_token: runtime.bytes_per_token,
                num_layers: runtime.num_layers,
            });
        }
        let bytes_per_token_per_layer = runtime.bytes_per_token / runtime.num_layers;
        if !bytes_per_token_per_layer.is_multiple_of(runtime.kv_tensors_per_token) {
            return Err(CudaKvCachePoolError::UnevenTensorLayout {
                bytes_per_token_per_layer,
                kv_tensors_per_token: runtime.kv_tensors_per_token,
            });
        }
        let bytes_per_token_per_tensor = bytes_per_token_per_layer / runtime.kv_tensors_per_token;
        let bytes_per_layer_page = runtime
            .page_size
            .checked_mul(bytes_per_token_per_layer)
            .ok_or(CudaKvCachePoolError::SizeOverflow)?;
        let bytes_per_tensor_page = runtime
            .page_size
            .checked_mul(bytes_per_token_per_tensor)
            .ok_or(CudaKvCachePoolError::SizeOverflow)?;
        let total_byte_len = page_count
            .checked_mul(runtime.page_size_bytes)
            .ok_or(CudaKvCachePoolError::SizeOverflow)?;
        let slot_count = page_count
            .checked_mul(runtime.page_size)
            .ok_or(CudaKvCachePoolError::SizeOverflow)?;

        Ok(Self {
            runtime,
            page_count,
            bytes_per_token_per_layer,
            bytes_per_token_per_tensor,
            bytes_per_layer_page,
            bytes_per_tensor_page,
            total_byte_len,
            slot_count,
        })
    }

    pub fn runtime(&self) -> KvCacheRuntimeLayout {
        self.runtime
    }

    pub fn page_count(&self) -> usize {
        self.page_count
    }

    pub fn total_byte_len(&self) -> usize {
        self.total_byte_len
    }

    pub fn slot_count(&self) -> usize {
        self.slot_count
    }

    pub fn bytes_per_token_per_layer(&self) -> usize {
        self.bytes_per_token_per_layer
    }

    pub fn bytes_per_token_per_tensor(&self) -> usize {
        self.bytes_per_token_per_tensor
    }

    pub fn bytes_per_layer_page(&self) -> usize {
        self.bytes_per_layer_page
    }

    pub fn bytes_per_tensor_page(&self) -> usize {
        self.bytes_per_tensor_page
    }

    pub fn page_byte_range(&self, page_index: usize) -> Result<Range<usize>, CudaKvCachePoolError> {
        self.validate_page(page_index)?;
        let start = page_index
            .checked_mul(self.runtime.page_size_bytes)
            .ok_or(CudaKvCachePoolError::SizeOverflow)?;
        let end = start
            .checked_add(self.runtime.page_size_bytes)
            .ok_or(CudaKvCachePoolError::SizeOverflow)?;
        Ok(start..end)
    }

    pub fn tensor_token_byte_range(
        &self,
        page_index: usize,
        layer_index: usize,
        tensor_index: usize,
        token_index: usize,
    ) -> Result<Range<usize>, CudaKvCachePoolError> {
        self.validate_page(page_index)?;
        if layer_index >= self.runtime.num_layers {
            return Err(CudaKvCachePoolError::LayerOutOfRange {
                layer_index,
                layer_count: self.runtime.num_layers,
            });
        }
        if tensor_index >= self.runtime.kv_tensors_per_token {
            return Err(CudaKvCachePoolError::TensorOutOfRange {
                tensor_index,
                tensor_count: self.runtime.kv_tensors_per_token,
            });
        }
        if token_index >= self.runtime.page_size {
            return Err(CudaKvCachePoolError::TokenOutOfRange {
                token_index,
                page_size: self.runtime.page_size,
            });
        }

        let page_offset = page_index
            .checked_mul(self.runtime.page_size_bytes)
            .ok_or(CudaKvCachePoolError::SizeOverflow)?;
        let layer_offset = layer_index
            .checked_mul(self.bytes_per_layer_page)
            .ok_or(CudaKvCachePoolError::SizeOverflow)?;
        let tensor_offset = tensor_index
            .checked_mul(self.bytes_per_tensor_page)
            .ok_or(CudaKvCachePoolError::SizeOverflow)?;
        let token_offset = token_index
            .checked_mul(self.bytes_per_token_per_tensor)
            .ok_or(CudaKvCachePoolError::SizeOverflow)?;
        let start = page_offset
            .checked_add(layer_offset)
            .and_then(|offset| offset.checked_add(tensor_offset))
            .and_then(|offset| offset.checked_add(token_offset))
            .ok_or(CudaKvCachePoolError::SizeOverflow)?;
        let end = start
            .checked_add(self.bytes_per_token_per_tensor)
            .ok_or(CudaKvCachePoolError::SizeOverflow)?;
        Ok(start..end)
    }

    pub fn tensor_slot_byte_range(
        &self,
        layer_index: usize,
        tensor_index: usize,
        slot_index: usize,
    ) -> Result<Range<usize>, CudaKvCachePoolError> {
        if slot_index >= self.slot_count {
            return Err(CudaKvCachePoolError::SlotOutOfRange {
                slot_index,
                slot_count: self.slot_count,
            });
        }
        self.tensor_token_byte_range(
            slot_index / self.runtime.page_size,
            layer_index,
            tensor_index,
            slot_index % self.runtime.page_size,
        )
    }

    pub fn validate_slot_indices(&self, slots: &[usize]) -> Result<(), CudaKvCachePoolError> {
        if slots.is_empty() {
            return Err(CudaKvCachePoolError::EmptySlotMap);
        }
        for (batch_index, slot_index) in slots.iter().copied().enumerate() {
            if slot_index >= self.slot_count {
                return Err(CudaKvCachePoolError::BatchSlotOutOfRange {
                    batch_index,
                    slot_index,
                    slot_count: self.slot_count,
                });
            }
        }
        Ok(())
    }

    pub fn kv_pair_copy_plan(
        &self,
        layer_index: usize,
        row_count: usize,
        key_row_stride_bytes: usize,
        value_row_stride_bytes: usize,
    ) -> Result<CudaKvPairCopyPlan, CudaKvCachePoolError> {
        if layer_index >= self.runtime.num_layers {
            return Err(CudaKvCachePoolError::LayerOutOfRange {
                layer_index,
                layer_count: self.runtime.num_layers,
            });
        }
        if self.runtime.kv_tensors_per_token < 2 {
            return Err(CudaKvCachePoolError::KvPairRequiresTwoTensors {
                tensor_count: self.runtime.kv_tensors_per_token,
            });
        }
        let layer_offset = layer_index
            .checked_mul(self.bytes_per_layer_page)
            .ok_or(CudaKvCachePoolError::SizeOverflow)?;
        let value_offset = layer_offset
            .checked_add(self.bytes_per_tensor_page)
            .ok_or(CudaKvCachePoolError::SizeOverflow)?;
        let copy_layout = CudaKvPairCopyLayout::new(
            self.runtime.page_size,
            self.runtime.page_size_bytes,
            layer_offset,
            value_offset,
            self.bytes_per_token_per_tensor,
        );
        Ok(CudaKvPairCopyPlan::new(
            row_count,
            self.slot_count,
            copy_layout,
            key_row_stride_bytes,
            value_row_stride_bytes,
        )?)
    }

    fn validate_page(&self, page_index: usize) -> Result<(), CudaKvCachePoolError> {
        if page_index >= self.page_count {
            Err(CudaKvCachePoolError::PageOutOfRange {
                page_index,
                page_count: self.page_count,
            })
        } else {
            Ok(())
        }
    }
}

pub struct CudaKvCacheSlotMap {
    allocation: CudaDeviceAllocation,
    slots: Vec<usize>,
    slot_count: usize,
}

impl fmt::Debug for CudaKvCacheSlotMap {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaKvCacheSlotMap")
            .field("device_ordinal", &self.allocation.device_ordinal())
            .field("slots", &self.slots)
            .field("slot_count", &self.slot_count)
            .finish()
    }
}

impl CudaKvCacheSlotMap {
    pub fn slots(&self) -> &[usize] {
        &self.slots
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn allocation(&self) -> &CudaDeviceAllocation {
        &self.allocation
    }
}

pub struct CudaKvSlotScatterLaunch<'a> {
    pub kernels: &'a mut CudaKvPairCopyKernels,
    pub layer_index: usize,
    pub slot_map: &'a CudaKvCacheSlotMap,
    pub keys: &'a CudaDeviceAllocation,
    pub keys_offset: usize,
    pub key_row_stride_bytes: usize,
    pub values: &'a CudaDeviceAllocation,
    pub values_offset: usize,
    pub value_row_stride_bytes: usize,
}

pub struct CudaKvSlotGatherLaunch<'a> {
    pub kernels: &'a mut CudaKvPairCopyKernels,
    pub layer_index: usize,
    pub slot_map: &'a CudaKvCacheSlotMap,
    pub keys: &'a mut CudaDeviceAllocation,
    pub keys_offset: usize,
    pub key_row_stride_bytes: usize,
    pub values: &'a mut CudaDeviceAllocation,
    pub values_offset: usize,
    pub value_row_stride_bytes: usize,
}

pub struct CudaKvCachePool {
    context: CudaContext,
    layout: CudaKvCachePoolLayout,
    allocation: CudaDeviceAllocation,
}

impl CudaKvCachePool {
    pub fn allocate(
        context: &CudaContext,
        runtime: KvCacheRuntimeLayout,
        page_count: usize,
    ) -> Result<Self, CudaKvCachePoolError> {
        let layout = CudaKvCachePoolLayout::new(runtime, page_count)?;
        let mut allocation = context.allocate(layout.total_byte_len())?;
        allocation.fill(0)?;
        Ok(Self {
            context: context.clone(),
            layout,
            allocation,
        })
    }

    pub fn layout(&self) -> CudaKvCachePoolLayout {
        self.layout
    }

    pub fn allocation(&self) -> &CudaDeviceAllocation {
        &self.allocation
    }

    pub fn allocation_mut(&mut self) -> &mut CudaDeviceAllocation {
        &mut self.allocation
    }

    pub fn clear(&mut self) -> Result<(), CudaKvCachePoolError> {
        self.allocation.fill(0)?;
        Ok(())
    }

    pub fn upload_slot_map(
        &self,
        slots: &[usize],
    ) -> Result<CudaKvCacheSlotMap, CudaKvCachePoolError> {
        self.layout.validate_slot_indices(slots)?;
        let byte_len = slots
            .len()
            .checked_mul(size_of::<u64>())
            .ok_or(CudaKvCachePoolError::SizeOverflow)?;
        let mut bytes = Vec::with_capacity(byte_len);
        for slot in slots {
            let slot = u64::try_from(*slot).map_err(|_| CudaKvCachePoolError::SizeOverflow)?;
            bytes.extend_from_slice(&slot.to_ne_bytes());
        }
        let mut allocation = self.context.allocate(byte_len)?;
        allocation.copy_from_host(0, &bytes)?;
        Ok(CudaKvCacheSlotMap {
            allocation,
            slots: slots.to_vec(),
            slot_count: self.layout.slot_count,
        })
    }

    pub fn write_page(
        &mut self,
        page_index: usize,
        bytes: &[u8],
    ) -> Result<(), CudaKvCachePoolError> {
        if bytes.len() != self.layout.runtime.page_size_bytes {
            return Err(CudaKvCachePoolError::PageBufferSizeMismatch {
                expected: self.layout.runtime.page_size_bytes,
                actual: bytes.len(),
            });
        }
        let range = self.layout.page_byte_range(page_index)?;
        self.allocation.copy_from_host(range.start, bytes)?;
        Ok(())
    }

    pub fn read_page(
        &self,
        page_index: usize,
        bytes: &mut [u8],
    ) -> Result<(), CudaKvCachePoolError> {
        if bytes.len() != self.layout.runtime.page_size_bytes {
            return Err(CudaKvCachePoolError::PageBufferSizeMismatch {
                expected: self.layout.runtime.page_size_bytes,
                actual: bytes.len(),
            });
        }
        let range = self.layout.page_byte_range(page_index)?;
        self.allocation.copy_to_host(range.start, bytes)?;
        Ok(())
    }

    pub fn write_tensor_slot_from_device(
        &mut self,
        layer_index: usize,
        tensor_index: usize,
        slot_index: usize,
        source: &CudaDeviceAllocation,
        source_offset: usize,
    ) -> Result<(), CudaKvCachePoolError> {
        let range = self
            .layout
            .tensor_slot_byte_range(layer_index, tensor_index, slot_index)?;
        self.allocation
            .copy_from_device(range.start, source, source_offset, range.len())?;
        Ok(())
    }

    pub fn read_tensor_slot_to_device(
        &self,
        layer_index: usize,
        tensor_index: usize,
        slot_index: usize,
        destination: &mut CudaDeviceAllocation,
        destination_offset: usize,
    ) -> Result<(), CudaKvCachePoolError> {
        let range = self
            .layout
            .tensor_slot_byte_range(layer_index, tensor_index, slot_index)?;
        destination.copy_from_device(
            destination_offset,
            &self.allocation,
            range.start,
            range.len(),
        )?;
        Ok(())
    }

    pub fn write_kv_slots_from_device(
        &mut self,
        launch: CudaKvSlotScatterLaunch<'_>,
    ) -> Result<(), CudaKvCachePoolError> {
        let CudaKvSlotScatterLaunch {
            kernels,
            layer_index,
            slot_map,
            keys,
            keys_offset,
            key_row_stride_bytes,
            values,
            values_offset,
            value_row_stride_bytes,
        } = launch;
        self.validate_slot_map(slot_map)?;
        let plan = self.layout.kv_pair_copy_plan(
            layer_index,
            slot_map.len(),
            key_row_stride_bytes,
            value_row_stride_bytes,
        )?;
        kernels.scatter(
            plan,
            CudaKvPairScatter {
                keys,
                keys_offset,
                values,
                values_offset,
                slot_indices: &slot_map.allocation,
                slot_indices_offset: 0,
                pool: &mut self.allocation,
                pool_offset: 0,
            },
        )?;
        Ok(())
    }

    pub fn read_kv_slots_to_device(
        &self,
        launch: CudaKvSlotGatherLaunch<'_>,
    ) -> Result<(), CudaKvCachePoolError> {
        let CudaKvSlotGatherLaunch {
            kernels,
            layer_index,
            slot_map,
            keys,
            keys_offset,
            key_row_stride_bytes,
            values,
            values_offset,
            value_row_stride_bytes,
        } = launch;
        self.validate_slot_map(slot_map)?;
        let plan = self.layout.kv_pair_copy_plan(
            layer_index,
            slot_map.len(),
            key_row_stride_bytes,
            value_row_stride_bytes,
        )?;
        kernels.gather(
            plan,
            CudaKvPairGather {
                pool: &self.allocation,
                pool_offset: 0,
                slot_indices: &slot_map.allocation,
                slot_indices_offset: 0,
                keys,
                keys_offset,
                values,
                values_offset,
            },
        )?;
        Ok(())
    }

    pub fn tensor_location(
        &self,
        page_index: usize,
        layer_index: usize,
        tensor_index: usize,
        token_index: usize,
    ) -> Result<CudaKvCacheTensorLocation, CudaKvCachePoolError> {
        let range = self.layout.tensor_token_byte_range(
            page_index,
            layer_index,
            tensor_index,
            token_index,
        )?;
        let device_ptr = self.allocation.device_ptr_at(range.start, range.len())?;
        Ok(CudaKvCacheTensorLocation {
            device_ptr,
            byte_offset: range.start,
            byte_len: range.len(),
        })
    }

    pub fn slot_location(
        &self,
        layer_index: usize,
        tensor_index: usize,
        slot_index: usize,
    ) -> Result<CudaKvCacheTensorLocation, CudaKvCachePoolError> {
        let range = self
            .layout
            .tensor_slot_byte_range(layer_index, tensor_index, slot_index)?;
        let device_ptr = self.allocation.device_ptr_at(range.start, range.len())?;
        Ok(CudaKvCacheTensorLocation {
            device_ptr,
            byte_offset: range.start,
            byte_len: range.len(),
        })
    }

    pub fn transferable_memory(&self) -> Result<TransferableKvCacheMemory, CudaKvCachePoolError> {
        let device_ptr = self
            .allocation
            .device_ptr_at(0, self.layout.total_byte_len())?;
        let base_addr =
            usize::try_from(device_ptr).map_err(|_| CudaKvCachePoolError::SizeOverflow)?;
        Ok(TransferableKvCacheMemory::new(
            vec![TransferableKvCacheRegion {
                base_addr,
                byte_len: self.allocation.byte_len(),
                page_size_bytes: self.layout.runtime.page_size_bytes,
            }],
            self.layout.runtime.page_size_bytes,
            KvCacheMemoryLocation::Cuda {
                device_id: self.allocation.device_ordinal(),
            },
        )
        .map_err(KvCacheTransferError::from)?)
    }

    fn validate_slot_map(&self, slot_map: &CudaKvCacheSlotMap) -> Result<(), CudaKvCachePoolError> {
        if slot_map.slot_count != self.layout.slot_count {
            return Err(CudaKvCachePoolError::SlotMapCapacityMismatch {
                map_slot_count: slot_map.slot_count,
                pool_slot_count: self.layout.slot_count,
            });
        }
        self.layout.validate_slot_indices(&slot_map.slots)
    }
}

impl KvCacheMemoryProvider for CudaKvCachePool {
    type Error = KvCacheTransferError;

    fn transferable_kv_cache_memory(&self) -> Result<TransferableKvCacheMemory, Self::Error> {
        self.transferable_memory().map_err(|error| {
            KvCacheTransferError::Runtime(format!(
                "CUDA KV cache pool is not transferable: {error}"
            ))
        })
    }
}
