use std::fmt;

use sglang_kernel::cuda::{CudaContext, CudaDeviceAllocation, CudaError};
use sglang_kernel::cuda_kv_kernels::{
    CudaKvPairCopyError, CudaKvPairCopyKernels, CudaKvPairCopyLayout, CudaKvPairCopyPlan,
    CudaKvPairGather, CudaKvPairScatter,
};

use crate::kv_cache::{
    KvCachePool, KvCachePoolError, KvCacheStorage, PagedKvCacheLayout, PagedKvCacheLayoutError,
};
use crate::transfer::{
    KvCacheMemoryLocation, KvCacheMemoryProvider, KvCacheRuntimeLayout, KvCacheTransferError,
    TransferableKvCacheMemory, TransferableKvCacheRegion,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CudaKvStorageError {
    Layout(PagedKvCacheLayoutError),
    Pool(KvCachePoolError),
    StorageSizeMismatch {
        expected: usize,
        actual: usize,
    },
    SlotMapCapacityMismatch {
        map_slot_count: usize,
        layout_slot_count: usize,
    },
    PageBufferSizeMismatch {
        expected: usize,
        actual: usize,
    },
    SizeOverflow,
    Cuda(CudaError),
    CudaKvCopy(CudaKvPairCopyError),
    Transfer(KvCacheTransferError),
}

impl fmt::Display for CudaKvStorageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Layout(error) => write!(formatter, "invalid CUDA KV storage layout: {error}"),
            Self::Pool(error) => write!(formatter, "invalid CUDA KV cache pool: {error}"),
            Self::StorageSizeMismatch { expected, actual } => write!(
                formatter,
                "CUDA KV storage has {actual} bytes but layout requires exactly {expected} bytes"
            ),
            Self::SlotMapCapacityMismatch {
                map_slot_count,
                layout_slot_count,
            } => write!(
                formatter,
                "CUDA KV slot map was validated for {map_slot_count} slots but layout contains {layout_slot_count} slots"
            ),
            Self::PageBufferSizeMismatch { expected, actual } => write!(
                formatter,
                "CUDA KV page buffer has {actual} bytes but requires exactly {expected} bytes"
            ),
            Self::SizeOverflow => formatter.write_str("CUDA KV storage size overflowed"),
            Self::Cuda(error) => write!(formatter, "CUDA KV storage operation failed: {error}"),
            Self::CudaKvCopy(error) => write!(formatter, "CUDA KV device copy failed: {error}"),
            Self::Transfer(error) => {
                write!(formatter, "CUDA KV transfer layout failed: {error}")
            }
        }
    }
}

impl std::error::Error for CudaKvStorageError {}

impl From<PagedKvCacheLayoutError> for CudaKvStorageError {
    fn from(value: PagedKvCacheLayoutError) -> Self {
        Self::Layout(value)
    }
}

impl From<KvCachePoolError> for CudaKvStorageError {
    fn from(value: KvCachePoolError) -> Self {
        Self::Pool(value)
    }
}

impl From<CudaError> for CudaKvStorageError {
    fn from(value: CudaError) -> Self {
        Self::Cuda(value)
    }
}

impl From<CudaKvPairCopyError> for CudaKvStorageError {
    fn from(value: CudaKvPairCopyError) -> Self {
        Self::CudaKvCopy(value)
    }
}

impl From<KvCacheTransferError> for CudaKvStorageError {
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

pub struct CudaKvStorage {
    context: CudaContext,
    allocation: CudaDeviceAllocation,
}

impl CudaKvStorage {
    pub fn allocate(context: &CudaContext, byte_len: usize) -> Result<Self, CudaKvStorageError> {
        let mut allocation = context.allocate(byte_len)?;
        allocation.fill(0)?;
        Ok(Self {
            context: context.clone(),
            allocation,
        })
    }

    pub fn allocation(&self) -> &CudaDeviceAllocation {
        &self.allocation
    }

    pub fn allocation_mut(&mut self) -> &mut CudaDeviceAllocation {
        &mut self.allocation
    }

    pub fn upload_slot_map(
        &self,
        layout: PagedKvCacheLayout,
        slots: &[usize],
    ) -> Result<CudaKvCacheSlotMap, CudaKvStorageError> {
        self.validate_layout(layout)?;
        layout.validate_slot_indices(slots)?;
        let byte_len = slots
            .len()
            .checked_mul(size_of::<u64>())
            .ok_or(CudaKvStorageError::SizeOverflow)?;
        let mut bytes = Vec::with_capacity(byte_len);
        for slot in slots {
            let slot = u64::try_from(*slot).map_err(|_| CudaKvStorageError::SizeOverflow)?;
            bytes.extend_from_slice(&slot.to_ne_bytes());
        }
        let mut allocation = self.context.allocate(byte_len)?;
        allocation.copy_from_host(0, &bytes)?;
        Ok(CudaKvCacheSlotMap {
            allocation,
            slots: slots.to_vec(),
            slot_count: layout.slot_count(),
        })
    }

    pub fn write_page(
        &mut self,
        layout: PagedKvCacheLayout,
        page_index: usize,
        bytes: &[u8],
    ) -> Result<(), CudaKvStorageError> {
        self.validate_layout(layout)?;
        if bytes.len() != layout.runtime().page_size_bytes {
            return Err(CudaKvStorageError::PageBufferSizeMismatch {
                expected: layout.runtime().page_size_bytes,
                actual: bytes.len(),
            });
        }
        let range = layout.page_byte_range(page_index)?;
        self.allocation.copy_from_host(range.start, bytes)?;
        Ok(())
    }

    pub fn read_page(
        &self,
        layout: PagedKvCacheLayout,
        page_index: usize,
        bytes: &mut [u8],
    ) -> Result<(), CudaKvStorageError> {
        self.validate_layout(layout)?;
        if bytes.len() != layout.runtime().page_size_bytes {
            return Err(CudaKvStorageError::PageBufferSizeMismatch {
                expected: layout.runtime().page_size_bytes,
                actual: bytes.len(),
            });
        }
        let range = layout.page_byte_range(page_index)?;
        self.allocation.copy_to_host(range.start, bytes)?;
        Ok(())
    }

    pub fn write_tensor_slot_from_device(
        &mut self,
        layout: PagedKvCacheLayout,
        layer_index: usize,
        tensor_index: usize,
        slot_index: usize,
        source: &CudaDeviceAllocation,
        source_offset: usize,
    ) -> Result<(), CudaKvStorageError> {
        self.validate_layout(layout)?;
        let range = layout.tensor_slot_byte_range(layer_index, tensor_index, slot_index)?;
        self.allocation
            .copy_from_device(range.start, source, source_offset, range.len())?;
        Ok(())
    }

    pub fn read_tensor_slot_to_device(
        &self,
        layout: PagedKvCacheLayout,
        layer_index: usize,
        tensor_index: usize,
        slot_index: usize,
        destination: &mut CudaDeviceAllocation,
        destination_offset: usize,
    ) -> Result<(), CudaKvStorageError> {
        self.validate_layout(layout)?;
        let range = layout.tensor_slot_byte_range(layer_index, tensor_index, slot_index)?;
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
        layout: PagedKvCacheLayout,
        launch: CudaKvSlotScatterLaunch<'_>,
    ) -> Result<(), CudaKvStorageError> {
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
        self.validate_slot_map(layout, slot_map)?;
        let plan = kv_pair_copy_plan(
            layout,
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
        layout: PagedKvCacheLayout,
        launch: CudaKvSlotGatherLaunch<'_>,
    ) -> Result<(), CudaKvStorageError> {
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
        self.validate_slot_map(layout, slot_map)?;
        let plan = kv_pair_copy_plan(
            layout,
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
        layout: PagedKvCacheLayout,
        page_index: usize,
        layer_index: usize,
        tensor_index: usize,
        token_index: usize,
    ) -> Result<CudaKvCacheTensorLocation, CudaKvStorageError> {
        self.validate_layout(layout)?;
        let range =
            layout.tensor_token_byte_range(page_index, layer_index, tensor_index, token_index)?;
        let device_ptr = self.allocation.device_ptr_at(range.start, range.len())?;
        Ok(CudaKvCacheTensorLocation {
            device_ptr,
            byte_offset: range.start,
            byte_len: range.len(),
        })
    }

    pub fn slot_location(
        &self,
        layout: PagedKvCacheLayout,
        layer_index: usize,
        tensor_index: usize,
        slot_index: usize,
    ) -> Result<CudaKvCacheTensorLocation, CudaKvStorageError> {
        self.validate_layout(layout)?;
        let range = layout.tensor_slot_byte_range(layer_index, tensor_index, slot_index)?;
        let device_ptr = self.allocation.device_ptr_at(range.start, range.len())?;
        Ok(CudaKvCacheTensorLocation {
            device_ptr,
            byte_offset: range.start,
            byte_len: range.len(),
        })
    }

    pub fn transferable_memory(
        &self,
        layout: PagedKvCacheLayout,
    ) -> Result<TransferableKvCacheMemory, CudaKvStorageError> {
        self.validate_layout(layout)?;
        let device_ptr = self.allocation.device_ptr_at(0, layout.total_byte_len())?;
        let base_addr =
            usize::try_from(device_ptr).map_err(|_| CudaKvStorageError::SizeOverflow)?;
        Ok(TransferableKvCacheMemory::new(
            vec![TransferableKvCacheRegion {
                base_addr,
                byte_len: self.allocation.byte_len(),
                page_size_bytes: layout.runtime().page_size_bytes,
            }],
            layout.runtime().page_size_bytes,
            KvCacheMemoryLocation::Cuda {
                device_id: self.allocation.device_ordinal(),
            },
        )
        .map_err(KvCacheTransferError::from)?)
    }

    pub fn validate_layout(&self, layout: PagedKvCacheLayout) -> Result<(), CudaKvStorageError> {
        let actual = self.allocation.byte_len();
        let expected = layout.total_byte_len();
        if actual != expected {
            return Err(CudaKvStorageError::StorageSizeMismatch { expected, actual });
        }
        Ok(())
    }

    fn validate_slot_map(
        &self,
        layout: PagedKvCacheLayout,
        slot_map: &CudaKvCacheSlotMap,
    ) -> Result<(), CudaKvStorageError> {
        self.validate_layout(layout)?;
        if slot_map.slot_count != layout.slot_count() {
            return Err(CudaKvStorageError::SlotMapCapacityMismatch {
                map_slot_count: slot_map.slot_count,
                layout_slot_count: layout.slot_count(),
            });
        }
        layout.validate_slot_indices(&slot_map.slots)?;
        Ok(())
    }
}

impl KvCacheStorage for CudaKvStorage {
    type Error = CudaKvStorageError;

    fn byte_len(&self) -> usize {
        self.allocation.byte_len()
    }

    fn clear(&mut self) -> Result<(), Self::Error> {
        self.allocation.fill(0)?;
        Ok(())
    }
}

pub fn allocate_cuda_kv_cache(
    context: &CudaContext,
    runtime: KvCacheRuntimeLayout,
    page_count: usize,
) -> Result<KvCachePool<CudaKvStorage>, CudaKvStorageError> {
    let layout = PagedKvCacheLayout::new(runtime, page_count)?;
    let storage = CudaKvStorage::allocate(context, layout.total_byte_len())?;
    Ok(KvCachePool::new(layout, storage)?)
}

impl KvCacheMemoryProvider for KvCachePool<CudaKvStorage> {
    type Error = KvCacheTransferError;

    fn transferable_kv_cache_memory(&self) -> Result<TransferableKvCacheMemory, Self::Error> {
        self.storage()
            .transferable_memory(self.layout())
            .map_err(|error| {
                KvCacheTransferError::Runtime(format!(
                    "CUDA KV storage is not transferable: {error}"
                ))
            })
    }
}

fn kv_pair_copy_plan(
    layout: PagedKvCacheLayout,
    layer_index: usize,
    row_count: usize,
    key_row_stride_bytes: usize,
    value_row_stride_bytes: usize,
) -> Result<CudaKvPairCopyPlan, CudaKvStorageError> {
    let geometry = layout.kv_pair_copy_geometry(layer_index)?;
    let copy_layout = CudaKvPairCopyLayout::new(
        geometry.page_size,
        geometry.page_stride_bytes,
        geometry.key_offset_bytes,
        geometry.value_offset_bytes,
        geometry.token_bytes,
    );
    Ok(CudaKvPairCopyPlan::new(
        row_count,
        layout.slot_count(),
        copy_layout,
        key_row_stride_bytes,
        value_row_stride_bytes,
    )?)
}
