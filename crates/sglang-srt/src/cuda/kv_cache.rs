use std::fmt;

use nexus_transfer::{
    KvCacheMemoryProvider, TransferableKvCacheMemory, TransferableKvCacheMemoryError,
};
use sglang_kernel::cuda::{CudaContext, CudaDeviceAllocation, CudaError};
use sglang_kernel::cuda_kv_kernels::{
    CudaKvPairCopyError, CudaKvPairCopyKernels, CudaKvPairCopyLayout, CudaKvPairCopyPlan,
    CudaKvPairGather, CudaKvPairScatter,
};

use crate::device_storage::{
    DeviceSlotMap, DeviceTensorLocation, PagedDeviceStorage, PagedStorageError,
};
use crate::kv_cache::{
    KvCachePool, KvCachePoolError, KvCacheRuntimeLayout, KvCacheStorage, PagedKvCacheLayout,
    PagedKvCacheLayoutError,
};
use crate::runtime_kv_cache::ActiveKvCache;
use crate::transfer::KvCacheTransferError;

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
    Descriptor(TransferableKvCacheMemoryError),
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
            Self::Descriptor(error) => {
                write!(formatter, "invalid NexusKV CUDA memory descriptor: {error}")
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

impl From<TransferableKvCacheMemoryError> for CudaKvStorageError {
    fn from(value: TransferableKvCacheMemoryError) -> Self {
        Self::Descriptor(value)
    }
}

impl From<PagedStorageError<CudaError>> for CudaKvStorageError {
    fn from(value: PagedStorageError<CudaError>) -> Self {
        match value {
            PagedStorageError::Layout(error) => Self::Layout(error),
            PagedStorageError::Pool(error) => Self::Pool(error),
            PagedStorageError::StorageSizeMismatch { expected, actual } => {
                Self::StorageSizeMismatch { expected, actual }
            }
            PagedStorageError::SlotMapCapacityMismatch {
                map_slot_count,
                layout_slot_count,
            } => Self::SlotMapCapacityMismatch {
                map_slot_count,
                layout_slot_count,
            },
            PagedStorageError::PageBufferSizeMismatch { expected, actual } => {
                Self::PageBufferSizeMismatch { expected, actual }
            }
            PagedStorageError::SizeOverflow => Self::SizeOverflow,
            PagedStorageError::Device(error) => Self::Cuda(error),
            PagedStorageError::Descriptor(error) => Self::Descriptor(error),
        }
    }
}

pub type CudaKvCacheTensorLocation = DeviceTensorLocation;

pub struct CudaKvCacheSlotMap {
    inner: DeviceSlotMap<CudaDeviceAllocation>,
}

impl fmt::Debug for CudaKvCacheSlotMap {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaKvCacheSlotMap")
            .field("device_ordinal", &self.inner.allocation().device_ordinal())
            .field("slots", &self.inner.slots())
            .field("slot_count", &self.inner.slot_count())
            .finish()
    }
}

impl CudaKvCacheSlotMap {
    pub fn slots(&self) -> &[usize] {
        self.inner.slots()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn allocation(&self) -> &CudaDeviceAllocation {
        self.inner.allocation()
    }

    pub fn slot_count(&self) -> usize {
        self.inner.slot_count()
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
    storage: PagedDeviceStorage<CudaDeviceAllocation>,
}

impl CudaKvStorage {
    pub fn allocate(
        context: &CudaContext,
        byte_len: usize,
        page_size_bytes: usize,
    ) -> Result<Self, CudaKvStorageError> {
        let mut allocation = context.allocate(byte_len)?;
        allocation.fill(0)?;
        let memory_location = nexus_transfer::KvCacheMemoryLocation::Cuda {
            device_id: allocation.device_ordinal(),
        };
        Ok(Self {
            context: context.clone(),
            storage: PagedDeviceStorage::new(allocation, page_size_bytes, memory_location),
        })
    }

    pub fn inner(&self) -> &PagedDeviceStorage<CudaDeviceAllocation> {
        &self.storage
    }

    pub fn inner_mut(&mut self) -> &mut PagedDeviceStorage<CudaDeviceAllocation> {
        &mut self.storage
    }

    pub fn allocation(&self) -> &CudaDeviceAllocation {
        self.storage.allocation()
    }

    pub fn allocation_mut(&mut self) -> &mut CudaDeviceAllocation {
        self.storage.allocation_mut()
    }

    pub fn upload_slot_map(
        &self,
        layout: PagedKvCacheLayout,
        slots: &[usize],
    ) -> Result<CudaKvCacheSlotMap, CudaKvStorageError> {
        let inner = self
            .storage
            .upload_slot_map(layout, slots, |byte_len| self.context.allocate(byte_len))?;
        Ok(CudaKvCacheSlotMap { inner })
    }

    pub fn write_page(
        &mut self,
        layout: PagedKvCacheLayout,
        page_index: usize,
        bytes: &[u8],
    ) -> Result<(), CudaKvStorageError> {
        self.storage
            .write_page(layout, page_index, bytes)
            .map_err(Into::into)
    }

    pub fn read_page(
        &self,
        layout: PagedKvCacheLayout,
        page_index: usize,
        bytes: &mut [u8],
    ) -> Result<(), CudaKvStorageError> {
        self.storage
            .read_page(layout, page_index, bytes)
            .map_err(Into::into)
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
        self.storage
            .write_tensor_slot_from_device(
                layout,
                layer_index,
                tensor_index,
                slot_index,
                source,
                source_offset,
            )
            .map_err(Into::into)
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
        self.storage
            .read_tensor_slot_to_device(
                layout,
                layer_index,
                tensor_index,
                slot_index,
                destination,
                destination_offset,
            )
            .map_err(Into::into)
    }

    pub fn tensor_location(
        &self,
        layout: PagedKvCacheLayout,
        page_index: usize,
        layer_index: usize,
        tensor_index: usize,
        token_index: usize,
    ) -> Result<CudaKvCacheTensorLocation, CudaKvStorageError> {
        self.storage
            .tensor_location(layout, page_index, layer_index, tensor_index, token_index)
            .map_err(Into::into)
    }

    pub fn slot_location(
        &self,
        layout: PagedKvCacheLayout,
        layer_index: usize,
        tensor_index: usize,
        slot_index: usize,
    ) -> Result<CudaKvCacheTensorLocation, CudaKvStorageError> {
        self.storage
            .slot_location(layout, layer_index, tensor_index, slot_index)
            .map_err(Into::into)
    }

    pub fn validate_layout(&self, layout: PagedKvCacheLayout) -> Result<(), CudaKvStorageError> {
        self.storage.validate_layout(layout).map_err(Into::into)
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
        self.storage.validate_slot_map(layout, &slot_map.inner)?;
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
                slot_indices: slot_map.allocation(),
                slot_indices_offset: 0,
                pool: self.storage.allocation_mut(),
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
        self.storage.validate_slot_map(layout, &slot_map.inner)?;
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
                pool: self.storage.allocation(),
                pool_offset: 0,
                slot_indices: slot_map.allocation(),
                slot_indices_offset: 0,
                keys,
                keys_offset,
                values,
                values_offset,
            },
        )?;
        Ok(())
    }
}

impl KvCacheStorage for CudaKvStorage {
    fn byte_len(&self) -> usize {
        self.storage.byte_len()
    }

    fn clear(&mut self) -> Result<(), Self::Error> {
        self.storage.clear().map_err(Into::into)
    }
}

impl KvCacheMemoryProvider for CudaKvStorage {
    type Error = CudaKvStorageError;

    fn transferable_kv_cache_memory(&self) -> Result<TransferableKvCacheMemory, Self::Error> {
        self.storage
            .transferable_kv_cache_memory()
            .map_err(Into::into)
    }
}

pub fn allocate_cuda_kv_cache(
    context: &CudaContext,
    runtime: KvCacheRuntimeLayout,
    page_count: usize,
) -> Result<KvCachePool<CudaKvStorage>, CudaKvStorageError> {
    let layout = PagedKvCacheLayout::new(runtime, page_count)?;
    let storage = CudaKvStorage::allocate(
        context,
        layout.total_byte_len(),
        layout.runtime().page_size_bytes,
    )?;
    Ok(KvCachePool::new(layout, storage)?)
}

impl ActiveKvCache for KvCachePool<CudaKvStorage> {
    fn backend(&self) -> crate::backend::RuntimeBackend {
        crate::backend::RuntimeBackend::Cuda
    }

    fn layout(&self) -> PagedKvCacheLayout {
        self.layout()
    }

    fn transferable_memory(&self) -> Result<TransferableKvCacheMemory, KvCacheTransferError> {
        self.transferable_kv_cache_memory().map_err(|error| {
            KvCacheTransferError::Runtime(format!(
                "CUDA runtime KV memory cannot be described by NexusKV: {error}"
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
