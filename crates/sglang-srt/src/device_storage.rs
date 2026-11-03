use std::fmt;

use nexus_transfer::{
    KvCacheMemoryLocation, KvCacheMemoryProvider, TransferableKvCacheMemory,
    TransferableKvCacheMemoryError, TransferableKvCacheRegion,
};
use sglang_kernel::cuda::{CudaDeviceAllocation, CudaError};

use crate::kv_cache::{
    KvCachePoolError, KvCacheStorage, PagedKvCacheLayout, PagedKvCacheLayoutError,
};

pub trait DeviceBuffer: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn device_ptr(&self) -> u64;
    fn byte_len(&self) -> usize;
    fn device_ordinal(&self) -> usize;
    fn device_ptr_at(&self, offset: usize, len: usize) -> Result<u64, Self::Error>;
    fn copy_from_host(&mut self, offset: usize, bytes: &[u8]) -> Result<(), Self::Error>;
    fn copy_to_host(&self, offset: usize, bytes: &mut [u8]) -> Result<(), Self::Error>;
    fn copy_from_device(
        &mut self,
        destination_offset: usize,
        source: &Self,
        source_offset: usize,
        byte_len: usize,
    ) -> Result<(), Self::Error>;
    fn fill(&mut self, value: u8) -> Result<(), Self::Error>;
}

impl DeviceBuffer for CudaDeviceAllocation {
    type Error = CudaError;

    fn device_ptr(&self) -> u64 {
        self.device_ptr()
    }

    fn byte_len(&self) -> usize {
        self.byte_len()
    }

    fn device_ordinal(&self) -> usize {
        self.device_ordinal()
    }

    fn device_ptr_at(&self, offset: usize, len: usize) -> Result<u64, Self::Error> {
        self.device_ptr_at(offset, len)
    }

    fn copy_from_host(&mut self, offset: usize, bytes: &[u8]) -> Result<(), Self::Error> {
        self.copy_from_host(offset, bytes)
    }

    fn copy_to_host(&self, offset: usize, bytes: &mut [u8]) -> Result<(), Self::Error> {
        self.copy_to_host(offset, bytes)
    }

    fn copy_from_device(
        &mut self,
        destination_offset: usize,
        source: &Self,
        source_offset: usize,
        byte_len: usize,
    ) -> Result<(), Self::Error> {
        self.copy_from_device(destination_offset, source, source_offset, byte_len)
    }

    fn fill(&mut self, value: u8) -> Result<(), Self::Error> {
        self.fill(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceTensorLocation {
    pub device_ptr: u64,
    pub byte_offset: usize,
    pub byte_len: usize,
}

pub struct DeviceSlotMap<B> {
    allocation: B,
    slots: Vec<usize>,
    slot_count: usize,
}

impl<B: fmt::Debug> fmt::Debug for DeviceSlotMap<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeviceSlotMap")
            .field("allocation", &self.allocation)
            .field("slots", &self.slots)
            .field("slot_count", &self.slot_count)
            .finish()
    }
}

impl<B> DeviceSlotMap<B> {
    pub fn new(allocation: B, slots: Vec<usize>, slot_count: usize) -> Self {
        Self {
            allocation,
            slots,
            slot_count,
        }
    }

    pub fn slots(&self) -> &[usize] {
        &self.slots
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn allocation(&self) -> &B {
        &self.allocation
    }

    pub fn slot_count(&self) -> usize {
        self.slot_count
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PagedStorageError<E> {
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
    Device(E),
    Descriptor(TransferableKvCacheMemoryError),
}

impl<E: fmt::Display> fmt::Display for PagedStorageError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Layout(error) => write!(formatter, "invalid paged KV storage layout: {error}"),
            Self::Pool(error) => write!(formatter, "invalid paged KV cache pool: {error}"),
            Self::StorageSizeMismatch { expected, actual } => write!(
                formatter,
                "paged KV storage has {actual} bytes but layout requires exactly {expected} bytes"
            ),
            Self::SlotMapCapacityMismatch {
                map_slot_count,
                layout_slot_count,
            } => write!(
                formatter,
                "paged KV slot map was validated for {map_slot_count} slots but layout contains {layout_slot_count} slots"
            ),
            Self::PageBufferSizeMismatch { expected, actual } => write!(
                formatter,
                "paged KV page buffer has {actual} bytes but requires exactly {expected} bytes"
            ),
            Self::SizeOverflow => formatter.write_str("paged KV storage size overflowed"),
            Self::Device(error) => write!(formatter, "paged KV storage operation failed: {error}"),
            Self::Descriptor(error) => {
                write!(formatter, "invalid NexusKV memory descriptor: {error}")
            }
        }
    }
}

impl<E: fmt::Debug + fmt::Display> std::error::Error for PagedStorageError<E> {}

impl<E> From<PagedKvCacheLayoutError> for PagedStorageError<E> {
    fn from(value: PagedKvCacheLayoutError) -> Self {
        Self::Layout(value)
    }
}

impl<E> From<KvCachePoolError> for PagedStorageError<E> {
    fn from(value: KvCachePoolError) -> Self {
        Self::Pool(value)
    }
}

impl<E> From<TransferableKvCacheMemoryError> for PagedStorageError<E> {
    fn from(value: TransferableKvCacheMemoryError) -> Self {
        Self::Descriptor(value)
    }
}

pub struct PagedDeviceStorage<B> {
    allocation: B,
    page_size_bytes: usize,
    memory_location: KvCacheMemoryLocation,
}

impl<B> PagedDeviceStorage<B> {
    pub fn new(
        allocation: B,
        page_size_bytes: usize,
        memory_location: KvCacheMemoryLocation,
    ) -> Self {
        Self {
            allocation,
            page_size_bytes,
            memory_location,
        }
    }

    pub fn allocation(&self) -> &B {
        &self.allocation
    }

    pub fn allocation_mut(&mut self) -> &mut B {
        &mut self.allocation
    }

    pub fn page_size_bytes(&self) -> usize {
        self.page_size_bytes
    }

    pub fn memory_location(&self) -> &KvCacheMemoryLocation {
        &self.memory_location
    }
}

impl<B: DeviceBuffer> PagedDeviceStorage<B> {
    pub fn validate_layout(
        &self,
        layout: PagedKvCacheLayout,
    ) -> Result<(), PagedStorageError<B::Error>> {
        let actual = self.allocation.byte_len();
        let expected = layout.total_byte_len();
        if actual != expected {
            return Err(PagedStorageError::StorageSizeMismatch { expected, actual });
        }
        if self.page_size_bytes != layout.runtime().page_size_bytes {
            return Err(PagedStorageError::PageBufferSizeMismatch {
                expected: layout.runtime().page_size_bytes,
                actual: self.page_size_bytes,
            });
        }
        Ok(())
    }

    pub fn validate_slot_map(
        &self,
        layout: PagedKvCacheLayout,
        slot_map: &DeviceSlotMap<B>,
    ) -> Result<(), PagedStorageError<B::Error>> {
        self.validate_layout(layout)?;
        if slot_map.slot_count != layout.slot_count() {
            return Err(PagedStorageError::SlotMapCapacityMismatch {
                map_slot_count: slot_map.slot_count,
                layout_slot_count: layout.slot_count(),
            });
        }
        layout.validate_slot_indices(&slot_map.slots)?;
        Ok(())
    }

    pub fn upload_slot_map<F>(
        &self,
        layout: PagedKvCacheLayout,
        slots: &[usize],
        allocate: F,
    ) -> Result<DeviceSlotMap<B>, PagedStorageError<B::Error>>
    where
        F: FnOnce(usize) -> Result<B, B::Error>,
    {
        self.validate_layout(layout)?;
        layout.validate_slot_indices(slots)?;
        let byte_len = slots
            .len()
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or(PagedStorageError::SizeOverflow)?;
        let mut bytes = Vec::with_capacity(byte_len);
        for slot in slots {
            let slot = u64::try_from(*slot).map_err(|_| PagedStorageError::SizeOverflow)?;
            bytes.extend_from_slice(&slot.to_ne_bytes());
        }
        let mut allocation = allocate(byte_len).map_err(PagedStorageError::Device)?;
        allocation
            .copy_from_host(0, &bytes)
            .map_err(PagedStorageError::Device)?;
        Ok(DeviceSlotMap::new(
            allocation,
            slots.to_vec(),
            layout.slot_count(),
        ))
    }

    pub fn write_page(
        &mut self,
        layout: PagedKvCacheLayout,
        page_index: usize,
        bytes: &[u8],
    ) -> Result<(), PagedStorageError<B::Error>> {
        self.validate_layout(layout)?;
        if bytes.len() != layout.runtime().page_size_bytes {
            return Err(PagedStorageError::PageBufferSizeMismatch {
                expected: layout.runtime().page_size_bytes,
                actual: bytes.len(),
            });
        }
        let range = layout.page_byte_range(page_index)?;
        self.allocation
            .copy_from_host(range.start, bytes)
            .map_err(PagedStorageError::Device)?;
        Ok(())
    }

    pub fn read_page(
        &self,
        layout: PagedKvCacheLayout,
        page_index: usize,
        bytes: &mut [u8],
    ) -> Result<(), PagedStorageError<B::Error>> {
        self.validate_layout(layout)?;
        if bytes.len() != layout.runtime().page_size_bytes {
            return Err(PagedStorageError::PageBufferSizeMismatch {
                expected: layout.runtime().page_size_bytes,
                actual: bytes.len(),
            });
        }
        let range = layout.page_byte_range(page_index)?;
        self.allocation
            .copy_to_host(range.start, bytes)
            .map_err(PagedStorageError::Device)?;
        Ok(())
    }

    pub fn write_tensor_slot_from_device(
        &mut self,
        layout: PagedKvCacheLayout,
        layer_index: usize,
        tensor_index: usize,
        slot_index: usize,
        source: &B,
        source_offset: usize,
    ) -> Result<(), PagedStorageError<B::Error>> {
        self.validate_layout(layout)?;
        let range = layout.tensor_slot_byte_range(layer_index, tensor_index, slot_index)?;
        self.allocation
            .copy_from_device(range.start, source, source_offset, range.len())
            .map_err(PagedStorageError::Device)?;
        Ok(())
    }

    pub fn read_tensor_slot_to_device(
        &self,
        layout: PagedKvCacheLayout,
        layer_index: usize,
        tensor_index: usize,
        slot_index: usize,
        destination: &mut B,
        destination_offset: usize,
    ) -> Result<(), PagedStorageError<B::Error>> {
        self.validate_layout(layout)?;
        let range = layout.tensor_slot_byte_range(layer_index, tensor_index, slot_index)?;
        destination
            .copy_from_device(
                destination_offset,
                &self.allocation,
                range.start,
                range.len(),
            )
            .map_err(PagedStorageError::Device)?;
        Ok(())
    }

    pub fn tensor_location(
        &self,
        layout: PagedKvCacheLayout,
        page_index: usize,
        layer_index: usize,
        tensor_index: usize,
        token_index: usize,
    ) -> Result<DeviceTensorLocation, PagedStorageError<B::Error>> {
        self.validate_layout(layout)?;
        let range =
            layout.tensor_token_byte_range(page_index, layer_index, tensor_index, token_index)?;
        let device_ptr = self
            .allocation
            .device_ptr_at(range.start, range.len())
            .map_err(PagedStorageError::Device)?;
        Ok(DeviceTensorLocation {
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
    ) -> Result<DeviceTensorLocation, PagedStorageError<B::Error>> {
        self.validate_layout(layout)?;
        let range = layout.tensor_slot_byte_range(layer_index, tensor_index, slot_index)?;
        let device_ptr = self
            .allocation
            .device_ptr_at(range.start, range.len())
            .map_err(PagedStorageError::Device)?;
        Ok(DeviceTensorLocation {
            device_ptr,
            byte_offset: range.start,
            byte_len: range.len(),
        })
    }
}

impl<B: DeviceBuffer> KvCacheStorage for PagedDeviceStorage<B> {
    fn byte_len(&self) -> usize {
        self.allocation.byte_len()
    }

    fn clear(&mut self) -> Result<(), Self::Error> {
        self.allocation.fill(0).map_err(PagedStorageError::Device)?;
        Ok(())
    }
}

impl<B: DeviceBuffer> KvCacheMemoryProvider for PagedDeviceStorage<B> {
    type Error = PagedStorageError<B::Error>;

    fn transferable_kv_cache_memory(&self) -> Result<TransferableKvCacheMemory, Self::Error> {
        let device_ptr = self
            .allocation
            .device_ptr_at(0, self.allocation.byte_len())
            .map_err(PagedStorageError::Device)?;
        let base_addr = usize::try_from(device_ptr).map_err(|_| PagedStorageError::SizeOverflow)?;
        Ok(TransferableKvCacheMemory::new(
            vec![TransferableKvCacheRegion {
                base_addr,
                byte_len: self.allocation.byte_len(),
                page_size_bytes: self.page_size_bytes,
            }],
            self.page_size_bytes,
            self.memory_location,
        )?)
    }
}
