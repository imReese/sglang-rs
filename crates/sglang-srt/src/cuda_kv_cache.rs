use std::fmt;
use std::ops::Range;

use sglang_kernel::cuda::{CudaContext, CudaDeviceAllocation, CudaError};

use crate::transfer::{
    KvCacheMemoryLocation, KvCacheRuntimeLayout, KvCacheTransferError,
    MooncakeKvCacheMemoryProvider, TransferableKvCacheMemory, TransferableKvCacheRegion,
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
    PageBufferSizeMismatch {
        expected: usize,
        actual: usize,
    },
    Cuda(CudaError),
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
            Self::PageBufferSizeMismatch { expected, actual } => write!(
                formatter,
                "CUDA KV cache page buffer has {actual} bytes but requires exactly {expected} bytes"
            ),
            Self::Cuda(error) => write!(formatter, "CUDA KV cache operation failed: {error}"),
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

pub struct CudaKvCachePool {
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
        Ok(Self { layout, allocation })
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
        )?)
    }
}

impl MooncakeKvCacheMemoryProvider for CudaKvCachePool {
    fn mooncake_kv_cache_memory(&self) -> Result<TransferableKvCacheMemory, KvCacheTransferError> {
        self.transferable_memory().map_err(|error| {
            KvCacheTransferError::Runtime(format!(
                "CUDA KV cache pool is not transferable: {error}"
            ))
        })
    }
}
