use std::fmt;
use std::ops::Range;

use crate::transfer::KvCacheRuntimeLayout;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PagedKvCacheLayoutError {
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
    KvPairRequiresTwoTensors {
        tensor_count: usize,
    },
}

impl fmt::Display for PagedKvCacheLayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroPageCount => formatter.write_str("KV cache page count must be non-zero"),
            Self::ZeroLayoutField(field) => {
                write!(formatter, "KV cache layout field {field} must be non-zero")
            }
            Self::RuntimePageSizeMismatch { expected, actual } => write!(
                formatter,
                "KV cache runtime page size is {actual} bytes but page_size * bytes_per_token requires {expected} bytes"
            ),
            Self::UnevenLayerLayout {
                bytes_per_token,
                num_layers,
            } => write!(
                formatter,
                "KV cache token size {bytes_per_token} bytes is not divisible by {num_layers} layers"
            ),
            Self::UnevenTensorLayout {
                bytes_per_token_per_layer,
                kv_tensors_per_token,
            } => write!(
                formatter,
                "KV cache per-layer token size {bytes_per_token_per_layer} bytes is not divisible by {kv_tensors_per_token} KV tensors"
            ),
            Self::SizeOverflow => formatter.write_str("KV cache layout size overflowed"),
            Self::PageOutOfRange {
                page_index,
                page_count,
            } => write!(
                formatter,
                "KV cache page index {page_index} is outside {page_count} pages"
            ),
            Self::LayerOutOfRange {
                layer_index,
                layer_count,
            } => write!(
                formatter,
                "KV cache layer index {layer_index} is outside {layer_count} layers"
            ),
            Self::TensorOutOfRange {
                tensor_index,
                tensor_count,
            } => write!(
                formatter,
                "KV cache tensor index {tensor_index} is outside {tensor_count} tensors per token"
            ),
            Self::TokenOutOfRange {
                token_index,
                page_size,
            } => write!(
                formatter,
                "KV cache token index {token_index} is outside page size {page_size}"
            ),
            Self::SlotOutOfRange {
                slot_index,
                slot_count,
            } => write!(
                formatter,
                "KV cache slot index {slot_index} is outside {slot_count} token slots"
            ),
            Self::EmptySlotMap => formatter.write_str("KV cache slot map must not be empty"),
            Self::BatchSlotOutOfRange {
                batch_index,
                slot_index,
                slot_count,
            } => write!(
                formatter,
                "KV cache batch slot {batch_index} references physical slot {slot_index}, outside {slot_count} token slots"
            ),
            Self::KvPairRequiresTwoTensors { tensor_count } => write!(
                formatter,
                "KV cache K/V pair layout requires at least two tensors per token, layout has {tensor_count}"
            ),
        }
    }
}

impl std::error::Error for PagedKvCacheLayoutError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvPairCopyGeometry {
    pub page_size: usize,
    pub page_stride_bytes: usize,
    pub key_offset_bytes: usize,
    pub value_offset_bytes: usize,
    pub token_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PagedKvCacheLayout {
    runtime: KvCacheRuntimeLayout,
    page_count: usize,
    bytes_per_token_per_layer: usize,
    bytes_per_token_per_tensor: usize,
    bytes_per_layer_page: usize,
    bytes_per_tensor_page: usize,
    total_byte_len: usize,
    slot_count: usize,
}

impl PagedKvCacheLayout {
    pub fn new(
        runtime: KvCacheRuntimeLayout,
        page_count: usize,
    ) -> Result<Self, PagedKvCacheLayoutError> {
        if page_count == 0 {
            return Err(PagedKvCacheLayoutError::ZeroPageCount);
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
                return Err(PagedKvCacheLayoutError::ZeroLayoutField(field));
            }
        }

        let expected_page_size_bytes = runtime
            .page_size
            .checked_mul(runtime.bytes_per_token)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;
        if runtime.page_size_bytes != expected_page_size_bytes {
            return Err(PagedKvCacheLayoutError::RuntimePageSizeMismatch {
                expected: expected_page_size_bytes,
                actual: runtime.page_size_bytes,
            });
        }
        if !runtime.bytes_per_token.is_multiple_of(runtime.num_layers) {
            return Err(PagedKvCacheLayoutError::UnevenLayerLayout {
                bytes_per_token: runtime.bytes_per_token,
                num_layers: runtime.num_layers,
            });
        }
        let bytes_per_token_per_layer = runtime.bytes_per_token / runtime.num_layers;
        if !bytes_per_token_per_layer.is_multiple_of(runtime.kv_tensors_per_token) {
            return Err(PagedKvCacheLayoutError::UnevenTensorLayout {
                bytes_per_token_per_layer,
                kv_tensors_per_token: runtime.kv_tensors_per_token,
            });
        }
        let bytes_per_token_per_tensor = bytes_per_token_per_layer / runtime.kv_tensors_per_token;
        let bytes_per_layer_page = runtime
            .page_size
            .checked_mul(bytes_per_token_per_layer)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;
        let bytes_per_tensor_page = runtime
            .page_size
            .checked_mul(bytes_per_token_per_tensor)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;
        let total_byte_len = page_count
            .checked_mul(runtime.page_size_bytes)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;
        let slot_count = page_count
            .checked_mul(runtime.page_size)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;

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

    pub fn page_byte_range(
        &self,
        page_index: usize,
    ) -> Result<Range<usize>, PagedKvCacheLayoutError> {
        self.validate_page(page_index)?;
        let start = page_index
            .checked_mul(self.runtime.page_size_bytes)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;
        let end = start
            .checked_add(self.runtime.page_size_bytes)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;
        Ok(start..end)
    }

    pub fn tensor_token_byte_range(
        &self,
        page_index: usize,
        layer_index: usize,
        tensor_index: usize,
        token_index: usize,
    ) -> Result<Range<usize>, PagedKvCacheLayoutError> {
        self.validate_page(page_index)?;
        if layer_index >= self.runtime.num_layers {
            return Err(PagedKvCacheLayoutError::LayerOutOfRange {
                layer_index,
                layer_count: self.runtime.num_layers,
            });
        }
        if tensor_index >= self.runtime.kv_tensors_per_token {
            return Err(PagedKvCacheLayoutError::TensorOutOfRange {
                tensor_index,
                tensor_count: self.runtime.kv_tensors_per_token,
            });
        }
        if token_index >= self.runtime.page_size {
            return Err(PagedKvCacheLayoutError::TokenOutOfRange {
                token_index,
                page_size: self.runtime.page_size,
            });
        }

        let page_offset = page_index
            .checked_mul(self.runtime.page_size_bytes)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;
        let layer_offset = layer_index
            .checked_mul(self.bytes_per_layer_page)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;
        let tensor_offset = tensor_index
            .checked_mul(self.bytes_per_tensor_page)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;
        let token_offset = token_index
            .checked_mul(self.bytes_per_token_per_tensor)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;
        let start = page_offset
            .checked_add(layer_offset)
            .and_then(|offset| offset.checked_add(tensor_offset))
            .and_then(|offset| offset.checked_add(token_offset))
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;
        let end = start
            .checked_add(self.bytes_per_token_per_tensor)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;
        Ok(start..end)
    }

    pub fn tensor_slot_byte_range(
        &self,
        layer_index: usize,
        tensor_index: usize,
        slot_index: usize,
    ) -> Result<Range<usize>, PagedKvCacheLayoutError> {
        if slot_index >= self.slot_count {
            return Err(PagedKvCacheLayoutError::SlotOutOfRange {
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

    pub fn validate_slot_indices(&self, slots: &[usize]) -> Result<(), PagedKvCacheLayoutError> {
        if slots.is_empty() {
            return Err(PagedKvCacheLayoutError::EmptySlotMap);
        }
        for (batch_index, slot_index) in slots.iter().copied().enumerate() {
            if slot_index >= self.slot_count {
                return Err(PagedKvCacheLayoutError::BatchSlotOutOfRange {
                    batch_index,
                    slot_index,
                    slot_count: self.slot_count,
                });
            }
        }
        Ok(())
    }

    pub fn kv_pair_copy_geometry(
        &self,
        layer_index: usize,
    ) -> Result<KvPairCopyGeometry, PagedKvCacheLayoutError> {
        if layer_index >= self.runtime.num_layers {
            return Err(PagedKvCacheLayoutError::LayerOutOfRange {
                layer_index,
                layer_count: self.runtime.num_layers,
            });
        }
        if self.runtime.kv_tensors_per_token < 2 {
            return Err(PagedKvCacheLayoutError::KvPairRequiresTwoTensors {
                tensor_count: self.runtime.kv_tensors_per_token,
            });
        }
        let key_offset_bytes = layer_index
            .checked_mul(self.bytes_per_layer_page)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;
        let value_offset_bytes = key_offset_bytes
            .checked_add(self.bytes_per_tensor_page)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;
        Ok(KvPairCopyGeometry {
            page_size: self.runtime.page_size,
            page_stride_bytes: self.runtime.page_size_bytes,
            key_offset_bytes,
            value_offset_bytes,
            token_bytes: self.bytes_per_token_per_tensor,
        })
    }

    fn validate_page(&self, page_index: usize) -> Result<(), PagedKvCacheLayoutError> {
        if page_index >= self.page_count {
            Err(PagedKvCacheLayoutError::PageOutOfRange {
                page_index,
                page_count: self.page_count,
            })
        } else {
            Ok(())
        }
    }
}

pub trait KvCacheStorage {
    type Error;

    fn byte_len(&self) -> usize;
    fn clear(&mut self) -> Result<(), Self::Error>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvCachePoolError {
    StorageSizeMismatch { expected: usize, actual: usize },
}

impl fmt::Display for KvCachePoolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StorageSizeMismatch { expected, actual } => write!(
                formatter,
                "KV cache storage has {actual} bytes but layout requires exactly {expected} bytes"
            ),
        }
    }
}

impl std::error::Error for KvCachePoolError {}

pub struct KvCachePool<S> {
    layout: PagedKvCacheLayout,
    storage: S,
}

impl<S> KvCachePool<S>
where
    S: KvCacheStorage,
{
    pub fn new(layout: PagedKvCacheLayout, storage: S) -> Result<Self, KvCachePoolError> {
        let actual = storage.byte_len();
        let expected = layout.total_byte_len();
        if actual != expected {
            return Err(KvCachePoolError::StorageSizeMismatch { expected, actual });
        }
        Ok(Self { layout, storage })
    }

    pub fn layout(&self) -> PagedKvCacheLayout {
        self.layout
    }

    pub fn storage(&self) -> &S {
        &self.storage
    }

    pub fn storage_mut(&mut self) -> &mut S {
        &mut self.storage
    }

    pub fn clear(&mut self) -> Result<(), S::Error> {
        self.storage.clear()
    }

    pub fn into_storage(self) -> S {
        self.storage
    }
}
