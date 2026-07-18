use std::fmt;
use std::ops::Range;

use nexus_transfer::{KvCacheMemoryProvider, TransferableKvCacheMemory};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvCacheDtype {
    Auto,
    Float32,
    Bfloat16,
    Fp8E4M3,
    Fp8E5M2,
    Fp4E2M1,
}

impl KvCacheDtype {
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "bf16" | "bfloat16" => Some(Self::Bfloat16),
            "fp8_e4m3" => Some(Self::Fp8E4M3),
            "fp8_e5m2" => Some(Self::Fp8E5M2),
            "fp4_e2m1" => Some(Self::Fp4E2M1),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Float32 => "float32",
            Self::Bfloat16 => "bfloat16",
            Self::Fp8E4M3 => "fp8_e4m3",
            Self::Fp8E5M2 => "fp8_e5m2",
            Self::Fp4E2M1 => "fp4_e2m1",
        }
    }

    pub fn bytes_per_element(self) -> Option<usize> {
        match self {
            Self::Auto | Self::Bfloat16 => Some(2),
            Self::Float32 => Some(4),
            Self::Fp8E4M3 | Self::Fp8E5M2 => Some(1),
            Self::Fp4E2M1 => None,
        }
    }

    pub fn runtime_storage_dtype(self) -> Self {
        match self {
            Self::Auto => Self::Bfloat16,
            dtype => dtype,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvCacheRuntimeLayout {
    pub dtype: KvCacheDtype,
    pub page_size: usize,
    pub num_layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub kv_tensors_per_token: usize,
    pub bytes_per_token: usize,
    pub page_size_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvCacheModelLayout {
    pub num_layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub kv_tensors_per_token: usize,
    pub bytes_per_token_per_layer: Option<usize>,
    tensor_geometry: KvCacheTensorGeometry,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KvCacheTensorGeometry {
    Uniform,
    Pair {
        key_elements: usize,
        value_elements: usize,
    },
    PackedBytes,
}

impl KvCacheModelLayout {
    pub fn multi_tensor(
        num_layers: usize,
        kv_heads: usize,
        head_dim: usize,
        kv_tensors_per_token: usize,
    ) -> Self {
        Self {
            num_layers,
            kv_heads,
            head_dim,
            kv_tensors_per_token,
            bytes_per_token_per_layer: None,
            tensor_geometry: KvCacheTensorGeometry::Uniform,
        }
    }

    pub fn tensor_pair(
        num_layers: usize,
        key_heads: usize,
        key_head_dim: usize,
        value_heads: usize,
        value_head_dim: usize,
    ) -> Result<Self, KvCacheLayoutError> {
        let key_elements = key_heads
            .checked_mul(key_head_dim)
            .ok_or(KvCacheLayoutError::SizeOverflow)?;
        let value_elements = value_heads
            .checked_mul(value_head_dim)
            .ok_or(KvCacheLayoutError::SizeOverflow)?;
        if key_elements == 0 || value_elements == 0 {
            return Err(KvCacheLayoutError::ZeroTensorWidth);
        }
        Ok(Self {
            num_layers,
            kv_heads: key_heads,
            head_dim: key_head_dim,
            kv_tensors_per_token: 2,
            bytes_per_token_per_layer: None,
            tensor_geometry: KvCacheTensorGeometry::Pair {
                key_elements,
                value_elements,
            },
        })
    }

    pub fn packed_bytes_per_layer(num_layers: usize, bytes_per_token_per_layer: usize) -> Self {
        Self {
            num_layers,
            kv_heads: 1,
            head_dim: bytes_per_token_per_layer,
            kv_tensors_per_token: 1,
            bytes_per_token_per_layer: Some(bytes_per_token_per_layer),
            tensor_geometry: KvCacheTensorGeometry::PackedBytes,
        }
    }

    pub fn elements_per_token(self) -> Option<usize> {
        let elements_per_layer = match self.tensor_geometry {
            KvCacheTensorGeometry::Uniform => self
                .kv_tensors_per_token
                .checked_mul(self.kv_heads)?
                .checked_mul(self.head_dim)?,
            KvCacheTensorGeometry::Pair {
                key_elements,
                value_elements,
            } => key_elements.checked_add(value_elements)?,
            KvCacheTensorGeometry::PackedBytes => return None,
        };
        self.num_layers.checked_mul(elements_per_layer)
    }

    pub fn token_size_bytes(self, dtype: KvCacheDtype) -> Result<usize, KvCacheLayoutError> {
        if let Some(bytes_per_token_per_layer) = self.bytes_per_token_per_layer {
            return self
                .num_layers
                .checked_mul(bytes_per_token_per_layer)
                .ok_or(KvCacheLayoutError::SizeOverflow);
        }

        self.elements_per_token()
            .ok_or(KvCacheLayoutError::SizeOverflow)?
            .checked_mul(
                dtype
                    .bytes_per_element()
                    .ok_or(KvCacheLayoutError::DtypeRequiresModelMetadata(dtype))?,
            )
            .ok_or(KvCacheLayoutError::SizeOverflow)
    }

    pub(crate) fn tensor_pair_size_bytes(
        self,
        dtype: KvCacheDtype,
    ) -> Result<Option<[usize; 2]>, KvCacheLayoutError> {
        let KvCacheTensorGeometry::Pair {
            key_elements,
            value_elements,
        } = self.tensor_geometry
        else {
            return Ok(None);
        };
        let element_bytes = dtype
            .bytes_per_element()
            .ok_or(KvCacheLayoutError::DtypeRequiresModelMetadata(dtype))?;
        let key_bytes = key_elements
            .checked_mul(element_bytes)
            .ok_or(KvCacheLayoutError::SizeOverflow)?;
        let value_bytes = value_elements
            .checked_mul(element_bytes)
            .ok_or(KvCacheLayoutError::SizeOverflow)?;
        Ok(Some([key_bytes, value_bytes]))
    }

    pub fn packed_mla(
        num_layers: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
    ) -> Result<Self, KvCacheLayoutError> {
        if !qk_nope_head_dim.is_multiple_of(64) {
            return Err(KvCacheLayoutError::InvalidPackedMlaNopeHeadDim(
                qk_nope_head_dim,
            ));
        }
        let bytes_per_token_per_layer = qk_nope_head_dim
            .checked_add(
                qk_rope_head_dim
                    .checked_mul(2)
                    .ok_or(KvCacheLayoutError::SizeOverflow)?,
            )
            .and_then(|value| value.checked_add(qk_nope_head_dim / 64))
            .and_then(|value| value.checked_add(1))
            .ok_or(KvCacheLayoutError::SizeOverflow)?;
        Ok(Self::packed_bytes_per_layer(
            num_layers,
            bytes_per_token_per_layer,
        ))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvCacheLayoutError {
    DtypeRequiresModelMetadata(KvCacheDtype),
    InvalidPackedMlaNopeHeadDim(usize),
    ZeroTensorWidth,
    SizeOverflow,
}

impl fmt::Display for KvCacheLayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DtypeRequiresModelMetadata(dtype) => write!(
                formatter,
                "KV cache dtype {} requires model metadata for byte width",
                dtype.as_str()
            ),
            Self::InvalidPackedMlaNopeHeadDim(head_dim) => write!(
                formatter,
                "qk_nope_head_dim must be divisible by 64 for packed MLA KV layout: {head_dim}"
            ),
            Self::ZeroTensorWidth => formatter.write_str("KV cache tensor width must be non-zero"),
            Self::SizeOverflow => formatter.write_str("KV cache layout byte size overflowed"),
        }
    }
}

impl std::error::Error for KvCacheLayoutError {}

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
    TensorPairRequiresTwoTensors {
        tensor_count: usize,
    },
    TensorPairSizeMismatch {
        bytes_per_token_per_layer: usize,
        key_token_bytes: usize,
        value_token_bytes: usize,
    },
    UnevenKvPairCopy {
        key_token_bytes: usize,
        value_token_bytes: usize,
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
            Self::TensorPairRequiresTwoTensors { tensor_count } => write!(
                formatter,
                "explicit KV tensor-pair layout requires exactly two tensors per token, layout has {tensor_count}"
            ),
            Self::TensorPairSizeMismatch {
                bytes_per_token_per_layer,
                key_token_bytes,
                value_token_bytes,
            } => write!(
                formatter,
                "KV tensor-pair widths ({key_token_bytes} key bytes + {value_token_bytes} value bytes) do not match the runtime per-layer token size of {bytes_per_token_per_layer} bytes"
            ),
            Self::UnevenKvPairCopy {
                key_token_bytes,
                value_token_bytes,
            } => write!(
                formatter,
                "the selected KV copy/attention kernel requires equal key and value widths, but the layout has {key_token_bytes} key bytes and {value_token_bytes} value bytes per token"
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
enum PagedKvTensorGeometry {
    Uniform {
        token_bytes: usize,
        page_bytes: usize,
    },
    Pair {
        key_token_bytes: usize,
        value_token_bytes: usize,
        key_page_bytes: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PagedKvCacheLayout {
    runtime: KvCacheRuntimeLayout,
    page_count: usize,
    bytes_per_token_per_layer: usize,
    bytes_per_layer_page: usize,
    tensor_geometry: PagedKvTensorGeometry,
    total_byte_len: usize,
    slot_count: usize,
}

impl PagedKvCacheLayout {
    pub fn new(
        runtime: KvCacheRuntimeLayout,
        page_count: usize,
    ) -> Result<Self, PagedKvCacheLayoutError> {
        let (bytes_per_token_per_layer, bytes_per_layer_page, total_byte_len, slot_count) =
            Self::validate_runtime(runtime, page_count)?;
        if !bytes_per_token_per_layer.is_multiple_of(runtime.kv_tensors_per_token) {
            return Err(PagedKvCacheLayoutError::UnevenTensorLayout {
                bytes_per_token_per_layer,
                kv_tensors_per_token: runtime.kv_tensors_per_token,
            });
        }
        let token_bytes = bytes_per_token_per_layer / runtime.kv_tensors_per_token;
        let page_bytes = runtime
            .page_size
            .checked_mul(token_bytes)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;

        Ok(Self {
            runtime,
            page_count,
            bytes_per_token_per_layer,
            bytes_per_layer_page,
            tensor_geometry: PagedKvTensorGeometry::Uniform {
                token_bytes,
                page_bytes,
            },
            total_byte_len,
            slot_count,
        })
    }

    pub fn new_with_tensor_pair(
        runtime: KvCacheRuntimeLayout,
        page_count: usize,
        key_token_bytes: usize,
        value_token_bytes: usize,
    ) -> Result<Self, PagedKvCacheLayoutError> {
        let (bytes_per_token_per_layer, bytes_per_layer_page, total_byte_len, slot_count) =
            Self::validate_runtime(runtime, page_count)?;
        if runtime.kv_tensors_per_token != 2 {
            return Err(PagedKvCacheLayoutError::TensorPairRequiresTwoTensors {
                tensor_count: runtime.kv_tensors_per_token,
            });
        }
        for (field, value) in [
            ("key_token_bytes", key_token_bytes),
            ("value_token_bytes", value_token_bytes),
        ] {
            if value == 0 {
                return Err(PagedKvCacheLayoutError::ZeroLayoutField(field));
            }
        }
        let pair_bytes = key_token_bytes
            .checked_add(value_token_bytes)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;
        if pair_bytes != bytes_per_token_per_layer {
            return Err(PagedKvCacheLayoutError::TensorPairSizeMismatch {
                bytes_per_token_per_layer,
                key_token_bytes,
                value_token_bytes,
            });
        }
        let key_page_bytes = runtime
            .page_size
            .checked_mul(key_token_bytes)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;

        Ok(Self {
            runtime,
            page_count,
            bytes_per_token_per_layer,
            bytes_per_layer_page,
            tensor_geometry: PagedKvTensorGeometry::Pair {
                key_token_bytes,
                value_token_bytes,
                key_page_bytes,
            },
            total_byte_len,
            slot_count,
        })
    }

    fn validate_runtime(
        runtime: KvCacheRuntimeLayout,
        page_count: usize,
    ) -> Result<(usize, usize, usize, usize), PagedKvCacheLayoutError> {
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
        let bytes_per_layer_page = runtime
            .page_size
            .checked_mul(bytes_per_token_per_layer)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;
        let total_byte_len = page_count
            .checked_mul(runtime.page_size_bytes)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;
        let slot_count = page_count
            .checked_mul(runtime.page_size)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;

        Ok((
            bytes_per_token_per_layer,
            bytes_per_layer_page,
            total_byte_len,
            slot_count,
        ))
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

    pub fn bytes_per_token_per_tensor(&self) -> Option<usize> {
        match self.tensor_geometry {
            PagedKvTensorGeometry::Uniform { token_bytes, .. } => Some(token_bytes),
            PagedKvTensorGeometry::Pair { .. } => None,
        }
    }

    pub fn bytes_per_layer_page(&self) -> usize {
        self.bytes_per_layer_page
    }

    pub fn bytes_per_tensor_page(&self) -> Option<usize> {
        match self.tensor_geometry {
            PagedKvTensorGeometry::Uniform { page_bytes, .. } => Some(page_bytes),
            PagedKvTensorGeometry::Pair { .. } => None,
        }
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
        let tensor_offset = self.tensor_page_offset(tensor_index)?;
        let tensor_token_bytes = self.tensor_token_bytes(tensor_index)?;
        let token_offset = token_index
            .checked_mul(tensor_token_bytes)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;
        let start = page_offset
            .checked_add(layer_offset)
            .and_then(|offset| offset.checked_add(tensor_offset))
            .and_then(|offset| offset.checked_add(token_offset))
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;
        let end = start
            .checked_add(tensor_token_bytes)
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
        let layer_offset_bytes = layer_index
            .checked_mul(self.bytes_per_layer_page)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;
        let key_token_bytes = self.tensor_token_bytes(0)?;
        let value_token_bytes = self.tensor_token_bytes(1)?;
        if key_token_bytes != value_token_bytes {
            return Err(PagedKvCacheLayoutError::UnevenKvPairCopy {
                key_token_bytes,
                value_token_bytes,
            });
        }
        let key_offset_bytes = layer_offset_bytes
            .checked_add(self.tensor_page_offset(0)?)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;
        let value_offset_bytes = layer_offset_bytes
            .checked_add(self.tensor_page_offset(1)?)
            .ok_or(PagedKvCacheLayoutError::SizeOverflow)?;
        Ok(KvPairCopyGeometry {
            page_size: self.runtime.page_size,
            page_stride_bytes: self.runtime.page_size_bytes,
            key_offset_bytes,
            value_offset_bytes,
            token_bytes: key_token_bytes,
        })
    }

    fn tensor_token_bytes(&self, tensor_index: usize) -> Result<usize, PagedKvCacheLayoutError> {
        match self.tensor_geometry {
            PagedKvTensorGeometry::Uniform { token_bytes, .. } => Ok(token_bytes),
            PagedKvTensorGeometry::Pair {
                key_token_bytes,
                value_token_bytes,
                ..
            } => match tensor_index {
                0 => Ok(key_token_bytes),
                1 => Ok(value_token_bytes),
                _ => Err(PagedKvCacheLayoutError::TensorOutOfRange {
                    tensor_index,
                    tensor_count: self.runtime.kv_tensors_per_token,
                }),
            },
        }
    }

    fn tensor_page_offset(&self, tensor_index: usize) -> Result<usize, PagedKvCacheLayoutError> {
        match self.tensor_geometry {
            PagedKvTensorGeometry::Uniform { page_bytes, .. } => tensor_index
                .checked_mul(page_bytes)
                .ok_or(PagedKvCacheLayoutError::SizeOverflow),
            PagedKvTensorGeometry::Pair { key_page_bytes, .. } => match tensor_index {
                0 => Ok(0),
                1 => Ok(key_page_bytes),
                _ => Err(PagedKvCacheLayoutError::TensorOutOfRange {
                    tensor_index,
                    tensor_count: self.runtime.kv_tensors_per_token,
                }),
            },
        }
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

pub trait KvCacheStorage: KvCacheMemoryProvider {
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

impl<S> KvCacheMemoryProvider for KvCachePool<S>
where
    S: KvCacheStorage,
{
    type Error = S::Error;

    fn transferable_kv_cache_memory(&self) -> Result<TransferableKvCacheMemory, Self::Error> {
        self.storage.transferable_kv_cache_memory()
    }
}
