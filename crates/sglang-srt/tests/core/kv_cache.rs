use sglang_srt::kv_cache::{
    KvCacheModelLayout, KvCachePool, KvCachePoolError, KvCacheStorage, PagedKvCacheLayout,
    PagedKvCacheLayoutError,
};
use sglang_srt::transfer::{
    KvCacheDtype, KvCacheMemoryLocation, KvCacheMemoryProvider, KvCacheRuntimeLayout,
    TransferableKvCacheMemory, TransferableKvCacheRegion,
};

struct ByteStorage {
    bytes: Vec<u8>,
    descriptor: TransferableKvCacheMemory,
}

impl ByteStorage {
    fn new(byte_len: usize, page_size_bytes: usize, fill: u8) -> Self {
        let bytes = vec![fill; byte_len];
        let descriptor = TransferableKvCacheMemory::new(
            vec![TransferableKvCacheRegion {
                base_addr: bytes.as_ptr() as usize,
                byte_len,
                page_size_bytes,
            }],
            page_size_bytes,
            KvCacheMemoryLocation::Cpu { numa_node: 0 },
        )
        .expect("test storage must produce a valid NexusKV descriptor");
        Self { bytes, descriptor }
    }
}

impl KvCacheMemoryProvider for ByteStorage {
    type Error = std::convert::Infallible;

    fn transferable_kv_cache_memory(&self) -> Result<TransferableKvCacheMemory, Self::Error> {
        Ok(self.descriptor.clone())
    }
}

impl KvCacheStorage for ByteStorage {
    fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    fn clear(&mut self) -> Result<(), Self::Error> {
        self.bytes.fill(0);
        Ok(())
    }
}

fn runtime_layout() -> KvCacheRuntimeLayout {
    KvCacheRuntimeLayout {
        dtype: KvCacheDtype::Bfloat16,
        page_size: 4,
        num_layers: 2,
        kv_heads: 2,
        head_dim: 8,
        kv_tensors_per_token: 2,
        bytes_per_token: 128,
        page_size_bytes: 512,
    }
}

#[test]
fn page_major_layout_matches_page_layer_tensor_token_order() {
    let layout = PagedKvCacheLayout::new(runtime_layout(), 3).expect("layout should be valid");

    assert_eq!(layout.runtime(), runtime_layout());
    assert_eq!(layout.page_count(), 3);
    assert_eq!(layout.bytes_per_token_per_layer(), 64);
    assert_eq!(layout.bytes_per_token_per_tensor(), Some(32));
    assert_eq!(layout.bytes_per_layer_page(), 256);
    assert_eq!(layout.bytes_per_tensor_page(), Some(128));
    assert_eq!(layout.total_byte_len(), 1_536);
    assert_eq!(layout.slot_count(), 12);
    assert_eq!(layout.page_byte_range(1).unwrap(), 512..1_024);
    assert_eq!(
        layout.tensor_token_byte_range(1, 1, 1, 3).unwrap(),
        992..1_024
    );
    assert_eq!(layout.tensor_slot_byte_range(1, 1, 7).unwrap(), 992..1_024);
}

#[test]
fn page_major_layout_rejects_out_of_range_coordinates() {
    let layout = PagedKvCacheLayout::new(runtime_layout(), 3).expect("layout should be valid");

    assert_eq!(
        layout.page_byte_range(3),
        Err(PagedKvCacheLayoutError::PageOutOfRange {
            page_index: 3,
            page_count: 3,
        })
    );
    assert_eq!(
        layout.tensor_token_byte_range(0, 2, 0, 0),
        Err(PagedKvCacheLayoutError::LayerOutOfRange {
            layer_index: 2,
            layer_count: 2,
        })
    );
    assert_eq!(
        layout.tensor_token_byte_range(0, 0, 2, 0),
        Err(PagedKvCacheLayoutError::TensorOutOfRange {
            tensor_index: 2,
            tensor_count: 2,
        })
    );
    assert_eq!(
        layout.tensor_token_byte_range(0, 0, 0, 4),
        Err(PagedKvCacheLayoutError::TokenOutOfRange {
            token_index: 4,
            page_size: 4,
        })
    );
    assert_eq!(
        layout.tensor_slot_byte_range(0, 0, 12),
        Err(PagedKvCacheLayoutError::SlotOutOfRange {
            slot_index: 12,
            slot_count: 12,
        })
    );
}

#[test]
fn page_major_layout_fails_fast_on_inconsistent_runtime_metadata() {
    let mut layout = runtime_layout();
    layout.page_size_bytes = 511;
    assert_eq!(
        PagedKvCacheLayout::new(layout, 1),
        Err(PagedKvCacheLayoutError::RuntimePageSizeMismatch {
            expected: 512,
            actual: 511,
        })
    );

    let mut layout = runtime_layout();
    layout.bytes_per_token = 129;
    layout.page_size_bytes = 516;
    assert_eq!(
        PagedKvCacheLayout::new(layout, 1),
        Err(PagedKvCacheLayoutError::UnevenLayerLayout {
            bytes_per_token: 129,
            num_layers: 2,
        })
    );

    let mut layout = runtime_layout();
    layout.bytes_per_token = 130;
    layout.page_size_bytes = 520;
    assert_eq!(
        PagedKvCacheLayout::new(layout, 1),
        Err(PagedKvCacheLayoutError::UnevenTensorLayout {
            bytes_per_token_per_layer: 65,
            kv_tensors_per_token: 2,
        })
    );

    assert_eq!(
        PagedKvCacheLayout::new(runtime_layout(), usize::MAX),
        Err(PagedKvCacheLayoutError::SizeOverflow)
    );
}

#[test]
fn page_major_layout_describes_a_layer_kv_pair() {
    let layout = PagedKvCacheLayout::new(runtime_layout(), 3).expect("layout should be valid");
    let geometry = layout
        .kv_pair_copy_geometry(1)
        .expect("second layer should have a K/V pair");

    assert_eq!(geometry.page_size, 4);
    assert_eq!(geometry.page_stride_bytes, 512);
    assert_eq!(geometry.key_offset_bytes, 256);
    assert_eq!(geometry.value_offset_bytes, 384);
    assert_eq!(geometry.token_bytes, 32);
}

#[test]
fn asymmetric_tensor_pair_preserves_each_tensor_width() {
    let model_layout =
        KvCacheModelLayout::tensor_pair(2, 1, 5, 1, 3).expect("tensor widths are valid");
    assert_eq!(model_layout.elements_per_token(), Some(16));
    assert_eq!(
        model_layout
            .token_size_bytes(KvCacheDtype::Bfloat16)
            .expect("BF16 has a known element width"),
        32
    );

    let runtime = KvCacheRuntimeLayout {
        dtype: KvCacheDtype::Bfloat16,
        page_size: 4,
        num_layers: 2,
        kv_heads: 1,
        head_dim: 5,
        kv_tensors_per_token: 2,
        bytes_per_token: 32,
        page_size_bytes: 128,
    };
    let layout = PagedKvCacheLayout::new_with_tensor_pair(runtime, 2, 10, 6)
        .expect("asymmetric pair must match the aggregate runtime layout");

    assert_eq!(layout.bytes_per_token_per_tensor(), None);
    assert_eq!(layout.bytes_per_tensor_page(), None);
    assert_eq!(layout.total_byte_len(), 256);
    assert_eq!(layout.tensor_token_byte_range(0, 0, 0, 3).unwrap(), 30..40);
    assert_eq!(layout.tensor_token_byte_range(0, 0, 1, 3).unwrap(), 58..64);
    assert_eq!(layout.tensor_token_byte_range(0, 1, 0, 0).unwrap(), 64..74);
    assert_eq!(
        layout.kv_pair_copy_geometry(0),
        Err(PagedKvCacheLayoutError::UnevenKvPairCopy {
            key_token_bytes: 10,
            value_token_bytes: 6,
        })
    );
}

#[test]
fn explicit_tensor_pair_rejects_inconsistent_runtime_geometry() {
    let mut runtime = runtime_layout();
    runtime.kv_tensors_per_token = 3;
    assert_eq!(
        PagedKvCacheLayout::new_with_tensor_pair(runtime, 1, 32, 32),
        Err(PagedKvCacheLayoutError::TensorPairRequiresTwoTensors { tensor_count: 3 })
    );

    let runtime = runtime_layout();
    assert_eq!(
        PagedKvCacheLayout::new_with_tensor_pair(runtime, 1, 40, 20),
        Err(PagedKvCacheLayoutError::TensorPairSizeMismatch {
            bytes_per_token_per_layer: 64,
            key_token_bytes: 40,
            value_token_bytes: 20,
        })
    );
}

#[test]
fn page_major_layout_validates_scheduler_slot_maps_before_cuda_upload() {
    let layout = PagedKvCacheLayout::new(runtime_layout(), 3).expect("layout should be valid");

    layout
        .validate_slot_indices(&[0, 4, 11])
        .expect("physical slots spanning pages should validate");
    assert_eq!(
        layout.validate_slot_indices(&[]),
        Err(PagedKvCacheLayoutError::EmptySlotMap)
    );
    assert_eq!(
        layout.validate_slot_indices(&[0, 12, 1]),
        Err(PagedKvCacheLayoutError::BatchSlotOutOfRange {
            batch_index: 1,
            slot_index: 12,
            slot_count: 12,
        })
    );

    let mut single_tensor_runtime = runtime_layout();
    single_tensor_runtime.kv_tensors_per_token = 1;
    let single_tensor_layout = PagedKvCacheLayout::new(single_tensor_runtime, 1)
        .expect("single-tensor metadata is valid but cannot hold a K/V pair");
    assert_eq!(
        single_tensor_layout.kv_pair_copy_geometry(0),
        Err(PagedKvCacheLayoutError::KvPairRequiresTwoTensors { tensor_count: 1 })
    );
}

#[test]
fn pool_owns_backend_storage_without_knowing_its_platform() {
    let layout = PagedKvCacheLayout::new(runtime_layout(), 1).expect("layout should be valid");
    let storage = ByteStorage::new(layout.total_byte_len(), layout.runtime().page_size_bytes, 7);
    let mut pool = KvCachePool::new(layout, storage).expect("storage capacity should match");

    pool.clear().expect("byte storage clear is infallible");

    assert_eq!(pool.layout(), layout);
    assert!(pool.storage().bytes.iter().all(|byte| *byte == 0));
    let descriptor = pool
        .transferable_kv_cache_memory()
        .expect("the common pool must forward its storage's NexusKV contract");
    assert_eq!(descriptor, pool.storage().descriptor);
}

#[test]
fn pool_rejects_storage_with_the_wrong_capacity() {
    let layout = PagedKvCacheLayout::new(runtime_layout(), 1).expect("layout should be valid");
    let storage = ByteStorage::new(layout.total_byte_len() - 1, 1, 0);

    assert!(matches!(
        KvCachePool::new(layout, storage),
        Err(KvCachePoolError::StorageSizeMismatch {
            expected: 512,
            actual: 511,
        })
    ));
}
