use sglang_srt::cuda_kv_cache::{CudaKvCachePoolError, CudaKvCachePoolLayout};
use sglang_srt::transfer::{KvCacheDtype, KvCacheRuntimeLayout};

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
    let layout = CudaKvCachePoolLayout::new(runtime_layout(), 3).expect("layout should be valid");

    assert_eq!(layout.runtime(), runtime_layout());
    assert_eq!(layout.page_count(), 3);
    assert_eq!(layout.bytes_per_token_per_layer(), 64);
    assert_eq!(layout.bytes_per_token_per_tensor(), 32);
    assert_eq!(layout.bytes_per_layer_page(), 256);
    assert_eq!(layout.bytes_per_tensor_page(), 128);
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
    let layout = CudaKvCachePoolLayout::new(runtime_layout(), 3).expect("layout should be valid");

    assert_eq!(
        layout.page_byte_range(3),
        Err(CudaKvCachePoolError::PageOutOfRange {
            page_index: 3,
            page_count: 3,
        })
    );
    assert_eq!(
        layout.tensor_token_byte_range(0, 2, 0, 0),
        Err(CudaKvCachePoolError::LayerOutOfRange {
            layer_index: 2,
            layer_count: 2,
        })
    );
    assert_eq!(
        layout.tensor_token_byte_range(0, 0, 2, 0),
        Err(CudaKvCachePoolError::TensorOutOfRange {
            tensor_index: 2,
            tensor_count: 2,
        })
    );
    assert_eq!(
        layout.tensor_token_byte_range(0, 0, 0, 4),
        Err(CudaKvCachePoolError::TokenOutOfRange {
            token_index: 4,
            page_size: 4,
        })
    );
    assert_eq!(
        layout.tensor_slot_byte_range(0, 0, 12),
        Err(CudaKvCachePoolError::SlotOutOfRange {
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
        CudaKvCachePoolLayout::new(layout, 1),
        Err(CudaKvCachePoolError::RuntimePageSizeMismatch {
            expected: 512,
            actual: 511,
        })
    );

    let mut layout = runtime_layout();
    layout.bytes_per_token = 129;
    layout.page_size_bytes = 516;
    assert_eq!(
        CudaKvCachePoolLayout::new(layout, 1),
        Err(CudaKvCachePoolError::UnevenLayerLayout {
            bytes_per_token: 129,
            num_layers: 2,
        })
    );

    let mut layout = runtime_layout();
    layout.bytes_per_token = 130;
    layout.page_size_bytes = 520;
    assert_eq!(
        CudaKvCachePoolLayout::new(layout, 1),
        Err(CudaKvCachePoolError::UnevenTensorLayout {
            bytes_per_token_per_layer: 65,
            kv_tensors_per_token: 2,
        })
    );

    assert_eq!(
        CudaKvCachePoolLayout::new(runtime_layout(), usize::MAX),
        Err(CudaKvCachePoolError::SizeOverflow)
    );
}
