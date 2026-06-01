use sglang_srt::cache::{CacheAllocationError, CachePageAllocator, CachePageId};

#[test]
fn allocator_returns_contiguous_free_pages_in_page_order() {
    let mut allocator = CachePageAllocator::new(4);

    let first = allocator.allocate(3).expect("allocation should succeed");
    let second = allocator.allocate(1).expect("allocation should succeed");

    assert_eq!(
        first,
        vec![
            CachePageId::from(0),
            CachePageId::from(1),
            CachePageId::from(2)
        ]
    );
    assert_eq!(second, vec![CachePageId::from(3)]);
    assert_eq!(allocator.available_pages(), 0);
}

#[test]
fn allocator_does_not_consume_pages_when_allocation_cannot_be_satisfied() {
    let mut allocator = CachePageAllocator::new(2);

    let result = allocator.allocate(3);

    assert_eq!(
        result,
        Err(CacheAllocationError::OutOfPages {
            requested: 3,
            available: 2
        })
    );
    assert_eq!(allocator.available_pages(), 2);
    assert_eq!(
        allocator
            .allocate(2)
            .expect("allocation should still succeed"),
        vec![CachePageId::from(0), CachePageId::from(1)]
    );
}

#[test]
fn allocator_reuses_released_pages_before_never_allocated_pages() {
    let mut allocator = CachePageAllocator::new(5);
    let allocated = allocator.allocate(3).expect("allocation should succeed");

    allocator.release(&allocated[1..]);

    assert_eq!(
        allocator.allocate(3).expect("allocation should succeed"),
        vec![
            CachePageId::from(1),
            CachePageId::from(2),
            CachePageId::from(3)
        ]
    );
}

#[test]
fn allocator_reset_restores_all_pages_in_page_order() {
    let mut allocator = CachePageAllocator::new(4);
    let _allocated = allocator.allocate(3).expect("allocation should succeed");

    allocator.reset();

    assert_eq!(allocator.available_pages(), 4);
    assert_eq!(
        allocator.allocate(4).expect("allocation should succeed"),
        vec![
            CachePageId::from(0),
            CachePageId::from(1),
            CachePageId::from(2),
            CachePageId::from(3),
        ]
    );
}
