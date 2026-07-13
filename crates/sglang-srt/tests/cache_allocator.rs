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
        Err(CacheAllocationError::OutOfSlots {
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

#[test]
fn page_aware_allocator_keeps_independent_requests_on_separate_pages() {
    let mut allocator =
        CachePageAllocator::with_page_size(8, 4).expect("page layout should be valid");

    let first = allocator.allocate(3).expect("first request should fit");
    let second = allocator.allocate(2).expect("second request should fit");

    assert_eq!(
        first,
        vec![
            CachePageId::from(0),
            CachePageId::from(1),
            CachePageId::from(2),
        ]
    );
    assert_eq!(second, vec![CachePageId::from(4), CachePageId::from(5)]);
    assert_eq!(allocator.available_pages(), 0);
    assert_eq!(allocator.available_slots(), 0);
}

#[test]
fn page_aware_allocator_extends_a_sequence_tail_before_reserving_another_page() {
    let mut allocator =
        CachePageAllocator::with_page_size(8, 4).expect("page layout should be valid");
    let mut sequence = allocator.allocate(3).expect("prefill should fit");

    let extension = allocator
        .allocate_for_sequence(&sequence, 2)
        .expect("decode extension should fit");
    assert_eq!(extension, vec![CachePageId::from(3), CachePageId::from(4)]);
    sequence.extend_from_slice(&extension);

    allocator.release(&extension);
    assert_eq!(allocator.available_pages(), 1);
    assert_eq!(
        allocator.allocate(1).expect("new page should be reusable"),
        vec![CachePageId::from(4)]
    );

    allocator.release(&sequence[..3]);
    assert_eq!(allocator.available_pages(), 1);
}

#[test]
fn page_aware_allocator_rejects_invalid_pool_geometry() {
    assert_eq!(
        CachePageAllocator::with_page_size(8, 0),
        Err(CacheAllocationError::ZeroPageSize)
    );
    assert_eq!(
        CachePageAllocator::with_page_size(0, 4),
        Err(CacheAllocationError::ZeroSlotCapacity)
    );
    assert_eq!(
        CachePageAllocator::with_page_size(10, 4),
        Err(CacheAllocationError::SlotCapacityNotPageAligned {
            slot_capacity: 10,
            page_size: 4,
        })
    );
}
