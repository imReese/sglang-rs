use std::collections::{BTreeSet, HashMap, VecDeque};
use std::fmt;

use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CachePageId(usize);

impl CachePageId {
    pub fn as_usize(&self) -> usize {
        self.0
    }

    pub fn page_index(&self, page_size: usize) -> usize {
        assert!(page_size > 0, "KV cache page size must be non-zero");
        self.0 / page_size
    }

    pub fn slot_index_in_page(&self, page_size: usize) -> usize {
        assert!(page_size > 0, "KV cache page size must be non-zero");
        self.0 % page_size
    }
}

impl From<usize> for CachePageId {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixMatch {
    pub matched_token_count: usize,
    pub cache_pages: Vec<CachePageId>,
    pub remaining_input_ids: Vec<u32>,
}

#[derive(Debug, Eq, PartialEq)]
pub enum RadixCacheError {
    TokenPageLengthMismatch {
        token_count: usize,
        cache_page_count: usize,
    },
}

impl fmt::Display for RadixCacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokenPageLengthMismatch {
                token_count,
                cache_page_count,
            } => write!(
                formatter,
                "token count ({token_count}) must match cache page count ({cache_page_count})"
            ),
        }
    }
}

impl std::error::Error for RadixCacheError {}

#[derive(Debug, Eq, PartialEq)]
pub enum CacheAllocationError {
    ZeroPageSize,
    ZeroSlotCapacity,
    SlotCapacityNotPageAligned {
        slot_capacity: usize,
        page_size: usize,
    },
    OutOfSlots {
        requested: usize,
        available: usize,
    },
}

impl fmt::Display for CacheAllocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroPageSize => formatter.write_str("KV cache page size must be non-zero"),
            Self::ZeroSlotCapacity => {
                formatter.write_str("KV cache slot capacity must be non-zero")
            }
            Self::SlotCapacityNotPageAligned {
                slot_capacity,
                page_size,
            } => write!(
                formatter,
                "KV cache slot capacity {slot_capacity} must be divisible by page size {page_size}"
            ),
            Self::OutOfSlots {
                requested,
                available,
            } => write!(
                formatter,
                "requested {requested} KV cache slots but only {available} are available"
            ),
        }
    }
}

impl std::error::Error for CacheAllocationError {}

pub fn compute_sglang_block_hashes(input_ids: &[u32], block_size: usize) -> Vec<i64> {
    assert!(block_size > 0, "block_size must be positive");
    if input_ids.is_empty() {
        return Vec::new();
    }

    let mut hashes = Vec::with_capacity(input_ids.len().div_ceil(block_size));
    let mut parent_digest = None;

    for block in input_ids.chunks(block_size) {
        let digest = sglang_hash_block(parent_digest.as_ref(), block);
        hashes.push(sglang_sha256_digest_to_i64(&digest));
        parent_digest = Some(digest);
    }

    hashes
}

pub fn sglang_sha256_digest_to_i64(digest: &[u8; 32]) -> i64 {
    let mut top_bytes = [0u8; 8];
    top_bytes.copy_from_slice(&digest[..8]);
    i64::from_be_bytes(top_bytes)
}

fn sglang_hash_block(parent_digest: Option<&[u8; 32]>, input_ids: &[u32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    if let Some(parent_digest) = parent_digest {
        hasher.update(parent_digest);
    }
    for input_id in input_ids {
        hasher.update(input_id.to_le_bytes());
    }
    hasher.finalize().into()
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KvCacheWorkerId {
    pub endpoint: String,
    pub dp_rank: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvCacheWorkerSnapshot {
    pub id: KvCacheWorkerId,
    pub active_load: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KvCacheAwareSelectionConfig {
    pub cache_threshold: f32,
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
}

impl Default for KvCacheAwareSelectionConfig {
    fn default() -> Self {
        Self {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvBlockPrefixMatch {
    pub matched_blocks: usize,
    pub workers: BTreeSet<KvCacheWorkerId>,
}

#[derive(Default)]
pub struct KvBlockPrefixIndex {
    root: KvBlockPrefixNode,
}

impl KvBlockPrefixIndex {
    pub fn insert(&mut self, worker: &KvCacheWorkerId, block_hashes: &[i64]) {
        let mut node = &mut self.root;
        for block_hash in block_hashes {
            node = node.children.entry(*block_hash).or_default();
            node.workers.insert(worker.clone());
        }
    }

    pub fn remove(&mut self, worker: &KvCacheWorkerId, block_hashes: &[i64]) {
        remove_worker_from_chain(&mut self.root, worker, block_hashes);
    }

    pub fn clear_worker(&mut self, worker: &KvCacheWorkerId) {
        clear_worker_from_block_node(&mut self.root, worker);
    }

    pub fn match_prefix(&self, block_hashes: &[i64]) -> KvBlockPrefixMatch {
        let mut node = &self.root;
        let mut matched_blocks = 0;
        let mut workers = BTreeSet::new();

        for block_hash in block_hashes {
            let Some(child) = node.children.get(block_hash) else {
                break;
            };
            matched_blocks += 1;
            workers = child.workers.clone();
            node = child;
        }

        KvBlockPrefixMatch {
            matched_blocks,
            workers,
        }
    }

    pub fn select_cache_aware_worker(
        &self,
        candidates: &[KvCacheWorkerSnapshot],
        block_hashes: &[i64],
        cache_threshold: f32,
    ) -> Option<KvCacheWorkerId> {
        self.select_cache_aware_worker_with_config(
            candidates,
            block_hashes,
            KvCacheAwareSelectionConfig {
                cache_threshold,
                balance_abs_threshold: usize::MAX,
                balance_rel_threshold: f32::MAX,
            },
        )
    }

    pub fn select_cache_aware_worker_with_config(
        &self,
        candidates: &[KvCacheWorkerSnapshot],
        block_hashes: &[i64],
        config: KvCacheAwareSelectionConfig,
    ) -> Option<KvCacheWorkerId> {
        if candidates.is_empty() {
            return None;
        }

        if is_cache_aware_selection_imbalanced(candidates, config) {
            return min_load_worker(candidates);
        }

        let matched = self.match_prefix(block_hashes);
        let match_rate = if block_hashes.is_empty() {
            0.0
        } else {
            matched.matched_blocks as f32 / block_hashes.len() as f32
        };

        if match_rate > config.cache_threshold && !matched.workers.is_empty() {
            if let Some(worker) = candidates
                .iter()
                .filter(|candidate| matched.workers.contains(&candidate.id))
                .min_by_key(|candidate| candidate.active_load)
            {
                return Some(worker.id.clone());
            }
        }

        min_load_worker(candidates)
    }

    pub fn select_cache_aware_worker_for_tokens(
        &self,
        candidates: &[KvCacheWorkerSnapshot],
        input_ids: &[u32],
        block_size: usize,
        cache_threshold: f32,
    ) -> Option<KvCacheWorkerId> {
        let block_hashes = compute_sglang_block_hashes(input_ids, block_size);
        self.select_cache_aware_worker(candidates, &block_hashes, cache_threshold)
    }
}

fn is_cache_aware_selection_imbalanced(
    candidates: &[KvCacheWorkerSnapshot],
    config: KvCacheAwareSelectionConfig,
) -> bool {
    let Some(min_load) = candidates
        .iter()
        .map(|candidate| candidate.active_load)
        .min()
    else {
        return false;
    };
    let max_load = candidates
        .iter()
        .map(|candidate| candidate.active_load)
        .max()
        .unwrap_or(0);
    let absolute_diff = max_load.saturating_sub(min_load);
    let relative_threshold = (min_load as f32 * config.balance_rel_threshold) as usize;
    absolute_diff > config.balance_abs_threshold && max_load > relative_threshold
}

fn min_load_worker(candidates: &[KvCacheWorkerSnapshot]) -> Option<KvCacheWorkerId> {
    candidates
        .iter()
        .min_by_key(|candidate| candidate.active_load)
        .map(|worker| worker.id.clone())
}

fn remove_worker_from_chain(
    node: &mut KvBlockPrefixNode,
    worker: &KvCacheWorkerId,
    block_hashes: &[i64],
) -> bool {
    let Some((block_hash, remaining_hashes)) = block_hashes.split_first() else {
        return node.workers.is_empty() && node.children.is_empty();
    };

    if let Some(child) = node.children.get_mut(block_hash) {
        child.workers.remove(worker);
        if remove_worker_from_chain(child, worker, remaining_hashes) {
            node.children.remove(block_hash);
        }
    }

    node.workers.is_empty() && node.children.is_empty()
}

fn clear_worker_from_block_node(node: &mut KvBlockPrefixNode, worker: &KvCacheWorkerId) -> bool {
    node.workers.remove(worker);
    node.children.retain(|_, child| {
        clear_worker_from_block_node(child, worker);
        !(child.workers.is_empty() && child.children.is_empty())
    });
    node.workers.is_empty() && node.children.is_empty()
}

#[derive(Default)]
struct KvBlockPrefixNode {
    workers: BTreeSet<KvCacheWorkerId>,
    children: HashMap<i64, KvBlockPrefixNode>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CachePageAllocator {
    slot_capacity: usize,
    page_size: usize,
    free_page_indices: VecDeque<usize>,
    allocated_page_indices: BTreeSet<usize>,
}

impl CachePageAllocator {
    pub fn new(slot_capacity: usize) -> Self {
        Self::with_page_size(slot_capacity, 1).expect("KV cache slot capacity must be non-zero")
    }

    pub fn with_page_size(
        slot_capacity: usize,
        page_size: usize,
    ) -> Result<Self, CacheAllocationError> {
        if page_size == 0 {
            return Err(CacheAllocationError::ZeroPageSize);
        }
        if slot_capacity == 0 {
            return Err(CacheAllocationError::ZeroSlotCapacity);
        }
        if !slot_capacity.is_multiple_of(page_size) {
            return Err(CacheAllocationError::SlotCapacityNotPageAligned {
                slot_capacity,
                page_size,
            });
        }
        let page_count = slot_capacity / page_size;
        Ok(Self {
            slot_capacity,
            page_size,
            free_page_indices: (0..page_count).collect(),
            allocated_page_indices: BTreeSet::new(),
        })
    }

    pub fn available_pages(&self) -> usize {
        self.free_page_indices.len()
    }

    pub fn available_slots(&self) -> usize {
        self.free_page_indices.len() * self.page_size
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn allocate(
        &mut self,
        slot_count: usize,
    ) -> Result<Vec<CachePageId>, CacheAllocationError> {
        self.allocate_for_sequence(&[], slot_count)
    }

    pub fn allocate_for_sequence(
        &mut self,
        sequence_slots: &[CachePageId],
        slot_count: usize,
    ) -> Result<Vec<CachePageId>, CacheAllocationError> {
        if slot_count == 0 {
            return Ok(Vec::new());
        }

        let reusable_tail_slots = sequence_slots
            .last()
            .filter(|slot| {
                self.allocated_page_indices
                    .contains(&slot.page_index(self.page_size))
            })
            .map(|slot| self.page_size - slot.slot_index_in_page(self.page_size) - 1)
            .unwrap_or(0)
            .min(slot_count);
        let new_slot_count = slot_count - reusable_tail_slots;
        let required_page_count = new_slot_count.div_ceil(self.page_size);
        if required_page_count > self.free_page_indices.len() {
            return Err(CacheAllocationError::OutOfSlots {
                requested: slot_count,
                available: reusable_tail_slots + self.available_slots(),
            });
        }

        let mut slots = Vec::with_capacity(slot_count);
        if reusable_tail_slots > 0 {
            let first_slot = sequence_slots
                .last()
                .expect("reusable tail requires a sequence slot")
                .as_usize()
                + 1;
            slots.extend((first_slot..first_slot + reusable_tail_slots).map(CachePageId::from));
        }

        let mut remaining_new_slots = new_slot_count;
        for _ in 0..required_page_count {
            let page_index = self
                .free_page_indices
                .pop_front()
                .expect("page availability was checked before allocation");
            self.allocated_page_indices.insert(page_index);
            let slots_in_page = remaining_new_slots.min(self.page_size);
            let first_slot = page_index * self.page_size;
            slots.extend((first_slot..first_slot + slots_in_page).map(CachePageId::from));
            remaining_new_slots -= slots_in_page;
        }
        Ok(slots)
    }

    pub fn release(&mut self, cache_pages: &[CachePageId]) {
        let mut released_page_indices = Vec::new();
        for slot in cache_pages {
            // A page is owned by the allocation that contains its first slot.
            // This keeps a failed tail-slot decode from releasing the sequence's
            // existing physical page.
            if slot.as_usize() >= self.slot_capacity || slot.slot_index_in_page(self.page_size) != 0
            {
                continue;
            }
            let page_index = slot.page_index(self.page_size);
            if self.allocated_page_indices.remove(&page_index) {
                released_page_indices.push(page_index);
            }
        }
        for page_index in released_page_indices.into_iter().rev() {
            self.free_page_indices.push_front(page_index);
        }
    }

    pub fn reset(&mut self) {
        self.free_page_indices = (0..self.slot_capacity / self.page_size).collect();
        self.allocated_page_indices.clear();
    }
}

#[derive(Default)]
pub struct RadixCache {
    root: RadixNode,
}

impl RadixCache {
    pub fn clear(&mut self) {
        self.root = RadixNode::default();
    }

    pub fn insert(
        &mut self,
        input_ids: &[u32],
        cache_pages: &[CachePageId],
    ) -> Result<(), RadixCacheError> {
        if input_ids.len() != cache_pages.len() {
            return Err(RadixCacheError::TokenPageLengthMismatch {
                token_count: input_ids.len(),
                cache_page_count: cache_pages.len(),
            });
        }

        let mut node = &mut self.root;
        for (input_id, cache_page) in input_ids.iter().zip(cache_pages.iter()) {
            node = node.children.entry(*input_id).or_default();
            node.cache_page = Some(*cache_page);
        }

        Ok(())
    }

    pub fn match_prefix(&self, input_ids: &[u32]) -> PrefixMatch {
        let mut matched_token_count = 0;
        let mut cache_pages = Vec::new();
        let mut node = &self.root;

        for input_id in input_ids {
            let Some(child) = node.children.get(input_id) else {
                break;
            };

            let Some(cache_page) = child.cache_page else {
                break;
            };

            matched_token_count += 1;
            cache_pages.push(cache_page);
            node = child;
        }

        PrefixMatch {
            matched_token_count,
            cache_pages,
            remaining_input_ids: input_ids[matched_token_count..].to_vec(),
        }
    }

    pub fn match_prefix_page_aligned(&self, input_ids: &[u32], page_size: usize) -> PrefixMatch {
        assert!(page_size > 0, "KV cache page size must be non-zero");
        let mut prefix_match = self.match_prefix(input_ids);
        let aligned_token_count = prefix_match.matched_token_count / page_size * page_size;
        prefix_match.matched_token_count = aligned_token_count;
        prefix_match.cache_pages.truncate(aligned_token_count);
        prefix_match.remaining_input_ids = input_ids[aligned_token_count..].to_vec();
        prefix_match
    }
}

#[derive(Default)]
struct RadixNode {
    cache_page: Option<CachePageId>,
    children: HashMap<u32, RadixNode>,
}
