use std::collections::{BTreeSet, HashMap, VecDeque};
use std::fmt;

use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CachePageId(usize);

impl CachePageId {
    pub fn as_usize(&self) -> usize {
        self.0
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
    OutOfPages { requested: usize, available: usize },
}

impl fmt::Display for CacheAllocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfPages {
                requested,
                available,
            } => write!(
                formatter,
                "requested {requested} cache pages but only {available} are available"
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
        if candidates.is_empty() {
            return None;
        }

        let matched = self.match_prefix(block_hashes);
        let match_rate = if block_hashes.is_empty() {
            0.0
        } else {
            matched.matched_blocks as f32 / block_hashes.len() as f32
        };

        if match_rate > cache_threshold && !matched.workers.is_empty() {
            if let Some(worker) = candidates
                .iter()
                .filter(|candidate| matched.workers.contains(&candidate.id))
                .min_by_key(|candidate| candidate.active_load)
            {
                return Some(worker.id.clone());
            }
        }

        candidates
            .iter()
            .min_by_key(|candidate| candidate.active_load)
            .map(|worker| worker.id.clone())
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
    page_count: usize,
    free_pages: VecDeque<CachePageId>,
}

impl CachePageAllocator {
    pub fn new(page_count: usize) -> Self {
        Self {
            page_count,
            free_pages: (0..page_count).map(CachePageId::from).collect(),
        }
    }

    pub fn available_pages(&self) -> usize {
        self.free_pages.len()
    }

    pub fn allocate(
        &mut self,
        page_count: usize,
    ) -> Result<Vec<CachePageId>, CacheAllocationError> {
        if page_count > self.free_pages.len() {
            return Err(CacheAllocationError::OutOfPages {
                requested: page_count,
                available: self.free_pages.len(),
            });
        }

        Ok((0..page_count)
            .filter_map(|_| self.free_pages.pop_front())
            .collect())
    }

    pub fn release(&mut self, cache_pages: &[CachePageId]) {
        for cache_page in cache_pages.iter().rev() {
            self.free_pages.push_front(*cache_page);
        }
    }

    pub fn reset(&mut self) {
        self.free_pages = (0..self.page_count).map(CachePageId::from).collect();
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
}

#[derive(Default)]
struct RadixNode {
    cache_page: Option<CachePageId>,
    children: HashMap<u32, RadixNode>,
}
