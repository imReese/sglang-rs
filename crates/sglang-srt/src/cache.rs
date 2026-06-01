use std::collections::{HashMap, VecDeque};
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CachePageId(usize);

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CachePageAllocator {
    free_pages: VecDeque<CachePageId>,
}

impl CachePageAllocator {
    pub fn new(page_count: usize) -> Self {
        Self {
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
}

#[derive(Default)]
pub struct RadixCache {
    root: RadixNode,
}

impl RadixCache {
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
