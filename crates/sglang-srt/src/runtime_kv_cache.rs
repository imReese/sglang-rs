use std::fmt;

use nexus_transfer::{KvCacheMemoryProvider, TransferableKvCacheMemory};

use crate::backend::RuntimeBackend;
use crate::kv_cache::PagedKvCacheLayout;
use crate::transfer::KvCacheTransferError;

pub(crate) trait ActiveKvCache: Send {
    fn backend(&self) -> RuntimeBackend;
    fn layout(&self) -> PagedKvCacheLayout;
    fn transferable_memory(&self) -> Result<TransferableKvCacheMemory, KvCacheTransferError>;
}

pub(crate) struct RuntimeKvCache<K> {
    allocation: K,
}

impl<K> RuntimeKvCache<K>
where
    K: ActiveKvCache,
{
    pub(crate) fn new(allocation: K) -> Self {
        Self { allocation }
    }

    pub(crate) fn layout(&self) -> PagedKvCacheLayout {
        self.allocation.layout()
    }

    pub(crate) fn allocation_mut(&mut self) -> &mut K {
        &mut self.allocation
    }
}

impl<K> fmt::Debug for RuntimeKvCache<K>
where
    K: ActiveKvCache,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeKvCache")
            .field("backend", &self.allocation.backend())
            .field("layout", &self.allocation.layout())
            .finish_non_exhaustive()
    }
}

impl<K> KvCacheMemoryProvider for RuntimeKvCache<K>
where
    K: ActiveKvCache,
{
    type Error = KvCacheTransferError;

    fn transferable_kv_cache_memory(&self) -> Result<TransferableKvCacheMemory, Self::Error> {
        self.allocation.transferable_memory()
    }
}

pub(crate) trait RuntimeKvCacheMetadata {
    fn active_kv_cache_layout(&self) -> Option<PagedKvCacheLayout>;
}
