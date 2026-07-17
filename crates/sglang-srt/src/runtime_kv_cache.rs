use std::any::Any;
use std::fmt;

use nexus_transfer::{KvCacheMemoryProvider, TransferableKvCacheMemory};

use crate::backend::RuntimeBackend;
use crate::kv_cache::PagedKvCacheLayout;
use crate::transfer::KvCacheTransferError;

pub(crate) trait ActiveKvCache: Send {
    fn backend(&self) -> RuntimeBackend;
    fn layout(&self) -> PagedKvCacheLayout;
    fn transferable_memory(&self) -> Result<TransferableKvCacheMemory, KvCacheTransferError>;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub(crate) struct RuntimeKvCache {
    allocation: Box<dyn ActiveKvCache>,
}

impl RuntimeKvCache {
    pub(crate) fn new(allocation: impl ActiveKvCache + 'static) -> Self {
        Self {
            allocation: Box::new(allocation),
        }
    }

    pub(crate) fn execution_resources(&mut self) -> ModelExecutionResources<'_> {
        ModelExecutionResources {
            kv_cache: Some(self.allocation.as_mut()),
        }
    }

    pub(crate) fn layout(&self) -> PagedKvCacheLayout {
        self.allocation.layout()
    }
}

impl fmt::Debug for RuntimeKvCache {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeKvCache")
            .field("backend", &self.allocation.backend())
            .field("layout", &self.allocation.layout())
            .finish_non_exhaustive()
    }
}

impl KvCacheMemoryProvider for RuntimeKvCache {
    type Error = KvCacheTransferError;

    fn transferable_kv_cache_memory(&self) -> Result<TransferableKvCacheMemory, Self::Error> {
        self.allocation.transferable_memory()
    }
}

pub struct ModelExecutionResources<'a> {
    kv_cache: Option<&'a mut dyn ActiveKvCache>,
}

impl<'a> ModelExecutionResources<'a> {
    pub fn without_kv_cache() -> Self {
        Self { kv_cache: None }
    }

    pub(crate) fn active_kv_cache(self) -> Option<&'a mut dyn ActiveKvCache> {
        self.kv_cache
    }
}
