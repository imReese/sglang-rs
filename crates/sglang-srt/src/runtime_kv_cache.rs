use std::fmt;

use nexus_transfer::{KvCacheMemoryProvider, TransferableKvCacheMemory};

use crate::cuda_kv_cache::CudaKvStorage;
use crate::kv_cache::{KvCachePool, PagedKvCacheLayout};
use crate::transfer::KvCacheTransferError;

pub(crate) enum RuntimeKvCache {
    Cuda(KvCachePool<CudaKvStorage>),
}

impl RuntimeKvCache {
    pub(crate) fn cuda(pool: KvCachePool<CudaKvStorage>) -> Self {
        Self::Cuda(pool)
    }

    pub(crate) fn execution_resources(&mut self) -> ModelExecutionResources<'_> {
        match self {
            Self::Cuda(pool) => ModelExecutionResources {
                kv_cache: Some(RuntimeKvCacheView::Cuda {
                    layout: pool.layout(),
                    storage: pool.storage_mut(),
                }),
            },
        }
    }
}

impl fmt::Debug for RuntimeKvCache {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(pool) => formatter
                .debug_struct("RuntimeKvCache")
                .field("backend", &"cuda")
                .field("layout", &pool.layout())
                .finish_non_exhaustive(),
        }
    }
}

impl KvCacheMemoryProvider for RuntimeKvCache {
    type Error = KvCacheTransferError;

    fn transferable_kv_cache_memory(&self) -> Result<TransferableKvCacheMemory, Self::Error> {
        match self {
            Self::Cuda(pool) => pool.transferable_kv_cache_memory().map_err(|error| {
                KvCacheTransferError::Runtime(format!(
                    "CUDA runtime KV memory cannot be described by NexusKV: {error}"
                ))
            }),
        }
    }
}

pub struct ModelExecutionResources<'a> {
    kv_cache: Option<RuntimeKvCacheView<'a>>,
}

impl<'a> ModelExecutionResources<'a> {
    pub fn without_kv_cache() -> Self {
        Self { kv_cache: None }
    }

    pub(crate) fn cuda_kv_cache(self) -> Option<(PagedKvCacheLayout, &'a mut CudaKvStorage)> {
        match self.kv_cache {
            Some(RuntimeKvCacheView::Cuda { layout, storage }) => Some((layout, storage)),
            None => None,
        }
    }
}

enum RuntimeKvCacheView<'a> {
    Cuda {
        layout: PagedKvCacheLayout,
        storage: &'a mut CudaKvStorage,
    },
}
