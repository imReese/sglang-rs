use nexus_transfer::{KvCacheMemoryProvider, TransferableKvCacheMemory};

use crate::backend::RuntimeBackend;
use crate::cuda_kv_cache::CudaKvStorage;
use crate::cuda_recurrent_state::CudaRecurrentStateStorage;
use crate::kv_cache::KvCachePool;
use crate::model_runtime::BackendExecutionResources;
use crate::runtime_kv_cache::{RuntimeKvCache, RuntimeKvCacheMetadata};
use crate::transfer::KvCacheTransferError;
use crate::types::RequestId;

#[derive(Debug)]
pub(crate) struct CudaExecutionResources {
    active_kv_cache: RuntimeKvCache<KvCachePool<CudaKvStorage>>,
    recurrent_state: Option<CudaRecurrentStateStorage>,
}

impl CudaExecutionResources {
    pub(crate) fn new(
        active_kv_cache: KvCachePool<CudaKvStorage>,
        recurrent_state: Option<CudaRecurrentStateStorage>,
    ) -> Self {
        Self {
            active_kv_cache: RuntimeKvCache::new(active_kv_cache),
            recurrent_state,
        }
    }

    pub(crate) fn active_kv_cache_mut(&mut self) -> &mut KvCachePool<CudaKvStorage> {
        self.active_kv_cache.allocation_mut()
    }

    pub(crate) fn has_recurrent_state(&self) -> bool {
        self.recurrent_state.is_some()
    }

    pub(crate) fn execution_memory_mut(
        &mut self,
    ) -> (
        &mut KvCachePool<CudaKvStorage>,
        Option<&mut CudaRecurrentStateStorage>,
    ) {
        (
            self.active_kv_cache.allocation_mut(),
            self.recurrent_state.as_mut(),
        )
    }
}

impl KvCacheMemoryProvider for CudaExecutionResources {
    type Error = KvCacheTransferError;

    fn transferable_kv_cache_memory(&self) -> Result<TransferableKvCacheMemory, Self::Error> {
        self.active_kv_cache.transferable_kv_cache_memory()
    }
}

impl RuntimeKvCacheMetadata for CudaExecutionResources {
    fn active_kv_cache_layout(&self) -> Option<crate::kv_cache::PagedKvCacheLayout> {
        self.active_kv_cache.active_kv_cache_layout()
    }
}

impl BackendExecutionResources for CudaExecutionResources {
    fn runtime_backend(&self) -> RuntimeBackend {
        RuntimeBackend::Cuda
    }

    fn complete_request(&mut self, request_id: &RequestId) {
        if let Some(recurrent_state) = self.recurrent_state.as_mut() {
            recurrent_state.release_request(request_id);
        }
    }

    fn recurrent_state_layout(&self) -> Option<crate::models::RecurrentStateLayout> {
        self.recurrent_state
            .as_ref()
            .map(CudaRecurrentStateStorage::layout)
    }
}
