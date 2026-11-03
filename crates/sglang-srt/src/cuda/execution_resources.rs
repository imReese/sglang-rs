use crate::cuda_kv_cache::CudaKvStorage;
use crate::cuda_recurrent_state::CudaRecurrentStateStorage;
use crate::kv_cache::KvCachePool;
use crate::model_runtime::ModelExecutionResources;

pub(crate) type CudaExecutionResources =
    ModelExecutionResources<KvCachePool<CudaKvStorage>, CudaRecurrentStateStorage>;
