use std::fmt;
use std::sync::{Arc, Mutex};

use nexus_transfer::{KvCacheMemoryProvider, TransferableKvCacheMemory};
use sglang_kernel::nccl::{NcclCommunicator, NcclLibrary, NcclRank, NcclUniqueId};

use crate::backend::{
    CapabilityStatus, CudaBackend, RuntimeBackend, RuntimeCapability, RuntimeDtype,
};
use crate::backend_model::paged_kv_cache_layout_for_model;
use crate::cuda_dense_decoder::CudaBf16DenseDecoder;
use crate::cuda_execution_resources::CudaExecutionResources;
use crate::cuda_kv_cache::allocate_cuda_kv_cache;
use crate::kv_cache::{KvCacheDtype, KvCacheModelLayout, PagedKvCacheLayout};
use crate::model_artifacts::LocalModelArtifacts;
use crate::model_executor::{
    ForwardModel, ModelForwardError, ModelForwardOutput, ModelWorkerBatch,
};
use crate::model_runtime::{
    BackendExecutionResources, BackendExecutionRuntime, BackendModelExecutor, ModelRuntimeConfig,
    ModelRuntimeLoadError,
};
use crate::models::ModelDefinition;
use crate::parallel::{
    RankWorker, TensorParallelRank, TensorParallelTopology, WorkerGroup, WorkerGroupError,
};
use crate::runtime_kv_cache::RuntimeKvCacheMetadata;
use crate::transfer::KvCacheTransferError;
use crate::transformer_parallel::DenseTensorParallelPlan;
use crate::types::RequestId;
use crate::worker::WorkerWeightUpdateRequest;

pub(crate) struct CudaTensorParallelDenseRuntime {
    workers: WorkerGroup<CudaDenseRankWorker>,
    local_kv_layout: PagedKvCacheLayout,
    capability: RuntimeCapability,
}

impl CudaTensorParallelDenseRuntime {
    pub(crate) fn launch(
        definition: &ModelDefinition,
        artifacts: &LocalModelArtifacts,
        rank_zero_backend: CudaBackend,
        config: ModelRuntimeConfig,
    ) -> Result<Self, ModelRuntimeLoadError> {
        if config.tensor_parallel_node_count != 1 || config.tensor_parallel_node_rank != 0 {
            return Err(ModelRuntimeLoadError::MissingCapabilities(vec![
                "multi-node CUDA tensor parallel rendezvous and rank lifecycle; only single-node TP is currently implemented"
                    .to_string(),
            ]));
        }
        let kv_config = config.kv_cache.ok_or_else(|| {
            ModelRuntimeLoadError::MissingCapabilities(vec![
                "runtime KV cache allocation configuration".to_string(),
            ])
        })?;
        let topology = TensorParallelTopology::new(
            config.tensor_parallel_size,
            config.tensor_parallel_node_count,
            config.tensor_parallel_node_rank,
            config.device_placement.base_gpu_id,
            config.device_placement.gpu_id_step,
        )
        .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?;
        let first_rank = topology.local_ranks().first().copied().ok_or_else(|| {
            ModelRuntimeLoadError::Load("tensor parallel topology has no local ranks".to_string())
        })?;
        if first_rank.global_rank() != config.device_placement.tensor_parallel_rank
            || first_rank.local_rank() != config.device_placement.local_rank
            || first_rank.device_ordinal() != rank_zero_backend.device().ordinal
        {
            return Err(ModelRuntimeLoadError::Load(format!(
                "initialized CUDA backend placement does not match the first tensor parallel rank: placement global/local/device={}/{}/{}, topology={}/{}/{}",
                config.device_placement.tensor_parallel_rank,
                config.device_placement.local_rank,
                rank_zero_backend.device().ordinal,
                first_rank.global_rank(),
                first_rank.local_rank(),
                first_rank.device_ordinal()
            )));
        }

        let mut rank_zero_backend = Some(rank_zero_backend);
        let mut rank_backends = Vec::with_capacity(topology.local_ranks().len());
        for rank in topology.local_ranks().iter().copied() {
            let backend = if rank.global_rank() == first_rank.global_rank() {
                rank_zero_backend.take().ok_or_else(|| {
                    ModelRuntimeLoadError::Load(
                        "rank-zero CUDA backend was already consumed during preflight".to_string(),
                    )
                })?
            } else {
                CudaBackend::initialize(rank.device_ordinal())
                    .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?
            };
            if backend.device().ordinal != rank.device_ordinal() {
                return Err(ModelRuntimeLoadError::Load(format!(
                    "tensor parallel rank {} initialized CUDA device {}, expected {}",
                    rank.global_rank(),
                    backend.device().ordinal,
                    rank.device_ordinal()
                )));
            }
            if !backend
                .capabilities()
                .supported_dtypes
                .contains(&RuntimeDtype::Bf16)
            {
                return Err(ModelRuntimeLoadError::MissingCapabilities(vec![format!(
                    "CUDA BF16 compute capability 8.0 or newer on tensor parallel rank {} device {}",
                    rank.global_rank(),
                    rank.device_ordinal()
                )]));
            }
            rank_backends.push(Some(backend));
        }

        let mut capability = rank_backends
            .iter()
            .flatten()
            .min_by_key(|backend| backend.device().compute_capability)
            .expect("tensor parallel topology has at least one local rank")
            .capabilities();
        capability.runtime_name = "cuda-bf16-dense-tensor-parallel";
        capability.supports_forward = true;
        capability.supports_transferable_kv = false;
        capability.supported_dtypes = vec![RuntimeDtype::Bf16];
        capability.attention_backends = vec!["cuda-paged-bf16"];
        capability.tensor_parallel = CapabilityStatus::Supported;
        capability.kv_cache_memory_registration = CapabilityStatus::Unsupported;
        capability.mooncake = CapabilityStatus::Unsupported;
        capability.rdma = CapabilityStatus::Unsupported;

        let nccl_library =
            NcclLibrary::load().map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?;
        let unique_id = nccl_library
            .unique_id()
            .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?;
        drop(nccl_library);

        let definition = definition.clone();
        let artifacts = artifacts.clone();
        let global_kv_layout = definition.kv_cache_layout().ok_or_else(|| {
            ModelRuntimeLoadError::MissingCapabilities(vec![
                "model paged KV cache geometry".to_string(),
            ])
        })?;
        let rank_backends = Arc::new(Mutex::new(rank_backends));
        let initializer_definition = definition.clone();
        let initializer_artifacts = artifacts.clone();
        let initializer_backends = Arc::clone(&rank_backends);
        let workers = WorkerGroup::launch(&topology, move |rank| {
            initialize_rank(
                rank,
                unique_id,
                &initializer_definition,
                &initializer_artifacts,
                global_kv_layout,
                kv_config,
                &initializer_backends,
            )
        })
        .map_err(worker_group_load_error)?;
        let local_plan = DenseTensorParallelPlan::from_execution(
            definition.execution(),
            topology.local_ranks()[0],
        )
        .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?;
        let local_model_layout = local_dense_kv_layout(global_kv_layout, local_plan);
        let local_kv_layout =
            paged_kv_cache_layout_for_model(local_model_layout, KvCacheDtype::Bfloat16, kv_config)?;
        Ok(Self {
            workers,
            local_kv_layout,
            capability,
        })
    }

    fn execute(
        &mut self,
        command: CudaDenseRankCommand,
    ) -> Result<Vec<CudaDenseRankOutput>, ModelForwardError> {
        self.workers
            .execute_all(command)
            .map(|outputs| {
                outputs
                    .into_iter()
                    .map(|output| output.into_output())
                    .collect()
            })
            .map_err(|error| ModelForwardError::Runtime(error.to_string()))
    }
}

impl fmt::Debug for CudaTensorParallelDenseRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaTensorParallelDenseRuntime")
            .field("workers", &self.workers)
            .field("local_kv_layout", &self.local_kv_layout)
            .finish_non_exhaustive()
    }
}

impl ForwardModel for CudaTensorParallelDenseRuntime {
    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        let outputs = self.execute(CudaDenseRankCommand::Forward(Arc::new(batch.clone())))?;
        let mut outputs = outputs.into_iter();
        let first = match outputs.next() {
            Some(CudaDenseRankOutput::Forward(output)) => output,
            Some(CudaDenseRankOutput::Complete) => {
                return Err(ModelForwardError::Runtime(
                    "tensor parallel rank returned completion acknowledgement for forward"
                        .to_string(),
                ));
            }
            None => {
                return Err(ModelForwardError::Runtime(
                    "tensor parallel worker group returned no forward output".to_string(),
                ));
            }
        };
        for output in outputs {
            let CudaDenseRankOutput::Forward(output) = output else {
                return Err(ModelForwardError::Runtime(
                    "tensor parallel rank returned completion acknowledgement for forward"
                        .to_string(),
                ));
            };
            if output != first {
                return Err(ModelForwardError::Runtime(
                    "tensor parallel ranks produced different replicated logits after collective reduction"
                        .to_string(),
                ));
            }
        }
        Ok(first)
    }

    fn complete_request(&mut self, request_id: &RequestId) {
        let _ = self.execute(CudaDenseRankCommand::Complete(request_id.clone()));
    }

    fn update_weights_from_disk(
        &mut self,
        _request: &WorkerWeightUpdateRequest,
    ) -> Result<(), ModelForwardError> {
        Err(ModelForwardError::Runtime(
            "tensor parallel CUDA runtime requires a coordinated rank reload for update_weights_from_disk"
                .to_string(),
        ))
    }
}

impl KvCacheMemoryProvider for CudaTensorParallelDenseRuntime {
    type Error = KvCacheTransferError;

    fn transferable_kv_cache_memory(&self) -> Result<TransferableKvCacheMemory, Self::Error> {
        Err(KvCacheTransferError::Runtime(
            "tensor parallel CUDA KV memory is rank-local; NexusKV exposes one device location per descriptor, and rank-wise PD transfer descriptors are not implemented"
                .to_string(),
        ))
    }
}

impl RuntimeKvCacheMetadata for CudaTensorParallelDenseRuntime {
    fn active_kv_cache_layout(&self) -> Option<PagedKvCacheLayout> {
        Some(self.local_kv_layout)
    }
}

impl BackendExecutionRuntime for CudaTensorParallelDenseRuntime {
    fn resources_backend(&self) -> RuntimeBackend {
        RuntimeBackend::Cuda
    }

    fn recurrent_state_layout(&self) -> Option<crate::models::RecurrentStateLayout> {
        None
    }

    fn runtime_capability(&self) -> RuntimeCapability {
        self.capability.clone()
    }

    fn execution_dtype(&self) -> RuntimeDtype {
        RuntimeDtype::Bf16
    }
}

struct CudaDenseRankWorker {
    decoder: CudaBf16DenseDecoder,
    resources: CudaExecutionResources,
}

impl RankWorker for CudaDenseRankWorker {
    type Command = CudaDenseRankCommand;
    type Output = CudaDenseRankOutput;
    type Error = ModelForwardError;

    fn execute(&mut self, command: Self::Command) -> Result<Self::Output, Self::Error> {
        match command {
            CudaDenseRankCommand::Forward(batch) => self
                .decoder
                .forward(&batch, &mut self.resources)
                .map(CudaDenseRankOutput::Forward),
            CudaDenseRankCommand::Complete(request_id) => {
                self.resources.complete_request(&request_id);
                Ok(CudaDenseRankOutput::Complete)
            }
        }
    }

    fn shutdown(&mut self) -> Result<(), Self::Error> {
        self.decoder.shutdown_collective();
        Ok(())
    }
}

#[derive(Clone)]
enum CudaDenseRankCommand {
    Forward(Arc<ModelWorkerBatch>),
    Complete(RequestId),
}

enum CudaDenseRankOutput {
    Forward(ModelForwardOutput),
    Complete,
}

fn initialize_rank(
    rank: TensorParallelRank,
    unique_id: NcclUniqueId,
    definition: &ModelDefinition,
    artifacts: &LocalModelArtifacts,
    global_kv_layout: KvCacheModelLayout,
    kv_config: crate::model_executor::KvCacheAllocationConfig,
    rank_backends: &Mutex<Vec<Option<CudaBackend>>>,
) -> Result<CudaDenseRankWorker, String> {
    let backend = rank_backends
        .lock()
        .map_err(|_| "tensor parallel CUDA backend lock is poisoned".to_string())?
        .get_mut(rank.local_rank())
        .and_then(Option::take)
        .ok_or_else(|| {
            format!(
                "CUDA backend for tensor parallel local rank {} is missing or already consumed",
                rank.local_rank()
            )
        })?;
    let communicator = NcclCommunicator::initialize(
        NcclLibrary::load().map_err(|error| error.to_string())?,
        backend.context(),
        unique_id,
        NcclRank::new(rank.world_size(), rank.global_rank()).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let parallel = DenseTensorParallelPlan::from_execution(definition.execution(), rank)
        .map_err(|error| error.to_string())?;
    let local_model_layout = local_dense_kv_layout(global_kv_layout, parallel);
    let local_kv_layout =
        paged_kv_cache_layout_for_model(local_model_layout, KvCacheDtype::Bfloat16, kv_config)
            .map_err(|error| error.to_string())?;
    let kv_cache = allocate_cuda_kv_cache(
        backend.context(),
        local_kv_layout.runtime(),
        local_kv_layout.page_count(),
    )
    .map_err(|error| error.to_string())?;
    let decoder = CudaBf16DenseDecoder::load_tensor_parallel(
        definition,
        artifacts,
        backend,
        rank,
        Some(communicator),
    )
    .map_err(|error| error.to_string())?;
    if decoder.local_kv_head_count() != local_model_layout.kv_heads {
        return Err(format!(
            "rank-local decoder KV head count {} does not match allocated KV layout {}",
            decoder.local_kv_head_count(),
            local_model_layout.kv_heads
        ));
    }
    Ok(CudaDenseRankWorker {
        decoder,
        resources: CudaExecutionResources::new(kv_cache, None),
    })
}

fn local_dense_kv_layout(
    global: KvCacheModelLayout,
    parallel: DenseTensorParallelPlan,
) -> KvCacheModelLayout {
    KvCacheModelLayout::multi_tensor(
        global.num_layers,
        parallel.local_kv_head_count(),
        global.head_dim,
        global.kv_tensors_per_token,
    )
}

fn worker_group_load_error(error: WorkerGroupError) -> ModelRuntimeLoadError {
    ModelRuntimeLoadError::Load(format!(
        "CUDA tensor parallel worker group failed to initialize: {error}"
    ))
}
