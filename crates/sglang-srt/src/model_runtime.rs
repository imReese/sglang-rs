use std::fmt;

use crate::backend::{InitializedRuntimeBackend, RuntimeBackend, RuntimeCapability, RuntimeDtype};
use crate::cpu_hybrid::CpuReferenceHybridDecoder;
use crate::cpu_reference::CpuReferenceDenseDecoder;
use crate::cuda_dense_decoder::CudaBf16DenseDecoder;
use crate::cuda_kv_cache::allocate_cuda_kv_cache;
use crate::cuda_runtime::CudaEmbeddingLmModel;
use crate::model_artifacts::LocalModelArtifacts;
use crate::model_executor::{
    CpuEmbeddingLmModel, ForwardModel, KvCacheAllocationConfig, ModelForwardError,
    ModelForwardOutput, ModelWorkerBatch,
};
use crate::models::{ModelDefinition, ModelExecutionArchitecture};
use crate::runtime_kv_cache::{ModelExecutionResources, RuntimeKvCache};
use crate::transfer::{KvCacheDtype, KvCacheRuntimeLayout};
use crate::worker::WorkerWeightUpdateRequest;

pub(crate) trait ModelExecutor: ForwardModel + fmt::Debug + Send {
    fn runtime_capability(&self) -> RuntimeCapability;
    fn execution_dtype(&self) -> RuntimeDtype;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ModelRuntimeConfig {
    pub(crate) tensor_parallel_size: usize,
    pub(crate) kv_cache: Option<KvCacheAllocationConfig>,
}

impl ModelExecutor for CpuEmbeddingLmModel {
    fn runtime_capability(&self) -> RuntimeCapability {
        RuntimeCapability::cpu_reference("cpu-embedding-lm", false)
    }

    fn execution_dtype(&self) -> RuntimeDtype {
        RuntimeDtype::F32
    }
}

impl ModelExecutor for CudaEmbeddingLmModel {
    fn runtime_capability(&self) -> RuntimeCapability {
        CudaEmbeddingLmModel::runtime_capability(self)
    }

    fn execution_dtype(&self) -> RuntimeDtype {
        RuntimeDtype::F32
    }
}

impl ModelExecutor for CudaBf16DenseDecoder {
    fn runtime_capability(&self) -> RuntimeCapability {
        CudaBf16DenseDecoder::runtime_capability(self)
    }

    fn execution_dtype(&self) -> RuntimeDtype {
        RuntimeDtype::Bf16
    }
}

impl ModelExecutor for CpuReferenceDenseDecoder {
    fn runtime_capability(&self) -> RuntimeCapability {
        CpuReferenceDenseDecoder::runtime_capability(self)
    }

    fn execution_dtype(&self) -> RuntimeDtype {
        CpuReferenceDenseDecoder::execution_dtype(self)
    }
}

impl ModelExecutor for CpuReferenceHybridDecoder {
    fn runtime_capability(&self) -> RuntimeCapability {
        CpuReferenceHybridDecoder::runtime_capability(self)
    }

    fn execution_dtype(&self) -> RuntimeDtype {
        CpuReferenceHybridDecoder::execution_dtype(self)
    }
}

#[derive(Debug)]
pub(crate) struct LoadedModelRuntime {
    runtime_backend: RuntimeBackend,
    executor: Box<dyn ModelExecutor>,
    config: ModelRuntimeConfig,
    runtime_kv_cache: Option<RuntimeKvCache>,
}

impl LoadedModelRuntime {
    pub(crate) fn load(
        definition: &ModelDefinition,
        artifacts: &LocalModelArtifacts,
        backend: InitializedRuntimeBackend,
        config: ModelRuntimeConfig,
    ) -> Result<Self, ModelRuntimeLoadError> {
        validate_runtime_support(definition, &backend, config.tensor_parallel_size)?;
        let runtime_backend = backend.runtime_backend();
        let runtime_kv_cache = allocate_runtime_kv_cache(definition, &backend, config.kv_cache)?;
        let executor = load_executor(definition, artifacts, backend)?;
        let capability = executor.runtime_capability();
        let execution_dtype = executor.execution_dtype();
        if !definition.supported_dtypes().contains(&execution_dtype) {
            return Err(ModelRuntimeLoadError::MissingCapabilities(vec![format!(
                "model execution dtype {execution_dtype}"
            )]));
        }
        capability
            .validate_requirements(&definition.runtime_requirements(
                execution_dtype,
                config.tensor_parallel_size,
                None,
            ))
            .map_err(|mismatch| ModelRuntimeLoadError::MissingCapabilities(mismatch.missing))?;
        Ok(Self {
            runtime_backend,
            executor,
            config,
            runtime_kv_cache,
        })
    }

    pub(crate) fn runtime_backend(&self) -> RuntimeBackend {
        self.runtime_backend
    }

    pub(crate) fn runtime_capability(&self) -> RuntimeCapability {
        self.executor.runtime_capability()
    }

    pub(crate) fn execution_dtype(&self) -> RuntimeDtype {
        self.executor.execution_dtype()
    }

    pub(crate) fn config(&self) -> ModelRuntimeConfig {
        self.config
    }

    pub(crate) fn take_runtime_kv_cache(&mut self) -> Option<RuntimeKvCache> {
        self.runtime_kv_cache.take()
    }

    pub(crate) fn has_runtime_kv_cache(&self) -> bool {
        self.runtime_kv_cache.is_some()
    }
}

impl ForwardModel for LoadedModelRuntime {
    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        let resources = self
            .runtime_kv_cache
            .as_mut()
            .map(RuntimeKvCache::execution_resources)
            .unwrap_or_else(ModelExecutionResources::without_kv_cache);
        self.executor.forward_with_resources(batch, resources)
    }

    fn forward_with_resources(
        &mut self,
        batch: &ModelWorkerBatch,
        resources: ModelExecutionResources<'_>,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        self.executor.forward_with_resources(batch, resources)
    }

    fn complete_request(&mut self, request_id: &crate::types::RequestId) {
        self.executor.complete_request(request_id);
    }

    fn update_weights_from_disk(
        &mut self,
        request: &WorkerWeightUpdateRequest,
    ) -> Result<(), ModelForwardError> {
        self.executor.update_weights_from_disk(request)
    }
}

pub(crate) fn validate_runtime_support(
    definition: &ModelDefinition,
    backend: &InitializedRuntimeBackend,
    tensor_parallel_size: usize,
) -> Result<(), ModelRuntimeLoadError> {
    if let InitializedRuntimeBackend::Unavailable(runtime_backend) = backend {
        return Err(ModelRuntimeLoadError::MissingCapabilities(vec![format!(
            "{} runtime backend implementation",
            runtime_backend.as_str()
        )]));
    }

    definition
        .validate_tensor_parallel(tensor_parallel_size)
        .map_err(|message| ModelRuntimeLoadError::MissingCapabilities(vec![message]))?;

    let backend_capabilities = backend.capabilities();
    let execution_dtype = definition
        .supported_dtypes()
        .iter()
        .copied()
        .find(|dtype| backend_capabilities.supported_dtypes.contains(dtype));
    let mut missing = execution_dtype
        .is_none()
        .then(|| {
            format!(
                "execution dtype supported by both model ({}) and {} backend ({})",
                format_dtypes(definition.supported_dtypes()),
                backend.runtime_backend(),
                format_dtypes(&backend_capabilities.supported_dtypes)
            )
        })
        .into_iter()
        .collect::<Vec<_>>();

    if let ModelExecutionArchitecture::Transformer {
        attention,
        feed_forward,
    } = definition.execution()
    {
        let has_cpu_reference_executor = matches!(backend, InitializedRuntimeBackend::CpuReference)
            && matches!(
                (attention, feed_forward),
                (
                    crate::models::AttentionArchitecture::MultiHead { .. },
                    crate::models::FeedForwardArchitecture::Dense { .. }
                ) | (
                    crate::models::AttentionArchitecture::Hybrid { .. },
                    crate::models::FeedForwardArchitecture::Dense { .. }
                )
            )
            && (definition.dense_decoder().is_some() || definition.hybrid_decoder().is_some());
        let has_cuda_dense_executor = matches!(backend, InitializedRuntimeBackend::Cuda(_))
            && tensor_parallel_size == 1
            && definition.dense_decoder().is_some()
            && matches!(
                (attention, feed_forward),
                (
                    crate::models::AttentionArchitecture::MultiHead { .. },
                    crate::models::FeedForwardArchitecture::Dense { .. }
                )
            )
            && backend_capabilities
                .supported_dtypes
                .contains(&RuntimeDtype::Bf16);
        if !has_cpu_reference_executor && !has_cuda_dense_executor {
            missing.push(format!(
                "{} {} decoder executor for {} backend",
                attention.family(),
                feed_forward.family(),
                backend.runtime_backend()
            ));
            missing.push("runtime-owned KV cache allocation".to_string());
        }
        if matches!(backend, InitializedRuntimeBackend::Cuda(_))
            && definition.dense_decoder().is_some()
            && tensor_parallel_size != 1
        {
            missing.push(format!(
                "CUDA dense decoder tensor parallel size 1 (requested {tensor_parallel_size})"
            ));
        }
        if matches!(backend, InitializedRuntimeBackend::Cuda(_))
            && definition.dense_decoder().is_some()
            && !backend_capabilities
                .supported_dtypes
                .contains(&RuntimeDtype::Bf16)
        {
            missing.push("CUDA BF16 compute capability 8.0 or newer".to_string());
        }
    }

    if missing.is_empty() {
        Ok(())
    } else {
        Err(ModelRuntimeLoadError::MissingCapabilities(missing))
    }
}

fn load_executor(
    definition: &ModelDefinition,
    artifacts: &LocalModelArtifacts,
    backend: InitializedRuntimeBackend,
) -> Result<Box<dyn ModelExecutor>, ModelRuntimeLoadError> {
    match definition.execution() {
        ModelExecutionArchitecture::Embedding => load_embedding_executor(artifacts, backend),
        ModelExecutionArchitecture::Transformer {
            attention,
            feed_forward,
        } => match backend {
            InitializedRuntimeBackend::CpuReference if definition.hybrid_decoder().is_some() => {
                CpuReferenceHybridDecoder::load(definition, artifacts)
                    .map(|model| Box::new(model) as Box<dyn ModelExecutor>)
                    .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))
            }
            InitializedRuntimeBackend::CpuReference if definition.dense_decoder().is_some() => {
                CpuReferenceDenseDecoder::load(definition, artifacts)
                    .map(|model| Box::new(model) as Box<dyn ModelExecutor>)
                    .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))
            }
            InitializedRuntimeBackend::CpuReference => {
                Err(ModelRuntimeLoadError::MissingCapabilities(vec![format!(
                    "{} {} decoder executor for cpu backend",
                    attention.family(),
                    feed_forward.family()
                )]))
            }
            InitializedRuntimeBackend::Cuda(backend) if definition.dense_decoder().is_some() => {
                CudaBf16DenseDecoder::load(definition, artifacts, backend)
                    .map(|model| Box::new(model) as Box<dyn ModelExecutor>)
                    .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))
            }
            InitializedRuntimeBackend::Cuda(_) => {
                Err(ModelRuntimeLoadError::MissingCapabilities(vec![format!(
                    "{} {} decoder executor for cuda backend",
                    attention.family(),
                    feed_forward.family()
                )]))
            }
            InitializedRuntimeBackend::Unavailable(runtime_backend) => {
                Err(ModelRuntimeLoadError::MissingCapabilities(vec![format!(
                    "{} runtime backend implementation",
                    runtime_backend.as_str()
                )]))
            }
        },
    }
}

fn allocate_runtime_kv_cache(
    definition: &ModelDefinition,
    backend: &InitializedRuntimeBackend,
    config: Option<KvCacheAllocationConfig>,
) -> Result<Option<RuntimeKvCache>, ModelRuntimeLoadError> {
    let InitializedRuntimeBackend::Cuda(backend) = backend else {
        return Ok(None);
    };
    if definition.dense_decoder().is_none() {
        return Ok(None);
    }

    let config = config.ok_or_else(|| {
        ModelRuntimeLoadError::MissingCapabilities(vec![
            "runtime KV cache allocation configuration".to_string(),
        ])
    })?;
    validate_cache_config(config)?;
    let model_layout = definition.kv_cache_layout().ok_or_else(|| {
        ModelRuntimeLoadError::MissingCapabilities(vec![
            "model paged KV cache geometry".to_string(),
        ])
    })?;
    let bytes_per_token = model_layout
        .token_size_bytes(KvCacheDtype::Bfloat16)
        .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?;
    let page_size_bytes = config
        .page_size
        .checked_mul(bytes_per_token)
        .ok_or_else(|| ModelRuntimeLoadError::Load("KV page size overflowed".to_string()))?;
    let runtime_layout = KvCacheRuntimeLayout {
        dtype: KvCacheDtype::Bfloat16,
        page_size: config.page_size,
        num_layers: model_layout.num_layers,
        kv_heads: model_layout.kv_heads,
        head_dim: model_layout.head_dim,
        kv_tensors_per_token: model_layout.kv_tensors_per_token,
        bytes_per_token,
        page_size_bytes,
    };
    let page_count = config.slot_capacity / config.page_size;
    let pool = allocate_cuda_kv_cache(backend.context(), runtime_layout, page_count)
        .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?;
    Ok(Some(RuntimeKvCache::cuda(pool)))
}

fn validate_cache_config(config: KvCacheAllocationConfig) -> Result<(), ModelRuntimeLoadError> {
    if config.slot_capacity == 0 {
        return Err(ModelRuntimeLoadError::MissingCapabilities(vec![
            "non-zero KV cache slot capacity".to_string(),
        ]));
    }
    if config.page_size == 0 {
        return Err(ModelRuntimeLoadError::MissingCapabilities(vec![
            "non-zero KV cache page size".to_string(),
        ]));
    }
    if !config.slot_capacity.is_multiple_of(config.page_size) {
        return Err(ModelRuntimeLoadError::MissingCapabilities(vec![format!(
            "KV cache slot capacity {} divisible by page size {}",
            config.slot_capacity, config.page_size
        )]));
    }
    Ok(())
}

fn format_dtypes(dtypes: &[RuntimeDtype]) -> String {
    if dtypes.is_empty() {
        "none".to_string()
    } else {
        dtypes
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn load_embedding_executor(
    artifacts: &LocalModelArtifacts,
    backend: InitializedRuntimeBackend,
) -> Result<Box<dyn ModelExecutor>, ModelRuntimeLoadError> {
    match backend {
        InitializedRuntimeBackend::CpuReference => {
            CpuEmbeddingLmModel::from_local_model_artifacts(artifacts)
                .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?
                .map(|model| Box::new(model) as Box<dyn ModelExecutor>)
                .ok_or_else(|| {
                    ModelRuntimeLoadError::Load(
                        "embedding model configuration was rejected after registry resolution"
                            .to_string(),
                    )
                })
        }
        InitializedRuntimeBackend::Cuda(backend) => {
            CudaEmbeddingLmModel::from_local_model_artifacts(artifacts, backend)
                .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?
                .map(|model| Box::new(model) as Box<dyn ModelExecutor>)
                .ok_or_else(|| {
                    ModelRuntimeLoadError::Load(
                        "embedding model configuration was rejected after registry resolution"
                            .to_string(),
                    )
                })
        }
        InitializedRuntimeBackend::Unavailable(runtime_backend) => {
            Err(ModelRuntimeLoadError::MissingCapabilities(vec![format!(
                "{} runtime backend implementation",
                runtime_backend.as_str()
            )]))
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ModelRuntimeLoadError {
    MissingCapabilities(Vec<String>),
    Load(String),
}

impl fmt::Display for ModelRuntimeLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingCapabilities(missing) => {
                write!(formatter, "missing capabilities: {}", missing.join(", "))
            }
            Self::Load(message) => formatter.write_str(message),
        }
    }
}
