use crate::backend::{CudaBackend, InitializedRuntimeBackend, RuntimeCapability, RuntimeDtype};
use crate::cpu_hybrid::CpuReferenceHybridDecoder;
use crate::cpu_reference::{CpuReferenceDenseDecoder, CpuReferenceKvCache};
use crate::cuda_dense_decoder::CudaBf16DenseDecoder;
use crate::cuda_kv_cache::allocate_cuda_kv_cache;
use crate::cuda_runtime::CudaEmbeddingLmModel;
use crate::kv_cache::{KvCacheDtype, KvCacheRuntimeLayout, PagedKvCacheLayout};
use crate::model_artifacts::LocalModelArtifacts;
use crate::model_executor::{CpuEmbeddingLmModel, KvCacheAllocationConfig};
use crate::model_runtime::{ModelExecutor, ModelRuntimeConfig, ModelRuntimeLoadError};
use crate::models::{ModelDefinition, ModelExecutionArchitecture};
use crate::runtime_kv_cache::RuntimeKvCache;

pub(crate) struct BackendModelRuntime {
    pub(crate) executor: Box<dyn ModelExecutor>,
    pub(crate) active_kv_cache: Option<RuntimeKvCache>,
}

impl InitializedRuntimeBackend {
    pub(crate) fn create_model_runtime(
        self,
        definition: &ModelDefinition,
        artifacts: &LocalModelArtifacts,
        config: ModelRuntimeConfig,
    ) -> Result<BackendModelRuntime, ModelRuntimeLoadError> {
        match self {
            Self::CpuReference => create_cpu_model_runtime(definition, artifacts, config),
            Self::Cuda(backend) => {
                create_cuda_model_runtime(definition, artifacts, backend, config)
            }
            Self::Unavailable(runtime_backend) => {
                Err(ModelRuntimeLoadError::MissingCapabilities(vec![format!(
                    "{} runtime backend implementation",
                    runtime_backend.as_str()
                )]))
            }
        }
    }

    pub(crate) fn validate_model_runtime(
        &self,
        definition: &ModelDefinition,
        tensor_parallel_size: usize,
    ) -> Vec<String> {
        match self {
            Self::CpuReference => validate_cpu_model_runtime(definition),
            Self::Cuda(backend) => {
                validate_cuda_model_runtime(definition, backend, tensor_parallel_size)
            }
            Self::Unavailable(runtime_backend) => {
                vec![format!(
                    "{} runtime backend implementation",
                    runtime_backend.as_str()
                )]
            }
        }
    }
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

fn create_cpu_model_runtime(
    definition: &ModelDefinition,
    artifacts: &LocalModelArtifacts,
    config: ModelRuntimeConfig,
) -> Result<BackendModelRuntime, ModelRuntimeLoadError> {
    match definition.execution() {
        ModelExecutionArchitecture::Embedding => {
            let executor = CpuEmbeddingLmModel::from_local_model_artifacts(artifacts)
                .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?
                .map(|model| Box::new(model) as Box<dyn ModelExecutor>)
                .ok_or_else(|| {
                    ModelRuntimeLoadError::Load(
                        "embedding model configuration was rejected after registry resolution"
                            .to_string(),
                    )
                })?;
            Ok(BackendModelRuntime {
                executor,
                active_kv_cache: None,
            })
        }
        ModelExecutionArchitecture::Transformer { .. } if definition.hybrid_decoder().is_some() => {
            let executor = Box::new(
                CpuReferenceHybridDecoder::load(definition, artifacts)
                    .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?,
            );
            Ok(BackendModelRuntime {
                executor,
                active_kv_cache: Some(allocate_cpu_active_kv_cache(
                    definition,
                    required_kv_cache_config(config)?,
                )?),
            })
        }
        ModelExecutionArchitecture::Transformer { .. } if definition.dense_decoder().is_some() => {
            let executor = Box::new(
                CpuReferenceDenseDecoder::load(definition, artifacts)
                    .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?,
            );
            Ok(BackendModelRuntime {
                executor,
                active_kv_cache: Some(allocate_cpu_active_kv_cache(
                    definition,
                    required_kv_cache_config(config)?,
                )?),
            })
        }
        ModelExecutionArchitecture::Transformer {
            attention,
            feed_forward,
        } => Err(ModelRuntimeLoadError::MissingCapabilities(vec![format!(
            "{} {} decoder executor for cpu backend",
            attention.family(),
            feed_forward.family()
        )])),
    }
}

fn required_kv_cache_config(
    config: ModelRuntimeConfig,
) -> Result<KvCacheAllocationConfig, ModelRuntimeLoadError> {
    config.kv_cache.ok_or_else(|| {
        ModelRuntimeLoadError::MissingCapabilities(vec![
            "runtime KV cache allocation configuration".to_string(),
        ])
    })
}

fn allocate_cpu_active_kv_cache(
    definition: &ModelDefinition,
    config: KvCacheAllocationConfig,
) -> Result<RuntimeKvCache, ModelRuntimeLoadError> {
    let layout = paged_kv_cache_layout(definition, KvCacheDtype::Float32, config)?;
    let cache = CpuReferenceKvCache::new(layout)
        .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?;
    Ok(RuntimeKvCache::new(cache))
}

fn create_cuda_model_runtime(
    definition: &ModelDefinition,
    artifacts: &LocalModelArtifacts,
    backend: CudaBackend,
    config: ModelRuntimeConfig,
) -> Result<BackendModelRuntime, ModelRuntimeLoadError> {
    match definition.execution() {
        ModelExecutionArchitecture::Embedding => {
            let executor = CudaEmbeddingLmModel::from_local_model_artifacts(artifacts, backend)
                .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?
                .map(|model| Box::new(model) as Box<dyn ModelExecutor>)
                .ok_or_else(|| {
                    ModelRuntimeLoadError::Load(
                        "embedding model configuration was rejected after registry resolution"
                            .to_string(),
                    )
                })?;
            Ok(BackendModelRuntime {
                executor,
                active_kv_cache: None,
            })
        }
        ModelExecutionArchitecture::Transformer { .. } if definition.dense_decoder().is_some() => {
            let active_kv_cache = allocate_cuda_active_kv_cache(
                definition,
                &backend,
                required_kv_cache_config(config)?,
            )?;
            let executor = CudaBf16DenseDecoder::load(definition, artifacts, backend)
                .map(|model| Box::new(model) as Box<dyn ModelExecutor>)
                .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?;
            Ok(BackendModelRuntime {
                executor,
                active_kv_cache: Some(active_kv_cache),
            })
        }
        ModelExecutionArchitecture::Transformer {
            attention,
            feed_forward,
        } => Err(ModelRuntimeLoadError::MissingCapabilities(vec![format!(
            "{} {} decoder executor for cuda backend",
            attention.family(),
            feed_forward.family()
        )])),
    }
}

fn allocate_cuda_active_kv_cache(
    definition: &ModelDefinition,
    backend: &CudaBackend,
    config: KvCacheAllocationConfig,
) -> Result<RuntimeKvCache, ModelRuntimeLoadError> {
    let layout = paged_kv_cache_layout(definition, KvCacheDtype::Bfloat16, config)?;
    let pool = allocate_cuda_kv_cache(backend.context(), layout.runtime(), layout.page_count())
        .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?;
    Ok(RuntimeKvCache::new(pool))
}

fn paged_kv_cache_layout(
    definition: &ModelDefinition,
    dtype: KvCacheDtype,
    config: KvCacheAllocationConfig,
) -> Result<PagedKvCacheLayout, ModelRuntimeLoadError> {
    validate_cache_config(config)?;
    let model_layout = definition.kv_cache_layout().ok_or_else(|| {
        ModelRuntimeLoadError::MissingCapabilities(vec![
            "model paged KV cache geometry".to_string(),
        ])
    })?;
    let bytes_per_token = model_layout
        .token_size_bytes(dtype)
        .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?;
    let page_size_bytes = config
        .page_size
        .checked_mul(bytes_per_token)
        .ok_or_else(|| ModelRuntimeLoadError::Load("KV page size overflowed".to_string()))?;
    let runtime_layout = KvCacheRuntimeLayout {
        dtype,
        page_size: config.page_size,
        num_layers: model_layout.num_layers,
        kv_heads: model_layout.kv_heads,
        head_dim: model_layout.head_dim,
        kv_tensors_per_token: model_layout.kv_tensors_per_token,
        bytes_per_token,
        page_size_bytes,
    };
    let page_count = config.slot_capacity / config.page_size;
    PagedKvCacheLayout::new(runtime_layout, page_count)
        .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))
}

fn validate_cpu_model_runtime(definition: &ModelDefinition) -> Vec<String> {
    match definition.execution() {
        ModelExecutionArchitecture::Embedding => Vec::new(),
        ModelExecutionArchitecture::Transformer { .. }
            if definition.dense_decoder().is_some() || definition.hybrid_decoder().is_some() =>
        {
            Vec::new()
        }
        ModelExecutionArchitecture::Transformer {
            attention,
            feed_forward,
        } => vec![
            format!(
                "{} {} decoder executor for cpu backend",
                attention.family(),
                feed_forward.family()
            ),
            "runtime-owned KV cache allocation".to_string(),
        ],
    }
}

fn validate_cuda_model_runtime(
    definition: &ModelDefinition,
    backend: &CudaBackend,
    tensor_parallel_size: usize,
) -> Vec<String> {
    if matches!(
        definition.execution(),
        ModelExecutionArchitecture::Embedding
    ) {
        return Vec::new();
    }
    let mut missing = Vec::new();
    let capability = backend.capabilities();
    if definition.dense_decoder().is_none() {
        if let ModelExecutionArchitecture::Transformer {
            attention,
            feed_forward,
        } = definition.execution()
        {
            missing.push(format!(
                "{} {} decoder executor for cuda backend",
                attention.family(),
                feed_forward.family()
            ));
            missing.push("runtime-owned KV cache allocation".to_string());
        }
        return missing;
    }
    if tensor_parallel_size != 1 {
        missing.push(format!(
            "CUDA dense decoder tensor parallel size 1 (requested {tensor_parallel_size})"
        ));
    }
    if !capability.supported_dtypes.contains(&RuntimeDtype::Bf16) {
        missing.push("CUDA BF16 compute capability 8.0 or newer".to_string());
    }
    missing
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
