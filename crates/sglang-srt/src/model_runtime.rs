use std::fmt;

use crate::backend::{InitializedRuntimeBackend, RuntimeBackend, RuntimeCapability, RuntimeDtype};
use crate::cpu_hybrid::CpuReferenceHybridDecoder;
use crate::cpu_reference::CpuReferenceDenseDecoder;
use crate::cuda_dense_decoder::CudaBf16DenseDecoder;
use crate::cuda_runtime::CudaEmbeddingLmModel;
use crate::model_artifacts::LocalModelArtifacts;
use crate::model_executor::{
    CpuEmbeddingLmModel, ForwardModel, KvCacheAllocationConfig, ModelForwardError,
    ModelForwardOutput, ModelWorkerBatch,
};
use crate::models::{ModelDefinition, ModelExecutionArchitecture};
use crate::transfer::TransferableKvCacheMemory;
use crate::worker::WorkerWeightUpdateRequest;

pub(crate) trait ModelExecutor: ForwardModel + fmt::Debug + Send {
    fn runtime_capability(&self) -> RuntimeCapability;
    fn execution_dtype(&self) -> RuntimeDtype;

    fn transferable_kv_cache_memory(&self) -> Option<&TransferableKvCacheMemory> {
        None
    }
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

    fn transferable_kv_cache_memory(&self) -> Option<&TransferableKvCacheMemory> {
        Some(CudaBf16DenseDecoder::transferable_kv_cache_memory(self))
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
        let executor = load_executor(definition, artifacts, backend, config)?;
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

    pub(crate) fn transferable_kv_cache_memory(&self) -> Option<TransferableKvCacheMemory> {
        self.executor.transferable_kv_cache_memory().cloned()
    }
}

impl ForwardModel for LoadedModelRuntime {
    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        self.executor.forward(batch)
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
    config: ModelRuntimeConfig,
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
                let kv_cache = config.kv_cache.ok_or_else(|| {
                    ModelRuntimeLoadError::MissingCapabilities(vec![
                        "runtime KV cache allocation configuration".to_string(),
                    ])
                })?;
                CudaBf16DenseDecoder::load(definition, artifacts, backend, kv_cache)
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
