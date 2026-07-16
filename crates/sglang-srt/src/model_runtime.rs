use std::fmt;

use crate::backend::{
    InitializedRuntimeBackend, RuntimeBackend, RuntimeCapability, RuntimeRequirements,
};
use crate::cuda_runtime::CudaEmbeddingLmModel;
use crate::model_artifacts::LocalModelArtifacts;
use crate::model_executor::{
    CpuEmbeddingLmModel, ForwardModel, ModelForwardError, ModelForwardOutput, ModelWorkerBatch,
};
use crate::models::{ModelDefinition, ModelExecutionArchitecture};
use crate::worker::WorkerWeightUpdateRequest;

pub(crate) trait ModelExecutor: ForwardModel + fmt::Debug + Send {
    fn runtime_capability(&self) -> RuntimeCapability;
}

impl ModelExecutor for CpuEmbeddingLmModel {
    fn runtime_capability(&self) -> RuntimeCapability {
        RuntimeCapability::cpu_reference("cpu-embedding-lm", false)
    }
}

impl ModelExecutor for CudaEmbeddingLmModel {
    fn runtime_capability(&self) -> RuntimeCapability {
        CudaEmbeddingLmModel::runtime_capability(self)
    }
}

#[derive(Debug)]
pub(crate) struct LoadedModelRuntime {
    runtime_backend: RuntimeBackend,
    executor: Box<dyn ModelExecutor>,
}

impl LoadedModelRuntime {
    pub(crate) fn load(
        definition: &ModelDefinition,
        artifacts: &LocalModelArtifacts,
        backend: InitializedRuntimeBackend,
        tensor_parallel_size: usize,
    ) -> Result<Self, ModelRuntimeLoadError> {
        validate_runtime_support(definition, &backend, tensor_parallel_size)?;
        let runtime_backend = backend.runtime_backend();
        let executor = load_executor(definition, artifacts, backend)?;
        let capability = executor.runtime_capability();
        capability
            .validate_requirements(&definition.runtime_requirements(tensor_parallel_size, None))
            .map_err(|mismatch| ModelRuntimeLoadError::MissingCapabilities(mismatch.missing))?;
        Ok(Self {
            runtime_backend,
            executor,
        })
    }

    pub(crate) fn runtime_backend(&self) -> RuntimeBackend {
        self.runtime_backend
    }

    pub(crate) fn runtime_capability(&self) -> RuntimeCapability {
        self.executor.runtime_capability()
    }
}

impl ForwardModel for LoadedModelRuntime {
    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        self.executor.forward(batch)
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

    let requirements = RuntimeRequirements {
        requires_forward: false,
        dtype: Some(definition.dtype()),
        attention_backend: None,
        tensor_parallel_size,
        requires_kv_cache_registration: false,
        requires_mooncake: false,
    };
    let mut missing = backend
        .capabilities()
        .validate_requirements(&requirements)
        .err()
        .map(|mismatch| mismatch.missing)
        .unwrap_or_default();

    if let ModelExecutionArchitecture::Transformer {
        attention,
        feed_forward,
    } = definition.execution()
    {
        missing.push(format!("{} decoder execution", attention.family()));
        missing.push(format!("{} kernels", feed_forward.family()));
        missing.push("runtime-owned KV cache allocation".to_string());
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
        } => Err(ModelRuntimeLoadError::MissingCapabilities(vec![
            format!("{} decoder execution", attention.family()),
            format!("{} kernels", feed_forward.family()),
            "runtime-owned KV cache allocation".to_string(),
        ])),
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
