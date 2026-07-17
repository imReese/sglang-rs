use std::fmt;

use crate::backend::{InitializedRuntimeBackend, RuntimeBackend, RuntimeCapability, RuntimeDtype};
use crate::model_artifacts::LocalModelArtifacts;
use crate::model_executor::{
    ForwardModel, KvCacheAllocationConfig, ModelForwardError, ModelForwardOutput, ModelWorkerBatch,
};
use crate::models::ModelDefinition;
use crate::runtime_kv_cache::{ModelExecutionResources, RuntimeKvCache};
use crate::worker::WorkerWeightUpdateRequest;

pub(crate) trait ModelExecutor: ForwardModel + fmt::Debug + Send {
    fn runtime_capability(&self) -> RuntimeCapability;
    fn execution_dtype(&self) -> RuntimeDtype;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ModelRuntimeConfig {
    pub(crate) tensor_parallel_size: usize,
    pub(crate) device_placement: crate::backend::RuntimeDevicePlacement,
    pub(crate) kv_cache: Option<KvCacheAllocationConfig>,
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
        let backend_runtime = backend.create_model_runtime(definition, artifacts, config)?;
        let executor = backend_runtime.executor;
        let runtime_kv_cache = backend_runtime.active_kv_cache;
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

    missing.extend(backend.validate_model_runtime(definition, tensor_parallel_size));

    if missing.is_empty() {
        Ok(())
    } else {
        Err(ModelRuntimeLoadError::MissingCapabilities(missing))
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
