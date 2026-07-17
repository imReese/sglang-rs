use std::fmt;

use nexus_transfer::{KvCacheMemoryProvider, TransferableKvCacheMemory};

use crate::backend::{RuntimeBackend, RuntimeCapability, RuntimeDtype};
use crate::model_artifacts::LocalModelArtifacts;
use crate::model_executor::{
    ForwardModel, KvCacheAllocationConfig, ModelForwardError, ModelForwardOutput, ModelWorkerBatch,
};
use crate::models::ModelDefinition;
use crate::runtime_kv_cache::{ActiveKvCache, RuntimeKvCache, RuntimeKvCacheMetadata};
use crate::transfer::KvCacheTransferError;
use crate::types::RequestId;
use crate::worker::WorkerWeightUpdateRequest;

pub(crate) trait BackendModelExecutor<K>: fmt::Debug + Send
where
    K: ActiveKvCache,
{
    fn runtime_capability(&self) -> RuntimeCapability;
    fn execution_dtype(&self) -> RuntimeDtype;
    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
        kv_cache: &mut K,
    ) -> Result<ModelForwardOutput, ModelForwardError>;
    fn complete_request(&mut self, _request_id: &RequestId) {}
    fn update_weights_from_disk(
        &mut self,
        _request: &WorkerWeightUpdateRequest,
    ) -> Result<(), ModelForwardError> {
        Err(ModelForwardError::Runtime(
            "backend model executor does not support update_weights_from_disk".to_string(),
        ))
    }
}

pub(crate) trait BackendExecutionRuntime:
    ForwardModel
    + KvCacheMemoryProvider<Error = KvCacheTransferError>
    + RuntimeKvCacheMetadata
    + fmt::Debug
    + Send
{
    fn runtime_capability(&self) -> RuntimeCapability;
    fn execution_dtype(&self) -> RuntimeDtype;
}

pub(crate) struct BackendExecutionBundle<E, K>
where
    K: ActiveKvCache,
{
    executor: E,
    active_kv_cache: RuntimeKvCache<K>,
}

impl<E, K> fmt::Debug for BackendExecutionBundle<E, K>
where
    E: fmt::Debug,
    K: ActiveKvCache,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BackendExecutionBundle")
            .field("executor", &self.executor)
            .field("active_kv_cache", &self.active_kv_cache)
            .finish()
    }
}

impl<E, K> BackendExecutionBundle<E, K>
where
    K: ActiveKvCache,
{
    pub(crate) fn new(executor: E, active_kv_cache: K) -> Self {
        Self {
            executor,
            active_kv_cache: RuntimeKvCache::new(active_kv_cache),
        }
    }
}

impl<E, K> ForwardModel for BackendExecutionBundle<E, K>
where
    E: BackendModelExecutor<K>,
    K: ActiveKvCache,
{
    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        self.executor
            .forward(batch, self.active_kv_cache.allocation_mut())
    }

    fn complete_request(&mut self, request_id: &RequestId) {
        self.executor.complete_request(request_id);
    }

    fn update_weights_from_disk(
        &mut self,
        request: &WorkerWeightUpdateRequest,
    ) -> Result<(), ModelForwardError> {
        self.executor.update_weights_from_disk(request)
    }
}

impl<E, K> KvCacheMemoryProvider for BackendExecutionBundle<E, K>
where
    E: BackendModelExecutor<K>,
    K: ActiveKvCache,
{
    type Error = KvCacheTransferError;

    fn transferable_kv_cache_memory(&self) -> Result<TransferableKvCacheMemory, Self::Error> {
        self.active_kv_cache.transferable_kv_cache_memory()
    }
}

impl<E, K> RuntimeKvCacheMetadata for BackendExecutionBundle<E, K>
where
    E: BackendModelExecutor<K>,
    K: ActiveKvCache,
{
    fn active_kv_cache_layout(&self) -> Option<crate::kv_cache::PagedKvCacheLayout> {
        Some(self.active_kv_cache.layout())
    }
}

impl<E, K> BackendExecutionRuntime for BackendExecutionBundle<E, K>
where
    E: BackendModelExecutor<K> + 'static,
    K: ActiveKvCache + 'static,
{
    fn runtime_capability(&self) -> RuntimeCapability {
        self.executor.runtime_capability()
    }

    fn execution_dtype(&self) -> RuntimeDtype {
        self.executor.execution_dtype()
    }
}

pub(crate) trait InitializedRuntimeBackend: Send {
    fn runtime_backend(&self) -> RuntimeBackend;
    fn capabilities(&self) -> RuntimeCapability;
    fn validate_model_runtime(
        &self,
        definition: &ModelDefinition,
        tensor_parallel_size: usize,
    ) -> Vec<String>;
    fn create_model_runtime(
        self: Box<Self>,
        definition: &ModelDefinition,
        artifacts: &LocalModelArtifacts,
        config: ModelRuntimeConfig,
    ) -> Result<Box<dyn BackendExecutionRuntime>, ModelRuntimeLoadError>;
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
    execution: Box<dyn BackendExecutionRuntime>,
    config: ModelRuntimeConfig,
}

impl LoadedModelRuntime {
    pub(crate) fn load(
        definition: &ModelDefinition,
        artifacts: &LocalModelArtifacts,
        backend: Box<dyn InitializedRuntimeBackend>,
        config: ModelRuntimeConfig,
    ) -> Result<Self, ModelRuntimeLoadError> {
        validate_runtime_support(definition, backend.as_ref(), config.tensor_parallel_size)?;
        let runtime_backend = backend.runtime_backend();
        let execution = backend.create_model_runtime(definition, artifacts, config)?;
        let capability = execution.runtime_capability();
        let execution_dtype = execution.execution_dtype();
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
            execution,
            config,
        })
    }

    pub(crate) fn runtime_backend(&self) -> RuntimeBackend {
        self.runtime_backend
    }

    pub(crate) fn runtime_capability(&self) -> RuntimeCapability {
        self.execution.runtime_capability()
    }

    pub(crate) fn execution_dtype(&self) -> RuntimeDtype {
        self.execution.execution_dtype()
    }

    pub(crate) fn config(&self) -> ModelRuntimeConfig {
        self.config
    }

    pub(crate) fn has_runtime_kv_cache(&self) -> bool {
        self.execution.active_kv_cache_layout().is_some()
    }
}

impl ForwardModel for LoadedModelRuntime {
    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        self.execution.forward(batch)
    }

    fn complete_request(&mut self, request_id: &RequestId) {
        self.execution.complete_request(request_id);
    }

    fn update_weights_from_disk(
        &mut self,
        request: &WorkerWeightUpdateRequest,
    ) -> Result<(), ModelForwardError> {
        self.execution.update_weights_from_disk(request)
    }
}

impl KvCacheMemoryProvider for LoadedModelRuntime {
    type Error = KvCacheTransferError;

    fn transferable_kv_cache_memory(&self) -> Result<TransferableKvCacheMemory, Self::Error> {
        self.execution.transferable_kv_cache_memory()
    }
}

impl RuntimeKvCacheMetadata for LoadedModelRuntime {
    fn active_kv_cache_layout(&self) -> Option<crate::kv_cache::PagedKvCacheLayout> {
        self.execution.active_kv_cache_layout()
    }
}

pub(crate) fn validate_runtime_support(
    definition: &ModelDefinition,
    backend: &dyn InitializedRuntimeBackend,
    tensor_parallel_size: usize,
) -> Result<(), ModelRuntimeLoadError> {
    validate_runtime_parallelism(tensor_parallel_size)
        .map_err(|message| ModelRuntimeLoadError::MissingCapabilities(vec![message]))?;

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

pub(crate) fn validate_runtime_parallelism(tensor_parallel_size: usize) -> Result<(), String> {
    match tensor_parallel_size {
        0 => Err("tensor parallel size must be positive".to_string()),
        1 => Ok(()),
        requested => Err(format!(
            "tensor parallel execution requires a WorkerGroup, rank lifecycle, and collective backend; the runtime currently supports tp_size=1 (requested {requested})"
        )),
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
