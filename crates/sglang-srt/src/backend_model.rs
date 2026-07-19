use crate::backend::{
    CudaBackend, RuntimeBackend, RuntimeCapability, RuntimeDevicePlacement, RuntimeDtype,
};
use crate::cpu_hybrid::{CpuHybridExecutionResources, CpuReferenceHybridDecoder};
use crate::cpu_reference::{CpuReferenceDenseDecoder, CpuReferenceKvCache};
use crate::cuda_dense_decoder::CudaBf16DenseDecoder;
use crate::cuda_execution_resources::CudaExecutionResources;
use crate::cuda_hybrid_decoder::CudaBf16HybridDecoder;
use crate::cuda_kv_cache::allocate_cuda_kv_cache;
use crate::cuda_recurrent_state::CudaRecurrentStateStorage;
use crate::cuda_tensor_parallel::CudaTensorParallelDenseRuntime;
use crate::kv_cache::{KvCacheDtype, KvCacheModelLayout, KvCacheRuntimeLayout, PagedKvCacheLayout};
use crate::model_artifacts::LocalModelArtifacts;
use crate::model_executor::KvCacheAllocationConfig;
use crate::model_runtime::{
    BackendExecutionBundle, BackendExecutionRuntime, InitializedRuntimeBackend, ModelRuntimeConfig,
    ModelRuntimeLoadError,
};
use crate::models::{ModelDefinition, ModelExecutionArchitecture};

pub(crate) struct BackendProviderRegistry;

impl BackendProviderRegistry {
    pub(crate) fn initialize(
        requested: RuntimeBackend,
        placement: RuntimeDevicePlacement,
    ) -> Result<Box<dyn InitializedRuntimeBackend>, RuntimeBackendInitializationError> {
        if requested == RuntimeBackend::Auto {
            return initialize_auto_backend(placement);
        }

        let provider = runtime_backend_providers()
            .iter()
            .copied()
            .find(|provider| provider.backend() == requested)
            .ok_or_else(|| RuntimeBackendInitializationError {
                requested,
                message: format!(
                    "{} backend provider is not registered; registered providers: {}",
                    requested,
                    registered_backend_names()
                ),
            })?;
        provider
            .initialize(placement)
            .map_err(|message| RuntimeBackendInitializationError { requested, message })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RuntimeBackendInitializationError {
    pub(crate) requested: RuntimeBackend,
    pub(crate) message: String,
}

trait RuntimeBackendProvider: Sync {
    fn backend(&self) -> RuntimeBackend;
    fn is_production(&self) -> bool;
    fn initialize(
        &self,
        placement: RuntimeDevicePlacement,
    ) -> Result<Box<dyn InitializedRuntimeBackend>, String>;
}

struct CpuReferenceBackendProvider;
struct CudaBackendProvider;
struct CpuReferenceBackend;

static CPU_REFERENCE_BACKEND_PROVIDER: CpuReferenceBackendProvider = CpuReferenceBackendProvider;
static CUDA_BACKEND_PROVIDER: CudaBackendProvider = CudaBackendProvider;
static RUNTIME_BACKEND_PROVIDERS: [&'static dyn RuntimeBackendProvider; 2] =
    [&CPU_REFERENCE_BACKEND_PROVIDER, &CUDA_BACKEND_PROVIDER];

fn runtime_backend_providers() -> &'static [&'static dyn RuntimeBackendProvider] {
    &RUNTIME_BACKEND_PROVIDERS
}

fn registered_backend_names() -> String {
    runtime_backend_providers()
        .iter()
        .map(|provider| provider.backend().as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

fn initialize_auto_backend(
    placement: RuntimeDevicePlacement,
) -> Result<Box<dyn InitializedRuntimeBackend>, RuntimeBackendInitializationError> {
    let mut failures = Vec::new();
    for provider in runtime_backend_providers()
        .iter()
        .copied()
        .filter(|provider| provider.is_production())
    {
        match provider.initialize(placement) {
            Ok(backend) => return Ok(backend),
            Err(message) => failures.push(format!("{}: {message}", provider.backend())),
        }
    }

    let attempted = if failures.is_empty() {
        "no production backend providers are registered".to_string()
    } else {
        failures.join("; ")
    };
    Err(RuntimeBackendInitializationError {
        requested: RuntimeBackend::Auto,
        message: format!(
            "no executable production backend was detected; {attempted}; auto never falls back to the CPU reference backend"
        ),
    })
}

impl RuntimeBackendProvider for CpuReferenceBackendProvider {
    fn backend(&self) -> RuntimeBackend {
        RuntimeBackend::Cpu
    }

    fn is_production(&self) -> bool {
        false
    }

    fn initialize(
        &self,
        _placement: RuntimeDevicePlacement,
    ) -> Result<Box<dyn InitializedRuntimeBackend>, String> {
        Ok(Box::new(CpuReferenceBackend))
    }
}

impl RuntimeBackendProvider for CudaBackendProvider {
    fn backend(&self) -> RuntimeBackend {
        RuntimeBackend::Cuda
    }

    fn is_production(&self) -> bool {
        true
    }

    fn initialize(
        &self,
        placement: RuntimeDevicePlacement,
    ) -> Result<Box<dyn InitializedRuntimeBackend>, String> {
        let device_ordinal = placement.device_ordinal()?;
        CudaBackend::initialize(device_ordinal)
            .map(|backend| Box::new(backend) as Box<dyn InitializedRuntimeBackend>)
            .map_err(|error| {
                format!(
                    "failed to initialize CUDA device ordinal {device_ordinal} for local rank {} / TP rank {}: {error}",
                    placement.local_rank, placement.tensor_parallel_rank
                )
            })
    }
}

impl InitializedRuntimeBackend for CpuReferenceBackend {
    fn runtime_backend(&self) -> RuntimeBackend {
        RuntimeBackend::Cpu
    }

    fn capabilities(&self) -> RuntimeCapability {
        RuntimeCapability::cpu_reference("cpu-reference-backend", false)
    }

    fn validate_model_runtime(
        &self,
        definition: &ModelDefinition,
        tensor_parallel_size: usize,
    ) -> Vec<String> {
        validate_cpu_model_runtime(definition, tensor_parallel_size)
    }

    fn create_model_runtime(
        self: Box<Self>,
        definition: &ModelDefinition,
        artifacts: &LocalModelArtifacts,
        config: ModelRuntimeConfig,
    ) -> Result<Box<dyn BackendExecutionRuntime>, ModelRuntimeLoadError> {
        create_cpu_model_runtime(definition, artifacts, config)
    }
}

impl InitializedRuntimeBackend for CudaBackend {
    fn runtime_backend(&self) -> RuntimeBackend {
        RuntimeBackend::Cuda
    }

    fn capabilities(&self) -> RuntimeCapability {
        CudaBackend::capabilities(self)
    }

    fn validate_model_runtime(
        &self,
        definition: &ModelDefinition,
        tensor_parallel_size: usize,
    ) -> Vec<String> {
        validate_cuda_model_runtime(definition, self, tensor_parallel_size)
    }

    fn create_model_runtime(
        self: Box<Self>,
        definition: &ModelDefinition,
        artifacts: &LocalModelArtifacts,
        config: ModelRuntimeConfig,
    ) -> Result<Box<dyn BackendExecutionRuntime>, ModelRuntimeLoadError> {
        create_cuda_model_runtime(definition, artifacts, *self, config)
    }
}

fn create_cpu_model_runtime(
    definition: &ModelDefinition,
    artifacts: &LocalModelArtifacts,
    config: ModelRuntimeConfig,
) -> Result<Box<dyn BackendExecutionRuntime>, ModelRuntimeLoadError> {
    let kv_cache = allocate_cpu_active_kv_cache(definition, required_kv_cache_config(config)?)?;
    match definition.execution() {
        ModelExecutionArchitecture::Transformer { .. } if definition.hybrid_decoder().is_some() => {
            let executor = CpuReferenceHybridDecoder::load(definition, artifacts)
                .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?;
            let recurrent_state_layout = definition.recurrent_state_layout();
            let resources = CpuHybridExecutionResources::new(kv_cache, recurrent_state_layout)
                .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?;
            Ok(Box::new(BackendExecutionBundle::new(executor, resources)))
        }
        ModelExecutionArchitecture::Transformer { .. } if definition.dense_decoder().is_some() => {
            let executor = CpuReferenceDenseDecoder::load(definition, artifacts)
                .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?;
            Ok(Box::new(BackendExecutionBundle::from_kv_cache(
                executor, kv_cache,
            )))
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
) -> Result<CpuReferenceKvCache, ModelRuntimeLoadError> {
    let layout = paged_kv_cache_layout(definition, KvCacheDtype::Float32, config)?;
    CpuReferenceKvCache::new(layout).map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))
}

fn create_cuda_model_runtime(
    definition: &ModelDefinition,
    artifacts: &LocalModelArtifacts,
    backend: CudaBackend,
    config: ModelRuntimeConfig,
) -> Result<Box<dyn BackendExecutionRuntime>, ModelRuntimeLoadError> {
    match definition.execution() {
        ModelExecutionArchitecture::Transformer { .. } if definition.dense_decoder().is_some() => {
            if config.tensor_parallel_size > 1 {
                return CudaTensorParallelDenseRuntime::launch(
                    definition, artifacts, backend, config,
                )
                .map(|runtime| Box::new(runtime) as Box<dyn BackendExecutionRuntime>);
            }
            let kv_cache = allocate_cuda_active_kv_cache(
                definition,
                &backend,
                required_kv_cache_config(config)?,
            )?;
            let executor = CudaBf16DenseDecoder::load(definition, artifacts, backend)
                .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?;
            let resources = CudaExecutionResources::new(kv_cache, None);
            Ok(Box::new(BackendExecutionBundle::new(executor, resources)))
        }
        ModelExecutionArchitecture::Transformer { .. } if definition.hybrid_decoder().is_some() => {
            if config.tensor_parallel_size > 1 {
                return Err(ModelRuntimeLoadError::MissingCapabilities(vec![
                    "CUDA tensor parallel executor for the shared hybrid decoder".to_string(),
                ]));
            }
            let kv_cache = allocate_cuda_active_kv_cache(
                definition,
                &backend,
                required_kv_cache_config(config)?,
            )?;
            let recurrent_state = definition
                .recurrent_state_layout()
                .map(|recurrent_layout| {
                    CudaRecurrentStateStorage::allocate(
                        backend.context(),
                        recurrent_layout,
                        config.recurrent_state_slot_capacity,
                    )
                })
                .transpose()
                .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?;
            let executor = CudaBf16HybridDecoder::load(definition, artifacts, backend)
                .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?;
            let resources = CudaExecutionResources::new(kv_cache, recurrent_state);
            Ok(Box::new(BackendExecutionBundle::new(executor, resources)))
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
) -> Result<crate::kv_cache::KvCachePool<crate::cuda_kv_cache::CudaKvStorage>, ModelRuntimeLoadError>
{
    let layout = paged_kv_cache_layout(definition, KvCacheDtype::Bfloat16, config)?;
    allocate_cuda_kv_cache(backend.context(), layout.runtime(), layout.page_count())
        .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))
}

fn paged_kv_cache_layout(
    definition: &ModelDefinition,
    dtype: KvCacheDtype,
    config: KvCacheAllocationConfig,
) -> Result<PagedKvCacheLayout, ModelRuntimeLoadError> {
    let model_layout = definition.kv_cache_layout().ok_or_else(|| {
        ModelRuntimeLoadError::MissingCapabilities(vec![
            "model paged KV cache geometry".to_string(),
        ])
    })?;
    paged_kv_cache_layout_for_model(model_layout, dtype, config)
}

pub(crate) fn paged_kv_cache_layout_for_model(
    model_layout: KvCacheModelLayout,
    dtype: KvCacheDtype,
    config: KvCacheAllocationConfig,
) -> Result<PagedKvCacheLayout, ModelRuntimeLoadError> {
    validate_cache_config(config)?;
    let bytes_per_token = model_layout
        .token_size_bytes(dtype)
        .map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))?;
    let tensor_pair_size_bytes = model_layout
        .tensor_pair_size_bytes(dtype)
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
    let layout = match tensor_pair_size_bytes {
        Some([key_token_bytes, value_token_bytes]) => PagedKvCacheLayout::new_with_tensor_pair(
            runtime_layout,
            page_count,
            key_token_bytes,
            value_token_bytes,
        ),
        None => PagedKvCacheLayout::new(runtime_layout, page_count),
    };
    layout.map_err(|error| ModelRuntimeLoadError::Load(error.to_string()))
}

fn validate_cpu_model_runtime(
    definition: &ModelDefinition,
    tensor_parallel_size: usize,
) -> Vec<String> {
    if tensor_parallel_size > 1 {
        return vec![format!(
            "CPU reference backend supports only tp_size=1 (requested {tensor_parallel_size})"
        )];
    }
    match definition.execution() {
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
    let mut missing = Vec::new();
    let capability = backend.capabilities();
    if definition.dense_decoder().is_none() && definition.hybrid_decoder().is_none() {
        let ModelExecutionArchitecture::Transformer {
            attention,
            feed_forward,
        } = definition.execution();
        missing.push(format!(
            "{} {} decoder executor for cuda backend",
            attention.family(),
            feed_forward.family()
        ));
        missing.push("runtime-owned KV cache allocation".to_string());
        return missing;
    }
    if let Some(plan) = definition.hybrid_decoder() {
        missing.extend(CudaBf16HybridDecoder::missing_components(definition));
        if plan.has_recurrent_layers() && definition.recurrent_state_layout().is_none() {
            missing.push("model recurrent-state layout".to_string());
        }
    }
    if !capability.supported_dtypes.contains(&RuntimeDtype::Bf16) {
        missing.push("CUDA BF16 compute capability 8.0 or newer".to_string());
    }
    if tensor_parallel_size > 1 {
        if definition.dense_decoder().is_none() || definition.hybrid_decoder().is_some() {
            missing.push(
                "CUDA tensor parallel executor for the shared dense decoder; hybrid and MLA/MoE tensor parallel execution are not implemented"
                    .to_string(),
            );
        } else if let Err(error) = sglang_kernel::nccl::NcclLibrary::load() {
            missing.push(format!("NCCL collective backend: {error}"));
        }
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

#[cfg(test)]
mod tests {
    use super::BackendProviderRegistry;
    use crate::backend::{RuntimeBackend, RuntimeCapabilityClass, RuntimeDevicePlacement};

    fn placement() -> RuntimeDevicePlacement {
        RuntimeDevicePlacement::for_tensor_parallel_rank(0, 1, 0, 1, 1)
            .expect("single-rank placement")
    }

    #[test]
    fn explicit_cpu_uses_registered_reference_provider() {
        let backend = BackendProviderRegistry::initialize(RuntimeBackend::Cpu, placement())
            .expect("CPU reference provider should initialize");

        assert_eq!(backend.runtime_backend(), RuntimeBackend::Cpu);
        assert_eq!(
            backend.capabilities().class,
            RuntimeCapabilityClass::CpuReference
        );
    }

    #[test]
    fn unregistered_accelerator_provider_fails_before_model_loading() {
        let error = match BackendProviderRegistry::initialize(RuntimeBackend::Metal, placement()) {
            Ok(_) => panic!("Metal provider is not implemented"),
            Err(error) => error,
        };

        assert_eq!(error.requested, RuntimeBackend::Metal);
        assert!(
            error
                .message
                .contains("metal backend provider is not registered")
        );
        assert!(error.message.contains("cpu, cuda"));
    }
}
