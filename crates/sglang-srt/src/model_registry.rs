use std::fmt;
use std::path::{Path, PathBuf};

use crate::backend::{
    InitializedRuntimeBackend, RuntimeBackend, RuntimeCapability, RuntimeDtype, RuntimeRequirements,
};
use crate::cli::ServerArgs;
use crate::cuda_runtime::CudaEmbeddingLmModel;
use crate::model_artifacts::{HfModelConfig, LocalModelArtifacts, ModelArtifactError};
use crate::model_executor::{
    CpuEmbeddingLmModel, ForwardModel, ModelForwardError, ModelForwardOutput, ModelWorkerBatch,
};
use crate::transfer::{KvCacheTransferError, TransferableKvCacheMemory};
use crate::worker::WorkerWeightUpdateRequest;

const EMBEDDING_LM_ARCHITECTURE: &str = "SglangEmbeddingLmForCausalLM";
const DEEPSEEK_V4_ARCHITECTURE: &str = "DeepseekV4ForCausalLM";
const GLM_MOE_DSA_ARCHITECTURE: &str = "GlmMoeDsaForCausalLM";

static EMBEDDING_LM_ADAPTER: EmbeddingLmAdapter = EmbeddingLmAdapter;
static DEEPSEEK_V4_ADAPTER: DeepSeekV4Adapter = DeepSeekV4Adapter;
static GLM_MOE_DSA_ADAPTER: GlmMoeDsaAdapter = GlmMoeDsaAdapter;
static MODEL_ADAPTERS: [&'static dyn ModelAdapter; 3] = [
    &EMBEDDING_LM_ADAPTER,
    &DEEPSEEK_V4_ADAPTER,
    &GLM_MOE_DSA_ADAPTER,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelCapabilityRequirements {
    dtype: Option<RuntimeDtype>,
    default_attention_backend: Option<&'static str>,
}

impl ModelCapabilityRequirements {
    pub fn dtype(self) -> Option<RuntimeDtype> {
        self.dtype
    }

    pub fn default_attention_backend(self) -> Option<&'static str> {
        self.default_attention_backend
    }

    pub fn runtime_requirements<'a>(
        self,
        tensor_parallel_size: usize,
        requested_attention_backend: Option<&'a str>,
    ) -> RuntimeRequirements<'a> {
        RuntimeRequirements {
            requires_forward: true,
            dtype: self.dtype,
            attention_backend: requested_attention_backend.or(self.default_attention_backend),
            tensor_parallel_size,
            ..RuntimeRequirements::default()
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ModelRegistry;

impl ModelRegistry {
    pub fn resolve(
        self,
        model_path: &Path,
        config: &HfModelConfig,
    ) -> Result<ResolvedModelAdapter, ModelRegistryError> {
        if config.architectures.is_empty() {
            return Err(ModelRegistryError::MissingArchitectures {
                model_path: model_path.to_path_buf(),
            });
        }

        for requested_architecture in &config.architectures {
            for adapter in MODEL_ADAPTERS {
                if let Some(architecture) = adapter
                    .architectures()
                    .iter()
                    .copied()
                    .find(|architecture| *architecture == requested_architecture)
                {
                    return Ok(ResolvedModelAdapter {
                        adapter,
                        architecture,
                    });
                }
            }
        }

        Err(ModelRegistryError::UnsupportedArchitectures {
            model_path: model_path.to_path_buf(),
            requested: config.architectures.clone(),
            supported: self
                .supported_architectures()
                .into_iter()
                .map(str::to_string)
                .collect(),
        })
    }

    pub fn supported_architectures(self) -> Vec<&'static str> {
        MODEL_ADAPTERS
            .iter()
            .flat_map(|adapter| adapter.architectures().iter().copied())
            .collect()
    }

    pub fn validate_checkpoint(
        self,
        artifacts: &LocalModelArtifacts,
    ) -> Result<(), ModelRegistryError> {
        self.resolve(artifacts.model_path(), artifacts.config())?
            .validate_checkpoint(artifacts)
    }

    fn load(
        self,
        artifacts: &LocalModelArtifacts,
        requested_backend: RuntimeBackend,
        tensor_parallel_size: usize,
    ) -> Result<RegisteredModel, ModelRegistryError> {
        let adapter = self.resolve(artifacts.model_path(), artifacts.config())?;
        let backend =
            InitializedRuntimeBackend::initialize(requested_backend).map_err(|error| {
                ModelRegistryError::BackendInitialization {
                    requested: error.requested,
                    message: error.message,
                }
            })?;
        adapter.load(artifacts, backend, tensor_parallel_size)
    }
}

#[derive(Clone, Copy)]
pub struct ResolvedModelAdapter {
    adapter: &'static dyn ModelAdapter,
    architecture: &'static str,
}

impl fmt::Debug for ResolvedModelAdapter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResolvedModelAdapter")
            .field("architecture", &self.architecture)
            .finish_non_exhaustive()
    }
}

impl ResolvedModelAdapter {
    pub fn architecture(self) -> &'static str {
        self.architecture
    }

    pub fn requirements(self, backend: RuntimeBackend) -> ModelCapabilityRequirements {
        self.adapter.requirements(backend)
    }

    fn validate_checkpoint(
        self,
        artifacts: &LocalModelArtifacts,
    ) -> Result<(), ModelRegistryError> {
        self.adapter.validate_checkpoint(artifacts)
    }

    fn load(
        self,
        artifacts: &LocalModelArtifacts,
        backend: InitializedRuntimeBackend,
        tensor_parallel_size: usize,
    ) -> Result<RegisteredModel, ModelRegistryError> {
        let runtime_backend = backend.runtime_backend();
        let requirements = self.adapter.requirements(runtime_backend);
        let runtime = self
            .adapter
            .load_runtime(artifacts, backend, tensor_parallel_size)?;
        Ok(RegisteredModel {
            architecture: self.architecture,
            model_path: artifacts.model_path().to_path_buf(),
            model_type: artifacts.config().model_type.clone(),
            runtime_backend,
            tensor_parallel_size,
            requirements,
            runtime,
        })
    }
}

trait ModelAdapter: Sync {
    fn architectures(&self) -> &'static [&'static str];

    fn requirements(&self, backend: RuntimeBackend) -> ModelCapabilityRequirements;

    fn validate_checkpoint(
        &self,
        artifacts: &LocalModelArtifacts,
    ) -> Result<(), ModelRegistryError>;

    fn load_runtime(
        &self,
        artifacts: &LocalModelArtifacts,
        backend: InitializedRuntimeBackend,
        tensor_parallel_size: usize,
    ) -> Result<Box<dyn RegisteredModelExecutor>, ModelRegistryError>;
}

struct EmbeddingLmAdapter;

impl ModelAdapter for EmbeddingLmAdapter {
    fn architectures(&self) -> &'static [&'static str] {
        &[EMBEDDING_LM_ARCHITECTURE]
    }

    fn requirements(&self, _backend: RuntimeBackend) -> ModelCapabilityRequirements {
        ModelCapabilityRequirements {
            dtype: Some(RuntimeDtype::F32),
            default_attention_backend: None,
        }
    }

    fn validate_checkpoint(
        &self,
        artifacts: &LocalModelArtifacts,
    ) -> Result<(), ModelRegistryError> {
        CpuEmbeddingLmModel::from_local_model_artifacts(artifacts)
            .map_err(ModelRegistryError::from)?
            .ok_or_else(|| adapter_config_error(artifacts, EMBEDDING_LM_ARCHITECTURE))?;
        Ok(())
    }

    fn load_runtime(
        &self,
        artifacts: &LocalModelArtifacts,
        backend: InitializedRuntimeBackend,
        _tensor_parallel_size: usize,
    ) -> Result<Box<dyn RegisteredModelExecutor>, ModelRegistryError> {
        match backend {
            InitializedRuntimeBackend::CpuReference => {
                CpuEmbeddingLmModel::from_local_model_artifacts(artifacts)
                    .map_err(ModelRegistryError::from)?
                    .map(|model| Box::new(model) as Box<dyn RegisteredModelExecutor>)
                    .ok_or_else(|| adapter_config_error(artifacts, EMBEDDING_LM_ARCHITECTURE))
            }
            InitializedRuntimeBackend::Cuda(backend) => {
                CudaEmbeddingLmModel::from_local_model_artifacts(artifacts, backend)
                    .map_err(|error| load_error(artifacts, EMBEDDING_LM_ARCHITECTURE, error))?
                    .map(|model| Box::new(model) as Box<dyn RegisteredModelExecutor>)
                    .ok_or_else(|| adapter_config_error(artifacts, EMBEDDING_LM_ARCHITECTURE))
            }
            InitializedRuntimeBackend::Unavailable(backend) => Err(missing_executor_error(
                artifacts,
                EMBEDDING_LM_ARCHITECTURE,
                backend,
            )),
        }
    }
}

struct DeepSeekV4Adapter;

impl ModelAdapter for DeepSeekV4Adapter {
    fn architectures(&self) -> &'static [&'static str] {
        &[DEEPSEEK_V4_ARCHITECTURE]
    }

    fn requirements(&self, _backend: RuntimeBackend) -> ModelCapabilityRequirements {
        ModelCapabilityRequirements {
            dtype: Some(RuntimeDtype::Bf16),
            default_attention_backend: None,
        }
    }

    fn validate_checkpoint(
        &self,
        artifacts: &LocalModelArtifacts,
    ) -> Result<(), ModelRegistryError> {
        artifacts
            .checkpoint_catalog()?
            .deepseek_model_weights()
            .map(|_| ())
            .map_err(ModelRegistryError::from)
    }

    fn load_runtime(
        &self,
        artifacts: &LocalModelArtifacts,
        backend: InitializedRuntimeBackend,
        _tensor_parallel_size: usize,
    ) -> Result<Box<dyn RegisteredModelExecutor>, ModelRegistryError> {
        Err(missing_executor_error(
            artifacts,
            DEEPSEEK_V4_ARCHITECTURE,
            backend.runtime_backend(),
        ))
    }
}

struct GlmMoeDsaAdapter;

impl ModelAdapter for GlmMoeDsaAdapter {
    fn architectures(&self) -> &'static [&'static str] {
        &[GLM_MOE_DSA_ARCHITECTURE]
    }

    fn requirements(&self, backend: RuntimeBackend) -> ModelCapabilityRequirements {
        ModelCapabilityRequirements {
            dtype: Some(match backend {
                RuntimeBackend::Cpu => RuntimeDtype::F32,
                _ => RuntimeDtype::Bf16,
            }),
            default_attention_backend: (backend == RuntimeBackend::Cpu).then_some("reference"),
        }
    }

    fn validate_checkpoint(
        &self,
        artifacts: &LocalModelArtifacts,
    ) -> Result<(), ModelRegistryError> {
        artifacts
            .checkpoint_catalog()?
            .glm_moe_dsa_model_weights()
            .map(|_| ())
            .map_err(ModelRegistryError::from)
    }

    fn load_runtime(
        &self,
        artifacts: &LocalModelArtifacts,
        backend: InitializedRuntimeBackend,
        _tensor_parallel_size: usize,
    ) -> Result<Box<dyn RegisteredModelExecutor>, ModelRegistryError> {
        Err(missing_executor_error(
            artifacts,
            GLM_MOE_DSA_ARCHITECTURE,
            backend.runtime_backend(),
        ))
    }
}

trait RegisteredModelExecutor: ForwardModel + fmt::Debug + Send {
    fn runtime_capability(&self) -> RuntimeCapability;
}

impl RegisteredModelExecutor for CpuEmbeddingLmModel {
    fn runtime_capability(&self) -> RuntimeCapability {
        RuntimeCapability::cpu_reference("cpu-embedding-lm", false)
    }
}

impl RegisteredModelExecutor for CudaEmbeddingLmModel {
    fn runtime_capability(&self) -> RuntimeCapability {
        CudaEmbeddingLmModel::runtime_capability(self)
    }
}

#[derive(Debug)]
pub struct RegisteredModel {
    architecture: &'static str,
    model_path: PathBuf,
    model_type: Option<String>,
    runtime_backend: RuntimeBackend,
    tensor_parallel_size: usize,
    requirements: ModelCapabilityRequirements,
    runtime: Box<dyn RegisteredModelExecutor>,
}

impl RegisteredModel {
    pub fn architecture(&self) -> &'static str {
        self.architecture
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn model_type(&self) -> Option<&str> {
        self.model_type.as_deref()
    }

    pub fn runtime_backend(&self) -> RuntimeBackend {
        self.runtime_backend
    }

    pub fn tensor_parallel_size(&self) -> usize {
        self.tensor_parallel_size
    }

    pub fn requirements(&self) -> ModelCapabilityRequirements {
        self.requirements
    }
}

#[derive(Debug, Default)]
pub enum BootstrapForwardModel {
    #[default]
    Space,
    Registered(RegisteredModel),
}

impl BootstrapForwardModel {
    pub(crate) fn from_server_args(args: &ServerArgs) -> Result<Self, ModelRegistryError> {
        let model_path = Path::new(&args.model_path);
        if model_path.is_dir() && !model_path.join("config.json").is_file() {
            return Ok(Self::Space);
        }

        let artifacts = match LocalModelArtifacts::from_model_path(&args.model_path) {
            Ok(artifacts) => artifacts,
            Err(ModelArtifactError::ModelPathNotLocalDirectory { .. })
            | Err(ModelArtifactError::NoSafetensorsWeights { .. }) => return Ok(Self::Space),
            Err(error) => return Err(error.into()),
        };
        let requested_backend = RuntimeBackend::parse(&args.device)
            .ok_or_else(|| ModelRegistryError::InvalidDevice(args.device.clone()))?;
        ModelRegistry
            .load(&artifacts, requested_backend, args.tp_size)
            .map(Self::Registered)
    }

    pub fn is_space_reference(&self) -> bool {
        matches!(self, Self::Space)
    }

    pub fn runtime_capability(&self) -> RuntimeCapability {
        match self {
            Self::Space => RuntimeCapability::cpu_reference("space-reference", false),
            Self::Registered(model) => model.runtime.runtime_capability(),
        }
    }

    pub fn runtime_requirements<'a>(
        &self,
        tensor_parallel_size: usize,
        requested_attention_backend: Option<&'a str>,
    ) -> RuntimeRequirements<'a> {
        match self {
            Self::Space => ModelCapabilityRequirements {
                dtype: Some(RuntimeDtype::F32),
                default_attention_backend: Some("reference"),
            }
            .runtime_requirements(tensor_parallel_size, requested_attention_backend),
            Self::Registered(model) => model
                .requirements
                .runtime_requirements(tensor_parallel_size, requested_attention_backend),
        }
    }

    pub fn reserve_mooncake_kv_cache_slots(
        &mut self,
        slot_capacity: usize,
        page_size: usize,
    ) -> Result<(), KvCacheTransferError> {
        match self {
            Self::Space => Err(KvCacheTransferError::Runtime(
                "Space reference model does not expose transferable Mooncake KV memory".to_string(),
            )),
            Self::Registered(_) => Err(KvCacheTransferError::Runtime(format!(
                "generic ModelRunner KV cache resources are not registered; cannot reserve {slot_capacity} slots with page size {page_size} for Mooncake"
            ))),
        }
    }

    pub fn mooncake_kv_cache_memory(
        &self,
    ) -> Result<TransferableKvCacheMemory, KvCacheTransferError> {
        match self {
            Self::Space => Err(KvCacheTransferError::Runtime(
                "Space reference model does not expose transferable Mooncake KV memory".to_string(),
            )),
            Self::Registered(_) => Err(KvCacheTransferError::Runtime(
                "generic ModelRunner does not own registered Mooncake KV cache memory".to_string(),
            )),
        }
    }

    fn reload_backend(&self) -> RuntimeBackend {
        match self {
            Self::Space => RuntimeBackend::Cpu,
            Self::Registered(model) => model.runtime_backend,
        }
    }

    fn reload_tp_size(&self) -> usize {
        match self {
            Self::Space => 1,
            Self::Registered(model) => model.tensor_parallel_size,
        }
    }
}

impl ForwardModel for BootstrapForwardModel {
    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        match self {
            Self::Space => {
                let mut logits = Vec::with_capacity(batch.request_ids().len());
                for _ in batch.request_ids() {
                    let mut row = vec![0.0; (b' ' as usize) + 1];
                    row[b' ' as usize] = 1.0;
                    logits.push(row);
                }
                ModelForwardOutput::new(logits)
            }
            Self::Registered(model) => model.runtime.forward(batch),
        }
    }

    fn update_weights_from_disk(
        &mut self,
        request: &WorkerWeightUpdateRequest,
    ) -> Result<(), ModelForwardError> {
        let artifacts = LocalModelArtifacts::from_model_path(&request.model_path)
            .map_err(|error| ModelForwardError::Runtime(error.to_string()))?;
        let next = ModelRegistry
            .load(&artifacts, self.reload_backend(), self.reload_tp_size())
            .map(Self::Registered)
            .map_err(|error| ModelForwardError::Runtime(error.to_string()))?;
        *self = next;
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelRegistryError {
    ModelArtifact(ModelArtifactError),
    InvalidDevice(String),
    MissingArchitectures {
        model_path: PathBuf,
    },
    UnsupportedArchitectures {
        model_path: PathBuf,
        requested: Vec<String>,
        supported: Vec<String>,
    },
    BackendInitialization {
        requested: RuntimeBackend,
        message: String,
    },
    MissingCapabilities {
        model_path: PathBuf,
        architecture: &'static str,
        backend: RuntimeBackend,
        missing: Vec<String>,
    },
    InvalidAdapterConfig {
        model_path: PathBuf,
        architecture: &'static str,
        model_type: Option<String>,
    },
    ModelLoad {
        model_path: PathBuf,
        architecture: &'static str,
        message: String,
    },
}

impl fmt::Display for ModelRegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelArtifact(error) => write!(formatter, "model artifact error: {error}"),
            Self::InvalidDevice(device) => write!(formatter, "invalid --device: {device}"),
            Self::MissingArchitectures { model_path } => write!(
                formatter,
                "model {} has no Hugging Face architectures; model selection requires config.json architectures",
                model_path.display()
            ),
            Self::UnsupportedArchitectures {
                model_path,
                requested,
                supported,
            } => write!(
                formatter,
                "model {} architectures {:?} are unsupported; registered architectures: {:?}",
                model_path.display(),
                requested,
                supported
            ),
            Self::BackendInitialization { requested, message } => write!(
                formatter,
                "failed to initialize requested runtime backend {requested}: {message}"
            ),
            Self::MissingCapabilities {
                model_path,
                architecture,
                backend,
                missing,
            } => write!(
                formatter,
                "model {} architecture {architecture} cannot start on backend {backend}; missing capabilities: {}",
                model_path.display(),
                missing.join(", ")
            ),
            Self::InvalidAdapterConfig {
                model_path,
                architecture,
                model_type,
            } => write!(
                formatter,
                "model {} resolved architecture {architecture}, but its adapter rejected model_type {}",
                model_path.display(),
                model_type.as_deref().unwrap_or("<unknown>")
            ),
            Self::ModelLoad {
                model_path,
                architecture,
                message,
            } => write!(
                formatter,
                "failed to load model {} architecture {architecture}: {message}",
                model_path.display()
            ),
        }
    }
}

impl std::error::Error for ModelRegistryError {}

impl From<ModelArtifactError> for ModelRegistryError {
    fn from(value: ModelArtifactError) -> Self {
        Self::ModelArtifact(value)
    }
}

fn missing_executor_error(
    artifacts: &LocalModelArtifacts,
    architecture: &'static str,
    backend: RuntimeBackend,
) -> ModelRegistryError {
    ModelRegistryError::MissingCapabilities {
        model_path: artifacts.model_path().to_path_buf(),
        architecture,
        backend,
        missing: vec!["registered model forward executor".to_string()],
    }
}

fn adapter_config_error(
    artifacts: &LocalModelArtifacts,
    architecture: &'static str,
) -> ModelRegistryError {
    ModelRegistryError::InvalidAdapterConfig {
        model_path: artifacts.model_path().to_path_buf(),
        architecture,
        model_type: artifacts.config().model_type.clone(),
    }
}

fn load_error(
    artifacts: &LocalModelArtifacts,
    architecture: &'static str,
    error: impl fmt::Display,
) -> ModelRegistryError {
    ModelRegistryError::ModelLoad {
        model_path: artifacts.model_path().to_path_buf(),
        architecture,
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_resolves_first_registered_hugging_face_architecture() {
        let config = HfModelConfig {
            model_type: Some("not-a-registry-key".to_string()),
            architectures: vec![
                "UnsupportedForCausalLM".to_string(),
                GLM_MOE_DSA_ARCHITECTURE.to_string(),
            ],
            ..HfModelConfig::default()
        };

        let resolved = ModelRegistry
            .resolve(Path::new("/models/glm"), &config)
            .expect("registered architecture should resolve");

        assert_eq!(resolved.architecture(), GLM_MOE_DSA_ARCHITECTURE);
        assert_eq!(
            resolved
                .requirements(RuntimeBackend::Cpu)
                .default_attention_backend(),
            Some("reference")
        );
    }

    #[test]
    fn registry_requires_hugging_face_architectures() {
        let error = ModelRegistry
            .resolve(
                Path::new("/models/missing-architecture"),
                &HfModelConfig::default(),
            )
            .expect_err("missing architectures must fail before model loading");

        assert!(matches!(
            error,
            ModelRegistryError::MissingArchitectures { .. }
        ));
    }

    #[test]
    fn registry_reports_unsupported_architectures() {
        let config = HfModelConfig {
            architectures: vec!["Qwen2ForCausalLM".to_string()],
            ..HfModelConfig::default()
        };

        let error = ModelRegistry
            .resolve(Path::new("/models/qwen"), &config)
            .expect_err("unimplemented dense decoder must fail at registry resolution");

        assert!(matches!(
            error,
            ModelRegistryError::UnsupportedArchitectures { requested, .. }
                if requested == ["Qwen2ForCausalLM"]
        ));
    }
}
