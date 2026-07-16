use std::fmt;
use std::path::{Path, PathBuf};

use crate::backend::{
    InitializedRuntimeBackend, RuntimeBackend, RuntimeCapability, RuntimeRequirements,
};
use crate::cli::ServerArgs;
use crate::model_artifacts::{HfModelConfig, LocalModelArtifacts, ModelArtifactError};
use crate::model_executor::{
    ForwardModel, ModelForwardError, ModelForwardOutput, ModelWorkerBatch,
};
use crate::model_runtime::{LoadedModelRuntime, ModelRuntimeLoadError, validate_runtime_support};
use crate::models::{
    DEEPSEEK_V4_ADAPTER, EMBEDDING_LM_ADAPTER, GLM_MOE_DSA_ADAPTER, ModelAdapter,
    ModelAdapterError, ModelDefinition, QWEN2_ADAPTER,
};
use crate::transfer::KvCacheModelLayout;
use crate::worker::WorkerWeightUpdateRequest;

static MODEL_ADAPTERS: [&'static dyn ModelAdapter; 4] = [
    &EMBEDDING_LM_ADAPTER,
    &DEEPSEEK_V4_ADAPTER,
    &GLM_MOE_DSA_ADAPTER,
    &QWEN2_ADAPTER,
];

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

    pub fn definition(
        self,
        model_path: &Path,
        config: &HfModelConfig,
    ) -> Result<ModelDefinition, ModelRegistryError> {
        self.resolve(model_path, config)?
            .build_definition(model_path, config)
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
        let resolved = self.resolve(artifacts.model_path(), artifacts.config())?;
        resolved.build_definition(artifacts.model_path(), artifacts.config())?;
        resolved.validate_checkpoint(artifacts)
    }

    fn load(
        self,
        artifacts: &LocalModelArtifacts,
        requested_backend: RuntimeBackend,
        tensor_parallel_size: usize,
    ) -> Result<RegisteredModel, ModelRegistryError> {
        let resolved = self.resolve(artifacts.model_path(), artifacts.config())?;
        let definition = resolved.build_definition(artifacts.model_path(), artifacts.config())?;
        let backend =
            InitializedRuntimeBackend::initialize(requested_backend).map_err(|error| {
                ModelRegistryError::BackendInitialization {
                    requested: error.requested,
                    message: error.message,
                }
            })?;

        validate_runtime_support(&definition, &backend, tensor_parallel_size).map_err(|error| {
            runtime_error(
                artifacts,
                definition.architecture(),
                backend.runtime_backend(),
                error,
            )
        })?;
        resolved.validate_checkpoint(artifacts)?;
        let runtime_backend = backend.runtime_backend();
        let runtime =
            LoadedModelRuntime::load(&definition, artifacts, backend, tensor_parallel_size)
                .map_err(|error| {
                    runtime_error(artifacts, definition.architecture(), runtime_backend, error)
                })?;

        Ok(RegisteredModel {
            model_path: artifacts.model_path().to_path_buf(),
            definition,
            tensor_parallel_size,
            runtime,
        })
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

    pub fn build_definition(
        self,
        model_path: &Path,
        config: &HfModelConfig,
    ) -> Result<ModelDefinition, ModelRegistryError> {
        self.adapter
            .build_definition(model_path, config)
            .map_err(|error| adapter_error(model_path, self.architecture, error))
    }

    fn validate_checkpoint(
        self,
        artifacts: &LocalModelArtifacts,
    ) -> Result<(), ModelRegistryError> {
        self.adapter
            .validate_checkpoint(artifacts)
            .map_err(ModelRegistryError::from)
    }
}

#[derive(Debug)]
pub struct RegisteredModel {
    model_path: PathBuf,
    definition: ModelDefinition,
    tensor_parallel_size: usize,
    runtime: LoadedModelRuntime,
}

impl RegisteredModel {
    pub fn architecture(&self) -> &'static str {
        self.definition.architecture()
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn model_type(&self) -> Option<&str> {
        self.definition.model_type()
    }

    pub fn definition(&self) -> &ModelDefinition {
        &self.definition
    }

    pub fn runtime_backend(&self) -> RuntimeBackend {
        self.runtime.runtime_backend()
    }

    pub fn tensor_parallel_size(&self) -> usize {
        self.tensor_parallel_size
    }
}

#[derive(Debug)]
pub struct BootstrapForwardModel {
    registered: RegisteredModel,
}

impl BootstrapForwardModel {
    pub(crate) fn from_server_args(args: &ServerArgs) -> Result<Self, ModelRegistryError> {
        let artifacts = LocalModelArtifacts::from_model_path(&args.model_path)?;
        let requested_backend = RuntimeBackend::parse(&args.device)
            .ok_or_else(|| ModelRegistryError::InvalidDevice(args.device.clone()))?;
        ModelRegistry
            .load(&artifacts, requested_backend, args.tp_size)
            .map(|registered| Self { registered })
    }

    pub fn runtime_capability(&self) -> RuntimeCapability {
        self.registered.runtime.runtime_capability()
    }

    pub fn runtime_requirements<'a>(
        &self,
        tensor_parallel_size: usize,
        requested_attention_backend: Option<&'a str>,
    ) -> RuntimeRequirements<'a> {
        self.registered
            .definition
            .runtime_requirements(tensor_parallel_size, requested_attention_backend)
    }

    pub fn kv_cache_layout(&self) -> Option<KvCacheModelLayout> {
        self.registered.definition.kv_cache_layout()
    }

    fn reload_backend(&self) -> RuntimeBackend {
        self.registered.runtime_backend()
    }

    fn reload_tp_size(&self) -> usize {
        self.registered.tensor_parallel_size
    }
}

impl ForwardModel for BootstrapForwardModel {
    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        self.registered.runtime.forward(batch)
    }

    fn update_weights_from_disk(
        &mut self,
        request: &WorkerWeightUpdateRequest,
    ) -> Result<(), ModelForwardError> {
        let artifacts = LocalModelArtifacts::from_model_path(&request.model_path)
            .map_err(|error| ModelForwardError::Runtime(error.to_string()))?;
        let next = ModelRegistry
            .load(&artifacts, self.reload_backend(), self.reload_tp_size())
            .map(|registered| Self { registered })
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
        message: String,
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
                message,
            } => write!(
                formatter,
                "model {} resolved architecture {architecture}, but its adapter rejected the configuration: {message}",
                model_path.display()
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

fn adapter_error(
    model_path: &Path,
    architecture: &'static str,
    error: ModelAdapterError,
) -> ModelRegistryError {
    ModelRegistryError::InvalidAdapterConfig {
        model_path: model_path.to_path_buf(),
        architecture,
        message: error.to_string(),
    }
}

fn runtime_error(
    artifacts: &LocalModelArtifacts,
    architecture: &'static str,
    backend: RuntimeBackend,
    error: ModelRuntimeLoadError,
) -> ModelRegistryError {
    match error {
        ModelRuntimeLoadError::MissingCapabilities(missing) => {
            ModelRegistryError::MissingCapabilities {
                model_path: artifacts.model_path().to_path_buf(),
                architecture,
                backend,
                missing,
            }
        }
        ModelRuntimeLoadError::Load(message) => ModelRegistryError::ModelLoad {
            model_path: artifacts.model_path().to_path_buf(),
            architecture,
            message,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{
        AttentionArchitecture, FeedForwardArchitecture, ModelExecutionArchitecture,
    };

    fn mla_moe_config(architecture: &str, model_type: &str) -> HfModelConfig {
        HfModelConfig {
            model_type: Some(model_type.to_string()),
            architectures: vec![architecture.to_string()],
            num_hidden_layers: Some(4),
            num_attention_heads: Some(16),
            qk_nope_head_dim: Some(128),
            qk_rope_head_dim: Some(64),
            v_head_dim: Some(128),
            n_routed_experts: Some(32),
            n_shared_experts: Some(1),
            num_experts_per_tok: Some(4),
            moe_intermediate_size: Some(256),
            ..HfModelConfig::default()
        }
    }

    fn qwen_config() -> HfModelConfig {
        HfModelConfig {
            model_type: Some("qwen2".to_string()),
            architectures: vec!["Qwen2ForCausalLM".to_string()],
            num_hidden_layers: Some(4),
            hidden_size: Some(1024),
            intermediate_size: Some(4096),
            num_attention_heads: Some(16),
            num_key_value_heads: Some(4),
            ..HfModelConfig::default()
        }
    }

    #[test]
    fn glm_and_deepseek_share_mla_moe_execution_components() {
        let glm = ModelRegistry
            .definition(
                Path::new("/models/glm"),
                &mla_moe_config("GlmMoeDsaForCausalLM", "glm_moe_dsa"),
            )
            .expect("GLM adapter should build a model definition");
        let deepseek = ModelRegistry
            .definition(
                Path::new("/models/deepseek"),
                &mla_moe_config("DeepseekV4ForCausalLM", "deepseek_v4"),
            )
            .expect("DeepSeek adapter should build a model definition");

        assert_eq!(glm.execution(), deepseek.execution());
        assert_eq!(glm.kv_cache_layout(), deepseek.kv_cache_layout());
        assert!(matches!(
            glm.execution(),
            ModelExecutionArchitecture::Transformer {
                attention: AttentionArchitecture::MultiLatent { .. },
                feed_forward: FeedForwardArchitecture::MixtureOfExperts { .. },
            }
        ));
        glm.validate_tensor_parallel(8)
            .expect("shared MLA definition should validate TP");
        deepseek
            .validate_tensor_parallel(8)
            .expect("shared MLA definition should validate TP");
    }

    #[test]
    fn qwen_uses_the_same_definition_boundary_for_dense_decoder_components() {
        let qwen = ModelRegistry
            .definition(Path::new("/models/qwen"), &qwen_config())
            .expect("Qwen adapter should build a model definition");

        assert!(matches!(
            qwen.execution(),
            ModelExecutionArchitecture::Transformer {
                attention: AttentionArchitecture::MultiHead { .. },
                feed_forward: FeedForwardArchitecture::Dense { .. },
            }
        ));
        assert_eq!(qwen.kv_cache_layout().expect("Qwen KV layout").kv_heads, 4);
        qwen.validate_tensor_parallel(4)
            .expect("Qwen attention heads should shard over TP");
        assert!(qwen.validate_tensor_parallel(3).is_err());
    }

    #[test]
    fn generic_runtime_preflight_distinguishes_component_families_without_model_branches() {
        let backend = InitializedRuntimeBackend::CpuReference;
        let mla = ModelRegistry
            .definition(
                Path::new("/models/glm"),
                &mla_moe_config("GlmMoeDsaForCausalLM", "glm_moe_dsa"),
            )
            .expect("MLA model definition");
        let dense = ModelRegistry
            .definition(Path::new("/models/qwen"), &qwen_config())
            .expect("dense model definition");

        let mla_error = validate_runtime_support(&mla, &backend, 1)
            .expect_err("CPU reference backend has no production MLA executor");
        let dense_error = validate_runtime_support(&dense, &backend, 1)
            .expect_err("CPU reference backend has no production dense executor");

        assert!(matches!(
            mla_error,
            ModelRuntimeLoadError::MissingCapabilities(ref missing)
                if missing.iter().any(|item| item == "multi-latent attention decoder execution")
                    && missing.iter().any(|item| item == "mixture-of-experts kernels")
        ));
        assert!(matches!(
            dense_error,
            ModelRuntimeLoadError::MissingCapabilities(ref missing)
                if missing.iter().any(|item| item == "multi-head attention decoder execution")
                    && missing.iter().any(|item| item == "dense feed-forward kernels")
        ));
    }

    #[test]
    fn registry_resolves_first_registered_hugging_face_architecture() {
        let mut config = mla_moe_config("GlmMoeDsaForCausalLM", "not-a-registry-key");
        config
            .architectures
            .insert(0, "UnsupportedForCausalLM".to_string());

        let resolved = ModelRegistry
            .resolve(Path::new("/models/glm"), &config)
            .expect("registered architecture should resolve");

        assert_eq!(resolved.architecture(), "GlmMoeDsaForCausalLM");
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
    fn registry_reports_truly_unsupported_architectures() {
        let config = HfModelConfig {
            architectures: vec!["UnknownForCausalLM".to_string()],
            ..HfModelConfig::default()
        };

        let error = ModelRegistry
            .resolve(Path::new("/models/unknown"), &config)
            .expect_err("unknown architecture must fail at registry resolution");

        assert!(matches!(
            error,
            ModelRegistryError::UnsupportedArchitectures { requested, .. }
                if requested == ["UnknownForCausalLM"]
        ));
    }
}
