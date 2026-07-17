use std::fmt;
use std::path::{Path, PathBuf};

use crate::backend::{
    InitializedRuntimeBackend, RuntimeBackend, RuntimeCapability, RuntimeRequirements,
};
use crate::cli::ServerArgs;
use crate::model_artifacts::{HfModelConfig, LocalModelArtifacts, ModelArtifactError};
use crate::model_executor::{
    ForwardModel, KvCacheAllocationConfig, ModelForwardError, ModelForwardOutput, ModelWorkerBatch,
};
use crate::model_runtime::{
    LoadedModelRuntime, ModelRuntimeConfig, ModelRuntimeLoadError, validate_runtime_support,
};
use crate::models::{
    DEEPSEEK_V4_ADAPTER, EMBEDDING_LM_ADAPTER, GLM_MOE_DSA_ADAPTER, ModelAdapter,
    ModelAdapterError, ModelDefinition, QWEN2_ADAPTER, QWEN3_5_ADAPTER, QWEN3_ADAPTER,
};
use crate::transfer::{KvCacheModelLayout, TransferableKvCacheMemory};
use crate::worker::WorkerWeightUpdateRequest;

static MODEL_ADAPTERS: [&'static dyn ModelAdapter; 6] = [
    &EMBEDDING_LM_ADAPTER,
    &DEEPSEEK_V4_ADAPTER,
    &GLM_MOE_DSA_ADAPTER,
    &QWEN2_ADAPTER,
    &QWEN3_ADAPTER,
    &QWEN3_5_ADAPTER,
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
        runtime_config: ModelRuntimeConfig,
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

        validate_runtime_support(&definition, &backend, runtime_config.tensor_parallel_size)
            .map_err(|error| {
                runtime_error(
                    artifacts,
                    definition.architecture(),
                    backend.runtime_backend(),
                    error,
                )
            })?;
        resolved.validate_checkpoint(artifacts)?;
        let runtime_backend = backend.runtime_backend();
        let runtime = LoadedModelRuntime::load(&definition, artifacts, backend, runtime_config)
            .map_err(|error| {
                runtime_error(artifacts, definition.architecture(), runtime_backend, error)
            })?;

        Ok(RegisteredModel {
            model_path: artifacts.model_path().to_path_buf(),
            definition,
            tensor_parallel_size: runtime_config.tensor_parallel_size,
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
            .load(
                &artifacts,
                requested_backend,
                ModelRuntimeConfig {
                    tensor_parallel_size: args.tp_size,
                    kv_cache: Some(KvCacheAllocationConfig {
                        slot_capacity: args.num_reserved_decode_tokens,
                        page_size: args.page_size,
                    }),
                },
            )
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
        self.registered.definition.runtime_requirements(
            self.registered.runtime.execution_dtype(),
            tensor_parallel_size,
            requested_attention_backend,
        )
    }

    pub fn kv_cache_layout(&self) -> Option<KvCacheModelLayout> {
        self.registered.definition.kv_cache_layout()
    }

    pub fn cache_architecture(&self) -> crate::models::ModelCacheArchitecture {
        self.registered.definition.cache_architecture()
    }

    pub(crate) fn transferable_kv_cache_memory(&self) -> Option<TransferableKvCacheMemory> {
        self.registered.runtime.transferable_kv_cache_memory()
    }

    fn reload_backend(&self) -> RuntimeBackend {
        self.registered.runtime_backend()
    }

    fn reload_runtime_config(&self) -> ModelRuntimeConfig {
        self.registered.runtime.config()
    }
}

impl ForwardModel for BootstrapForwardModel {
    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        self.registered.runtime.forward(batch)
    }

    fn complete_request(&mut self, request_id: &crate::types::RequestId) {
        self.registered.runtime.complete_request(request_id);
    }

    fn update_weights_from_disk(
        &mut self,
        request: &WorkerWeightUpdateRequest,
    ) -> Result<(), ModelForwardError> {
        if self.transferable_kv_cache_memory().is_some() {
            return Err(ModelForwardError::Runtime(
                "update_weights_from_disk requires a runtime restart when the model owns transferable KV memory; replacing a registered allocation in place is unsupported"
                    .to_string(),
            ));
        }
        let artifacts = LocalModelArtifacts::from_model_path(&request.model_path)
            .map_err(|error| ModelForwardError::Runtime(error.to_string()))?;
        let next = ModelRegistry
            .load(
                &artifacts,
                self.reload_backend(),
                self.reload_runtime_config(),
            )
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
    use crate::model_artifacts::HfConfigFloat;
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
            vocab_size: Some(32_000),
            max_position_embeddings: Some(32_768),
            num_hidden_layers: Some(4),
            hidden_size: Some(1024),
            intermediate_size: Some(4096),
            num_attention_heads: Some(16),
            num_key_value_heads: Some(4),
            hidden_act: Some("silu".to_string()),
            rms_norm_eps: Some(HfConfigFloat::new(1e-6)),
            rope_theta: Some(HfConfigFloat::new(1_000_000.0)),
            tie_word_embeddings: Some(false),
            ..HfModelConfig::default()
        }
    }

    fn qwen3_config() -> HfModelConfig {
        HfModelConfig {
            model_type: Some("qwen3".to_string()),
            architectures: vec!["Qwen3ForCausalLM".to_string()],
            vocab_size: Some(151_936),
            max_position_embeddings: Some(40_960),
            num_hidden_layers: Some(28),
            hidden_size: Some(1024),
            intermediate_size: Some(3072),
            num_attention_heads: Some(16),
            num_key_value_heads: Some(8),
            head_dim: Some(128),
            hidden_act: Some("silu".to_string()),
            attention_bias: Some(false),
            rms_norm_eps: Some(HfConfigFloat::new(1e-6)),
            rope_theta: Some(HfConfigFloat::new(1_000_000.0)),
            tie_word_embeddings: Some(true),
            ..HfModelConfig::default()
        }
    }

    fn qwen3_5_config() -> HfModelConfig {
        HfModelConfig {
            model_type: Some("qwen3_5".to_string()),
            text_model_type: Some("qwen3_5_text".to_string()),
            architectures: vec!["Qwen3_5ForConditionalGeneration".to_string()],
            vocab_size: Some(248_320),
            max_position_embeddings: Some(262_144),
            num_hidden_layers: Some(4),
            hidden_size: Some(1024),
            intermediate_size: Some(3584),
            num_attention_heads: Some(8),
            num_key_value_heads: Some(2),
            head_dim: Some(256),
            hidden_act: Some("silu".to_string()),
            attention_bias: Some(false),
            rms_norm_eps: Some(HfConfigFloat::new(1e-6)),
            rope_theta: Some(HfConfigFloat::new(10_000_000.0)),
            partial_rotary_factor: Some(HfConfigFloat::new(0.25)),
            attn_output_gate: Some(true),
            linear_conv_kernel_dim: Some(4),
            linear_key_head_dim: Some(128),
            linear_value_head_dim: Some(128),
            linear_num_key_heads: Some(16),
            linear_num_value_heads: Some(16),
            layer_types: vec![
                "linear_attention".to_string(),
                "linear_attention".to_string(),
                "linear_attention".to_string(),
                "full_attention".to_string(),
            ],
            mamba_ssm_dtype: Some("float32".to_string()),
            tie_word_embeddings: Some(true),
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
    fn qwen2_and_qwen3_share_dense_execution_and_kv_boundaries() {
        let qwen2 = ModelRegistry
            .definition(Path::new("/models/qwen2"), &qwen_config())
            .expect("Qwen2 adapter should build a model definition");
        let qwen3 = ModelRegistry
            .definition(Path::new("/models/qwen3"), &qwen3_config())
            .expect("Qwen3 adapter should build a model definition");

        assert_eq!(
            qwen2.execution().attention_family(),
            qwen3.execution().attention_family()
        );
        assert_eq!(
            qwen2.execution().feed_forward_family(),
            qwen3.execution().feed_forward_family()
        );
        assert_eq!(
            qwen3.kv_cache_layout().expect("Qwen3 KV layout").kv_heads,
            8
        );
        qwen3
            .validate_tensor_parallel(8)
            .expect("Qwen3 attention heads should shard over TP");
    }

    #[test]
    fn qwen3_5_uses_shared_hybrid_execution_without_changing_qwen3_dense_boundary() {
        let qwen3 = ModelRegistry
            .definition(Path::new("/models/qwen3"), &qwen3_config())
            .expect("Qwen3 dense definition");
        let qwen3_5 = ModelRegistry
            .definition(Path::new("/models/qwen3.5"), &qwen3_5_config())
            .expect("Qwen3.5 hybrid definition");

        assert!(matches!(
            qwen3.execution(),
            ModelExecutionArchitecture::Transformer {
                attention: AttentionArchitecture::MultiHead { .. },
                feed_forward: FeedForwardArchitecture::Dense { .. },
            }
        ));
        assert!(matches!(
            qwen3_5.execution(),
            ModelExecutionArchitecture::Transformer {
                attention: AttentionArchitecture::Hybrid { .. },
                feed_forward: FeedForwardArchitecture::Dense { .. },
            }
        ));
        assert_eq!(
            qwen3_5.cache_architecture(),
            crate::models::ModelCacheArchitecture::HybridState {
                full_attention_layer_count: 1,
                recurrent_state_layer_count: 3,
            }
        );
        assert_eq!(
            qwen3_5
                .kv_cache_layout()
                .expect("full-attention KV layout")
                .num_layers,
            1
        );
        validate_runtime_support(&qwen3_5, &InitializedRuntimeBackend::CpuReference, 1)
            .expect("CPU reference backend should execute a shared hybrid decoder plan");
        qwen3_5
            .validate_tensor_parallel(8)
            .expect("all hybrid head families should support TP 8");
        assert!(qwen3_5.validate_tensor_parallel(3).is_err());
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
        validate_runtime_support(&dense, &backend, 1)
            .expect("CPU reference backend should execute the shared dense decoder plan");

        assert!(matches!(
            mla_error,
            ModelRuntimeLoadError::MissingCapabilities(ref missing)
                if missing.iter().any(|item| item.contains("multi-latent attention"))
                    && missing.iter().any(|item| item.contains("mixture-of-experts"))
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
