use std::fmt;
use std::path::{Path, PathBuf};

use crate::backend::{RuntimeBackend, RuntimeCapability, RuntimeRequirements};
use crate::backend_model::BackendProviderRegistry;
use crate::cli::ServerArgs;
use crate::kv_cache::KvCacheModelLayout;
use crate::model_artifacts::{HfModelConfig, LocalModelArtifacts, ModelArtifactError};
use crate::model_executor::{
    ForwardModel, KvCacheAllocationConfig, ModelForwardError, ModelForwardOutput, ModelWorkerBatch,
};
use crate::model_runtime::{
    LoadedModelRuntime, ModelRuntimeConfig, ModelRuntimeLoadError, validate_runtime_parallelism,
    validate_runtime_support,
};
use crate::models::{
    DEEPSEEK_V3_ADAPTER, DEEPSEEK_V4_ADAPTER, GLM_MOE_DSA_ADAPTER, KIMI_LINEAR_ADAPTER,
    ModelAdapter, ModelAdapterError, ModelDefinition, QWEN2_ADAPTER, QWEN3_5_ADAPTER,
    QWEN3_ADAPTER,
};
use crate::runtime_kv_cache::RuntimeKvCacheMetadata;
use crate::transfer::{KvCacheMemoryProvider, KvCacheTransferError, TransferableKvCacheMemory};
use crate::worker::WorkerWeightUpdateRequest;

static MODEL_ADAPTERS: [&'static dyn ModelAdapter; 7] = [
    &DEEPSEEK_V3_ADAPTER,
    &DEEPSEEK_V4_ADAPTER,
    &GLM_MOE_DSA_ADAPTER,
    &QWEN2_ADAPTER,
    &QWEN3_ADAPTER,
    &QWEN3_5_ADAPTER,
    &KIMI_LINEAR_ADAPTER,
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
        let definition = resolved.build_definition(artifacts.model_path(), artifacts.config())?;
        definition
            .validate_checkpoint(artifacts)
            .map_err(ModelRegistryError::from)
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
            BackendProviderRegistry::initialize(requested_backend, runtime_config.device_placement)
                .map_err(|error| ModelRegistryError::BackendInitialization {
                    requested: error.requested,
                    message: error.message,
                })?;

        validate_runtime_support(
            &definition,
            backend.as_ref(),
            runtime_config.tensor_parallel_size,
        )
        .map_err(|error| {
            runtime_error(
                artifacts,
                definition.architecture(),
                backend.runtime_backend(),
                error,
            )
        })?;
        definition.validate_checkpoint(artifacts)?;
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
        let requested_backend = RuntimeBackend::parse(&args.device)
            .ok_or_else(|| ModelRegistryError::InvalidDevice(args.device.clone()))?;
        validate_runtime_parallelism(args.tp_size).map_err(|message| {
            ModelRegistryError::BackendInitialization {
                requested: requested_backend,
                message,
            }
        })?;
        let artifacts = LocalModelArtifacts::from_model_path(&args.model_path)?;
        let ranks_per_node = args.tp_size.checked_div(args.nnodes).ok_or_else(|| {
            ModelRegistryError::BackendInitialization {
                requested: requested_backend,
                message: "node count must be positive".to_string(),
            }
        })?;
        let tensor_parallel_rank = args.node_rank.checked_mul(ranks_per_node).ok_or_else(|| {
            ModelRegistryError::BackendInitialization {
                requested: requested_backend,
                message: "tensor parallel rank overflowed".to_string(),
            }
        })?;
        let device_placement = crate::backend::RuntimeDevicePlacement::for_tensor_parallel_rank(
            args.base_gpu_id,
            args.gpu_id_step,
            tensor_parallel_rank,
            args.tp_size,
            args.nnodes,
        )
        .map_err(|message| ModelRegistryError::BackendInitialization {
            requested: requested_backend,
            message,
        })?;
        ModelRegistry
            .load(
                &artifacts,
                requested_backend,
                ModelRuntimeConfig {
                    tensor_parallel_size: args.tp_size,
                    device_placement,
                    kv_cache: Some(KvCacheAllocationConfig {
                        slot_capacity: args.num_reserved_decode_tokens,
                        page_size: args.page_size,
                    }),
                    recurrent_state_slot_capacity: args.max_running_requests.unwrap_or(256),
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

    #[cfg(feature = "mooncake-link")]
    pub(crate) fn runtime_device_ordinal(&self) -> Result<usize, String> {
        self.registered
            .runtime
            .config()
            .device_placement
            .device_ordinal()
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
        if self.registered.runtime.has_runtime_kv_cache() {
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

impl KvCacheMemoryProvider for BootstrapForwardModel {
    type Error = KvCacheTransferError;

    fn transferable_kv_cache_memory(&self) -> Result<TransferableKvCacheMemory, Self::Error> {
        self.registered.runtime.transferable_kv_cache_memory()
    }
}

impl RuntimeKvCacheMetadata for BootstrapForwardModel {
    fn active_kv_cache_layout(&self) -> Option<crate::kv_cache::PagedKvCacheLayout> {
        self.registered.runtime.active_kv_cache_layout()
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
        AttentionArchitecture, FeedForwardArchitecture, HybridDecoderLayerKind,
        HybridFullAttentionConfig, ModelExecutionArchitecture, MultiLatentQueryConfig,
        MultiLatentQueryWeightNames,
    };
    use serde_json::json;

    fn cpu_reference_backend() -> Box<dyn crate::model_runtime::InitializedRuntimeBackend> {
        let placement =
            crate::backend::RuntimeDevicePlacement::for_tensor_parallel_rank(0, 1, 0, 1, 1)
                .expect("CPU reference placement");
        BackendProviderRegistry::initialize(RuntimeBackend::Cpu, placement)
            .expect("CPU reference provider should initialize")
    }

    fn mla_moe_config(architecture: &str, model_type: &str) -> HfModelConfig {
        HfModelConfig::from_json_value(json!({
            "model_type": model_type,
            "architectures": [architecture],
            "vocab_size": 32_000,
            "max_position_embeddings": 32_768,
            "num_hidden_layers": 4,
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "n_routed_experts": 32,
            "n_shared_experts": 1,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 256,
            "hc_mult": 1
        }))
        .expect("valid MLA/MoE config")
    }

    fn deepseek_v3_kimi_k2_config() -> HfModelConfig {
        HfModelConfig::from_json_value(json!({
            "model_type": "kimi_k2",
            "architectures": ["DeepseekV3ForCausalLM"],
            "vocab_size": 3,
            "max_position_embeddings": 32,
            "num_hidden_layers": 2,
            "hidden_size": 2,
            "intermediate_size": 2,
            "num_attention_heads": 1,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-5,
            "rope_theta": 10_000.0,
            "rope_scaling": null,
            "attention_bias": false,
            "tie_word_embeddings": false,
            "q_lora_rank": 2,
            "kv_lora_rank": 2,
            "qk_nope_head_dim": 1,
            "qk_rope_head_dim": 2,
            "v_head_dim": 2,
            "moe_intermediate_size": 2,
            "n_routed_experts": 1,
            "n_shared_experts": 1,
            "num_experts_per_tok": 1,
            "routed_scaling_factor": 1.0,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": true,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "num_nextn_predict_layers": 0,
            "quantization_config": null
        }))
        .expect("valid Kimi-K2-compatible DeepSeek V3 config")
    }

    fn qwen_config() -> HfModelConfig {
        HfModelConfig::from_json_value(json!({
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "vocab_size": 32_000,
            "max_position_embeddings": 32_768,
            "num_hidden_layers": 4,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "tie_word_embeddings": false
        }))
        .expect("valid Qwen2 config")
    }

    fn qwen3_config() -> HfModelConfig {
        HfModelConfig::from_json_value(json!({
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "vocab_size": 151_936,
            "max_position_embeddings": 40_960,
            "num_hidden_layers": 28,
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "hidden_act": "silu",
            "attention_bias": false,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "tie_word_embeddings": true
        }))
        .expect("valid Qwen3 config")
    }

    fn qwen3_5_config() -> HfModelConfig {
        HfModelConfig::from_json_value(json!({
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {
                "model_type": "qwen3_5_text",
                "vocab_size": 248_320,
                "max_position_embeddings": 262_144,
                "num_hidden_layers": 4,
                "hidden_size": 1024,
                "intermediate_size": 3584,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "hidden_act": "silu",
                "attention_bias": false,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10_000_000.0,
                "partial_rotary_factor": 0.25,
                "attn_output_gate": true,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 16,
                "layer_types": [
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention"
                ],
                "mamba_ssm_dtype": "float32",
                "tie_word_embeddings": true
            }
        }))
        .expect("valid Qwen3.5 config")
    }

    fn kimi_linear_config() -> HfModelConfig {
        HfModelConfig::from_json_value(json!({
            "model_type": "kimi_linear",
            "architectures": ["KimiLinearForCausalLM"],
            "vocab_size": 3,
            "model_max_length": 32,
            "num_hidden_layers": 2,
            "hidden_size": 2,
            "intermediate_size": 2,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-5,
            "rope_theta": 10_000.0,
            "tie_word_embeddings": false,
            "kv_lora_rank": 2,
            "q_lora_rank": null,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 2,
            "mla_use_nope": true,
            "moe_intermediate_size": 2,
            "moe_renormalize": true,
            "moe_router_activation_func": "sigmoid",
            "num_experts": 1,
            "num_experts_per_token": 1,
            "num_shared_experts": 1,
            "routed_scaling_factor": 1.0,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "use_grouped_topk": true,
            "num_expert_group": 1,
            "topk_group": 1,
            "num_nextn_predict_layers": 0,
            "linear_attn_config": {
                "head_dim": 2,
                "num_heads": 1,
                "short_conv_kernel_size": 2,
                "kda_layers": [1],
                "full_attn_layers": [2]
            }
        }))
        .expect("valid Kimi Linear config")
    }

    fn kimi_linear_query_lora_config() -> HfModelConfig {
        let mut value = kimi_linear_config().raw_document().clone();
        value["q_lora_rank"] = json!(2);
        HfModelConfig::from_json_value(value).expect("valid Kimi Linear query LoRA config")
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
    fn kimi_k2_model_type_routes_to_the_shared_deepseek_v3_mla_moe_plan() {
        let deepseek = ModelRegistry
            .definition(
                Path::new("/models/kimi-k2-unquantized"),
                &deepseek_v3_kimi_k2_config(),
            )
            .expect("DeepSeek V3 adapter should build the Kimi-K2-compatible definition");
        let glm = ModelRegistry
            .definition(
                Path::new("/models/glm"),
                &mla_moe_config("GlmMoeDsaForCausalLM", "glm_moe_dsa"),
            )
            .expect("GLM adapter should build its shared MLA/MoE definition");

        assert_eq!(deepseek.architecture(), "DeepseekV3ForCausalLM");
        assert_eq!(
            deepseek.execution().attention_family(),
            glm.execution().attention_family()
        );
        assert_eq!(
            deepseek.execution().feed_forward_family(),
            glm.execution().feed_forward_family()
        );
        assert_eq!(
            deepseek.cache_architecture(),
            crate::models::ModelCacheArchitecture::PagedKv
        );
        assert!(deepseek.recurrent_state_layout().is_none());
        let layout = deepseek.kv_cache_layout().expect("DeepSeek V3 KV layout");
        assert_eq!(layout.num_layers, 2);
        assert_eq!(
            layout
                .tensor_pair_size_bytes(crate::kv_cache::KvCacheDtype::Float32)
                .expect("DeepSeek V3 tensor-pair geometry"),
            Some([16, 8])
        );
        let plan = deepseek.hybrid_decoder().expect("shared component plan");
        assert!(plan.linear_attention.is_none());
        assert!(plan.weights.layers.iter().all(|layer| matches!(
            layer.mixer,
            HybridDecoderLayerKind::MultiLatentAttention { .. }
        )));
        assert!(matches!(
            plan.weights.layers[0].feed_forward,
            crate::models::HybridFeedForward::Dense { .. }
        ));
        assert!(matches!(
            plan.weights.layers[1].feed_forward,
            crate::models::HybridFeedForward::MixtureOfExperts { .. }
        ));
        validate_runtime_support(&deepseek, cpu_reference_backend().as_ref(), 1)
            .expect("CPU reference backend should execute the shared pure MLA/MoE plan");
    }

    #[test]
    fn deepseek_v3_quantized_checkpoint_fails_during_adapter_loading() {
        let mut value = deepseek_v3_kimi_k2_config().raw_document().clone();
        value["quantization_config"] = json!({"quant_method": "fp8"});
        let config = HfModelConfig::from_json_value(value).expect("routing config should parse");

        let error = ModelRegistry
            .definition(Path::new("/models/kimi-k2-fp8"), &config)
            .expect_err("unsupported quantization must fail before backend initialization");

        assert!(matches!(
            error,
            ModelRegistryError::InvalidAdapterConfig { ref message, .. }
                if message.contains("quantized DeepSeek/Kimi checkpoints are not implemented")
        ));
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
        let recurrent = qwen3_5
            .recurrent_state_layout()
            .expect("Qwen3.5 recurrent-state layout");
        assert_eq!(recurrent.layer_count, 3);
        assert_eq!(recurrent.conv_elements_per_layer(), Some(18_432));
        assert_eq!(recurrent.temporal_elements_per_layer(), Some(262_144));
        assert_eq!(recurrent.elements_per_request(), Some(841_728));
        assert_eq!(
            qwen3_5
                .kv_cache_layout()
                .expect("full-attention KV layout")
                .num_layers,
            1
        );
        let backend = cpu_reference_backend();
        validate_runtime_support(&qwen3_5, backend.as_ref(), 1)
            .expect("CPU reference backend should execute a shared hybrid decoder plan");
        qwen3_5
            .validate_tensor_parallel(8)
            .expect("all hybrid head families should support TP 8");
        assert!(qwen3_5.validate_tensor_parallel(3).is_err());
    }

    #[test]
    fn kimi_and_qwen3_5_use_distinct_components_through_the_shared_hybrid_executor() {
        let kimi = ModelRegistry
            .definition(Path::new("/models/kimi-linear"), &kimi_linear_config())
            .expect("Kimi Linear definition");
        let qwen3_5 = ModelRegistry
            .definition(Path::new("/models/qwen3.5"), &qwen3_5_config())
            .expect("Qwen3.5 definition");

        assert!(matches!(
            kimi.execution(),
            ModelExecutionArchitecture::Transformer {
                attention: AttentionArchitecture::HybridMultiLatent { .. },
                feed_forward: FeedForwardArchitecture::MixtureOfExperts { .. },
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
            kimi.cache_architecture(),
            crate::models::ModelCacheArchitecture::HybridState {
                full_attention_layer_count: 1,
                recurrent_state_layer_count: 1,
            }
        );
        let recurrent = kimi
            .recurrent_state_layout()
            .expect("Kimi recurrent-state layout");
        assert_eq!(recurrent.layer_count, 1);
        assert_eq!(recurrent.conv_elements_per_layer(), Some(6));
        assert_eq!(recurrent.temporal_elements_per_layer(), Some(4));
        assert_eq!(recurrent.elements_per_request(), Some(10));
        let layout = kimi.kv_cache_layout().expect("Kimi MLA KV layout");
        assert_eq!(layout.num_layers, 1);
        assert_eq!(
            layout
                .tensor_pair_size_bytes(crate::kv_cache::KvCacheDtype::Float32)
                .expect("Kimi tensor-pair geometry"),
            Some([16, 8])
        );
        let plan = kimi.hybrid_decoder().expect("shared hybrid plan");
        assert!(matches!(
            plan.weights.layers[0].mixer,
            crate::models::HybridDecoderLayerKind::KeyGatedDelta { .. }
        ));
        assert!(matches!(
            plan.weights.layers[1].mixer,
            crate::models::HybridDecoderLayerKind::MultiLatentAttention { .. }
        ));
        assert!(matches!(
            plan.weights.layers[0].feed_forward,
            crate::models::HybridFeedForward::Dense { .. }
        ));
        assert!(matches!(
            plan.weights.layers[1].feed_forward,
            crate::models::HybridFeedForward::MixtureOfExperts { .. }
        ));
        let backend = cpu_reference_backend();
        validate_runtime_support(&kimi, backend.as_ref(), 1)
            .expect("CPU reference backend should execute the shared Kimi component plan");
    }

    #[test]
    fn kimi_query_lora_uses_the_shared_mla_projection_component() {
        let kimi = ModelRegistry
            .definition(
                Path::new("/models/kimi-linear-query-lora"),
                &kimi_linear_query_lora_config(),
            )
            .expect("Kimi Linear query LoRA definition");
        let plan = kimi.hybrid_decoder().expect("shared hybrid plan");

        assert!(matches!(
            plan.full_attention,
            HybridFullAttentionConfig::MultiLatent {
                query: MultiLatentQueryConfig::LowRank { rank: 2 },
                ..
            }
        ));
        assert!(matches!(
            &plan.weights.layers[1].mixer,
            HybridDecoderLayerKind::MultiLatentAttention { weights, .. }
                if matches!(
                    &weights.query,
                    MultiLatentQueryWeightNames::LowRank { .. }
                )
        ));
    }

    #[test]
    fn generic_runtime_preflight_distinguishes_component_families_without_model_branches() {
        let backend = cpu_reference_backend();
        let mla = ModelRegistry
            .definition(
                Path::new("/models/glm"),
                &mla_moe_config("GlmMoeDsaForCausalLM", "glm_moe_dsa"),
            )
            .expect("MLA model definition");
        let dense = ModelRegistry
            .definition(Path::new("/models/qwen"), &qwen_config())
            .expect("dense model definition");

        let mla_error = validate_runtime_support(&mla, backend.as_ref(), 1)
            .expect_err("CPU reference backend has no production MLA executor");
        validate_runtime_support(&dense, backend.as_ref(), 1)
            .expect("CPU reference backend should execute the shared dense decoder plan");

        assert!(matches!(
            mla_error,
            ModelRuntimeLoadError::MissingCapabilities(ref missing)
                if missing.iter().any(|item| item.contains("multi-latent attention"))
                    && missing.iter().any(|item| item.contains("mixture-of-experts"))
        ));
    }

    #[test]
    fn cuda_hybrid_preflight_accepts_shared_kimi_components() {
        let kimi = ModelRegistry
            .definition(Path::new("/models/kimi"), &kimi_linear_config())
            .expect("Kimi Linear definition");

        let missing = crate::cuda_hybrid_decoder::CudaBf16HybridDecoder::missing_components(&kimi);

        assert!(
            missing.is_empty(),
            "unexpected missing components: {missing:?}"
        );
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
    fn adapter_typed_config_errors_fail_definition_build() {
        let config = HfModelConfig::from_json_value(json!({
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "num_hidden_layers": "not-an-integer"
        }))
        .expect("routing config should preserve the raw document");

        let error = ModelRegistry
            .definition(Path::new("/models/invalid-qwen"), &config)
            .expect_err("typed Qwen config parsing must fail before runtime startup");

        assert!(matches!(
            error,
            ModelRegistryError::InvalidAdapterConfig { .. }
        ));
        assert!(error.to_string().contains("num_hidden_layers"));
    }

    #[test]
    fn registry_reports_truly_unsupported_architectures() {
        let config = HfModelConfig::from_json_value(json!({
            "architectures": ["SglangEmbeddingLmForCausalLM"]
        }))
        .expect("valid unsupported architecture config");

        let error = ModelRegistry
            .resolve(Path::new("/models/unknown"), &config)
            .expect_err("unknown architecture must fail at registry resolution");

        assert!(matches!(
            error,
            ModelRegistryError::UnsupportedArchitectures { requested, .. }
                if requested == ["SglangEmbeddingLmForCausalLM"]
        ));
        assert!(
            !ModelRegistry
                .supported_architectures()
                .contains(&"SglangEmbeddingLmForCausalLM")
        );
    }
}
