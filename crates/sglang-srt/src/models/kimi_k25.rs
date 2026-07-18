use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::Deserialize;

use crate::model_artifacts::HfModelConfig;

use super::deepseek::{
    DeepSeekV3DefinitionSpec, MlaMoeConfig, build_deepseek_v3_definition_for_adapter,
};
use super::{ModelAdapter, ModelAdapterError, ModelDefinition, RoutedExpertWeightFormat};

pub(crate) const KIMI_K25_ARCHITECTURE: &str = "KimiK25ForConditionalGeneration";
pub(crate) static KIMI_K25_ADAPTER: KimiK25Adapter = KimiK25Adapter;

pub(crate) struct KimiK25Adapter;

#[derive(Debug, Deserialize)]
struct KimiK25Config {
    text_config: MlaMoeConfig,
    #[serde(default)]
    encoder_only: bool,
    #[serde(default)]
    quantization_config: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct CompressedTensorsConfig {
    quant_method: String,
    format: String,
    config_groups: BTreeMap<String, CompressedTensorsGroup>,
    ignore: Vec<String>,
    kv_cache_scheme: Option<serde_json::Value>,
    quantization_status: String,
}

#[derive(Debug, Deserialize)]
struct CompressedTensorsGroup {
    targets: Vec<String>,
    input_activations: Option<serde_json::Value>,
    output_activations: Option<serde_json::Value>,
    weights: CompressedTensorsWeights,
}

#[derive(Debug, Deserialize)]
struct CompressedTensorsWeights {
    r#type: String,
    num_bits: usize,
    group_size: usize,
    strategy: String,
    symmetric: bool,
    dynamic: bool,
    observer: String,
    actorder: Option<serde_json::Value>,
    block_structure: Option<serde_json::Value>,
}

impl ModelAdapter for KimiK25Adapter {
    fn architectures(&self) -> &'static [&'static str] {
        &[KIMI_K25_ARCHITECTURE]
    }

    fn build_definition(
        &self,
        _model_path: &Path,
        hf_config: &HfModelConfig,
    ) -> Result<ModelDefinition, ModelAdapterError> {
        let config: KimiK25Config = serde_path_to_error::deserialize(
            hf_config.raw_document().clone(),
        )
        .map_err(|error| {
            ModelAdapterError::invalid(
                KIMI_K25_ARCHITECTURE,
                format!("invalid Kimi K2.5 config document: {error}"),
            )
        })?;
        if config.encoder_only {
            return Err(ModelAdapterError::invalid(
                KIMI_K25_ARCHITECTURE,
                "encoder_only Kimi K2.5 has no language model to serve",
            ));
        }
        if config
            .quantization_config
            .as_ref()
            .is_some_and(|value| !value.is_null())
        {
            return Err(ModelAdapterError::invalid(
                KIMI_K25_ARCHITECTURE,
                "Kimi K2.5 quantization_config must belong to text_config; top-level multimodal quantization is not supported",
            ));
        }
        let routed_expert_weight_format =
            kimi_routed_expert_weight_format(config.text_config.quantization_config.as_ref())?;

        build_deepseek_v3_definition_for_adapter(
            hf_config,
            config.text_config,
            DeepSeekV3DefinitionSpec {
                architecture: KIMI_K25_ARCHITECTURE,
                model_prefix: "language_model.model",
                lm_head: "language_model.lm_head.weight",
                routed_expert_weight_format,
            },
        )
    }
}

fn kimi_routed_expert_weight_format(
    quantization: Option<&serde_json::Value>,
) -> Result<RoutedExpertWeightFormat, ModelAdapterError> {
    let Some(quantization) = quantization.filter(|value| !value.is_null()) else {
        return Ok(RoutedExpertWeightFormat::Unquantized);
    };
    let config: CompressedTensorsConfig = serde_path_to_error::deserialize(quantization.clone())
        .map_err(|error| {
            ModelAdapterError::invalid(
                KIMI_K25_ARCHITECTURE,
                format!("invalid Kimi K2.5 compressed-tensors config: {error}"),
            )
        })?;
    if config.quant_method != "compressed-tensors"
        || config.format != "pack-quantized"
        || config.quantization_status != "compressed"
    {
        return Err(ModelAdapterError::invalid(
            KIMI_K25_ARCHITECTURE,
            "Kimi K2.5 requires compressed-tensors pack-quantized weights with compressed status",
        ));
    }
    if config
        .kv_cache_scheme
        .as_ref()
        .is_some_and(|value| !value.is_null())
    {
        return Err(ModelAdapterError::invalid(
            KIMI_K25_ARCHITECTURE,
            "Kimi K2.5 quantized KV cache is not supported",
        ));
    }
    if config.config_groups.len() != 1 {
        return Err(ModelAdapterError::invalid(
            KIMI_K25_ARCHITECTURE,
            "Kimi K2.5 compressed-tensors config must contain exactly one weight group",
        ));
    }
    let group = config.config_groups.values().next().ok_or_else(|| {
        ModelAdapterError::invalid(
            KIMI_K25_ARCHITECTURE,
            "Kimi K2.5 compressed-tensors weight group is missing",
        )
    })?;
    let weights = &group.weights;
    if group.targets.len() != 1
        || group.targets[0] != "Linear"
        || group
            .input_activations
            .as_ref()
            .is_some_and(|value| !value.is_null())
        || group
            .output_activations
            .as_ref()
            .is_some_and(|value| !value.is_null())
        || weights.r#type != "int"
        || weights.num_bits != 4
        || weights.group_size != 32
        || weights.strategy != "group"
        || !weights.symmetric
        || weights.dynamic
        || weights.observer != "minmax"
        || weights
            .actorder
            .as_ref()
            .is_some_and(|value| !value.is_null())
        || weights
            .block_structure
            .as_ref()
            .is_some_and(|value| !value.is_null())
    {
        return Err(ModelAdapterError::invalid(
            KIMI_K25_ARCHITECTURE,
            "Kimi K2.5 currently supports only symmetric static INT4 group-32 routed-expert weights without activation quantization",
        ));
    }
    let expected_ignore = BTreeSet::from([
        "re:.*self_attn.*",
        "re:.*shared_experts.*",
        "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
        "re:.*lm_head.*",
        "re:vision_tower.*",
        "re:mm_projector.*",
    ]);
    let actual_ignore = config
        .ignore
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if actual_ignore != expected_ignore {
        return Err(ModelAdapterError::invalid(
            KIMI_K25_ARCHITECTURE,
            "Kimi K2.5 compressed-tensors ignore rules must leave only routed experts quantized",
        ));
    }
    Ok(RoutedExpertWeightFormat::CompressedTensorsInt4 { group_size: 32 })
}
