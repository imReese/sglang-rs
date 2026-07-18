use std::path::Path;

use serde::Deserialize;

use crate::model_artifacts::HfModelConfig;

use super::deepseek::{
    DeepSeekV3DefinitionSpec, MlaMoeConfig, build_deepseek_v3_definition_for_adapter,
};
use super::{ModelAdapter, ModelAdapterError, ModelDefinition};

pub(crate) const KIMI_K25_ARCHITECTURE: &str = "KimiK25ForConditionalGeneration";
pub(crate) static KIMI_K25_ADAPTER: KimiK25Adapter = KimiK25Adapter;

pub(crate) struct KimiK25Adapter;

#[derive(Debug, Deserialize)]
struct KimiK25Config {
    text_config: MlaMoeConfig,
    #[serde(default)]
    encoder_only: bool,
    quantization_config: Option<serde_json::Value>,
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
                "quantized Kimi K2.5 checkpoints are not implemented; compressed-tensors support is required",
            ));
        }

        build_deepseek_v3_definition_for_adapter(
            hf_config,
            config.text_config,
            DeepSeekV3DefinitionSpec {
                architecture: KIMI_K25_ARCHITECTURE,
                model_prefix: "language_model.model",
                lm_head: "language_model.lm_head.weight",
            },
        )
    }
}
