use std::path::Path;

use crate::model_artifacts::{HfModelConfig, LocalModelArtifacts, ModelArtifactError};

use super::deepseek::build_mla_moe_definition;
use super::{ModelAdapter, ModelAdapterError, ModelDefinition};

pub(crate) const GLM_MOE_DSA_ARCHITECTURE: &str = "GlmMoeDsaForCausalLM";
pub(crate) static GLM_MOE_DSA_ADAPTER: GlmMoeDsaAdapter = GlmMoeDsaAdapter;

pub(crate) struct GlmMoeDsaAdapter;

impl ModelAdapter for GlmMoeDsaAdapter {
    fn architectures(&self) -> &'static [&'static str] {
        &[GLM_MOE_DSA_ARCHITECTURE]
    }

    fn build_definition(
        &self,
        _model_path: &Path,
        config: &HfModelConfig,
    ) -> Result<ModelDefinition, ModelAdapterError> {
        build_mla_moe_definition(GLM_MOE_DSA_ARCHITECTURE, config)
    }

    fn validate_checkpoint(
        &self,
        artifacts: &LocalModelArtifacts,
    ) -> Result<(), ModelArtifactError> {
        artifacts
            .checkpoint_catalog()?
            .glm_moe_dsa_model_weights()?;
        Ok(())
    }
}
