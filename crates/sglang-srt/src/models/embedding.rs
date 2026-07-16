use std::path::Path;

use crate::backend::RuntimeDtype;
use crate::model_artifacts::{HfModelConfig, LocalModelArtifacts, ModelArtifactError};
use crate::model_executor::CpuEmbeddingLmModel;

use super::{ModelAdapter, ModelAdapterError, ModelDefinition, ModelExecutionArchitecture};

pub(crate) const EMBEDDING_LM_ARCHITECTURE: &str = "SglangEmbeddingLmForCausalLM";
pub(crate) static EMBEDDING_LM_ADAPTER: EmbeddingLmAdapter = EmbeddingLmAdapter;

pub(crate) struct EmbeddingLmAdapter;

impl ModelAdapter for EmbeddingLmAdapter {
    fn architectures(&self) -> &'static [&'static str] {
        &[EMBEDDING_LM_ARCHITECTURE]
    }

    fn build_definition(
        &self,
        _model_path: &Path,
        config: &HfModelConfig,
    ) -> Result<ModelDefinition, ModelAdapterError> {
        Ok(ModelDefinition::new(
            EMBEDDING_LM_ARCHITECTURE,
            config,
            ModelExecutionArchitecture::Embedding,
            vec![RuntimeDtype::F32],
            None,
        ))
    }

    fn validate_checkpoint(
        &self,
        artifacts: &LocalModelArtifacts,
    ) -> Result<(), ModelArtifactError> {
        CpuEmbeddingLmModel::from_local_model_artifacts(artifacts)?.ok_or_else(|| {
            ModelArtifactError::InvalidSafetensorsData {
                path: artifacts.model_path().to_path_buf(),
                message: "embedding LM adapter rejected the model configuration".to_string(),
            }
        })?;
        Ok(())
    }
}
