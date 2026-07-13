use std::fmt;

use sglang_kernel::cublas::{CudaBlas, CudaBlasError};
use sglang_kernel::cuda::{CudaDeviceAllocation, CudaDeviceInfo, CudaError};

use crate::backend::{CapabilityStatus, CudaBackend, RuntimeCapability, RuntimeDtype};
use crate::model_artifacts::{LocalModelArtifacts, ModelArtifactError};
use crate::model_executor::{
    EmbeddingLmWeights, ForwardModel, ModelForwardError, ModelForwardOutput, ModelWorkerBatch,
};

#[derive(Debug)]
pub enum CudaEmbeddingLmError {
    ModelArtifact(ModelArtifactError),
    Cuda(CudaError),
    CudaBlas(CudaBlasError),
    WeightByteSizeOverflow {
        tensor_name: &'static str,
        element_count: usize,
    },
}

impl fmt::Display for CudaEmbeddingLmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelArtifact(error) => write!(formatter, "CUDA model artifact error: {error}"),
            Self::Cuda(error) => write!(formatter, "CUDA model operation failed: {error}"),
            Self::CudaBlas(error) => {
                write!(formatter, "CUDA model cuBLAS operation failed: {error}")
            }
            Self::WeightByteSizeOverflow {
                tensor_name,
                element_count,
            } => write!(
                formatter,
                "CUDA tensor {tensor_name} with {element_count} elements exceeds host addressable size"
            ),
        }
    }
}

impl std::error::Error for CudaEmbeddingLmError {}

impl From<ModelArtifactError> for CudaEmbeddingLmError {
    fn from(value: ModelArtifactError) -> Self {
        Self::ModelArtifact(value)
    }
}

impl From<CudaError> for CudaEmbeddingLmError {
    fn from(value: CudaError) -> Self {
        Self::Cuda(value)
    }
}

impl From<CudaBlasError> for CudaEmbeddingLmError {
    fn from(value: CudaBlasError) -> Self {
        Self::CudaBlas(value)
    }
}

pub struct CudaEmbeddingLmModel {
    backend: CudaBackend,
    blas: CudaBlas,
    token_embeddings: CudaDeviceAllocation,
    lm_head: CudaDeviceAllocation,
    logits: CudaDeviceAllocation,
    vocab_size: usize,
    hidden_size: usize,
}

impl fmt::Debug for CudaEmbeddingLmModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaEmbeddingLmModel")
            .field("device", self.backend.device())
            .field("vocab_size", &self.vocab_size)
            .field("hidden_size", &self.hidden_size)
            .finish_non_exhaustive()
    }
}

impl CudaEmbeddingLmModel {
    pub fn from_local_model_artifacts(
        artifacts: &LocalModelArtifacts,
        backend: CudaBackend,
    ) -> Result<Option<Self>, CudaEmbeddingLmError> {
        let Some(weights) = EmbeddingLmWeights::from_local_model_artifacts(artifacts)? else {
            return Ok(None);
        };
        Self::from_weights(weights, backend).map(Some)
    }

    pub fn from_weights(
        weights: EmbeddingLmWeights,
        backend: CudaBackend,
    ) -> Result<Self, CudaEmbeddingLmError> {
        let token_embeddings = upload_f32_tensor(
            &backend,
            "model.embed_tokens.weight",
            weights.token_embeddings(),
        )?;
        let lm_head = upload_f32_tensor(&backend, "lm_head.weight", weights.lm_head())?;
        let logits_byte_len = checked_f32_byte_len("logits", weights.vocab_size())?;
        let mut logits = backend.context().allocate(logits_byte_len)?;
        logits.fill(0)?;
        let blas = CudaBlas::load(backend.context())?;

        Ok(Self {
            backend,
            blas,
            token_embeddings,
            lm_head,
            logits,
            vocab_size: weights.vocab_size(),
            hidden_size: weights.hidden_size(),
        })
    }

    pub fn device(&self) -> &CudaDeviceInfo {
        self.backend.device()
    }

    pub fn runtime_capability(&self) -> RuntimeCapability {
        let mut capability = self.backend.capabilities();
        capability.runtime_name = "cuda-cublas-embedding-lm";
        capability.supports_forward = true;
        capability.supports_transferable_kv = false;
        capability.supported_dtypes = vec![RuntimeDtype::F32];
        capability.attention_backends.clear();
        capability.tensor_parallel = CapabilityStatus::Unsupported;
        capability.kv_cache_memory_registration = CapabilityStatus::Unsupported;
        capability.mooncake = CapabilityStatus::Unsupported;
        capability
    }

    fn logits_for_token(&mut self, token_id: u32) -> Result<Vec<f32>, ModelForwardError> {
        let token_id = usize::try_from(token_id).map_err(|_| {
            ModelForwardError::Runtime(format!("token id {token_id} does not fit usize"))
        })?;
        if token_id >= self.vocab_size {
            return Err(ModelForwardError::Runtime(format!(
                "token id {token_id} is outside CUDA embedding LM vocabulary {}",
                self.vocab_size
            )));
        }
        let vector_offset_bytes = token_id
            .checked_mul(self.hidden_size)
            .and_then(|offset| offset.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                ModelForwardError::Runtime(
                    "CUDA embedding row device offset overflowed".to_string(),
                )
            })?;
        self.blas
            .sgemv_row_major(
                &self.lm_head,
                self.vocab_size,
                self.hidden_size,
                &self.token_embeddings,
                vector_offset_bytes,
                &mut self.logits,
            )
            .map_err(|error| ModelForwardError::Runtime(error.to_string()))?;

        let mut logits = vec![0.0_f32; self.vocab_size];
        self.logits
            .copy_to_host(0, f32_slice_as_bytes_mut(&mut logits))
            .map_err(|error| ModelForwardError::Runtime(error.to_string()))?;
        Ok(logits)
    }
}

impl ForwardModel for CudaEmbeddingLmModel {
    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        let logits = batch
            .last_input_token_ids()
            .into_iter()
            .map(|token_id| self.logits_for_token(token_id))
            .collect::<Result<Vec<_>, _>>()?;
        ModelForwardOutput::new(logits)
    }
}

fn upload_f32_tensor(
    backend: &CudaBackend,
    tensor_name: &'static str,
    values: &[f32],
) -> Result<CudaDeviceAllocation, CudaEmbeddingLmError> {
    let byte_len = checked_f32_byte_len(tensor_name, values.len())?;
    let mut allocation = backend.context().allocate(byte_len)?;
    allocation.copy_from_host(0, f32_slice_as_bytes(values))?;
    Ok(allocation)
}

fn checked_f32_byte_len(
    tensor_name: &'static str,
    element_count: usize,
) -> Result<usize, CudaEmbeddingLmError> {
    element_count.checked_mul(std::mem::size_of::<f32>()).ok_or(
        CudaEmbeddingLmError::WeightByteSizeOverflow {
            tensor_name,
            element_count,
        },
    )
}

fn f32_slice_as_bytes(values: &[f32]) -> &[u8] {
    // f32 has no invalid bit patterns and the returned byte slice cannot outlive `values`.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn f32_slice_as_bytes_mut(values: &mut [f32]) -> &mut [u8] {
    // f32 has no invalid bit patterns and the returned byte slice exclusively borrows `values`.
    unsafe {
        std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), std::mem::size_of_val(values))
    }
}
