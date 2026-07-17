use std::fmt;

use sglang_kernel::cublas::{CudaBlas, CudaBlasError};
use sglang_kernel::cuda::{CudaContext, CudaDeviceAllocation, CudaError};
use sglang_kernel::cuda_bf16_kernels::{CudaBf16DenseKernels, CudaBf16KernelError};
use sglang_kernel::cuda_kv_kernels::{CudaKvPairCopyError, CudaKvPairCopyKernels};

use crate::backend::{CapabilityStatus, CudaBackend, RuntimeCapability, RuntimeDtype};
use crate::cuda_attention::{
    CudaBf16PagedAttentionExecutor, CudaPagedAttentionError, CudaPagedAttentionForward,
    CudaPagedAttentionMetadata,
};
use crate::cuda_kv_cache::{CudaKvCachePool, CudaKvCachePoolError, CudaKvSlotScatterLaunch};
use crate::model_artifacts::{
    LocalModelArtifacts, ModelArtifactError, SafetensorsTensorDecodeError,
};
use crate::model_executor::{
    ForwardModel, KvCacheAllocationConfig, ModelForwardError, ModelForwardOutput, ModelWorkerBatch,
    validate_model_worker_batch,
};
use crate::models::{
    AttentionArchitecture, DenseDecoderExecutionPlan, DenseDecoderLayerWeightNames,
    FeedForwardArchitecture, ModelDefinition, ModelExecutionArchitecture,
};
use crate::transfer::{KvCacheDtype, KvCacheRuntimeLayout, TransferableKvCacheMemory};

const BF16_BYTES: usize = 2;

#[derive(Debug)]
pub(crate) enum CudaDenseDecoderError {
    Unsupported(String),
    Shape(String),
    MissingTensor(String),
    ModelArtifact(ModelArtifactError),
    TensorDecode(SafetensorsTensorDecodeError),
    Cuda(CudaError),
    CudaBlas(CudaBlasError),
    Kernel(CudaBf16KernelError),
    Attention(CudaPagedAttentionError),
    KvCache(CudaKvCachePoolError),
    KvCopy(CudaKvPairCopyError),
}

impl fmt::Display for CudaDenseDecoderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(message) => {
                write!(formatter, "unsupported CUDA dense decoder: {message}")
            }
            Self::Shape(message) => write!(formatter, "CUDA dense decoder shape error: {message}"),
            Self::MissingTensor(name) => {
                write!(formatter, "CUDA dense decoder tensor {name} is missing")
            }
            Self::ModelArtifact(error) => {
                write!(formatter, "CUDA dense decoder artifact error: {error}")
            }
            Self::TensorDecode(error) => write!(
                formatter,
                "CUDA dense decoder tensor decode failed: {error}"
            ),
            Self::Cuda(error) => write!(
                formatter,
                "CUDA dense decoder device operation failed: {error}"
            ),
            Self::CudaBlas(error) => write!(
                formatter,
                "CUDA dense decoder cuBLAS operation failed: {error}"
            ),
            Self::Kernel(error) => write!(formatter, "CUDA dense decoder kernel failed: {error}"),
            Self::Attention(error) => {
                write!(formatter, "CUDA dense decoder attention failed: {error}")
            }
            Self::KvCache(error) => {
                write!(formatter, "CUDA dense decoder KV cache failed: {error}")
            }
            Self::KvCopy(error) => {
                write!(formatter, "CUDA dense decoder KV scatter failed: {error}")
            }
        }
    }
}

impl std::error::Error for CudaDenseDecoderError {}

macro_rules! error_conversion {
    ($source:ty, $variant:ident) => {
        impl From<$source> for CudaDenseDecoderError {
            fn from(value: $source) -> Self {
                Self::$variant(value)
            }
        }
    };
}

error_conversion!(ModelArtifactError, ModelArtifact);
error_conversion!(SafetensorsTensorDecodeError, TensorDecode);
error_conversion!(CudaError, Cuda);
error_conversion!(CudaBlasError, CudaBlas);
error_conversion!(CudaBf16KernelError, Kernel);
error_conversion!(CudaPagedAttentionError, Attention);
error_conversion!(CudaKvCachePoolError, KvCache);
error_conversion!(CudaKvPairCopyError, KvCopy);

pub(crate) struct CudaBf16DenseDecoder {
    backend: CudaBackend,
    plan: DenseDecoderExecutionPlan,
    shape: CudaDenseDecoderShape,
    blas: CudaBlas,
    kernels: CudaBf16DenseKernels,
    attention: CudaBf16PagedAttentionExecutor,
    kv_copy: CudaKvPairCopyKernels,
    kv_cache: CudaKvCachePool,
    transferable_kv_cache_memory: TransferableKvCacheMemory,
    token_embeddings: CudaBf16Matrix,
    final_norm: CudaDeviceAllocation,
    lm_head: Option<CudaBf16Matrix>,
    layers: Vec<CudaDenseDecoderLayer>,
}

impl fmt::Debug for CudaBf16DenseDecoder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaBf16DenseDecoder")
            .field("device", self.backend.device())
            .field("shape", &self.shape)
            .field("slot_count", &self.kv_cache.layout().slot_count())
            .finish_non_exhaustive()
    }
}

impl CudaBf16DenseDecoder {
    pub(crate) fn load(
        definition: &ModelDefinition,
        artifacts: &LocalModelArtifacts,
        backend: CudaBackend,
        config: KvCacheAllocationConfig,
    ) -> Result<Self, CudaDenseDecoderError> {
        let plan = definition.dense_decoder().cloned().ok_or_else(|| {
            CudaDenseDecoderError::Unsupported(
                "model definition has no shared dense decoder execution plan".to_string(),
            )
        })?;
        let shape = CudaDenseDecoderShape::from_definition(definition, &plan)?;
        validate_cache_config(config)?;
        let model_layout = definition.kv_cache_layout().ok_or_else(|| {
            CudaDenseDecoderError::Unsupported(
                "model definition has no paged KV cache geometry".to_string(),
            )
        })?;
        let bytes_per_token = model_layout
            .token_size_bytes(KvCacheDtype::Bfloat16)
            .map_err(|error| CudaDenseDecoderError::Shape(error.to_string()))?;
        let page_size_bytes = config
            .page_size
            .checked_mul(bytes_per_token)
            .ok_or_else(|| CudaDenseDecoderError::Shape("KV page size overflowed".to_string()))?;
        let kv_runtime_layout = KvCacheRuntimeLayout {
            dtype: KvCacheDtype::Bfloat16,
            page_size: config.page_size,
            num_layers: model_layout.num_layers,
            kv_heads: model_layout.kv_heads,
            head_dim: model_layout.head_dim,
            kv_tensors_per_token: model_layout.kv_tensors_per_token,
            bytes_per_token,
            page_size_bytes,
        };
        let page_count = config.slot_capacity / config.page_size;
        let kv_cache = CudaKvCachePool::allocate(backend.context(), kv_runtime_layout, page_count)?;
        let transferable_kv_cache_memory = kv_cache.transferable_memory()?;
        let compute_capability = backend.device().compute_capability;
        let blas = CudaBlas::load(backend.context())?;
        let kernels = CudaBf16DenseKernels::compile(backend.context(), compute_capability)?;
        let attention =
            CudaBf16PagedAttentionExecutor::compile(backend.context(), compute_capability)?;
        let kv_copy = CudaKvPairCopyKernels::compile(backend.context(), compute_capability)?;

        let token_embeddings = CudaBf16Matrix::load(
            artifacts,
            backend.context(),
            &plan.weights.token_embeddings,
            plan.vocab_size,
            plan.hidden_size,
        )?;
        let final_norm = upload_required_bf16(
            artifacts,
            backend.context(),
            &plan.weights.final_norm,
            plan.hidden_size,
        )?;
        let lm_head = plan
            .weights
            .lm_head
            .as_deref()
            .map(|name| {
                CudaBf16Matrix::load(
                    artifacts,
                    backend.context(),
                    name,
                    plan.vocab_size,
                    plan.hidden_size,
                )
            })
            .transpose()?;
        let layers = plan
            .weights
            .layers
            .iter()
            .map(|names| CudaDenseDecoderLayer::load(artifacts, backend.context(), names, shape))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            backend,
            plan,
            shape,
            blas,
            kernels,
            attention,
            kv_copy,
            kv_cache,
            transferable_kv_cache_memory,
            token_embeddings,
            final_norm,
            lm_head,
            layers,
        })
    }

    pub(crate) fn runtime_capability(&self) -> RuntimeCapability {
        let mut capability = self.backend.capabilities();
        capability.runtime_name = "cuda-bf16-dense-decoder";
        capability.supports_forward = true;
        capability.supported_dtypes = vec![RuntimeDtype::Bf16];
        capability.attention_backends = vec!["cuda-paged-bf16"];
        capability.tensor_parallel = CapabilityStatus::Unsupported;
        capability
    }

    pub(crate) fn transferable_kv_cache_memory(&self) -> &TransferableKvCacheMemory {
        &self.transferable_kv_cache_memory
    }

    fn forward_batch(
        &mut self,
        batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, CudaDenseDecoderError> {
        validate_model_worker_batch(batch)
            .map_err(|error| CudaDenseDecoderError::Shape(error.to_string()))?;
        let row_count = batch.input_ids().len();
        if row_count == 0 {
            return Err(CudaDenseDecoderError::Shape(
                "forward batch contains no input tokens".to_string(),
            ));
        }
        let metadata =
            CudaPagedAttentionMetadata::from_model_worker_batch(batch, self.kv_cache.layout())?;
        let device_metadata = metadata.upload(self.backend.context())?;
        let token_ids = upload_u32(self.backend.context(), batch.input_ids())?;
        let positions =
            upload_usize_as_u64(self.backend.context(), "positions", batch.positions())?;
        let output_slots = batch
            .out_cache_pages()
            .iter()
            .map(|slot| slot.as_usize())
            .collect::<Vec<_>>();
        let output_slot_map = self.kv_cache.upload_slot_map(&output_slots)?;

        let mut hidden = allocate_bf16(
            self.backend.context(),
            checked_product(row_count, self.shape.hidden_size, "embedding output")?,
        )?;
        self.kernels.embedding_lookup(
            &token_ids,
            &self.token_embeddings.allocation,
            &mut hidden,
            row_count,
            self.plan.vocab_size,
            self.shape.hidden_size,
        )?;

        for layer_index in 0..self.layers.len() {
            hidden = self.forward_layer(
                layer_index,
                row_count,
                hidden,
                &positions,
                &output_slot_map,
                &device_metadata,
            )?;
        }

        let normalized =
            self.rms_norm(&hidden, &self.final_norm, row_count, self.shape.hidden_size)?;
        let last_rows = batch
            .request_offsets()
            .iter()
            .zip(batch.input_token_counts())
            .map(|(offset, count)| {
                offset
                    .checked_add(*count)
                    .and_then(|value| value.checked_sub(1))
                    .ok_or_else(|| {
                        CudaDenseDecoderError::Shape(
                            "request has no input row for logits collection".to_string(),
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let last_row_indices =
            upload_usize_as_u64(self.backend.context(), "last row indices", &last_rows)?;
        let mut last_hidden = allocate_bf16(
            self.backend.context(),
            checked_product(last_rows.len(), self.shape.hidden_size, "last hidden rows")?,
        )?;
        self.kernels.gather_rows(
            &normalized,
            &last_row_indices,
            &mut last_hidden,
            row_count,
            last_rows.len(),
            self.shape.hidden_size,
        )?;
        let lm_head = self.lm_head.as_ref().unwrap_or(&self.token_embeddings);
        let logits = linear(
            &self.blas,
            self.backend.context(),
            &last_hidden,
            last_rows.len(),
            self.shape.hidden_size,
            lm_head,
        )?;
        let logits = download_bf16(
            &logits,
            checked_product(last_rows.len(), self.plan.vocab_size, "logits")?,
        )?;
        let rows = logits
            .chunks_exact(self.plan.vocab_size)
            .map(<[f32]>::to_vec)
            .collect::<Vec<_>>();
        ModelForwardOutput::new(rows)
            .map_err(|error| CudaDenseDecoderError::Shape(error.to_string()))
    }

    fn forward_layer(
        &mut self,
        layer_index: usize,
        row_count: usize,
        hidden: CudaDeviceAllocation,
        positions: &CudaDeviceAllocation,
        output_slot_map: &crate::cuda_kv_cache::CudaKvCacheSlotMap,
        metadata: &crate::cuda_attention::CudaPagedAttentionDeviceMetadata,
    ) -> Result<CudaDeviceAllocation, CudaDenseDecoderError> {
        let layer = &self.layers[layer_index];
        let normalized = rms_norm(
            &self.kernels,
            self.backend.context(),
            &hidden,
            &layer.input_norm,
            row_count,
            self.shape.hidden_size,
            self.plan.rms_norm_eps,
        )?;
        let mut query = linear(
            &self.blas,
            self.backend.context(),
            &normalized,
            row_count,
            self.shape.hidden_size,
            &layer.query,
        )?;
        add_optional_bias(
            &self.kernels,
            &mut query,
            layer.query_bias.as_ref(),
            row_count,
            self.shape.query_size,
        )?;
        let mut key = linear(
            &self.blas,
            self.backend.context(),
            &normalized,
            row_count,
            self.shape.hidden_size,
            &layer.key,
        )?;
        add_optional_bias(
            &self.kernels,
            &mut key,
            layer.key_bias.as_ref(),
            row_count,
            self.shape.kv_size,
        )?;
        let mut value = linear(
            &self.blas,
            self.backend.context(),
            &normalized,
            row_count,
            self.shape.hidden_size,
            &layer.value,
        )?;
        add_optional_bias(
            &self.kernels,
            &mut value,
            layer.value_bias.as_ref(),
            row_count,
            self.shape.kv_size,
        )?;

        if let Some(weight) = &layer.query_norm {
            query = rms_norm(
                &self.kernels,
                self.backend.context(),
                &query,
                weight,
                checked_product(row_count, self.shape.query_head_count, "query norm rows")?,
                self.shape.head_dim,
                self.plan.rms_norm_eps,
            )?;
        }
        if let Some(weight) = &layer.key_norm {
            key = rms_norm(
                &self.kernels,
                self.backend.context(),
                &key,
                weight,
                checked_product(row_count, self.shape.kv_head_count, "key norm rows")?,
                self.shape.head_dim,
                self.plan.rms_norm_eps,
            )?;
        }
        self.kernels.neox_rope(
            &mut query,
            positions,
            row_count,
            self.shape.query_head_count,
            self.shape.head_dim,
            self.plan.rope_theta,
        )?;
        self.kernels.neox_rope(
            &mut key,
            positions,
            row_count,
            self.shape.kv_head_count,
            self.shape.head_dim,
            self.plan.rope_theta,
        )?;
        let kv_row_bytes = checked_product(self.shape.kv_size, BF16_BYTES, "KV row bytes")?;
        self.kv_cache
            .write_kv_slots_from_device(CudaKvSlotScatterLaunch {
                kernels: &mut self.kv_copy,
                layer_index,
                slot_map: output_slot_map,
                keys: &key,
                keys_offset: 0,
                key_row_stride_bytes: kv_row_bytes,
                values: &value,
                values_offset: 0,
                value_row_stride_bytes: kv_row_bytes,
            })?;

        let mut attention_output = allocate_bf16(
            self.backend.context(),
            checked_product(row_count, self.shape.query_size, "attention output")?,
        )?;
        self.attention.forward(CudaPagedAttentionForward {
            kv_cache: &self.kv_cache,
            layer_index,
            metadata,
            queries: &query,
            queries_offset: 0,
            query_head_count: self.shape.query_head_count,
            scale: (self.shape.head_dim as f32).sqrt().recip(),
            output: &mut attention_output,
            output_offset: 0,
        })?;
        let mut projected_attention = linear(
            &self.blas,
            self.backend.context(),
            &attention_output,
            row_count,
            self.shape.query_size,
            &layer.output,
        )?;
        add_optional_bias(
            &self.kernels,
            &mut projected_attention,
            layer.output_bias.as_ref(),
            row_count,
            self.shape.hidden_size,
        )?;
        let after_attention = add(
            &self.kernels,
            self.backend.context(),
            &hidden,
            &projected_attention,
            checked_product(row_count, self.shape.hidden_size, "attention residual")?,
        )?;
        let normalized = rms_norm(
            &self.kernels,
            self.backend.context(),
            &after_attention,
            &layer.post_attention_norm,
            row_count,
            self.shape.hidden_size,
            self.plan.rms_norm_eps,
        )?;
        let gate = linear(
            &self.blas,
            self.backend.context(),
            &normalized,
            row_count,
            self.shape.hidden_size,
            &layer.gate,
        )?;
        let up = linear(
            &self.blas,
            self.backend.context(),
            &normalized,
            row_count,
            self.shape.hidden_size,
            &layer.up,
        )?;
        let intermediate_elements = checked_product(
            row_count,
            self.shape.intermediate_size,
            "feed-forward intermediate",
        )?;
        let mut activated = allocate_bf16(self.backend.context(), intermediate_elements)?;
        self.kernels
            .silu_mul(&gate, &up, &mut activated, intermediate_elements)?;
        let feed_forward = linear(
            &self.blas,
            self.backend.context(),
            &activated,
            row_count,
            self.shape.intermediate_size,
            &layer.down,
        )?;
        add(
            &self.kernels,
            self.backend.context(),
            &after_attention,
            &feed_forward,
            checked_product(row_count, self.shape.hidden_size, "feed-forward residual")?,
        )
    }

    fn rms_norm(
        &self,
        input: &CudaDeviceAllocation,
        weight: &CudaDeviceAllocation,
        rows: usize,
        width: usize,
    ) -> Result<CudaDeviceAllocation, CudaDenseDecoderError> {
        rms_norm(
            &self.kernels,
            self.backend.context(),
            input,
            weight,
            rows,
            width,
            self.plan.rms_norm_eps,
        )
    }
}

impl ForwardModel for CudaBf16DenseDecoder {
    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        self.forward_batch(batch)
            .map_err(|error| ModelForwardError::Runtime(error.to_string()))
    }
}

#[derive(Clone, Copy, Debug)]
struct CudaDenseDecoderShape {
    hidden_size: usize,
    intermediate_size: usize,
    query_head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    query_size: usize,
    kv_size: usize,
}

impl CudaDenseDecoderShape {
    fn from_definition(
        definition: &ModelDefinition,
        plan: &DenseDecoderExecutionPlan,
    ) -> Result<Self, CudaDenseDecoderError> {
        let ModelExecutionArchitecture::Transformer {
            attention:
                AttentionArchitecture::MultiHead {
                    num_attention_heads,
                    num_key_value_heads,
                    head_dim,
                },
            feed_forward: FeedForwardArchitecture::Dense { intermediate_size },
        } = definition.execution()
        else {
            return Err(CudaDenseDecoderError::Unsupported(
                "shared CUDA dense decoder requires multi-head attention and dense feed-forward"
                    .to_string(),
            ));
        };
        let query_size = checked_product(num_attention_heads, head_dim, "query width")?;
        let kv_size = checked_product(num_key_value_heads, head_dim, "KV width")?;
        Ok(Self {
            hidden_size: plan.hidden_size,
            intermediate_size,
            query_head_count: num_attention_heads,
            kv_head_count: num_key_value_heads,
            head_dim,
            query_size,
            kv_size,
        })
    }
}

struct CudaDenseDecoderLayer {
    input_norm: CudaDeviceAllocation,
    query: CudaBf16Matrix,
    query_bias: Option<CudaDeviceAllocation>,
    query_norm: Option<CudaDeviceAllocation>,
    key: CudaBf16Matrix,
    key_bias: Option<CudaDeviceAllocation>,
    key_norm: Option<CudaDeviceAllocation>,
    value: CudaBf16Matrix,
    value_bias: Option<CudaDeviceAllocation>,
    output: CudaBf16Matrix,
    output_bias: Option<CudaDeviceAllocation>,
    post_attention_norm: CudaDeviceAllocation,
    gate: CudaBf16Matrix,
    up: CudaBf16Matrix,
    down: CudaBf16Matrix,
}

impl CudaDenseDecoderLayer {
    fn load(
        artifacts: &LocalModelArtifacts,
        context: &CudaContext,
        names: &DenseDecoderLayerWeightNames,
        shape: CudaDenseDecoderShape,
    ) -> Result<Self, CudaDenseDecoderError> {
        Ok(Self {
            input_norm: upload_required_bf16(
                artifacts,
                context,
                &names.input_norm,
                shape.hidden_size,
            )?,
            query: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.query_weight,
                shape.query_size,
                shape.hidden_size,
            )?,
            query_bias: upload_optional_bf16(
                artifacts,
                context,
                names.query_bias.as_deref(),
                shape.query_size,
            )?,
            query_norm: upload_optional_bf16(
                artifacts,
                context,
                names.query_norm.as_deref(),
                shape.head_dim,
            )?,
            key: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.key_weight,
                shape.kv_size,
                shape.hidden_size,
            )?,
            key_bias: upload_optional_bf16(
                artifacts,
                context,
                names.key_bias.as_deref(),
                shape.kv_size,
            )?,
            key_norm: upload_optional_bf16(
                artifacts,
                context,
                names.key_norm.as_deref(),
                shape.head_dim,
            )?,
            value: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.value_weight,
                shape.kv_size,
                shape.hidden_size,
            )?,
            value_bias: upload_optional_bf16(
                artifacts,
                context,
                names.value_bias.as_deref(),
                shape.kv_size,
            )?,
            output: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.output_weight,
                shape.hidden_size,
                shape.query_size,
            )?,
            output_bias: upload_optional_bf16(
                artifacts,
                context,
                names.output_bias.as_deref(),
                shape.hidden_size,
            )?,
            post_attention_norm: upload_required_bf16(
                artifacts,
                context,
                &names.post_attention_norm,
                shape.hidden_size,
            )?,
            gate: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.gate_weight,
                shape.intermediate_size,
                shape.hidden_size,
            )?,
            up: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.up_weight,
                shape.intermediate_size,
                shape.hidden_size,
            )?,
            down: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.down_weight,
                shape.hidden_size,
                shape.intermediate_size,
            )?,
        })
    }
}

struct CudaBf16Matrix {
    allocation: CudaDeviceAllocation,
    rows: usize,
    columns: usize,
}

impl CudaBf16Matrix {
    fn load(
        artifacts: &LocalModelArtifacts,
        context: &CudaContext,
        name: &str,
        rows: usize,
        columns: usize,
    ) -> Result<Self, CudaDenseDecoderError> {
        let element_count = checked_product(rows, columns, name)?;
        Ok(Self {
            allocation: upload_required_bf16(artifacts, context, name, element_count)?,
            rows,
            columns,
        })
    }
}

fn validate_cache_config(config: KvCacheAllocationConfig) -> Result<(), CudaDenseDecoderError> {
    if config.slot_capacity == 0 {
        return Err(CudaDenseDecoderError::Shape(
            "KV cache slot capacity must be non-zero".to_string(),
        ));
    }
    if config.page_size == 0 {
        return Err(CudaDenseDecoderError::Shape(
            "KV cache page size must be non-zero".to_string(),
        ));
    }
    if !config.slot_capacity.is_multiple_of(config.page_size) {
        return Err(CudaDenseDecoderError::Shape(format!(
            "KV cache slot capacity {} must be divisible by page size {}",
            config.slot_capacity, config.page_size
        )));
    }
    Ok(())
}

fn linear(
    blas: &CudaBlas,
    context: &CudaContext,
    input: &CudaDeviceAllocation,
    rows: usize,
    input_columns: usize,
    weight: &CudaBf16Matrix,
) -> Result<CudaDeviceAllocation, CudaDenseDecoderError> {
    if weight.columns != input_columns {
        return Err(CudaDenseDecoderError::Shape(format!(
            "linear input width {input_columns} does not match weight width {}",
            weight.columns
        )));
    }
    let mut output = allocate_bf16(
        context,
        checked_product(rows, weight.rows, "linear output")?,
    )?;
    blas.bf16_gemm_row_major(
        input,
        rows,
        input_columns,
        &weight.allocation,
        weight.rows,
        &mut output,
    )?;
    Ok(output)
}

fn rms_norm(
    kernels: &CudaBf16DenseKernels,
    context: &CudaContext,
    input: &CudaDeviceAllocation,
    weight: &CudaDeviceAllocation,
    rows: usize,
    width: usize,
    epsilon: f32,
) -> Result<CudaDeviceAllocation, CudaDenseDecoderError> {
    let mut output = allocate_bf16(context, checked_product(rows, width, "RMSNorm output")?)?;
    kernels.rms_norm(input, weight, &mut output, rows, width, epsilon)?;
    Ok(output)
}

fn add_optional_bias(
    kernels: &CudaBf16DenseKernels,
    values: &mut CudaDeviceAllocation,
    bias: Option<&CudaDeviceAllocation>,
    rows: usize,
    width: usize,
) -> Result<(), CudaDenseDecoderError> {
    if let Some(bias) = bias {
        kernels.add_bias(values, bias, rows, width)?;
    }
    Ok(())
}

fn add(
    kernels: &CudaBf16DenseKernels,
    context: &CudaContext,
    left: &CudaDeviceAllocation,
    right: &CudaDeviceAllocation,
    element_count: usize,
) -> Result<CudaDeviceAllocation, CudaDenseDecoderError> {
    let mut output = allocate_bf16(context, element_count)?;
    kernels.add(left, right, &mut output, element_count)?;
    Ok(output)
}

fn allocate_bf16(
    context: &CudaContext,
    element_count: usize,
) -> Result<CudaDeviceAllocation, CudaDenseDecoderError> {
    let byte_len = checked_product(element_count, BF16_BYTES, "BF16 allocation")?;
    Ok(context.allocate(byte_len)?)
}

fn upload_required_bf16(
    artifacts: &LocalModelArtifacts,
    context: &CudaContext,
    name: &str,
    expected_elements: usize,
) -> Result<CudaDeviceAllocation, CudaDenseDecoderError> {
    let tensor = artifacts
        .safetensors()
        .read_tensor(name)?
        .ok_or_else(|| CudaDenseDecoderError::MissingTensor(name.to_string()))?;
    if !matches!(tensor.metadata.dtype.as_str(), "F32" | "F16" | "BF16") {
        return Err(CudaDenseDecoderError::Unsupported(format!(
            "tensor {name} uses checkpoint dtype {}; the BF16 executor currently accepts unquantized F32, F16, or BF16 weights and does not apply quantized weight scales",
            tensor.metadata.dtype
        )));
    }
    if tensor.element_count() != expected_elements {
        return Err(CudaDenseDecoderError::Shape(format!(
            "tensor {name} has {} elements, expected {expected_elements}",
            tensor.element_count()
        )));
    }
    let values = tensor.decode_f32_values()?;
    let bytes = f32_values_to_bf16_bytes(&values);
    let mut allocation = context.allocate(bytes.len())?;
    allocation.copy_from_host(0, &bytes)?;
    Ok(allocation)
}

fn upload_optional_bf16(
    artifacts: &LocalModelArtifacts,
    context: &CudaContext,
    name: Option<&str>,
    expected_elements: usize,
) -> Result<Option<CudaDeviceAllocation>, CudaDenseDecoderError> {
    name.map(|name| upload_required_bf16(artifacts, context, name, expected_elements))
        .transpose()
}

fn upload_u32(
    context: &CudaContext,
    values: &[u32],
) -> Result<CudaDeviceAllocation, CudaDenseDecoderError> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    upload_bytes(context, &bytes)
}

fn upload_usize_as_u64(
    context: &CudaContext,
    field: &'static str,
    values: &[usize],
) -> Result<CudaDeviceAllocation, CudaDenseDecoderError> {
    let bytes = values
        .iter()
        .map(|value| {
            u64::try_from(*value).map_err(|_| {
                CudaDenseDecoderError::Shape(format!(
                    "{field} value {value} cannot be represented as u64"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flat_map(u64::to_ne_bytes)
        .collect::<Vec<_>>();
    upload_bytes(context, &bytes)
}

fn upload_bytes(
    context: &CudaContext,
    bytes: &[u8],
) -> Result<CudaDeviceAllocation, CudaDenseDecoderError> {
    let mut allocation = context.allocate(bytes.len())?;
    allocation.copy_from_host(0, bytes)?;
    Ok(allocation)
}

fn download_bf16(
    allocation: &CudaDeviceAllocation,
    element_count: usize,
) -> Result<Vec<f32>, CudaDenseDecoderError> {
    let mut bytes = vec![0_u8; checked_product(element_count, BF16_BYTES, "BF16 download")?];
    allocation.copy_to_host(0, &mut bytes)?;
    Ok(bytes
        .chunks_exact(BF16_BYTES)
        .map(|chunk| {
            let bits = u16::from_ne_bytes([chunk[0], chunk[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect())
}

fn f32_values_to_bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| {
            let bits = value.to_bits();
            let rounding_bias = 0x7fff + ((bits >> 16) & 1);
            ((bits.wrapping_add(rounding_bias) >> 16) as u16).to_ne_bytes()
        })
        .collect()
}

fn checked_product(
    left: usize,
    right: usize,
    name: impl fmt::Display,
) -> Result<usize, CudaDenseDecoderError> {
    left.checked_mul(right)
        .ok_or_else(|| CudaDenseDecoderError::Shape(format!("{name} size overflowed")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_conversion_rounds_to_nearest_even() {
        let values = [0.0_f32, 1.0, -2.5, f32::INFINITY];
        let bytes = f32_values_to_bf16_bytes(&values);
        let decoded = bytes
            .chunks_exact(2)
            .map(|chunk| f32::from_bits((u16::from_ne_bytes([chunk[0], chunk[1]]) as u32) << 16))
            .collect::<Vec<_>>();
        assert_eq!(decoded, values);
    }

    #[test]
    fn cache_config_requires_scheduler_aligned_capacity() {
        assert!(
            validate_cache_config(KvCacheAllocationConfig {
                slot_capacity: 512,
                page_size: 16,
            })
            .is_ok()
        );
        assert!(
            validate_cache_config(KvCacheAllocationConfig {
                slot_capacity: 511,
                page_size: 16,
            })
            .is_err()
        );
    }
}
