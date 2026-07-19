use std::fmt;

use sglang_kernel::cublas::CudaBlas;
use sglang_kernel::cuda::{CudaContext, CudaDeviceAllocation};
use sglang_kernel::cuda_bf16_kernels::CudaBf16DenseKernels;
use sglang_kernel::cuda_kv_kernels::CudaKvPairCopyKernels;
use sglang_kernel::nccl::NcclCommunicator;

use crate::backend::{CapabilityStatus, CudaBackend, RuntimeCapability, RuntimeDtype};
use crate::cuda_attention::{
    CudaBf16PagedAttentionExecutor, CudaPagedAttentionForward, CudaPagedAttentionMetadata,
};
use crate::cuda_execution_resources::CudaExecutionResources;
use crate::cuda_kv_cache::{CudaKvSlotScatterLaunch, CudaKvStorage};
use crate::cuda_transformer::{
    CudaBf16DenseFeedForward, CudaBf16Matrix, CudaExecutorError, add, add_optional_bias,
    allocate_bf16, checked_product, download_bf16, linear, rms_norm, upload_optional_bf16,
    upload_optional_bf16_partition, upload_required_bf16, upload_u32, upload_usize_as_u64,
};
use crate::kv_cache::PagedKvCacheLayout;
use crate::model_artifacts::LocalModelArtifacts;
use crate::model_executor::{
    ModelForwardError, ModelForwardOutput, ModelWorkerBatch, validate_model_worker_batch,
};
use crate::model_runtime::BackendModelExecutor;
use crate::models::{
    AttentionArchitecture, DenseDecoderExecutionPlan, DenseDecoderLayerWeightNames,
    DenseFeedForwardWeightNames, FeedForwardArchitecture, ModelDefinition,
    ModelExecutionArchitecture,
};
use crate::parallel::{TensorParallelRank, TensorParallelTopology};
use crate::transformer_parallel::DenseTensorParallelPlan;

type CudaDenseDecoderError = CudaExecutorError;

pub(crate) struct CudaBf16DenseDecoder {
    backend: CudaBackend,
    plan: DenseDecoderExecutionPlan,
    shape: CudaDenseDecoderShape,
    blas: CudaBlas,
    kernels: CudaBf16DenseKernels,
    attention: CudaBf16PagedAttentionExecutor,
    kv_copy: CudaKvPairCopyKernels,
    token_embeddings: CudaBf16Matrix,
    final_norm: CudaDeviceAllocation,
    lm_head: Option<CudaBf16Matrix>,
    layers: Vec<CudaDenseDecoderLayer>,
    rank: TensorParallelRank,
    communicator: Option<NcclCommunicator>,
}

impl fmt::Debug for CudaBf16DenseDecoder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaBf16DenseDecoder")
            .field("device", self.backend.device())
            .field("rank", &self.rank)
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

impl CudaBf16DenseDecoder {
    pub(crate) fn load(
        definition: &ModelDefinition,
        artifacts: &LocalModelArtifacts,
        backend: CudaBackend,
    ) -> Result<Self, CudaDenseDecoderError> {
        let topology = TensorParallelTopology::new(1, 1, 0, backend.device().ordinal, 1)
            .map_err(|error| CudaDenseDecoderError::Shape(error.to_string()))?;
        Self::load_tensor_parallel(
            definition,
            artifacts,
            backend,
            topology.local_ranks()[0],
            None,
        )
    }

    pub(crate) fn load_tensor_parallel(
        definition: &ModelDefinition,
        artifacts: &LocalModelArtifacts,
        backend: CudaBackend,
        rank: TensorParallelRank,
        communicator: Option<NcclCommunicator>,
    ) -> Result<Self, CudaDenseDecoderError> {
        if backend.device().ordinal != rank.device_ordinal() {
            return Err(CudaDenseDecoderError::Shape(format!(
                "tensor parallel rank {} is assigned CUDA device {}, but backend initialized device {}",
                rank.global_rank(),
                rank.device_ordinal(),
                backend.device().ordinal
            )));
        }
        if rank.world_size() > 1 && communicator.is_none() {
            return Err(CudaDenseDecoderError::Unsupported(format!(
                "tensor parallel rank {} / {} requires an initialized collective backend",
                rank.global_rank(),
                rank.world_size()
            )));
        }
        let plan = definition.dense_decoder().cloned().ok_or_else(|| {
            CudaDenseDecoderError::Unsupported(
                "model definition has no shared dense decoder execution plan".to_string(),
            )
        })?;
        let shape = CudaDenseDecoderShape::from_definition(definition, &plan, rank)?;
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
            .map(|names| {
                CudaDenseDecoderLayer::load_tensor_parallel(
                    artifacts,
                    backend.context(),
                    names,
                    shape,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            backend,
            plan,
            shape,
            blas,
            kernels,
            attention,
            kv_copy,
            token_embeddings,
            final_norm,
            lm_head,
            layers,
            rank,
            communicator,
        })
    }

    pub(crate) fn runtime_capability(&self) -> RuntimeCapability {
        let mut capability = self.backend.capabilities();
        capability.runtime_name = "cuda-bf16-dense-decoder";
        capability.supports_forward = true;
        capability.supported_dtypes = vec![RuntimeDtype::Bf16];
        capability.attention_backends = vec!["cuda-paged-bf16"];
        capability.tensor_parallel = CapabilityStatus::Supported;
        capability
    }

    pub(crate) fn local_kv_head_count(&self) -> usize {
        self.shape.kv_head_count
    }

    pub(crate) fn shutdown_collective(&mut self) {
        self.communicator.take();
    }

    fn forward_batch(
        &mut self,
        batch: &ModelWorkerBatch,
        kv_layout: PagedKvCacheLayout,
        kv_storage: &mut CudaKvStorage,
    ) -> Result<ModelForwardOutput, CudaDenseDecoderError> {
        validate_model_worker_batch(batch)
            .map_err(|error| CudaDenseDecoderError::Shape(error.to_string()))?;
        let row_count = batch.input_ids().len();
        if row_count == 0 {
            return Err(CudaDenseDecoderError::Shape(
                "forward batch contains no input tokens".to_string(),
            ));
        }
        let metadata = CudaPagedAttentionMetadata::from_model_worker_batch(batch, kv_layout)?;
        let device_metadata = metadata.upload(self.backend.context())?;
        let token_ids = upload_u32(self.backend.context(), batch.input_ids())?;
        let positions =
            upload_usize_as_u64(self.backend.context(), "positions", batch.positions())?;
        let output_slots = batch
            .out_cache_pages()
            .iter()
            .map(|slot| slot.as_usize())
            .collect::<Vec<_>>();
        let output_slot_map = kv_storage.upload_slot_map(kv_layout, &output_slots)?;

        let mut hidden = allocate_bf16(
            self.backend.context(),
            checked_product(row_count, self.shape.hidden_size, "embedding output")?,
        )?;
        self.kernels.embedding_lookup(
            &token_ids,
            self.token_embeddings.allocation(),
            &mut hidden,
            row_count,
            self.plan.vocab_size,
            self.shape.hidden_size,
        )?;

        for layer_index in 0..self.layers.len() {
            hidden = self.forward_layer(CudaDenseLayerForward {
                layer_index,
                row_count,
                hidden,
                positions: &positions,
                output_slot_map: &output_slot_map,
                metadata: &device_metadata,
                kv_layout,
                kv_storage,
            })?;
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
        launch: CudaDenseLayerForward<'_>,
    ) -> Result<CudaDeviceAllocation, CudaDenseDecoderError> {
        let CudaDenseLayerForward {
            layer_index,
            row_count,
            hidden,
            positions,
            output_slot_map,
            metadata,
            kv_layout,
            kv_storage,
        } = launch;
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
        let kv_row_bytes = checked_product(self.shape.kv_size, 2, "KV row bytes")?;
        kv_storage.write_kv_slots_from_device(
            kv_layout,
            CudaKvSlotScatterLaunch {
                kernels: &mut self.kv_copy,
                layer_index,
                slot_map: output_slot_map,
                keys: &key,
                keys_offset: 0,
                key_row_stride_bytes: kv_row_bytes,
                values: &value,
                values_offset: 0,
                value_row_stride_bytes: kv_row_bytes,
            },
        )?;

        let mut attention_output = allocate_bf16(
            self.backend.context(),
            checked_product(row_count, self.shape.query_size, "attention output")?,
        )?;
        self.attention.forward(CudaPagedAttentionForward {
            kv_layout,
            kv_storage,
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
        self.all_reduce_if_needed(
            &mut projected_attention,
            checked_product(row_count, self.shape.hidden_size, "attention projection")?,
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
        let mut feed_forward = layer.feed_forward.forward(
            &self.blas,
            &self.kernels,
            self.backend.context(),
            &normalized,
            row_count,
            self.shape.hidden_size,
        )?;
        self.all_reduce_if_needed(
            &mut feed_forward,
            checked_product(row_count, self.shape.hidden_size, "feed-forward projection")?,
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

    fn all_reduce_if_needed(
        &self,
        allocation: &mut CudaDeviceAllocation,
        element_count: usize,
    ) -> Result<(), CudaDenseDecoderError> {
        if self.rank.world_size() == 1 {
            return Ok(());
        }
        self.communicator
            .as_ref()
            .ok_or_else(|| {
                CudaDenseDecoderError::Execution(format!(
                    "tensor parallel rank {} lost its collective backend",
                    self.rank.global_rank()
                ))
            })?
            .all_reduce_bf16_sum_in_place(allocation, element_count)?;
        Ok(())
    }
}

struct CudaDenseLayerForward<'a> {
    layer_index: usize,
    row_count: usize,
    hidden: CudaDeviceAllocation,
    positions: &'a CudaDeviceAllocation,
    output_slot_map: &'a crate::cuda_kv_cache::CudaKvCacheSlotMap,
    metadata: &'a crate::cuda_attention::CudaPagedAttentionDeviceMetadata,
    kv_layout: PagedKvCacheLayout,
    kv_storage: &'a mut CudaKvStorage,
}

impl BackendModelExecutor<CudaExecutionResources> for CudaBf16DenseDecoder {
    fn runtime_capability(&self) -> RuntimeCapability {
        CudaBf16DenseDecoder::runtime_capability(self)
    }

    fn execution_dtype(&self) -> RuntimeDtype {
        RuntimeDtype::Bf16
    }

    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
        resources: &mut CudaExecutionResources,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        let kv_pool = resources.active_kv_cache_mut();
        let kv_layout = kv_pool.layout();
        let kv_storage = kv_pool.storage_mut();
        self.forward_batch(batch, kv_layout, kv_storage)
            .map_err(|error| ModelForwardError::Runtime(error.to_string()))
    }
}

#[derive(Clone, Copy, Debug)]
struct CudaDenseDecoderShape {
    hidden_size: usize,
    global_intermediate_size: usize,
    query_head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    query_size: usize,
    kv_size: usize,
    global_query_size: usize,
    global_kv_size: usize,
    parallel: DenseTensorParallelPlan,
}

impl CudaDenseDecoderShape {
    fn from_definition(
        definition: &ModelDefinition,
        plan: &DenseDecoderExecutionPlan,
        rank: TensorParallelRank,
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
        let parallel = DenseTensorParallelPlan::from_execution(definition.execution(), rank)
            .map_err(|error| CudaDenseDecoderError::Shape(error.to_string()))?;
        let global_query_size = checked_product(num_attention_heads, head_dim, "query width")?;
        let global_kv_size = checked_product(num_key_value_heads, head_dim, "KV width")?;
        Ok(Self {
            hidden_size: plan.hidden_size,
            global_intermediate_size: intermediate_size,
            query_head_count: parallel.local_query_head_count(),
            kv_head_count: parallel.local_kv_head_count(),
            head_dim,
            query_size: parallel
                .local_query_size()
                .map_err(|error| CudaDenseDecoderError::Shape(error.to_string()))?,
            kv_size: parallel
                .local_kv_size()
                .map_err(|error| CudaDenseDecoderError::Shape(error.to_string()))?,
            global_query_size,
            global_kv_size,
            parallel,
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
    feed_forward: CudaBf16DenseFeedForward,
}

impl CudaDenseDecoderLayer {
    fn load_tensor_parallel(
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
            query: CudaBf16Matrix::load_axis_partition(
                artifacts,
                context,
                &names.query_weight,
                shape.global_query_size,
                shape.hidden_size,
                0,
                shape.parallel.query_partition(),
            )?,
            query_bias: upload_optional_bf16_partition(
                artifacts,
                context,
                names.query_bias.as_deref(),
                shape.global_query_size,
                shape.parallel.query_partition(),
            )?,
            query_norm: upload_optional_bf16(
                artifacts,
                context,
                names.query_norm.as_deref(),
                shape.head_dim,
            )?,
            key: CudaBf16Matrix::load_axis_partition(
                artifacts,
                context,
                &names.key_weight,
                shape.global_kv_size,
                shape.hidden_size,
                0,
                shape.parallel.kv_partition(),
            )?,
            key_bias: upload_optional_bf16_partition(
                artifacts,
                context,
                names.key_bias.as_deref(),
                shape.global_kv_size,
                shape.parallel.kv_partition(),
            )?,
            key_norm: upload_optional_bf16(
                artifacts,
                context,
                names.key_norm.as_deref(),
                shape.head_dim,
            )?,
            value: CudaBf16Matrix::load_axis_partition(
                artifacts,
                context,
                &names.value_weight,
                shape.global_kv_size,
                shape.hidden_size,
                0,
                shape.parallel.kv_partition(),
            )?,
            value_bias: upload_optional_bf16_partition(
                artifacts,
                context,
                names.value_bias.as_deref(),
                shape.global_kv_size,
                shape.parallel.kv_partition(),
            )?,
            output: CudaBf16Matrix::load_axis_partition(
                artifacts,
                context,
                &names.output_weight,
                shape.hidden_size,
                shape.global_query_size,
                1,
                shape.parallel.query_partition(),
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
            feed_forward: CudaBf16DenseFeedForward::load_tensor_parallel(
                artifacts,
                context,
                &DenseFeedForwardWeightNames {
                    gate_weight: names.gate_weight.clone(),
                    up_weight: names.up_weight.clone(),
                    down_weight: names.down_weight.clone(),
                },
                shape.hidden_size,
                shape.global_intermediate_size,
                shape.parallel,
            )?,
        })
    }
}
