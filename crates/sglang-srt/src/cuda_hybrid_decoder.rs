use std::collections::BTreeSet;
use std::fmt;

use sglang_kernel::cublas::CudaBlas;
use sglang_kernel::cuda::{CudaContext, CudaDeviceAllocation};
use sglang_kernel::cuda_bf16_kernels::CudaBf16DenseKernels;
use sglang_kernel::cuda_hybrid_kernels::{CudaBf16HybridKernels, CudaKdaDecayLaunch};
use sglang_kernel::cuda_linear_attention::{
    CudaBf16LinearAttentionKernels, CudaCausalConv1dSegmentLaunch, CudaCausalConv1dSegmentShape,
    CudaKeyGatedDeltaLaunch, CudaKeyGatedDeltaShape,
};
use sglang_kernel::cuda_mla::CudaBf16MlaKernels;

use crate::backend::{CapabilityStatus, CudaBackend, RuntimeCapability, RuntimeDtype};
use crate::cuda_attention::CudaBf16PagedAttentionExecutor;
use crate::cuda_execution_resources::CudaExecutionResources;
use crate::cuda_mla::{CudaBf16MultiLatentAttention, CudaMultiLatentAttentionForward};
use crate::cuda_moe::CudaBf16MixtureOfExperts;
use crate::cuda_recurrent_state::CudaRecurrentLayerState;
use crate::cuda_transformer::{
    CudaBf16DenseFeedForward, CudaBf16Matrix, CudaExecutorError, add, allocate_bf16, allocate_f32,
    checked_product, download_bf16, linear, rms_norm, upload_required_bf16, upload_required_f32,
    upload_u32,
};
use crate::model_artifacts::LocalModelArtifacts;
use crate::model_executor::{
    ModelForwardError, ModelForwardOutput, ModelWorkerBatch, validate_model_worker_batch,
};
use crate::model_runtime::BackendModelExecutor;
use crate::models::{
    DecoderNormalization, HybridDecoderExecutionPlan, HybridDecoderLayerKind,
    HybridDecoderLayerWeightNames, HybridFeedForward, HybridLinearAttentionConfig,
    KeyGatedDeltaWeightNames, ModelDefinition,
};

const BF16_BYTES: usize = 2;

pub(crate) struct CudaBf16HybridDecoder {
    backend: CudaBackend,
    plan: HybridDecoderExecutionPlan,
    blas: CudaBlas,
    dense_kernels: CudaBf16DenseKernels,
    hybrid_kernels: CudaBf16HybridKernels,
    linear_attention: CudaBf16LinearAttentionKernels,
    mla_kernels: Option<CudaBf16MlaKernels>,
    paged_attention: Option<CudaBf16PagedAttentionExecutor>,
    token_embeddings: CudaBf16Matrix,
    final_norm: CudaDeviceAllocation,
    lm_head: Option<CudaBf16Matrix>,
    layers: Vec<CudaHybridLayer>,
}

impl fmt::Debug for CudaBf16HybridDecoder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaBf16HybridDecoder")
            .field("device", self.backend.device())
            .field("layer_count", &self.layers.len())
            .finish_non_exhaustive()
    }
}

impl CudaBf16HybridDecoder {
    pub(crate) fn missing_components(definition: &ModelDefinition) -> Vec<String> {
        let Some(plan) = definition.hybrid_decoder() else {
            return vec!["shared hybrid decoder execution plan".to_string()];
        };
        let mut missing = BTreeSet::new();
        if plan.normalization != DecoderNormalization::Rms {
            missing.insert("CUDA hybrid Gemma RMS normalization".to_string());
        }
        for layer in &plan.weights.layers {
            match layer.mixer {
                HybridDecoderLayerKind::KeyGatedDelta { .. } => {
                    if !matches!(
                        plan.linear_attention,
                        HybridLinearAttentionConfig::KeyGatedDelta { .. }
                    ) {
                        missing.insert("CUDA KDA typed execution config".to_string());
                    }
                }
                HybridDecoderLayerKind::FullAttention { .. } => {
                    missing.insert("CUDA hybrid full-attention component".to_string());
                }
                HybridDecoderLayerKind::GatedDeltaNet { .. } => {
                    missing.insert("CUDA gated-delta-net component".to_string());
                }
                HybridDecoderLayerKind::MultiLatentAttention { .. } => {
                    if !matches!(
                        plan.full_attention,
                        crate::models::HybridFullAttentionConfig::MultiLatent { .. }
                    ) {
                        missing.insert("CUDA MLA typed execution config".to_string());
                    }
                }
            }
        }
        missing.into_iter().collect()
    }

    pub(crate) fn load(
        definition: &ModelDefinition,
        artifacts: &LocalModelArtifacts,
        backend: CudaBackend,
    ) -> Result<Self, CudaExecutorError> {
        let missing = Self::missing_components(definition);
        if !missing.is_empty() {
            return Err(CudaExecutorError::Unsupported(missing.join(", ")));
        }
        definition.validate_hybrid_decoder_checkpoint(artifacts)?;
        let plan = definition.hybrid_decoder().cloned().ok_or_else(|| {
            CudaExecutorError::Unsupported(
                "model definition has no shared hybrid decoder execution plan".to_string(),
            )
        })?;
        let compute_capability = backend.device().compute_capability;
        let blas = CudaBlas::load(backend.context())?;
        let dense_kernels = CudaBf16DenseKernels::compile(backend.context(), compute_capability)?;
        let hybrid_kernels = CudaBf16HybridKernels::compile(backend.context(), compute_capability)?;
        let linear_attention =
            CudaBf16LinearAttentionKernels::compile(backend.context(), compute_capability)?;
        let has_mla = plan.weights.layers.iter().any(|layer| {
            matches!(
                layer.mixer,
                HybridDecoderLayerKind::MultiLatentAttention { .. }
            )
        });
        let mla_kernels = has_mla
            .then(|| CudaBf16MlaKernels::compile(backend.context(), compute_capability))
            .transpose()?;
        let paged_attention = has_mla
            .then(|| CudaBf16PagedAttentionExecutor::compile(backend.context(), compute_capability))
            .transpose()?;
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
            .map(|names| CudaHybridLayer::load(artifacts, backend.context(), names, &plan))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            backend,
            plan,
            blas,
            dense_kernels,
            hybrid_kernels,
            linear_attention,
            mla_kernels,
            paged_attention,
            token_embeddings,
            final_norm,
            lm_head,
            layers,
        })
    }

    pub(crate) fn runtime_capability(&self) -> RuntimeCapability {
        let mut capability = self.backend.capabilities();
        capability.runtime_name = "cuda-bf16-hybrid-decoder";
        capability.supports_forward = true;
        capability.supported_dtypes = vec![RuntimeDtype::Bf16];
        capability.attention_backends = vec!["cuda-kda-bf16", "cuda-mla-bf16"];
        capability.tensor_parallel = CapabilityStatus::Unsupported;
        capability
    }

    fn forward_batch(
        &mut self,
        batch: &ModelWorkerBatch,
        resources: &mut CudaExecutionResources,
    ) -> Result<ModelForwardOutput, CudaExecutorError> {
        validate_model_worker_batch(batch)
            .map_err(|error| CudaExecutorError::Shape(error.to_string()))?;
        if batch.input_ids().is_empty() {
            return Err(CudaExecutorError::Shape(
                "hybrid forward batch contains no input tokens".to_string(),
            ));
        }
        if let Some((request_index, cached)) = batch
            .cached_token_counts()
            .iter()
            .copied()
            .enumerate()
            .find(|(_, cached)| *cached != 0)
        {
            return Err(CudaExecutorError::Unsupported(format!(
                "CUDA hybrid request {request_index} has {cached} cached prefix tokens, but recurrent prefix-state restoration is not implemented"
            )));
        }
        let (active_kv_cache, mut recurrent_state) = resources.execution_memory_mut();
        let mut logits = Vec::with_capacity(batch.request_ids().len());
        for request_index in 0..batch.request_ids().len() {
            let request_id = &batch.request_ids()[request_index];
            let offset = batch.request_offsets()[request_index];
            let token_count = batch.input_token_counts()[request_index];
            let sequence_offset = batch.sequence_offsets()[request_index];
            let sequence_count = batch.sequence_token_counts()[request_index];
            let sequence_slots =
                &batch.sequence_cache_pages()[sequence_offset..sequence_offset + sequence_count];
            for relative_index in 0..token_count {
                let row_index = offset + relative_index;
                let position = batch.positions()[row_index];
                if position >= self.plan.max_position_embeddings {
                    return Err(CudaExecutorError::Shape(format!(
                        "token position {position} exceeds max_position_embeddings {}",
                        self.plan.max_position_embeddings
                    )));
                }
                if let Some(state) = recurrent_state.as_deref_mut() {
                    state.prepare_batch(std::slice::from_ref(request_id))?;
                }
                let token_id = upload_u32(self.backend.context(), &[batch.input_ids()[row_index]])?;
                let mut hidden = allocate_bf16(self.backend.context(), self.plan.hidden_size)?;
                self.dense_kernels.embedding_lookup(
                    &token_id,
                    self.token_embeddings.allocation(),
                    &mut hidden,
                    1,
                    self.plan.vocab_size,
                    self.plan.hidden_size,
                )?;
                for layer in &self.layers {
                    let normalized = rms_norm(
                        &self.dense_kernels,
                        self.backend.context(),
                        &hidden,
                        &layer.input_norm,
                        1,
                        self.plan.hidden_size,
                        self.plan.rms_norm_eps,
                    )?;
                    let mixer = match &layer.mixer {
                        CudaHybridMixer::KeyGatedDelta {
                            state_layer_index,
                            component,
                        } => {
                            let state = recurrent_state.as_deref_mut().ok_or_else(|| {
                                CudaExecutorError::Unsupported(
                                    "CUDA KDA requires backend-owned recurrent state".to_string(),
                                )
                            })?;
                            let mut layer_state = state.layer_state_mut(*state_layer_index)?;
                            component.forward(CudaKeyGatedDeltaForward {
                                context: self.backend.context(),
                                blas: &self.blas,
                                dense_kernels: &self.dense_kernels,
                                hybrid_kernels: &self.hybrid_kernels,
                                linear_attention: &mut self.linear_attention,
                                hidden: &normalized,
                                batch_size: 1,
                                rms_norm_eps: self.plan.rms_norm_eps,
                                state: &mut layer_state,
                            })?
                        }
                        CudaHybridMixer::MultiLatentAttention {
                            cache_layer_index,
                            component,
                        } => {
                            let kernels = self.mla_kernels.as_ref().ok_or_else(|| {
                                CudaExecutorError::Unsupported(
                                    "CUDA MLA kernels were not initialized".to_string(),
                                )
                            })?;
                            let attention = self.paged_attention.as_mut().ok_or_else(|| {
                                CudaExecutorError::Unsupported(
                                    "CUDA MLA paged attention was not initialized".to_string(),
                                )
                            })?;
                            let kv_layout = active_kv_cache.layout();
                            component.forward(CudaMultiLatentAttentionForward {
                                context: self.backend.context(),
                                blas: &self.blas,
                                dense_kernels: &self.dense_kernels,
                                kernels,
                                attention,
                                hidden: &normalized,
                                position,
                                output_slot: batch.out_cache_pages()[row_index],
                                sequence_slots,
                                cache_layer_index: *cache_layer_index,
                                kv_layout,
                                kv_storage: active_kv_cache.storage_mut(),
                                rms_norm_epsilon: self.plan.rms_norm_eps,
                                rope_theta: self.plan.rope_theta,
                            })?
                        }
                    };
                    let after_mixer = add(
                        &self.dense_kernels,
                        self.backend.context(),
                        &hidden,
                        &mixer,
                        self.plan.hidden_size,
                    )?;
                    let normalized = rms_norm(
                        &self.dense_kernels,
                        self.backend.context(),
                        &after_mixer,
                        &layer.post_attention_norm,
                        1,
                        self.plan.hidden_size,
                        self.plan.rms_norm_eps,
                    )?;
                    let feed_forward = layer.feed_forward.forward_single(
                        &self.blas,
                        &self.dense_kernels,
                        self.backend.context(),
                        &normalized,
                        self.plan.hidden_size,
                    )?;
                    hidden = add(
                        &self.dense_kernels,
                        self.backend.context(),
                        &after_mixer,
                        &feed_forward,
                        self.plan.hidden_size,
                    )?;
                }
                if relative_index + 1 == token_count {
                    let normalized = rms_norm(
                        &self.dense_kernels,
                        self.backend.context(),
                        &hidden,
                        &self.final_norm,
                        1,
                        self.plan.hidden_size,
                        self.plan.rms_norm_eps,
                    )?;
                    let lm_head = self.lm_head.as_ref().unwrap_or(&self.token_embeddings);
                    let token_logits = linear(
                        &self.blas,
                        self.backend.context(),
                        &normalized,
                        1,
                        self.plan.hidden_size,
                        lm_head,
                    )?;
                    logits.push(download_bf16(&token_logits, self.plan.vocab_size)?);
                }
            }
        }
        ModelForwardOutput::new(logits).map_err(|error| CudaExecutorError::Shape(error.to_string()))
    }
}

impl BackendModelExecutor<CudaExecutionResources> for CudaBf16HybridDecoder {
    fn runtime_capability(&self) -> RuntimeCapability {
        CudaBf16HybridDecoder::runtime_capability(self)
    }

    fn execution_dtype(&self) -> RuntimeDtype {
        RuntimeDtype::Bf16
    }

    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
        resources: &mut CudaExecutionResources,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        self.forward_batch(batch, resources)
            .map_err(|error| ModelForwardError::Runtime(error.to_string()))
    }
}

struct CudaHybridLayer {
    input_norm: CudaDeviceAllocation,
    mixer: CudaHybridMixer,
    post_attention_norm: CudaDeviceAllocation,
    feed_forward: CudaHybridFeedForward,
}

impl CudaHybridLayer {
    fn load(
        artifacts: &LocalModelArtifacts,
        context: &CudaContext,
        names: &HybridDecoderLayerWeightNames,
        plan: &HybridDecoderExecutionPlan,
    ) -> Result<Self, CudaExecutorError> {
        let mixer = match &names.mixer {
            HybridDecoderLayerKind::KeyGatedDelta {
                state_layer_index,
                weights,
            } => CudaHybridMixer::KeyGatedDelta {
                state_layer_index: *state_layer_index,
                component: Box::new(CudaBf16KeyGatedDelta::load(
                    artifacts,
                    context,
                    weights,
                    plan.hidden_size,
                    plan.linear_attention,
                )?),
            },
            HybridDecoderLayerKind::MultiLatentAttention {
                cache_layer_index,
                weights,
            } => CudaHybridMixer::MultiLatentAttention {
                cache_layer_index: *cache_layer_index,
                component: Box::new(CudaBf16MultiLatentAttention::load(
                    artifacts,
                    context,
                    weights,
                    plan.hidden_size,
                    plan.full_attention,
                )?),
            },
            HybridDecoderLayerKind::FullAttention { .. }
            | HybridDecoderLayerKind::GatedDeltaNet { .. } => {
                return Err(CudaExecutorError::Unsupported(
                    "CUDA hybrid layer loader received an unimplemented mixer".to_string(),
                ));
            }
        };
        let feed_forward = match &names.feed_forward {
            HybridFeedForward::Dense {
                intermediate_size,
                weights,
            } => CudaHybridFeedForward::Dense(Box::new(CudaBf16DenseFeedForward::load(
                artifacts,
                context,
                weights,
                plan.hidden_size,
                *intermediate_size,
            )?)),
            HybridFeedForward::MixtureOfExperts { config, weights } => {
                CudaHybridFeedForward::MixtureOfExperts(Box::new(CudaBf16MixtureOfExperts::load(
                    artifacts,
                    context,
                    config,
                    weights,
                    plan.hidden_size,
                )?))
            }
        };
        Ok(Self {
            input_norm: upload_required_bf16(
                artifacts,
                context,
                &names.input_norm,
                plan.hidden_size,
            )?,
            mixer,
            post_attention_norm: upload_required_bf16(
                artifacts,
                context,
                &names.post_attention_norm,
                plan.hidden_size,
            )?,
            feed_forward,
        })
    }
}

enum CudaHybridMixer {
    KeyGatedDelta {
        state_layer_index: usize,
        component: Box<CudaBf16KeyGatedDelta>,
    },
    MultiLatentAttention {
        cache_layer_index: usize,
        component: Box<CudaBf16MultiLatentAttention>,
    },
}

enum CudaHybridFeedForward {
    Dense(Box<CudaBf16DenseFeedForward>),
    MixtureOfExperts(Box<CudaBf16MixtureOfExperts>),
}

impl CudaHybridFeedForward {
    fn forward_single(
        &self,
        blas: &CudaBlas,
        kernels: &CudaBf16DenseKernels,
        context: &CudaContext,
        hidden: &CudaDeviceAllocation,
        hidden_size: usize,
    ) -> Result<CudaDeviceAllocation, CudaExecutorError> {
        match self {
            Self::Dense(component) => {
                component.forward(blas, kernels, context, hidden, 1, hidden_size)
            }
            Self::MixtureOfExperts(component) => {
                component.forward_single(blas, kernels, context, hidden)
            }
        }
    }
}

pub(crate) struct CudaBf16KeyGatedDelta {
    shape: CudaKeyGatedDeltaComponentShape,
    query: CudaBf16Matrix,
    key: CudaBf16Matrix,
    value: CudaBf16Matrix,
    beta: CudaBf16Matrix,
    forget_a: CudaBf16Matrix,
    forget_b: CudaBf16Matrix,
    gate_a: CudaBf16Matrix,
    gate_b: CudaBf16Matrix,
    query_conv: CudaDeviceAllocation,
    key_conv: CudaDeviceAllocation,
    value_conv: CudaDeviceAllocation,
    a_log: CudaDeviceAllocation,
    dt_bias: CudaDeviceAllocation,
    output_norm: CudaDeviceAllocation,
    output: CudaBf16Matrix,
}

impl CudaBf16KeyGatedDelta {
    pub(crate) fn load(
        artifacts: &LocalModelArtifacts,
        context: &CudaContext,
        names: &KeyGatedDeltaWeightNames,
        hidden_size: usize,
        config: HybridLinearAttentionConfig,
    ) -> Result<Self, CudaExecutorError> {
        let shape = CudaKeyGatedDeltaComponentShape::new(hidden_size, config)?;
        Ok(Self {
            query: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.query_weight,
                shape.key_size,
                hidden_size,
            )?,
            key: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.key_weight,
                shape.key_size,
                hidden_size,
            )?,
            value: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.value_weight,
                shape.value_size,
                hidden_size,
            )?,
            beta: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.beta_weight,
                shape.head_count,
                hidden_size,
            )?,
            forget_a: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.forget_a_weight,
                shape.key_head_dim,
                hidden_size,
            )?,
            forget_b: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.forget_b_weight,
                shape.key_size,
                shape.key_head_dim,
            )?,
            gate_a: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.gate_a_weight,
                shape.key_head_dim,
                hidden_size,
            )?,
            gate_b: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.gate_b_weight,
                shape.value_size,
                shape.key_head_dim,
            )?,
            query_conv: upload_required_bf16(
                artifacts,
                context,
                &names.query_conv_weight,
                checked_product(shape.key_size, shape.conv_kernel_dim, "KDA query conv")?,
            )?,
            key_conv: upload_required_bf16(
                artifacts,
                context,
                &names.key_conv_weight,
                checked_product(shape.key_size, shape.conv_kernel_dim, "KDA key conv")?,
            )?,
            value_conv: upload_required_bf16(
                artifacts,
                context,
                &names.value_conv_weight,
                checked_product(shape.value_size, shape.conv_kernel_dim, "KDA value conv")?,
            )?,
            a_log: upload_required_f32(artifacts, context, &names.a_log, shape.head_count)?,
            dt_bias: upload_required_f32(artifacts, context, &names.dt_bias, shape.key_size)?,
            output_norm: upload_required_bf16(
                artifacts,
                context,
                &names.output_norm,
                shape.value_head_dim,
            )?,
            output: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.output_weight,
                hidden_size,
                shape.value_size,
            )?,
            shape,
        })
    }

    pub(crate) fn forward(
        &self,
        launch: CudaKeyGatedDeltaForward<'_>,
    ) -> Result<CudaDeviceAllocation, CudaExecutorError> {
        let CudaKeyGatedDeltaForward {
            context,
            blas,
            dense_kernels,
            hybrid_kernels,
            linear_attention,
            hidden,
            batch_size,
            rms_norm_eps,
            state,
        } = launch;
        if state.batch_size != batch_size {
            return Err(CudaExecutorError::Shape(format!(
                "KDA batch has {batch_size} rows but recurrent state prepared {} indices",
                state.batch_size
            )));
        }
        let expected_conv_channels = self.shape.conv_channels()?;
        let expected_conv_state_elements = checked_product(
            checked_product(
                state.state_slot_count,
                self.shape.conv_kernel_dim - 1,
                "KDA conv state slots",
            )?,
            expected_conv_channels,
            "KDA conv state",
        )?;
        let conv_state_bytes = checked_product(
            expected_conv_state_elements,
            BF16_BYTES,
            "KDA conv state bytes",
        )?;
        state
            .conv_state
            .device_ptr_at(state.conv_state_offset, conv_state_bytes)?;

        let query = linear(
            blas,
            context,
            hidden,
            batch_size,
            self.shape.hidden_size,
            &self.query,
        )?;
        let key = linear(
            blas,
            context,
            hidden,
            batch_size,
            self.shape.hidden_size,
            &self.key,
        )?;
        let value = linear(
            blas,
            context,
            hidden,
            batch_size,
            self.shape.hidden_size,
            &self.value,
        )?;
        let mut query_conv = allocate_bf16(
            context,
            checked_product(batch_size, self.shape.key_size, "KDA convolved query")?,
        )?;
        let mut key_conv = allocate_bf16(
            context,
            checked_product(batch_size, self.shape.key_size, "KDA convolved key")?,
        )?;
        let mut value_conv = allocate_bf16(
            context,
            checked_product(batch_size, self.shape.value_size, "KDA convolved value")?,
        )?;

        self.convolve_segment(CudaKdaConvSegment {
            kernels: linear_attention,
            input: &query,
            weight: &self.query_conv,
            output: &mut query_conv,
            state,
            batch_size,
            channels: self.shape.key_size,
            state_channel_offset: 0,
        })?;
        self.convolve_segment(CudaKdaConvSegment {
            kernels: linear_attention,
            input: &key,
            weight: &self.key_conv,
            output: &mut key_conv,
            state,
            batch_size,
            channels: self.shape.key_size,
            state_channel_offset: self.shape.key_size,
        })?;
        self.convolve_segment(CudaKdaConvSegment {
            kernels: linear_attention,
            input: &value,
            weight: &self.value_conv,
            output: &mut value_conv,
            state,
            batch_size,
            channels: self.shape.value_size,
            state_channel_offset: checked_product(self.shape.key_size, 2, "KDA value conv offset")?,
        })?;
        hybrid_kernels.silu_inplace(
            &mut query_conv,
            checked_product(batch_size, self.shape.key_size, "KDA query activation")?,
        )?;
        hybrid_kernels.silu_inplace(
            &mut key_conv,
            checked_product(batch_size, self.shape.key_size, "KDA key activation")?,
        )?;
        hybrid_kernels.silu_inplace(
            &mut value_conv,
            checked_product(batch_size, self.shape.value_size, "KDA value activation")?,
        )?;
        let head_rows = checked_product(batch_size, self.shape.head_count, "KDA head rows")?;
        hybrid_kernels.l2_normalize_heads_inplace(
            &mut query_conv,
            0,
            head_rows,
            self.shape.key_head_dim,
            (self.shape.key_head_dim as f32).sqrt().recip(),
            1e-6,
        )?;
        hybrid_kernels.l2_normalize_heads_inplace(
            &mut key_conv,
            0,
            head_rows,
            self.shape.key_head_dim,
            1.0,
            1e-6,
        )?;

        let forget_hidden = linear(
            blas,
            context,
            hidden,
            batch_size,
            self.shape.hidden_size,
            &self.forget_a,
        )?;
        let raw_forget = linear(
            blas,
            context,
            &forget_hidden,
            batch_size,
            self.shape.key_head_dim,
            &self.forget_b,
        )?;
        let key_elements = checked_product(batch_size, self.shape.key_size, "KDA decay")?;
        let mut decay = allocate_f32(context, key_elements)?;
        hybrid_kernels.kda_decay(CudaKdaDecayLaunch {
            raw_forget: &raw_forget,
            dt_bias: &self.dt_bias,
            a_log: &self.a_log,
            decay: &mut decay,
            batch_size,
            head_count: self.shape.head_count,
            key_head_dim: self.shape.key_head_dim,
        })?;
        let beta_raw = linear(
            blas,
            context,
            hidden,
            batch_size,
            self.shape.hidden_size,
            &self.beta,
        )?;
        let beta_elements = checked_product(batch_size, self.shape.head_count, "KDA beta")?;
        let mut beta = allocate_f32(context, beta_elements)?;
        hybrid_kernels.sigmoid_to_f32(&beta_raw, &mut beta, beta_elements)?;

        let value_elements = checked_product(batch_size, self.shape.value_size, "KDA output")?;
        let mut core = allocate_bf16(context, value_elements)?;
        linear_attention.key_gated_delta_decode(CudaKeyGatedDeltaLaunch {
            query: &query_conv,
            query_offset: 0,
            key: &key_conv,
            key_offset: 0,
            value: &value_conv,
            value_offset: 0,
            decay: &decay,
            decay_offset: 0,
            beta: &beta,
            beta_offset: 0,
            state: state.temporal_state,
            state_offset: state.temporal_state_offset,
            state_indices: state.state_indices,
            state_indices_offset: 0,
            output: &mut core,
            output_offset: 0,
            shape: CudaKeyGatedDeltaShape {
                batch_size,
                state_slot_count: state.state_slot_count,
                head_count: self.shape.head_count,
                key_head_dim: self.shape.key_head_dim,
                value_head_dim: self.shape.value_head_dim,
            },
        })?;
        let mut normalized = rms_norm(
            dense_kernels,
            context,
            &core,
            &self.output_norm,
            head_rows,
            self.shape.value_head_dim,
            rms_norm_eps,
        )?;
        let gate_hidden = linear(
            blas,
            context,
            hidden,
            batch_size,
            self.shape.hidden_size,
            &self.gate_a,
        )?;
        let gate = linear(
            blas,
            context,
            &gate_hidden,
            batch_size,
            self.shape.key_head_dim,
            &self.gate_b,
        )?;
        hybrid_kernels.sigmoid_mul_inplace(&mut normalized, &gate, value_elements)?;
        linear(
            blas,
            context,
            &normalized,
            batch_size,
            self.shape.value_size,
            &self.output,
        )
    }

    fn convolve_segment(
        &self,
        launch: CudaKdaConvSegment<'_, '_>,
    ) -> Result<(), CudaExecutorError> {
        let CudaKdaConvSegment {
            kernels,
            input,
            weight,
            output,
            state,
            batch_size,
            channels,
            state_channel_offset,
        } = launch;
        kernels.causal_conv1d_update_segment(CudaCausalConv1dSegmentLaunch {
            input,
            input_offset: 0,
            weight,
            weight_offset: 0,
            state: state.conv_state,
            state_offset: state.conv_state_offset,
            state_indices: state.state_indices,
            state_indices_offset: 0,
            output,
            output_offset: 0,
            shape: CudaCausalConv1dSegmentShape {
                batch_size,
                state_slot_count: state.state_slot_count,
                channels,
                kernel_size: self.shape.conv_kernel_dim,
                state_slot_channels: self.shape.conv_channels()?,
                state_channel_offset,
            },
        })?;
        Ok(())
    }
}

struct CudaKdaConvSegment<'a, 'state> {
    kernels: &'a mut CudaBf16LinearAttentionKernels,
    input: &'a CudaDeviceAllocation,
    weight: &'a CudaDeviceAllocation,
    output: &'a mut CudaDeviceAllocation,
    state: &'a mut CudaRecurrentLayerState<'state>,
    batch_size: usize,
    channels: usize,
    state_channel_offset: usize,
}

pub(crate) struct CudaKeyGatedDeltaForward<'a> {
    pub(crate) context: &'a CudaContext,
    pub(crate) blas: &'a CudaBlas,
    pub(crate) dense_kernels: &'a CudaBf16DenseKernels,
    pub(crate) hybrid_kernels: &'a CudaBf16HybridKernels,
    pub(crate) linear_attention: &'a mut CudaBf16LinearAttentionKernels,
    pub(crate) hidden: &'a CudaDeviceAllocation,
    pub(crate) batch_size: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) state: &'a mut CudaRecurrentLayerState<'a>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CudaKeyGatedDeltaComponentShape {
    hidden_size: usize,
    conv_kernel_dim: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    head_count: usize,
    key_size: usize,
    value_size: usize,
}

impl CudaKeyGatedDeltaComponentShape {
    fn new(
        hidden_size: usize,
        config: HybridLinearAttentionConfig,
    ) -> Result<Self, CudaExecutorError> {
        let HybridLinearAttentionConfig::KeyGatedDelta {
            conv_kernel_dim,
            key_head_dim,
            value_head_dim,
            num_heads,
        } = config
        else {
            return Err(CudaExecutorError::Unsupported(
                "CUDA KDA component requires a key-gated delta execution plan".to_string(),
            ));
        };
        if hidden_size == 0
            || conv_kernel_dim < 2
            || key_head_dim == 0
            || value_head_dim == 0
            || num_heads == 0
        {
            return Err(CudaExecutorError::Shape(format!(
                "invalid CUDA KDA geometry: hidden={hidden_size}, kernel={conv_kernel_dim}, heads={num_heads}, key_dim={key_head_dim}, value_dim={value_head_dim}"
            )));
        }
        Ok(Self {
            hidden_size,
            conv_kernel_dim,
            key_head_dim,
            value_head_dim,
            head_count: num_heads,
            key_size: checked_product(num_heads, key_head_dim, "KDA key size")?,
            value_size: checked_product(num_heads, value_head_dim, "KDA value size")?,
        })
    }

    fn conv_channels(self) -> Result<usize, CudaExecutorError> {
        checked_product(self.key_size, 2, "KDA query/key channels")?
            .checked_add(self.value_size)
            .ok_or_else(|| CudaExecutorError::Shape("KDA conv channels overflowed".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kda_shape_uses_combined_conv_and_per_head_temporal_geometry() {
        let shape = CudaKeyGatedDeltaComponentShape::new(
            64,
            HybridLinearAttentionConfig::KeyGatedDelta {
                conv_kernel_dim: 4,
                key_head_dim: 8,
                value_head_dim: 16,
                num_heads: 2,
            },
        )
        .expect("valid KDA shape");
        assert_eq!(shape.key_size, 16);
        assert_eq!(shape.value_size, 32);
        assert_eq!(shape.conv_channels().expect("conv channels"), 64);
    }
}
