use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::PathBuf;

use sha2::{Digest, Sha256};

use crate::cache::CachePageId;
use crate::model_artifacts::{
    GlmMoeDsaLayerCheckpointWeights, GlmMoeDsaLayerFeedForwardCheckpointWeights,
    GlmMoeDsaModelCheckpointWeights, GlmMoeDsaModelTensorSpan, LocalModelArtifacts,
    LocalModelCheckpointCatalog, ModelArtifactError, SafetensorsLayerTensorSpan,
    SafetensorsTensorData, SafetensorsTensorDecodeError, SafetensorsTensorMetadata,
    SafetensorsTensorSpan,
};
use crate::model_executor::{
    ForwardModel, ModelForwardError, ModelForwardOutput, ModelWorkerBatch,
};
use crate::transfer::{
    KvCacheMemoryLocation, KvCacheModelLayout, KvCachePageSnapshotChecksum,
    KvCachePageSnapshotImporter, KvCachePageSnapshotProvider, KvCacheTransferError,
    MooncakeKvCacheMemoryProvider, PdConfigError, TransferableKvCacheMemory,
    TransferableKvCacheRegion,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmMoeDsaRuntime {
    model_path: PathBuf,
    root_tensors: GlmMoeDsaRootTensorDescriptors,
    layers: Vec<GlmMoeDsaLayerTensorDescriptors>,
    kv_cache_layout: KvCacheModelLayout,
    rms_norm_eps_bits: u32,
    rope_theta_bits: u32,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    routed_scaling_factor_bits: u32,
    attention_shape: GlmMoeDsaAttentionShape,
}

impl GlmMoeDsaRuntime {
    pub fn from_local_model_artifacts(
        artifacts: &LocalModelArtifacts,
    ) -> Result<Self, GlmMoeDsaRuntimeError> {
        if artifacts.config().model_type.as_deref() != Some("glm_moe_dsa") {
            return Err(GlmMoeDsaRuntimeError::UnsupportedModelType {
                model_type: artifacts.config().model_type.clone(),
            });
        }

        let checkpoint = LocalModelCheckpointCatalog::from_local_model_artifacts(artifacts)?;
        let weights = checkpoint.glm_moe_dsa_model_weights()?;
        let kv_cache_layout = KvCacheModelLayout::from_hf_config(artifacts.config())?
            .ok_or(GlmMoeDsaRuntimeError::MissingKvCacheLayout)?;

        Ok(Self {
            model_path: artifacts.model_path().to_path_buf(),
            root_tensors: GlmMoeDsaRootTensorDescriptors::from_checkpoint(&weights),
            layers: weights
                .layers()
                .iter()
                .map(GlmMoeDsaLayerTensorDescriptors::from_checkpoint)
                .collect(),
            kv_cache_layout,
            rms_norm_eps_bits: artifacts
                .config()
                .rms_norm_eps
                .map(|value| value.get() as f32)
                .unwrap_or(1e-6)
                .to_bits(),
            rope_theta_bits: artifacts
                .config()
                .rope_theta
                .map(|value| value.get() as f32)
                .unwrap_or(10000.0)
                .to_bits(),
            num_experts_per_tok: artifacts.config().num_experts_per_tok.unwrap_or(1),
            norm_topk_prob: artifacts.config().norm_topk_prob.unwrap_or(false),
            routed_scaling_factor_bits: artifacts
                .config()
                .routed_scaling_factor
                .map(|value| value.get() as f32)
                .unwrap_or(1.0)
                .to_bits(),
            attention_shape: GlmMoeDsaAttentionShape::from_config(artifacts.config()),
        })
    }

    pub fn model_path(&self) -> &PathBuf {
        &self.model_path
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn root_tensors(&self) -> &GlmMoeDsaRootTensorDescriptors {
        &self.root_tensors
    }

    pub fn layers(&self) -> &[GlmMoeDsaLayerTensorDescriptors] {
        &self.layers
    }

    pub fn kv_cache_layout(&self) -> KvCacheModelLayout {
        self.kv_cache_layout
    }

    pub fn rms_norm_eps(&self) -> f32 {
        f32::from_bits(self.rms_norm_eps_bits)
    }

    pub fn rope_theta(&self) -> f32 {
        f32::from_bits(self.rope_theta_bits)
    }

    pub fn num_experts_per_tok(&self) -> usize {
        self.num_experts_per_tok
    }

    pub fn norm_topk_prob(&self) -> bool {
        self.norm_topk_prob
    }

    pub fn routed_scaling_factor(&self) -> f32 {
        f32::from_bits(self.routed_scaling_factor_bits)
    }

    pub fn attention_shape(&self) -> GlmMoeDsaAttentionShape {
        self.attention_shape
    }

    pub fn tensor_parallel_placement_plan(
        &self,
        tensor_parallel_size: usize,
    ) -> GlmMoeDsaTensorPlacementPlan {
        GlmMoeDsaTensorPlacementPlan::from_runtime(self, tensor_parallel_size)
    }

    pub fn load_tensor_parallel_shards(
        &self,
        tensor_parallel_size: usize,
    ) -> Result<GlmMoeDsaLoadedTensorParallelRuntime, GlmMoeDsaTensorShardLoadError> {
        GlmMoeDsaLoadedTensorParallelRuntime::from_runtime(self, tensor_parallel_size)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlmMoeDsaAttentionShape {
    num_attention_heads: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
}

impl GlmMoeDsaAttentionShape {
    fn from_config(config: &crate::model_artifacts::HfModelConfig) -> Self {
        let head_dim = config.head_dim.unwrap_or(1);
        Self {
            num_attention_heads: config.num_attention_heads.unwrap_or(1),
            qk_nope_head_dim: config.qk_nope_head_dim.unwrap_or(head_dim),
            qk_rope_head_dim: config.qk_rope_head_dim.unwrap_or(0),
            v_head_dim: config.v_head_dim.unwrap_or(head_dim),
        }
    }

    fn qk_head_dim(self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    fn query_width(self) -> usize {
        self.num_attention_heads * self.qk_head_dim()
    }

    fn kv_width(self) -> usize {
        self.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim)
    }

    fn value_width(self) -> usize {
        self.num_attention_heads * self.v_head_dim
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmMoeDsaRootTensorDescriptors {
    token_embeddings: GlmTensorDescriptor,
    final_norm: GlmTensorDescriptor,
    lm_head: GlmTensorDescriptor,
}

impl GlmMoeDsaRootTensorDescriptors {
    fn from_checkpoint(weights: &GlmMoeDsaModelCheckpointWeights<'_>) -> Self {
        Self {
            token_embeddings: GlmTensorDescriptor::from_model_tensor(weights.token_embeddings()),
            final_norm: GlmTensorDescriptor::from_model_tensor(weights.final_norm()),
            lm_head: GlmTensorDescriptor::from_model_tensor(weights.lm_head()),
        }
    }

    pub fn token_embeddings(&self) -> &GlmTensorDescriptor {
        &self.token_embeddings
    }

    pub fn final_norm(&self) -> &GlmTensorDescriptor {
        &self.final_norm
    }

    pub fn lm_head(&self) -> &GlmTensorDescriptor {
        &self.lm_head
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmMoeDsaLayerTensorDescriptors {
    layer_id: usize,
    q_a_proj: GlmTensorDescriptor,
    q_a_layernorm: GlmTensorDescriptor,
    q_b_proj: GlmTensorDescriptor,
    kv_a_proj_with_mqa: GlmTensorDescriptor,
    kv_a_layernorm: GlmTensorDescriptor,
    kv_b_proj: GlmTensorDescriptor,
    o_proj: GlmTensorDescriptor,
    input_layernorm: GlmTensorDescriptor,
    post_attention_layernorm: GlmTensorDescriptor,
    feed_forward: GlmMoeDsaFeedForwardTensorDescriptors,
}

impl GlmMoeDsaLayerTensorDescriptors {
    fn from_checkpoint(weights: &GlmMoeDsaLayerCheckpointWeights<'_>) -> Self {
        Self {
            layer_id: weights.layer_id(),
            q_a_proj: GlmTensorDescriptor::from_layer_tensor(weights.q_a_proj()),
            q_a_layernorm: GlmTensorDescriptor::from_layer_tensor(weights.q_a_layernorm()),
            q_b_proj: GlmTensorDescriptor::from_layer_tensor(weights.q_b_proj()),
            kv_a_proj_with_mqa: GlmTensorDescriptor::from_layer_tensor(
                weights.kv_a_proj_with_mqa(),
            ),
            kv_a_layernorm: GlmTensorDescriptor::from_layer_tensor(weights.kv_a_layernorm()),
            kv_b_proj: GlmTensorDescriptor::from_layer_tensor(weights.kv_b_proj()),
            o_proj: GlmTensorDescriptor::from_layer_tensor(weights.o_proj()),
            input_layernorm: GlmTensorDescriptor::from_layer_tensor(weights.input_layernorm()),
            post_attention_layernorm: GlmTensorDescriptor::from_layer_tensor(
                weights.post_attention_layernorm(),
            ),
            feed_forward: GlmMoeDsaFeedForwardTensorDescriptors::from_checkpoint(
                weights.feed_forward(),
            ),
        }
    }

    pub fn layer_id(&self) -> usize {
        self.layer_id
    }

    pub fn feed_forward(&self) -> &GlmMoeDsaFeedForwardTensorDescriptors {
        &self.feed_forward
    }

    fn push_tensor_placements(&self, entries: &mut Vec<GlmMoeDsaTensorPlacement>) {
        push_placement(
            entries,
            &self.q_a_proj,
            GlmMoeDsaTensorPlacementKind::Replicated,
        );
        push_placement(
            entries,
            &self.q_a_layernorm,
            GlmMoeDsaTensorPlacementKind::Replicated,
        );
        push_placement(
            entries,
            &self.q_b_proj,
            GlmMoeDsaTensorPlacementKind::ColumnParallel { axis: 0 },
        );
        push_placement(
            entries,
            &self.kv_a_proj_with_mqa,
            GlmMoeDsaTensorPlacementKind::Replicated,
        );
        push_placement(
            entries,
            &self.kv_a_layernorm,
            GlmMoeDsaTensorPlacementKind::Replicated,
        );
        push_placement(
            entries,
            &self.kv_b_proj,
            GlmMoeDsaTensorPlacementKind::ColumnParallel { axis: 0 },
        );
        push_placement(
            entries,
            &self.o_proj,
            GlmMoeDsaTensorPlacementKind::RowParallel { axis: 1 },
        );
        push_placement(
            entries,
            &self.input_layernorm,
            GlmMoeDsaTensorPlacementKind::Replicated,
        );
        push_placement(
            entries,
            &self.post_attention_layernorm,
            GlmMoeDsaTensorPlacementKind::Replicated,
        );
        self.feed_forward.push_tensor_placements(entries);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GlmMoeDsaFeedForwardTensorDescriptors {
    Dense {
        gate_proj: GlmTensorDescriptor,
        up_proj: GlmTensorDescriptor,
        down_proj: GlmTensorDescriptor,
    },
    Moe {
        gate: GlmTensorDescriptor,
        routed_experts: Vec<GlmMoeDsaRoutedExpertTensorDescriptors>,
    },
}

impl GlmMoeDsaFeedForwardTensorDescriptors {
    fn from_checkpoint(weights: &GlmMoeDsaLayerFeedForwardCheckpointWeights<'_>) -> Self {
        match weights {
            GlmMoeDsaLayerFeedForwardCheckpointWeights::Dense {
                gate_proj,
                up_proj,
                down_proj,
            } => Self::Dense {
                gate_proj: GlmTensorDescriptor::from_layer_tensor(gate_proj),
                up_proj: GlmTensorDescriptor::from_layer_tensor(up_proj),
                down_proj: GlmTensorDescriptor::from_layer_tensor(down_proj),
            },
            GlmMoeDsaLayerFeedForwardCheckpointWeights::Moe {
                gate,
                routed_experts,
            } => Self::Moe {
                gate: GlmTensorDescriptor::from_layer_tensor(gate),
                routed_experts: routed_experts
                    .groups()
                    .map(|group| GlmMoeDsaRoutedExpertTensorDescriptors {
                        expert_id: group.expert_id,
                        gate: GlmTensorDescriptor::from_tensor_span(
                            format!(
                                "model.layers.{}.mlp.experts.{}.gate_proj.weight",
                                group.layer_id, group.expert_id
                            ),
                            &group.gate,
                        ),
                        up: GlmTensorDescriptor::from_tensor_span(
                            format!(
                                "model.layers.{}.mlp.experts.{}.up_proj.weight",
                                group.layer_id, group.expert_id
                            ),
                            &group.up,
                        ),
                        down: GlmTensorDescriptor::from_tensor_span(
                            format!(
                                "model.layers.{}.mlp.experts.{}.down_proj.weight",
                                group.layer_id, group.expert_id
                            ),
                            &group.down,
                        ),
                    })
                    .collect(),
            },
        }
    }

    fn push_tensor_placements(&self, entries: &mut Vec<GlmMoeDsaTensorPlacement>) {
        match self {
            Self::Dense {
                gate_proj,
                up_proj,
                down_proj,
            } => {
                push_placement(
                    entries,
                    gate_proj,
                    GlmMoeDsaTensorPlacementKind::ColumnParallel { axis: 0 },
                );
                push_placement(
                    entries,
                    up_proj,
                    GlmMoeDsaTensorPlacementKind::ColumnParallel { axis: 0 },
                );
                push_placement(
                    entries,
                    down_proj,
                    GlmMoeDsaTensorPlacementKind::RowParallel { axis: 1 },
                );
            }
            Self::Moe {
                gate,
                routed_experts,
            } => {
                push_placement(entries, gate, GlmMoeDsaTensorPlacementKind::Replicated);
                for expert in routed_experts {
                    expert.push_tensor_placements(entries);
                }
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmMoeDsaRoutedExpertTensorDescriptors {
    expert_id: usize,
    gate: GlmTensorDescriptor,
    up: GlmTensorDescriptor,
    down: GlmTensorDescriptor,
}

impl GlmMoeDsaRoutedExpertTensorDescriptors {
    pub fn expert_id(&self) -> usize {
        self.expert_id
    }

    pub fn gate(&self) -> &GlmTensorDescriptor {
        &self.gate
    }

    pub fn up(&self) -> &GlmTensorDescriptor {
        &self.up
    }

    pub fn down(&self) -> &GlmTensorDescriptor {
        &self.down
    }

    fn push_tensor_placements(&self, entries: &mut Vec<GlmMoeDsaTensorPlacement>) {
        push_placement(
            entries,
            &self.gate,
            GlmMoeDsaTensorPlacementKind::ColumnParallel { axis: 0 },
        );
        push_placement(
            entries,
            &self.up,
            GlmMoeDsaTensorPlacementKind::ColumnParallel { axis: 0 },
        );
        push_placement(
            entries,
            &self.down,
            GlmMoeDsaTensorPlacementKind::RowParallel { axis: 1 },
        );
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmTensorDescriptor {
    tensor_name: String,
    path: PathBuf,
    dtype: String,
    shape: Vec<usize>,
    absolute_byte_offset: u64,
    byte_len: usize,
}

impl GlmTensorDescriptor {
    fn from_model_tensor(tensor: &GlmMoeDsaModelTensorSpan) -> Self {
        Self::from_tensor_span(tensor.tensor_name.clone(), &tensor.span)
    }

    fn from_layer_tensor(tensor: &SafetensorsLayerTensorSpan) -> Self {
        Self::from_tensor_span(tensor.tensor_name.clone(), &tensor.span)
    }

    fn from_tensor_span(tensor_name: String, span: &SafetensorsTensorSpan) -> Self {
        Self {
            tensor_name,
            path: span.path.clone(),
            dtype: span.metadata.dtype.clone(),
            shape: span.metadata.shape.clone(),
            absolute_byte_offset: span.absolute_byte_offset,
            byte_len: span.byte_len,
        }
    }

    pub fn tensor_name(&self) -> &str {
        &self.tensor_name
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    pub fn dtype(&self) -> &str {
        &self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn absolute_byte_offset(&self) -> u64 {
        self.absolute_byte_offset
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    fn read(&self) -> Result<SafetensorsTensorData, GlmMoeDsaTensorShardLoadError> {
        let mut file =
            fs::File::open(&self.path).map_err(|error| ModelArtifactError::ReadWeightShard {
                path: self.path.clone(),
                message: error.to_string(),
            })?;
        file.seek(SeekFrom::Start(self.absolute_byte_offset))
            .map_err(|error| ModelArtifactError::ReadWeightShard {
                path: self.path.clone(),
                message: error.to_string(),
            })?;
        let mut bytes = vec![0_u8; self.byte_len];
        file.read_exact(&mut bytes)
            .map_err(|error| ModelArtifactError::ReadWeightShard {
                path: self.path.clone(),
                message: error.to_string(),
            })?;

        Ok(SafetensorsTensorData {
            metadata: SafetensorsTensorMetadata {
                dtype: self.dtype.clone(),
                shape: self.shape.clone(),
                data_offsets: [0, self.byte_len],
            },
            bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlmMoeDsaTensorPlacementKind {
    Replicated,
    VocabParallel { axis: usize },
    ColumnParallel { axis: usize },
    RowParallel { axis: usize },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmMoeDsaTensorPlacementPlan {
    tensor_parallel_size: usize,
    entries: Vec<GlmMoeDsaTensorPlacement>,
}

impl GlmMoeDsaTensorPlacementPlan {
    fn from_runtime(runtime: &GlmMoeDsaRuntime, tensor_parallel_size: usize) -> Self {
        let mut entries = Vec::new();
        push_placement(
            &mut entries,
            runtime.root_tensors().token_embeddings(),
            GlmMoeDsaTensorPlacementKind::VocabParallel { axis: 0 },
        );
        push_placement(
            &mut entries,
            runtime.root_tensors().final_norm(),
            GlmMoeDsaTensorPlacementKind::Replicated,
        );
        push_placement(
            &mut entries,
            runtime.root_tensors().lm_head(),
            GlmMoeDsaTensorPlacementKind::VocabParallel { axis: 0 },
        );

        for layer in runtime.layers() {
            layer.push_tensor_placements(&mut entries);
        }

        Self {
            tensor_parallel_size,
            entries,
        }
    }

    pub fn tensor_parallel_size(&self) -> usize {
        self.tensor_parallel_size
    }

    pub fn entries(&self) -> &[GlmMoeDsaTensorPlacement] {
        &self.entries
    }

    pub fn kind_for(&self, tensor_name: &str) -> Option<GlmMoeDsaTensorPlacementKind> {
        self.entries
            .iter()
            .find(|entry| entry.tensor().tensor_name() == tensor_name)
            .map(GlmMoeDsaTensorPlacement::kind)
    }

    pub fn rank_shard_plan(
        &self,
        tensor_parallel_rank: usize,
    ) -> Result<GlmMoeDsaTensorRankShardPlan, GlmMoeDsaTensorShardPlanError> {
        GlmMoeDsaTensorRankShardPlan::from_placement_plan(self, tensor_parallel_rank)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmMoeDsaTensorPlacement {
    tensor: GlmTensorDescriptor,
    kind: GlmMoeDsaTensorPlacementKind,
}

impl GlmMoeDsaTensorPlacement {
    pub fn tensor(&self) -> &GlmTensorDescriptor {
        &self.tensor
    }

    pub fn kind(&self) -> GlmMoeDsaTensorPlacementKind {
        self.kind
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GlmMoeDsaTensorShardSelection {
    Full,
    Slice { axis: usize, range: Range<usize> },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmMoeDsaTensorRankShardPlan {
    tensor_parallel_size: usize,
    tensor_parallel_rank: usize,
    shards: Vec<GlmMoeDsaTensorShard>,
}

impl GlmMoeDsaTensorRankShardPlan {
    fn from_placement_plan(
        plan: &GlmMoeDsaTensorPlacementPlan,
        tensor_parallel_rank: usize,
    ) -> Result<Self, GlmMoeDsaTensorShardPlanError> {
        if tensor_parallel_rank >= plan.tensor_parallel_size {
            return Err(GlmMoeDsaTensorShardPlanError::RankOutOfBounds {
                tensor_parallel_rank,
                tensor_parallel_size: plan.tensor_parallel_size,
            });
        }

        let shards = plan
            .entries()
            .iter()
            .map(|entry| {
                let selection = shard_selection_for_entry(
                    entry,
                    plan.tensor_parallel_size,
                    tensor_parallel_rank,
                )?;
                Ok(GlmMoeDsaTensorShard {
                    tensor: entry.tensor().clone(),
                    kind: entry.kind(),
                    selection,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            tensor_parallel_size: plan.tensor_parallel_size,
            tensor_parallel_rank,
            shards,
        })
    }

    pub fn tensor_parallel_size(&self) -> usize {
        self.tensor_parallel_size
    }

    pub fn tensor_parallel_rank(&self) -> usize {
        self.tensor_parallel_rank
    }

    pub fn shards(&self) -> &[GlmMoeDsaTensorShard] {
        &self.shards
    }

    pub fn selection_for(&self, tensor_name: &str) -> Option<GlmMoeDsaTensorShardSelection> {
        self.shards
            .iter()
            .find(|shard| shard.tensor().tensor_name() == tensor_name)
            .map(|shard| shard.selection().clone())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmMoeDsaTensorShard {
    tensor: GlmTensorDescriptor,
    kind: GlmMoeDsaTensorPlacementKind,
    selection: GlmMoeDsaTensorShardSelection,
}

impl GlmMoeDsaTensorShard {
    pub fn tensor(&self) -> &GlmTensorDescriptor {
        &self.tensor
    }

    pub fn kind(&self) -> GlmMoeDsaTensorPlacementKind {
        self.kind
    }

    pub fn selection(&self) -> &GlmMoeDsaTensorShardSelection {
        &self.selection
    }

    pub fn load(&self) -> Result<GlmMoeDsaLoadedTensorShard, GlmMoeDsaTensorShardLoadError> {
        let tensor_data = self.tensor.read()?;
        let (shape, bytes) = materialize_tensor_shard(
            self.tensor.tensor_name(),
            self.tensor.shape(),
            tensor_data.dtype_byte_width(),
            &tensor_data.bytes,
            &self.selection,
        )?;

        Ok(GlmMoeDsaLoadedTensorShard {
            tensor_name: self.tensor.tensor_name().to_string(),
            dtype: self.tensor.dtype().to_string(),
            shape,
            selection: self.selection.clone(),
            bytes,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmMoeDsaLoadedTensorShard {
    tensor_name: String,
    dtype: String,
    shape: Vec<usize>,
    selection: GlmMoeDsaTensorShardSelection,
    bytes: Vec<u8>,
}

impl GlmMoeDsaLoadedTensorShard {
    pub fn tensor_name(&self) -> &str {
        &self.tensor_name
    }

    pub fn dtype(&self) -> &str {
        &self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn selection(&self) -> &GlmMoeDsaTensorShardSelection {
        &self.selection
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn decode_f32_values(&self) -> Result<Vec<f32>, SafetensorsTensorDecodeError> {
        SafetensorsTensorData {
            metadata: SafetensorsTensorMetadata {
                dtype: self.dtype.clone(),
                shape: self.shape.clone(),
                data_offsets: [0, self.bytes.len()],
            },
            bytes: self.bytes.clone(),
        }
        .decode_f32_values()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmMoeDsaLoadedTensorParallelRuntime {
    runtime: GlmMoeDsaRuntime,
    tensor_parallel_size: usize,
    ranks: Vec<GlmMoeDsaLoadedTensorRank>,
}

impl GlmMoeDsaLoadedTensorParallelRuntime {
    fn from_runtime(
        runtime: &GlmMoeDsaRuntime,
        tensor_parallel_size: usize,
    ) -> Result<Self, GlmMoeDsaTensorShardLoadError> {
        if tensor_parallel_size == 0 {
            return Err(GlmMoeDsaTensorShardLoadError::ShardPlan(
                GlmMoeDsaTensorShardPlanError::RankOutOfBounds {
                    tensor_parallel_rank: 0,
                    tensor_parallel_size,
                },
            ));
        }

        let placement_plan = runtime.tensor_parallel_placement_plan(tensor_parallel_size);
        let ranks = (0..tensor_parallel_size)
            .map(|rank| {
                let rank_plan = placement_plan.rank_shard_plan(rank)?;
                GlmMoeDsaLoadedTensorRank::from_rank_shard_plan(&rank_plan)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            runtime: runtime.clone(),
            tensor_parallel_size,
            ranks,
        })
    }

    pub fn tensor_parallel_size(&self) -> usize {
        self.tensor_parallel_size
    }

    pub fn rank_count(&self) -> usize {
        self.ranks.len()
    }

    pub fn ranks(&self) -> &[GlmMoeDsaLoadedTensorRank] {
        &self.ranks
    }

    pub fn rank(&self, tensor_parallel_rank: usize) -> Option<&GlmMoeDsaLoadedTensorRank> {
        self.ranks
            .iter()
            .find(|rank| rank.tensor_parallel_rank() == tensor_parallel_rank)
    }

    pub fn layer_count(&self) -> usize {
        self.runtime.layer_count()
    }

    pub fn kv_cache_layout(&self) -> KvCacheModelLayout {
        self.runtime.kv_cache_layout()
    }

    pub fn loaded_shard_count(&self) -> usize {
        self.ranks
            .iter()
            .map(GlmMoeDsaLoadedTensorRank::shard_count)
            .sum()
    }

    pub fn loaded_byte_len(&self) -> usize {
        self.ranks
            .iter()
            .map(GlmMoeDsaLoadedTensorRank::loaded_byte_len)
            .sum()
    }

    pub fn decode_f32_tensor_parallel_shards(
        &self,
    ) -> Result<GlmMoeDsaF32TensorParallelRuntime, SafetensorsTensorDecodeError> {
        GlmMoeDsaF32TensorParallelRuntime::from_loaded_runtime(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmMoeDsaLoadedTensorRank {
    tensor_parallel_rank: usize,
    shards: Vec<GlmMoeDsaLoadedTensorShard>,
}

impl GlmMoeDsaLoadedTensorRank {
    fn from_rank_shard_plan(
        rank_plan: &GlmMoeDsaTensorRankShardPlan,
    ) -> Result<Self, GlmMoeDsaTensorShardLoadError> {
        let shards = rank_plan
            .shards()
            .iter()
            .map(GlmMoeDsaTensorShard::load)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            tensor_parallel_rank: rank_plan.tensor_parallel_rank(),
            shards,
        })
    }

    pub fn tensor_parallel_rank(&self) -> usize {
        self.tensor_parallel_rank
    }

    pub fn shards(&self) -> &[GlmMoeDsaLoadedTensorShard] {
        &self.shards
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub fn loaded_byte_len(&self) -> usize {
        self.shards.iter().map(|shard| shard.bytes().len()).sum()
    }

    pub fn tensor_shard(&self, tensor_name: &str) -> Option<&GlmMoeDsaLoadedTensorShard> {
        self.shards
            .iter()
            .find(|shard| shard.tensor_name() == tensor_name)
    }

    fn decode_f32_shards(&self) -> Result<GlmMoeDsaF32TensorRank, SafetensorsTensorDecodeError> {
        let shards = self
            .shards
            .iter()
            .map(GlmMoeDsaF32TensorShard::from_loaded_shard)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(GlmMoeDsaF32TensorRank {
            tensor_parallel_rank: self.tensor_parallel_rank,
            shards,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlmMoeDsaF32TensorParallelRuntime {
    tensor_parallel_size: usize,
    layer_count: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
    attention_shape: GlmMoeDsaAttentionShape,
    feed_forward_kinds: Vec<GlmMoeDsaF32FeedForwardKind>,
    ranks: Vec<GlmMoeDsaF32TensorRank>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GlmMoeDsaF32FeedForwardKind {
    Dense,
    Moe,
}

impl From<&GlmMoeDsaFeedForwardTensorDescriptors> for GlmMoeDsaF32FeedForwardKind {
    fn from(descriptors: &GlmMoeDsaFeedForwardTensorDescriptors) -> Self {
        match descriptors {
            GlmMoeDsaFeedForwardTensorDescriptors::Dense { .. } => Self::Dense,
            GlmMoeDsaFeedForwardTensorDescriptors::Moe { .. } => Self::Moe,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlmMoeDsaF32LayerOutput {
    hidden_states: Vec<f32>,
    residual: Vec<f32>,
}

impl GlmMoeDsaF32LayerOutput {
    fn new(hidden_states: Vec<f32>, residual: Vec<f32>) -> Self {
        Self {
            hidden_states,
            residual,
        }
    }

    pub fn hidden_states(&self) -> &[f32] {
        &self.hidden_states
    }

    pub fn residual(&self) -> &[f32] {
        &self.residual
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlmMoeDsaF32AttentionProjectionOutput {
    q_lora: Vec<f32>,
    q: Vec<f32>,
    kv_lora: Vec<f32>,
    k_rope: Vec<f32>,
    kv: Vec<f32>,
}

impl GlmMoeDsaF32AttentionProjectionOutput {
    fn new(
        q_lora: Vec<f32>,
        q: Vec<f32>,
        kv_lora: Vec<f32>,
        k_rope: Vec<f32>,
        kv: Vec<f32>,
    ) -> Self {
        Self {
            q_lora,
            q,
            kv_lora,
            k_rope,
            kv,
        }
    }

    pub fn q_lora(&self) -> &[f32] {
        &self.q_lora
    }

    pub fn q(&self) -> &[f32] {
        &self.q
    }

    pub fn kv_lora(&self) -> &[f32] {
        &self.kv_lora
    }

    pub fn k_rope(&self) -> &[f32] {
        &self.k_rope
    }

    pub fn kv(&self) -> &[f32] {
        &self.kv
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GlmMoeDsaF32KvPageStore {
    entries: HashMap<(usize, CachePageId), GlmMoeDsaF32KvPageEntry>,
}

#[derive(Clone, Debug, PartialEq)]
struct GlmMoeDsaF32KvPageEntry {
    position: usize,
    projection: GlmMoeDsaF32AttentionProjectionOutput,
}

impl GlmMoeDsaF32KvPageStore {
    pub fn contains(&self, layer_id: usize, cache_page: CachePageId) -> bool {
        self.entries.contains_key(&(layer_id, cache_page))
    }

    fn insert(
        &mut self,
        layer_id: usize,
        cache_page: CachePageId,
        position: usize,
        projection: GlmMoeDsaF32AttentionProjectionOutput,
    ) {
        self.entries.insert(
            (layer_id, cache_page),
            GlmMoeDsaF32KvPageEntry {
                position,
                projection,
            },
        );
    }

    fn get(&self, layer_id: usize, cache_page: CachePageId) -> Option<&GlmMoeDsaF32KvPageEntry> {
        self.entries.get(&(layer_id, cache_page))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlmMoeDsaF32KvPageSnapshot {
    layer_id: usize,
    cache_page: CachePageId,
    position: usize,
    projection: GlmMoeDsaF32AttentionProjectionOutput,
}

impl GlmMoeDsaF32KvPageSnapshot {
    pub fn layer_id(&self) -> usize {
        self.layer_id
    }

    pub fn cache_page(&self) -> CachePageId {
        self.cache_page
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn projection(&self) -> &GlmMoeDsaF32AttentionProjectionOutput {
        &self.projection
    }
}

impl KvCachePageSnapshotChecksum for GlmMoeDsaF32KvPageSnapshot {
    fn update_content_checksum(&self, hasher: &mut Sha256) {
        hasher.update((self.layer_id as u64).to_le_bytes());
        hasher.update((self.cache_page.as_usize() as u64).to_le_bytes());
        hasher.update((self.position as u64).to_le_bytes());
        update_f32_slice_checksum(hasher, self.projection.q_lora());
        update_f32_slice_checksum(hasher, self.projection.q());
        update_f32_slice_checksum(hasher, self.projection.kv_lora());
        update_f32_slice_checksum(hasher, self.projection.k_rope());
        update_f32_slice_checksum(hasher, self.projection.kv());
    }
}

fn update_f32_slice_checksum(hasher: &mut Sha256, values: &[f32]) {
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(value.to_le_bytes());
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlmMoeDsaF32CachedForwardModel {
    runtime: GlmMoeDsaF32TensorParallelRuntime,
    kv_cache: GlmMoeDsaF32KvPageStore,
    transfer_pages: GlmMoeDsaF32TransferPageStore,
}

struct GlmMoeDsaF32KvAttentionBatch<'a> {
    positions: &'a [usize],
    out_cache_pages: &'a [CachePageId],
    sequence_cache_pages: &'a [CachePageId],
}

#[derive(Clone, Debug, PartialEq)]
struct GlmMoeDsaF32TransferPageStore {
    token_slot_size_bytes: usize,
    page_size_bytes: usize,
    slot_capacity: usize,
    registration_locked: bool,
    pages: Vec<u8>,
}

impl GlmMoeDsaF32CachedForwardModel {
    pub fn new(runtime: GlmMoeDsaF32TensorParallelRuntime) -> Self {
        let transfer_pages = GlmMoeDsaF32TransferPageStore::initial_for_runtime(&runtime);
        Self {
            runtime,
            kv_cache: GlmMoeDsaF32KvPageStore::default(),
            transfer_pages,
        }
    }

    pub fn runtime(&self) -> &GlmMoeDsaF32TensorParallelRuntime {
        &self.runtime
    }

    pub fn rank_count(&self) -> usize {
        self.runtime.rank_count()
    }

    pub fn kv_cache_contains(&self, layer_id: usize, cache_page: CachePageId) -> bool {
        self.kv_cache.contains(layer_id, cache_page)
    }

    pub fn export_kv_cache_pages(
        &self,
        cache_pages: &[CachePageId],
    ) -> Result<Vec<GlmMoeDsaF32KvPageSnapshot>, GlmMoeDsaF32KernelError> {
        let mut snapshots = Vec::with_capacity(self.runtime.layer_count() * cache_pages.len());
        for layer_id in 0..self.runtime.layer_count() {
            for cache_page in cache_pages {
                let Some(entry) = self.kv_cache.get(layer_id, *cache_page) else {
                    return Err(GlmMoeDsaF32KernelError::MissingKvCachePage {
                        layer_id,
                        cache_page: cache_page.as_usize(),
                    });
                };
                snapshots.push(GlmMoeDsaF32KvPageSnapshot {
                    layer_id,
                    cache_page: *cache_page,
                    position: entry.position,
                    projection: entry.projection.clone(),
                });
            }
        }

        Ok(snapshots)
    }

    pub fn import_kv_cache_pages(
        &mut self,
        snapshots: impl IntoIterator<Item = GlmMoeDsaF32KvPageSnapshot>,
    ) -> Result<(), GlmMoeDsaF32KernelError> {
        for snapshot in snapshots {
            if snapshot.layer_id >= self.runtime.layer_count() {
                return Err(GlmMoeDsaF32KernelError::LayerOutOfBounds {
                    layer_id: snapshot.layer_id,
                    layer_count: self.runtime.layer_count(),
                });
            }
            self.kv_cache.insert(
                snapshot.layer_id,
                snapshot.cache_page,
                snapshot.position,
                snapshot.projection,
            );
        }

        Ok(())
    }

    pub fn reserve_transfer_slots(
        &mut self,
        slot_capacity: usize,
        page_size: usize,
    ) -> Result<(), KvCacheTransferError> {
        if page_size == 0 {
            return Err(KvCacheTransferError::Runtime(
                "GLM transfer KV page size must be non-zero".to_string(),
            ));
        }
        if slot_capacity == 0 {
            return Err(KvCacheTransferError::Runtime(
                "GLM transfer KV slot capacity must be non-zero".to_string(),
            ));
        }
        if !slot_capacity.is_multiple_of(page_size) {
            return Err(KvCacheTransferError::Runtime(format!(
                "GLM transfer KV slot capacity {slot_capacity} must be divisible by page size {page_size}"
            )));
        }
        let token_slot_size_bytes = self
            .runtime
            .zero_transfer_page_size_bytes()
            .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))?;
        let page_size_bytes = page_size
            .checked_mul(token_slot_size_bytes)
            .ok_or_else(|| {
                KvCacheTransferError::Runtime(
                    "GLM transfer physical page size overflowed".to_string(),
                )
            })?;
        let byte_len = slot_capacity
            .checked_mul(token_slot_size_bytes)
            .ok_or_else(|| {
                KvCacheTransferError::Runtime(
                    "GLM transfer KV page backing store length overflowed".to_string(),
                )
            })?;
        self.transfer_pages.token_slot_size_bytes = token_slot_size_bytes;
        self.transfer_pages.page_size_bytes = page_size_bytes;
        self.transfer_pages.slot_capacity = slot_capacity;
        self.transfer_pages.registration_locked = true;
        self.transfer_pages.pages.resize(byte_len, 0);
        Ok(())
    }

    fn refresh_transfer_pages(&mut self) -> Result<(), KvCacheTransferError> {
        let mut cache_pages = Vec::new();
        for (_, cache_page) in self.kv_cache.entries.keys() {
            cache_pages.push(cache_page.as_usize());
        }
        cache_pages.sort_unstable();
        cache_pages.dedup();

        let Some(max_cache_slot) = cache_pages.iter().copied().max() else {
            self.transfer_pages.pages.fill(0);
            return Ok(());
        };

        let mut serialized_pages = Vec::with_capacity(cache_pages.len());
        let token_slot_size_bytes = self.transfer_pages.token_slot_size_bytes;
        for cache_page_index in cache_pages {
            let cache_page = CachePageId::from(cache_page_index);
            let page = self.serialize_transfer_page(cache_page)?;
            if page.is_empty() {
                return Err(KvCacheTransferError::Runtime(
                    "GLM transfer KV page serialization produced an empty page".to_string(),
                ));
            }
            if page.len() != token_slot_size_bytes {
                return Err(KvCacheTransferError::Runtime(format!(
                    "GLM transfer KV slot {} serialized to {} bytes but expected {token_slot_size_bytes}",
                    cache_page_index,
                    page.len()
                )));
            }
            serialized_pages.push((cache_page, page));
        }

        let required_slot_capacity = max_cache_slot.checked_add(1).ok_or_else(|| {
            KvCacheTransferError::Runtime("GLM transfer KV slot capacity overflowed".to_string())
        })?;
        if required_slot_capacity > self.transfer_pages.slot_capacity {
            if self.transfer_pages.registration_locked {
                return Err(KvCacheTransferError::Runtime(format!(
                    "GLM transfer KV slot {max_cache_slot} exceeds registered slot capacity {}",
                    self.transfer_pages.slot_capacity
                )));
            }
            let byte_len = required_slot_capacity
                .checked_mul(token_slot_size_bytes)
                .ok_or_else(|| {
                    KvCacheTransferError::Runtime(
                        "GLM transfer KV page backing store length overflowed".to_string(),
                    )
                })?;
            self.transfer_pages.slot_capacity = required_slot_capacity;
            self.transfer_pages.pages.resize(byte_len, 0);
        }
        self.transfer_pages.pages.fill(0);

        for (cache_page, page) in serialized_pages {
            let offset = cache_page
                .as_usize()
                .checked_mul(token_slot_size_bytes)
                .ok_or_else(|| {
                    KvCacheTransferError::Runtime(
                        "GLM transfer KV page offset overflowed".to_string(),
                    )
                })?;
            self.transfer_pages.pages[offset..offset + token_slot_size_bytes]
                .copy_from_slice(&page);
        }

        Ok(())
    }

    fn serialize_transfer_page(
        &self,
        cache_page: CachePageId,
    ) -> Result<Vec<u8>, KvCacheTransferError> {
        let mut page = Vec::new();
        for layer_id in 0..self.runtime.layer_count() {
            let entry = self.kv_cache.get(layer_id, cache_page).ok_or_else(|| {
                KvCacheTransferError::Runtime(format!(
                    "missing GLM layer {layer_id} KV cache page {} for Mooncake transfer memory",
                    cache_page.as_usize()
                ))
            })?;
            append_f32_slice_bytes(&mut page, entry.projection.q_lora());
            append_f32_slice_bytes(&mut page, entry.projection.q());
            append_f32_slice_bytes(&mut page, entry.projection.kv_lora());
            append_f32_slice_bytes(&mut page, entry.projection.k_rope());
            append_f32_slice_bytes(&mut page, entry.projection.kv());
        }
        Ok(page)
    }
}

impl GlmMoeDsaF32TransferPageStore {
    fn initial_for_runtime(runtime: &GlmMoeDsaF32TensorParallelRuntime) -> Self {
        let token_slot_size_bytes = runtime
            .zero_transfer_page_size_bytes()
            .expect("GLM runtime should expose transfer page dimensions");
        Self {
            token_slot_size_bytes,
            page_size_bytes: token_slot_size_bytes,
            slot_capacity: 1,
            registration_locked: false,
            pages: vec![0; token_slot_size_bytes],
        }
    }
}

fn append_f32_slice_bytes(output: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        output.extend_from_slice(&value.to_le_bytes());
    }
}

impl KvCachePageSnapshotProvider for GlmMoeDsaF32CachedForwardModel {
    type Snapshot = GlmMoeDsaF32KvPageSnapshot;

    fn export_kv_cache_pages(
        &self,
        cache_pages: &[CachePageId],
    ) -> Result<Vec<Self::Snapshot>, KvCacheTransferError> {
        GlmMoeDsaF32CachedForwardModel::export_kv_cache_pages(self, cache_pages)
            .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))
    }
}

impl KvCachePageSnapshotImporter for GlmMoeDsaF32CachedForwardModel {
    type Snapshot = GlmMoeDsaF32KvPageSnapshot;

    fn import_kv_cache_pages(
        &mut self,
        snapshots: Vec<Self::Snapshot>,
    ) -> Result<(), KvCacheTransferError> {
        GlmMoeDsaF32CachedForwardModel::import_kv_cache_pages(self, snapshots)
            .map_err(|error| KvCacheTransferError::Runtime(error.to_string()))
    }
}

impl MooncakeKvCacheMemoryProvider for GlmMoeDsaF32CachedForwardModel {
    fn mooncake_kv_cache_memory(&self) -> Result<TransferableKvCacheMemory, KvCacheTransferError> {
        TransferableKvCacheMemory::new(
            vec![TransferableKvCacheRegion {
                base_addr: self.transfer_pages.pages.as_ptr() as usize,
                byte_len: self.transfer_pages.pages.len(),
                page_size_bytes: self.transfer_pages.page_size_bytes,
            }],
            self.transfer_pages.page_size_bytes,
            KvCacheMemoryLocation::Cpu { numa_node: 0 },
        )
    }
}

impl ForwardModel for GlmMoeDsaF32CachedForwardModel {
    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        let logits = self
            .runtime
            .transformer_lm_head_logits_with_kv_cache(batch, &mut self.kv_cache)
            .map_err(|error| ModelForwardError::Runtime(error.to_string()))?;
        self.refresh_transfer_pages()
            .map_err(|error| ModelForwardError::Runtime(error.to_string()))?;
        ModelForwardOutput::new(logits)
    }
}

impl GlmMoeDsaF32TensorParallelRuntime {
    fn from_loaded_runtime(
        runtime: &GlmMoeDsaLoadedTensorParallelRuntime,
    ) -> Result<Self, SafetensorsTensorDecodeError> {
        let ranks = runtime
            .ranks()
            .iter()
            .map(GlmMoeDsaLoadedTensorRank::decode_f32_shards)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            tensor_parallel_size: runtime.tensor_parallel_size(),
            layer_count: runtime.layer_count(),
            rms_norm_eps: runtime.runtime.rms_norm_eps(),
            rope_theta: runtime.runtime.rope_theta(),
            num_experts_per_tok: runtime.runtime.num_experts_per_tok(),
            norm_topk_prob: runtime.runtime.norm_topk_prob(),
            routed_scaling_factor: runtime.runtime.routed_scaling_factor(),
            attention_shape: runtime.runtime.attention_shape(),
            feed_forward_kinds: runtime
                .runtime
                .layers()
                .iter()
                .map(|layer| GlmMoeDsaF32FeedForwardKind::from(layer.feed_forward()))
                .collect(),
            ranks,
        })
    }

    pub fn tensor_parallel_size(&self) -> usize {
        self.tensor_parallel_size
    }

    pub fn rank_count(&self) -> usize {
        self.ranks.len()
    }

    pub fn layer_count(&self) -> usize {
        self.layer_count
    }

    pub fn rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }

    pub fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    pub fn num_experts_per_tok(&self) -> usize {
        self.num_experts_per_tok
    }

    pub fn norm_topk_prob(&self) -> bool {
        self.norm_topk_prob
    }

    pub fn routed_scaling_factor(&self) -> f32 {
        self.routed_scaling_factor
    }

    pub fn attention_shape(&self) -> GlmMoeDsaAttentionShape {
        self.attention_shape
    }

    fn zero_transfer_page_size_bytes(&self) -> Result<usize, GlmMoeDsaF32KernelError> {
        let mut f32_values = 0_usize;
        for layer_id in 0..self.layer_count {
            let q_lora_width = self
                .layer_norm_weight(&format!(
                    "model.layers.{layer_id}.self_attn.q_a_layernorm.weight"
                ))?
                .len();
            let kv_lora_width = self
                .layer_norm_weight(&format!(
                    "model.layers.{layer_id}.self_attn.kv_a_layernorm.weight"
                ))?
                .len();
            f32_values = f32_values
                .checked_add(q_lora_width)
                .and_then(|value| value.checked_add(self.attention_shape.query_width()))
                .and_then(|value| value.checked_add(kv_lora_width))
                .and_then(|value| value.checked_add(self.attention_shape.qk_rope_head_dim))
                .and_then(|value| value.checked_add(self.attention_shape.kv_width()))
                .ok_or_else(|| GlmMoeDsaF32KernelError::Runtime {
                    message: "GLM transfer page size overflowed".to_string(),
                })?;
        }
        f32_values
            .checked_mul(std::mem::size_of::<f32>())
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| GlmMoeDsaF32KernelError::Runtime {
                message: "GLM transfer page size must be non-zero".to_string(),
            })
    }

    pub fn ranks(&self) -> &[GlmMoeDsaF32TensorRank] {
        &self.ranks
    }

    pub fn rank(&self, tensor_parallel_rank: usize) -> Option<&GlmMoeDsaF32TensorRank> {
        self.ranks
            .iter()
            .find(|rank| rank.tensor_parallel_rank() == tensor_parallel_rank)
    }

    pub fn embedding_lm_head_logits(
        &self,
        batch: &ModelWorkerBatch,
    ) -> Result<Vec<Vec<f32>>, GlmMoeDsaF32KernelError> {
        batch
            .last_input_token_ids()
            .into_iter()
            .map(|token_id| self.embedding_lm_head_logits_for_token(token_id))
            .collect()
    }

    pub fn transformer_lm_head_logits(
        &self,
        batch: &ModelWorkerBatch,
    ) -> Result<Vec<Vec<f32>>, GlmMoeDsaF32KernelError> {
        batch
            .sequence_offsets()
            .iter()
            .zip(batch.sequence_token_counts())
            .map(|(offset, sequence_token_count)| {
                let input_end = *offset + *sequence_token_count;
                let positions = (0..*sequence_token_count).collect::<Vec<_>>();
                self.transformer_lm_head_logits_for_request(
                    &batch.sequence_token_ids()[*offset..input_end],
                    &positions,
                )
            })
            .collect()
    }

    pub fn transformer_lm_head_logits_with_kv_cache(
        &self,
        batch: &ModelWorkerBatch,
        kv_cache: &mut GlmMoeDsaF32KvPageStore,
    ) -> Result<Vec<Vec<f32>>, GlmMoeDsaF32KernelError> {
        if batch.out_cache_pages().len() != batch.input_ids().len()
            || batch.sequence_cache_pages().len() != batch.sequence_token_ids().len()
        {
            return self.transformer_lm_head_logits(batch);
        }

        batch
            .sequence_offsets()
            .iter()
            .zip(batch.sequence_token_counts())
            .zip(batch.request_offsets())
            .zip(batch.input_token_counts())
            .map(
                |(((sequence_offset, sequence_token_count), request_offset), input_token_count)| {
                    let sequence_end = *sequence_offset + *sequence_token_count;
                    let input_end = *request_offset + *input_token_count;
                    self.transformer_lm_head_logits_for_cached_request(
                        &batch.input_ids()[*request_offset..input_end],
                        &batch.positions()[*request_offset..input_end],
                        &batch.out_cache_pages()[*request_offset..input_end],
                        &batch.sequence_cache_pages()[*sequence_offset..sequence_end],
                        kv_cache,
                    )
                },
            )
            .collect()
    }

    pub fn attention_projection_output(
        &self,
        layer_id: usize,
        hidden: &[f32],
    ) -> Result<GlmMoeDsaF32AttentionProjectionOutput, GlmMoeDsaF32KernelError> {
        let q_lora = self.replicated_projection_with_norm(
            &format!("model.layers.{layer_id}.self_attn.q_a_proj.weight"),
            &format!("model.layers.{layer_id}.self_attn.q_a_layernorm.weight"),
            hidden,
        )?;
        let q = self.column_parallel_projection(
            &format!("model.layers.{layer_id}.self_attn.q_b_proj.weight"),
            &q_lora,
        )?;

        let kv_projection_name =
            format!("model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight");
        let kv_norm_name = format!("model.layers.{layer_id}.self_attn.kv_a_layernorm.weight");
        let kv_a = self.replicated_projection(&kv_projection_name, hidden)?;
        let kv_lora_width = self.layer_norm_weight(&kv_norm_name)?.len();
        let expected_kv_a_width = kv_lora_width + self.attention_shape.qk_rope_head_dim;
        if kv_a.len() != expected_kv_a_width {
            return Err(GlmMoeDsaF32KernelError::HiddenSizeMismatch {
                tensor_name: kv_projection_name,
                expected: expected_kv_a_width,
                actual: kv_a.len(),
            });
        }
        let mut kv_lora = kv_a[..kv_lora_width].to_vec();
        let norm_weight = self.layer_norm_weight(&kv_norm_name)?;
        apply_rms_norm(&kv_norm_name, &mut kv_lora, norm_weight, self.rms_norm_eps)?;
        let k_rope = kv_a[kv_lora_width..].to_vec();
        let kv = self.column_parallel_projection(
            &format!("model.layers.{layer_id}.self_attn.kv_b_proj.weight"),
            &kv_lora,
        )?;

        Ok(GlmMoeDsaF32AttentionProjectionOutput::new(
            q_lora, q, kv_lora, k_rope, kv,
        ))
    }

    pub fn attention_output(
        &self,
        layer_id: usize,
        hidden_states: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, GlmMoeDsaF32KernelError> {
        let positions = (0..hidden_states.len()).collect::<Vec<_>>();
        self.attention_output_with_positions(layer_id, hidden_states, &positions)
    }

    pub fn attention_output_with_kv_cache(
        &self,
        layer_id: usize,
        hidden_states: &[Vec<f32>],
        positions: &[usize],
        out_cache_pages: &[CachePageId],
        sequence_cache_pages: &[CachePageId],
        kv_cache: &mut GlmMoeDsaF32KvPageStore,
    ) -> Result<Vec<Vec<f32>>, GlmMoeDsaF32KernelError> {
        if positions.len() != hidden_states.len() {
            return Err(GlmMoeDsaF32KernelError::TokenCountMismatch {
                tensor_name: format!("model.layers.{layer_id}.self_attn.rotary_emb"),
                expected: hidden_states.len(),
                actual: positions.len(),
            });
        }
        if out_cache_pages.len() != hidden_states.len() {
            return Err(GlmMoeDsaF32KernelError::TokenCountMismatch {
                tensor_name: format!("model.layers.{layer_id}.self_attn.kv_cache"),
                expected: hidden_states.len(),
                actual: out_cache_pages.len(),
            });
        }

        for ((hidden, position), cache_page) in hidden_states
            .iter()
            .zip(positions)
            .zip(out_cache_pages.iter().copied())
        {
            let projection = self.attention_projection_output(layer_id, hidden)?;
            kv_cache.insert(layer_id, cache_page, *position, projection);
        }

        let mut projections = Vec::with_capacity(sequence_cache_pages.len());
        let mut sequence_positions = Vec::with_capacity(sequence_cache_pages.len());
        for cache_page in sequence_cache_pages {
            let Some(entry) = kv_cache.get(layer_id, *cache_page) else {
                return Err(GlmMoeDsaF32KernelError::MissingKvCachePage {
                    layer_id,
                    cache_page: cache_page.as_usize(),
                });
            };
            projections.push(entry.projection.clone());
            sequence_positions.push(entry.position);
        }

        let outputs = self.causal_attention_values(layer_id, &projections, &sequence_positions)?;
        let output_start = outputs.len().checked_sub(hidden_states.len()).ok_or(
            GlmMoeDsaF32KernelError::TokenCountMismatch {
                tensor_name: format!("model.layers.{layer_id}.self_attn.kv_cache"),
                expected: hidden_states.len(),
                actual: outputs.len(),
            },
        )?;
        outputs[output_start..]
            .iter()
            .map(|values| {
                self.row_parallel_projection(
                    &format!("model.layers.{layer_id}.self_attn.o_proj.weight"),
                    values,
                )
            })
            .collect()
    }

    fn attention_output_with_positions(
        &self,
        layer_id: usize,
        hidden_states: &[Vec<f32>],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>, GlmMoeDsaF32KernelError> {
        let projections = hidden_states
            .iter()
            .map(|hidden| self.attention_projection_output(layer_id, hidden))
            .collect::<Result<Vec<_>, _>>()?;
        let attention_values = self.causal_attention_values(layer_id, &projections, positions)?;
        attention_values
            .iter()
            .map(|values| {
                self.row_parallel_projection(
                    &format!("model.layers.{layer_id}.self_attn.o_proj.weight"),
                    values,
                )
            })
            .collect()
    }

    pub fn transformer_layer_output(
        &self,
        layer_id: usize,
        hidden_states: &[Vec<f32>],
        residuals: Option<&[Vec<f32>]>,
    ) -> Result<Vec<GlmMoeDsaF32LayerOutput>, GlmMoeDsaF32KernelError> {
        let positions = (0..hidden_states.len()).collect::<Vec<_>>();
        self.transformer_layer_output_with_positions(layer_id, hidden_states, &positions, residuals)
    }

    fn transformer_layer_output_with_positions(
        &self,
        layer_id: usize,
        hidden_states: &[Vec<f32>],
        positions: &[usize],
        residuals: Option<&[Vec<f32>]>,
    ) -> Result<Vec<GlmMoeDsaF32LayerOutput>, GlmMoeDsaF32KernelError> {
        if let Some(residuals) = residuals
            && residuals.len() != hidden_states.len()
        {
            return Err(GlmMoeDsaF32KernelError::TokenCountMismatch {
                tensor_name: format!("model.layers.{layer_id}.input_layernorm.weight"),
                expected: hidden_states.len(),
                actual: residuals.len(),
            });
        }

        let mut attention_input = Vec::with_capacity(hidden_states.len());
        let mut attention_residual = Vec::with_capacity(hidden_states.len());
        for (token_index, hidden) in hidden_states.iter().enumerate() {
            let residual = residuals.and_then(|residuals| residuals.get(token_index));
            let (normalized, residual) =
                self.prepare_attention_input(layer_id, hidden, residual.map(Vec::as_slice))?;
            attention_input.push(normalized);
            attention_residual.push(residual);
        }

        let attention_output =
            self.attention_output_with_positions(layer_id, &attention_input, positions)?;
        attention_output
            .iter()
            .zip(&attention_residual)
            .map(|(attention_output, residual)| {
                self.feed_forward_layer_output(layer_id, attention_output, Some(residual))
            })
            .collect()
    }

    fn transformer_layer_output_with_kv_cache(
        &self,
        layer_id: usize,
        hidden_states: &[Vec<f32>],
        batch: GlmMoeDsaF32KvAttentionBatch<'_>,
        kv_cache: &mut GlmMoeDsaF32KvPageStore,
        residuals: Option<&[Vec<f32>]>,
    ) -> Result<Vec<GlmMoeDsaF32LayerOutput>, GlmMoeDsaF32KernelError> {
        if let Some(residuals) = residuals
            && residuals.len() != hidden_states.len()
        {
            return Err(GlmMoeDsaF32KernelError::TokenCountMismatch {
                tensor_name: format!("model.layers.{layer_id}.input_layernorm.weight"),
                expected: hidden_states.len(),
                actual: residuals.len(),
            });
        }

        let mut attention_input = Vec::with_capacity(hidden_states.len());
        let mut attention_residual = Vec::with_capacity(hidden_states.len());
        for (token_index, hidden) in hidden_states.iter().enumerate() {
            let residual = residuals.and_then(|residuals| residuals.get(token_index));
            let (normalized, residual) =
                self.prepare_attention_input(layer_id, hidden, residual.map(Vec::as_slice))?;
            attention_input.push(normalized);
            attention_residual.push(residual);
        }

        let attention_output = self.attention_output_with_kv_cache(
            layer_id,
            &attention_input,
            batch.positions,
            batch.out_cache_pages,
            batch.sequence_cache_pages,
            kv_cache,
        )?;
        attention_output
            .iter()
            .zip(&attention_residual)
            .map(|(attention_output, residual)| {
                self.feed_forward_layer_output(layer_id, attention_output, Some(residual))
            })
            .collect()
    }

    pub fn dense_mlp_output(
        &self,
        layer_id: usize,
        hidden: &[f32],
    ) -> Result<Vec<f32>, GlmMoeDsaF32KernelError> {
        let mut output = Vec::<f32>::new();
        let mut saw_rank = false;
        for rank in &self.ranks {
            let contribution = rank.dense_mlp_partial_output(layer_id, hidden)?;
            if output.is_empty() {
                output.resize(contribution.len(), 0.0);
            } else if output.len() != contribution.len() {
                return Err(GlmMoeDsaF32KernelError::HiddenSizeMismatch {
                    tensor_name: format!("model.layers.{layer_id}.mlp.down_proj.weight"),
                    expected: output.len(),
                    actual: contribution.len(),
                });
            }
            for (output, contribution) in output.iter_mut().zip(contribution) {
                *output += contribution;
            }
            saw_rank = true;
        }

        if saw_rank {
            Ok(output)
        } else {
            Err(GlmMoeDsaF32KernelError::MissingTensor {
                tensor_name: format!("model.layers.{layer_id}.mlp.gate_proj.weight"),
            })
        }
    }

    pub fn feed_forward_layer_output(
        &self,
        layer_id: usize,
        attention_output: &[f32],
        residual: Option<&[f32]>,
    ) -> Result<GlmMoeDsaF32LayerOutput, GlmMoeDsaF32KernelError> {
        let feed_forward_kind = self.feed_forward_kind(layer_id)?;
        let mut residual_out = attention_output.to_vec();
        if let Some(residual) = residual {
            if residual.len() != attention_output.len() {
                return Err(GlmMoeDsaF32KernelError::HiddenSizeMismatch {
                    tensor_name: format!("model.layers.{layer_id}.post_attention_layernorm.weight"),
                    expected: attention_output.len(),
                    actual: residual.len(),
                });
            }
            for (output, residual) in residual_out.iter_mut().zip(residual) {
                *output += residual;
            }
        }

        let mut mlp_input = residual_out.clone();
        let norm_name = format!("model.layers.{layer_id}.post_attention_layernorm.weight");
        let norm_weight = self.layer_norm_weight(&norm_name)?;
        apply_rms_norm(&norm_name, &mut mlp_input, norm_weight, self.rms_norm_eps)?;

        let hidden_states = match feed_forward_kind {
            GlmMoeDsaF32FeedForwardKind::Dense => self.dense_mlp_output(layer_id, &mlp_input)?,
            GlmMoeDsaF32FeedForwardKind::Moe => self.moe_mlp_output(layer_id, &mlp_input)?,
        };

        Ok(GlmMoeDsaF32LayerOutput::new(hidden_states, residual_out))
    }

    pub fn moe_mlp_output(
        &self,
        layer_id: usize,
        hidden: &[f32],
    ) -> Result<Vec<f32>, GlmMoeDsaF32KernelError> {
        let router_logits = self.moe_router_logits(layer_id, hidden)?;
        let topk = topk_softmax_weights(
            &router_logits,
            self.num_experts_per_tok,
            self.norm_topk_prob,
            layer_id,
        )?;

        let mut output = Vec::<f32>::new();
        for (expert_id, route_weight) in topk {
            let expert_output = self.routed_expert_output(layer_id, expert_id, hidden)?;
            if output.is_empty() {
                output.resize(expert_output.len(), 0.0);
            } else if output.len() != expert_output.len() {
                return Err(GlmMoeDsaF32KernelError::HiddenSizeMismatch {
                    tensor_name: format!(
                        "model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight"
                    ),
                    expected: output.len(),
                    actual: expert_output.len(),
                });
            }
            let scale = route_weight * self.routed_scaling_factor;
            for (output, expert_value) in output.iter_mut().zip(expert_output) {
                *output += scale * expert_value;
            }
        }

        Ok(output)
    }

    fn moe_router_logits(
        &self,
        layer_id: usize,
        hidden: &[f32],
    ) -> Result<Vec<f32>, GlmMoeDsaF32KernelError> {
        for rank in &self.ranks {
            if let Some(logits) = rank.moe_router_logits(layer_id, hidden)? {
                return Ok(logits);
            }
        }

        Err(GlmMoeDsaF32KernelError::MissingTensor {
            tensor_name: format!("model.layers.{layer_id}.mlp.gate.weight"),
        })
    }

    fn routed_expert_output(
        &self,
        layer_id: usize,
        expert_id: usize,
        hidden: &[f32],
    ) -> Result<Vec<f32>, GlmMoeDsaF32KernelError> {
        let mut output = Vec::<f32>::new();
        let mut saw_rank = false;
        for rank in &self.ranks {
            let contribution = rank.routed_expert_partial_output(layer_id, expert_id, hidden)?;
            if output.is_empty() {
                output.resize(contribution.len(), 0.0);
            } else if output.len() != contribution.len() {
                return Err(GlmMoeDsaF32KernelError::HiddenSizeMismatch {
                    tensor_name: format!(
                        "model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight"
                    ),
                    expected: output.len(),
                    actual: contribution.len(),
                });
            }
            for (output, contribution) in output.iter_mut().zip(contribution) {
                *output += contribution;
            }
            saw_rank = true;
        }

        if saw_rank {
            Ok(output)
        } else {
            Err(GlmMoeDsaF32KernelError::MissingTensor {
                tensor_name: format!(
                    "model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight"
                ),
            })
        }
    }

    fn embedding_lm_head_logits_for_token(
        &self,
        token_id: u32,
    ) -> Result<Vec<f32>, GlmMoeDsaF32KernelError> {
        let mut hidden = self.token_embedding(token_id)?;
        let norm_weight = self.final_norm_weight()?;
        apply_rms_norm(
            "model.norm.weight",
            &mut hidden,
            norm_weight,
            self.rms_norm_eps,
        )?;

        self.lm_head_logits_for_hidden(&hidden)
    }

    fn transformer_lm_head_logits_for_request(
        &self,
        input_ids: &[u32],
        positions: &[usize],
    ) -> Result<Vec<f32>, GlmMoeDsaF32KernelError> {
        if input_ids.is_empty() {
            return Err(GlmMoeDsaF32KernelError::TokenCountMismatch {
                tensor_name: "model.embed_tokens.weight".to_string(),
                expected: 1,
                actual: 0,
            });
        }
        if positions.len() != input_ids.len() {
            return Err(GlmMoeDsaF32KernelError::TokenCountMismatch {
                tensor_name: "model.layers.0.self_attn.rotary_emb".to_string(),
                expected: input_ids.len(),
                actual: positions.len(),
            });
        }

        let mut hidden_states = input_ids
            .iter()
            .copied()
            .map(|token_id| self.token_embedding(token_id))
            .collect::<Result<Vec<_>, _>>()?;
        let mut residuals = None::<Vec<Vec<f32>>>;

        for layer_id in 0..self.layer_count {
            let layer_output = self.transformer_layer_output_with_positions(
                layer_id,
                &hidden_states,
                positions,
                residuals.as_deref(),
            )?;
            hidden_states = layer_output
                .iter()
                .map(|output| output.hidden_states().to_vec())
                .collect();
            residuals = Some(
                layer_output
                    .iter()
                    .map(|output| output.residual().to_vec())
                    .collect(),
            );
        }

        let last_token_index = hidden_states.len() - 1;
        let mut hidden = hidden_states[last_token_index].clone();
        if let Some(residuals) = residuals {
            let residual = &residuals[last_token_index];
            if residual.len() != hidden.len() {
                return Err(GlmMoeDsaF32KernelError::HiddenSizeMismatch {
                    tensor_name: "model.norm.weight".to_string(),
                    expected: hidden.len(),
                    actual: residual.len(),
                });
            }
            for (hidden, residual) in hidden.iter_mut().zip(residual) {
                *hidden += residual;
            }
        }

        let norm_weight = self.final_norm_weight()?;
        apply_rms_norm(
            "model.norm.weight",
            &mut hidden,
            norm_weight,
            self.rms_norm_eps,
        )?;
        self.lm_head_logits_for_hidden(&hidden)
    }

    fn transformer_lm_head_logits_for_cached_request(
        &self,
        input_ids: &[u32],
        positions: &[usize],
        out_cache_pages: &[CachePageId],
        sequence_cache_pages: &[CachePageId],
        kv_cache: &mut GlmMoeDsaF32KvPageStore,
    ) -> Result<Vec<f32>, GlmMoeDsaF32KernelError> {
        if input_ids.is_empty() {
            return Err(GlmMoeDsaF32KernelError::TokenCountMismatch {
                tensor_name: "model.embed_tokens.weight".to_string(),
                expected: 1,
                actual: 0,
            });
        }
        if positions.len() != input_ids.len() {
            return Err(GlmMoeDsaF32KernelError::TokenCountMismatch {
                tensor_name: "model.layers.0.self_attn.rotary_emb".to_string(),
                expected: input_ids.len(),
                actual: positions.len(),
            });
        }
        if out_cache_pages.len() != input_ids.len() {
            return Err(GlmMoeDsaF32KernelError::TokenCountMismatch {
                tensor_name: "model.layers.0.self_attn.kv_cache".to_string(),
                expected: input_ids.len(),
                actual: out_cache_pages.len(),
            });
        }

        let mut hidden_states = input_ids
            .iter()
            .copied()
            .map(|token_id| self.token_embedding(token_id))
            .collect::<Result<Vec<_>, _>>()?;
        let mut residuals = None::<Vec<Vec<f32>>>;

        for layer_id in 0..self.layer_count {
            let layer_output = self.transformer_layer_output_with_kv_cache(
                layer_id,
                &hidden_states,
                GlmMoeDsaF32KvAttentionBatch {
                    positions,
                    out_cache_pages,
                    sequence_cache_pages,
                },
                kv_cache,
                residuals.as_deref(),
            )?;
            hidden_states = layer_output
                .iter()
                .map(|output| output.hidden_states().to_vec())
                .collect();
            residuals = Some(
                layer_output
                    .iter()
                    .map(|output| output.residual().to_vec())
                    .collect(),
            );
        }

        let last_token_index = hidden_states.len() - 1;
        let mut hidden = hidden_states[last_token_index].clone();
        if let Some(residuals) = residuals {
            let residual = &residuals[last_token_index];
            if residual.len() != hidden.len() {
                return Err(GlmMoeDsaF32KernelError::HiddenSizeMismatch {
                    tensor_name: "model.norm.weight".to_string(),
                    expected: hidden.len(),
                    actual: residual.len(),
                });
            }
            for (hidden, residual) in hidden.iter_mut().zip(residual) {
                *hidden += residual;
            }
        }

        let norm_weight = self.final_norm_weight()?;
        apply_rms_norm(
            "model.norm.weight",
            &mut hidden,
            norm_weight,
            self.rms_norm_eps,
        )?;
        self.lm_head_logits_for_hidden(&hidden)
    }

    fn lm_head_logits_for_hidden(
        &self,
        hidden: &[f32],
    ) -> Result<Vec<f32>, GlmMoeDsaF32KernelError> {
        let mut logits = Vec::<Option<f32>>::new();
        for rank in &self.ranks {
            for (token_id, logit) in rank.lm_head_partial_logits(hidden)? {
                if logits.len() <= token_id {
                    logits.resize(token_id + 1, None);
                }
                logits[token_id] = Some(logit);
            }
        }

        logits
            .into_iter()
            .enumerate()
            .map(|(token_id, logit)| {
                logit.ok_or(GlmMoeDsaF32KernelError::MissingVocabularyLogit { token_id })
            })
            .collect()
    }

    fn token_embedding(&self, token_id: u32) -> Result<Vec<f32>, GlmMoeDsaF32KernelError> {
        let token_id =
            usize::try_from(token_id).map_err(|_| GlmMoeDsaF32KernelError::TokenIdOutOfRange {
                token_id: token_id.to_string(),
            })?;
        let mut saw_embedding = false;
        for rank in &self.ranks {
            let Some(embedding) = rank.tensor_shard("model.embed_tokens.weight") else {
                continue;
            };
            saw_embedding = true;
            if let Some(values) = embedding.vocab_parallel_row(token_id)? {
                return Ok(values.to_vec());
            }
        }

        if saw_embedding {
            Err(GlmMoeDsaF32KernelError::TokenOutOfVocabulary { token_id })
        } else {
            Err(GlmMoeDsaF32KernelError::MissingTensor {
                tensor_name: "model.embed_tokens.weight".to_string(),
            })
        }
    }

    fn final_norm_weight(&self) -> Result<&[f32], GlmMoeDsaF32KernelError> {
        self.layer_norm_weight("model.norm.weight")
    }

    fn prepare_attention_input(
        &self,
        layer_id: usize,
        hidden: &[f32],
        residual: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Vec<f32>), GlmMoeDsaF32KernelError> {
        let norm_name = format!("model.layers.{layer_id}.input_layernorm.weight");
        let mut residual_out = hidden.to_vec();
        if let Some(residual) = residual {
            if residual.len() != hidden.len() {
                return Err(GlmMoeDsaF32KernelError::HiddenSizeMismatch {
                    tensor_name: norm_name,
                    expected: hidden.len(),
                    actual: residual.len(),
                });
            }
            for (output, residual) in residual_out.iter_mut().zip(residual) {
                *output += residual;
            }
        }

        let mut normalized = residual_out.clone();
        let norm_weight = self.layer_norm_weight(&norm_name)?;
        apply_rms_norm(&norm_name, &mut normalized, norm_weight, self.rms_norm_eps)?;
        Ok((normalized, residual_out))
    }

    fn replicated_projection_with_norm(
        &self,
        projection_name: &str,
        norm_name: &str,
        hidden: &[f32],
    ) -> Result<Vec<f32>, GlmMoeDsaF32KernelError> {
        for rank in &self.ranks {
            if let Some(projection) = rank.tensor_shard(projection_name) {
                let mut output = projection.output_parallel_matvec(hidden)?;
                let norm_weight = self.layer_norm_weight(norm_name)?;
                apply_rms_norm(norm_name, &mut output, norm_weight, self.rms_norm_eps)?;
                return Ok(output);
            }
        }

        Err(GlmMoeDsaF32KernelError::MissingTensor {
            tensor_name: projection_name.to_string(),
        })
    }

    fn replicated_projection(
        &self,
        projection_name: &str,
        hidden: &[f32],
    ) -> Result<Vec<f32>, GlmMoeDsaF32KernelError> {
        for rank in &self.ranks {
            if let Some(projection) = rank.tensor_shard(projection_name) {
                return projection.output_parallel_matvec(hidden);
            }
        }

        Err(GlmMoeDsaF32KernelError::MissingTensor {
            tensor_name: projection_name.to_string(),
        })
    }

    fn column_parallel_projection(
        &self,
        tensor_name: &str,
        hidden: &[f32],
    ) -> Result<Vec<f32>, GlmMoeDsaF32KernelError> {
        let mut output = Vec::<Option<f32>>::new();
        let mut saw_tensor = false;
        for rank in &self.ranks {
            let Some(projection) = rank.tensor_shard(tensor_name) else {
                continue;
            };
            saw_tensor = true;
            for (output_index, value) in projection.output_parallel_indexed_matvec(hidden)? {
                if output.len() <= output_index {
                    output.resize(output_index + 1, None);
                }
                output[output_index] = Some(value);
            }
        }

        if !saw_tensor {
            return Err(GlmMoeDsaF32KernelError::MissingTensor {
                tensor_name: tensor_name.to_string(),
            });
        }

        output
            .into_iter()
            .enumerate()
            .map(|(output_index, value)| {
                value.ok_or_else(|| GlmMoeDsaF32KernelError::MissingTensorShardOutput {
                    tensor_name: tensor_name.to_string(),
                    output_index,
                })
            })
            .collect()
    }

    fn row_parallel_projection(
        &self,
        tensor_name: &str,
        hidden: &[f32],
    ) -> Result<Vec<f32>, GlmMoeDsaF32KernelError> {
        let mut output = Vec::<f32>::new();
        let mut saw_tensor = false;
        for rank in &self.ranks {
            let Some(projection) = rank.tensor_shard(tensor_name) else {
                continue;
            };
            saw_tensor = true;
            let hidden_slice = match projection.selection() {
                GlmMoeDsaTensorShardSelection::Full => hidden,
                GlmMoeDsaTensorShardSelection::Slice { axis: 1, range } => {
                    if range.end > hidden.len() {
                        return Err(GlmMoeDsaF32KernelError::HiddenSizeMismatch {
                            tensor_name: tensor_name.to_string(),
                            expected: range.end,
                            actual: hidden.len(),
                        });
                    }
                    &hidden[range.clone()]
                }
                GlmMoeDsaTensorShardSelection::Slice { axis, .. } => {
                    return Err(GlmMoeDsaF32KernelError::TensorSelectionMismatch {
                        tensor_name: tensor_name.to_string(),
                        expected_axis: 1,
                        actual_axis: *axis,
                    });
                }
            };
            let contribution = projection.row_parallel_matvec_contribution(hidden_slice)?;
            if output.is_empty() {
                output.resize(contribution.len(), 0.0);
            } else if output.len() != contribution.len() {
                return Err(GlmMoeDsaF32KernelError::HiddenSizeMismatch {
                    tensor_name: tensor_name.to_string(),
                    expected: output.len(),
                    actual: contribution.len(),
                });
            }
            for (output, contribution) in output.iter_mut().zip(contribution) {
                *output += contribution;
            }
        }

        if saw_tensor {
            Ok(output)
        } else {
            Err(GlmMoeDsaF32KernelError::MissingTensor {
                tensor_name: tensor_name.to_string(),
            })
        }
    }

    fn causal_attention_values(
        &self,
        layer_id: usize,
        projections: &[GlmMoeDsaF32AttentionProjectionOutput],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>, GlmMoeDsaF32KernelError> {
        let shape = self.attention_shape;
        let query_width = shape.query_width();
        let kv_width = shape.kv_width();
        let value_width = shape.value_width();
        if positions.len() != projections.len() {
            return Err(GlmMoeDsaF32KernelError::TokenCountMismatch {
                tensor_name: format!("model.layers.{layer_id}.self_attn.rotary_emb"),
                expected: projections.len(),
                actual: positions.len(),
            });
        }
        if !shape.qk_rope_head_dim.is_multiple_of(2) {
            return Err(GlmMoeDsaF32KernelError::HiddenSizeMismatch {
                tensor_name: format!("model.layers.{layer_id}.self_attn.rotary_emb"),
                expected: shape.qk_rope_head_dim + 1,
                actual: shape.qk_rope_head_dim,
            });
        }

        for projection in projections {
            if projection.q().len() != query_width {
                return Err(GlmMoeDsaF32KernelError::HiddenSizeMismatch {
                    tensor_name: format!("model.layers.{layer_id}.self_attn.q_b_proj.weight"),
                    expected: query_width,
                    actual: projection.q().len(),
                });
            }
            if projection.kv().len() != kv_width {
                return Err(GlmMoeDsaF32KernelError::HiddenSizeMismatch {
                    tensor_name: format!("model.layers.{layer_id}.self_attn.kv_b_proj.weight"),
                    expected: kv_width,
                    actual: projection.kv().len(),
                });
            }
            if projection.k_rope().len() != shape.qk_rope_head_dim {
                return Err(GlmMoeDsaF32KernelError::HiddenSizeMismatch {
                    tensor_name: format!(
                        "model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight"
                    ),
                    expected: shape.qk_rope_head_dim,
                    actual: projection.k_rope().len(),
                });
            }
        }

        let scale = (shape.qk_head_dim() as f32).sqrt().recip();
        let mut outputs = Vec::with_capacity(projections.len());
        for token_index in 0..projections.len() {
            let mut token_output = vec![0.0; value_width];
            for head_index in 0..shape.num_attention_heads {
                let scores = (0..=token_index)
                    .map(|key_index| {
                        self.attention_score_for_head(
                            &projections[token_index],
                            &projections[key_index],
                            head_index,
                            positions[token_index],
                            positions[key_index],
                        ) * scale
                    })
                    .collect::<Vec<_>>();
                let weights = softmax(&scores);

                for (key_index, weight) in (0..=token_index).zip(weights) {
                    let value = value_for_head(&projections[key_index], shape, head_index);
                    let output_start = head_index * shape.v_head_dim;
                    for (output, value) in token_output
                        [output_start..output_start + shape.v_head_dim]
                        .iter_mut()
                        .zip(value)
                    {
                        *output += weight * value;
                    }
                }
            }
            outputs.push(token_output);
        }

        Ok(outputs)
    }

    fn attention_score_for_head(
        &self,
        query_projection: &GlmMoeDsaF32AttentionProjectionOutput,
        key_projection: &GlmMoeDsaF32AttentionProjectionOutput,
        head_index: usize,
        query_position: usize,
        key_position: usize,
    ) -> f32 {
        let shape = self.attention_shape;
        let query_start = head_index * shape.qk_head_dim();
        let query_nope = &query_projection.q()[query_start..query_start + shape.qk_nope_head_dim];
        let key = key_for_head(key_projection, shape, head_index);
        let nope_score = query_nope
            .iter()
            .zip(key)
            .map(|(query, key)| query * key)
            .sum::<f32>();

        let query_rope_start = query_start + shape.qk_nope_head_dim;
        let query_rope =
            &query_projection.q()[query_rope_start..query_rope_start + shape.qk_rope_head_dim];
        nope_score
            + rotary_dot_product(
                query_rope,
                key_projection.k_rope(),
                query_position,
                key_position,
                self.rope_theta,
            )
    }

    fn layer_norm_weight(&self, tensor_name: &str) -> Result<&[f32], GlmMoeDsaF32KernelError> {
        for rank in &self.ranks {
            if let Some(norm) = rank.tensor_shard(tensor_name) {
                return norm.vector_values();
            }
        }

        Err(GlmMoeDsaF32KernelError::MissingTensor {
            tensor_name: tensor_name.to_string(),
        })
    }

    fn feed_forward_kind(
        &self,
        layer_id: usize,
    ) -> Result<GlmMoeDsaF32FeedForwardKind, GlmMoeDsaF32KernelError> {
        self.feed_forward_kinds.get(layer_id).copied().ok_or(
            GlmMoeDsaF32KernelError::LayerOutOfBounds {
                layer_id,
                layer_count: self.layer_count,
            },
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlmMoeDsaF32TensorRank {
    tensor_parallel_rank: usize,
    shards: Vec<GlmMoeDsaF32TensorShard>,
}

impl GlmMoeDsaF32TensorRank {
    pub fn tensor_parallel_rank(&self) -> usize {
        self.tensor_parallel_rank
    }

    pub fn shards(&self) -> &[GlmMoeDsaF32TensorShard] {
        &self.shards
    }

    pub fn tensor_shard(&self, tensor_name: &str) -> Option<&GlmMoeDsaF32TensorShard> {
        self.shards
            .iter()
            .find(|shard| shard.tensor_name() == tensor_name)
    }

    pub fn lm_head_partial_logits(
        &self,
        hidden: &[f32],
    ) -> Result<Vec<(usize, f32)>, GlmMoeDsaF32KernelError> {
        let lm_head = self.tensor_shard("lm_head.weight").ok_or_else(|| {
            GlmMoeDsaF32KernelError::MissingTensor {
                tensor_name: "lm_head.weight".to_string(),
            }
        })?;
        lm_head.vocab_parallel_matvec(hidden)
    }

    fn dense_mlp_partial_output(
        &self,
        layer_id: usize,
        hidden: &[f32],
    ) -> Result<Vec<f32>, GlmMoeDsaF32KernelError> {
        let gate_name = format!("model.layers.{layer_id}.mlp.gate_proj.weight");
        let up_name = format!("model.layers.{layer_id}.mlp.up_proj.weight");
        let down_name = format!("model.layers.{layer_id}.mlp.down_proj.weight");
        let gate = self.tensor_shard(&gate_name).ok_or_else(|| {
            GlmMoeDsaF32KernelError::MissingTensor {
                tensor_name: gate_name.clone(),
            }
        })?;
        let up =
            self.tensor_shard(&up_name)
                .ok_or_else(|| GlmMoeDsaF32KernelError::MissingTensor {
                    tensor_name: up_name.clone(),
                })?;
        let down = self.tensor_shard(&down_name).ok_or_else(|| {
            GlmMoeDsaF32KernelError::MissingTensor {
                tensor_name: down_name.clone(),
            }
        })?;

        let gate_values = gate.output_parallel_matvec(hidden)?;
        let up_values = up.output_parallel_matvec(hidden)?;
        if gate_values.len() != up_values.len() {
            return Err(GlmMoeDsaF32KernelError::HiddenSizeMismatch {
                tensor_name: up_name,
                expected: gate_values.len(),
                actual: up_values.len(),
            });
        }
        let activated = gate_values
            .iter()
            .zip(&up_values)
            .map(|(gate, up)| silu(*gate) * up)
            .collect::<Vec<_>>();
        down.row_parallel_matvec_contribution(&activated)
    }

    fn moe_router_logits(
        &self,
        layer_id: usize,
        hidden: &[f32],
    ) -> Result<Option<Vec<f32>>, GlmMoeDsaF32KernelError> {
        let gate_name = format!("model.layers.{layer_id}.mlp.gate.weight");
        let Some(gate) = self.tensor_shard(&gate_name) else {
            return Ok(None);
        };

        gate.output_parallel_matvec(hidden).map(Some)
    }

    fn routed_expert_partial_output(
        &self,
        layer_id: usize,
        expert_id: usize,
        hidden: &[f32],
    ) -> Result<Vec<f32>, GlmMoeDsaF32KernelError> {
        let gate_name = format!("model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight");
        let up_name = format!("model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight");
        let down_name = format!("model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight");
        let gate = self.tensor_shard(&gate_name).ok_or_else(|| {
            GlmMoeDsaF32KernelError::MissingTensor {
                tensor_name: gate_name.clone(),
            }
        })?;
        let up =
            self.tensor_shard(&up_name)
                .ok_or_else(|| GlmMoeDsaF32KernelError::MissingTensor {
                    tensor_name: up_name.clone(),
                })?;
        let down = self.tensor_shard(&down_name).ok_or_else(|| {
            GlmMoeDsaF32KernelError::MissingTensor {
                tensor_name: down_name.clone(),
            }
        })?;

        let gate_values = gate.output_parallel_matvec(hidden)?;
        let up_values = up.output_parallel_matvec(hidden)?;
        if gate_values.len() != up_values.len() {
            return Err(GlmMoeDsaF32KernelError::HiddenSizeMismatch {
                tensor_name: up_name,
                expected: gate_values.len(),
                actual: up_values.len(),
            });
        }
        let activated = gate_values
            .iter()
            .zip(&up_values)
            .map(|(gate, up)| silu(*gate) * up)
            .collect::<Vec<_>>();
        down.row_parallel_matvec_contribution(&activated)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlmMoeDsaF32TensorShard {
    tensor_name: String,
    shape: Vec<usize>,
    selection: GlmMoeDsaTensorShardSelection,
    values: Vec<f32>,
}

impl GlmMoeDsaF32TensorShard {
    fn from_loaded_shard(
        shard: &GlmMoeDsaLoadedTensorShard,
    ) -> Result<Self, SafetensorsTensorDecodeError> {
        Ok(Self {
            tensor_name: shard.tensor_name().to_string(),
            shape: shard.shape().to_vec(),
            selection: shard.selection().clone(),
            values: shard.decode_f32_values()?,
        })
    }

    pub fn tensor_name(&self) -> &str {
        &self.tensor_name
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn selection(&self) -> &GlmMoeDsaTensorShardSelection {
        &self.selection
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    fn vocab_parallel_matvec(
        &self,
        hidden: &[f32],
    ) -> Result<Vec<(usize, f32)>, GlmMoeDsaF32KernelError> {
        let [rows, columns] = self.shape.as_slice() else {
            return Err(GlmMoeDsaF32KernelError::TensorRankMismatch {
                tensor_name: self.tensor_name.clone(),
                expected_rank: 2,
                shape: self.shape.clone(),
            });
        };
        if *columns != hidden.len() {
            return Err(GlmMoeDsaF32KernelError::HiddenSizeMismatch {
                tensor_name: self.tensor_name.clone(),
                expected: *columns,
                actual: hidden.len(),
            });
        }

        let global_row_start = match &self.selection {
            GlmMoeDsaTensorShardSelection::Full => 0,
            GlmMoeDsaTensorShardSelection::Slice { axis: 0, range } => range.start,
            GlmMoeDsaTensorShardSelection::Slice { axis, .. } => {
                return Err(GlmMoeDsaF32KernelError::TensorSelectionMismatch {
                    tensor_name: self.tensor_name.clone(),
                    expected_axis: 0,
                    actual_axis: *axis,
                });
            }
        };

        Ok(self
            .values
            .chunks_exact(*columns)
            .take(*rows)
            .enumerate()
            .map(|(row_index, row)| {
                let logit = row
                    .iter()
                    .zip(hidden)
                    .map(|(weight, value)| weight * value)
                    .sum();
                (global_row_start + row_index, logit)
            })
            .collect())
    }

    fn output_parallel_matvec(&self, hidden: &[f32]) -> Result<Vec<f32>, GlmMoeDsaF32KernelError> {
        let [rows, columns] = self.shape.as_slice() else {
            return Err(GlmMoeDsaF32KernelError::TensorRankMismatch {
                tensor_name: self.tensor_name.clone(),
                expected_rank: 2,
                shape: self.shape.clone(),
            });
        };
        if *columns != hidden.len() {
            return Err(GlmMoeDsaF32KernelError::HiddenSizeMismatch {
                tensor_name: self.tensor_name.clone(),
                expected: *columns,
                actual: hidden.len(),
            });
        }
        match &self.selection {
            GlmMoeDsaTensorShardSelection::Full
            | GlmMoeDsaTensorShardSelection::Slice { axis: 0, .. } => {}
            GlmMoeDsaTensorShardSelection::Slice { axis, .. } => {
                return Err(GlmMoeDsaF32KernelError::TensorSelectionMismatch {
                    tensor_name: self.tensor_name.clone(),
                    expected_axis: 0,
                    actual_axis: *axis,
                });
            }
        }

        Ok(self
            .values
            .chunks_exact(*columns)
            .take(*rows)
            .map(|row| {
                row.iter()
                    .zip(hidden)
                    .map(|(weight, value)| weight * value)
                    .sum()
            })
            .collect())
    }

    fn output_parallel_indexed_matvec(
        &self,
        hidden: &[f32],
    ) -> Result<Vec<(usize, f32)>, GlmMoeDsaF32KernelError> {
        let [rows, columns] = self.shape.as_slice() else {
            return Err(GlmMoeDsaF32KernelError::TensorRankMismatch {
                tensor_name: self.tensor_name.clone(),
                expected_rank: 2,
                shape: self.shape.clone(),
            });
        };
        if *columns != hidden.len() {
            return Err(GlmMoeDsaF32KernelError::HiddenSizeMismatch {
                tensor_name: self.tensor_name.clone(),
                expected: *columns,
                actual: hidden.len(),
            });
        }

        let global_row_start = match &self.selection {
            GlmMoeDsaTensorShardSelection::Full => 0,
            GlmMoeDsaTensorShardSelection::Slice { axis: 0, range } => range.start,
            GlmMoeDsaTensorShardSelection::Slice { axis, .. } => {
                return Err(GlmMoeDsaF32KernelError::TensorSelectionMismatch {
                    tensor_name: self.tensor_name.clone(),
                    expected_axis: 0,
                    actual_axis: *axis,
                });
            }
        };

        Ok(self
            .values
            .chunks_exact(*columns)
            .take(*rows)
            .enumerate()
            .map(|(row_index, row)| {
                let value = row
                    .iter()
                    .zip(hidden)
                    .map(|(weight, value)| weight * value)
                    .sum();
                (global_row_start + row_index, value)
            })
            .collect())
    }

    fn row_parallel_matvec_contribution(
        &self,
        hidden: &[f32],
    ) -> Result<Vec<f32>, GlmMoeDsaF32KernelError> {
        let [rows, columns] = self.shape.as_slice() else {
            return Err(GlmMoeDsaF32KernelError::TensorRankMismatch {
                tensor_name: self.tensor_name.clone(),
                expected_rank: 2,
                shape: self.shape.clone(),
            });
        };
        if *columns != hidden.len() {
            return Err(GlmMoeDsaF32KernelError::HiddenSizeMismatch {
                tensor_name: self.tensor_name.clone(),
                expected: *columns,
                actual: hidden.len(),
            });
        }
        match &self.selection {
            GlmMoeDsaTensorShardSelection::Full
            | GlmMoeDsaTensorShardSelection::Slice { axis: 1, .. } => {}
            GlmMoeDsaTensorShardSelection::Slice { axis, .. } => {
                return Err(GlmMoeDsaF32KernelError::TensorSelectionMismatch {
                    tensor_name: self.tensor_name.clone(),
                    expected_axis: 1,
                    actual_axis: *axis,
                });
            }
        }

        Ok(self
            .values
            .chunks_exact(*columns)
            .take(*rows)
            .map(|row| {
                row.iter()
                    .zip(hidden)
                    .map(|(weight, value)| weight * value)
                    .sum()
            })
            .collect())
    }

    fn vocab_parallel_row(
        &self,
        global_row: usize,
    ) -> Result<Option<&[f32]>, GlmMoeDsaF32KernelError> {
        let [rows, columns] = self.shape.as_slice() else {
            return Err(GlmMoeDsaF32KernelError::TensorRankMismatch {
                tensor_name: self.tensor_name.clone(),
                expected_rank: 2,
                shape: self.shape.clone(),
            });
        };

        let global_row_start = match &self.selection {
            GlmMoeDsaTensorShardSelection::Full => 0,
            GlmMoeDsaTensorShardSelection::Slice { axis: 0, range } => range.start,
            GlmMoeDsaTensorShardSelection::Slice { axis, .. } => {
                return Err(GlmMoeDsaF32KernelError::TensorSelectionMismatch {
                    tensor_name: self.tensor_name.clone(),
                    expected_axis: 0,
                    actual_axis: *axis,
                });
            }
        };
        let Some(local_row) = global_row.checked_sub(global_row_start) else {
            return Ok(None);
        };
        if local_row >= *rows {
            return Ok(None);
        }

        let start = local_row * columns;
        let end = start + columns;
        Ok(Some(&self.values[start..end]))
    }

    fn vector_values(&self) -> Result<&[f32], GlmMoeDsaF32KernelError> {
        let [length] = self.shape.as_slice() else {
            return Err(GlmMoeDsaF32KernelError::TensorRankMismatch {
                tensor_name: self.tensor_name.clone(),
                expected_rank: 1,
                shape: self.shape.clone(),
            });
        };
        if self.values.len() != *length {
            return Err(GlmMoeDsaF32KernelError::TensorDataLengthMismatch {
                tensor_name: self.tensor_name.clone(),
                expected: *length,
                actual: self.values.len(),
            });
        }
        Ok(&self.values)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GlmMoeDsaRuntimeError {
    UnsupportedModelType { model_type: Option<String> },
    MissingKvCacheLayout,
    ModelArtifact(ModelArtifactError),
    PdConfig(PdConfigError),
}

impl fmt::Display for GlmMoeDsaRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedModelType { model_type } => {
                write!(formatter, "unsupported GLM-DSA model_type {model_type:?}")
            }
            Self::MissingKvCacheLayout => {
                formatter.write_str("GLM-DSA model config does not define a KV cache layout")
            }
            Self::ModelArtifact(error) => write!(formatter, "{error}"),
            Self::PdConfig(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for GlmMoeDsaRuntimeError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GlmMoeDsaF32KernelError {
    MissingTensor {
        tensor_name: String,
    },
    TokenIdOutOfRange {
        token_id: String,
    },
    TokenOutOfVocabulary {
        token_id: usize,
    },
    MissingVocabularyLogit {
        token_id: usize,
    },
    MissingTensorShardOutput {
        tensor_name: String,
        output_index: usize,
    },
    TensorRankMismatch {
        tensor_name: String,
        expected_rank: usize,
        shape: Vec<usize>,
    },
    TensorDataLengthMismatch {
        tensor_name: String,
        expected: usize,
        actual: usize,
    },
    HiddenSizeMismatch {
        tensor_name: String,
        expected: usize,
        actual: usize,
    },
    TokenCountMismatch {
        tensor_name: String,
        expected: usize,
        actual: usize,
    },
    MissingKvCachePage {
        layer_id: usize,
        cache_page: usize,
    },
    TensorSelectionMismatch {
        tensor_name: String,
        expected_axis: usize,
        actual_axis: usize,
    },
    InvalidTopK {
        layer_id: usize,
        top_k: usize,
        expert_count: usize,
    },
    LayerOutOfBounds {
        layer_id: usize,
        layer_count: usize,
    },
    Runtime {
        message: String,
    },
}

impl fmt::Display for GlmMoeDsaF32KernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingTensor { tensor_name } => {
                write!(formatter, "missing f32 tensor {tensor_name}")
            }
            Self::TokenIdOutOfRange { token_id } => {
                write!(formatter, "token id {token_id} does not fit usize")
            }
            Self::TokenOutOfVocabulary { token_id } => {
                write!(formatter, "token id {token_id} is outside GLM vocabulary")
            }
            Self::MissingVocabularyLogit { token_id } => {
                write!(
                    formatter,
                    "missing GLM vocab-parallel logit for token {token_id}"
                )
            }
            Self::MissingTensorShardOutput {
                tensor_name,
                output_index,
            } => write!(
                formatter,
                "missing GLM tensor-parallel output index {output_index} for tensor {tensor_name}"
            ),
            Self::TensorRankMismatch {
                tensor_name,
                expected_rank,
                shape,
            } => write!(
                formatter,
                "f32 tensor {tensor_name} expected rank {expected_rank} but shape is {shape:?}"
            ),
            Self::TensorDataLengthMismatch {
                tensor_name,
                expected,
                actual,
            } => write!(
                formatter,
                "f32 tensor {tensor_name} expected {expected} value(s) but decoded {actual}"
            ),
            Self::HiddenSizeMismatch {
                tensor_name,
                expected,
                actual,
            } => write!(
                formatter,
                "f32 tensor {tensor_name} expected hidden size {expected} but got {actual}"
            ),
            Self::TokenCountMismatch {
                tensor_name,
                expected,
                actual,
            } => write!(
                formatter,
                "f32 tensor {tensor_name} expected {expected} token(s) but got {actual}"
            ),
            Self::MissingKvCachePage {
                layer_id,
                cache_page,
            } => write!(
                formatter,
                "missing GLM layer {layer_id} KV cache page {cache_page}"
            ),
            Self::TensorSelectionMismatch {
                tensor_name,
                expected_axis,
                actual_axis,
            } => write!(
                formatter,
                "f32 tensor {tensor_name} expected selection axis {expected_axis} but got {actual_axis}"
            ),
            Self::InvalidTopK {
                layer_id,
                top_k,
                expert_count,
            } => write!(
                formatter,
                "GLM layer {layer_id} requested top_k {top_k} for {expert_count} routed expert(s)"
            ),
            Self::LayerOutOfBounds {
                layer_id,
                layer_count,
            } => write!(
                formatter,
                "GLM layer id {layer_id} is outside loaded layer count {layer_count}"
            ),
            Self::Runtime { message } => write!(formatter, "{message}"),
        }
    }
}

impl std::error::Error for GlmMoeDsaF32KernelError {}

fn apply_rms_norm(
    tensor_name: &str,
    hidden: &mut [f32],
    weight: &[f32],
    eps: f32,
) -> Result<(), GlmMoeDsaF32KernelError> {
    if hidden.len() != weight.len() {
        return Err(GlmMoeDsaF32KernelError::HiddenSizeMismatch {
            tensor_name: tensor_name.to_string(),
            expected: weight.len(),
            actual: hidden.len(),
        });
    }

    let mean_square = hidden.iter().map(|value| value * value).sum::<f32>() / hidden.len() as f32;
    let inv_rms = (mean_square + eps).sqrt().recip();
    for (value, weight) in hidden.iter_mut().zip(weight) {
        *value *= inv_rms * weight;
    }
    Ok(())
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn key_for_head(
    projection: &GlmMoeDsaF32AttentionProjectionOutput,
    shape: GlmMoeDsaAttentionShape,
    head_index: usize,
) -> &[f32] {
    let head_width = shape.qk_nope_head_dim + shape.v_head_dim;
    let head_start = head_index * head_width;
    &projection.kv()[head_start..head_start + shape.qk_nope_head_dim]
}

fn value_for_head(
    projection: &GlmMoeDsaF32AttentionProjectionOutput,
    shape: GlmMoeDsaAttentionShape,
    head_index: usize,
) -> &[f32] {
    let head_width = shape.qk_nope_head_dim + shape.v_head_dim;
    let head_start = head_index * head_width;
    let value_start = head_start + shape.qk_nope_head_dim;
    &projection.kv()[value_start..value_start + shape.v_head_dim]
}

fn rotary_dot_product(
    query: &[f32],
    key: &[f32],
    query_position: usize,
    key_position: usize,
    rope_theta: f32,
) -> f32 {
    if query.is_empty() {
        return 0.0;
    }

    let query = apply_rotary_embedding(query, query_position, rope_theta);
    let key = apply_rotary_embedding(key, key_position, rope_theta);
    query.iter().zip(key).map(|(query, key)| query * key).sum()
}

fn apply_rotary_embedding(values: &[f32], position: usize, rope_theta: f32) -> Vec<f32> {
    let rotary_dim = values.len();
    let mut output = vec![0.0; rotary_dim];
    for pair_index in 0..rotary_dim / 2 {
        let even_index = pair_index * 2;
        let odd_index = even_index + 1;
        let inv_freq = rope_theta.powf(-(even_index as f32) / rotary_dim as f32);
        let angle = position as f32 * inv_freq;
        let cos = angle.cos();
        let sin = angle.sin();
        let even = values[even_index];
        let odd = values[odd_index];
        output[even_index] = even * cos - odd * sin;
        output[odd_index] = odd * cos + even * sin;
    }
    output
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |max, value| max.max(value));
    let exp_values = logits
        .iter()
        .map(|logit| (*logit - max_logit).exp())
        .collect::<Vec<_>>();
    let exp_sum = exp_values.iter().sum::<f32>();
    exp_values
        .into_iter()
        .map(|exp_value| exp_value / exp_sum)
        .collect()
}

fn topk_softmax_weights(
    logits: &[f32],
    top_k: usize,
    renormalize: bool,
    layer_id: usize,
) -> Result<Vec<(usize, f32)>, GlmMoeDsaF32KernelError> {
    if top_k == 0 || top_k > logits.len() {
        return Err(GlmMoeDsaF32KernelError::InvalidTopK {
            layer_id,
            top_k,
            expert_count: logits.len(),
        });
    }

    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |max, value| max.max(value));
    let exp_values = logits
        .iter()
        .map(|logit| (*logit - max_logit).exp())
        .collect::<Vec<_>>();
    let exp_sum = exp_values.iter().sum::<f32>();
    let mut weights = exp_values
        .into_iter()
        .enumerate()
        .map(|(expert_id, exp_value)| (expert_id, exp_value / exp_sum))
        .collect::<Vec<_>>();
    weights.sort_by(|(left_id, left), (right_id, right)| {
        right.total_cmp(left).then_with(|| left_id.cmp(right_id))
    });
    weights.truncate(top_k);

    if renormalize {
        let selected_sum = weights.iter().map(|(_, weight)| *weight).sum::<f32>();
        for (_, weight) in &mut weights {
            *weight /= selected_sum;
        }
    }

    Ok(weights)
}

impl From<ModelArtifactError> for GlmMoeDsaRuntimeError {
    fn from(error: ModelArtifactError) -> Self {
        Self::ModelArtifact(error)
    }
}

impl From<PdConfigError> for GlmMoeDsaRuntimeError {
    fn from(error: PdConfigError) -> Self {
        Self::PdConfig(error)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GlmMoeDsaTensorShardPlanError {
    RankOutOfBounds {
        tensor_parallel_rank: usize,
        tensor_parallel_size: usize,
    },
    TensorAxisOutOfBounds {
        tensor_name: String,
        axis: usize,
        shape: Vec<usize>,
    },
    TensorDimensionNotDivisible {
        tensor_name: String,
        axis: usize,
        dimension: usize,
        tensor_parallel_size: usize,
    },
}

impl fmt::Display for GlmMoeDsaTensorShardPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankOutOfBounds {
                tensor_parallel_rank,
                tensor_parallel_size,
            } => write!(
                formatter,
                "tensor parallel rank {tensor_parallel_rank} must be < tensor parallel size {tensor_parallel_size}"
            ),
            Self::TensorAxisOutOfBounds {
                tensor_name,
                axis,
                shape,
            } => write!(
                formatter,
                "tensor {tensor_name} placement axis {axis} is out of bounds for shape {shape:?}"
            ),
            Self::TensorDimensionNotDivisible {
                tensor_name,
                axis,
                dimension,
                tensor_parallel_size,
            } => write!(
                formatter,
                "tensor {tensor_name} axis {axis} dimension {dimension} is not divisible by tensor parallel size {tensor_parallel_size}"
            ),
        }
    }
}

impl std::error::Error for GlmMoeDsaTensorShardPlanError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GlmMoeDsaTensorShardLoadError {
    ShardPlan(GlmMoeDsaTensorShardPlanError),
    ModelArtifact(ModelArtifactError),
    TensorRankTooHigh {
        tensor_name: String,
        shape: Vec<usize>,
    },
    TensorAxisOutOfBounds {
        tensor_name: String,
        axis: usize,
        shape: Vec<usize>,
    },
    TensorDataLengthMismatch {
        tensor_name: String,
        expected_byte_len: usize,
        actual_byte_len: usize,
    },
    TensorShardByteLengthOverflow {
        tensor_name: String,
    },
    TensorShardOffsetOverflow {
        tensor_name: String,
    },
}

impl fmt::Display for GlmMoeDsaTensorShardLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShardPlan(error) => write!(formatter, "{error}"),
            Self::ModelArtifact(error) => write!(formatter, "{error}"),
            Self::TensorRankTooHigh { tensor_name, shape } => write!(
                formatter,
                "tensor {tensor_name} with shape {shape:?} has unsupported rank for shard loading"
            ),
            Self::TensorAxisOutOfBounds {
                tensor_name,
                axis,
                shape,
            } => write!(
                formatter,
                "tensor {tensor_name} shard axis {axis} is out of bounds for shape {shape:?}"
            ),
            Self::TensorDataLengthMismatch {
                tensor_name,
                expected_byte_len,
                actual_byte_len,
            } => write!(
                formatter,
                "tensor {tensor_name} expected {expected_byte_len} payload bytes but loaded {actual_byte_len}"
            ),
            Self::TensorShardByteLengthOverflow { tensor_name } => {
                write!(
                    formatter,
                    "tensor {tensor_name} shard byte length overflowed"
                )
            }
            Self::TensorShardOffsetOverflow { tensor_name } => {
                write!(
                    formatter,
                    "tensor {tensor_name} shard byte offset overflowed"
                )
            }
        }
    }
}

impl std::error::Error for GlmMoeDsaTensorShardLoadError {}

impl From<GlmMoeDsaTensorShardPlanError> for GlmMoeDsaTensorShardLoadError {
    fn from(error: GlmMoeDsaTensorShardPlanError) -> Self {
        Self::ShardPlan(error)
    }
}

impl From<ModelArtifactError> for GlmMoeDsaTensorShardLoadError {
    fn from(error: ModelArtifactError) -> Self {
        Self::ModelArtifact(error)
    }
}

fn shard_selection_for_entry(
    entry: &GlmMoeDsaTensorPlacement,
    tensor_parallel_size: usize,
    tensor_parallel_rank: usize,
) -> Result<GlmMoeDsaTensorShardSelection, GlmMoeDsaTensorShardPlanError> {
    if tensor_parallel_size == 1 {
        return Ok(GlmMoeDsaTensorShardSelection::Full);
    }

    let axis = match entry.kind() {
        GlmMoeDsaTensorPlacementKind::Replicated => {
            return Ok(GlmMoeDsaTensorShardSelection::Full);
        }
        GlmMoeDsaTensorPlacementKind::VocabParallel { axis }
        | GlmMoeDsaTensorPlacementKind::ColumnParallel { axis }
        | GlmMoeDsaTensorPlacementKind::RowParallel { axis } => axis,
    };
    let shape = entry.tensor().shape();
    let Some(&dimension) = shape.get(axis) else {
        return Err(GlmMoeDsaTensorShardPlanError::TensorAxisOutOfBounds {
            tensor_name: entry.tensor().tensor_name().to_string(),
            axis,
            shape: shape.to_vec(),
        });
    };
    if dimension % tensor_parallel_size != 0 {
        return Err(GlmMoeDsaTensorShardPlanError::TensorDimensionNotDivisible {
            tensor_name: entry.tensor().tensor_name().to_string(),
            axis,
            dimension,
            tensor_parallel_size,
        });
    }
    let shard_size = dimension / tensor_parallel_size;
    let start = tensor_parallel_rank * shard_size;

    Ok(GlmMoeDsaTensorShardSelection::Slice {
        axis,
        range: start..start + shard_size,
    })
}

fn materialize_tensor_shard(
    tensor_name: &str,
    shape: &[usize],
    dtype_byte_width: usize,
    bytes: &[u8],
    selection: &GlmMoeDsaTensorShardSelection,
) -> Result<(Vec<usize>, Vec<u8>), GlmMoeDsaTensorShardLoadError> {
    let expected_byte_len = shape
        .iter()
        .try_fold(dtype_byte_width, |accumulator, dimension| {
            accumulator.checked_mul(*dimension)
        })
        .ok_or_else(
            || GlmMoeDsaTensorShardLoadError::TensorShardByteLengthOverflow {
                tensor_name: tensor_name.to_string(),
            },
        )?;
    if expected_byte_len != bytes.len() {
        return Err(GlmMoeDsaTensorShardLoadError::TensorDataLengthMismatch {
            tensor_name: tensor_name.to_string(),
            expected_byte_len,
            actual_byte_len: bytes.len(),
        });
    }

    match selection {
        GlmMoeDsaTensorShardSelection::Full => Ok((shape.to_vec(), bytes.to_vec())),
        GlmMoeDsaTensorShardSelection::Slice { axis, range } => {
            materialize_tensor_slice(tensor_name, shape, dtype_byte_width, bytes, *axis, range)
        }
    }
}

fn materialize_tensor_slice(
    tensor_name: &str,
    shape: &[usize],
    dtype_byte_width: usize,
    bytes: &[u8],
    axis: usize,
    range: &Range<usize>,
) -> Result<(Vec<usize>, Vec<u8>), GlmMoeDsaTensorShardLoadError> {
    if axis >= shape.len() {
        return Err(GlmMoeDsaTensorShardLoadError::TensorAxisOutOfBounds {
            tensor_name: tensor_name.to_string(),
            axis,
            shape: shape.to_vec(),
        });
    }
    if shape.len() > 2 {
        return Err(GlmMoeDsaTensorShardLoadError::TensorRankTooHigh {
            tensor_name: tensor_name.to_string(),
            shape: shape.to_vec(),
        });
    }

    let mut shard_shape = shape.to_vec();
    shard_shape[axis] = range.len();

    match shape {
        [_] => {
            let start = range.start.checked_mul(dtype_byte_width).ok_or_else(|| {
                GlmMoeDsaTensorShardLoadError::TensorShardOffsetOverflow {
                    tensor_name: tensor_name.to_string(),
                }
            })?;
            let end = range.end.checked_mul(dtype_byte_width).ok_or_else(|| {
                GlmMoeDsaTensorShardLoadError::TensorShardOffsetOverflow {
                    tensor_name: tensor_name.to_string(),
                }
            })?;
            Ok((shard_shape, bytes[start..end].to_vec()))
        }
        [rows, columns] if axis == 0 => {
            let row_byte_len = columns.checked_mul(dtype_byte_width).ok_or_else(|| {
                GlmMoeDsaTensorShardLoadError::TensorShardByteLengthOverflow {
                    tensor_name: tensor_name.to_string(),
                }
            })?;
            let start = range.start.checked_mul(row_byte_len).ok_or_else(|| {
                GlmMoeDsaTensorShardLoadError::TensorShardOffsetOverflow {
                    tensor_name: tensor_name.to_string(),
                }
            })?;
            let end = range.end.checked_mul(row_byte_len).ok_or_else(|| {
                GlmMoeDsaTensorShardLoadError::TensorShardOffsetOverflow {
                    tensor_name: tensor_name.to_string(),
                }
            })?;
            debug_assert!(range.end <= *rows);
            Ok((shard_shape, bytes[start..end].to_vec()))
        }
        [rows, columns] => {
            let row_byte_len = columns.checked_mul(dtype_byte_width).ok_or_else(|| {
                GlmMoeDsaTensorShardLoadError::TensorShardByteLengthOverflow {
                    tensor_name: tensor_name.to_string(),
                }
            })?;
            let column_start = range.start.checked_mul(dtype_byte_width).ok_or_else(|| {
                GlmMoeDsaTensorShardLoadError::TensorShardOffsetOverflow {
                    tensor_name: tensor_name.to_string(),
                }
            })?;
            let column_end = range.end.checked_mul(dtype_byte_width).ok_or_else(|| {
                GlmMoeDsaTensorShardLoadError::TensorShardOffsetOverflow {
                    tensor_name: tensor_name.to_string(),
                }
            })?;
            let shard_row_byte_len = column_end.checked_sub(column_start).ok_or_else(|| {
                GlmMoeDsaTensorShardLoadError::TensorShardOffsetOverflow {
                    tensor_name: tensor_name.to_string(),
                }
            })?;
            let shard_byte_len = rows.checked_mul(shard_row_byte_len).ok_or_else(|| {
                GlmMoeDsaTensorShardLoadError::TensorShardByteLengthOverflow {
                    tensor_name: tensor_name.to_string(),
                }
            })?;
            let mut shard_bytes = Vec::with_capacity(shard_byte_len);
            for row in 0..*rows {
                let row_start = row.checked_mul(row_byte_len).ok_or_else(|| {
                    GlmMoeDsaTensorShardLoadError::TensorShardOffsetOverflow {
                        tensor_name: tensor_name.to_string(),
                    }
                })?;
                let start = row_start.checked_add(column_start).ok_or_else(|| {
                    GlmMoeDsaTensorShardLoadError::TensorShardOffsetOverflow {
                        tensor_name: tensor_name.to_string(),
                    }
                })?;
                let end = row_start.checked_add(column_end).ok_or_else(|| {
                    GlmMoeDsaTensorShardLoadError::TensorShardOffsetOverflow {
                        tensor_name: tensor_name.to_string(),
                    }
                })?;
                shard_bytes.extend_from_slice(&bytes[start..end]);
            }

            Ok((shard_shape, shard_bytes))
        }
        [] => Ok((shard_shape, Vec::new())),
        _ => unreachable!("tensor ranks above 2 are rejected before slicing"),
    }
}

fn push_placement(
    entries: &mut Vec<GlmMoeDsaTensorPlacement>,
    tensor: &GlmTensorDescriptor,
    kind: GlmMoeDsaTensorPlacementKind,
) {
    entries.push(GlmMoeDsaTensorPlacement {
        tensor: tensor.clone(),
        kind,
    });
}
