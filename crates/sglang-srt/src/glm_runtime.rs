use std::fmt;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::PathBuf;

use crate::model_artifacts::{
    GlmMoeDsaLayerCheckpointWeights, GlmMoeDsaLayerFeedForwardCheckpointWeights,
    GlmMoeDsaModelCheckpointWeights, GlmMoeDsaModelTensorSpan, LocalModelArtifacts,
    LocalModelCheckpointCatalog, ModelArtifactError, SafetensorsLayerTensorSpan,
    SafetensorsTensorData, SafetensorsTensorMetadata, SafetensorsTensorSpan,
};
use crate::transfer::{KvCacheModelLayout, PdConfigError};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmMoeDsaRuntime {
    model_path: PathBuf,
    root_tensors: GlmMoeDsaRootTensorDescriptors,
    layers: Vec<GlmMoeDsaLayerTensorDescriptors>,
    kv_cache_layout: KvCacheModelLayout,
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
