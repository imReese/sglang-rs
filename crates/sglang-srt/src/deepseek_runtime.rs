use std::fmt;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::PathBuf;

use crate::cache::CachePageId;
use crate::model_artifacts::{
    DeepSeekLayerCheckpointWeights, DeepSeekLayerFeedForwardCheckpointWeights,
    DeepSeekModelCheckpointWeights, DeepSeekModelTensorSpan, LocalModelArtifacts,
    LocalModelCheckpointCatalog, ModelArtifactError, SafetensorsLayerTensorSpan,
    SafetensorsTensorData, SafetensorsTensorDecodeError, SafetensorsTensorMetadata,
    SafetensorsTensorSpan,
};
use crate::model_executor::ModelWorkerBatch;
use crate::scheduler::ForwardMode;
use crate::transfer::{KvCacheModelLayout, PdConfigError};
use crate::types::{BootstrapRoom, RequestId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4Runtime {
    model_path: PathBuf,
    root_tensors: DeepSeekV4RootTensorDescriptors,
    layers: Vec<DeepSeekV4LayerTensorDescriptors>,
    kv_cache_layout: KvCacheModelLayout,
}

impl DeepSeekV4Runtime {
    pub fn from_local_model_artifacts(
        artifacts: &LocalModelArtifacts,
    ) -> Result<Self, DeepSeekRuntimeError> {
        if artifacts.config().model_type.as_deref() != Some("deepseek_v4") {
            return Err(DeepSeekRuntimeError::UnsupportedModelType {
                model_type: artifacts.config().model_type.clone(),
            });
        }

        let checkpoint = LocalModelCheckpointCatalog::from_local_model_artifacts(artifacts)?;
        let weights = checkpoint.deepseek_model_weights()?;
        let kv_cache_layout = KvCacheModelLayout::from_hf_config(artifacts.config())?
            .ok_or(DeepSeekRuntimeError::MissingKvCacheLayout)?;

        Ok(Self {
            model_path: artifacts.model_path().to_path_buf(),
            root_tensors: DeepSeekV4RootTensorDescriptors::from_checkpoint(&weights),
            layers: weights
                .layers()
                .iter()
                .map(DeepSeekV4LayerTensorDescriptors::from_checkpoint)
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

    pub fn root_tensors(&self) -> &DeepSeekV4RootTensorDescriptors {
        &self.root_tensors
    }

    pub fn layers(&self) -> &[DeepSeekV4LayerTensorDescriptors] {
        &self.layers
    }

    pub fn kv_cache_layout(&self) -> KvCacheModelLayout {
        self.kv_cache_layout
    }

    pub fn forward_plan(&self, batch: &ModelWorkerBatch) -> DeepSeekV4ForwardPlan {
        DeepSeekV4ForwardPlan::from_model_worker_batch(batch)
    }

    pub fn tensor_parallel_placement_plan(
        &self,
        tensor_parallel_size: usize,
    ) -> DeepSeekV4TensorPlacementPlan {
        DeepSeekV4TensorPlacementPlan::from_runtime(self, tensor_parallel_size)
    }

    pub fn load_tensor_parallel_shards(
        &self,
        tensor_parallel_size: usize,
    ) -> Result<DeepSeekV4LoadedTensorParallelRuntime, DeepSeekV4TensorShardLoadError> {
        DeepSeekV4LoadedTensorParallelRuntime::from_runtime(self, tensor_parallel_size)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4RootTensorDescriptors {
    token_embeddings: DeepSeekTensorDescriptor,
    final_norm: DeepSeekTensorDescriptor,
    hc_head_fn: DeepSeekTensorDescriptor,
    hc_head_base: DeepSeekTensorDescriptor,
    hc_head_scale: DeepSeekTensorDescriptor,
    lm_head: DeepSeekTensorDescriptor,
}

impl DeepSeekV4RootTensorDescriptors {
    fn from_checkpoint(weights: &DeepSeekModelCheckpointWeights<'_>) -> Self {
        Self {
            token_embeddings: DeepSeekTensorDescriptor::from_model_tensor(
                weights.token_embeddings(),
            ),
            final_norm: DeepSeekTensorDescriptor::from_model_tensor(weights.final_norm()),
            hc_head_fn: DeepSeekTensorDescriptor::from_model_tensor(weights.hc_head_fn()),
            hc_head_base: DeepSeekTensorDescriptor::from_model_tensor(weights.hc_head_base()),
            hc_head_scale: DeepSeekTensorDescriptor::from_model_tensor(weights.hc_head_scale()),
            lm_head: DeepSeekTensorDescriptor::from_model_tensor(weights.lm_head()),
        }
    }

    pub fn token_embeddings(&self) -> &DeepSeekTensorDescriptor {
        &self.token_embeddings
    }

    pub fn final_norm(&self) -> &DeepSeekTensorDescriptor {
        &self.final_norm
    }

    pub fn hc_head_fn(&self) -> &DeepSeekTensorDescriptor {
        &self.hc_head_fn
    }

    pub fn hc_head_base(&self) -> &DeepSeekTensorDescriptor {
        &self.hc_head_base
    }

    pub fn hc_head_scale(&self) -> &DeepSeekTensorDescriptor {
        &self.hc_head_scale
    }

    pub fn lm_head(&self) -> &DeepSeekTensorDescriptor {
        &self.lm_head
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4LayerTensorDescriptors {
    layer_id: usize,
    wq_a: DeepSeekTensorDescriptor,
    wq_b: DeepSeekTensorDescriptor,
    wkv: DeepSeekTensorDescriptor,
    q_norm: DeepSeekTensorDescriptor,
    kv_norm: DeepSeekTensorDescriptor,
    wo_a: DeepSeekTensorDescriptor,
    wo_b: DeepSeekTensorDescriptor,
    input_layernorm: DeepSeekTensorDescriptor,
    post_attention_layernorm: DeepSeekTensorDescriptor,
    hc_attn_fn: DeepSeekTensorDescriptor,
    hc_attn_base: DeepSeekTensorDescriptor,
    hc_attn_scale: DeepSeekTensorDescriptor,
    hc_ffn_fn: DeepSeekTensorDescriptor,
    hc_ffn_base: DeepSeekTensorDescriptor,
    hc_ffn_scale: DeepSeekTensorDescriptor,
    feed_forward: DeepSeekV4FeedForwardTensorDescriptors,
}

impl DeepSeekV4LayerTensorDescriptors {
    fn from_checkpoint(weights: &DeepSeekLayerCheckpointWeights<'_>) -> Self {
        Self {
            layer_id: weights.layer_id(),
            wq_a: DeepSeekTensorDescriptor::from_layer_tensor(weights.wq_a()),
            wq_b: DeepSeekTensorDescriptor::from_layer_tensor(weights.wq_b()),
            wkv: DeepSeekTensorDescriptor::from_layer_tensor(weights.wkv()),
            q_norm: DeepSeekTensorDescriptor::from_layer_tensor(weights.q_norm()),
            kv_norm: DeepSeekTensorDescriptor::from_layer_tensor(weights.kv_norm()),
            wo_a: DeepSeekTensorDescriptor::from_layer_tensor(weights.wo_a()),
            wo_b: DeepSeekTensorDescriptor::from_layer_tensor(weights.wo_b()),
            input_layernorm: DeepSeekTensorDescriptor::from_layer_tensor(weights.input_layernorm()),
            post_attention_layernorm: DeepSeekTensorDescriptor::from_layer_tensor(
                weights.post_attention_layernorm(),
            ),
            hc_attn_fn: DeepSeekTensorDescriptor::from_layer_tensor(weights.hc_attn_fn()),
            hc_attn_base: DeepSeekTensorDescriptor::from_layer_tensor(weights.hc_attn_base()),
            hc_attn_scale: DeepSeekTensorDescriptor::from_layer_tensor(weights.hc_attn_scale()),
            hc_ffn_fn: DeepSeekTensorDescriptor::from_layer_tensor(weights.hc_ffn_fn()),
            hc_ffn_base: DeepSeekTensorDescriptor::from_layer_tensor(weights.hc_ffn_base()),
            hc_ffn_scale: DeepSeekTensorDescriptor::from_layer_tensor(weights.hc_ffn_scale()),
            feed_forward: DeepSeekV4FeedForwardTensorDescriptors::from_checkpoint(
                weights.feed_forward(),
            ),
        }
    }

    pub fn layer_id(&self) -> usize {
        self.layer_id
    }

    pub fn feed_forward(&self) -> &DeepSeekV4FeedForwardTensorDescriptors {
        &self.feed_forward
    }

    fn push_tensor_placements(&self, entries: &mut Vec<DeepSeekV4TensorPlacement>) {
        push_placement(
            entries,
            &self.wq_a,
            DeepSeekV4TensorPlacementKind::Replicated,
        );
        push_placement(
            entries,
            &self.wq_b,
            DeepSeekV4TensorPlacementKind::ColumnParallel { axis: 0 },
        );
        push_placement(
            entries,
            &self.wkv,
            DeepSeekV4TensorPlacementKind::Replicated,
        );
        push_placement(
            entries,
            &self.q_norm,
            DeepSeekV4TensorPlacementKind::Replicated,
        );
        push_placement(
            entries,
            &self.kv_norm,
            DeepSeekV4TensorPlacementKind::Replicated,
        );
        push_placement(
            entries,
            &self.wo_a,
            DeepSeekV4TensorPlacementKind::ColumnParallel { axis: 0 },
        );
        push_placement(
            entries,
            &self.wo_b,
            DeepSeekV4TensorPlacementKind::RowParallel { axis: 1 },
        );
        push_placement(
            entries,
            &self.input_layernorm,
            DeepSeekV4TensorPlacementKind::Replicated,
        );
        push_placement(
            entries,
            &self.post_attention_layernorm,
            DeepSeekV4TensorPlacementKind::Replicated,
        );
        push_placement(
            entries,
            &self.hc_attn_fn,
            DeepSeekV4TensorPlacementKind::Replicated,
        );
        push_placement(
            entries,
            &self.hc_attn_base,
            DeepSeekV4TensorPlacementKind::Replicated,
        );
        push_placement(
            entries,
            &self.hc_attn_scale,
            DeepSeekV4TensorPlacementKind::Replicated,
        );
        push_placement(
            entries,
            &self.hc_ffn_fn,
            DeepSeekV4TensorPlacementKind::Replicated,
        );
        push_placement(
            entries,
            &self.hc_ffn_base,
            DeepSeekV4TensorPlacementKind::Replicated,
        );
        push_placement(
            entries,
            &self.hc_ffn_scale,
            DeepSeekV4TensorPlacementKind::Replicated,
        );
        self.feed_forward.push_tensor_placements(entries);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeepSeekV4FeedForwardTensorDescriptors {
    Dense {
        gate_up_proj: DeepSeekTensorDescriptor,
        down_proj: DeepSeekTensorDescriptor,
    },
    Moe {
        gate: DeepSeekTensorDescriptor,
        routed_experts: Vec<DeepSeekV4RoutedExpertTensorDescriptors>,
    },
}

impl DeepSeekV4FeedForwardTensorDescriptors {
    fn from_checkpoint(weights: &DeepSeekLayerFeedForwardCheckpointWeights<'_>) -> Self {
        match weights {
            DeepSeekLayerFeedForwardCheckpointWeights::Dense {
                gate_up_proj,
                down_proj,
            } => Self::Dense {
                gate_up_proj: DeepSeekTensorDescriptor::from_layer_tensor(gate_up_proj),
                down_proj: DeepSeekTensorDescriptor::from_layer_tensor(down_proj),
            },
            DeepSeekLayerFeedForwardCheckpointWeights::Moe {
                gate,
                routed_experts,
            } => Self::Moe {
                gate: DeepSeekTensorDescriptor::from_layer_tensor(gate),
                routed_experts: routed_experts
                    .groups()
                    .map(|group| DeepSeekV4RoutedExpertTensorDescriptors {
                        expert_id: group.expert_id,
                        gate: DeepSeekTensorDescriptor::from_tensor_span(
                            format!(
                                "model.layers.{}.ffn.experts.{}.w1.weight",
                                group.layer_id, group.expert_id
                            ),
                            &group.gate,
                        ),
                        up: DeepSeekTensorDescriptor::from_tensor_span(
                            format!(
                                "model.layers.{}.ffn.experts.{}.w3.weight",
                                group.layer_id, group.expert_id
                            ),
                            &group.up,
                        ),
                        down: DeepSeekTensorDescriptor::from_tensor_span(
                            format!(
                                "model.layers.{}.ffn.experts.{}.w2.weight",
                                group.layer_id, group.expert_id
                            ),
                            &group.down,
                        ),
                    })
                    .collect(),
            },
        }
    }

    fn push_tensor_placements(&self, entries: &mut Vec<DeepSeekV4TensorPlacement>) {
        match self {
            Self::Dense {
                gate_up_proj,
                down_proj,
            } => {
                push_placement(
                    entries,
                    gate_up_proj,
                    DeepSeekV4TensorPlacementKind::ColumnParallel { axis: 0 },
                );
                push_placement(
                    entries,
                    down_proj,
                    DeepSeekV4TensorPlacementKind::RowParallel { axis: 1 },
                );
            }
            Self::Moe {
                gate,
                routed_experts,
            } => {
                push_placement(entries, gate, DeepSeekV4TensorPlacementKind::Replicated);
                for expert in routed_experts {
                    expert.push_tensor_placements(entries);
                }
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4RoutedExpertTensorDescriptors {
    expert_id: usize,
    gate: DeepSeekTensorDescriptor,
    up: DeepSeekTensorDescriptor,
    down: DeepSeekTensorDescriptor,
}

impl DeepSeekV4RoutedExpertTensorDescriptors {
    pub fn expert_id(&self) -> usize {
        self.expert_id
    }

    pub fn gate(&self) -> &DeepSeekTensorDescriptor {
        &self.gate
    }

    pub fn up(&self) -> &DeepSeekTensorDescriptor {
        &self.up
    }

    pub fn down(&self) -> &DeepSeekTensorDescriptor {
        &self.down
    }

    fn push_tensor_placements(&self, entries: &mut Vec<DeepSeekV4TensorPlacement>) {
        push_placement(
            entries,
            &self.gate,
            DeepSeekV4TensorPlacementKind::ColumnParallel { axis: 0 },
        );
        push_placement(
            entries,
            &self.up,
            DeepSeekV4TensorPlacementKind::ColumnParallel { axis: 0 },
        );
        push_placement(
            entries,
            &self.down,
            DeepSeekV4TensorPlacementKind::RowParallel { axis: 1 },
        );
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekTensorDescriptor {
    tensor_name: String,
    path: PathBuf,
    dtype: String,
    shape: Vec<usize>,
    absolute_byte_offset: u64,
    byte_len: usize,
}

impl DeepSeekTensorDescriptor {
    fn from_model_tensor(tensor: &DeepSeekModelTensorSpan) -> Self {
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

    fn read(&self) -> Result<SafetensorsTensorData, DeepSeekV4TensorShardLoadError> {
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
pub enum DeepSeekV4TensorPlacementKind {
    Replicated,
    VocabParallel { axis: usize },
    ColumnParallel { axis: usize },
    RowParallel { axis: usize },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4TensorPlacementPlan {
    tensor_parallel_size: usize,
    entries: Vec<DeepSeekV4TensorPlacement>,
}

impl DeepSeekV4TensorPlacementPlan {
    fn from_runtime(runtime: &DeepSeekV4Runtime, tensor_parallel_size: usize) -> Self {
        let mut entries = Vec::new();
        push_placement(
            &mut entries,
            runtime.root_tensors().token_embeddings(),
            DeepSeekV4TensorPlacementKind::VocabParallel { axis: 0 },
        );
        push_placement(
            &mut entries,
            runtime.root_tensors().final_norm(),
            DeepSeekV4TensorPlacementKind::Replicated,
        );
        push_placement(
            &mut entries,
            runtime.root_tensors().hc_head_fn(),
            DeepSeekV4TensorPlacementKind::Replicated,
        );
        push_placement(
            &mut entries,
            runtime.root_tensors().hc_head_base(),
            DeepSeekV4TensorPlacementKind::Replicated,
        );
        push_placement(
            &mut entries,
            runtime.root_tensors().hc_head_scale(),
            DeepSeekV4TensorPlacementKind::Replicated,
        );
        push_placement(
            &mut entries,
            runtime.root_tensors().lm_head(),
            DeepSeekV4TensorPlacementKind::VocabParallel { axis: 0 },
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

    pub fn entries(&self) -> &[DeepSeekV4TensorPlacement] {
        &self.entries
    }

    pub fn kind_for(&self, tensor_name: &str) -> Option<DeepSeekV4TensorPlacementKind> {
        self.entries
            .iter()
            .find(|entry| entry.tensor().tensor_name() == tensor_name)
            .map(DeepSeekV4TensorPlacement::kind)
    }

    pub fn rank_shard_plan(
        &self,
        tensor_parallel_rank: usize,
    ) -> Result<DeepSeekV4TensorRankShardPlan, DeepSeekV4TensorShardPlanError> {
        DeepSeekV4TensorRankShardPlan::from_placement_plan(self, tensor_parallel_rank)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4TensorPlacement {
    tensor: DeepSeekTensorDescriptor,
    kind: DeepSeekV4TensorPlacementKind,
}

impl DeepSeekV4TensorPlacement {
    pub fn tensor(&self) -> &DeepSeekTensorDescriptor {
        &self.tensor
    }

    pub fn kind(&self) -> DeepSeekV4TensorPlacementKind {
        self.kind
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeepSeekV4TensorShardSelection {
    Full,
    Slice { axis: usize, range: Range<usize> },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4TensorRankShardPlan {
    tensor_parallel_size: usize,
    tensor_parallel_rank: usize,
    shards: Vec<DeepSeekV4TensorShard>,
}

impl DeepSeekV4TensorRankShardPlan {
    fn from_placement_plan(
        plan: &DeepSeekV4TensorPlacementPlan,
        tensor_parallel_rank: usize,
    ) -> Result<Self, DeepSeekV4TensorShardPlanError> {
        if tensor_parallel_rank >= plan.tensor_parallel_size {
            return Err(DeepSeekV4TensorShardPlanError::RankOutOfBounds {
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
                Ok(DeepSeekV4TensorShard {
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

    pub fn shards(&self) -> &[DeepSeekV4TensorShard] {
        &self.shards
    }

    pub fn selection_for(&self, tensor_name: &str) -> Option<DeepSeekV4TensorShardSelection> {
        self.shards
            .iter()
            .find(|shard| shard.tensor().tensor_name() == tensor_name)
            .map(|shard| shard.selection().clone())
    }

    pub fn load_tensor_shard(
        &self,
        tensor_name: &str,
    ) -> Result<Option<DeepSeekV4LoadedTensorShard>, DeepSeekV4TensorShardLoadError> {
        let Some(shard) = self
            .shards
            .iter()
            .find(|shard| shard.tensor().tensor_name() == tensor_name)
        else {
            return Ok(None);
        };

        shard.load().map(Some)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4TensorShard {
    tensor: DeepSeekTensorDescriptor,
    kind: DeepSeekV4TensorPlacementKind,
    selection: DeepSeekV4TensorShardSelection,
}

impl DeepSeekV4TensorShard {
    pub fn tensor(&self) -> &DeepSeekTensorDescriptor {
        &self.tensor
    }

    pub fn kind(&self) -> DeepSeekV4TensorPlacementKind {
        self.kind
    }

    pub fn selection(&self) -> &DeepSeekV4TensorShardSelection {
        &self.selection
    }

    pub fn load(&self) -> Result<DeepSeekV4LoadedTensorShard, DeepSeekV4TensorShardLoadError> {
        let tensor_data = self.tensor.read()?;
        let (shape, bytes) = materialize_tensor_shard(
            self.tensor.tensor_name(),
            self.tensor.path(),
            self.tensor.shape(),
            tensor_data.dtype_byte_width(),
            &tensor_data.bytes,
            &self.selection,
        )?;

        Ok(DeepSeekV4LoadedTensorShard {
            tensor_name: self.tensor.tensor_name().to_string(),
            dtype: self.tensor.dtype().to_string(),
            shape,
            selection: self.selection.clone(),
            bytes,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4LoadedTensorShard {
    tensor_name: String,
    dtype: String,
    shape: Vec<usize>,
    selection: DeepSeekV4TensorShardSelection,
    bytes: Vec<u8>,
}

impl DeepSeekV4LoadedTensorShard {
    pub fn tensor_name(&self) -> &str {
        &self.tensor_name
    }

    pub fn dtype(&self) -> &str {
        &self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn selection(&self) -> &DeepSeekV4TensorShardSelection {
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
pub struct DeepSeekV4LoadedTensorParallelRuntime {
    runtime: DeepSeekV4Runtime,
    tensor_parallel_size: usize,
    ranks: Vec<DeepSeekV4LoadedTensorRank>,
}

impl DeepSeekV4LoadedTensorParallelRuntime {
    fn from_runtime(
        runtime: &DeepSeekV4Runtime,
        tensor_parallel_size: usize,
    ) -> Result<Self, DeepSeekV4TensorShardLoadError> {
        if tensor_parallel_size == 0 {
            return Err(DeepSeekV4TensorShardLoadError::ShardPlan(
                DeepSeekV4TensorShardPlanError::RankOutOfBounds {
                    tensor_parallel_rank: 0,
                    tensor_parallel_size,
                },
            ));
        }

        let placement_plan = runtime.tensor_parallel_placement_plan(tensor_parallel_size);
        let ranks = (0..tensor_parallel_size)
            .map(|rank| {
                let rank_plan = placement_plan.rank_shard_plan(rank)?;
                DeepSeekV4LoadedTensorRank::from_rank_shard_plan(&rank_plan)
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

    pub fn ranks(&self) -> &[DeepSeekV4LoadedTensorRank] {
        &self.ranks
    }

    pub fn rank(&self, tensor_parallel_rank: usize) -> Option<&DeepSeekV4LoadedTensorRank> {
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

    pub fn forward_plan(&self, batch: &ModelWorkerBatch) -> DeepSeekV4ForwardPlan {
        self.runtime.forward_plan(batch)
    }

    pub fn loaded_shard_count(&self) -> usize {
        self.ranks
            .iter()
            .map(DeepSeekV4LoadedTensorRank::shard_count)
            .sum()
    }

    pub fn loaded_byte_len(&self) -> usize {
        self.ranks
            .iter()
            .map(DeepSeekV4LoadedTensorRank::loaded_byte_len)
            .sum()
    }

    pub fn decode_f32_tensor_parallel_shards(
        &self,
    ) -> Result<DeepSeekV4F32TensorParallelRuntime, SafetensorsTensorDecodeError> {
        DeepSeekV4F32TensorParallelRuntime::from_loaded_runtime(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4LoadedTensorRank {
    tensor_parallel_rank: usize,
    shards: Vec<DeepSeekV4LoadedTensorShard>,
}

impl DeepSeekV4LoadedTensorRank {
    fn from_rank_shard_plan(
        rank_plan: &DeepSeekV4TensorRankShardPlan,
    ) -> Result<Self, DeepSeekV4TensorShardLoadError> {
        let shards = rank_plan
            .shards()
            .iter()
            .map(DeepSeekV4TensorShard::load)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            tensor_parallel_rank: rank_plan.tensor_parallel_rank(),
            shards,
        })
    }

    pub fn tensor_parallel_rank(&self) -> usize {
        self.tensor_parallel_rank
    }

    pub fn shards(&self) -> &[DeepSeekV4LoadedTensorShard] {
        &self.shards
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub fn loaded_byte_len(&self) -> usize {
        self.shards.iter().map(|shard| shard.bytes().len()).sum()
    }

    pub fn tensor_shard(&self, tensor_name: &str) -> Option<&DeepSeekV4LoadedTensorShard> {
        self.shards
            .iter()
            .find(|shard| shard.tensor_name() == tensor_name)
    }

    fn decode_f32_shards(&self) -> Result<DeepSeekV4F32TensorRank, SafetensorsTensorDecodeError> {
        let shards = self
            .shards
            .iter()
            .map(DeepSeekV4F32TensorShard::from_loaded_shard)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(DeepSeekV4F32TensorRank {
            tensor_parallel_rank: self.tensor_parallel_rank,
            shards,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeepSeekV4F32TensorParallelRuntime {
    tensor_parallel_size: usize,
    layer_count: usize,
    ranks: Vec<DeepSeekV4F32TensorRank>,
}

impl DeepSeekV4F32TensorParallelRuntime {
    fn from_loaded_runtime(
        runtime: &DeepSeekV4LoadedTensorParallelRuntime,
    ) -> Result<Self, SafetensorsTensorDecodeError> {
        let ranks = runtime
            .ranks()
            .iter()
            .map(DeepSeekV4LoadedTensorRank::decode_f32_shards)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            tensor_parallel_size: runtime.tensor_parallel_size(),
            layer_count: runtime.layer_count(),
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

    pub fn ranks(&self) -> &[DeepSeekV4F32TensorRank] {
        &self.ranks
    }

    pub fn rank(&self, tensor_parallel_rank: usize) -> Option<&DeepSeekV4F32TensorRank> {
        self.ranks
            .iter()
            .find(|rank| rank.tensor_parallel_rank() == tensor_parallel_rank)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeepSeekV4F32TensorRank {
    tensor_parallel_rank: usize,
    shards: Vec<DeepSeekV4F32TensorShard>,
}

impl DeepSeekV4F32TensorRank {
    pub fn tensor_parallel_rank(&self) -> usize {
        self.tensor_parallel_rank
    }

    pub fn shards(&self) -> &[DeepSeekV4F32TensorShard] {
        &self.shards
    }

    pub fn tensor_shard(&self, tensor_name: &str) -> Option<&DeepSeekV4F32TensorShard> {
        self.shards
            .iter()
            .find(|shard| shard.tensor_name() == tensor_name)
    }

    pub fn lm_head_partial_logits(
        &self,
        hidden: &[f32],
    ) -> Result<Vec<(usize, f32)>, DeepSeekV4F32KernelError> {
        let lm_head = self.tensor_shard("lm_head.weight").ok_or_else(|| {
            DeepSeekV4F32KernelError::MissingTensor {
                tensor_name: "lm_head.weight".to_string(),
            }
        })?;
        lm_head.vocab_parallel_matvec(hidden)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeepSeekV4F32TensorShard {
    tensor_name: String,
    shape: Vec<usize>,
    selection: DeepSeekV4TensorShardSelection,
    values: Vec<f32>,
}

impl DeepSeekV4F32TensorShard {
    fn from_loaded_shard(
        shard: &DeepSeekV4LoadedTensorShard,
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

    pub fn selection(&self) -> &DeepSeekV4TensorShardSelection {
        &self.selection
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    fn vocab_parallel_matvec(
        &self,
        hidden: &[f32],
    ) -> Result<Vec<(usize, f32)>, DeepSeekV4F32KernelError> {
        let [rows, columns] = self.shape.as_slice() else {
            return Err(DeepSeekV4F32KernelError::TensorRankMismatch {
                tensor_name: self.tensor_name.clone(),
                expected_rank: 2,
                shape: self.shape.clone(),
            });
        };
        if *columns != hidden.len() {
            return Err(DeepSeekV4F32KernelError::HiddenSizeMismatch {
                tensor_name: self.tensor_name.clone(),
                expected: *columns,
                actual: hidden.len(),
            });
        }

        let global_row_start = match &self.selection {
            DeepSeekV4TensorShardSelection::Full => 0,
            DeepSeekV4TensorShardSelection::Slice { axis: 0, range } => range.start,
            DeepSeekV4TensorShardSelection::Slice { axis, .. } => {
                return Err(DeepSeekV4F32KernelError::TensorSelectionMismatch {
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
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeepSeekV4TensorShardPlanError {
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

impl fmt::Display for DeepSeekV4TensorShardPlanError {
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

impl std::error::Error for DeepSeekV4TensorShardPlanError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeepSeekV4TensorShardLoadError {
    ShardPlan(DeepSeekV4TensorShardPlanError),
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

impl fmt::Display for DeepSeekV4TensorShardLoadError {
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

impl std::error::Error for DeepSeekV4TensorShardLoadError {}

impl From<DeepSeekV4TensorShardPlanError> for DeepSeekV4TensorShardLoadError {
    fn from(error: DeepSeekV4TensorShardPlanError) -> Self {
        Self::ShardPlan(error)
    }
}

impl From<ModelArtifactError> for DeepSeekV4TensorShardLoadError {
    fn from(error: ModelArtifactError) -> Self {
        Self::ModelArtifact(error)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeepSeekV4F32KernelError {
    MissingTensor {
        tensor_name: String,
    },
    TensorRankMismatch {
        tensor_name: String,
        expected_rank: usize,
        shape: Vec<usize>,
    },
    HiddenSizeMismatch {
        tensor_name: String,
        expected: usize,
        actual: usize,
    },
    TensorSelectionMismatch {
        tensor_name: String,
        expected_axis: usize,
        actual_axis: usize,
    },
}

impl fmt::Display for DeepSeekV4F32KernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingTensor { tensor_name } => {
                write!(formatter, "missing f32 tensor {tensor_name}")
            }
            Self::TensorRankMismatch {
                tensor_name,
                expected_rank,
                shape,
            } => write!(
                formatter,
                "f32 tensor {tensor_name} expected rank {expected_rank} but shape is {shape:?}"
            ),
            Self::HiddenSizeMismatch {
                tensor_name,
                expected,
                actual,
            } => write!(
                formatter,
                "f32 tensor {tensor_name} expected hidden size {expected} but got {actual}"
            ),
            Self::TensorSelectionMismatch {
                tensor_name,
                expected_axis,
                actual_axis,
            } => write!(
                formatter,
                "f32 tensor {tensor_name} expected selection axis {expected_axis} but got {actual_axis}"
            ),
        }
    }
}

impl std::error::Error for DeepSeekV4F32KernelError {}

fn shard_selection_for_entry(
    entry: &DeepSeekV4TensorPlacement,
    tensor_parallel_size: usize,
    tensor_parallel_rank: usize,
) -> Result<DeepSeekV4TensorShardSelection, DeepSeekV4TensorShardPlanError> {
    if tensor_parallel_size == 1 {
        return Ok(DeepSeekV4TensorShardSelection::Full);
    }

    let axis = match entry.kind() {
        DeepSeekV4TensorPlacementKind::Replicated => {
            return Ok(DeepSeekV4TensorShardSelection::Full);
        }
        DeepSeekV4TensorPlacementKind::VocabParallel { axis }
        | DeepSeekV4TensorPlacementKind::ColumnParallel { axis }
        | DeepSeekV4TensorPlacementKind::RowParallel { axis } => axis,
    };
    let shape = entry.tensor().shape();
    let Some(&dimension) = shape.get(axis) else {
        return Err(DeepSeekV4TensorShardPlanError::TensorAxisOutOfBounds {
            tensor_name: entry.tensor().tensor_name().to_string(),
            axis,
            shape: shape.to_vec(),
        });
    };
    if dimension % tensor_parallel_size != 0 {
        return Err(
            DeepSeekV4TensorShardPlanError::TensorDimensionNotDivisible {
                tensor_name: entry.tensor().tensor_name().to_string(),
                axis,
                dimension,
                tensor_parallel_size,
            },
        );
    }
    let shard_size = dimension / tensor_parallel_size;
    let start = tensor_parallel_rank * shard_size;

    Ok(DeepSeekV4TensorShardSelection::Slice {
        axis,
        range: start..start + shard_size,
    })
}

fn materialize_tensor_shard(
    tensor_name: &str,
    tensor_path: &PathBuf,
    shape: &[usize],
    dtype_byte_width: usize,
    bytes: &[u8],
    selection: &DeepSeekV4TensorShardSelection,
) -> Result<(Vec<usize>, Vec<u8>), DeepSeekV4TensorShardLoadError> {
    let expected_byte_len = shape
        .iter()
        .try_fold(dtype_byte_width, |accumulator, dimension| {
            accumulator.checked_mul(*dimension)
        })
        .ok_or_else(
            || DeepSeekV4TensorShardLoadError::TensorShardByteLengthOverflow {
                tensor_name: tensor_name.to_string(),
            },
        )?;
    if expected_byte_len != bytes.len() {
        return Err(DeepSeekV4TensorShardLoadError::TensorDataLengthMismatch {
            tensor_name: tensor_name.to_string(),
            expected_byte_len,
            actual_byte_len: bytes.len(),
        });
    }

    match selection {
        DeepSeekV4TensorShardSelection::Full => Ok((shape.to_vec(), bytes.to_vec())),
        DeepSeekV4TensorShardSelection::Slice { axis, range } => materialize_tensor_slice(
            tensor_name,
            tensor_path,
            shape,
            dtype_byte_width,
            bytes,
            *axis,
            range,
        ),
    }
}

fn materialize_tensor_slice(
    tensor_name: &str,
    _tensor_path: &PathBuf,
    shape: &[usize],
    dtype_byte_width: usize,
    bytes: &[u8],
    axis: usize,
    range: &Range<usize>,
) -> Result<(Vec<usize>, Vec<u8>), DeepSeekV4TensorShardLoadError> {
    if axis >= shape.len() {
        return Err(DeepSeekV4TensorShardLoadError::TensorAxisOutOfBounds {
            tensor_name: tensor_name.to_string(),
            axis,
            shape: shape.to_vec(),
        });
    }
    if shape.len() > 2 {
        return Err(DeepSeekV4TensorShardLoadError::TensorRankTooHigh {
            tensor_name: tensor_name.to_string(),
            shape: shape.to_vec(),
        });
    }

    let mut shard_shape = shape.to_vec();
    shard_shape[axis] = range.len();

    match shape {
        [_] => {
            let start = range.start.checked_mul(dtype_byte_width).ok_or_else(|| {
                DeepSeekV4TensorShardLoadError::TensorShardOffsetOverflow {
                    tensor_name: tensor_name.to_string(),
                }
            })?;
            let end = range.end.checked_mul(dtype_byte_width).ok_or_else(|| {
                DeepSeekV4TensorShardLoadError::TensorShardOffsetOverflow {
                    tensor_name: tensor_name.to_string(),
                }
            })?;
            Ok((shard_shape, bytes[start..end].to_vec()))
        }
        [rows, columns] if axis == 0 => {
            let row_byte_len = columns.checked_mul(dtype_byte_width).ok_or_else(|| {
                DeepSeekV4TensorShardLoadError::TensorShardByteLengthOverflow {
                    tensor_name: tensor_name.to_string(),
                }
            })?;
            let start = range.start.checked_mul(row_byte_len).ok_or_else(|| {
                DeepSeekV4TensorShardLoadError::TensorShardOffsetOverflow {
                    tensor_name: tensor_name.to_string(),
                }
            })?;
            let end = range.end.checked_mul(row_byte_len).ok_or_else(|| {
                DeepSeekV4TensorShardLoadError::TensorShardOffsetOverflow {
                    tensor_name: tensor_name.to_string(),
                }
            })?;
            debug_assert!(range.end <= *rows);
            Ok((shard_shape, bytes[start..end].to_vec()))
        }
        [rows, columns] => {
            let row_byte_len = columns.checked_mul(dtype_byte_width).ok_or_else(|| {
                DeepSeekV4TensorShardLoadError::TensorShardByteLengthOverflow {
                    tensor_name: tensor_name.to_string(),
                }
            })?;
            let column_start = range.start.checked_mul(dtype_byte_width).ok_or_else(|| {
                DeepSeekV4TensorShardLoadError::TensorShardOffsetOverflow {
                    tensor_name: tensor_name.to_string(),
                }
            })?;
            let column_end = range.end.checked_mul(dtype_byte_width).ok_or_else(|| {
                DeepSeekV4TensorShardLoadError::TensorShardOffsetOverflow {
                    tensor_name: tensor_name.to_string(),
                }
            })?;
            let shard_row_byte_len = column_end.checked_sub(column_start).ok_or_else(|| {
                DeepSeekV4TensorShardLoadError::TensorShardOffsetOverflow {
                    tensor_name: tensor_name.to_string(),
                }
            })?;
            let shard_byte_len = rows.checked_mul(shard_row_byte_len).ok_or_else(|| {
                DeepSeekV4TensorShardLoadError::TensorShardByteLengthOverflow {
                    tensor_name: tensor_name.to_string(),
                }
            })?;
            let mut shard_bytes = Vec::with_capacity(shard_byte_len);
            for row in 0..*rows {
                let row_start = row.checked_mul(row_byte_len).ok_or_else(|| {
                    DeepSeekV4TensorShardLoadError::TensorShardOffsetOverflow {
                        tensor_name: tensor_name.to_string(),
                    }
                })?;
                let start = row_start.checked_add(column_start).ok_or_else(|| {
                    DeepSeekV4TensorShardLoadError::TensorShardOffsetOverflow {
                        tensor_name: tensor_name.to_string(),
                    }
                })?;
                let end = row_start.checked_add(column_end).ok_or_else(|| {
                    DeepSeekV4TensorShardLoadError::TensorShardOffsetOverflow {
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
    entries: &mut Vec<DeepSeekV4TensorPlacement>,
    tensor: &DeepSeekTensorDescriptor,
    kind: DeepSeekV4TensorPlacementKind,
) {
    entries.push(DeepSeekV4TensorPlacement {
        tensor: tensor.clone(),
        kind,
    });
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4ForwardPlan {
    forward_mode: ForwardMode,
    request_ids: Vec<RequestId>,
    input_ids: Vec<u32>,
    positions: Vec<usize>,
    sequence_lengths: Vec<usize>,
    request_offsets: Vec<usize>,
    cached_token_counts: Vec<usize>,
    input_token_counts: Vec<usize>,
    out_cache_pages: Vec<CachePageId>,
    data_parallel_ranks: Vec<i32>,
    bootstrap_rooms: Vec<Option<BootstrapRoom>>,
    request_spans: Vec<DeepSeekV4RequestForwardSpan>,
}

impl DeepSeekV4ForwardPlan {
    fn from_model_worker_batch(batch: &ModelWorkerBatch) -> Self {
        let bootstrap_rooms = batch
            .disaggregated_params()
            .iter()
            .map(|params| params.as_ref().map(|params| params.bootstrap_room))
            .collect::<Vec<_>>();
        let mut request_spans = Vec::with_capacity(batch.request_ids().len());
        for request_index in 0..batch.request_ids().len() {
            let token_start = batch.request_offsets()[request_index];
            let token_count = batch.input_token_counts()[request_index];
            let token_end = token_start + token_count;
            let out_cache_pages = if token_end <= batch.out_cache_pages().len() {
                batch.out_cache_pages()[token_start..token_end].to_vec()
            } else {
                Vec::new()
            };
            request_spans.push(DeepSeekV4RequestForwardSpan {
                request_id: batch.request_ids()[request_index].clone(),
                token_range: token_start..token_end,
                prefix_cache_pages: batch.prefix_cache_pages()[request_index].clone(),
                out_cache_pages,
                data_parallel_rank: batch.data_parallel_ranks()[request_index],
                bootstrap_room: bootstrap_rooms[request_index],
            });
        }

        Self {
            forward_mode: batch.forward_mode(),
            request_ids: batch.request_ids().to_vec(),
            input_ids: batch.input_ids().to_vec(),
            positions: batch.positions().to_vec(),
            sequence_lengths: batch.sequence_lengths().to_vec(),
            request_offsets: batch.request_offsets().to_vec(),
            cached_token_counts: batch.cached_token_counts().to_vec(),
            input_token_counts: batch.input_token_counts().to_vec(),
            out_cache_pages: batch.out_cache_pages().to_vec(),
            data_parallel_ranks: batch.data_parallel_ranks().to_vec(),
            bootstrap_rooms,
            request_spans,
        }
    }

    pub fn forward_mode(&self) -> ForwardMode {
        self.forward_mode
    }

    pub fn request_ids(&self) -> &[RequestId] {
        &self.request_ids
    }

    pub fn input_ids(&self) -> &[u32] {
        &self.input_ids
    }

    pub fn positions(&self) -> &[usize] {
        &self.positions
    }

    pub fn sequence_lengths(&self) -> &[usize] {
        &self.sequence_lengths
    }

    pub fn request_offsets(&self) -> &[usize] {
        &self.request_offsets
    }

    pub fn cached_token_counts(&self) -> &[usize] {
        &self.cached_token_counts
    }

    pub fn input_token_counts(&self) -> &[usize] {
        &self.input_token_counts
    }

    pub fn out_cache_pages(&self) -> &[CachePageId] {
        &self.out_cache_pages
    }

    pub fn data_parallel_ranks(&self) -> &[i32] {
        &self.data_parallel_ranks
    }

    pub fn bootstrap_rooms(&self) -> &[Option<BootstrapRoom>] {
        &self.bootstrap_rooms
    }

    pub fn request_spans(&self) -> &[DeepSeekV4RequestForwardSpan] {
        &self.request_spans
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4RequestForwardSpan {
    request_id: RequestId,
    token_range: Range<usize>,
    prefix_cache_pages: Vec<CachePageId>,
    out_cache_pages: Vec<CachePageId>,
    data_parallel_rank: i32,
    bootstrap_room: Option<BootstrapRoom>,
}

impl DeepSeekV4RequestForwardSpan {
    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    pub fn token_range(&self) -> Range<usize> {
        self.token_range.clone()
    }

    pub fn prefix_cache_pages(&self) -> &[CachePageId] {
        &self.prefix_cache_pages
    }

    pub fn out_cache_pages(&self) -> &[CachePageId] {
        &self.out_cache_pages
    }

    pub fn data_parallel_rank(&self) -> i32 {
        self.data_parallel_rank
    }

    pub fn bootstrap_room(&self) -> Option<BootstrapRoom> {
        self.bootstrap_room
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum DeepSeekRuntimeError {
    UnsupportedModelType { model_type: Option<String> },
    MissingKvCacheLayout,
    ModelArtifact(ModelArtifactError),
    PdConfig(PdConfigError),
}

impl fmt::Display for DeepSeekRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedModelType { model_type } => write!(
                formatter,
                "DeepSeek V4 runtime requires model_type deepseek_v4, got {}",
                model_type.as_deref().unwrap_or("<unknown>")
            ),
            Self::MissingKvCacheLayout => {
                formatter.write_str("DeepSeek V4 runtime requires a packed KV cache layout")
            }
            Self::ModelArtifact(error) => write!(formatter, "{error}"),
            Self::PdConfig(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for DeepSeekRuntimeError {}

impl From<ModelArtifactError> for DeepSeekRuntimeError {
    fn from(value: ModelArtifactError) -> Self {
        Self::ModelArtifact(value)
    }
}

impl From<PdConfigError> for DeepSeekRuntimeError {
    fn from(value: PdConfigError) -> Self {
        Self::PdConfig(value)
    }
}
