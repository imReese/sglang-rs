use std::fmt;
use std::ops::Range;
use std::path::PathBuf;

use crate::cache::CachePageId;
use crate::model_artifacts::{
    DeepSeekLayerCheckpointWeights, DeepSeekLayerFeedForwardCheckpointWeights,
    DeepSeekModelCheckpointWeights, DeepSeekModelTensorSpan, LocalModelArtifacts,
    LocalModelCheckpointCatalog, ModelArtifactError, SafetensorsLayerTensorSpan,
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
