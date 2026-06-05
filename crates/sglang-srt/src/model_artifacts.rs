use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalModelArtifacts {
    model_path: PathBuf,
    config: HfModelConfig,
    safetensors: SafetensorsManifest,
}

impl LocalModelArtifacts {
    pub fn from_model_path(path: impl AsRef<Path>) -> Result<Self, ModelArtifactError> {
        let model_path = resolve_model_path(path.as_ref());
        Self::from_resolved_model_path(&model_path)
    }

    pub fn from_model_path_with_hf_cache(
        model_path: &str,
        hub_cache: impl AsRef<Path>,
    ) -> Result<Self, ModelArtifactError> {
        let model_path = resolve_model_path_from_hf_cache(model_path, hub_cache)
            .unwrap_or_else(|| PathBuf::from(model_path));
        Self::from_resolved_model_path(&model_path)
    }

    fn from_resolved_model_path(model_path: &Path) -> Result<Self, ModelArtifactError> {
        if !model_path.is_dir() {
            return Err(ModelArtifactError::ModelPathNotLocalDirectory {
                path: model_path.to_path_buf(),
            });
        }

        let config = HfModelConfig::from_model_path(model_path)?;
        let safetensors = SafetensorsManifest::from_model_path(model_path)?;

        Ok(Self {
            model_path: model_path.to_path_buf(),
            config,
            safetensors,
        })
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn config(&self) -> &HfModelConfig {
        &self.config
    }

    pub fn safetensors(&self) -> &SafetensorsManifest {
        &self.safetensors
    }

    pub fn validate_routed_expert_checkpoint_coverage(
        &self,
    ) -> Result<RoutedExpertCheckpointCoverage, ModelArtifactError> {
        let groups = self.safetensors.routed_expert_weight_groups()?;
        self.validate_routed_expert_checkpoint_coverage_for_groups(&groups)
    }

    pub fn routed_expert_weight_catalog(
        &self,
    ) -> Result<SafetensorsRoutedExpertWeightCatalog, ModelArtifactError> {
        SafetensorsRoutedExpertWeightCatalog::from_local_model_artifacts(self)
    }

    pub fn checkpoint_catalog(&self) -> Result<LocalModelCheckpointCatalog, ModelArtifactError> {
        LocalModelCheckpointCatalog::from_local_model_artifacts(self)
    }

    pub fn validate_checkpoint_for_supported_model(&self) -> Result<(), ModelArtifactError> {
        let checkpoint = self.checkpoint_catalog()?;
        match self.config.model_type.as_deref() {
            Some("deepseek_v4") => {
                checkpoint.deepseek_model_weights()?;
            }
            _ => {}
        }
        Ok(())
    }

    fn validate_routed_expert_checkpoint_coverage_for_groups(
        &self,
        groups: &[SafetensorsRoutedExpertWeightGroup],
    ) -> Result<RoutedExpertCheckpointCoverage, ModelArtifactError> {
        let Some(expected_group_count) = self.config.expected_routed_expert_group_count() else {
            return Ok(RoutedExpertCheckpointCoverage {
                expected_group_count: 0,
                actual_group_count: 0,
                expected_weight_count: 0,
                actual_weight_count: 0,
            });
        };
        let expected_weight_count = self
            .config
            .expected_routed_expert_weight_count()
            .ok_or_else(|| {
                invalid_safetensors_data(
                    &self.model_path,
                    "expected routed expert weight count overflowed",
                )
            })?;

        let actual_group_count = groups.len();
        let actual_weight_count = actual_group_count.checked_mul(3).ok_or_else(|| {
            invalid_safetensors_data(
                &self.model_path,
                "actual routed expert weight count overflowed",
            )
        })?;

        if actual_group_count != expected_group_count {
            return Err(invalid_safetensors_data(
                &self.model_path,
                format!(
                    "expected {expected_group_count} routed expert groups from model config but found {actual_group_count}"
                ),
            ));
        }

        let n_routed_experts = self.config.n_routed_experts.ok_or_else(|| {
            invalid_safetensors_data(
                &self.model_path,
                "expected routed expert count is missing from model config",
            )
        })?;
        let expected_coordinates: BTreeSet<(usize, usize)> = self
            .config
            .moe_layer_ids()
            .into_iter()
            .flat_map(|layer_id| (0..n_routed_experts).map(move |expert_id| (layer_id, expert_id)))
            .collect();
        let actual_coordinates: BTreeSet<(usize, usize)> = groups
            .iter()
            .map(|group| (group.layer_id, group.expert_id))
            .collect();
        if actual_coordinates != expected_coordinates {
            let missing = expected_coordinates
                .difference(&actual_coordinates)
                .next()
                .map(|(layer_id, expert_id)| {
                    format!(
                        "missing expected routed expert group layer {layer_id} expert {expert_id}"
                    )
                });
            let unexpected = actual_coordinates
                .difference(&expected_coordinates)
                .next()
                .map(|(layer_id, expert_id)| {
                    format!("unexpected routed expert group layer {layer_id} expert {expert_id}")
                });
            let detail = [missing, unexpected]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join("; ");

            return Err(invalid_safetensors_data(
                &self.model_path,
                format!("routed expert checkpoint coordinate mismatch: {detail}"),
            ));
        }

        Ok(RoutedExpertCheckpointCoverage {
            expected_group_count,
            actual_group_count,
            expected_weight_count,
            actual_weight_count,
        })
    }
}

pub fn resolve_model_path(path: impl AsRef<Path>) -> PathBuf {
    let path = path.as_ref();
    if path.is_dir() {
        return path.to_path_buf();
    }

    path.to_str()
        .and_then(|model_id| {
            default_hf_hub_cache().and_then(|hub| resolve_model_path_from_hf_cache(model_id, hub))
        })
        .unwrap_or_else(|| path.to_path_buf())
}

pub fn resolve_model_path_from_hf_cache(
    model_id: &str,
    hub_cache: impl AsRef<Path>,
) -> Option<PathBuf> {
    resolve_model_path_from_hf_cache_with_required_file(model_id, hub_cache, "config.json")
}

pub fn resolve_model_path_from_hf_cache_with_required_file(
    model_id: &str,
    hub_cache: impl AsRef<Path>,
    required_file: &str,
) -> Option<PathBuf> {
    if model_id.starts_with('-') || model_id.starts_with('/') || model_id.contains('\\') {
        return None;
    }

    let repo_cache = hub_cache
        .as_ref()
        .join(format!("models--{}", model_id.replace('/', "--")));
    let refs_main = repo_cache.join("refs").join("main");
    if let Ok(commit) = fs::read_to_string(refs_main) {
        let snapshot = repo_cache.join("snapshots").join(commit.trim());
        if snapshot.join(required_file).is_file() {
            return Some(snapshot);
        }
    }

    let snapshots = repo_cache.join("snapshots");
    let mut candidates = fs::read_dir(snapshots)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.join(required_file).is_file())
        .collect::<Vec<_>>();
    candidates.sort();
    candidates.pop()
}

fn default_hf_hub_cache() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("HUGGINGFACE_HUB_CACHE") {
        return Some(PathBuf::from(path));
    }
    if let Some(path) = std::env::var_os("HF_HOME") {
        return Some(PathBuf::from(path).join("hub"));
    }
    std::env::var_os("HOME").map(|home| {
        PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub")
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutedExpertCheckpointCoverage {
    pub expected_group_count: usize,
    pub actual_group_count: usize,
    pub expected_weight_count: usize,
    pub actual_weight_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalModelCheckpointCatalog {
    model_path: PathBuf,
    config: HfModelConfig,
    safetensors: SafetensorsManifest,
    layer_tensors: SafetensorsLayerTensorCatalog,
    routed_experts: SafetensorsRoutedExpertWeightCatalog,
}

impl LocalModelCheckpointCatalog {
    pub fn from_local_model_artifacts(
        artifacts: &LocalModelArtifacts,
    ) -> Result<Self, ModelArtifactError> {
        let layer_tensors = artifacts.safetensors().layer_tensor_catalog()?;
        let routed_experts = artifacts.routed_expert_weight_catalog()?;

        Ok(Self {
            model_path: artifacts.model_path().to_path_buf(),
            config: artifacts.config().clone(),
            safetensors: artifacts.safetensors().clone(),
            layer_tensors,
            routed_experts,
        })
    }

    pub fn layer_tensors(&self) -> &SafetensorsLayerTensorCatalog {
        &self.layer_tensors
    }

    pub fn routed_experts(&self) -> &SafetensorsRoutedExpertWeightCatalog {
        &self.routed_experts
    }

    pub fn deepseek_model_weights(
        &self,
    ) -> Result<DeepSeekModelCheckpointWeights<'_>, ModelArtifactError> {
        let num_hidden_layers = self.config.num_hidden_layers.ok_or_else(|| {
            invalid_safetensors_data(
                &self.model_path,
                "missing DeepSeek model num_hidden_layers config",
            )
        })?;
        let layers = (0..num_hidden_layers)
            .map(|layer_id| self.deepseek_layer_weights(layer_id))
            .collect::<Result<Vec<_>, _>>()?;

        let weights = DeepSeekModelCheckpointWeights {
            token_embeddings: self.required_deepseek_model_tensor("model.embed_tokens.weight")?,
            final_norm: self.required_deepseek_model_tensor("model.norm.weight")?,
            lm_head: self.required_deepseek_model_tensor("lm_head.weight")?,
            hc_head_fn: self.required_deepseek_model_tensor("model.hc_head_fn")?,
            hc_head_base: self.required_deepseek_model_tensor("model.hc_head_base")?,
            hc_head_scale: self.required_deepseek_model_tensor("model.hc_head_scale")?,
            layers,
        };
        self.validate_deepseek_hc_head_shapes(&weights)?;
        Ok(weights)
    }

    pub fn deepseek_layer_weights(
        &self,
        layer_id: usize,
    ) -> Result<DeepSeekLayerCheckpointWeights<'_>, ModelArtifactError> {
        let feed_forward = if self.config.is_moe_layer(layer_id) {
            let routed_experts = self.routed_experts.layer(layer_id).ok_or_else(|| {
                invalid_safetensors_data(
                    &self.model_path,
                    format!("missing DeepSeek layer {layer_id} routed expert weights"),
                )
            })?;
            DeepSeekLayerFeedForwardCheckpointWeights::Moe {
                gate: self.required_deepseek_layer_tensor(layer_id, "mlp.gate.weight")?,
                routed_experts,
            }
        } else {
            DeepSeekLayerFeedForwardCheckpointWeights::Dense {
                gate_up_proj: self
                    .required_deepseek_layer_tensor(layer_id, "mlp.gate_up_proj.weight")?,
                down_proj: self.required_deepseek_layer_tensor(layer_id, "mlp.down_proj.weight")?,
            }
        };

        Ok(DeepSeekLayerCheckpointWeights {
            layer_id,
            wq_a: self.required_deepseek_layer_tensor(layer_id, "self_attn.wq_a.weight")?,
            wq_b: self.required_deepseek_layer_tensor(layer_id, "self_attn.wq_b.weight")?,
            wkv: self.required_deepseek_layer_tensor(layer_id, "self_attn.wkv.weight")?,
            q_norm: self.required_deepseek_layer_tensor(layer_id, "self_attn.q_norm.weight")?,
            kv_norm: self.required_deepseek_layer_tensor(layer_id, "self_attn.kv_norm.weight")?,
            wo_a: self.required_deepseek_layer_tensor(layer_id, "self_attn.wo_a.weight")?,
            wo_b: self.required_deepseek_layer_tensor(layer_id, "self_attn.wo_b.weight")?,
            input_layernorm: self
                .required_deepseek_layer_tensor(layer_id, "input_layernorm.weight")?,
            post_attention_layernorm: self
                .required_deepseek_layer_tensor(layer_id, "post_attention_layernorm.weight")?,
            hc_attn_fn: self.required_deepseek_layer_tensor(layer_id, "hc_attn_fn")?,
            hc_attn_base: self.required_deepseek_layer_tensor(layer_id, "hc_attn_base")?,
            hc_attn_scale: self.required_deepseek_layer_tensor(layer_id, "hc_attn_scale")?,
            hc_ffn_fn: self.required_deepseek_layer_tensor(layer_id, "hc_ffn_fn")?,
            hc_ffn_base: self.required_deepseek_layer_tensor(layer_id, "hc_ffn_base")?,
            hc_ffn_scale: self.required_deepseek_layer_tensor(layer_id, "hc_ffn_scale")?,
            feed_forward,
        })
    }

    fn required_deepseek_layer_tensor(
        &self,
        layer_id: usize,
        suffix: &str,
    ) -> Result<&SafetensorsLayerTensorSpan, ModelArtifactError> {
        self.layer_tensors.span(layer_id, suffix).ok_or_else(|| {
            invalid_safetensors_data(
                &self.model_path,
                format!("missing DeepSeek layer {layer_id} tensor {suffix}"),
            )
        })
    }

    fn required_deepseek_model_tensor(
        &self,
        tensor_name: &str,
    ) -> Result<DeepSeekModelTensorSpan, ModelArtifactError> {
        self.safetensors
            .tensor_span(tensor_name)?
            .map(|span| DeepSeekModelTensorSpan {
                tensor_name: tensor_name.to_string(),
                span,
            })
            .ok_or_else(|| {
                invalid_safetensors_data(
                    &self.model_path,
                    format!("missing DeepSeek model tensor {tensor_name}"),
                )
            })
    }

    fn validate_deepseek_hc_head_shapes(
        &self,
        weights: &DeepSeekModelCheckpointWeights<'_>,
    ) -> Result<(), ModelArtifactError> {
        let hidden_size = self.config.hidden_size.ok_or_else(|| {
            invalid_safetensors_data(
                &self.model_path,
                "missing DeepSeek model hidden_size config for HC head validation",
            )
        })?;
        let hc_mult = self.config.hc_mult.ok_or_else(|| {
            invalid_safetensors_data(
                &self.model_path,
                "missing DeepSeek model hc_mult config for HC head validation",
            )
        })?;
        let hc_dim = hc_mult.checked_mul(hidden_size).ok_or_else(|| {
            invalid_safetensors_data(&self.model_path, "DeepSeek HC head dimension overflowed")
        })?;

        self.validate_deepseek_model_tensor_shape(weights.hc_head_fn(), &[hc_mult, hc_dim])?;
        self.validate_deepseek_model_tensor_shape(weights.hc_head_base(), &[hc_mult])?;
        self.validate_deepseek_model_tensor_shape(weights.hc_head_scale(), &[1])?;
        Ok(())
    }

    fn validate_deepseek_model_tensor_shape(
        &self,
        tensor: &DeepSeekModelTensorSpan,
        expected_shape: &[usize],
    ) -> Result<(), ModelArtifactError> {
        if tensor.span.metadata.shape == expected_shape {
            return Ok(());
        }

        Err(invalid_safetensors_data(
            &self.model_path,
            format!(
                "DeepSeek model tensor {} shape {:?} does not match expected {:?}",
                tensor.tensor_name, tensor.span.metadata.shape, expected_shape
            ),
        ))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekModelTensorSpan {
    pub tensor_name: String,
    pub span: SafetensorsTensorSpan,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekModelCheckpointWeights<'a> {
    token_embeddings: DeepSeekModelTensorSpan,
    final_norm: DeepSeekModelTensorSpan,
    hc_head_fn: DeepSeekModelTensorSpan,
    hc_head_base: DeepSeekModelTensorSpan,
    hc_head_scale: DeepSeekModelTensorSpan,
    lm_head: DeepSeekModelTensorSpan,
    layers: Vec<DeepSeekLayerCheckpointWeights<'a>>,
}

impl<'a> DeepSeekModelCheckpointWeights<'a> {
    pub fn token_embeddings(&self) -> &DeepSeekModelTensorSpan {
        &self.token_embeddings
    }

    pub fn final_norm(&self) -> &DeepSeekModelTensorSpan {
        &self.final_norm
    }

    pub fn hc_head_fn(&self) -> &DeepSeekModelTensorSpan {
        &self.hc_head_fn
    }

    pub fn hc_head_base(&self) -> &DeepSeekModelTensorSpan {
        &self.hc_head_base
    }

    pub fn hc_head_scale(&self) -> &DeepSeekModelTensorSpan {
        &self.hc_head_scale
    }

    pub fn lm_head(&self) -> &DeepSeekModelTensorSpan {
        &self.lm_head
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn layers(&self) -> &[DeepSeekLayerCheckpointWeights<'a>] {
        &self.layers
    }

    pub fn layer(&self, layer_id: usize) -> Option<&DeepSeekLayerCheckpointWeights<'a>> {
        self.layers.get(layer_id)
    }

    pub fn read_root_tensors(&self) -> Result<DeepSeekLoadedModelRootWeights, ModelArtifactError> {
        Ok(DeepSeekLoadedModelRootWeights {
            token_embeddings: read_required_tensor_span(&self.token_embeddings.span)?,
            final_norm: read_required_tensor_span(&self.final_norm.span)?,
            hc_head_fn: read_required_tensor_span(&self.hc_head_fn.span)?,
            hc_head_base: read_required_tensor_span(&self.hc_head_base.span)?,
            hc_head_scale: read_required_tensor_span(&self.hc_head_scale.span)?,
            lm_head: read_required_tensor_span(&self.lm_head.span)?,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekLoadedModelRootWeights {
    token_embeddings: SafetensorsTensorData,
    final_norm: SafetensorsTensorData,
    hc_head_fn: SafetensorsTensorData,
    hc_head_base: SafetensorsTensorData,
    hc_head_scale: SafetensorsTensorData,
    lm_head: SafetensorsTensorData,
}

impl DeepSeekLoadedModelRootWeights {
    pub fn token_embeddings(&self) -> &SafetensorsTensorData {
        &self.token_embeddings
    }

    pub fn final_norm(&self) -> &SafetensorsTensorData {
        &self.final_norm
    }

    pub fn hc_head_fn(&self) -> &SafetensorsTensorData {
        &self.hc_head_fn
    }

    pub fn hc_head_base(&self) -> &SafetensorsTensorData {
        &self.hc_head_base
    }

    pub fn hc_head_scale(&self) -> &SafetensorsTensorData {
        &self.hc_head_scale
    }

    pub fn lm_head(&self) -> &SafetensorsTensorData {
        &self.lm_head
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekLayerCheckpointWeights<'a> {
    layer_id: usize,
    wq_a: &'a SafetensorsLayerTensorSpan,
    wq_b: &'a SafetensorsLayerTensorSpan,
    wkv: &'a SafetensorsLayerTensorSpan,
    q_norm: &'a SafetensorsLayerTensorSpan,
    kv_norm: &'a SafetensorsLayerTensorSpan,
    wo_a: &'a SafetensorsLayerTensorSpan,
    wo_b: &'a SafetensorsLayerTensorSpan,
    input_layernorm: &'a SafetensorsLayerTensorSpan,
    post_attention_layernorm: &'a SafetensorsLayerTensorSpan,
    hc_attn_fn: &'a SafetensorsLayerTensorSpan,
    hc_attn_base: &'a SafetensorsLayerTensorSpan,
    hc_attn_scale: &'a SafetensorsLayerTensorSpan,
    hc_ffn_fn: &'a SafetensorsLayerTensorSpan,
    hc_ffn_base: &'a SafetensorsLayerTensorSpan,
    hc_ffn_scale: &'a SafetensorsLayerTensorSpan,
    feed_forward: DeepSeekLayerFeedForwardCheckpointWeights<'a>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeepSeekLayerFeedForwardCheckpointWeights<'a> {
    Dense {
        gate_up_proj: &'a SafetensorsLayerTensorSpan,
        down_proj: &'a SafetensorsLayerTensorSpan,
    },
    Moe {
        gate: &'a SafetensorsLayerTensorSpan,
        routed_experts: SafetensorsRoutedExpertLayerWeights<'a>,
    },
}

impl<'a> DeepSeekLayerCheckpointWeights<'a> {
    pub fn read_tensors(&self) -> Result<DeepSeekLoadedLayerCheckpointWeights, ModelArtifactError> {
        Ok(DeepSeekLoadedLayerCheckpointWeights {
            layer_id: self.layer_id,
            wq_a: read_required_tensor_span(&self.wq_a.span)?,
            wq_b: read_required_tensor_span(&self.wq_b.span)?,
            wkv: read_required_tensor_span(&self.wkv.span)?,
            q_norm: read_required_tensor_span(&self.q_norm.span)?,
            kv_norm: read_required_tensor_span(&self.kv_norm.span)?,
            wo_a: read_required_tensor_span(&self.wo_a.span)?,
            wo_b: read_required_tensor_span(&self.wo_b.span)?,
            input_layernorm: read_required_tensor_span(&self.input_layernorm.span)?,
            post_attention_layernorm: read_required_tensor_span(
                &self.post_attention_layernorm.span,
            )?,
            hc_attn_fn: read_required_tensor_span(&self.hc_attn_fn.span)?,
            hc_attn_base: read_required_tensor_span(&self.hc_attn_base.span)?,
            hc_attn_scale: read_required_tensor_span(&self.hc_attn_scale.span)?,
            hc_ffn_fn: read_required_tensor_span(&self.hc_ffn_fn.span)?,
            hc_ffn_base: read_required_tensor_span(&self.hc_ffn_base.span)?,
            hc_ffn_scale: read_required_tensor_span(&self.hc_ffn_scale.span)?,
            feed_forward: match &self.feed_forward {
                DeepSeekLayerFeedForwardCheckpointWeights::Dense {
                    gate_up_proj,
                    down_proj,
                } => DeepSeekLoadedLayerFeedForwardWeights::Dense {
                    gate_up_proj: read_required_tensor_span(&gate_up_proj.span)?,
                    down_proj: read_required_tensor_span(&down_proj.span)?,
                },
                DeepSeekLayerFeedForwardCheckpointWeights::Moe {
                    gate,
                    routed_experts,
                } => DeepSeekLoadedLayerFeedForwardWeights::Moe {
                    gate: read_required_tensor_span(&gate.span)?,
                    routed_experts: routed_experts
                        .groups()
                        .map(DeepSeekLoadedRoutedExpertWeights::from_group)
                        .collect::<Result<Vec<_>, _>>()?,
                },
            },
        })
    }

    pub fn layer_id(&self) -> usize {
        self.layer_id
    }

    pub fn wq_a(&self) -> &'a SafetensorsLayerTensorSpan {
        self.wq_a
    }

    pub fn wq_b(&self) -> &'a SafetensorsLayerTensorSpan {
        self.wq_b
    }

    pub fn wkv(&self) -> &'a SafetensorsLayerTensorSpan {
        self.wkv
    }

    pub fn q_norm(&self) -> &'a SafetensorsLayerTensorSpan {
        self.q_norm
    }

    pub fn kv_norm(&self) -> &'a SafetensorsLayerTensorSpan {
        self.kv_norm
    }

    pub fn wo_a(&self) -> &'a SafetensorsLayerTensorSpan {
        self.wo_a
    }

    pub fn wo_b(&self) -> &'a SafetensorsLayerTensorSpan {
        self.wo_b
    }

    pub fn input_layernorm(&self) -> &'a SafetensorsLayerTensorSpan {
        self.input_layernorm
    }

    pub fn post_attention_layernorm(&self) -> &'a SafetensorsLayerTensorSpan {
        self.post_attention_layernorm
    }

    pub fn hc_attn_fn(&self) -> &'a SafetensorsLayerTensorSpan {
        self.hc_attn_fn
    }

    pub fn hc_attn_base(&self) -> &'a SafetensorsLayerTensorSpan {
        self.hc_attn_base
    }

    pub fn hc_attn_scale(&self) -> &'a SafetensorsLayerTensorSpan {
        self.hc_attn_scale
    }

    pub fn hc_ffn_fn(&self) -> &'a SafetensorsLayerTensorSpan {
        self.hc_ffn_fn
    }

    pub fn hc_ffn_base(&self) -> &'a SafetensorsLayerTensorSpan {
        self.hc_ffn_base
    }

    pub fn hc_ffn_scale(&self) -> &'a SafetensorsLayerTensorSpan {
        self.hc_ffn_scale
    }

    pub fn feed_forward(&self) -> &DeepSeekLayerFeedForwardCheckpointWeights<'a> {
        &self.feed_forward
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekLoadedLayerCheckpointWeights {
    layer_id: usize,
    wq_a: SafetensorsTensorData,
    wq_b: SafetensorsTensorData,
    wkv: SafetensorsTensorData,
    q_norm: SafetensorsTensorData,
    kv_norm: SafetensorsTensorData,
    wo_a: SafetensorsTensorData,
    wo_b: SafetensorsTensorData,
    input_layernorm: SafetensorsTensorData,
    post_attention_layernorm: SafetensorsTensorData,
    hc_attn_fn: SafetensorsTensorData,
    hc_attn_base: SafetensorsTensorData,
    hc_attn_scale: SafetensorsTensorData,
    hc_ffn_fn: SafetensorsTensorData,
    hc_ffn_base: SafetensorsTensorData,
    hc_ffn_scale: SafetensorsTensorData,
    feed_forward: DeepSeekLoadedLayerFeedForwardWeights,
}

impl DeepSeekLoadedLayerCheckpointWeights {
    pub fn layer_id(&self) -> usize {
        self.layer_id
    }

    pub fn wq_a(&self) -> &SafetensorsTensorData {
        &self.wq_a
    }

    pub fn wq_b(&self) -> &SafetensorsTensorData {
        &self.wq_b
    }

    pub fn wkv(&self) -> &SafetensorsTensorData {
        &self.wkv
    }

    pub fn q_norm(&self) -> &SafetensorsTensorData {
        &self.q_norm
    }

    pub fn kv_norm(&self) -> &SafetensorsTensorData {
        &self.kv_norm
    }

    pub fn wo_a(&self) -> &SafetensorsTensorData {
        &self.wo_a
    }

    pub fn wo_b(&self) -> &SafetensorsTensorData {
        &self.wo_b
    }

    pub fn input_layernorm(&self) -> &SafetensorsTensorData {
        &self.input_layernorm
    }

    pub fn post_attention_layernorm(&self) -> &SafetensorsTensorData {
        &self.post_attention_layernorm
    }

    pub fn hc_attn_fn(&self) -> &SafetensorsTensorData {
        &self.hc_attn_fn
    }

    pub fn hc_attn_base(&self) -> &SafetensorsTensorData {
        &self.hc_attn_base
    }

    pub fn hc_attn_scale(&self) -> &SafetensorsTensorData {
        &self.hc_attn_scale
    }

    pub fn hc_ffn_fn(&self) -> &SafetensorsTensorData {
        &self.hc_ffn_fn
    }

    pub fn hc_ffn_base(&self) -> &SafetensorsTensorData {
        &self.hc_ffn_base
    }

    pub fn hc_ffn_scale(&self) -> &SafetensorsTensorData {
        &self.hc_ffn_scale
    }

    pub fn feed_forward(&self) -> &DeepSeekLoadedLayerFeedForwardWeights {
        &self.feed_forward
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeepSeekLoadedLayerFeedForwardWeights {
    Dense {
        gate_up_proj: SafetensorsTensorData,
        down_proj: SafetensorsTensorData,
    },
    Moe {
        gate: SafetensorsTensorData,
        routed_experts: Vec<DeepSeekLoadedRoutedExpertWeights>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekLoadedRoutedExpertWeights {
    expert_id: usize,
    gate: SafetensorsTensorData,
    up: SafetensorsTensorData,
    down: SafetensorsTensorData,
}

impl DeepSeekLoadedRoutedExpertWeights {
    fn from_group(group: &SafetensorsRoutedExpertWeightGroup) -> Result<Self, ModelArtifactError> {
        Ok(Self {
            expert_id: group.expert_id,
            gate: read_required_tensor_span(&group.gate)?,
            up: read_required_tensor_span(&group.up)?,
            down: read_required_tensor_span(&group.down)?,
        })
    }

    pub fn expert_id(&self) -> usize {
        self.expert_id
    }

    pub fn gate(&self) -> &SafetensorsTensorData {
        &self.gate
    }

    pub fn up(&self) -> &SafetensorsTensorData {
        &self.up
    }

    pub fn down(&self) -> &SafetensorsTensorData {
        &self.down
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HfConfigFloat(f64);

impl HfConfigFloat {
    pub fn get(self) -> f64 {
        self.0
    }
}

impl PartialEq for HfConfigFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for HfConfigFloat {}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HfModelConfig {
    pub model_type: Option<String>,
    pub architectures: Vec<String>,
    pub eos_token_ids: Vec<u32>,
    pub vocab_size: Option<usize>,
    pub max_position_embeddings: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub hidden_size: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub moe_intermediate_size: Option<usize>,
    pub n_routed_experts: Option<usize>,
    pub n_shared_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub first_k_dense_replace: Option<usize>,
    pub moe_layer_freq: Option<usize>,
    pub hc_mult: Option<usize>,
    pub hc_sinkhorn_iters: Option<usize>,
    pub rms_norm_eps: Option<HfConfigFloat>,
    pub hc_eps: Option<HfConfigFloat>,
    pub tie_word_embeddings: Option<bool>,
}

impl HfModelConfig {
    pub fn from_model_path(path: impl AsRef<Path>) -> Result<Self, ModelArtifactError> {
        let model_path = resolve_model_path(path.as_ref());
        Self::from_resolved_model_path(&model_path)
    }

    pub fn from_model_path_with_hf_cache(
        model_path: &str,
        hub_cache: impl AsRef<Path>,
    ) -> Result<Self, ModelArtifactError> {
        let model_path = resolve_model_path_from_hf_cache(model_path, hub_cache)
            .unwrap_or_else(|| PathBuf::from(model_path));
        Self::from_resolved_model_path(&model_path)
    }

    fn from_resolved_model_path(path: &Path) -> Result<Self, ModelArtifactError> {
        let config_path = path.join("config.json");
        let raw = fs::read_to_string(&config_path).map_err(|error| {
            ModelArtifactError::ReadModelConfig {
                path: config_path.clone(),
                message: error.to_string(),
            }
        })?;
        let value: serde_json::Value =
            serde_json::from_str(&raw).map_err(|error| ModelArtifactError::InvalidModelConfig {
                path: config_path.clone(),
                message: error.to_string(),
            })?;

        Ok(Self {
            model_type: read_string_field(&value, "model_type"),
            architectures: read_string_array_field(&value, "architectures"),
            eos_token_ids: read_u32_or_array_field(&value, "eos_token_id", &config_path)?,
            vocab_size: read_usize_field(&value, "vocab_size", &config_path)?,
            max_position_embeddings: read_usize_field(
                &value,
                "max_position_embeddings",
                &config_path,
            )?,
            num_hidden_layers: read_usize_field(&value, "num_hidden_layers", &config_path)?,
            hidden_size: read_usize_field(&value, "hidden_size", &config_path)?,
            intermediate_size: read_usize_field(&value, "intermediate_size", &config_path)?,
            moe_intermediate_size: read_usize_field(&value, "moe_intermediate_size", &config_path)?,
            n_routed_experts: read_usize_field(&value, "n_routed_experts", &config_path)?,
            n_shared_experts: read_usize_field(&value, "n_shared_experts", &config_path)?,
            num_experts_per_tok: read_usize_field(&value, "num_experts_per_tok", &config_path)?,
            first_k_dense_replace: read_usize_field(&value, "first_k_dense_replace", &config_path)?,
            moe_layer_freq: read_usize_field(&value, "moe_layer_freq", &config_path)?,
            hc_mult: read_usize_field(&value, "hc_mult", &config_path)?,
            hc_sinkhorn_iters: read_usize_field(&value, "hc_sinkhorn_iters", &config_path)?,
            rms_norm_eps: read_f64_field(&value, "rms_norm_eps", &config_path)?,
            hc_eps: read_f64_field(&value, "hc_eps", &config_path)?,
            tie_word_embeddings: read_bool_field(&value, "tie_word_embeddings", &config_path)?,
        })
    }

    pub fn is_moe_layer(&self, layer_id: usize) -> bool {
        let Some(num_hidden_layers) = self.num_hidden_layers else {
            return false;
        };
        if layer_id >= num_hidden_layers || self.n_routed_experts.is_none() {
            return false;
        }

        let first_k_dense_replace = self.first_k_dense_replace.unwrap_or(0);
        let moe_layer_freq = self.moe_layer_freq.unwrap_or(1);
        moe_layer_freq > 0 && layer_id >= first_k_dense_replace && layer_id % moe_layer_freq == 0
    }

    pub fn moe_layer_ids(&self) -> Vec<usize> {
        let Some(num_hidden_layers) = self.num_hidden_layers else {
            return Vec::new();
        };

        (0..num_hidden_layers)
            .filter(|layer_id| self.is_moe_layer(*layer_id))
            .collect()
    }

    pub fn expected_routed_expert_group_count(&self) -> Option<usize> {
        self.moe_layer_ids()
            .len()
            .checked_mul(self.n_routed_experts?)
    }

    pub fn expected_routed_expert_weight_count(&self) -> Option<usize> {
        self.expected_routed_expert_group_count()?.checked_mul(3)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsManifest {
    index_path: PathBuf,
    tensor_names: Vec<String>,
    tensor_to_shard: BTreeMap<String, PathBuf>,
    shard_paths: Vec<PathBuf>,
}

impl SafetensorsManifest {
    pub fn from_model_path(path: impl AsRef<Path>) -> Result<Self, ModelArtifactError> {
        let model_path = path.as_ref();
        let index_path = model_path.join("model.safetensors.index.json");
        if index_path.is_file() {
            return Self::from_index_path(model_path, index_path);
        }

        let shard_paths = sorted_safetensors_files(model_path)?;
        if shard_paths.is_empty() {
            return Err(ModelArtifactError::NoSafetensorsWeights {
                path: model_path.to_path_buf(),
            });
        }

        Ok(Self {
            index_path,
            tensor_names: Vec::new(),
            tensor_to_shard: BTreeMap::new(),
            shard_paths,
        })
    }

    pub fn tensor_names(&self) -> &[String] {
        &self.tensor_names
    }

    pub fn shard_for_tensor(&self, tensor_name: &str) -> Option<&Path> {
        self.tensor_to_shard.get(tensor_name).map(PathBuf::as_path)
    }

    pub fn shard_paths(&self) -> &[PathBuf] {
        &self.shard_paths
    }

    pub fn tensor_metadata(
        &self,
        tensor_name: &str,
    ) -> Result<Option<SafetensorsTensorMetadata>, ModelArtifactError> {
        if let Some(shard_path) = self.shard_for_tensor(tensor_name) {
            return SafetensorsHeader::from_file(shard_path)
                .map(|header| header.tensor_metadata(tensor_name));
        }

        for shard_path in &self.shard_paths {
            let header = SafetensorsHeader::from_file(shard_path)?;
            if let Some(metadata) = header.tensor_metadata(tensor_name) {
                return Ok(Some(metadata));
            }
        }

        Ok(None)
    }

    pub fn tensor_span(
        &self,
        tensor_name: &str,
    ) -> Result<Option<SafetensorsTensorSpan>, ModelArtifactError> {
        if let Some(shard_path) = self.shard_for_tensor(tensor_name) {
            let header = SafetensorsHeader::from_file(shard_path)?;
            return header.tensor_span(shard_path, tensor_name);
        }

        for shard_path in &self.shard_paths {
            let header = SafetensorsHeader::from_file(shard_path)?;
            if let Some(span) = header.tensor_span(shard_path, tensor_name)? {
                return Ok(Some(span));
            }
        }

        Ok(None)
    }

    pub fn tensor_span_entries(
        &self,
    ) -> Result<Vec<(String, SafetensorsTensorSpan)>, ModelArtifactError> {
        if self.tensor_names.is_empty() {
            let mut entries = Vec::new();
            for shard_path in &self.shard_paths {
                let header = SafetensorsHeader::from_file(shard_path)?;
                entries.extend(header.tensor_span_entries(shard_path)?);
            }
            return Ok(entries);
        }

        let mut headers = BTreeMap::new();
        let mut entries = Vec::with_capacity(self.tensor_names.len());
        for tensor_name in &self.tensor_names {
            let Some(shard_path) = self.shard_for_tensor(tensor_name) else {
                continue;
            };
            let header = match headers.entry(shard_path.to_path_buf()) {
                std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(SafetensorsHeader::from_file(shard_path)?)
                }
            };
            if let Some(span) = header.tensor_span(shard_path, tensor_name)? {
                entries.push((tensor_name.clone(), span));
            }
        }

        Ok(entries)
    }

    pub fn layer_tensor_spans(
        &self,
    ) -> Result<Vec<SafetensorsLayerTensorSpan>, ModelArtifactError> {
        let spans = self
            .tensor_span_entries()?
            .into_iter()
            .filter_map(|(tensor_name, span)| {
                parse_layer_tensor_name(&tensor_name).map(|(layer_id, suffix)| {
                    SafetensorsLayerTensorSpan {
                        tensor_name,
                        layer_id,
                        suffix,
                        span,
                    }
                })
            })
            .collect();
        Ok(spans)
    }

    pub fn layer_tensor_span(
        &self,
        layer_id: usize,
        suffix: &str,
    ) -> Result<Option<SafetensorsLayerTensorSpan>, ModelArtifactError> {
        let tensor_name = format!("model.layers.{layer_id}.{suffix}");
        let Some(span) = self.tensor_span(&tensor_name)? else {
            return Ok(None);
        };

        Ok(Some(SafetensorsLayerTensorSpan {
            tensor_name,
            layer_id,
            suffix: suffix.to_string(),
            span,
        }))
    }

    pub fn layer_tensor_catalog(
        &self,
    ) -> Result<SafetensorsLayerTensorCatalog, ModelArtifactError> {
        SafetensorsLayerTensorCatalog::from_safetensors_manifest(self)
    }

    pub fn checkpoint_fingerprint_entries(
        &self,
    ) -> Result<Vec<SafetensorsCheckpointFingerprintEntry>, ModelArtifactError> {
        self.tensor_span_entries()?
            .into_iter()
            .map(|(tensor_name, span)| {
                SafetensorsCheckpointFingerprintEntry::from_span(tensor_name, span)
            })
            .collect()
    }

    pub fn routed_expert_weight_spans(
        &self,
    ) -> Result<Vec<SafetensorsRoutedExpertWeightSpan>, ModelArtifactError> {
        let spans = self
            .tensor_span_entries()?
            .into_iter()
            .filter_map(|(tensor_name, span)| {
                parse_routed_expert_weight_name(&tensor_name).map(
                    |(layer_id, expert_id, projection)| SafetensorsRoutedExpertWeightSpan {
                        tensor_name,
                        layer_id,
                        expert_id,
                        projection,
                        span,
                    },
                )
            })
            .collect();
        Ok(spans)
    }

    pub fn routed_expert_weight_groups(
        &self,
    ) -> Result<Vec<SafetensorsRoutedExpertWeightGroup>, ModelArtifactError> {
        let mut builders = BTreeMap::new();
        for weight in self.routed_expert_weight_spans()? {
            let key = (weight.layer_id, weight.expert_id);
            builders
                .entry(key)
                .or_insert_with(|| {
                    RoutedExpertWeightGroupBuilder::new(weight.layer_id, weight.expert_id)
                })
                .insert(weight)?;
        }

        builders
            .into_values()
            .map(RoutedExpertWeightGroupBuilder::finish)
            .collect()
    }

    pub fn read_tensor(
        &self,
        tensor_name: &str,
    ) -> Result<Option<SafetensorsTensorData>, ModelArtifactError> {
        let Some(span) = self.tensor_span(tensor_name)? else {
            return Ok(None);
        };
        span.read()
    }

    pub fn probe_routed_expert_weight_dtype(&self) -> Result<Option<String>, ModelArtifactError> {
        if let Some(tensor_name) = self
            .tensor_names
            .iter()
            .find(|name| is_routed_expert_weight(name))
        {
            return Ok(self
                .tensor_metadata(tensor_name)?
                .map(|metadata| metadata.dtype));
        }

        for shard_path in &self.shard_paths {
            let header = SafetensorsHeader::from_file(shard_path)?;
            if let Some(dtype) = header.routed_expert_weight_dtype() {
                return Ok(Some(dtype));
            }
        }

        Ok(None)
    }

    fn from_index_path(model_path: &Path, index_path: PathBuf) -> Result<Self, ModelArtifactError> {
        let raw = fs::read_to_string(&index_path).map_err(|error| {
            ModelArtifactError::ReadSafetensorsIndex {
                path: index_path.clone(),
                message: error.to_string(),
            }
        })?;
        let value: serde_json::Value = serde_json::from_str(&raw).map_err(|error| {
            ModelArtifactError::InvalidSafetensorsIndex {
                path: index_path.clone(),
                message: error.to_string(),
            }
        })?;
        let weight_map = value
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| ModelArtifactError::InvalidSafetensorsIndex {
                path: index_path.clone(),
                message: "missing object field weight_map".to_string(),
            })?;
        if weight_map.is_empty() {
            return Err(ModelArtifactError::InvalidSafetensorsIndex {
                path: index_path,
                message: "weight_map is empty".to_string(),
            });
        }

        let mut tensor_to_shard = BTreeMap::new();
        let mut shard_paths = BTreeSet::new();
        for (tensor_name, shard_name) in weight_map {
            let Some(shard_name) = shard_name.as_str() else {
                return Err(ModelArtifactError::InvalidSafetensorsIndex {
                    path: index_path.clone(),
                    message: format!("weight_map entry {tensor_name} is not a string"),
                });
            };
            let shard_path = model_path.join(shard_name);
            if !shard_path.is_file() {
                return Err(ModelArtifactError::MissingWeightShard { path: shard_path });
            }

            tensor_to_shard.insert(tensor_name.clone(), shard_path.clone());
            shard_paths.insert(shard_path);
        }

        Ok(Self {
            index_path,
            tensor_names: tensor_to_shard.keys().cloned().collect(),
            tensor_to_shard,
            shard_paths: shard_paths.into_iter().collect(),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsTensorMetadata {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

impl SafetensorsTensorMetadata {
    pub fn element_count(&self) -> Option<usize> {
        shape_element_count(&self.shape)
    }

    pub fn dtype_byte_width(&self) -> Option<usize> {
        safetensors_dtype_byte_width(&self.dtype)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsTensorData {
    pub metadata: SafetensorsTensorMetadata,
    pub bytes: Vec<u8>,
}

impl SafetensorsTensorData {
    pub fn element_count(&self) -> usize {
        self.metadata
            .element_count()
            .expect("loaded safetensors tensor element count should be validated")
    }

    pub fn dtype_byte_width(&self) -> usize {
        self.metadata
            .dtype_byte_width()
            .expect("loaded safetensors tensor dtype should be validated")
    }

    pub fn decode_f32_values(&self) -> Result<Vec<f32>, SafetensorsTensorDecodeError> {
        match self.metadata.dtype.as_str() {
            "F32" => decode_f32_bytes(&self.bytes),
            "BF16" => decode_u16_float_bytes("BF16", &self.bytes, bf16_to_f32),
            "F16" => decode_u16_float_bytes("F16", &self.bytes, f16_to_f32),
            "F8_E4M3" => Ok(self.bytes.iter().copied().map(f8_e4m3fn_to_f32).collect()),
            dtype => Err(SafetensorsTensorDecodeError::UnsupportedDtype {
                dtype: dtype.to_string(),
            }),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum SafetensorsTensorDecodeError {
    UnsupportedDtype { dtype: String },
    InvalidByteLength { dtype: String, byte_len: usize },
}

impl fmt::Display for SafetensorsTensorDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedDtype { dtype } => {
                write!(formatter, "cannot decode safetensors dtype {dtype} as f32")
            }
            Self::InvalidByteLength { dtype, byte_len } => write!(
                formatter,
                "cannot decode safetensors dtype {dtype} from {byte_len} bytes"
            ),
        }
    }
}

impl std::error::Error for SafetensorsTensorDecodeError {}

fn decode_f32_bytes(bytes: &[u8]) -> Result<Vec<f32>, SafetensorsTensorDecodeError> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(SafetensorsTensorDecodeError::InvalidByteLength {
            dtype: "F32".to_string(),
            byte_len: bytes.len(),
        });
    }

    Ok(chunks
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn decode_u16_float_bytes(
    dtype: &str,
    bytes: &[u8],
    decode: impl Fn(u16) -> f32,
) -> Result<Vec<f32>, SafetensorsTensorDecodeError> {
    let chunks = bytes.chunks_exact(2);
    if !chunks.remainder().is_empty() {
        return Err(SafetensorsTensorDecodeError::InvalidByteLength {
            dtype: dtype.to_string(),
            byte_len: bytes.len(),
        });
    }

    Ok(chunks
        .map(|chunk| decode(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect())
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x03ff;

    match exponent {
        0 => {
            if mantissa == 0 {
                sign * 0.0
            } else {
                sign * 2_f32.powi(-14) * (f32::from(mantissa) / 1024.0)
            }
        }
        0x1f => {
            if mantissa == 0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * 2_f32.powi(i32::from(exponent) - 15) * (1.0 + f32::from(mantissa) / 1024.0),
    }
}

fn f8_e4m3fn_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 3) & 0x0f;
    let mantissa = bits & 0x07;

    match exponent {
        0 => {
            if mantissa == 0 {
                sign * 0.0
            } else {
                sign * 2_f32.powi(-6) * (f32::from(mantissa) / 8.0)
            }
        }
        0x0f if mantissa == 0x07 => f32::NAN,
        _ => sign * 2_f32.powi(i32::from(exponent) - 7) * (1.0 + f32::from(mantissa) / 8.0),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsCheckpointFingerprintEntry {
    pub tensor_name: String,
    pub path: PathBuf,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub absolute_byte_offset: u64,
    pub byte_len: usize,
    pub fnv1a64: u64,
}

impl SafetensorsCheckpointFingerprintEntry {
    fn from_span(
        tensor_name: String,
        span: SafetensorsTensorSpan,
    ) -> Result<Self, ModelArtifactError> {
        let fnv1a64 = span.fnv1a64_checksum()?;
        Ok(Self {
            tensor_name,
            path: span.path,
            dtype: span.metadata.dtype,
            shape: span.metadata.shape,
            absolute_byte_offset: span.absolute_byte_offset,
            byte_len: span.byte_len,
            fnv1a64,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsLayerTensorSpan {
    pub tensor_name: String,
    pub layer_id: usize,
    pub suffix: String,
    pub span: SafetensorsTensorSpan,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsLayerTensorCatalog {
    tensors: BTreeMap<(usize, String), SafetensorsLayerTensorSpan>,
}

impl SafetensorsLayerTensorCatalog {
    pub fn from_safetensors_manifest(
        manifest: &SafetensorsManifest,
    ) -> Result<Self, ModelArtifactError> {
        let mut tensors = BTreeMap::new();
        for entry in manifest.layer_tensor_spans()? {
            let key = (entry.layer_id, entry.suffix.clone());
            match tensors.entry(key) {
                std::collections::btree_map::Entry::Vacant(slot) => {
                    slot.insert(entry);
                }
                std::collections::btree_map::Entry::Occupied(_) => {
                    return Err(invalid_safetensors_data(
                        &entry.span.path,
                        format!(
                            "duplicate layer tensor suffix for layer {}: {}",
                            entry.layer_id, entry.suffix
                        ),
                    ));
                }
            }
        }

        Ok(Self { tensors })
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn layer_ids(&self) -> impl Iterator<Item = usize> + '_ {
        let mut previous = None;
        self.tensors.keys().filter_map(move |(layer_id, _)| {
            if previous == Some(*layer_id) {
                return None;
            }
            previous = Some(*layer_id);
            Some(*layer_id)
        })
    }

    pub fn suffixes(&self, layer_id: usize) -> impl Iterator<Item = &str> + '_ {
        self.tensors
            .range((layer_id, String::new())..)
            .take_while(move |((entry_layer_id, _), _)| *entry_layer_id == layer_id)
            .map(|((_, suffix), _)| suffix.as_str())
    }

    pub fn span(&self, layer_id: usize, suffix: &str) -> Option<&SafetensorsLayerTensorSpan> {
        self.tensors.get(&(layer_id, suffix.to_string()))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SafetensorsRoutedExpertProjection {
    Gate,
    Up,
    Down,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsRoutedExpertWeightSpan {
    pub tensor_name: String,
    pub layer_id: usize,
    pub expert_id: usize,
    pub projection: SafetensorsRoutedExpertProjection,
    pub span: SafetensorsTensorSpan,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsRoutedExpertWeightGroup {
    pub layer_id: usize,
    pub expert_id: usize,
    pub gate: SafetensorsTensorSpan,
    pub up: SafetensorsTensorSpan,
    pub down: SafetensorsTensorSpan,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsRoutedExpertWeightCatalog {
    groups: BTreeMap<(usize, usize), SafetensorsRoutedExpertWeightGroup>,
}

impl SafetensorsRoutedExpertWeightCatalog {
    pub fn from_local_model_artifacts(
        artifacts: &LocalModelArtifacts,
    ) -> Result<Self, ModelArtifactError> {
        let groups = artifacts
            .safetensors()
            .routed_expert_weight_groups()?
            .into_iter()
            .collect::<Vec<_>>();
        artifacts.validate_routed_expert_checkpoint_coverage_for_groups(&groups)?;
        let groups = groups
            .into_iter()
            .map(|group| ((group.layer_id, group.expert_id), group))
            .collect();

        Ok(Self { groups })
    }

    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    pub fn coordinates(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.groups.keys().copied()
    }

    pub fn layer_ids(&self) -> impl Iterator<Item = usize> + '_ {
        let mut previous = None;
        self.groups.keys().filter_map(move |(layer_id, _)| {
            if previous == Some(*layer_id) {
                return None;
            }
            previous = Some(*layer_id);
            Some(*layer_id)
        })
    }

    pub fn layer(&self, layer_id: usize) -> Option<SafetensorsRoutedExpertLayerWeights<'_>> {
        let groups = self
            .groups
            .range((layer_id, 0)..=(layer_id, usize::MAX))
            .map(|(_, group)| group)
            .collect::<Vec<_>>();
        if groups.is_empty() {
            return None;
        }

        Some(SafetensorsRoutedExpertLayerWeights { layer_id, groups })
    }

    pub fn group(
        &self,
        layer_id: usize,
        expert_id: usize,
    ) -> Option<&SafetensorsRoutedExpertWeightGroup> {
        self.groups.get(&(layer_id, expert_id))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsRoutedExpertLayerWeights<'a> {
    layer_id: usize,
    groups: Vec<&'a SafetensorsRoutedExpertWeightGroup>,
}

impl<'a> SafetensorsRoutedExpertLayerWeights<'a> {
    pub fn layer_id(&self) -> usize {
        self.layer_id
    }

    pub fn expert_count(&self) -> usize {
        self.groups.len()
    }

    pub fn expert_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.groups.iter().map(|group| group.expert_id)
    }

    pub fn group(&self, expert_id: usize) -> Option<&'a SafetensorsRoutedExpertWeightGroup> {
        self.groups
            .iter()
            .copied()
            .find(|group| group.expert_id == expert_id)
    }

    pub fn groups(&self) -> impl Iterator<Item = &'a SafetensorsRoutedExpertWeightGroup> + '_ {
        self.groups.iter().copied()
    }
}

#[derive(Debug)]
struct RoutedExpertWeightGroupBuilder {
    layer_id: usize,
    expert_id: usize,
    gate: Option<SafetensorsTensorSpan>,
    up: Option<SafetensorsTensorSpan>,
    down: Option<SafetensorsTensorSpan>,
    error_path: Option<PathBuf>,
}

impl RoutedExpertWeightGroupBuilder {
    fn new(layer_id: usize, expert_id: usize) -> Self {
        Self {
            layer_id,
            expert_id,
            gate: None,
            up: None,
            down: None,
            error_path: None,
        }
    }

    fn insert(
        &mut self,
        weight: SafetensorsRoutedExpertWeightSpan,
    ) -> Result<(), ModelArtifactError> {
        self.error_path
            .get_or_insert_with(|| weight.span.path.clone());
        let slot = match weight.projection {
            SafetensorsRoutedExpertProjection::Gate => &mut self.gate,
            SafetensorsRoutedExpertProjection::Up => &mut self.up,
            SafetensorsRoutedExpertProjection::Down => &mut self.down,
        };
        if slot.is_some() {
            return Err(invalid_safetensors_data(
                &weight.span.path,
                format!(
                    "duplicate routed expert projection for layer {} expert {}",
                    self.layer_id, self.expert_id
                ),
            ));
        }

        *slot = Some(weight.span);
        Ok(())
    }

    fn finish(self) -> Result<SafetensorsRoutedExpertWeightGroup, ModelArtifactError> {
        let path = self
            .error_path
            .as_deref()
            .unwrap_or_else(|| Path::new("<unknown>"));
        let gate = self.gate.ok_or_else(|| {
            invalid_safetensors_data(
                path,
                format!(
                    "routed expert weight group for layer {} expert {} is missing gate projection",
                    self.layer_id, self.expert_id
                ),
            )
        })?;
        let up = self.up.ok_or_else(|| {
            invalid_safetensors_data(
                path,
                format!(
                    "routed expert weight group for layer {} expert {} is missing up projection",
                    self.layer_id, self.expert_id
                ),
            )
        })?;
        let down = self.down.ok_or_else(|| {
            invalid_safetensors_data(
                path,
                format!(
                    "routed expert weight group for layer {} expert {} is missing down projection",
                    self.layer_id, self.expert_id
                ),
            )
        })?;

        Ok(SafetensorsRoutedExpertWeightGroup {
            layer_id: self.layer_id,
            expert_id: self.expert_id,
            gate,
            up,
            down,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsTensorSpan {
    pub path: PathBuf,
    pub metadata: SafetensorsTensorMetadata,
    pub absolute_byte_offset: u64,
    pub byte_len: usize,
}

impl SafetensorsTensorSpan {
    pub fn fnv1a64_checksum(&self) -> Result<u64, ModelArtifactError> {
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

        let mut remaining = self.byte_len;
        let mut buffer = [0_u8; 64 * 1024];
        let mut checksum = FNV1A64_OFFSET_BASIS;
        while remaining > 0 {
            let read_len = remaining.min(buffer.len());
            file.read_exact(&mut buffer[..read_len]).map_err(|error| {
                ModelArtifactError::ReadWeightShard {
                    path: self.path.clone(),
                    message: error.to_string(),
                }
            })?;
            checksum = fnv1a64_update(checksum, &buffer[..read_len]);
            remaining -= read_len;
        }

        Ok(checksum)
    }

    pub fn read(&self) -> Result<Option<SafetensorsTensorData>, ModelArtifactError> {
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

        Ok(Some(SafetensorsTensorData {
            metadata: self.metadata.clone(),
            bytes,
        }))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsHeader {
    header_len: usize,
    tensors: BTreeMap<String, SafetensorsTensorMetadata>,
}

impl SafetensorsHeader {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ModelArtifactError> {
        let path = path.as_ref();
        let mut file =
            fs::File::open(path).map_err(|error| ModelArtifactError::ReadWeightShard {
                path: path.to_path_buf(),
                message: error.to_string(),
            })?;
        let mut header_len_bytes = [0_u8; 8];
        file.read_exact(&mut header_len_bytes).map_err(|error| {
            ModelArtifactError::ReadWeightShard {
                path: path.to_path_buf(),
                message: error.to_string(),
            }
        })?;
        let header_len = u64::from_le_bytes(header_len_bytes);
        let header_len = usize::try_from(header_len).map_err(|_| {
            ModelArtifactError::InvalidSafetensorsHeader {
                path: path.to_path_buf(),
                message: "header length does not fit in usize".to_string(),
            }
        })?;
        let mut header = vec![0_u8; header_len];
        file.read_exact(&mut header)
            .map_err(|error| ModelArtifactError::ReadWeightShard {
                path: path.to_path_buf(),
                message: error.to_string(),
            })?;
        let value: serde_json::Value = serde_json::from_slice(&header).map_err(|error| {
            ModelArtifactError::InvalidSafetensorsHeader {
                path: path.to_path_buf(),
                message: error.to_string(),
            }
        })?;
        let header_object =
            value
                .as_object()
                .ok_or_else(|| ModelArtifactError::InvalidSafetensorsHeader {
                    path: path.to_path_buf(),
                    message: "header must be a JSON object".to_string(),
                })?;
        let mut tensors = BTreeMap::new();
        for (tensor_name, metadata) in header_object {
            if tensor_name == "__metadata__" {
                continue;
            }
            let Some(metadata) = metadata.as_object() else {
                continue;
            };
            tensors.insert(
                tensor_name.clone(),
                parse_tensor_metadata(&path.to_path_buf(), tensor_name, metadata)?,
            );
        }

        Ok(Self {
            header_len,
            tensors,
        })
    }

    pub fn tensor_metadata(&self, tensor_name: &str) -> Option<SafetensorsTensorMetadata> {
        self.tensors.get(tensor_name).cloned()
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn tensor_span(
        &self,
        path: impl AsRef<Path>,
        tensor_name: &str,
    ) -> Result<Option<SafetensorsTensorSpan>, ModelArtifactError> {
        let path = path.as_ref();
        let Some(metadata) = self.tensors.get(tensor_name).cloned() else {
            return Ok(None);
        };
        let [start, end] = metadata.data_offsets;
        let tensor_len = end.checked_sub(start).ok_or_else(|| {
            invalid_safetensors_data(
                path,
                format!("tensor {tensor_name} data_offsets start is after end"),
            )
        })?;
        let expected_tensor_len = expected_tensor_byte_len(path, tensor_name, &metadata)?;
        if tensor_len != expected_tensor_len {
            return Err(invalid_safetensors_data(
                path,
                format!(
                    "tensor {tensor_name} metadata expects {expected_tensor_len} bytes but data_offsets describe {tensor_len} bytes"
                ),
            ));
        }
        let payload_start = 8_usize.checked_add(self.header_len).ok_or_else(|| {
            invalid_safetensors_data(path, "safetensors payload start offset overflowed")
        })?;
        let absolute_start = payload_start.checked_add(start).ok_or_else(|| {
            invalid_safetensors_data(
                path,
                format!("tensor {tensor_name} start offset overflowed"),
            )
        })?;
        let absolute_end = payload_start.checked_add(end).ok_or_else(|| {
            invalid_safetensors_data(path, format!("tensor {tensor_name} end offset overflowed"))
        })?;
        let absolute_start = u64::try_from(absolute_start).map_err(|_| {
            invalid_safetensors_data(
                path,
                format!("tensor {tensor_name} start offset overflows u64"),
            )
        })?;
        let absolute_end = u64::try_from(absolute_end).map_err(|_| {
            invalid_safetensors_data(
                path,
                format!("tensor {tensor_name} end offset overflows u64"),
            )
        })?;

        let file = fs::File::open(path).map_err(|error| ModelArtifactError::ReadWeightShard {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
        let shard_len = file
            .metadata()
            .map_err(|error| ModelArtifactError::ReadWeightShard {
                path: path.to_path_buf(),
                message: error.to_string(),
            })?
            .len();
        if absolute_end > shard_len {
            return Err(invalid_safetensors_data(
                path,
                format!("tensor {tensor_name} payload extends past end of shard"),
            ));
        }

        Ok(Some(SafetensorsTensorSpan {
            path: path.to_path_buf(),
            metadata,
            absolute_byte_offset: absolute_start,
            byte_len: tensor_len,
        }))
    }

    pub fn read_tensor(
        &self,
        path: impl AsRef<Path>,
        tensor_name: &str,
    ) -> Result<Option<SafetensorsTensorData>, ModelArtifactError> {
        let Some(span) = self.tensor_span(path, tensor_name)? else {
            return Ok(None);
        };
        span.read()
    }

    pub fn tensor_span_entries(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<Vec<(String, SafetensorsTensorSpan)>, ModelArtifactError> {
        let path = path.as_ref();
        let mut entries = Vec::with_capacity(self.tensors.len());
        for tensor_name in self.tensor_names() {
            if let Some(span) = self.tensor_span(path, tensor_name)? {
                entries.push((tensor_name.to_string(), span));
            }
        }

        Ok(entries)
    }

    fn routed_expert_weight_dtype(&self) -> Option<String> {
        self.tensors
            .iter()
            .find(|(tensor_name, _)| is_routed_expert_weight(tensor_name))
            .map(|(_, metadata)| metadata.dtype.clone())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelArtifactError {
    ModelPathNotLocalDirectory { path: PathBuf },
    ReadModelConfig { path: PathBuf, message: String },
    InvalidModelConfig { path: PathBuf, message: String },
    ReadSafetensorsIndex { path: PathBuf, message: String },
    InvalidSafetensorsIndex { path: PathBuf, message: String },
    MissingWeightShard { path: PathBuf },
    NoSafetensorsWeights { path: PathBuf },
    ReadModelDirectory { path: PathBuf, message: String },
    ReadWeightShard { path: PathBuf, message: String },
    InvalidSafetensorsHeader { path: PathBuf, message: String },
    InvalidSafetensorsData { path: PathBuf, message: String },
}

impl fmt::Display for ModelArtifactError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelPathNotLocalDirectory { path } => {
                write!(
                    formatter,
                    "model path is not a local directory: {}",
                    path.display()
                )
            }
            Self::ReadModelConfig { path, message } => {
                write!(
                    formatter,
                    "failed to read model config {}: {message}",
                    path.display()
                )
            }
            Self::InvalidModelConfig { path, message } => write!(
                formatter,
                "failed to parse model config {}: {message}",
                path.display()
            ),
            Self::ReadSafetensorsIndex { path, message } => write!(
                formatter,
                "failed to read safetensors index {}: {message}",
                path.display()
            ),
            Self::InvalidSafetensorsIndex { path, message } => write!(
                formatter,
                "failed to parse safetensors index {}: {message}",
                path.display()
            ),
            Self::MissingWeightShard { path } => {
                write!(
                    formatter,
                    "safetensors index references missing shard {}",
                    path.display()
                )
            }
            Self::NoSafetensorsWeights { path } => {
                write!(
                    formatter,
                    "no safetensors weights found under {}",
                    path.display()
                )
            }
            Self::ReadModelDirectory { path, message } => {
                write!(
                    formatter,
                    "failed to read model directory {}: {message}",
                    path.display()
                )
            }
            Self::ReadWeightShard { path, message } => {
                write!(
                    formatter,
                    "failed to read safetensors shard {}: {message}",
                    path.display()
                )
            }
            Self::InvalidSafetensorsHeader { path, message } => {
                write!(
                    formatter,
                    "failed to parse safetensors header {}: {message}",
                    path.display()
                )
            }
            Self::InvalidSafetensorsData { path, message } => {
                write!(
                    formatter,
                    "failed to read safetensors tensor data {}: {message}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for ModelArtifactError {}

fn sorted_safetensors_files(model_path: &Path) -> Result<Vec<PathBuf>, ModelArtifactError> {
    let entries =
        fs::read_dir(model_path).map_err(|error| ModelArtifactError::ReadModelDirectory {
            path: model_path.to_path_buf(),
            message: error.to_string(),
        })?;
    let mut shard_paths = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| ModelArtifactError::ReadModelDirectory {
            path: model_path.to_path_buf(),
            message: error.to_string(),
        })?;
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|extension| extension == "safetensors")
        {
            shard_paths.push(path);
        }
    }
    shard_paths.sort();
    Ok(shard_paths)
}

fn read_string_field(value: &serde_json::Value, field: &str) -> Option<String> {
    value.get(field)?.as_str().map(ToString::to_string)
}

fn read_string_array_field(value: &serde_json::Value, field: &str) -> Vec<String> {
    let Some(values) = value.get(field).and_then(serde_json::Value::as_array) else {
        return Vec::new();
    };
    values
        .iter()
        .filter_map(serde_json::Value::as_str)
        .map(ToString::to_string)
        .collect()
}

fn read_usize_field(
    value: &serde_json::Value,
    field: &'static str,
    config_path: &Path,
) -> Result<Option<usize>, ModelArtifactError> {
    let Some(raw) = value.get(field) else {
        return Ok(None);
    };
    if raw.is_null() {
        return Ok(None);
    }
    let Some(value) = raw.as_u64() else {
        return Err(ModelArtifactError::InvalidModelConfig {
            path: config_path.to_path_buf(),
            message: format!("field {field} must be an unsigned integer"),
        });
    };
    usize::try_from(value)
        .map(Some)
        .map_err(|_| ModelArtifactError::InvalidModelConfig {
            path: config_path.to_path_buf(),
            message: format!("field {field} does not fit in usize"),
        })
}

fn read_u32_or_array_field(
    value: &serde_json::Value,
    field: &'static str,
    config_path: &Path,
) -> Result<Vec<u32>, ModelArtifactError> {
    let Some(raw) = value.get(field) else {
        return Ok(Vec::new());
    };
    if raw.is_null() {
        return Ok(Vec::new());
    }
    if let Some(values) = raw.as_array() {
        return values
            .iter()
            .map(|value| read_u32_value(value, field, config_path))
            .collect();
    }

    Ok(vec![read_u32_value(raw, field, config_path)?])
}

fn read_u32_value(
    value: &serde_json::Value,
    field: &'static str,
    config_path: &Path,
) -> Result<u32, ModelArtifactError> {
    let Some(value) = value.as_u64() else {
        return Err(ModelArtifactError::InvalidModelConfig {
            path: config_path.to_path_buf(),
            message: format!(
                "field {field} must be an unsigned integer or array of unsigned integers"
            ),
        });
    };
    u32::try_from(value).map_err(|_| ModelArtifactError::InvalidModelConfig {
        path: config_path.to_path_buf(),
        message: format!("field {field} does not fit in u32"),
    })
}

fn read_f64_field(
    value: &serde_json::Value,
    field: &'static str,
    config_path: &Path,
) -> Result<Option<HfConfigFloat>, ModelArtifactError> {
    let Some(raw) = value.get(field) else {
        return Ok(None);
    };
    if raw.is_null() {
        return Ok(None);
    }
    let Some(value) = raw.as_f64() else {
        return Err(ModelArtifactError::InvalidModelConfig {
            path: config_path.to_path_buf(),
            message: format!("field {field} must be a finite number"),
        });
    };
    if !value.is_finite() {
        return Err(ModelArtifactError::InvalidModelConfig {
            path: config_path.to_path_buf(),
            message: format!("field {field} must be a finite number"),
        });
    }
    Ok(Some(HfConfigFloat(value)))
}

fn read_bool_field(
    value: &serde_json::Value,
    field: &'static str,
    config_path: &Path,
) -> Result<Option<bool>, ModelArtifactError> {
    let Some(raw) = value.get(field) else {
        return Ok(None);
    };
    if raw.is_null() {
        return Ok(None);
    }
    raw.as_bool()
        .map(Some)
        .ok_or_else(|| ModelArtifactError::InvalidModelConfig {
            path: config_path.to_path_buf(),
            message: format!("field {field} must be a boolean"),
        })
}

fn parse_tensor_metadata(
    path: &Path,
    tensor_name: &str,
    metadata: &serde_json::Map<String, serde_json::Value>,
) -> Result<SafetensorsTensorMetadata, ModelArtifactError> {
    let dtype = metadata
        .get("dtype")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ModelArtifactError::InvalidSafetensorsHeader {
            path: path.to_path_buf(),
            message: format!("tensor {tensor_name} is missing string dtype"),
        })?
        .to_string();
    let shape = metadata
        .get("shape")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| ModelArtifactError::InvalidSafetensorsHeader {
            path: path.to_path_buf(),
            message: format!("tensor {tensor_name} is missing shape array"),
        })?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| ModelArtifactError::InvalidSafetensorsHeader {
                    path: path.to_path_buf(),
                    message: format!("tensor {tensor_name} shape contains a non-usize dimension"),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let offsets = metadata
        .get("data_offsets")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| ModelArtifactError::InvalidSafetensorsHeader {
            path: path.to_path_buf(),
            message: format!("tensor {tensor_name} is missing data_offsets array"),
        })?;
    if offsets.len() != 2 {
        return Err(ModelArtifactError::InvalidSafetensorsHeader {
            path: path.to_path_buf(),
            message: format!("tensor {tensor_name} data_offsets must have two entries"),
        });
    }
    let mut data_offsets = [0_usize; 2];
    for (index, value) in offsets.iter().enumerate() {
        data_offsets[index] = value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| ModelArtifactError::InvalidSafetensorsHeader {
                path: path.to_path_buf(),
                message: format!("tensor {tensor_name} data_offsets contains a non-usize offset"),
            })?;
    }

    Ok(SafetensorsTensorMetadata {
        dtype,
        shape,
        data_offsets,
    })
}

fn invalid_safetensors_data(path: &Path, message: impl Into<String>) -> ModelArtifactError {
    ModelArtifactError::InvalidSafetensorsData {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

fn read_required_tensor_span(
    span: &SafetensorsTensorSpan,
) -> Result<SafetensorsTensorData, ModelArtifactError> {
    span.read()?.ok_or_else(|| {
        invalid_safetensors_data(
            &span.path,
            format!(
                "missing safetensors payload at byte offset {}",
                span.absolute_byte_offset
            ),
        )
    })
}

fn expected_tensor_byte_len(
    path: &Path,
    tensor_name: &str,
    metadata: &SafetensorsTensorMetadata,
) -> Result<usize, ModelArtifactError> {
    let element_count = metadata.element_count().ok_or_else(|| {
        invalid_safetensors_data(
            path,
            format!("tensor {tensor_name} element count overflowed"),
        )
    })?;
    let dtype_byte_width = metadata.dtype_byte_width().ok_or_else(|| {
        invalid_safetensors_data(
            path,
            format!(
                "tensor {tensor_name} has unsupported safetensors dtype {}",
                metadata.dtype
            ),
        )
    })?;

    element_count.checked_mul(dtype_byte_width).ok_or_else(|| {
        invalid_safetensors_data(path, format!("tensor {tensor_name} byte length overflowed"))
    })
}

fn shape_element_count(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .copied()
        .try_fold(1_usize, |count, dimension| count.checked_mul(dimension))
}

fn safetensors_dtype_byte_width(dtype: &str) -> Option<usize> {
    match dtype {
        "BOOL" | "I8" | "U8" | "F8_E4M3" | "F8_E5M2" | "F8_E8M0" => Some(1),
        "I16" | "U16" | "F16" | "BF16" => Some(2),
        "I32" | "U32" | "F32" => Some(4),
        "I64" | "U64" | "F64" => Some(8),
        _ => None,
    }
}

const FNV1A64_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV1A64_PRIME: u64 = 0x100000001b3;

fn fnv1a64_update(mut checksum: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        checksum ^= u64::from(*byte);
        checksum = checksum.wrapping_mul(FNV1A64_PRIME);
    }
    checksum
}

fn is_routed_expert_weight(tensor_name: &str) -> bool {
    parse_routed_expert_weight_name(tensor_name).is_some()
}

fn parse_layer_tensor_name(tensor_name: &str) -> Option<(usize, String)> {
    let remainder = tensor_name.strip_prefix("model.layers.")?;
    let (layer_id, suffix) = remainder.split_once('.')?;
    let layer_id = parse_usize_segment(layer_id)?;
    if suffix.is_empty() {
        return None;
    }

    Some((layer_id, suffix.to_string()))
}

fn parse_routed_expert_weight_name(
    tensor_name: &str,
) -> Option<(usize, usize, SafetensorsRoutedExpertProjection)> {
    let parts = tensor_name.split('.').collect::<Vec<_>>();
    let layer_id = parts.windows(2).find_map(|window| {
        if window[0] == "layers" {
            parse_usize_segment(window[1])
        } else {
            None
        }
    })?;

    let experts_index = parts.iter().position(|part| *part == "experts")?;
    let expert_id = parts
        .get(experts_index + 1)
        .and_then(|part| parse_usize_segment(part))?;
    let projection = match *parts.get(experts_index + 2)? {
        "w1" | "gate_proj" => SafetensorsRoutedExpertProjection::Gate,
        "w2" | "down_proj" => SafetensorsRoutedExpertProjection::Down,
        "w3" | "up_proj" => SafetensorsRoutedExpertProjection::Up,
        _ => return None,
    };
    if parts.get(experts_index + 3) != Some(&"weight") || experts_index + 4 != parts.len() {
        return None;
    }

    Some((layer_id, expert_id, projection))
}

fn parse_usize_segment(value: &str) -> Option<usize> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }

    value.parse().ok()
}
