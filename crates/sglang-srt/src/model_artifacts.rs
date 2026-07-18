use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;

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
        topology: &CheckpointTopology,
    ) -> Result<RoutedExpertCheckpointCoverage, ModelArtifactError> {
        let groups = self.safetensors.routed_expert_weight_groups()?;
        self.validate_routed_expert_checkpoint_coverage_for_groups(topology, &groups)
    }

    pub fn routed_expert_weight_catalog(
        &self,
    ) -> Result<SafetensorsRoutedExpertWeightCatalog, ModelArtifactError> {
        SafetensorsRoutedExpertWeightCatalog::from_local_model_artifacts(self)
    }

    pub fn checkpoint_catalog(&self) -> Result<LocalModelCheckpointCatalog, ModelArtifactError> {
        LocalModelCheckpointCatalog::from_local_model_artifacts(self)
    }

    fn validate_routed_expert_checkpoint_coverage_for_groups(
        &self,
        topology: &CheckpointTopology,
        groups: &[SafetensorsRoutedExpertWeightGroup],
    ) -> Result<RoutedExpertCheckpointCoverage, ModelArtifactError> {
        let expected_coordinates = topology.routed_expert_coordinates();
        let expected_group_count = expected_coordinates.len();
        let expected_weight_count = expected_group_count
            .checked_mul(topology.routed_expert_weights_per_group())
            .ok_or_else(|| {
                invalid_safetensors_data(
                    &self.model_path,
                    "expected routed expert weight count overflowed",
                )
            })?;

        let actual_group_count = groups.len();
        let actual_weight_count = groups.iter().try_fold(0_usize, |count, group| {
            count.checked_add(group.tensor_count()).ok_or_else(|| {
                invalid_safetensors_data(
                    &self.model_path,
                    "actual routed expert weight count overflowed",
                )
            })
        })?;

        if actual_group_count != expected_group_count {
            return Err(invalid_safetensors_data(
                &self.model_path,
                format!(
                    "expected {expected_group_count} routed expert groups from model config but found {actual_group_count}"
                ),
            ));
        }
        if actual_weight_count != expected_weight_count {
            return Err(invalid_safetensors_data(
                &self.model_path,
                format!(
                    "expected {expected_weight_count} routed expert tensors but found {actual_weight_count}"
                ),
            ));
        }

        let actual_coordinates: BTreeSet<(usize, usize)> = groups
            .iter()
            .map(|group| (group.layer_id, group.expert_id))
            .collect();
        if actual_coordinates != *expected_coordinates {
            let missing = expected_coordinates
                .difference(&actual_coordinates)
                .next()
                .map(|(layer_id, expert_id)| {
                    format!(
                        "missing expected routed expert group layer {layer_id} expert {expert_id}"
                    )
                });
            let unexpected = actual_coordinates
                .difference(expected_coordinates)
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

    pub fn validate_checkpoint_topology(
        &self,
        topology: &CheckpointTopology,
    ) -> Result<(), ModelArtifactError> {
        for tensor in topology.required_tensors() {
            let metadata = self
                .safetensors
                .tensor_metadata(tensor.name())?
                .ok_or_else(|| {
                    invalid_safetensors_data(
                        &self.model_path,
                        format!("missing checkpoint tensor {}", tensor.name()),
                    )
                })?;
            if let Some(expected_shape) = tensor.expected_shape()
                && metadata.shape != expected_shape
            {
                return Err(invalid_safetensors_data(
                    &self.model_path,
                    format!(
                        "checkpoint tensor {} shape {:?} does not match expected {:?}",
                        tensor.name(),
                        metadata.shape,
                        expected_shape
                    ),
                ));
            }
        }

        self.validate_routed_expert_checkpoint_coverage(topology)?;
        Ok(())
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
pub struct CheckpointTensorRequirement {
    name: String,
    expected_shape: Option<Vec<usize>>,
}

impl CheckpointTensorRequirement {
    pub fn present(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            expected_shape: None,
        }
    }

    pub fn with_shape(name: impl Into<String>, expected_shape: impl Into<Vec<usize>>) -> Self {
        Self {
            name: name.into(),
            expected_shape: Some(expected_shape.into()),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn expected_shape(&self) -> Option<&[usize]> {
        self.expected_shape.as_deref()
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CheckpointTopology {
    required_tensors: Vec<CheckpointTensorRequirement>,
    routed_expert_coordinates: BTreeSet<(usize, usize)>,
    routed_expert_weights_per_group: usize,
}

impl CheckpointTopology {
    pub fn new(required_tensors: Vec<CheckpointTensorRequirement>) -> Self {
        Self {
            required_tensors,
            routed_expert_coordinates: BTreeSet::new(),
            routed_expert_weights_per_group: 0,
        }
    }

    pub fn with_routed_experts(
        mut self,
        coordinates: impl IntoIterator<Item = (usize, usize)>,
        weights_per_group: usize,
    ) -> Self {
        self.routed_expert_coordinates = coordinates.into_iter().collect();
        self.routed_expert_weights_per_group = weights_per_group;
        self
    }

    pub fn required_tensors(&self) -> &[CheckpointTensorRequirement] {
        &self.required_tensors
    }

    pub fn routed_expert_coordinates(&self) -> &BTreeSet<(usize, usize)> {
        &self.routed_expert_coordinates
    }

    pub fn routed_expert_weights_per_group(&self) -> usize {
        self.routed_expert_weights_per_group
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalModelCheckpointCatalog {
    model_path: PathBuf,
    safetensors: SafetensorsManifest,
    layer_tensors: SafetensorsLayerTensorCatalog,
    quantized_linears: SafetensorsQuantizedLinearWeightCatalog,
    routed_experts: SafetensorsRoutedExpertWeightCatalog,
}

impl LocalModelCheckpointCatalog {
    pub fn from_local_model_artifacts(
        artifacts: &LocalModelArtifacts,
    ) -> Result<Self, ModelArtifactError> {
        let layer_tensors = artifacts.safetensors().layer_tensor_catalog()?;
        let quantized_linears = SafetensorsQuantizedLinearWeightCatalog::from_safetensors_manifest(
            artifacts.safetensors(),
        )?;
        let routed_experts = artifacts.routed_expert_weight_catalog()?;

        Ok(Self {
            model_path: artifacts.model_path().to_path_buf(),
            safetensors: artifacts.safetensors().clone(),
            layer_tensors,
            quantized_linears,
            routed_experts,
        })
    }

    pub fn layer_tensors(&self) -> &SafetensorsLayerTensorCatalog {
        &self.layer_tensors
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn safetensors(&self) -> &SafetensorsManifest {
        &self.safetensors
    }

    pub fn quantized_linear_weights(&self) -> &SafetensorsQuantizedLinearWeightCatalog {
        &self.quantized_linears
    }

    pub fn routed_experts(&self) -> &SafetensorsRoutedExpertWeightCatalog {
        &self.routed_experts
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HfModelConfig {
    raw_document: serde_json::Value,
    pub model_type: Option<String>,
    pub architectures: Vec<String>,
    pub eos_token_ids: Vec<u32>,
}

impl Default for HfModelConfig {
    fn default() -> Self {
        Self {
            raw_document: serde_json::Value::Object(serde_json::Map::new()),
            model_type: None,
            architectures: Vec::new(),
            eos_token_ids: Vec::new(),
        }
    }
}

impl HfModelConfig {
    pub fn from_json_value(value: serde_json::Value) -> Result<Self, ModelArtifactError> {
        Self::from_document(value, Path::new("<in-memory-config>"))
    }

    pub fn from_model_path(path: impl AsRef<Path>) -> Result<Self, ModelArtifactError> {
        let path = path.as_ref();
        let model_path = resolve_model_path(path);
        match Self::from_resolved_model_path(&model_path) {
            Ok(config) => Ok(config),
            Err(_) if looks_like_hf_model_id(path) => {
                let config_path = download_hf_model_config(path)?;
                Self::from_resolved_config_path(&config_path)
            }
            Err(error) => Err(error),
        }
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
        Self::from_resolved_config_path(&path.join("config.json"))
    }

    fn from_resolved_config_path(config_path: &Path) -> Result<Self, ModelArtifactError> {
        let raw = fs::read_to_string(config_path).map_err(|error| {
            ModelArtifactError::ReadModelConfig {
                path: config_path.to_path_buf(),
                message: error.to_string(),
            }
        })?;
        let value: serde_json::Value =
            serde_json::from_str(&raw).map_err(|error| ModelArtifactError::InvalidModelConfig {
                path: config_path.to_path_buf(),
                message: error.to_string(),
            })?;

        Self::from_document(value, config_path)
    }

    fn from_document(
        value: serde_json::Value,
        config_path: &Path,
    ) -> Result<Self, ModelArtifactError> {
        let text = value
            .get("text_config")
            .filter(|value| value.is_object())
            .unwrap_or(&value);

        Ok(Self {
            raw_document: value.clone(),
            model_type: read_string_field(&value, "model_type"),
            architectures: read_string_array_field(&value, "architectures"),
            eos_token_ids: read_u32_or_array_field(text, "eos_token_id", config_path)?,
        })
    }

    pub fn raw_document(&self) -> &serde_json::Value {
        &self.raw_document
    }

    pub(crate) fn text_document(&self) -> &serde_json::Value {
        self.raw_document
            .get("text_config")
            .filter(|value| value.is_object())
            .unwrap_or(&self.raw_document)
    }

    pub(crate) fn parse_text_config<T: DeserializeOwned>(&self) -> Result<T, String> {
        serde_path_to_error::deserialize(self.text_document().clone())
            .map_err(|error| error.to_string())
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

    pub fn quantized_linear_weight_spans(
        &self,
    ) -> Result<Vec<SafetensorsQuantizedLinearWeightSpan>, ModelArtifactError> {
        let entries = self
            .tensor_span_entries()?
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        let mut spans = Vec::new();

        for (tensor_name, weight) in &entries {
            if !is_quantized_linear_weight_tensor(tensor_name, weight) {
                continue;
            }

            let Some((scale_tensor_name, scale_kind, scale)) =
                quantized_linear_scale_tensor(tensor_name, &entries)
            else {
                continue;
            };
            spans.push(SafetensorsQuantizedLinearWeightSpan {
                tensor_name: tensor_name.clone(),
                scale_tensor_name,
                scale_kind,
                weight: weight.clone(),
                scale,
            });
        }

        Ok(spans)
    }

    pub fn routed_expert_weight_spans(
        &self,
    ) -> Result<Vec<SafetensorsRoutedExpertWeightSpan>, ModelArtifactError> {
        let spans = self
            .tensor_span_entries()?
            .into_iter()
            .filter_map(|(tensor_name, span)| {
                parse_routed_expert_weight_name(&tensor_name).map(
                    |(layer_id, expert_id, projection, component)| {
                        SafetensorsRoutedExpertWeightSpan {
                            tensor_name,
                            layer_id,
                            expert_id,
                            projection,
                            component,
                            span,
                        }
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
pub enum SafetensorsQuantizedLinearScaleKind {
    WeightScaleInv,
    WeightScale,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsQuantizedLinearWeightSpan {
    pub tensor_name: String,
    pub scale_tensor_name: String,
    pub scale_kind: SafetensorsQuantizedLinearScaleKind,
    pub weight: SafetensorsTensorSpan,
    pub scale: SafetensorsTensorSpan,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsQuantizedLinearWeightCatalog {
    weights: BTreeMap<String, SafetensorsQuantizedLinearWeightSpan>,
}

impl SafetensorsQuantizedLinearWeightCatalog {
    pub fn from_safetensors_manifest(
        manifest: &SafetensorsManifest,
    ) -> Result<Self, ModelArtifactError> {
        Ok(Self {
            weights: manifest
                .quantized_linear_weight_spans()?
                .into_iter()
                .map(|span| (span.tensor_name.clone(), span))
                .collect(),
        })
    }

    pub fn weight_count(&self) -> usize {
        self.weights.len()
    }

    pub fn span(&self, tensor_name: &str) -> Option<&SafetensorsQuantizedLinearWeightSpan> {
        self.weights.get(tensor_name)
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SafetensorsRoutedExpertWeightComponent {
    Weight,
    PackedWeight,
    WeightScale,
    WeightShape,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsRoutedExpertWeightSpan {
    pub tensor_name: String,
    pub layer_id: usize,
    pub expert_id: usize,
    pub projection: SafetensorsRoutedExpertProjection,
    pub component: SafetensorsRoutedExpertWeightComponent,
    pub span: SafetensorsTensorSpan,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SafetensorsRoutedExpertProjectionWeights {
    Unquantized { weight: SafetensorsTensorSpan },
    CompressedTensorsInt4(Box<SafetensorsCompressedInt4ProjectionWeights>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsCompressedInt4ProjectionWeights {
    pub packed: SafetensorsTensorSpan,
    pub scale: SafetensorsTensorSpan,
    pub shape: SafetensorsTensorSpan,
}

impl SafetensorsRoutedExpertProjectionWeights {
    pub fn tensor_count(&self) -> usize {
        match self {
            Self::Unquantized { .. } => 1,
            Self::CompressedTensorsInt4(_) => 3,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsRoutedExpertWeightGroup {
    pub layer_id: usize,
    pub expert_id: usize,
    pub gate: SafetensorsRoutedExpertProjectionWeights,
    pub up: SafetensorsRoutedExpertProjectionWeights,
    pub down: SafetensorsRoutedExpertProjectionWeights,
}

impl SafetensorsRoutedExpertWeightGroup {
    pub fn tensor_count(&self) -> usize {
        self.gate.tensor_count() + self.up.tensor_count() + self.down.tensor_count()
    }
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
    gate: RoutedExpertProjectionBuilder,
    up: RoutedExpertProjectionBuilder,
    down: RoutedExpertProjectionBuilder,
    error_path: Option<PathBuf>,
}

#[derive(Debug, Default)]
struct RoutedExpertProjectionBuilder {
    weight: Option<SafetensorsTensorSpan>,
    packed: Option<SafetensorsTensorSpan>,
    scale: Option<SafetensorsTensorSpan>,
    shape: Option<SafetensorsTensorSpan>,
}

impl RoutedExpertWeightGroupBuilder {
    fn new(layer_id: usize, expert_id: usize) -> Self {
        Self {
            layer_id,
            expert_id,
            gate: RoutedExpertProjectionBuilder::default(),
            up: RoutedExpertProjectionBuilder::default(),
            down: RoutedExpertProjectionBuilder::default(),
            error_path: None,
        }
    }

    fn insert(
        &mut self,
        weight: SafetensorsRoutedExpertWeightSpan,
    ) -> Result<(), ModelArtifactError> {
        self.error_path
            .get_or_insert_with(|| weight.span.path.clone());
        let projection = match weight.projection {
            SafetensorsRoutedExpertProjection::Gate => &mut self.gate,
            SafetensorsRoutedExpertProjection::Up => &mut self.up,
            SafetensorsRoutedExpertProjection::Down => &mut self.down,
        };
        projection.insert(weight.component, weight.span, self.layer_id, self.expert_id)
    }

    fn finish(self) -> Result<SafetensorsRoutedExpertWeightGroup, ModelArtifactError> {
        let path = self
            .error_path
            .as_deref()
            .unwrap_or_else(|| Path::new("<unknown>"));
        let gate = self
            .gate
            .finish(path, self.layer_id, self.expert_id, "gate")?;
        let up = self.up.finish(path, self.layer_id, self.expert_id, "up")?;
        let down = self
            .down
            .finish(path, self.layer_id, self.expert_id, "down")?;

        Ok(SafetensorsRoutedExpertWeightGroup {
            layer_id: self.layer_id,
            expert_id: self.expert_id,
            gate,
            up,
            down,
        })
    }
}

impl RoutedExpertProjectionBuilder {
    fn insert(
        &mut self,
        component: SafetensorsRoutedExpertWeightComponent,
        span: SafetensorsTensorSpan,
        layer_id: usize,
        expert_id: usize,
    ) -> Result<(), ModelArtifactError> {
        let slot = match component {
            SafetensorsRoutedExpertWeightComponent::Weight => &mut self.weight,
            SafetensorsRoutedExpertWeightComponent::PackedWeight => &mut self.packed,
            SafetensorsRoutedExpertWeightComponent::WeightScale => &mut self.scale,
            SafetensorsRoutedExpertWeightComponent::WeightShape => &mut self.shape,
        };
        if slot.is_some() {
            return Err(invalid_safetensors_data(
                &span.path,
                format!(
                    "duplicate routed expert weight component {component:?} for layer {layer_id} expert {expert_id}"
                ),
            ));
        }
        *slot = Some(span);
        Ok(())
    }

    fn finish(
        self,
        path: &Path,
        layer_id: usize,
        expert_id: usize,
        projection: &str,
    ) -> Result<SafetensorsRoutedExpertProjectionWeights, ModelArtifactError> {
        match (self.weight, self.packed, self.scale, self.shape) {
            (Some(weight), None, None, None) => {
                Ok(SafetensorsRoutedExpertProjectionWeights::Unquantized { weight })
            }
            (None, Some(packed), Some(scale), Some(shape)) => Ok(
                SafetensorsRoutedExpertProjectionWeights::CompressedTensorsInt4(Box::new(
                    SafetensorsCompressedInt4ProjectionWeights {
                        packed,
                        scale,
                        shape,
                    },
                )),
            ),
            (None, None, None, None) => Err(invalid_safetensors_data(
                path,
                format!(
                    "routed expert weight group for layer {layer_id} expert {expert_id} is missing {projection} projection"
                ),
            )),
            _ => Err(invalid_safetensors_data(
                path,
                format!(
                    "routed expert {projection} projection for layer {layer_id} expert {expert_id} mixes unquantized and incomplete compressed-tensors components"
                ),
            )),
        }
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
                parse_tensor_metadata(path, tensor_name, metadata)?,
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

fn looks_like_hf_model_id(path: &Path) -> bool {
    path.to_str().is_some_and(|model_id| {
        model_id.contains('/')
            && !model_id.starts_with('/')
            && !model_id.starts_with('-')
            && !model_id.contains('\\')
    })
}

fn download_hf_model_config(model_id: &Path) -> Result<PathBuf, ModelArtifactError> {
    let model_id = model_id
        .to_str()
        .expect("looks_like_hf_model_id should only accept UTF-8 repo ids");
    let api = hf_hub_api_builder_from_env().build().map_err(|error| {
        ModelArtifactError::ReadModelConfig {
            path: PathBuf::from(model_id).join("config.json"),
            message: format!("failed to initialize Hugging Face Hub client: {error}"),
        }
    })?;
    api.model(model_id.to_string())
        .get("config.json")
        .map_err(|error| ModelArtifactError::ReadModelConfig {
            path: PathBuf::from(model_id).join("config.json"),
            message: format!("failed to fetch Hugging Face config.json: {error}"),
        })
}

pub(crate) fn hf_hub_api_builder_from_env() -> hf_hub::api::sync::ApiBuilder {
    let mut builder = if let Some(cache) = std::env::var_os("HUGGINGFACE_HUB_CACHE") {
        hf_hub::api::sync::ApiBuilder::from_cache(hf_hub::Cache::new(PathBuf::from(cache)))
    } else {
        hf_hub::api::sync::ApiBuilder::from_env()
    };
    if let Ok(endpoint) = std::env::var("HF_ENDPOINT") {
        builder = builder.with_endpoint(endpoint);
    }
    builder.with_progress(false)
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

fn is_quantized_linear_weight_tensor(tensor_name: &str, span: &SafetensorsTensorSpan) -> bool {
    tensor_name.ends_with(".weight")
        && matches!(span.metadata.dtype.as_str(), "F8_E4M3" | "F8_E5M2" | "U8")
}

fn quantized_linear_scale_tensor(
    tensor_name: &str,
    entries: &BTreeMap<String, SafetensorsTensorSpan>,
) -> Option<(
    String,
    SafetensorsQuantizedLinearScaleKind,
    SafetensorsTensorSpan,
)> {
    let base_name = tensor_name.strip_suffix(".weight")?;
    let scale_inv_name = format!("{base_name}.weight_scale_inv");
    if let Some(scale) = entries.get(&scale_inv_name) {
        return Some((
            scale_inv_name,
            SafetensorsQuantizedLinearScaleKind::WeightScaleInv,
            scale.clone(),
        ));
    }

    let scale_name = format!("{base_name}.weight_scale");
    entries.get(&scale_name).map(|scale| {
        (
            scale_name,
            SafetensorsQuantizedLinearScaleKind::WeightScale,
            scale.clone(),
        )
    })
}

fn parse_routed_expert_weight_name(
    tensor_name: &str,
) -> Option<(
    usize,
    usize,
    SafetensorsRoutedExpertProjection,
    SafetensorsRoutedExpertWeightComponent,
)> {
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
    let component = match *parts.get(experts_index + 3)? {
        "weight" => SafetensorsRoutedExpertWeightComponent::Weight,
        "weight_packed" => SafetensorsRoutedExpertWeightComponent::PackedWeight,
        "weight_scale" => SafetensorsRoutedExpertWeightComponent::WeightScale,
        "weight_shape" => SafetensorsRoutedExpertWeightComponent::WeightShape,
        _ => return None,
    };
    if experts_index + 4 != parts.len() {
        return None;
    }

    Some((layer_id, expert_id, projection, component))
}

fn parse_usize_segment(value: &str) -> Option<usize> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }

    value.parse().ok()
}
