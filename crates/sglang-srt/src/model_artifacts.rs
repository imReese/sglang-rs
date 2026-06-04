use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalModelArtifacts {
    model_path: PathBuf,
    config: HfModelConfig,
    safetensors: SafetensorsManifest,
}

impl LocalModelArtifacts {
    pub fn from_model_path(path: impl AsRef<Path>) -> Result<Self, ModelArtifactError> {
        let model_path = path.as_ref();
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
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HfModelConfig {
    pub model_type: Option<String>,
    pub architectures: Vec<String>,
    pub vocab_size: Option<usize>,
    pub max_position_embeddings: Option<usize>,
    pub num_hidden_layers: Option<usize>,
}

impl HfModelConfig {
    pub fn from_model_path(path: impl AsRef<Path>) -> Result<Self, ModelArtifactError> {
        let config_path = path.as_ref().join("config.json");
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
            vocab_size: read_usize_field(&value, "vocab_size", &config_path)?,
            max_position_embeddings: read_usize_field(
                &value,
                "max_position_embeddings",
                &config_path,
            )?,
            num_hidden_layers: read_usize_field(&value, "num_hidden_layers", &config_path)?,
        })
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
        let Some(shard_path) = self.shard_for_tensor(tensor_name) else {
            return Ok(None);
        };
        SafetensorsHeader::from_file(shard_path).map(|header| header.tensor_metadata(tensor_name))
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsHeader {
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

        Ok(Self { tensors })
    }

    pub fn tensor_metadata(&self, tensor_name: &str) -> Option<SafetensorsTensorMetadata> {
        self.tensors.get(tensor_name).cloned()
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

fn is_routed_expert_weight(tensor_name: &str) -> bool {
    let Some((_, suffix)) = tensor_name.split_once(".experts.") else {
        return false;
    };
    let mut parts = suffix.split('.');
    let Some(expert_id) = parts.next() else {
        return false;
    };
    if expert_id.is_empty() || !expert_id.bytes().all(|byte| byte.is_ascii_digit()) {
        return false;
    }

    matches!(
        parts.collect::<Vec<_>>().as_slice(),
        ["w1", "weight"]
            | ["w2", "weight"]
            | ["w3", "weight"]
            | ["down_proj", "weight"]
            | ["up_proj", "weight"]
            | ["gate_proj", "weight"]
    )
}
