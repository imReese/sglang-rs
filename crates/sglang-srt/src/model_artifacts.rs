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

    pub fn tensor_span(
        &self,
        tensor_name: &str,
    ) -> Result<Option<SafetensorsTensorSpan>, ModelArtifactError> {
        let Some(shard_path) = self.shard_for_tensor(tensor_name) else {
            return Ok(None);
        };
        let header = SafetensorsHeader::from_file(shard_path)?;
        header.tensor_span(shard_path, tensor_name)
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
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsTensorSpan {
    pub path: PathBuf,
    pub metadata: SafetensorsTensorMetadata,
    pub absolute_byte_offset: u64,
    pub byte_len: usize,
}

impl SafetensorsTensorSpan {
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
