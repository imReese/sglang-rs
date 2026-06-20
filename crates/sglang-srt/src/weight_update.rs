use sha2::{Digest, Sha256};

use crate::model_artifacts::LocalModelArtifacts;
use crate::router::RouterGetModelInfoResponse;
use crate::worker::WorkerWeightUpdateRequest;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct WeightMetadataUpdate {
    pub model_info: RouterGetModelInfoResponse,
    pub worker_request: WorkerWeightUpdateRequest,
    pub message: String,
}

pub(crate) fn update_model_info_from_disk(
    current: RouterGetModelInfoResponse,
    model_path: &str,
    load_format: Option<&str>,
) -> Result<WeightMetadataUpdate, String> {
    let model_path = model_path.trim();
    if model_path.is_empty() {
        return Err("model_path cannot be empty or whitespace only".to_string());
    }
    let load_format = validate_update_weights_load_format(load_format)?;

    let artifacts = LocalModelArtifacts::from_model_path(model_path)
        .map_err(|error| format!("invalid local model artifacts: {error}"))?;
    artifacts
        .validate_checkpoint_for_supported_model()
        .map_err(|error| format!("unsupported local model checkpoint: {error}"))?;

    let weight_version = safetensors_weight_version(&artifacts, load_format)?;
    let model_info = updated_model_info_from_artifacts(current, &artifacts, weight_version.clone());
    let worker_request = WorkerWeightUpdateRequest {
        model_path: artifacts.model_path().to_string_lossy().to_string(),
        load_format: Some(load_format.to_string()),
        weight_version: weight_version.clone(),
    };
    let message = format!(
        "registered {load_format} weights from {} with version {weight_version}",
        artifacts.model_path().display()
    );

    Ok(WeightMetadataUpdate {
        model_info,
        worker_request,
        message,
    })
}

fn validate_update_weights_load_format(load_format: Option<&str>) -> Result<&'static str, String> {
    let Some(load_format) = load_format else {
        return Ok("safetensors");
    };
    let load_format = load_format.trim();
    match load_format {
        "" => Err("load_format cannot be empty or whitespace only".to_string()),
        "auto" | "safetensors" => Ok("safetensors"),
        other => Err(format!(
            "unsupported load_format {other:?}; supported values are auto and safetensors"
        )),
    }
}

fn safetensors_weight_version(
    artifacts: &LocalModelArtifacts,
    load_format: &str,
) -> Result<String, String> {
    let mut entries = artifacts
        .safetensors()
        .checkpoint_fingerprint_entries()
        .map_err(|error| format!("invalid safetensors weights: {error}"))?;
    entries.sort_by(|left, right| {
        left.tensor_name
            .cmp(&right.tensor_name)
            .then_with(|| left.path.cmp(&right.path))
    });

    let mut hasher = Sha256::new();
    hasher.update(load_format.as_bytes());
    hasher.update(b"\0");
    hasher.update(artifacts.model_path().to_string_lossy().as_bytes());
    hasher.update(b"\0");
    for entry in entries {
        hasher.update(entry.tensor_name.as_bytes());
        hasher.update(b"\0");
        hasher.update(entry.path.to_string_lossy().as_bytes());
        hasher.update(b"\0");
        hasher.update(entry.dtype.as_bytes());
        hasher.update(b"\0");
        for dimension in entry.shape {
            hasher.update(dimension.to_le_bytes());
        }
        hasher.update(entry.absolute_byte_offset.to_le_bytes());
        hasher.update(entry.byte_len.to_le_bytes());
        hasher.update(entry.fnv1a64.to_le_bytes());
    }

    let digest = hasher.finalize();
    let hex = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(format!("safetensors-sha256:{hex}"))
}

fn updated_model_info_from_artifacts(
    current: RouterGetModelInfoResponse,
    artifacts: &LocalModelArtifacts,
    weight_version: String,
) -> RouterGetModelInfoResponse {
    let model_path = artifacts.model_path().to_string_lossy().to_string();
    let tokenizer_path = if current.tokenizer_path.trim().is_empty()
        || current.tokenizer_path == current.model_path
    {
        model_path
    } else {
        current.tokenizer_path
    };

    RouterGetModelInfoResponse::from_local_model_artifacts(
        artifacts,
        current.served_model_name,
        tokenizer_path,
        weight_version,
    )
}
