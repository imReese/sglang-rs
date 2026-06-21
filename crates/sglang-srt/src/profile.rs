use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::json;

#[derive(Clone, Debug)]
pub(crate) struct ProfileSession {
    pub output_dir: PathBuf,
    pub started_at: SystemTime,
}

impl ProfileSession {
    pub(crate) fn new(output_dir: PathBuf) -> Self {
        Self {
            output_dir,
            started_at: SystemTime::now(),
        }
    }
}

#[derive(Debug)]
pub(crate) enum ProfileError {
    InvalidArgument(String),
    Internal(String),
}

impl fmt::Display for ProfileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument(message) | Self::Internal(message) => {
                formatter.write_str(message)
            }
        }
    }
}

impl std::error::Error for ProfileError {}

pub(crate) fn profile_output_dir(output_dir: Option<String>) -> Result<PathBuf, ProfileError> {
    let output_dir = output_dir.unwrap_or_else(|| {
        std::env::temp_dir()
            .join("sglang-rs-profile")
            .to_string_lossy()
            .to_string()
    });
    if output_dir.trim().is_empty() {
        return Err(ProfileError::InvalidArgument(
            "profile output_dir cannot be empty or whitespace only".to_string(),
        ));
    }
    Ok(PathBuf::from(output_dir))
}

pub(crate) fn ensure_profile_output_dir(output_dir: &PathBuf) -> Result<(), ProfileError> {
    fs::create_dir_all(output_dir).map_err(|error| {
        ProfileError::Internal(format!("create profile output directory: {error}"))
    })
}

fn unix_millis(time: SystemTime) -> u128 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

pub(crate) fn write_profile_file(
    session: ProfileSession,
    stopped_at: SystemTime,
    attributes: &HashMap<String, String>,
) -> Result<PathBuf, ProfileError> {
    ensure_profile_output_dir(&session.output_dir)?;

    let started_unix_ms = unix_millis(session.started_at);
    let stopped_unix_ms = unix_millis(stopped_at);
    let duration_ms = stopped_at
        .duration_since(session.started_at)
        .unwrap_or_default()
        .as_millis();
    let profile_path = session.output_dir.join(format!(
        "sglang-profile-{started_unix_ms}-{stopped_unix_ms}.json"
    ));
    let profile = json!({
        "profile": {
            "transport": attributes.get("transport").map(String::as_str).unwrap_or("tonic-grpc"),
            "output_dir": session.output_dir,
            "started_unix_ms": started_unix_ms,
            "stopped_unix_ms": stopped_unix_ms,
            "duration_ms": duration_ms,
            "attributes": attributes,
        }
    });
    let bytes = serde_json::to_vec_pretty(&profile)
        .map_err(|error| ProfileError::Internal(format!("serialize profile JSON: {error}")))?;
    fs::write(&profile_path, bytes)
        .map_err(|error| ProfileError::Internal(format!("write profile JSON: {error}")))?;

    Ok(profile_path)
}
