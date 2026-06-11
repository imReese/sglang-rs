use std::fmt;
use std::path::{Path, PathBuf};

use tokenizers::Tokenizer as HfTokenizerImpl;

use crate::model_artifacts::{
    hf_hub_api_builder_from_env, resolve_model_path,
    resolve_model_path_from_hf_cache_with_required_file,
};

#[derive(Debug, Eq, PartialEq)]
pub enum TokenizerError {
    InvalidUtf8,
    TokenizerFileNotFound { path: PathBuf },
    Load { path: PathBuf, message: String },
    Encode { message: String },
    Decode { message: String },
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidUtf8 => formatter.write_str("token ids are not valid UTF-8 bytes"),
            Self::TokenizerFileNotFound { path } => {
                write!(
                    formatter,
                    "tokenizer.json was not found under {}",
                    path.display()
                )
            }
            Self::Load { path, message } => {
                write!(
                    formatter,
                    "failed to load tokenizer {}: {message}",
                    path.display()
                )
            }
            Self::Encode { message } => write!(formatter, "tokenizer encode failed: {message}"),
            Self::Decode { message } => write!(formatter, "tokenizer decode failed: {message}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError>;
}

#[derive(Clone, Debug, Default)]
pub struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(u32::from).collect()
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError> {
        let bytes = token_ids
            .iter()
            .map(|token_id| u8::try_from(*token_id).map_err(|_| TokenizerError::InvalidUtf8))
            .collect::<Result<Vec<_>, _>>()?;

        String::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8)
    }
}

#[derive(Clone)]
pub struct HfTokenizer {
    inner: HfTokenizerImpl,
}

impl fmt::Debug for HfTokenizer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("HfTokenizer")
    }
}

impl HfTokenizer {
    pub fn from_tokenizer_path(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let source_path = path.as_ref();
        let tokenizer_json = resolve_tokenizer_json(source_path).ok_or_else(|| {
            TokenizerError::TokenizerFileNotFound {
                path: source_path.to_path_buf(),
            }
        })?;
        let inner =
            HfTokenizerImpl::from_file(&tokenizer_json).map_err(|error| TokenizerError::Load {
                path: tokenizer_json,
                message: error.to_string(),
            })?;

        Ok(Self { inner })
    }
}

impl Tokenizer for HfTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner
            .encode(text, true)
            .map(|encoding| encoding.get_ids().to_vec())
            .expect("HF tokenizer should encode valid UTF-8 text")
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError> {
        self.inner
            .decode(token_ids, true)
            .map_err(|error| TokenizerError::Decode {
                message: error.to_string(),
            })
    }
}

#[derive(Clone, Debug)]
pub enum RuntimeTokenizer {
    Byte(ByteTokenizer),
    Hf(HfTokenizer),
}

impl RuntimeTokenizer {
    pub fn from_model_or_tokenizer_path(
        model_path: &str,
        tokenizer_path: Option<&str>,
    ) -> Result<Self, TokenizerError> {
        if let Some(tokenizer_path) = tokenizer_path {
            return Self::from_tokenizer_source(tokenizer_path);
        }

        Self::from_model_source(model_path)
    }

    pub fn from_model_or_tokenizer_path_with_hf_cache(
        model_path: &str,
        tokenizer_path: Option<&str>,
        hub_cache: impl AsRef<Path>,
    ) -> Result<Self, TokenizerError> {
        let resolved_model_path = resolve_model_path_from_hf_cache_with_required_file(
            model_path,
            hub_cache,
            "tokenizer.json",
        )
        .unwrap_or_else(|| PathBuf::from(model_path));
        Self::from_model_or_tokenizer_path_with_resolved_model_path(
            resolved_model_path,
            tokenizer_path,
        )
    }

    fn from_model_or_tokenizer_path_with_resolved_model_path(
        resolved_model_path: PathBuf,
        tokenizer_path: Option<&str>,
    ) -> Result<Self, TokenizerError> {
        if let Some(tokenizer_path) = tokenizer_path {
            return Self::from_tokenizer_source(tokenizer_path);
        }

        if resolve_tokenizer_json(&resolved_model_path).is_some() {
            return Ok(Self::Hf(HfTokenizer::from_tokenizer_path(
                resolved_model_path,
            )?));
        }

        Ok(Self::Byte(ByteTokenizer))
    }

    fn from_model_source(model_path: &str) -> Result<Self, TokenizerError> {
        let resolved_model_path = resolve_model_path(Path::new(model_path));
        if resolve_tokenizer_json(&resolved_model_path).is_some() {
            return Ok(Self::Hf(HfTokenizer::from_tokenizer_path(
                resolved_model_path,
            )?));
        }

        if !looks_like_hf_model_id(model_path) {
            return Ok(Self::Byte(ByteTokenizer));
        }

        match download_hf_tokenizer_json(model_path) {
            Ok(tokenizer_json) => Ok(Self::Hf(HfTokenizer::from_tokenizer_path(tokenizer_json)?)),
            Err(_) => Ok(Self::Byte(ByteTokenizer)),
        }
    }

    fn from_tokenizer_source(tokenizer_path: &str) -> Result<Self, TokenizerError> {
        let path = Path::new(tokenizer_path);
        if resolve_tokenizer_json(path).is_some() {
            return Ok(Self::Hf(HfTokenizer::from_tokenizer_path(path)?));
        }
        if is_local_tokenizer_path(path) && !looks_like_hf_model_id(tokenizer_path) {
            return Ok(Self::Hf(HfTokenizer::from_tokenizer_path(path)?));
        }
        if looks_like_hf_model_id(tokenizer_path) {
            let tokenizer_json = download_hf_tokenizer_json(tokenizer_path)?;
            return Ok(Self::Hf(HfTokenizer::from_tokenizer_path(tokenizer_json)?));
        }

        Ok(Self::Byte(ByteTokenizer))
    }
}

impl Default for RuntimeTokenizer {
    fn default() -> Self {
        Self::Byte(ByteTokenizer)
    }
}

impl Tokenizer for RuntimeTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            Self::Byte(tokenizer) => tokenizer.encode(text),
            Self::Hf(tokenizer) => tokenizer.encode(text),
        }
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError> {
        match self {
            Self::Byte(tokenizer) => tokenizer.decode(token_ids),
            Self::Hf(tokenizer) => tokenizer.decode(token_ids),
        }
    }
}

fn resolve_tokenizer_json(path: &Path) -> Option<PathBuf> {
    if path.is_file() {
        return path
            .file_name()
            .filter(|file_name| *file_name == "tokenizer.json")
            .map(|_| path.to_path_buf());
    }

    let tokenizer_json = path.join("tokenizer.json");
    tokenizer_json.is_file().then_some(tokenizer_json)
}

fn is_local_tokenizer_path(path: &Path) -> bool {
    path.is_absolute()
        || path.components().count() > 1
        || matches!(
            path.components().next(),
            Some(std::path::Component::CurDir | std::path::Component::ParentDir)
        )
        || path
            .file_name()
            .is_some_and(|file_name| file_name == "tokenizer.json")
}

fn looks_like_hf_model_id(model_path: &str) -> bool {
    model_path.contains('/')
        && !model_path.starts_with('/')
        && !model_path.starts_with('-')
        && !model_path.contains('\\')
}

fn download_hf_tokenizer_json(model_id: &str) -> Result<PathBuf, TokenizerError> {
    let api = hf_hub_api_builder_from_env()
        .build()
        .map_err(|error| TokenizerError::Load {
            path: PathBuf::from(model_id).join("tokenizer.json"),
            message: format!("failed to initialize Hugging Face Hub client: {error}"),
        })?;
    api.model(model_id.to_string())
        .get("tokenizer.json")
        .map_err(|error| TokenizerError::Load {
            path: PathBuf::from(model_id).join("tokenizer.json"),
            message: format!("failed to fetch Hugging Face tokenizer.json: {error}"),
        })
}
