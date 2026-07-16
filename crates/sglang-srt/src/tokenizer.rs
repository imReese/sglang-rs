use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use minijinja::Environment;
use serde_json::{Map, Value, json};
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
    ChatTemplateUnavailable,
    ChatTemplateLoad { path: PathBuf, message: String },
    ChatTemplateRender { path: PathBuf, message: String },
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
            Self::ChatTemplateUnavailable => formatter.write_str(
                "this tokenizer has no Hugging Face chat template; use a completion endpoint or provide a model with tokenizer_config.json/chat_template.jinja",
            ),
            Self::ChatTemplateLoad { path, message } => {
                write!(
                    formatter,
                    "failed to load chat template {}: {message}",
                    path.display()
                )
            }
            Self::ChatTemplateRender { path, message } => {
                write!(
                    formatter,
                    "failed to render chat template {}: {message}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for TokenizerError {}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatTemplateInput {
    pub messages: Vec<Value>,
    pub tools: Option<Vec<Value>>,
    pub template_kwargs: Map<String, Value>,
}

pub trait IncrementalDecoder {
    fn step(&mut self, token_id: u32) -> Result<Option<String>, TokenizerError>;
}

pub trait Tokenizer: Clone {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError>;
    fn incremental_decoder(&self) -> Box<dyn IncrementalDecoder + '_>;
    fn apply_chat_template(&self, _input: &ChatTemplateInput) -> Result<String, TokenizerError> {
        Err(TokenizerError::ChatTemplateUnavailable)
    }
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

    fn incremental_decoder(&self) -> Box<dyn IncrementalDecoder + '_> {
        Box::new(ByteIncrementalDecoder::default())
    }

    fn apply_chat_template(&self, input: &ChatTemplateInput) -> Result<String, TokenizerError> {
        input
            .messages
            .iter()
            .map(|message| {
                message
                    .get("content")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
                    .ok_or_else(|| TokenizerError::ChatTemplateRender {
                        path: PathBuf::from("<byte-reference-tokenizer>"),
                        message: "message content must be a string".to_string(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|contents| contents.join("\n"))
    }
}

#[derive(Default)]
struct ByteIncrementalDecoder {
    pending: Vec<u8>,
}

impl IncrementalDecoder for ByteIncrementalDecoder {
    fn step(&mut self, token_id: u32) -> Result<Option<String>, TokenizerError> {
        self.pending
            .push(u8::try_from(token_id).map_err(|_| TokenizerError::InvalidUtf8)?);
        match std::str::from_utf8(&self.pending) {
            Ok(text) => {
                let text = text.to_string();
                self.pending.clear();
                Ok(Some(text))
            }
            Err(error) if error.error_len().is_none() => Ok(None),
            Err(_) => Err(TokenizerError::InvalidUtf8),
        }
    }
}

struct HfChatTemplate {
    environment: Environment<'static>,
    source_path: PathBuf,
}

#[derive(Clone)]
pub struct HfTokenizer {
    inner: HfTokenizerImpl,
    chat_template: Option<Arc<HfChatTemplate>>,
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
                path: tokenizer_json.clone(),
                message: error.to_string(),
            })?;
        let chat_template = load_chat_template(&tokenizer_json)?
            .map(|(source_path, source)| compile_chat_template(source_path, source))
            .transpose()?
            .map(Arc::new);

        Ok(Self {
            inner,
            chat_template,
        })
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

    fn incremental_decoder(&self) -> Box<dyn IncrementalDecoder + '_> {
        Box::new(HfIncrementalDecoder {
            tokenizer: &self.inner,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
        })
    }

    fn apply_chat_template(&self, input: &ChatTemplateInput) -> Result<String, TokenizerError> {
        let template = self
            .chat_template
            .as_ref()
            .ok_or(TokenizerError::ChatTemplateUnavailable)?;
        let mut context = input.template_kwargs.clone();
        context.insert("messages".to_string(), Value::Array(input.messages.clone()));
        context.insert(
            "tools".to_string(),
            input.tools.clone().map(Value::Array).unwrap_or(Value::Null),
        );
        context.insert("add_generation_prompt".to_string(), Value::Bool(true));

        template
            .environment
            .get_template("chat")
            .expect("compiled chat template should remain registered")
            .render(Value::Object(context))
            .map_err(|error| TokenizerError::ChatTemplateRender {
                path: template.source_path.clone(),
                message: error.to_string(),
            })
    }
}

struct HfIncrementalDecoder<'a> {
    tokenizer: &'a HfTokenizerImpl,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl IncrementalDecoder for HfIncrementalDecoder<'_> {
    fn step(&mut self, token_id: u32) -> Result<Option<String>, TokenizerError> {
        tokenizers::tokenizer::step_decode_stream(
            self.tokenizer,
            vec![token_id],
            true,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        )
        .map_err(|error| TokenizerError::Decode {
            message: error.to_string(),
        })
    }
}

#[derive(Clone, Debug)]
pub enum RuntimeTokenizer {
    Byte(ByteTokenizer),
    Hf(Box<HfTokenizer>),
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
            return Ok(Self::Hf(Box::new(HfTokenizer::from_tokenizer_path(
                resolved_model_path,
            )?)));
        }

        Err(TokenizerError::TokenizerFileNotFound {
            path: resolved_model_path,
        })
    }

    fn from_model_source(model_path: &str) -> Result<Self, TokenizerError> {
        let resolved_model_path = resolve_model_path(Path::new(model_path));
        if resolve_tokenizer_json(&resolved_model_path).is_some() {
            return Ok(Self::Hf(Box::new(HfTokenizer::from_tokenizer_path(
                resolved_model_path,
            )?)));
        }

        if resolved_model_path.exists() || is_local_tokenizer_path(Path::new(model_path)) {
            return Err(TokenizerError::TokenizerFileNotFound {
                path: resolved_model_path,
            });
        }

        if !looks_like_hf_model_id(model_path) {
            return Ok(Self::Byte(ByteTokenizer));
        }

        let tokenizer_json = download_hf_tokenizer_json(model_path)?;
        Ok(Self::Hf(Box::new(HfTokenizer::from_tokenizer_path(
            tokenizer_json,
        )?)))
    }

    fn from_tokenizer_source(tokenizer_path: &str) -> Result<Self, TokenizerError> {
        let path = Path::new(tokenizer_path);
        if resolve_tokenizer_json(path).is_some() {
            return Ok(Self::Hf(Box::new(HfTokenizer::from_tokenizer_path(path)?)));
        }
        if is_local_tokenizer_path(path) && !looks_like_hf_model_id(tokenizer_path) {
            return Ok(Self::Hf(Box::new(HfTokenizer::from_tokenizer_path(path)?)));
        }
        if looks_like_hf_model_id(tokenizer_path) {
            let tokenizer_json = download_hf_tokenizer_json(tokenizer_path)?;
            return Ok(Self::Hf(Box::new(HfTokenizer::from_tokenizer_path(
                tokenizer_json,
            )?)));
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

    fn incremental_decoder(&self) -> Box<dyn IncrementalDecoder + '_> {
        match self {
            Self::Byte(tokenizer) => tokenizer.incremental_decoder(),
            Self::Hf(tokenizer) => tokenizer.incremental_decoder(),
        }
    }

    fn apply_chat_template(&self, input: &ChatTemplateInput) -> Result<String, TokenizerError> {
        match self {
            Self::Byte(tokenizer) => tokenizer.apply_chat_template(input),
            Self::Hf(tokenizer) => tokenizer.apply_chat_template(input),
        }
    }
}

fn compile_chat_template(
    source_path: PathBuf,
    source: String,
) -> Result<HfChatTemplate, TokenizerError> {
    let mut environment = Environment::new();
    environment.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    environment
        .add_template_owned("chat", source)
        .map_err(|error| TokenizerError::ChatTemplateLoad {
            path: source_path.clone(),
            message: error.to_string(),
        })?;
    Ok(HfChatTemplate {
        environment,
        source_path,
    })
}

fn load_chat_template(tokenizer_json: &Path) -> Result<Option<(PathBuf, String)>, TokenizerError> {
    let Some(model_dir) = tokenizer_json.parent() else {
        return Ok(None);
    };
    let tokenizer_config = model_dir.join("tokenizer_config.json");
    if tokenizer_config.is_file() {
        let source = fs::read_to_string(&tokenizer_config).map_err(|error| {
            TokenizerError::ChatTemplateLoad {
                path: tokenizer_config.clone(),
                message: error.to_string(),
            }
        })?;
        let config: Value =
            serde_json::from_str(&source).map_err(|error| TokenizerError::ChatTemplateLoad {
                path: tokenizer_config.clone(),
                message: error.to_string(),
            })?;
        if let Some(chat_template) = config.get("chat_template") {
            let source = select_chat_template(chat_template, &tokenizer_config)?;
            return Ok(Some((tokenizer_config, source)));
        }
    }

    let jinja_path = model_dir.join("chat_template.jinja");
    if jinja_path.is_file() {
        let source =
            fs::read_to_string(&jinja_path).map_err(|error| TokenizerError::ChatTemplateLoad {
                path: jinja_path.clone(),
                message: error.to_string(),
            })?;
        return Ok(Some((jinja_path, source)));
    }

    Ok(None)
}

fn select_chat_template(value: &Value, path: &Path) -> Result<String, TokenizerError> {
    if let Some(source) = value.as_str() {
        return Ok(source.to_string());
    }
    let Some(templates) = value.as_object() else {
        return Err(TokenizerError::ChatTemplateLoad {
            path: path.to_path_buf(),
            message: "chat_template must be a string or named template object".to_string(),
        });
    };
    if let Some(source) = templates.get("default").and_then(Value::as_str) {
        return Ok(source.to_string());
    }
    if templates.len() == 1
        && let Some(source) = templates.values().next().and_then(Value::as_str)
    {
        return Ok(source.to_string());
    }
    Err(TokenizerError::ChatTemplateLoad {
        path: path.to_path_buf(),
        message: format!(
            "chat_template defines multiple named templates without a default: {}",
            json!(templates)
        ),
    })
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
    let repository = api.model(model_id.to_string());
    let tokenizer_json =
        repository
            .get("tokenizer.json")
            .map_err(|error| TokenizerError::Load {
                path: PathBuf::from(model_id).join("tokenizer.json"),
                message: format!("failed to fetch Hugging Face tokenizer.json: {error}"),
            })?;
    for optional_file in ["tokenizer_config.json", "chat_template.jinja"] {
        let _ = repository.get(optional_file);
    }
    Ok(tokenizer_json)
}
