use std::collections::BTreeMap;
use std::fmt;

use crate::cli::ServerArgs;
use crate::engine::{Engine, RuntimeError};
use crate::tokenizer::Tokenizer;
use crate::types::{
    DisaggregatedParams, RequestId, SamplingParams, TokenGenerateOutput, TokenGenerateRequest,
};
use crate::worker::WorkerExecutor;

pub const DEFAULT_MAX_NEW_TOKENS: usize = 128;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RouterSamplingParams {
    pub max_new_tokens: Option<i32>,
}

impl RouterSamplingParams {
    fn into_sampling_params(self) -> Result<SamplingParams, RouterProtocolError> {
        let max_new_tokens = match self.max_new_tokens {
            Some(value) if value < 0 => {
                return Err(RouterProtocolError::InvalidMaxNewTokens(value));
            }
            Some(value) => value as usize,
            None => DEFAULT_MAX_NEW_TOKENS,
        };

        Ok(SamplingParams { max_new_tokens })
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RouterTokenizedInput {
    pub original_text: String,
    pub input_ids: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouterDisaggregatedParams {
    pub bootstrap_host: String,
    pub bootstrap_port: u16,
    pub bootstrap_room: i32,
}

impl From<RouterDisaggregatedParams> for DisaggregatedParams {
    fn from(value: RouterDisaggregatedParams) -> Self {
        Self {
            bootstrap_host: value.bootstrap_host,
            bootstrap_port: value.bootstrap_port,
            bootstrap_room: value.bootstrap_room,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RouterGenerateRequest {
    pub request_id: String,
    pub tokenized: Option<RouterTokenizedInput>,
    pub sampling_params: Option<RouterSamplingParams>,
    pub disaggregated_params: Option<RouterDisaggregatedParams>,
    pub stream: bool,
    pub data_parallel_rank: i32,
    pub trace_headers: BTreeMap<String, String>,
}

impl RouterGenerateRequest {
    pub fn try_into_token_generate_request(
        self,
    ) -> Result<TokenGenerateRequest, RouterProtocolError> {
        if self.request_id.is_empty() {
            return Err(RouterProtocolError::MissingRequestId);
        }
        let tokenized = self
            .tokenized
            .ok_or(RouterProtocolError::MissingTokenizedInput)?;

        let sampling = self
            .sampling_params
            .unwrap_or_default()
            .into_sampling_params()?;

        Ok(TokenGenerateRequest {
            request_id: RequestId::from(self.request_id.as_str()),
            input_ids: tokenized.input_ids,
            sampling,
            disaggregated_params: self.disaggregated_params.map(Into::into),
        })
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RouterGenerateResponse {
    pub request_id: String,
    pub body: RouterGenerateResponseBody,
}

impl RouterGenerateResponse {
    pub fn from_token_generate_output(output: TokenGenerateOutput, prompt_tokens: i32) -> Self {
        let request_id = output.request_id.as_str().to_string();
        let completion_tokens = output.output_ids.len() as i32;
        let body = if output.finished {
            RouterGenerateResponseBody::Complete(RouterGenerateComplete {
                output_ids: output.output_ids,
                finish_reason: "stop".to_string(),
                prompt_tokens,
                completion_tokens,
                cached_tokens: 0,
                index: 0,
            })
        } else {
            RouterGenerateResponseBody::Chunk(RouterGenerateStreamChunk {
                token_ids: output.output_ids,
                prompt_tokens,
                completion_tokens,
                cached_tokens: 0,
                index: 0,
            })
        };

        Self { request_id, body }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RouterGenerateResponseBody {
    Chunk(RouterGenerateStreamChunk),
    Complete(RouterGenerateComplete),
    Error(RouterGenerateError),
}

impl Default for RouterGenerateResponseBody {
    fn default() -> Self {
        Self::Complete(RouterGenerateComplete::default())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RouterGenerateStreamChunk {
    pub token_ids: Vec<u32>,
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub cached_tokens: i32,
    pub index: u32,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RouterGenerateComplete {
    pub output_ids: Vec<u32>,
    pub finish_reason: String,
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub cached_tokens: i32,
    pub index: u32,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RouterGenerateError {
    pub message: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouterHealthCheckResponse {
    pub healthy: bool,
    pub message: String,
}

impl RouterHealthCheckResponse {
    pub fn healthy() -> Self {
        Self {
            healthy: true,
            message: "ready".to_string(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouterFlushCacheResponse {
    pub success: bool,
    pub message: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouterGetModelInfoResponse {
    pub model_path: String,
    pub tokenizer_path: String,
    pub is_generation: bool,
    pub preferred_sampling_params: String,
    pub weight_version: String,
    pub served_model_name: String,
    pub max_context_length: i32,
    pub vocab_size: i32,
    pub supports_vision: bool,
    pub model_type: String,
    pub eos_token_ids: Vec<i32>,
    pub pad_token_id: i32,
    pub bos_token_id: i32,
    pub max_req_input_len: i32,
}

impl RouterGetModelInfoResponse {
    pub fn from_server_args(args: &ServerArgs) -> Self {
        let tokenizer_path = args
            .tokenizer_path
            .clone()
            .unwrap_or_else(|| args.model_path.clone());
        let served_model_name = args
            .served_model_name
            .clone()
            .unwrap_or_else(|| args.model_path.clone());

        Self {
            model_path: args.model_path.clone(),
            tokenizer_path,
            is_generation: true,
            preferred_sampling_params: "{}".to_string(),
            weight_version: String::new(),
            served_model_name,
            max_context_length: 0,
            vocab_size: 0,
            supports_vision: false,
            model_type: String::new(),
            eos_token_ids: Vec::new(),
            pad_token_id: 0,
            bos_token_id: 0,
            max_req_input_len: 0,
        }
    }
}

pub struct RouterRuntime<T, W> {
    engine: Engine<T, W>,
}

impl<T, W> RouterRuntime<T, W> {
    pub fn new(engine: Engine<T, W>) -> Self {
        Self { engine }
    }

    pub fn engine(&self) -> &Engine<T, W> {
        &self.engine
    }
}

impl<T, W> RouterRuntime<T, W>
where
    T: Tokenizer,
    W: WorkerExecutor,
{
    pub fn generate(
        &mut self,
        request: RouterGenerateRequest,
    ) -> Result<RouterGenerateResponse, RouterRuntimeError> {
        let prompt_tokens = request
            .tokenized
            .as_ref()
            .map(|tokenized| tokenized.input_ids.len() as i32)
            .unwrap_or(0);
        let token_request = request.try_into_token_generate_request()?;
        let output = self.engine.generate_tokens(token_request)?;

        Ok(RouterGenerateResponse::from_token_generate_output(
            output,
            prompt_tokens,
        ))
    }

    pub fn generate_stream(
        &mut self,
        request: RouterGenerateRequest,
    ) -> Result<Vec<RouterGenerateResponse>, RouterRuntimeError> {
        let prompt_tokens = request
            .tokenized
            .as_ref()
            .map(|tokenized| tokenized.input_ids.len() as i32)
            .unwrap_or(0);
        let token_request = request.try_into_token_generate_request()?;
        let outputs = self.engine.generate_token_stream(token_request)?;
        let mut output_ids = Vec::new();
        let mut responses = Vec::with_capacity(outputs.len());

        for output in outputs {
            let request_id = output.request_id.as_str().to_string();
            output_ids.extend_from_slice(&output.output_ids);
            let completion_tokens = output_ids.len() as i32;

            let body = if output.finished {
                RouterGenerateResponseBody::Complete(RouterGenerateComplete {
                    output_ids: output_ids.clone(),
                    finish_reason: "stop".to_string(),
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens: 0,
                    index: 0,
                })
            } else {
                RouterGenerateResponseBody::Chunk(RouterGenerateStreamChunk {
                    token_ids: output.output_ids,
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens: 0,
                    index: 0,
                })
            };

            responses.push(RouterGenerateResponse { request_id, body });
        }

        Ok(responses)
    }

    pub fn flush_cache(&mut self) -> RouterFlushCacheResponse {
        if self.engine.flush_cache() {
            RouterFlushCacheResponse {
                success: true,
                message: "cache flushed".to_string(),
            }
        } else {
            RouterFlushCacheResponse {
                success: false,
                message: String::new(),
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RouterProtocolError {
    MissingRequestId,
    MissingTokenizedInput,
    InvalidMaxNewTokens(i32),
}

impl fmt::Display for RouterProtocolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingRequestId => formatter.write_str("missing router request id"),
            Self::MissingTokenizedInput => formatter.write_str("missing router tokenized input"),
            Self::InvalidMaxNewTokens(value) => {
                write!(formatter, "invalid router max_new_tokens: {value}")
            }
        }
    }
}

impl std::error::Error for RouterProtocolError {}

#[derive(Debug)]
pub enum RouterRuntimeError {
    Protocol(RouterProtocolError),
    Runtime(RuntimeError),
}

impl fmt::Display for RouterRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Protocol(error) => write!(formatter, "router protocol error: {error}"),
            Self::Runtime(error) => write!(formatter, "router runtime error: {error}"),
        }
    }
}

impl std::error::Error for RouterRuntimeError {}

impl From<RouterProtocolError> for RouterRuntimeError {
    fn from(value: RouterProtocolError) -> Self {
        Self::Protocol(value)
    }
}

impl From<RuntimeError> for RouterRuntimeError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}
