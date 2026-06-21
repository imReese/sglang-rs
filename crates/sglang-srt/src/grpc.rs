use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::future::{Future, pending};
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use serde_json::{Value, json};
use tonic::{Code, Request, Response, Status};

use crate::cli::ServerArgs;
use crate::engine::Engine;
use crate::openai_classify::{classify_response_json, parse_classify_request};
use crate::openai_embedding::{
    embeddings_response_json, parse_embedding_request, token_ids_to_embedding,
};
use crate::openai_rerank::{
    parse_rerank_request, rerank_results_to_json, score_rerank_documents, truncate_rerank_results,
};
use crate::openai_score::{parse_score_request, score_response_json};
use crate::profile::{
    ProfileError, ProfileSession, ensure_profile_output_dir, profile_output_dir, write_profile_file,
};
use crate::proto::sglang::runtime::v1::generate_response::Body as ProtoGenerateResponseBody;
use crate::proto::sglang::runtime::v1::sglang_service_server::{
    SglangService, SglangServiceServer,
};
use crate::proto::sglang::runtime::v1::{
    AbortRequest, Classification, ClassifyRequest, ClassifyResponse, ContinueGenerationRequest,
    ControlResponse, DetokenizeRequest, DetokenizeResponse, EmbedRequest, EmbedResponse, Embedding,
    FlushCacheRequest, GenerateComplete as ProtoGenerateComplete,
    GenerateError as ProtoGenerateError, GenerateRequest as ProtoGenerateRequest,
    GenerateResponse as ProtoGenerateResponse, GenerateStreamChunk as ProtoGenerateStreamChunk,
    GetLoadRequest, GetModelInfoRequest, GetServerInfoRequest, GetWeightsByNameRequest,
    GetWeightsByNameResponse, HealthCheckRequest, HealthCheckResponse, ListModelsRequest,
    ListModelsResponse, LoadResponse, ModelInfoResponse, OpenAiJsonRequest, OpenAiJsonResponse,
    PauseGenerationRequest, SamplingParams as ProtoSamplingParams, ServerInfoResponse,
    StartProfileRequest, StopProfileRequest, TextEmbedRequest, TextGenerateRequest,
    TokenizeRequest, TokenizeResponse, TokenizedInput, UpdateWeightVersionRequest,
    UpdateWeightsFromDiskRequest, Usage,
};
use crate::router::{
    RouterDisaggregatedParams, RouterGenerateComplete, RouterGenerateError, RouterGenerateRequest,
    RouterGenerateResponse, RouterGenerateResponseBody, RouterGenerateStreamChunk,
    RouterGetModelInfoResponse, RouterProtocolError, RouterRuntime, RouterRuntimeError,
    RouterSamplingParams, RouterStatusCode, RouterTextGenerateRequest,
};
use crate::tokenizer::Tokenizer;
use crate::transfer::PdConfig;
use crate::weight_update::{get_weights_by_name_from_disk, update_model_info_from_disk};
use crate::worker::WorkerExecutor;

type GenerateResponseStream = Pin<
    Box<
        dyn tonic::codegen::tokio_stream::Stream<Item = Result<ProtoGenerateResponse, Status>>
            + Send,
    >,
>;
type OpenAiJsonResponseStream = Pin<
    Box<dyn tonic::codegen::tokio_stream::Stream<Item = Result<OpenAiJsonResponse, Status>> + Send>,
>;

const DEFAULT_PROTO_EMBEDDING_DIMENSIONS: usize = 8;
const DEFAULT_PROTO_CLASS_COUNT: usize = 3;

enum GrpcChatResponse {
    Single(Vec<RouterGenerateResponse>),
    Batch(Vec<Vec<RouterGenerateResponse>>),
}

enum GrpcCompletionResponse {
    Single(Vec<RouterGenerateResponse>),
    Batch(Vec<Vec<RouterGenerateResponse>>),
}

pub const SGLANG_RUNTIME_FILE_DESCRIPTOR_SET: &[u8] =
    tonic::include_file_descriptor_set!("sglang_runtime_descriptor");

#[derive(Clone)]
pub struct GrpcRouterService<T, W> {
    runtime: Arc<Mutex<RouterRuntime<T, W>>>,
    model_info: Arc<Mutex<Option<RouterGetModelInfoResponse>>>,
    max_transfer_polls: usize,
    server_info_attributes: HashMap<String, String>,
    profile: Arc<Mutex<Option<ProfileSession>>>,
}

#[derive(Debug)]
pub enum GrpcServeError {
    Reflection(tonic_reflection::server::Error),
    Transport(tonic::transport::Error),
}

impl fmt::Display for GrpcServeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reflection(error) => write!(formatter, "gRPC reflection error: {error}"),
            Self::Transport(error) => write!(formatter, "gRPC transport error: {error}"),
        }
    }
}

impl std::error::Error for GrpcServeError {}

impl From<tonic_reflection::server::Error> for GrpcServeError {
    fn from(value: tonic_reflection::server::Error) -> Self {
        Self::Reflection(value)
    }
}

impl From<tonic::transport::Error> for GrpcServeError {
    fn from(value: tonic::transport::Error) -> Self {
        Self::Transport(value)
    }
}

impl<T, W> GrpcRouterService<T, W> {
    pub fn new(runtime: RouterRuntime<T, W>) -> Self {
        Self {
            runtime: Arc::new(Mutex::new(runtime)),
            model_info: Arc::new(Mutex::new(None)),
            max_transfer_polls: 0,
            server_info_attributes: default_server_info_attributes(),
            profile: Arc::new(Mutex::new(None)),
        }
    }

    pub fn with_model_info(
        runtime: RouterRuntime<T, W>,
        model_info: RouterGetModelInfoResponse,
    ) -> Self {
        Self {
            runtime: Arc::new(Mutex::new(runtime)),
            model_info: Arc::new(Mutex::new(Some(model_info))),
            max_transfer_polls: 0,
            server_info_attributes: default_server_info_attributes(),
            profile: Arc::new(Mutex::new(None)),
        }
    }

    pub fn with_max_transfer_polls(mut self, max_transfer_polls: usize) -> Self {
        self.max_transfer_polls = max_transfer_polls;
        self
    }

    pub fn with_server_args(runtime: RouterRuntime<T, W>, args: &ServerArgs) -> Self {
        Self::with_model_info(runtime, RouterGetModelInfoResponse::from_server_args(args))
            .with_server_info_attributes(server_info_attributes_from_args(args))
    }

    pub fn with_server_info_attributes(mut self, attributes: HashMap<String, String>) -> Self {
        self.server_info_attributes = attributes;
        self
    }

    pub fn from_engine(engine: Engine<T, W>) -> Self {
        Self::new(RouterRuntime::new(engine))
    }

    pub fn from_engine_with_model_info(
        engine: Engine<T, W>,
        model_info: RouterGetModelInfoResponse,
    ) -> Self {
        Self::with_model_info(RouterRuntime::new(engine), model_info)
    }

    pub fn runtime(&self) -> &Arc<Mutex<RouterRuntime<T, W>>> {
        &self.runtime
    }

    fn model_info(&self) -> Result<RouterGetModelInfoResponse, Status> {
        self.model_info
            .lock()
            .map_err(|_| Status::internal("model info mutex poisoned"))?
            .clone()
            .ok_or_else(|| Status::failed_precondition("model info is not configured"))
    }
}

pub fn router_status_code_to_grpc_code(status_code: RouterStatusCode) -> Code {
    match status_code {
        RouterStatusCode::InvalidArgument => Code::InvalidArgument,
        RouterStatusCode::ResourceExhausted => Code::ResourceExhausted,
        RouterStatusCode::FailedPrecondition => Code::FailedPrecondition,
    }
}

pub fn router_protocol_error_to_status(error: RouterProtocolError) -> Status {
    Status::new(
        router_status_code_to_grpc_code(error.status_code()),
        error.to_string(),
    )
}

pub async fn serve_grpc_router<T, W>(
    addr: SocketAddr,
    service: GrpcRouterService<T, W>,
    enable_reflection: bool,
) -> Result<(), GrpcServeError>
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    serve_grpc_router_with_shutdown(addr, service, enable_reflection, pending()).await
}

pub async fn serve_grpc_router_with_shutdown<T, W, F>(
    addr: SocketAddr,
    service: GrpcRouterService<T, W>,
    enable_reflection: bool,
    shutdown: F,
) -> Result<(), GrpcServeError>
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
    F: Future<Output = ()> + Send + 'static,
{
    let sglang_service = SglangServiceServer::new(service);

    if enable_reflection {
        let reflection_service = tonic_reflection::server::Builder::configure()
            .register_encoded_file_descriptor_set(SGLANG_RUNTIME_FILE_DESCRIPTOR_SET)
            .build_v1()?;

        tonic::transport::Server::builder()
            .add_service(sglang_service)
            .add_service(reflection_service)
            .serve_with_shutdown(addr, shutdown)
            .await?;

        return Ok(());
    }

    tonic::transport::Server::builder()
        .add_service(sglang_service)
        .serve_with_shutdown(addr, shutdown)
        .await?;

    Ok(())
}

fn router_runtime_error_to_status(error: RouterRuntimeError) -> Status {
    match error {
        RouterRuntimeError::Protocol(error) => router_protocol_error_to_status(error),
        RouterRuntimeError::Runtime(error) => Status::internal(error.to_string()),
    }
}

fn openai_chat_response_from_router_responses(
    mut responses: Vec<RouterGenerateResponse>,
    model: &str,
) -> Result<OpenAiJsonResponse, Status> {
    let Some(response) = responses.pop() else {
        return Err(Status::internal("generation produced no response"));
    };

    let json = match response.body {
        RouterGenerateResponseBody::Complete(complete) => json!({
            "id": format!("chatcmpl-{}", response.request_id),
            "object": "chat.completion",
            "model": model,
            "choices": [{
                "index": complete.index,
                "message": {
                    "role": "assistant",
                    "content": complete.text,
                },
                "finish_reason": complete.finish_reason,
            }],
            "usage": {
                "prompt_tokens": complete.prompt_tokens,
                "completion_tokens": complete.completion_tokens,
                "cached_tokens": complete.cached_tokens,
            }
        }),
        RouterGenerateResponseBody::Chunk(_) => {
            return Err(Status::internal(
                "non-stream gRPC chat completion returned a stream chunk",
            ));
        }
        RouterGenerateResponseBody::Error(error) => {
            return Err(Status::internal(error.message));
        }
    };

    let json = serde_json::to_vec(&json)
        .map_err(|e| Status::internal(format!("serialize OpenAI chat JSON: {e}")))?;
    Ok(OpenAiJsonResponse { json })
}

#[derive(Default)]
struct OpenAiChatUsage {
    prompt_tokens: i32,
    completion_tokens: i32,
    cached_tokens: i32,
}

fn openai_chat_batch_response_from_router_responses(
    batch_responses: Vec<Vec<RouterGenerateResponse>>,
    model: &str,
) -> Result<OpenAiJsonResponse, Status> {
    let mut choices = Vec::with_capacity(batch_responses.len());
    let mut usage = OpenAiChatUsage::default();
    let mut response_id = None;

    for (batch_index, mut responses) in batch_responses.into_iter().enumerate() {
        let Some(response) = responses.pop() else {
            return Err(Status::internal("generation produced no response"));
        };
        if response_id.is_none() {
            response_id = Some(response.request_id.clone());
        }

        match response.body {
            RouterGenerateResponseBody::Complete(complete) => {
                usage.prompt_tokens += complete.prompt_tokens;
                usage.completion_tokens += complete.completion_tokens;
                usage.cached_tokens += complete.cached_tokens;
                choices.push(json!({
                    "index": batch_index,
                    "message": {
                        "role": "assistant",
                        "content": complete.text,
                    },
                    "finish_reason": complete.finish_reason,
                }));
            }
            RouterGenerateResponseBody::Chunk(_) => {
                return Err(Status::internal(
                    "non-stream gRPC chat completion returned a stream chunk",
                ));
            }
            RouterGenerateResponseBody::Error(error) => {
                return Err(Status::internal(error.message));
            }
        }
    }

    let json = json!({
        "id": format!("chatcmpl-{}", response_id.unwrap_or_default()),
        "object": "chat.completion",
        "model": model,
        "choices": choices,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cached_tokens": usage.cached_tokens,
        }
    });
    let json = serde_json::to_vec(&json)
        .map_err(|e| Status::internal(format!("serialize OpenAI chat JSON: {e}")))?;
    Ok(OpenAiJsonResponse { json })
}

fn openai_chat_stream_response_from_router_response(
    response: RouterGenerateResponse,
    model: &str,
) -> Result<OpenAiJsonResponse, Status> {
    let json = match response.body {
        RouterGenerateResponseBody::Chunk(chunk) => json!({
            "id": format!("chatcmpl-{}", response.request_id),
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{
                "index": chunk.index,
                "delta": {
                    "content": chunk.text,
                },
                "finish_reason": Value::Null,
            }],
        }),
        RouterGenerateResponseBody::Complete(complete) => json!({
            "id": format!("chatcmpl-{}", response.request_id),
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{
                "index": complete.index,
                "delta": {},
                "finish_reason": complete.finish_reason,
            }],
            "usage": {
                "prompt_tokens": complete.prompt_tokens,
                "completion_tokens": complete.completion_tokens,
                "cached_tokens": complete.cached_tokens,
            }
        }),
        RouterGenerateResponseBody::Error(error) => {
            return Err(Status::internal(error.message));
        }
    };

    let json = serde_json::to_vec(&json)
        .map_err(|e| Status::internal(format!("serialize OpenAI chat stream JSON: {e}")))?;
    Ok(OpenAiJsonResponse { json })
}

fn openai_completion_response_from_router_responses(
    mut responses: Vec<RouterGenerateResponse>,
    model: &str,
) -> Result<OpenAiJsonResponse, Status> {
    let Some(response) = responses.pop() else {
        return Err(Status::internal("generation produced no response"));
    };

    let json = match response.body {
        RouterGenerateResponseBody::Complete(complete) => json!({
            "id": format!("cmpl-{}", response.request_id),
            "object": "text_completion",
            "model": model,
            "choices": [{
                "index": complete.index,
                "text": complete.text,
                "logprobs": Value::Null,
                "finish_reason": complete.finish_reason,
            }],
            "usage": {
                "prompt_tokens": complete.prompt_tokens,
                "completion_tokens": complete.completion_tokens,
                "total_tokens": complete.prompt_tokens + complete.completion_tokens,
                "cached_tokens": complete.cached_tokens,
            }
        }),
        RouterGenerateResponseBody::Chunk(_) => {
            return Err(Status::internal(
                "non-stream gRPC completion returned a stream chunk",
            ));
        }
        RouterGenerateResponseBody::Error(error) => {
            return Err(Status::internal(error.message));
        }
    };

    let json = serde_json::to_vec(&json)
        .map_err(|e| Status::internal(format!("serialize OpenAI completion JSON: {e}")))?;
    Ok(OpenAiJsonResponse { json })
}

#[derive(Default)]
struct OpenAiCompletionUsage {
    prompt_tokens: i32,
    completion_tokens: i32,
    cached_tokens: i32,
}

fn openai_completion_batch_response_from_router_responses(
    batch_responses: Vec<Vec<RouterGenerateResponse>>,
    model: &str,
) -> Result<OpenAiJsonResponse, Status> {
    let mut choices = Vec::with_capacity(batch_responses.len());
    let mut usage = OpenAiCompletionUsage::default();
    let mut response_id = None;

    for (batch_index, mut responses) in batch_responses.into_iter().enumerate() {
        let Some(response) = responses.pop() else {
            return Err(Status::internal("generation produced no response"));
        };
        if response_id.is_none() {
            response_id = Some(response.request_id.clone());
        }

        match response.body {
            RouterGenerateResponseBody::Complete(complete) => {
                usage.prompt_tokens += complete.prompt_tokens;
                usage.completion_tokens += complete.completion_tokens;
                usage.cached_tokens += complete.cached_tokens;
                choices.push(json!({
                    "index": batch_index,
                    "text": complete.text,
                    "logprobs": Value::Null,
                    "finish_reason": complete.finish_reason,
                }));
            }
            RouterGenerateResponseBody::Chunk(_) => {
                return Err(Status::internal(
                    "non-stream gRPC completion returned a stream chunk",
                ));
            }
            RouterGenerateResponseBody::Error(error) => {
                return Err(Status::internal(error.message));
            }
        }
    }

    let json = json!({
        "id": format!("cmpl-{}", response_id.unwrap_or_default()),
        "object": "text_completion",
        "model": model,
        "choices": choices,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.prompt_tokens + usage.completion_tokens,
            "cached_tokens": usage.cached_tokens,
        }
    });
    let json = serde_json::to_vec(&json)
        .map_err(|e| Status::internal(format!("serialize OpenAI completion JSON: {e}")))?;
    Ok(OpenAiJsonResponse { json })
}

fn openai_completion_stream_response_from_router_response(
    response: RouterGenerateResponse,
    model: &str,
) -> Result<OpenAiJsonResponse, Status> {
    let json = match response.body {
        RouterGenerateResponseBody::Chunk(chunk) => json!({
            "id": format!("cmpl-{}", response.request_id),
            "object": "text_completion",
            "model": model,
            "choices": [{
                "index": chunk.index,
                "text": chunk.text,
                "logprobs": Value::Null,
                "finish_reason": Value::Null,
            }],
        }),
        RouterGenerateResponseBody::Complete(complete) => json!({
            "id": format!("cmpl-{}", response.request_id),
            "object": "text_completion",
            "model": model,
            "choices": [{
                "index": complete.index,
                "text": "",
                "logprobs": Value::Null,
                "finish_reason": complete.finish_reason,
            }],
            "usage": {
                "prompt_tokens": complete.prompt_tokens,
                "completion_tokens": complete.completion_tokens,
                "total_tokens": complete.prompt_tokens + complete.completion_tokens,
                "cached_tokens": complete.cached_tokens,
            }
        }),
        RouterGenerateResponseBody::Error(error) => {
            return Err(Status::internal(error.message));
        }
    };

    let json = serde_json::to_vec(&json)
        .map_err(|e| Status::internal(format!("serialize OpenAI completion stream JSON: {e}")))?;
    Ok(OpenAiJsonResponse { json })
}

fn proto_embed_response_from_batches(
    token_batches: Vec<Vec<u32>>,
) -> Result<EmbedResponse, Status> {
    if token_batches.is_empty() {
        return Err(Status::invalid_argument("input cannot be empty"));
    }

    let prompt_tokens = token_batches
        .iter()
        .map(|token_ids| token_ids.len())
        .sum::<usize>();
    let embeddings = token_batches
        .into_iter()
        .enumerate()
        .map(|(index, token_ids)| {
            if token_ids.is_empty() {
                return Err(Status::invalid_argument("token ID input cannot be empty"));
            }
            Ok(Embedding {
                values: token_ids_to_embedding(&token_ids, DEFAULT_PROTO_EMBEDDING_DIMENSIONS),
                index: usize_to_u32(index)?,
            })
        })
        .collect::<Result<Vec<_>, Status>>()?;

    Ok(EmbedResponse {
        embeddings,
        usage: Some(proto_usage(prompt_tokens)?),
    })
}

fn proto_tokenized_inputs_to_batches(inputs: Vec<TokenizedInput>) -> Result<Vec<Vec<u32>>, Status> {
    if inputs.is_empty() {
        return Err(Status::invalid_argument("input cannot be empty"));
    }

    inputs
        .into_iter()
        .enumerate()
        .map(|(index, input)| {
            if input.input_ids.is_empty() {
                return Err(Status::invalid_argument(format!(
                    "token ID input at index {index} cannot be empty"
                )));
            }
            Ok(input.input_ids)
        })
        .collect()
}

fn proto_classify_response_from_batches(
    token_batches: Vec<Vec<u32>>,
    labels: Vec<String>,
) -> Result<ClassifyResponse, Status> {
    if token_batches.is_empty() {
        return Err(Status::invalid_argument("input cannot be empty"));
    }
    let labels = normalize_classification_labels(labels)?;
    let prompt_tokens = token_batches
        .iter()
        .map(|token_ids| token_ids.len())
        .sum::<usize>();
    let classifications = token_batches
        .into_iter()
        .enumerate()
        .map(|(index, token_ids)| {
            if token_ids.is_empty() {
                return Err(Status::invalid_argument("token ID input cannot be empty"));
            }
            let logits = token_ids_to_embedding(&token_ids, labels.len());
            let probs = softmax(&logits);
            let (label_index, score) = probs
                .iter()
                .copied()
                .enumerate()
                .max_by(|(left_index, left), (right_index, right)| {
                    left.total_cmp(right)
                        .then_with(|| right_index.cmp(left_index))
                })
                .unwrap_or((0, 0.0));

            Ok(Classification {
                label: labels[label_index].clone(),
                score,
                index: usize_to_u32(index)?,
            })
        })
        .collect::<Result<Vec<_>, Status>>()?;

    Ok(ClassifyResponse {
        classifications,
        usage: Some(proto_usage(prompt_tokens)?),
    })
}

fn normalize_classification_labels(labels: Vec<String>) -> Result<Vec<String>, Status> {
    if labels.is_empty() {
        return Ok((0..DEFAULT_PROTO_CLASS_COUNT)
            .map(|index| format!("LABEL_{index}"))
            .collect());
    }

    labels
        .into_iter()
        .enumerate()
        .map(|(index, label)| {
            if label.trim().is_empty() {
                return Err(Status::invalid_argument(format!(
                    "classification label at index {index} cannot be empty"
                )));
            }
            Ok(label)
        })
        .collect()
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp = logits
        .iter()
        .map(|logit| (*logit - max).exp())
        .collect::<Vec<_>>();
    let sum = exp.iter().sum::<f32>();
    if sum == 0.0 {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exp.into_iter().map(|value| value / sum).collect()
}

fn proto_usage(prompt_tokens: usize) -> Result<Usage, Status> {
    let prompt_tokens = usize_to_i32(prompt_tokens)?;
    Ok(Usage {
        prompt_tokens,
        completion_tokens: 0,
        total_tokens: prompt_tokens,
    })
}

fn profile_error_to_status(error: ProfileError) -> Status {
    match error {
        ProfileError::InvalidArgument(message) => Status::invalid_argument(message),
        ProfileError::Internal(message) => Status::internal(message),
    }
}

#[tonic::async_trait]
impl<T, W> SglangService for GrpcRouterService<T, W>
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    type TextGenerateStream = GenerateResponseStream;
    type GenerateStream = GenerateResponseStream;
    type ChatCompleteStream = OpenAiJsonResponseStream;
    type CompleteStream = OpenAiJsonResponseStream;

    async fn text_generate(
        &self,
        request: Request<TextGenerateRequest>,
    ) -> Result<Response<Self::TextGenerateStream>, Status> {
        let request = proto_text_generate_request_to_router_request(request.into_inner())?;
        let responses = self
            .runtime
            .lock()
            .map_err(|_| Status::internal("router runtime mutex poisoned"))?
            .generate_text_stream_with_transfer_polling(request, self.max_transfer_polls)
            .map_err(router_runtime_error_to_status)?
            .into_iter()
            .map(router_generate_response_to_proto_response)
            .map(Ok)
            .collect::<Vec<_>>();

        Ok(Response::new(Box::pin(tonic::codegen::tokio_stream::iter(
            responses,
        ))))
    }

    async fn generate(
        &self,
        request: Request<ProtoGenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let request = proto_generate_request_to_router_request(request.into_inner())?;
        let responses = self
            .runtime
            .lock()
            .map_err(|_| Status::internal("router runtime mutex poisoned"))?
            .generate_stream_with_transfer_polling(request, self.max_transfer_polls)
            .map_err(router_runtime_error_to_status)?
            .into_iter()
            .map(router_generate_response_to_proto_response)
            .map(Ok)
            .collect::<Vec<_>>();

        Ok(Response::new(Box::pin(tonic::codegen::tokio_stream::iter(
            responses,
        ))))
    }

    async fn text_embed(
        &self,
        request: Request<TextEmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        let request = request.into_inner();
        if request.texts.is_empty() {
            return Err(Status::invalid_argument("input cannot be empty"));
        }
        if let Some((index, _)) = request
            .texts
            .iter()
            .enumerate()
            .find(|(_, text)| text.trim().is_empty())
        {
            return Err(Status::invalid_argument(format!(
                "input text at index {index} cannot be empty or whitespace only"
            )));
        }

        let token_batches = {
            let runtime = self
                .runtime
                .lock()
                .map_err(|_| Status::internal("router runtime mutex poisoned"))?;
            request
                .texts
                .iter()
                .map(|text| runtime.tokenize(text).token_ids)
                .collect::<Vec<_>>()
        };

        Ok(Response::new(proto_embed_response_from_batches(
            token_batches,
        )?))
    }

    async fn embed(
        &self,
        request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        let request = request.into_inner();
        let token_batches = proto_tokenized_inputs_to_batches(request.inputs)?;
        Ok(Response::new(proto_embed_response_from_batches(
            token_batches,
        )?))
    }

    async fn classify(
        &self,
        request: Request<ClassifyRequest>,
    ) -> Result<Response<ClassifyResponse>, Status> {
        let request = request.into_inner();
        let token_batches = proto_tokenized_inputs_to_batches(request.inputs)?;
        Ok(Response::new(proto_classify_response_from_batches(
            token_batches,
            request.labels,
        )?))
    }

    async fn tokenize(
        &self,
        request: Request<TokenizeRequest>,
    ) -> Result<Response<TokenizeResponse>, Status> {
        let request = request.into_inner();
        let response = self
            .runtime
            .lock()
            .map_err(|_| Status::internal("router runtime mutex poisoned"))?
            .tokenize(&request.text);

        Ok(Response::new(TokenizeResponse {
            count: usize_to_u32(response.token_ids.len())?,
            token_ids: response.token_ids,
        }))
    }

    async fn detokenize(
        &self,
        request: Request<DetokenizeRequest>,
    ) -> Result<Response<DetokenizeResponse>, Status> {
        let request = request.into_inner();
        let response = self
            .runtime
            .lock()
            .map_err(|_| Status::internal("router runtime mutex poisoned"))?
            .detokenize(&request.token_ids)
            .map_err(|error| Status::invalid_argument(error.to_string()))?;

        Ok(Response::new(DetokenizeResponse {
            text: response.text,
        }))
    }

    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        Ok(Response::new(HealthCheckResponse {
            healthy: true,
            message: "ready".to_string(),
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<GetModelInfoRequest>,
    ) -> Result<Response<ModelInfoResponse>, Status> {
        Ok(Response::new(router_model_info_to_proto_response(
            self.model_info()?,
        )))
    }

    async fn get_server_info(
        &self,
        _request: Request<GetServerInfoRequest>,
    ) -> Result<Response<ServerInfoResponse>, Status> {
        Ok(Response::new(ServerInfoResponse {
            version: env!("CARGO_PKG_VERSION").to_string(),
            runtime: "sglang-rs".to_string(),
            attributes: self.server_info_attributes.clone(),
        }))
    }

    async fn list_models(
        &self,
        _request: Request<ListModelsRequest>,
    ) -> Result<Response<ListModelsResponse>, Status> {
        Ok(Response::new(ListModelsResponse {
            models: vec![router_model_info_to_proto_response(self.model_info()?)],
        }))
    }

    async fn get_load(
        &self,
        _request: Request<GetLoadRequest>,
    ) -> Result<Response<LoadResponse>, Status> {
        let response = self
            .runtime
            .lock()
            .map_err(|_| Status::internal("router runtime mutex poisoned"))?
            .load();

        Ok(Response::new(LoadResponse {
            waiting_queue_depth: usize_to_u32(response.waiting_queue_depth)?,
            decode_queue_depth: usize_to_u32(response.decode_queue_depth)?,
            available_cache_pages: response
                .available_cache_pages
                .map(usize_to_u32)
                .transpose()?,
        }))
    }

    async fn abort(
        &self,
        request: Request<AbortRequest>,
    ) -> Result<Response<ControlResponse>, Status> {
        let request = request.into_inner();
        let response = if request.abort_all {
            self.runtime
                .lock()
                .map_err(|_| Status::internal("router runtime mutex poisoned"))?
                .abort_all_requests()
        } else {
            self.runtime
                .lock()
                .map_err(|_| Status::internal("router runtime mutex poisoned"))?
                .abort_request(&request.request_id)
                .map_err(router_protocol_error_to_status)?
        };

        Ok(Response::new(ControlResponse {
            success: response.success,
            message: response.message,
        }))
    }

    async fn flush_cache(
        &self,
        _request: Request<FlushCacheRequest>,
    ) -> Result<Response<ControlResponse>, Status> {
        let response = self
            .runtime
            .lock()
            .map_err(|_| Status::internal("router runtime mutex poisoned"))?
            .flush_cache();

        Ok(Response::new(ControlResponse {
            success: response.success,
            message: response.message,
        }))
    }

    async fn pause_generation(
        &self,
        _request: Request<PauseGenerationRequest>,
    ) -> Result<Response<ControlResponse>, Status> {
        let response = self
            .runtime
            .lock()
            .map_err(|_| Status::internal("router runtime mutex poisoned"))?
            .pause_generation();

        Ok(Response::new(ControlResponse {
            success: response.success,
            message: response.message,
        }))
    }

    async fn continue_generation(
        &self,
        _request: Request<ContinueGenerationRequest>,
    ) -> Result<Response<ControlResponse>, Status> {
        let response = self
            .runtime
            .lock()
            .map_err(|_| Status::internal("router runtime mutex poisoned"))?
            .continue_generation();

        Ok(Response::new(ControlResponse {
            success: response.success,
            message: response.message,
        }))
    }

    async fn chat_complete(
        &self,
        request: Request<OpenAiJsonRequest>,
    ) -> Result<Response<Self::ChatCompleteStream>, Status> {
        let request = request.into_inner();
        let options = request.options.unwrap_or_default();
        let payload: Value = serde_json::from_slice(&request.json)
            .map_err(|e| Status::invalid_argument(format!("invalid OpenAI JSON payload: {e}")))?;
        let model = self.model_info()?.served_model_name;
        let mut request = crate::http::http_chat_payload_to_router_request(payload, &model)
            .map_err(Status::invalid_argument)?;
        match &mut request {
            crate::http::HttpChatRequest::Single(request) => {
                apply_openai_json_options_to_router_request(request, options);
            }
            crate::http::HttpChatRequest::Batch(requests) => {
                for request in requests {
                    apply_openai_json_options_to_router_request(request, options.clone());
                }
            }
        }
        let stream = request.stream();

        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| Status::internal("router runtime mutex poisoned"))?;
        let response = match request {
            crate::http::HttpChatRequest::Single(request) => runtime
                .generate_text_stream_with_transfer_polling(request, self.max_transfer_polls)
                .map(GrpcChatResponse::Single),
            crate::http::HttpChatRequest::Batch(requests) => {
                if stream {
                    return Err(Status::invalid_argument(
                        "streaming batched chat completions are not supported yet",
                    ));
                }
                runtime
                    .generate_text_batch_stream_with_transfer_polling(
                        requests,
                        self.max_transfer_polls,
                    )
                    .map(GrpcChatResponse::Batch)
            }
        }
        .map_err(router_runtime_error_to_status)?;

        match response {
            GrpcChatResponse::Single(responses) if stream => {
                let stream = responses
                    .into_iter()
                    .map(|response| {
                        openai_chat_stream_response_from_router_response(response, &model)
                    })
                    .collect::<Vec<_>>();
                Ok(Response::new(Box::pin(tonic::codegen::tokio_stream::iter(
                    stream,
                ))))
            }
            GrpcChatResponse::Single(responses) => {
                let response = openai_chat_response_from_router_responses(responses, &model)?;
                Ok(Response::new(Box::pin(tonic::codegen::tokio_stream::iter(
                    [Ok(response)],
                ))))
            }
            GrpcChatResponse::Batch(batch_responses) => {
                let response =
                    openai_chat_batch_response_from_router_responses(batch_responses, &model)?;
                Ok(Response::new(Box::pin(tonic::codegen::tokio_stream::iter(
                    [Ok(response)],
                ))))
            }
        }
    }

    async fn complete(
        &self,
        request: Request<OpenAiJsonRequest>,
    ) -> Result<Response<Self::CompleteStream>, Status> {
        let request = request.into_inner();
        let options = request.options.unwrap_or_default();
        let payload: Value = serde_json::from_slice(&request.json)
            .map_err(|e| Status::invalid_argument(format!("invalid OpenAI JSON payload: {e}")))?;
        let model = self.model_info()?.served_model_name;
        let mut request = crate::http::http_completion_payload_to_router_request(payload, &model)
            .map_err(Status::invalid_argument)?;
        match &mut request {
            crate::http::HttpCompletionRequest::Single(request) => {
                apply_openai_json_options_to_router_request(request, options);
            }
            crate::http::HttpCompletionRequest::Batch(requests) => {
                for request in requests {
                    apply_openai_json_options_to_router_request(request, options.clone());
                }
            }
        }
        let stream = request.stream();

        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| Status::internal("router runtime mutex poisoned"))?;
        let response = match request {
            crate::http::HttpCompletionRequest::Single(request) => runtime
                .generate_text_stream_with_transfer_polling(request, self.max_transfer_polls)
                .map(GrpcCompletionResponse::Single),
            crate::http::HttpCompletionRequest::Batch(requests) => {
                if stream {
                    return Err(Status::invalid_argument(
                        "streaming batched completions are not supported yet",
                    ));
                }
                runtime
                    .generate_text_batch_stream_with_transfer_polling(
                        requests,
                        self.max_transfer_polls,
                    )
                    .map(GrpcCompletionResponse::Batch)
            }
        }
        .map_err(router_runtime_error_to_status)?;

        match response {
            GrpcCompletionResponse::Single(responses) if stream => {
                let stream = responses
                    .into_iter()
                    .map(|response| {
                        openai_completion_stream_response_from_router_response(response, &model)
                    })
                    .collect::<Vec<_>>();
                Ok(Response::new(Box::pin(tonic::codegen::tokio_stream::iter(
                    stream,
                ))))
            }
            GrpcCompletionResponse::Single(responses) => {
                let response = openai_completion_response_from_router_responses(responses, &model)?;
                Ok(Response::new(Box::pin(tonic::codegen::tokio_stream::iter(
                    [Ok(response)],
                ))))
            }
            GrpcCompletionResponse::Batch(batch_responses) => {
                let response = openai_completion_batch_response_from_router_responses(
                    batch_responses,
                    &model,
                )?;
                Ok(Response::new(Box::pin(tonic::codegen::tokio_stream::iter(
                    [Ok(response)],
                ))))
            }
        }
    }

    async fn open_ai_embed(
        &self,
        request: Request<OpenAiJsonRequest>,
    ) -> Result<Response<OpenAiJsonResponse>, Status> {
        let model_info = self.model_info()?;
        let payload: Value = serde_json::from_slice(&request.into_inner().json)
            .map_err(|e| Status::invalid_argument(format!("invalid embeddings JSON: {e}")))?;
        let request = parse_embedding_request(&payload, &model_info.served_model_name)
            .map_err(Status::invalid_argument)?;
        let json = {
            let runtime = self
                .runtime
                .lock()
                .map_err(|_| Status::internal("router runtime mutex poisoned"))?;
            embeddings_response_json(&runtime, &request)
        };
        let json = serde_json::to_vec(&json)
            .map_err(|e| Status::internal(format!("serialize embeddings JSON: {e}")))?;
        Ok(Response::new(OpenAiJsonResponse { json }))
    }

    async fn open_ai_classify(
        &self,
        request: Request<OpenAiJsonRequest>,
    ) -> Result<Response<OpenAiJsonResponse>, Status> {
        let model_info = self.model_info()?;
        let payload: Value = serde_json::from_slice(&request.into_inner().json)
            .map_err(|e| Status::invalid_argument(format!("invalid classify JSON: {e}")))?;
        let request = parse_classify_request(&payload, &model_info.served_model_name)
            .map_err(Status::invalid_argument)?;
        let json = {
            let runtime = self
                .runtime
                .lock()
                .map_err(|_| Status::internal("router runtime mutex poisoned"))?;
            classify_response_json(&runtime, &request)
        };
        let json = serde_json::to_vec(&json)
            .map_err(|e| Status::internal(format!("serialize classify JSON: {e}")))?;
        Ok(Response::new(OpenAiJsonResponse { json }))
    }

    async fn score(
        &self,
        request: Request<OpenAiJsonRequest>,
    ) -> Result<Response<OpenAiJsonResponse>, Status> {
        let model_info = self.model_info()?;
        let payload: Value = serde_json::from_slice(&request.into_inner().json)
            .map_err(|e| Status::invalid_argument(format!("invalid score JSON: {e}")))?;
        let request = parse_score_request(&payload, &model_info.served_model_name)
            .map_err(Status::invalid_argument)?;
        let json = {
            let runtime = self
                .runtime
                .lock()
                .map_err(|_| Status::internal("router runtime mutex poisoned"))?;
            score_response_json(&runtime, &request)
        };
        let json = serde_json::to_vec(&json)
            .map_err(|e| Status::internal(format!("serialize score JSON: {e}")))?;
        Ok(Response::new(OpenAiJsonResponse { json }))
    }

    async fn rerank(
        &self,
        request: Request<OpenAiJsonRequest>,
    ) -> Result<Response<OpenAiJsonResponse>, Status> {
        let model_info = self.model_info()?;
        let payload: Value = serde_json::from_slice(&request.into_inner().json)
            .map_err(|e| Status::invalid_argument(format!("invalid rerank JSON: {e}")))?;
        let request = parse_rerank_request(&payload, &model_info.served_model_name)
            .map_err(Status::invalid_argument)?;
        let mut results = {
            let runtime = self
                .runtime
                .lock()
                .map_err(|_| Status::internal("router runtime mutex poisoned"))?;
            score_rerank_documents(&runtime, &request)
        };
        truncate_rerank_results(&request, &mut results);
        let json = rerank_results_to_json(&request, results);
        let json = serde_json::to_vec(&json)
            .map_err(|e| Status::internal(format!("serialize rerank JSON: {e}")))?;
        Ok(Response::new(OpenAiJsonResponse { json }))
    }

    async fn start_profile(
        &self,
        request: Request<StartProfileRequest>,
    ) -> Result<Response<ControlResponse>, Status> {
        let output_dir =
            profile_output_dir(request.into_inner().output_dir).map_err(profile_error_to_status)?;
        ensure_profile_output_dir(&output_dir).map_err(profile_error_to_status)?;
        let mut profile = self
            .profile
            .lock()
            .map_err(|_| Status::internal("profile mutex poisoned"))?;
        if profile.is_some() {
            return Err(Status::already_exists("profile is already running"));
        }
        *profile = Some(ProfileSession::new(output_dir.clone()));

        Ok(Response::new(ControlResponse {
            success: true,
            message: format!("profile started: {}", output_dir.display()),
        }))
    }

    async fn stop_profile(
        &self,
        _request: Request<StopProfileRequest>,
    ) -> Result<Response<ControlResponse>, Status> {
        let session = self
            .profile
            .lock()
            .map_err(|_| Status::internal("profile mutex poisoned"))?
            .take()
            .ok_or_else(|| Status::failed_precondition("profile is not running"))?;
        let profile_path =
            write_profile_file(session, SystemTime::now(), &self.server_info_attributes)
                .map_err(profile_error_to_status)?;

        Ok(Response::new(ControlResponse {
            success: true,
            message: format!("profile stopped: {}", profile_path.display()),
        }))
    }

    async fn update_weights_from_disk(
        &self,
        request: Request<UpdateWeightsFromDiskRequest>,
    ) -> Result<Response<ControlResponse>, Status> {
        let request = request.into_inner();
        let current = self.model_info()?;
        let update = update_model_info_from_disk(
            current,
            &request.model_path,
            request.load_format.as_deref(),
        )
        .map_err(Status::invalid_argument)?;
        {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| Status::internal("router runtime mutex poisoned"))?;
            runtime
                .update_weights_from_disk(update.worker_request.clone())
                .map_err(|error| Status::failed_precondition(error.to_string()))?;
        }

        *self
            .model_info
            .lock()
            .map_err(|_| Status::internal("model info mutex poisoned"))? = Some(update.model_info);

        Ok(Response::new(ControlResponse {
            success: true,
            message: update.message,
        }))
    }

    async fn update_weight_version(
        &self,
        request: Request<UpdateWeightVersionRequest>,
    ) -> Result<Response<ControlResponse>, Status> {
        let request = request.into_inner();
        let new_version = request.new_version.trim();
        if new_version.is_empty() {
            return Err(Status::invalid_argument(
                "new_version is required and must be a non-empty string",
            ));
        }

        if request.abort_all_requests.unwrap_or(true) {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| Status::internal("router runtime mutex poisoned"))?;
            runtime.abort_all_requests();
        }

        let mut model_info = self.model_info()?;
        model_info.weight_version = new_version.to_string();
        *self
            .model_info
            .lock()
            .map_err(|_| Status::internal("model info mutex poisoned"))? = Some(model_info);

        Ok(Response::new(ControlResponse {
            success: true,
            message: format!("Weight version updated to {new_version}"),
        }))
    }

    async fn get_weights_by_name(
        &self,
        request: Request<GetWeightsByNameRequest>,
    ) -> Result<Response<GetWeightsByNameResponse>, Status> {
        let request = request.into_inner();
        let model_info = self.model_info()?;
        let truncate_size = request.truncate_size.map(|value| value as usize);
        let parameter = get_weights_by_name_from_disk(&model_info, &request.name, truncate_size)
            .map_err(Status::invalid_argument)?;

        Ok(Response::new(GetWeightsByNameResponse { parameter }))
    }
}

fn default_server_info_attributes() -> HashMap<String, String> {
    HashMap::from([("transport".to_string(), "tonic-grpc".to_string())])
}

fn server_info_attributes_from_args(args: &ServerArgs) -> HashMap<String, String> {
    let mut attributes = default_server_info_attributes();
    attributes.insert(
        "served_model_name".to_string(),
        args.served_model_name
            .clone()
            .unwrap_or_else(|| args.model_path.clone()),
    );
    attributes.insert(
        "disaggregation_mode".to_string(),
        args.disaggregation_mode.clone(),
    );

    if args.disaggregation_mode == "prefill" {
        attributes.insert(
            "disaggregation_bootstrap_port".to_string(),
            args.disaggregation_bootstrap_port.to_string(),
        );
    }

    if let Some(ports) = args.disaggregation_zmq_ports {
        attributes.insert("kv_events.publisher".to_string(), "zmq".to_string());
        attributes.insert(
            "kv_events.endpoint_host".to_string(),
            grpc_kv_events_endpoint_host(args).to_string(),
        );
        attributes.insert(
            "kv_events.endpoint_port_base".to_string(),
            ports.start.to_string(),
        );
        attributes.insert("kv_events.topic".to_string(), String::new());
        attributes.insert(
            "kv_events.block_size".to_string(),
            args.page_size.to_string(),
        );
        attributes.insert("kv_events.dp_size".to_string(), args.dp_size.to_string());
    }

    if let Some(layout) = PdConfig::from_server_args(args)
        .ok()
        .and_then(|config| config.kv_cache_runtime_layout().ok().flatten())
    {
        attributes.insert(
            "kv_cache.dtype".to_string(),
            layout.dtype.as_str().to_string(),
        );
        attributes.insert(
            "kv_cache.page_size".to_string(),
            layout.page_size.to_string(),
        );
        attributes.insert(
            "kv_cache.num_layers".to_string(),
            layout.num_layers.to_string(),
        );
        attributes.insert("kv_cache.kv_heads".to_string(), layout.kv_heads.to_string());
        attributes.insert("kv_cache.head_dim".to_string(), layout.head_dim.to_string());
        attributes.insert(
            "kv_cache.kv_tensors_per_token".to_string(),
            layout.kv_tensors_per_token.to_string(),
        );
        attributes.insert(
            "kv_cache.bytes_per_token".to_string(),
            layout.bytes_per_token.to_string(),
        );
        attributes.insert(
            "kv_cache.page_size_bytes".to_string(),
            layout.page_size_bytes.to_string(),
        );
    }

    attributes
}

fn grpc_kv_events_endpoint_host(args: &ServerArgs) -> &str {
    if matches!(args.host.as_str(), "0.0.0.0" | "::" | "[::]") {
        if let Some(host) = args.dist_init_addr.as_deref().and_then(host_from_addr) {
            return host;
        }
    }
    &args.host
}

fn host_from_addr(addr: &str) -> Option<&str> {
    if let Some(rest) = addr.strip_prefix('[') {
        let (host, _) = rest.split_once(']')?;
        return Some(host);
    }
    addr.rsplit_once(':')
        .map(|(host, _)| host)
        .filter(|host| !host.is_empty())
}

fn apply_openai_json_options_to_router_request(
    request: &mut RouterTextGenerateRequest,
    options: crate::proto::sglang::runtime::v1::RequestOptions,
) {
    if request.request_id.is_empty() {
        request.request_id = options.request_id.unwrap_or_default();
    }
    if !request.stream {
        request.stream = options.stream;
    }
    if request.data_parallel_rank == 0 {
        request.data_parallel_rank = options.data_parallel_rank;
    }
    request
        .trace_headers
        .extend(options.trace_headers.into_iter());
}

fn proto_generate_request_to_router_request(
    request: ProtoGenerateRequest,
) -> Result<RouterGenerateRequest, Status> {
    let options = request.options.unwrap_or_default();

    Ok(RouterGenerateRequest {
        request_id: options.request_id.unwrap_or_default(),
        tokenized: Some(crate::router::RouterTokenizedInput {
            original_text: request.original_text,
            input_ids: request.input_ids,
        }),
        sampling_params: request.sampling_params.map(proto_sampling_params_to_router),
        disaggregated_params: request
            .disaggregated_params
            .map(proto_disaggregated_params_to_router)
            .transpose()?,
        stream: options.stream,
        data_parallel_rank: options.data_parallel_rank,
        trace_headers: options
            .trace_headers
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
    })
}

fn proto_text_generate_request_to_router_request(
    request: TextGenerateRequest,
) -> Result<RouterTextGenerateRequest, Status> {
    let options = request.options.unwrap_or_default();

    Ok(RouterTextGenerateRequest {
        request_id: options.request_id.unwrap_or_default(),
        text: request.text,
        sampling_params: request.sampling_params.map(proto_sampling_params_to_router),
        disaggregated_params: request
            .disaggregated_params
            .map(proto_disaggregated_params_to_router)
            .transpose()?,
        stream: options.stream,
        data_parallel_rank: options.data_parallel_rank,
        trace_headers: options
            .trace_headers
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
    })
}

fn proto_sampling_params_to_router(params: ProtoSamplingParams) -> RouterSamplingParams {
    RouterSamplingParams {
        max_new_tokens: params.max_new_tokens,
        temperature: params.temperature,
        top_p: params.top_p,
        top_k: params.top_k,
        min_p: params.min_p,
        frequency_penalty: params.frequency_penalty,
        presence_penalty: params.presence_penalty,
        repetition_penalty: params.repetition_penalty,
        stop_token_id: params.stop_token_id,
        stop_token_ids: params
            .stop_token_ids
            .into_iter()
            .map(|stop_token_id| stop_token_id as i32)
            .collect(),
        ignore_eos: params.ignore_eos,
        n: params.n,
        best_of: params.best_of,
    }
}

fn proto_disaggregated_params_to_router(
    params: crate::proto::sglang::runtime::v1::DisaggregatedParams,
) -> Result<RouterDisaggregatedParams, Status> {
    let bootstrap_port = u16::try_from(params.bootstrap_port).map_err(|_| {
        Status::invalid_argument(format!(
            "bootstrap_port must fit in u16: {}",
            params.bootstrap_port
        ))
    })?;

    Ok(RouterDisaggregatedParams {
        bootstrap_host: params.bootstrap_host,
        bootstrap_port,
        bootstrap_room: params.bootstrap_room,
    })
}

fn router_generate_response_to_proto_response(
    response: RouterGenerateResponse,
) -> ProtoGenerateResponse {
    ProtoGenerateResponse {
        request_id: response.request_id,
        body: Some(match response.body {
            RouterGenerateResponseBody::Chunk(chunk) => {
                ProtoGenerateResponseBody::Chunk(router_chunk_to_proto(chunk))
            }
            RouterGenerateResponseBody::Complete(complete) => {
                ProtoGenerateResponseBody::Complete(router_complete_to_proto(complete))
            }
            RouterGenerateResponseBody::Error(error) => {
                ProtoGenerateResponseBody::Error(router_error_to_proto(error))
            }
        }),
    }
}

fn router_chunk_to_proto(chunk: RouterGenerateStreamChunk) -> ProtoGenerateStreamChunk {
    ProtoGenerateStreamChunk {
        token_ids: chunk.token_ids,
        text: chunk.text,
        prompt_tokens: chunk.prompt_tokens,
        completion_tokens: chunk.completion_tokens,
        cached_tokens: chunk.cached_tokens,
        index: chunk.index,
    }
}

fn router_complete_to_proto(complete: RouterGenerateComplete) -> ProtoGenerateComplete {
    ProtoGenerateComplete {
        output_ids: complete.output_ids,
        text: complete.text,
        finish_reason: complete.finish_reason,
        prompt_tokens: complete.prompt_tokens,
        completion_tokens: complete.completion_tokens,
        cached_tokens: complete.cached_tokens,
        index: complete.index,
    }
}

fn router_error_to_proto(error: RouterGenerateError) -> ProtoGenerateError {
    ProtoGenerateError {
        message: error.message,
        code: String::new(),
    }
}

fn router_model_info_to_proto_response(
    model_info: RouterGetModelInfoResponse,
) -> ModelInfoResponse {
    ModelInfoResponse {
        model_path: model_info.model_path,
        tokenizer_path: model_info.tokenizer_path,
        is_generation: model_info.is_generation,
        preferred_sampling_params_json: model_info.preferred_sampling_params,
        weight_version: model_info.weight_version,
        served_model_name: model_info.served_model_name,
        max_context_length: model_info.max_context_length,
        vocab_size: model_info.vocab_size,
        supports_vision: model_info.supports_vision,
        model_type: model_info.model_type,
        eos_token_ids: model_info.eos_token_ids,
        pad_token_id: model_info.pad_token_id,
        bos_token_id: model_info.bos_token_id,
        max_request_input_length: model_info.max_req_input_len,
        routed_expert_expected_group_count: model_info.routed_expert_expected_group_count,
        routed_expert_actual_group_count: model_info.routed_expert_actual_group_count,
        routed_expert_expected_weight_count: model_info.routed_expert_expected_weight_count,
        routed_expert_actual_weight_count: model_info.routed_expert_actual_weight_count,
    }
}

fn usize_to_u32(value: usize) -> Result<u32, Status> {
    u32::try_from(value).map_err(|_| Status::internal("load metric overflowed uint32"))
}

fn usize_to_i32(value: usize) -> Result<i32, Status> {
    i32::try_from(value).map_err(|_| Status::internal("usage metric overflowed int32"))
}
