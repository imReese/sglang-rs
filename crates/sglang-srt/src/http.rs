use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::extract::{Query, State};
use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::engine_info_bootstrap::EngineInfoBootstrapService;
use crate::openai_classify::{classify_response_json, parse_classify_request};
use crate::openai_embedding::{embeddings_response_json, parse_embedding_request};
use crate::openai_rerank::{
    parse_rerank_request, rerank_results_to_json, score_rerank_documents, truncate_rerank_results,
};
use crate::openai_score::{parse_score_request, score_response_json};
use crate::profile::{
    ProfileError, ProfileSession, ensure_profile_output_dir, profile_output_dir, write_profile_file,
};
use crate::router::{
    RouterDisaggregatedParams, RouterGenerateComplete, RouterGenerateRequest,
    RouterGenerateResponse, RouterGenerateResponseBody, RouterGetModelInfoResponse,
    RouterProtocolError, RouterRuntime, RouterRuntimeError, RouterSamplingParams, RouterStatusCode,
    RouterTextGenerateRequest, RouterTokenizedInput,
};
use crate::tokenizer::Tokenizer;
use crate::types::BootstrapRoom;
use crate::weight_update::{get_weights_by_name_from_disk, update_model_info_from_disk};
use crate::worker::WorkerExecutor;

pub struct HttpRouterService<T, W> {
    runtime: Arc<Mutex<RouterRuntime<T, W>>>,
    model_info: Arc<Mutex<RouterGetModelInfoResponse>>,
    profile: Arc<Mutex<Option<ProfileSession>>>,
    profile_attributes: HashMap<String, String>,
    server_info: HttpServerInfo,
    engine_info_bootstrap: Option<EngineInfoBootstrapService>,
    allow_disaggregated_requests: bool,
    max_transfer_polls: usize,
}

impl<T, W> Clone for HttpRouterService<T, W> {
    fn clone(&self) -> Self {
        Self {
            runtime: Arc::clone(&self.runtime),
            model_info: Arc::clone(&self.model_info),
            profile: Arc::clone(&self.profile),
            profile_attributes: self.profile_attributes.clone(),
            server_info: self.server_info.clone(),
            engine_info_bootstrap: self.engine_info_bootstrap.clone(),
            allow_disaggregated_requests: self.allow_disaggregated_requests,
            max_transfer_polls: self.max_transfer_polls,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpServerInfo {
    pub disaggregation_mode: String,
    pub disaggregation_bootstrap_port: Option<u16>,
    pub kv_events: Option<HttpKvEventsInfo>,
    pub kv_cache: Option<HttpKvCacheInfo>,
}

impl Default for HttpServerInfo {
    fn default() -> Self {
        Self {
            disaggregation_mode: "null".to_string(),
            disaggregation_bootstrap_port: None,
            kv_events: None,
            kv_cache: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpKvEventsInfo {
    pub publisher: String,
    pub endpoint_host: String,
    pub endpoint_port_base: u16,
    pub topic: String,
    pub block_size: u32,
    pub dp_size: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpKvCacheInfo {
    pub dtype: String,
    pub page_size: u64,
    pub num_layers: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub kv_tensors_per_token: u64,
    pub bytes_per_token: u64,
    pub page_size_bytes: u64,
}

impl<T, W> HttpRouterService<T, W> {
    pub fn new(runtime: RouterRuntime<T, W>, model_info: RouterGetModelInfoResponse) -> Self {
        Self {
            runtime: Arc::new(Mutex::new(runtime)),
            model_info: Arc::new(Mutex::new(model_info)),
            profile: Arc::new(Mutex::new(None)),
            profile_attributes: HashMap::from([("transport".to_string(), "axum-http".to_string())]),
            server_info: HttpServerInfo::default(),
            engine_info_bootstrap: None,
            allow_disaggregated_requests: false,
            max_transfer_polls: 0,
        }
    }

    pub fn runtime(&self) -> &Arc<Mutex<RouterRuntime<T, W>>> {
        &self.runtime
    }

    fn model_info_snapshot(&self) -> Result<RouterGetModelInfoResponse, &'static str> {
        self.model_info
            .lock()
            .map_err(|_| "model info mutex poisoned")
            .map(|model_info| model_info.clone())
    }

    fn replace_model_info(
        &self,
        model_info: RouterGetModelInfoResponse,
    ) -> Result<(), &'static str> {
        *self
            .model_info
            .lock()
            .map_err(|_| "model info mutex poisoned")? = model_info;
        Ok(())
    }

    pub fn with_disaggregated_requests(mut self) -> Self {
        self.allow_disaggregated_requests = true;
        self
    }

    pub fn with_server_info(mut self, server_info: HttpServerInfo) -> Self {
        self.server_info = server_info;
        self
    }

    pub fn with_engine_info_bootstrap_service(
        mut self,
        engine_info_bootstrap: EngineInfoBootstrapService,
    ) -> Self {
        self.engine_info_bootstrap = Some(engine_info_bootstrap);
        self
    }

    pub fn with_max_transfer_polls(mut self, max_transfer_polls: usize) -> Self {
        self.max_transfer_polls = max_transfer_polls;
        self
    }
}

impl<T, W> HttpRouterService<T, W>
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    fn into_router(self) -> Router {
        Router::new()
            .route("/health", get(health))
            .route("/v1/models", get(list_models::<T, W>))
            .route("/model_info", get(model_info::<T, W>))
            .route("/get_model_info", get(model_info::<T, W>))
            .route("/server_info", get(server_info::<T, W>))
            .route("/get_server_info", get(server_info::<T, W>))
            .route(
                "/remote_instance_transfer_engine_info",
                get(remote_instance_transfer_engine_info::<T, W>),
            )
            .route(
                "/get_remote_instance_transfer_engine_info",
                get(remote_instance_transfer_engine_info::<T, W>),
            )
            .route("/v1/loads", get(loads::<T, W>))
            .route("/get_loads", get(loads::<T, W>))
            .route("/get_load", get(legacy_load::<T, W>))
            .route(
                "/flush_cache",
                get(flush_cache::<T, W>).post(flush_cache::<T, W>),
            )
            .route("/pause_generation", post(pause_generation::<T, W>))
            .route("/continue_generation", post(continue_generation::<T, W>))
            .route("/abort_request", post(abort_request::<T, W>))
            .route(
                "/start_profile",
                get(start_profile::<T, W>).post(start_profile::<T, W>),
            )
            .route(
                "/stop_profile",
                get(stop_profile::<T, W>).post(stop_profile::<T, W>),
            )
            .route(
                "/update_weights_from_disk",
                post(update_weights_from_disk::<T, W>),
            )
            .route(
                "/update_weight_version",
                post(update_weight_version::<T, W>),
            )
            .route(
                "/get_weights_by_name",
                get(get_weights_by_name::<T, W>).post(get_weights_by_name::<T, W>),
            )
            .route("/v1/tokenize", post(tokenize::<T, W>))
            .route("/tokenize", post(tokenize::<T, W>))
            .route("/v1/detokenize", post(detokenize::<T, W>))
            .route("/detokenize", post(detokenize::<T, W>))
            .route("/v1/chat/completions", post(chat_completions::<T, W>))
            .route("/v1/completions", post(completions::<T, W>))
            .route("/v1/rerank", post(rerank::<T, W>))
            .route("/rerank", post(rerank::<T, W>))
            .route("/v1/score", post(score::<T, W>))
            .route("/v1/embeddings", post(embeddings::<T, W>))
            .route("/v1/classify", post(classify::<T, W>))
            .route("/generate", post(generate::<T, W>))
            .with_state(self)
    }
}

#[derive(Debug)]
pub enum HttpServeError {
    Io(std::io::Error),
}

impl fmt::Display for HttpServeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "http server error: {error}"),
        }
    }
}

impl std::error::Error for HttpServeError {}

impl From<std::io::Error> for HttpServeError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

pub async fn serve_http_router<T, W>(
    addr: SocketAddr,
    service: HttpRouterService<T, W>,
) -> Result<(), HttpServeError>
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    serve_http_router_with_shutdown(addr, service, std::future::pending::<()>()).await
}

pub async fn serve_http_router_with_shutdown<T, W, F>(
    addr: SocketAddr,
    service: HttpRouterService<T, W>,
    shutdown: F,
) -> Result<(), HttpServeError>
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
    F: Future<Output = ()> + Send + 'static,
{
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, service.into_router())
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

async fn health() -> Json<Value> {
    Json(json!({
        "healthy": true,
        "message": "ready"
    }))
}

async fn list_models<T, W>(State(service): State<HttpRouterService<T, W>>) -> Response
where
    T: Send + 'static,
    W: Send + 'static,
{
    let info = match service.model_info_snapshot() {
        Ok(info) => info,
        Err(message) => return internal_error_json(message),
    };
    Json(json!({
        "object": "list",
        "data": [{
            "id": info.served_model_name,
            "object": "model",
            "owned_by": "sglang-rs",
            "root": info.model_path,
        }]
    }))
    .into_response()
}

async fn server_info<T, W>(State(service): State<HttpRouterService<T, W>>) -> Response
where
    T: Send + 'static,
    W: Send + 'static,
{
    let info = match service.model_info_snapshot() {
        Ok(info) => info,
        Err(message) => return internal_error_json(message),
    };
    let mut body = json!({
        "served_model_name": info.served_model_name,
        "disaggregation_mode": service.server_info.disaggregation_mode,
    });

    if let Some(port) = service.server_info.disaggregation_bootstrap_port {
        body["disaggregation_bootstrap_port"] = json!(port);
    }
    if let Some(kv_events) = service.server_info.kv_events {
        body["kv_events"] = json!({
            "publisher": kv_events.publisher,
            "endpoint_host": kv_events.endpoint_host,
            "endpoint_port_base": kv_events.endpoint_port_base,
            "topic": kv_events.topic,
            "block_size": kv_events.block_size,
            "dp_size": kv_events.dp_size,
        });
    }
    if let Some(kv_cache) = service.server_info.kv_cache {
        body["kv_cache"] = json!({
            "dtype": kv_cache.dtype,
            "page_size": kv_cache.page_size,
            "num_layers": kv_cache.num_layers,
            "kv_heads": kv_cache.kv_heads,
            "head_dim": kv_cache.head_dim,
            "kv_tensors_per_token": kv_cache.kv_tensors_per_token,
            "bytes_per_token": kv_cache.bytes_per_token,
            "page_size_bytes": kv_cache.page_size_bytes,
        });
    }

    Json(body).into_response()
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct RemoteInstanceTransferEngineInfoQuery {
    rank: Option<i32>,
}

async fn remote_instance_transfer_engine_info<T, W>(
    State(service): State<HttpRouterService<T, W>>,
    Query(query): Query<RemoteInstanceTransferEngineInfoQuery>,
) -> Response
where
    T: Send + 'static,
    W: Send + 'static,
{
    let Some(rank) = query.rank.filter(|rank| *rank >= 0) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {"message": "Missing or invalid rank parameter"}
            })),
        )
            .into_response();
    };

    let Some(engine_info_bootstrap) = service.engine_info_bootstrap else {
        return failed_transfer_engine_info_response(rank);
    };
    let Some(info) = engine_info_bootstrap.transfer_engine_info(rank) else {
        return failed_transfer_engine_info_response(rank);
    };

    (
        StatusCode::OK,
        Json(json!({
            "rank": rank,
            "remote_instance_transfer_engine_info": [
                info.session_id,
                info.weights_info_dict,
            ],
        })),
    )
        .into_response()
}

fn failed_transfer_engine_info_response(rank: i32) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({
            "error": {
                "message": format!("Failed to get transfer engine info for rank {rank}")
            }
        })),
    )
        .into_response()
}

async fn model_info<T, W>(State(service): State<HttpRouterService<T, W>>) -> Response
where
    T: Send + 'static,
    W: Send + 'static,
{
    let info = match service.model_info_snapshot() {
        Ok(info) => info,
        Err(message) => return internal_error_json(message),
    };
    Json(model_info_json(info)).into_response()
}

fn model_info_json(info: RouterGetModelInfoResponse) -> Value {
    json!({
        "model_path": info.model_path,
        "tokenizer_path": info.tokenizer_path,
        "is_generation": info.is_generation,
        "preferred_sampling_params": info.preferred_sampling_params,
        "weight_version": info.weight_version,
        "has_image_understanding": info.supports_vision,
        "has_audio_understanding": false,
        "model_type": info.model_type,
        "architectures": info.architectures,
        "max_context_length": info.max_context_length,
        "vocab_size": info.vocab_size,
        "eos_token_ids": info.eos_token_ids,
        "pad_token_id": info.pad_token_id,
        "bos_token_id": info.bos_token_id,
        "max_req_input_len": info.max_req_input_len,
    })
}

async fn loads<T, W>(State(service): State<HttpRouterService<T, W>>) -> Response
where
    T: Send + 'static,
    W: Send + 'static,
{
    let load = match service.runtime.lock() {
        Ok(runtime) => runtime.load(),
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {"message": "router runtime mutex poisoned"}
                })),
            )
                .into_response();
        }
    };
    let num_running_reqs = load.decode_queue_depth;
    let num_waiting_reqs = load.waiting_queue_depth;

    Json(json!({
        "timestamp": unix_timestamp_secs(),
        "version": env!("CARGO_PKG_VERSION"),
        "loads": [{
            "dp_rank": 0,
            "num_running_reqs": num_running_reqs,
            "num_waiting_reqs": num_waiting_reqs,
            "num_reqs": num_running_reqs + num_waiting_reqs,
            "waiting_queue_depth": load.waiting_queue_depth,
            "decode_queue_depth": load.decode_queue_depth,
            "available_cache_pages": load.available_cache_pages,
        }]
    }))
    .into_response()
}

async fn legacy_load<T, W>(State(service): State<HttpRouterService<T, W>>) -> Response
where
    T: Send + 'static,
    W: Send + 'static,
{
    let load = match service.runtime.lock() {
        Ok(runtime) => runtime.load(),
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {"message": "router runtime mutex poisoned"}
                })),
            )
                .into_response();
        }
    };
    let num_running_reqs = load.decode_queue_depth;
    let num_waiting_reqs = load.waiting_queue_depth;

    Json(json!([{
        "dp_rank": 0,
        "num_reqs": num_running_reqs + num_waiting_reqs,
        "num_waiting_reqs": num_waiting_reqs,
        "num_tokens": 0,
        "num_pending_tokens": 0,
    }]))
    .into_response()
}

async fn flush_cache<T, W>(State(service): State<HttpRouterService<T, W>>) -> Response
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let response = match service.runtime.lock() {
        Ok(mut runtime) => runtime.flush_cache(),
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "router runtime mutex poisoned",
            )
                .into_response();
        }
    };

    if response.success {
        (
            StatusCode::OK,
            "Cache flushed.\nPlease check backend logs for more details. (When there are running or waiting requests, the operation will not be performed.)\n",
        )
            .into_response()
    } else {
        let message = if response.message.is_empty() {
            "Flush cache failed.\n".to_string()
        } else {
            response.message
        };
        (StatusCode::BAD_REQUEST, message).into_response()
    }
}

async fn pause_generation<T, W>(State(service): State<HttpRouterService<T, W>>) -> Response
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let response = match service.runtime.lock() {
        Ok(mut runtime) => runtime.pause_generation(),
        Err(_) => return internal_error_json("router runtime mutex poisoned"),
    };

    Json(json!({
        "success": response.success,
        "message": response.message,
    }))
    .into_response()
}

async fn continue_generation<T, W>(State(service): State<HttpRouterService<T, W>>) -> Response
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let response = match service.runtime.lock() {
        Ok(mut runtime) => runtime.continue_generation(),
        Err(_) => return internal_error_json("router runtime mutex poisoned"),
    };

    Json(json!({
        "success": response.success,
        "message": response.message,
    }))
    .into_response()
}

async fn abort_request<T, W>(
    State(service): State<HttpRouterService<T, W>>,
    Json(payload): Json<Value>,
) -> Response
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    if payload
        .get("abort_all")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        let response = match service.runtime.lock() {
            Ok(mut runtime) => runtime.abort_all_requests(),
            Err(_) => return internal_error_json("router runtime mutex poisoned"),
        };

        return Json(json!({
            "success": response.success,
            "message": response.message,
        }))
        .into_response();
    }
    let request_id = payload
        .get("rid")
        .or_else(|| payload.get("request_id"))
        .and_then(Value::as_str)
        .unwrap_or_default();

    let response = match service.runtime.lock() {
        Ok(mut runtime) => match runtime.abort_request(request_id) {
            Ok(response) => response,
            Err(error) => return router_protocol_error_json(error),
        },
        Err(_) => return internal_error_json("router runtime mutex poisoned"),
    };

    Json(json!({
        "success": response.success,
        "message": response.message,
    }))
    .into_response()
}

async fn start_profile<T, W>(
    State(service): State<HttpRouterService<T, W>>,
    body: Bytes,
) -> Response
where
    T: Send + 'static,
    W: Send + 'static,
{
    let requested_dir = match profile_output_dir_from_body(&body) {
        Ok(output_dir) => output_dir,
        Err(response) => return response,
    };
    let output_dir = match profile_output_dir(requested_dir) {
        Ok(output_dir) => output_dir,
        Err(error) => return profile_error_json(error),
    };
    if let Err(error) = ensure_profile_output_dir(&output_dir) {
        return profile_error_json(error);
    }

    let mut profile = match service.profile.lock() {
        Ok(profile) => profile,
        Err(_) => return internal_error_json("profile mutex poisoned"),
    };
    if profile.is_some() {
        return (
            StatusCode::CONFLICT,
            Json(json!({ "error": { "message": "profile is already running" } })),
        )
            .into_response();
    }

    *profile = Some(ProfileSession::new(output_dir.clone()));
    Json(json!({
        "success": true,
        "message": format!("profile started: {}", output_dir.display()),
    }))
    .into_response()
}

async fn stop_profile<T, W>(State(service): State<HttpRouterService<T, W>>) -> Response
where
    T: Send + 'static,
    W: Send + 'static,
{
    let session = match service.profile.lock() {
        Ok(mut profile) => match profile.take() {
            Some(session) => session,
            None => {
                return (
                    StatusCode::PRECONDITION_FAILED,
                    Json(json!({ "error": { "message": "profile is not running" } })),
                )
                    .into_response();
            }
        },
        Err(_) => return internal_error_json("profile mutex poisoned"),
    };
    let profile_path =
        match write_profile_file(session, SystemTime::now(), &service.profile_attributes) {
            Ok(profile_path) => profile_path,
            Err(error) => return profile_error_json(error),
        };

    Json(json!({
        "success": true,
        "message": format!("profile stopped: {}", profile_path.display()),
    }))
    .into_response()
}

fn profile_output_dir_from_body(body: &[u8]) -> Result<Option<String>, Response> {
    if body.is_empty() {
        return Ok(None);
    }
    let value: Value = serde_json::from_slice(body).map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": { "message": "request body must be a JSON object" } })),
        )
            .into_response()
    })?;
    let object = value.as_object().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": { "message": "request body must be a JSON object" } })),
        )
            .into_response()
    })?;
    match object.get("output_dir") {
        Some(Value::String(output_dir)) => Ok(Some(output_dir.clone())),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": { "message": "output_dir must be a string when provided" } })),
        )
            .into_response()),
    }
}

fn profile_error_json(error: ProfileError) -> Response {
    match error {
        ProfileError::InvalidArgument(message) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": { "message": message } })),
        )
            .into_response(),
        ProfileError::Internal(message) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": { "message": message } })),
        )
            .into_response(),
    }
}

async fn update_weights_from_disk<T, W>(
    State(service): State<HttpRouterService<T, W>>,
    Json(payload): Json<Value>,
) -> Response
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let model_path = match payload.get("model_path").and_then(Value::as_str) {
        Some(model_path) => model_path,
        None => {
            return update_weights_bad_request(
                "model_path is required and must be a string".to_string(),
            );
        }
    };
    let load_format = match payload.get("load_format") {
        Some(Value::String(load_format)) => Some(load_format.as_str()),
        Some(Value::Null) | None => None,
        Some(_) => {
            return update_weights_bad_request(
                "load_format must be a string when provided".to_string(),
            );
        }
    };
    let current = match service.model_info_snapshot() {
        Ok(info) => info,
        Err(message) => return internal_error_json(message),
    };
    let update = match update_model_info_from_disk(current, model_path, load_format) {
        Ok(update) => update,
        Err(message) => return update_weights_bad_request(message),
    };

    let reload = match service.runtime.lock() {
        Ok(mut runtime) => runtime.update_weights_from_disk(update.worker_request.clone()),
        Err(_) => return internal_error_json("router runtime mutex poisoned"),
    };
    if let Err(error) = reload {
        return update_weights_bad_request(error.to_string());
    }

    if let Err(message) = service.replace_model_info(update.model_info) {
        return internal_error_json(message);
    }

    (
        StatusCode::OK,
        Json(json!({
            "success": true,
            "message": update.message,
            "num_paused_requests": 0,
        })),
    )
        .into_response()
}

async fn update_weight_version<T, W>(
    State(service): State<HttpRouterService<T, W>>,
    Json(payload): Json<Value>,
) -> Response
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let new_version = match payload.get("new_version").and_then(Value::as_str) {
        Some(new_version) if !new_version.is_empty() => new_version.to_string(),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "success": false,
                    "message": "new_version is required and must be a non-empty string",
                })),
            )
                .into_response();
        }
    };
    let abort_all_requests = payload
        .get("abort_all_requests")
        .and_then(Value::as_bool)
        .unwrap_or(true);

    if abort_all_requests {
        match service.runtime.lock() {
            Ok(mut runtime) => {
                runtime.abort_all_requests();
            }
            Err(_) => return internal_error_json("router runtime mutex poisoned"),
        }
    }

    let mut model_info = match service.model_info_snapshot() {
        Ok(info) => info,
        Err(message) => return internal_error_json(message),
    };
    model_info.weight_version = new_version.clone();
    if let Err(message) = service.replace_model_info(model_info) {
        return internal_error_json(message);
    }

    (
        StatusCode::OK,
        Json(json!({
            "success": true,
            "message": format!("Weight version updated to {new_version}"),
            "new_version": new_version,
        })),
    )
        .into_response()
}

async fn get_weights_by_name<T, W>(
    State(service): State<HttpRouterService<T, W>>,
    Json(payload): Json<Value>,
) -> Response
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let name = match payload.get("name").and_then(Value::as_str) {
        Some(name) => name,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": { "message": "name is required and must be a string" }
                })),
            )
                .into_response();
        }
    };
    let truncate_size = match payload.get("truncate_size") {
        Some(Value::Number(value)) => {
            match value.as_u64().and_then(|value| usize::try_from(value).ok()) {
                Some(value) => Some(value),
                None => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({
                            "error": { "message": "truncate_size must be a non-negative integer" }
                        })),
                    )
                        .into_response();
                }
            }
        }
        Some(Value::Null) | None => None,
        Some(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": { "message": "truncate_size must be a non-negative integer" }
                })),
            )
                .into_response();
        }
    };
    let model_info = match service.model_info_snapshot() {
        Ok(info) => info,
        Err(message) => return internal_error_json(message),
    };
    match get_weights_by_name_from_disk(&model_info, name, truncate_size) {
        Ok(parameter) => (StatusCode::OK, Json(json!({ "parameter": parameter }))).into_response(),
        Err(message) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": { "message": message } })),
        )
            .into_response(),
    }
}

fn update_weights_bad_request(message: String) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({
            "success": false,
            "message": message,
            "num_paused_requests": 0,
        })),
    )
        .into_response()
}

enum TokenizePrompt {
    Single(String),
    Batch(Vec<String>),
}

enum DetokenizeTokens {
    Single(Vec<u32>),
    Batch(Vec<Vec<u32>>),
}

async fn tokenize<T, W>(
    State(service): State<HttpRouterService<T, W>>,
    Json(payload): Json<Value>,
) -> Response
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let prompt = match parse_tokenize_prompt(&payload) {
        Ok(prompt) => prompt,
        Err(message) => return bad_request_json(message),
    };

    let info = match service.model_info_snapshot() {
        Ok(info) => info,
        Err(message) => return internal_error_json(message),
    };
    let max_model_len = if info.max_context_length > 0 {
        info.max_context_length
    } else {
        -1
    };
    let runtime = match service.runtime.lock() {
        Ok(runtime) => runtime,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {"message": "router runtime mutex poisoned"}
                })),
            )
                .into_response();
        }
    };

    match prompt {
        TokenizePrompt::Single(prompt) => {
            let response = runtime.tokenize(&prompt);
            Json(json!({
                "tokens": response.token_ids,
                "count": response.token_ids.len(),
                "max_model_len": max_model_len,
            }))
            .into_response()
        }
        TokenizePrompt::Batch(prompts) => {
            let mut token_batches = Vec::with_capacity(prompts.len());
            let mut counts = Vec::with_capacity(prompts.len());
            for prompt in prompts {
                let token_ids = runtime.tokenize(&prompt).token_ids;
                counts.push(token_ids.len());
                token_batches.push(token_ids);
            }
            Json(json!({
                "tokens": token_batches,
                "count": counts,
                "max_model_len": max_model_len,
            }))
            .into_response()
        }
    }
}

async fn detokenize<T, W>(
    State(service): State<HttpRouterService<T, W>>,
    Json(payload): Json<Value>,
) -> Response
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let tokens = match parse_detokenize_tokens(&payload) {
        Ok(tokens) => tokens,
        Err(message) => return bad_request_json(message),
    };

    let runtime = match service.runtime.lock() {
        Ok(runtime) => runtime,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {"message": "router runtime mutex poisoned"}
                })),
            )
                .into_response();
        }
    };

    match tokens {
        DetokenizeTokens::Single(token_ids) => match runtime.detokenize(&token_ids) {
            Ok(response) => Json(json!({ "text": response.text })).into_response(),
            Err(error) => bad_request_json(format!(
                "Error decoding tokens: {error}. Input tokens might be invalid for the model."
            )),
        },
        DetokenizeTokens::Batch(token_batches) => {
            let mut texts = Vec::with_capacity(token_batches.len());
            for token_ids in token_batches {
                match runtime.detokenize(&token_ids) {
                    Ok(response) => texts.push(response.text),
                    Err(error) => {
                        return bad_request_json(format!(
                            "Error decoding tokens: {error}. Input tokens might be invalid for the model."
                        ));
                    }
                }
            }
            Json(json!({ "text": texts })).into_response()
        }
    }
}

async fn rerank<T, W>(
    State(service): State<HttpRouterService<T, W>>,
    Json(payload): Json<Value>,
) -> Response
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let info = match service.model_info_snapshot() {
        Ok(info) => info,
        Err(message) => return internal_error_json(message),
    };
    let request = match parse_rerank_request(&payload, &info.served_model_name) {
        Ok(request) => request,
        Err(message) => return bad_request_json(message),
    };

    let mut results = {
        let runtime = match service.runtime.lock() {
            Ok(runtime) => runtime,
            Err(_) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({
                        "error": {"message": "router runtime mutex poisoned"}
                    })),
                )
                    .into_response();
            }
        };
        score_rerank_documents(&runtime, &request)
    };

    truncate_rerank_results(&request, &mut results);

    Json(rerank_results_to_json(&request, results)).into_response()
}

async fn score<T, W>(
    State(service): State<HttpRouterService<T, W>>,
    Json(payload): Json<Value>,
) -> Response
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let info = match service.model_info_snapshot() {
        Ok(info) => info,
        Err(message) => return internal_error_json(message),
    };
    let request = match parse_score_request(&payload, &info.served_model_name) {
        Ok(request) => request,
        Err(message) => return bad_request_json(message),
    };

    let json = {
        let runtime = match service.runtime.lock() {
            Ok(runtime) => runtime,
            Err(_) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({
                        "error": {"message": "router runtime mutex poisoned"}
                    })),
                )
                    .into_response();
            }
        };
        score_response_json(&runtime, &request)
    };

    Json(json).into_response()
}

async fn embeddings<T, W>(
    State(service): State<HttpRouterService<T, W>>,
    Json(payload): Json<Value>,
) -> Response
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let info = match service.model_info_snapshot() {
        Ok(info) => info,
        Err(message) => return internal_error_json(message),
    };
    let request = match parse_embedding_request(&payload, &info.served_model_name) {
        Ok(request) => request,
        Err(message) => return bad_request_json(message),
    };
    let runtime = match service.runtime.lock() {
        Ok(runtime) => runtime,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {"message": "router runtime mutex poisoned"}
                })),
            )
                .into_response();
        }
    };

    Json(embeddings_response_json(&runtime, &request)).into_response()
}

async fn classify<T, W>(
    State(service): State<HttpRouterService<T, W>>,
    Json(payload): Json<Value>,
) -> Response
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let info = match service.model_info_snapshot() {
        Ok(info) => info,
        Err(message) => return internal_error_json(message),
    };
    let request = match parse_classify_request(&payload, &info.served_model_name) {
        Ok(request) => request,
        Err(message) => return bad_request_json(message),
    };
    let runtime = match service.runtime.lock() {
        Ok(runtime) => runtime,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {"message": "router runtime mutex poisoned"}
                })),
            )
                .into_response();
        }
    };

    Json(classify_response_json(&runtime, &request)).into_response()
}

fn parse_tokenize_prompt(payload: &Value) -> Result<TokenizePrompt, String> {
    let has_prompt = payload.get("prompt").is_some();
    let has_messages = payload.get("messages").is_some();
    if has_prompt == has_messages {
        return Err("Exactly one of 'prompt' or 'messages' must be provided.".to_string());
    }
    if has_messages {
        return Err("messages tokenization requires chat template support".to_string());
    }

    match payload.get("prompt") {
        Some(Value::String(prompt)) => Ok(TokenizePrompt::Single(prompt.clone())),
        Some(Value::Array(prompts)) => {
            let mut out = Vec::with_capacity(prompts.len());
            for prompt in prompts {
                let Some(prompt) = prompt.as_str() else {
                    return Err(
                        "Invalid prompt type: expected string or list of strings.".to_string()
                    );
                };
                out.push(prompt.to_string());
            }
            Ok(TokenizePrompt::Batch(out))
        }
        _ => Err("Invalid prompt type: expected string or list of strings.".to_string()),
    }
}

fn parse_detokenize_tokens(payload: &Value) -> Result<DetokenizeTokens, String> {
    let tokens = payload
        .get("tokens")
        .ok_or_else(|| "missing `tokens` field".to_string())?;
    let Value::Array(items) = tokens else {
        return Err("Invalid tokens type: expected list of integers or list of lists.".to_string());
    };

    if items.is_empty() {
        return Ok(DetokenizeTokens::Single(Vec::new()));
    }
    if items[0].is_array() {
        let mut batches = Vec::with_capacity(items.len());
        for item in items {
            let Value::Array(token_list) = item else {
                return Err(
                    "Invalid tokens type: expected list of integers or list of lists.".to_string(),
                );
            };
            batches.push(parse_token_id_list(token_list)?);
        }
        return Ok(DetokenizeTokens::Batch(batches));
    }

    Ok(DetokenizeTokens::Single(parse_token_id_list(items)?))
}

fn parse_token_id_list(items: &[Value]) -> Result<Vec<u32>, String> {
    let mut token_ids = Vec::with_capacity(items.len());
    for item in items {
        let Some(token_id) = item.as_u64() else {
            return Err("Invalid input: 'tokens' must be a list of integers.".to_string());
        };
        let token_id = u32::try_from(token_id)
            .map_err(|_| "Invalid input: token id exceeds u32 range.".to_string())?;
        token_ids.push(token_id);
    }
    Ok(token_ids)
}

fn bad_request_json(message: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({ "error": { "message": message.into() } })),
    )
        .into_response()
}

fn internal_error_json(message: impl Into<String>) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({ "error": { "message": message.into() } })),
    )
        .into_response()
}

async fn generate<T, W>(
    State(service): State<HttpRouterService<T, W>>,
    Json(payload): Json<Value>,
) -> Response
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let request = match http_generate_payload_to_router_request(payload) {
        Ok(request) => request,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": { "message": error } })),
            )
                .into_response();
        }
    };
    let stream = request.stream();
    if request.disaggregated_params().is_some() && !service.allow_disaggregated_requests {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({
                "error": {
                    "message": "disaggregated HTTP generate requires a PD transfer-enabled runtime"
                }
            })),
        )
            .into_response();
    }

    let response = {
        let mut runtime = service
            .runtime
            .lock()
            .expect("HTTP router runtime lock should be held");
        match request {
            HttpGenerateRequest::Text(request) => {
                if service.max_transfer_polls == 0 {
                    runtime
                        .generate_text_stream(request)
                        .map(HttpGenerateResponse::Single)
                } else {
                    runtime
                        .generate_text_stream_with_transfer_polling(
                            request,
                            service.max_transfer_polls,
                        )
                        .map(HttpGenerateResponse::Single)
                }
            }
            HttpGenerateRequest::BatchText(requests) => {
                if stream {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({
                            "error": {
                                "message": "streaming batched /generate is not supported yet"
                            }
                        })),
                    )
                        .into_response();
                }

                if service.max_transfer_polls == 0 {
                    runtime
                        .generate_text_batch_stream(requests)
                        .map(HttpGenerateResponse::Batch)
                } else {
                    runtime
                        .generate_text_batch_stream_with_transfer_polling(
                            requests,
                            service.max_transfer_polls,
                        )
                        .map(HttpGenerateResponse::Batch)
                }
            }
            HttpGenerateRequest::Tokenized(request) => {
                if service.max_transfer_polls == 0 {
                    runtime
                        .generate_stream(request)
                        .map(HttpGenerateResponse::Single)
                } else {
                    runtime
                        .generate_stream_with_transfer_polling(request, service.max_transfer_polls)
                        .map(HttpGenerateResponse::Single)
                }
            }
            HttpGenerateRequest::BatchTokenized(requests) => {
                if stream {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({
                            "error": {
                                "message": "streaming batched /generate is not supported yet"
                            }
                        })),
                    )
                        .into_response();
                }

                if service.max_transfer_polls == 0 {
                    runtime
                        .generate_batch_stream(requests)
                        .map(HttpGenerateResponse::Batch)
                } else {
                    runtime
                        .generate_batch_stream_with_transfer_polling(
                            requests,
                            service.max_transfer_polls,
                        )
                        .map(HttpGenerateResponse::Batch)
                }
            }
        }
    };

    match response {
        Ok(HttpGenerateResponse::Single(mut responses)) => {
            if stream {
                return http_generate_stream_response_from_router_responses(responses);
            }
            let Some(response) = responses.pop() else {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": { "message": "generation produced no response" } })),
                )
                    .into_response();
            };
            match generate_complete_response_json(response) {
                Ok(body) => (StatusCode::OK, Json(body)).into_response(),
                Err(response) => match response.body {
                    RouterGenerateResponseBody::Chunk(_) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(
                            json!({ "error": { "message": "non-stream HTTP generate returned a stream chunk" } }),
                        ),
                    )
                        .into_response(),
                    RouterGenerateResponseBody::Error(error) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({ "error": { "message": error.message } })),
                    )
                        .into_response(),
                    RouterGenerateResponseBody::Complete(_) => unreachable!(),
                },
            }
        }
        Ok(HttpGenerateResponse::Batch(batch_responses)) => {
            let mut body = Vec::with_capacity(batch_responses.len());
            for mut responses in batch_responses {
                let Some(response) = responses.pop() else {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({ "error": { "message": "generation produced no response" } })),
                    )
                        .into_response();
                };
                match generate_complete_response_json(response) {
                    Ok(item) => body.push(item),
                    Err(response) => match response.body {
                        RouterGenerateResponseBody::Chunk(_) => {
                            return (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(json!({
                                    "error": {
                                        "message": "non-stream HTTP generate returned a stream chunk"
                                    }
                                })),
                            )
                                .into_response();
                        }
                        RouterGenerateResponseBody::Error(error) => {
                            return (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(json!({ "error": { "message": error.message } })),
                            )
                                .into_response();
                        }
                        RouterGenerateResponseBody::Complete(_) => unreachable!(),
                    },
                }
            }
            (StatusCode::OK, Json(Value::Array(body))).into_response()
        }
        Err(error) => router_runtime_error_json(error),
    }
}

enum HttpGenerateResponse {
    Single(Vec<RouterGenerateResponse>),
    Batch(Vec<Vec<RouterGenerateResponse>>),
}

fn generate_complete_response_json(
    response: RouterGenerateResponse,
) -> Result<Value, RouterGenerateResponse> {
    match response.body {
        RouterGenerateResponseBody::Complete(complete) => Ok(json!({
            "request_id": response.request_id,
            "text": complete.text,
            "output_ids": complete.output_ids,
            "finish_reason": complete.finish_reason,
            "usage": {
                "prompt_tokens": complete.prompt_tokens,
                "completion_tokens": complete.completion_tokens,
                "cached_tokens": complete.cached_tokens,
            }
        })),
        body => Err(RouterGenerateResponse {
            request_id: response.request_id,
            body,
        }),
    }
}

fn http_generate_stream_response_from_router_responses(
    responses: Vec<RouterGenerateResponse>,
) -> Response {
    let mut body = String::new();
    for response in responses {
        let json = match response.body {
            RouterGenerateResponseBody::Chunk(chunk) => json!({
                "request_id": response.request_id,
                "text": chunk.text,
                "output_ids": chunk.token_ids,
                "usage": {
                    "prompt_tokens": chunk.prompt_tokens,
                    "completion_tokens": chunk.completion_tokens,
                    "cached_tokens": chunk.cached_tokens,
                }
            }),
            RouterGenerateResponseBody::Complete(complete) => json!({
                "request_id": response.request_id,
                "text": complete.text,
                "output_ids": complete.output_ids,
                "finish_reason": complete.finish_reason,
                "usage": {
                    "prompt_tokens": complete.prompt_tokens,
                    "completion_tokens": complete.completion_tokens,
                    "cached_tokens": complete.cached_tokens,
                }
            }),
            RouterGenerateResponseBody::Error(error) => json!({
                "error": {
                    "message": error.message,
                }
            }),
        };
        let Ok(json) = serde_json::to_string(&json) else {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": { "message": "serialize SGLang generate stream JSON" } })),
            )
                .into_response();
        };
        body.push_str("data: ");
        body.push_str(&json);
        body.push_str("\n\n");
    }
    body.push_str("data: [DONE]\n\n");

    let mut response = Response::new(Body::from(body));
    response.headers_mut().insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("text/event-stream"),
    );
    response
}

fn router_runtime_error_json(error: RouterRuntimeError) -> Response {
    let status = match &error {
        RouterRuntimeError::Protocol(protocol) => {
            router_status_code_to_http_status(protocol.status_code())
        }
        RouterRuntimeError::Runtime(_) => StatusCode::INTERNAL_SERVER_ERROR,
    };

    (
        status,
        Json(json!({ "error": { "message": error.to_string() } })),
    )
        .into_response()
}

fn router_protocol_error_json(error: RouterProtocolError) -> Response {
    (
        router_status_code_to_http_status(error.status_code()),
        Json(json!({ "error": { "message": error.to_string() } })),
    )
        .into_response()
}

fn router_status_code_to_http_status(status_code: RouterStatusCode) -> StatusCode {
    match status_code {
        RouterStatusCode::InvalidArgument => StatusCode::BAD_REQUEST,
        RouterStatusCode::ResourceExhausted => StatusCode::TOO_MANY_REQUESTS,
        RouterStatusCode::FailedPrecondition => StatusCode::PRECONDITION_FAILED,
    }
}

async fn chat_completions<T, W>(
    State(service): State<HttpRouterService<T, W>>,
    Json(payload): Json<Value>,
) -> Response
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let model = match service.model_info_snapshot() {
        Ok(info) => info.served_model_name,
        Err(message) => return internal_error_json(message),
    };
    let request = match http_chat_payload_to_router_request(payload, &model) {
        Ok(request) => request,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": { "message": error } })),
            )
                .into_response();
        }
    };
    let stream = request.stream();
    if request.disaggregated_params().is_some() && !service.allow_disaggregated_requests {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({
                "error": {
                    "message": "disaggregated HTTP chat completions require a PD transfer-enabled runtime"
                }
            })),
        )
            .into_response();
    }

    let response = {
        let mut runtime = service
            .runtime
            .lock()
            .expect("HTTP router runtime lock should be held");
        match request {
            HttpChatRequest::Single(request) => {
                if service.max_transfer_polls == 0 {
                    runtime
                        .generate_text_stream(request)
                        .map(HttpChatResponse::Single)
                } else {
                    runtime
                        .generate_text_stream_with_transfer_polling(
                            request,
                            service.max_transfer_polls,
                        )
                        .map(HttpChatResponse::Single)
                }
            }
            HttpChatRequest::Batch(requests) => {
                if stream {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({
                            "error": {
                                "message": "streaming batched /v1/chat/completions is not supported yet"
                            }
                        })),
                    )
                        .into_response();
                }

                if service.max_transfer_polls == 0 {
                    runtime
                        .generate_text_batch_stream(requests)
                        .map(HttpChatResponse::Batch)
                } else {
                    runtime
                        .generate_text_batch_stream_with_transfer_polling(
                            requests,
                            service.max_transfer_polls,
                        )
                        .map(HttpChatResponse::Batch)
                }
            }
        }
    };

    match response {
        Ok(HttpChatResponse::Single(mut responses)) => {
            if stream {
                return http_chat_stream_response_from_router_responses(responses, &model);
            }
            let Some(response) = responses.pop() else {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": { "message": "generation produced no response" } })),
                )
                    .into_response();
            };
            match response.body {
                RouterGenerateResponseBody::Complete(complete) => (
                    StatusCode::OK,
                    Json(json!({
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
                    })),
                )
                    .into_response(),
                RouterGenerateResponseBody::Chunk(_) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({
                        "error": {
                            "message": "non-stream HTTP chat completion returned a stream chunk"
                        }
                    })),
                )
                    .into_response(),
                RouterGenerateResponseBody::Error(error) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": { "message": error.message } })),
                )
                    .into_response(),
            }
        }
        Ok(HttpChatResponse::Batch(batch_responses)) => {
            http_chat_batch_response_from_router_responses(batch_responses, &model)
        }
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": { "message": error.to_string() } })),
        )
            .into_response(),
    }
}

pub(crate) enum HttpChatRequest {
    Single(RouterTextGenerateRequest),
    Batch(Vec<RouterTextGenerateRequest>),
}

impl HttpChatRequest {
    pub(crate) fn stream(&self) -> bool {
        match self {
            Self::Single(request) => request.stream,
            Self::Batch(requests) => requests
                .first()
                .map(|request| request.stream)
                .unwrap_or(false),
        }
    }

    pub(crate) fn disaggregated_params(&self) -> Option<&RouterDisaggregatedParams> {
        match self {
            Self::Single(request) => request.disaggregated_params.as_ref(),
            Self::Batch(requests) => requests
                .iter()
                .find_map(|request| request.disaggregated_params.as_ref()),
        }
    }
}

enum HttpChatResponse {
    Single(Vec<RouterGenerateResponse>),
    Batch(Vec<Vec<RouterGenerateResponse>>),
}

#[derive(Default)]
struct ChatUsage {
    prompt_tokens: i32,
    completion_tokens: i32,
    cached_tokens: i32,
}

fn http_chat_batch_response_from_router_responses(
    batch_responses: Vec<Vec<RouterGenerateResponse>>,
    model: &str,
) -> Response {
    let mut choices = Vec::with_capacity(batch_responses.len());
    let mut usage = ChatUsage::default();
    let mut response_id = None;

    for (batch_index, mut responses) in batch_responses.into_iter().enumerate() {
        let Some(response) = responses.pop() else {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": { "message": "generation produced no response" } })),
            )
                .into_response();
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
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({
                        "error": {
                            "message": "non-stream HTTP chat completion returned a stream chunk"
                        }
                    })),
                )
                    .into_response();
            }
            RouterGenerateResponseBody::Error(error) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": { "message": error.message } })),
                )
                    .into_response();
            }
        }
    }

    (
        StatusCode::OK,
        Json(json!({
            "id": format!("chatcmpl-{}", response_id.unwrap_or_default()),
            "object": "chat.completion",
            "model": model,
            "choices": choices,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cached_tokens": usage.cached_tokens,
            }
        })),
    )
        .into_response()
}

fn http_chat_stream_response_from_router_responses(
    responses: Vec<RouterGenerateResponse>,
    model: &str,
) -> Response {
    let mut body = String::new();
    for response in responses {
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
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": { "message": error.message } })),
                )
                    .into_response();
            }
        };
        let Ok(json) = serde_json::to_string(&json) else {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": { "message": "serialize OpenAI chat stream JSON" } })),
            )
                .into_response();
        };
        body.push_str("data: ");
        body.push_str(&json);
        body.push_str("\n\n");
    }
    body.push_str("data: [DONE]\n\n");

    let mut response = Response::new(Body::from(body));
    response.headers_mut().insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("text/event-stream"),
    );
    response
}

async fn completions<T, W>(
    State(service): State<HttpRouterService<T, W>>,
    Json(payload): Json<Value>,
) -> Response
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let model = match service.model_info_snapshot() {
        Ok(info) => info.served_model_name,
        Err(message) => return internal_error_json(message),
    };
    let request = match http_completion_payload_to_router_request(payload, &model) {
        Ok(request) => request,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": { "message": error } })),
            )
                .into_response();
        }
    };
    let stream = request.stream();
    if request.disaggregated_params().is_some() && !service.allow_disaggregated_requests {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({
                "error": {
                    "message": "disaggregated HTTP completions require a PD transfer-enabled runtime"
                }
            })),
        )
            .into_response();
    }

    let response = {
        let mut runtime = service
            .runtime
            .lock()
            .expect("HTTP router runtime lock should be held");
        match request {
            HttpCompletionRequest::Single(request) => {
                if service.max_transfer_polls == 0 {
                    runtime
                        .generate_text_stream(request)
                        .map(HttpCompletionResponse::Single)
                } else {
                    runtime
                        .generate_text_stream_with_transfer_polling(
                            request,
                            service.max_transfer_polls,
                        )
                        .map(HttpCompletionResponse::Single)
                }
            }
            HttpCompletionRequest::Batch(requests) => {
                if stream {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({
                            "error": {
                                "message": "streaming batched /v1/completions is not supported yet"
                            }
                        })),
                    )
                        .into_response();
                }

                if service.max_transfer_polls == 0 {
                    runtime
                        .generate_text_batch_stream(requests)
                        .map(HttpCompletionResponse::Batch)
                } else {
                    runtime
                        .generate_text_batch_stream_with_transfer_polling(
                            requests,
                            service.max_transfer_polls,
                        )
                        .map(HttpCompletionResponse::Batch)
                }
            }
        }
    };

    match response {
        Ok(HttpCompletionResponse::Single(responses)) if stream => {
            http_completion_stream_response_from_router_responses(responses, &model)
        }
        Ok(HttpCompletionResponse::Single(responses)) => {
            http_completion_response_from_router_responses(responses, &model)
        }
        Ok(HttpCompletionResponse::Batch(batch_responses)) => {
            http_completion_batch_response_from_router_responses(batch_responses, &model)
        }
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": { "message": error.to_string() } })),
        )
            .into_response(),
    }
}

pub(crate) enum HttpCompletionRequest {
    Single(RouterTextGenerateRequest),
    Batch(Vec<RouterTextGenerateRequest>),
}

impl HttpCompletionRequest {
    pub(crate) fn stream(&self) -> bool {
        match self {
            Self::Single(request) => request.stream,
            Self::Batch(requests) => requests
                .first()
                .map(|request| request.stream)
                .unwrap_or(false),
        }
    }

    pub(crate) fn disaggregated_params(&self) -> Option<&RouterDisaggregatedParams> {
        match self {
            Self::Single(request) => request.disaggregated_params.as_ref(),
            Self::Batch(requests) => requests
                .iter()
                .find_map(|request| request.disaggregated_params.as_ref()),
        }
    }
}

enum HttpCompletionResponse {
    Single(Vec<RouterGenerateResponse>),
    Batch(Vec<Vec<RouterGenerateResponse>>),
}

fn http_completion_response_from_router_responses(
    mut responses: Vec<RouterGenerateResponse>,
    model: &str,
) -> Response {
    let Some(response) = responses.pop() else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": { "message": "generation produced no response" } })),
        )
            .into_response();
    };

    match response.body {
        RouterGenerateResponseBody::Complete(complete) => (
            StatusCode::OK,
            Json(json!({
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
            })),
        )
            .into_response(),
        RouterGenerateResponseBody::Chunk(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": "non-stream HTTP completion returned a stream chunk"
                }
            })),
        )
            .into_response(),
        RouterGenerateResponseBody::Error(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": { "message": error.message } })),
        )
            .into_response(),
    }
}

#[derive(Default)]
struct CompletionUsage {
    prompt_tokens: i32,
    completion_tokens: i32,
    cached_tokens: i32,
}

impl CompletionUsage {
    fn add_complete(&mut self, complete: &RouterGenerateComplete) {
        self.prompt_tokens += complete.prompt_tokens;
        self.completion_tokens += complete.completion_tokens;
        self.cached_tokens += complete.cached_tokens;
    }
}

fn http_completion_batch_response_from_router_responses(
    batch_responses: Vec<Vec<RouterGenerateResponse>>,
    model: &str,
) -> Response {
    let mut choices = Vec::with_capacity(batch_responses.len());
    let mut usage = CompletionUsage::default();
    let mut response_id = None;

    for (batch_index, mut responses) in batch_responses.into_iter().enumerate() {
        let Some(response) = responses.pop() else {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": { "message": "generation produced no response" } })),
            )
                .into_response();
        };
        if response_id.is_none() {
            response_id = Some(response.request_id.clone());
        }
        match response.body {
            RouterGenerateResponseBody::Complete(complete) => {
                usage.add_complete(&complete);
                choices.push(json!({
                    "index": batch_index,
                    "text": complete.text,
                    "logprobs": Value::Null,
                    "finish_reason": complete.finish_reason,
                }));
            }
            RouterGenerateResponseBody::Chunk(_) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({
                        "error": {
                            "message": "non-stream HTTP completion returned a stream chunk"
                        }
                    })),
                )
                    .into_response();
            }
            RouterGenerateResponseBody::Error(error) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": { "message": error.message } })),
                )
                    .into_response();
            }
        }
    }

    (
        StatusCode::OK,
        Json(json!({
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
        })),
    )
        .into_response()
}

fn http_completion_stream_response_from_router_responses(
    responses: Vec<RouterGenerateResponse>,
    model: &str,
) -> Response {
    let mut body = String::new();
    for response in responses {
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
            RouterGenerateResponseBody::Error(error) => json!({
                "error": {
                    "message": error.message,
                }
            }),
        };
        let Ok(json) = serde_json::to_string(&json) else {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": { "message": "serialize OpenAI completion stream JSON" } })),
            )
                .into_response();
        };
        body.push_str("data: ");
        body.push_str(&json);
        body.push_str("\n\n");
    }
    body.push_str("data: [DONE]\n\n");

    let mut response = Response::new(Body::from(body));
    response.headers_mut().insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("text/event-stream"),
    );
    response
}

enum HttpGenerateRequest {
    Text(RouterTextGenerateRequest),
    BatchText(Vec<RouterTextGenerateRequest>),
    Tokenized(RouterGenerateRequest),
    BatchTokenized(Vec<RouterGenerateRequest>),
}

impl HttpGenerateRequest {
    fn stream(&self) -> bool {
        match self {
            Self::Text(request) => request.stream,
            Self::BatchText(requests) => requests
                .first()
                .map(|request| request.stream)
                .unwrap_or(false),
            Self::Tokenized(request) => request.stream,
            Self::BatchTokenized(requests) => requests
                .first()
                .map(|request| request.stream)
                .unwrap_or(false),
        }
    }

    fn disaggregated_params(&self) -> Option<&RouterDisaggregatedParams> {
        match self {
            Self::Text(request) => request.disaggregated_params.as_ref(),
            Self::BatchText(requests) => requests
                .iter()
                .find_map(|request| request.disaggregated_params.as_ref()),
            Self::Tokenized(request) => request.disaggregated_params.as_ref(),
            Self::BatchTokenized(requests) => requests
                .iter()
                .find_map(|request| request.disaggregated_params.as_ref()),
        }
    }
}

fn http_generate_payload_to_router_request(payload: Value) -> Result<HttpGenerateRequest, String> {
    if payload.get("input_ids").is_some() {
        return match http_generate_payload_to_router_token_requests(payload)? {
            HttpTokenGenerateRequests::Single(request) => {
                Ok(HttpGenerateRequest::Tokenized(request))
            }
            HttpTokenGenerateRequests::Batch(requests) => {
                Ok(HttpGenerateRequest::BatchTokenized(requests))
            }
        };
    }
    match http_generate_payload_to_router_text_requests(payload)? {
        HttpTextGenerateRequests::Single(request) => Ok(HttpGenerateRequest::Text(request)),
        HttpTextGenerateRequests::Batch(requests) => Ok(HttpGenerateRequest::BatchText(requests)),
    }
}

enum HttpTextGenerateRequests {
    Single(RouterTextGenerateRequest),
    Batch(Vec<RouterTextGenerateRequest>),
}

fn http_generate_payload_to_router_text_requests(
    payload: Value,
) -> Result<HttpTextGenerateRequests, String> {
    let text_value = payload
        .get("text")
        .ok_or_else(|| "missing text".to_string())?;
    let sampling_params = payload
        .get("sampling_params")
        .map(json_to_sampling_params)
        .transpose()?;
    let stream = optional_bool(&payload, "stream")?.unwrap_or(false);

    if let Some(text_values) = text_value.as_array() {
        let batch_size = text_values.len();
        let texts = text_values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(ToString::to_string)
                    .ok_or_else(|| "text entries must be strings".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let request_ids = optional_string_values(&payload, "request_id", batch_size)?;
        let data_parallel_ranks = optional_i32_values(&payload, "data_parallel_rank", batch_size)?;
        let disaggregated_params = json_to_disaggregated_params_values(&payload, batch_size)?;
        let mut requests = Vec::with_capacity(batch_size);

        for index in 0..batch_size {
            requests.push(RouterTextGenerateRequest {
                request_id: request_ids[index].clone(),
                text: texts[index].clone(),
                sampling_params: sampling_params.clone(),
                disaggregated_params: disaggregated_params[index].clone(),
                stream,
                data_parallel_rank: data_parallel_ranks[index],
                ..Default::default()
            });
        }

        return Ok(HttpTextGenerateRequests::Batch(requests));
    }

    let text = text_value
        .as_str()
        .ok_or_else(|| "text must be a string or array of strings".to_string())?
        .to_string();
    let request_id = payload
        .get("request_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let disaggregated_params = json_to_disaggregated_params(&payload)?;
    let data_parallel_rank = optional_i32(&payload, "data_parallel_rank")?.unwrap_or_default();

    Ok(HttpTextGenerateRequests::Single(
        RouterTextGenerateRequest {
            request_id,
            text,
            sampling_params,
            disaggregated_params,
            stream,
            data_parallel_rank,
            ..Default::default()
        },
    ))
}

enum HttpTokenGenerateRequests {
    Single(RouterGenerateRequest),
    Batch(Vec<RouterGenerateRequest>),
}

fn http_generate_payload_to_router_token_requests(
    payload: Value,
) -> Result<HttpTokenGenerateRequests, String> {
    let input_ids_value = payload
        .get("input_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| "input_ids must be an array".to_string())?;
    let sampling_params = payload
        .get("sampling_params")
        .map(json_to_sampling_params)
        .transpose()?;
    let stream = optional_bool(&payload, "stream")?.unwrap_or(false);

    if input_ids_value
        .first()
        .is_some_and(serde_json::Value::is_array)
    {
        let input_batches = input_ids_value
            .iter()
            .map(token_id_array)
            .collect::<Result<Vec<_>, _>>()?;
        let batch_size = input_batches.len();
        let request_ids = optional_string_values(&payload, "request_id", batch_size)?;
        let original_texts = optional_string_values(&payload, "original_text", batch_size)?;
        let data_parallel_ranks = optional_i32_values(&payload, "data_parallel_rank", batch_size)?;
        let disaggregated_params = json_to_disaggregated_params_values(&payload, batch_size)?;
        let mut requests = Vec::with_capacity(batch_size);

        for index in 0..batch_size {
            requests.push(RouterGenerateRequest {
                request_id: request_ids[index].clone(),
                tokenized: Some(RouterTokenizedInput {
                    original_text: original_texts[index].clone(),
                    input_ids: input_batches[index].clone(),
                }),
                sampling_params: sampling_params.clone(),
                disaggregated_params: disaggregated_params[index].clone(),
                stream,
                data_parallel_rank: data_parallel_ranks[index],
                ..Default::default()
            });
        }

        return Ok(HttpTokenGenerateRequests::Batch(requests));
    }

    let input_ids = token_id_array(&Value::Array(input_ids_value.clone()))?;
    let request_id = payload
        .get("request_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let original_text = payload
        .get("original_text")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let disaggregated_params = json_to_disaggregated_params(&payload)?;
    let data_parallel_rank = optional_i32(&payload, "data_parallel_rank")?.unwrap_or_default();

    Ok(HttpTokenGenerateRequests::Single(RouterGenerateRequest {
        request_id,
        tokenized: Some(RouterTokenizedInput {
            original_text,
            input_ids,
        }),
        sampling_params,
        disaggregated_params,
        stream,
        data_parallel_rank,
        ..Default::default()
    }))
}

fn token_id_array(value: &Value) -> Result<Vec<u32>, String> {
    value
        .as_array()
        .ok_or_else(|| "input_ids must be an array".to_string())?
        .iter()
        .map(|value| {
            let Some(raw) = value.as_u64() else {
                return Err("input_ids entries must be unsigned integers".to_string());
            };
            u32::try_from(raw).map_err(|_| "input_ids entry is out of u32 range".to_string())
        })
        .collect()
}

fn optional_string_values(
    payload: &Value,
    field: &'static str,
    batch_size: usize,
) -> Result<Vec<String>, String> {
    let Some(value) = payload.get(field) else {
        return Ok(vec![String::new(); batch_size]);
    };

    if let Some(text) = value.as_str() {
        return Ok(vec![text.to_string(); batch_size]);
    }

    let values = value
        .as_array()
        .ok_or_else(|| format!("{field} must be a string or array of strings"))?;
    if values.len() != batch_size {
        return Err(format!(
            "{field} length {} does not match batch size {batch_size}",
            values.len()
        ));
    }

    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(ToString::to_string)
                .ok_or_else(|| format!("{field} entries must be strings"))
        })
        .collect()
}

fn optional_i32_values(
    payload: &Value,
    field: &'static str,
    batch_size: usize,
) -> Result<Vec<i32>, String> {
    let Some(value) = payload.get(field) else {
        return Ok(vec![0; batch_size]);
    };

    if let Some(raw) = value.as_i64() {
        let value = i32::try_from(raw).map_err(|_| format!("{field} is out of i32 range"))?;
        return Ok(vec![value; batch_size]);
    }

    let values = value
        .as_array()
        .ok_or_else(|| format!("{field} must be an integer or array of integers"))?;
    if values.len() != batch_size {
        return Err(format!(
            "{field} length {} does not match batch size {batch_size}",
            values.len()
        ));
    }

    values
        .iter()
        .map(|value| {
            let raw = value
                .as_i64()
                .ok_or_else(|| format!("{field} entries must be integers"))?;
            i32::try_from(raw).map_err(|_| format!("{field} entry is out of i32 range"))
        })
        .collect()
}

pub(crate) fn http_completion_payload_to_router_request(
    payload: Value,
    served_model_name: &str,
) -> Result<HttpCompletionRequest, String> {
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing model".to_string())?;
    if model != served_model_name {
        return Err(format!(
            "model {model} is not served by this worker ({served_model_name})"
        ));
    }

    let mut sampling_params = payload
        .get("sampling_params")
        .map(json_to_sampling_params)
        .transpose()?
        .unwrap_or_default();
    if let Some(max_tokens) = optional_i32(&payload, "max_tokens")? {
        sampling_params.max_new_tokens = Some(max_tokens);
    }
    let stream = optional_bool(&payload, "stream")?.unwrap_or(false);

    match completion_prompt_to_texts(&payload)? {
        CompletionPrompts::Single(text) => {
            Ok(HttpCompletionRequest::Single(RouterTextGenerateRequest {
                request_id: payload
                    .get("request_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                text,
                sampling_params: Some(sampling_params),
                disaggregated_params: json_to_disaggregated_params(&payload)?,
                stream,
                data_parallel_rank: optional_i32(&payload, "data_parallel_rank")?
                    .unwrap_or_default(),
                ..Default::default()
            }))
        }
        CompletionPrompts::Batch(texts) => {
            let batch_size = texts.len();
            if batch_size == 0 {
                return Err("prompt array must not be empty".to_string());
            }
            let request_ids = optional_string_values(&payload, "request_id", batch_size)?;
            let data_parallel_ranks =
                optional_i32_values(&payload, "data_parallel_rank", batch_size)?;
            let disaggregated_params = json_to_disaggregated_params_values(&payload, batch_size)?;
            let mut requests = Vec::with_capacity(batch_size);

            for index in 0..batch_size {
                requests.push(RouterTextGenerateRequest {
                    request_id: request_ids[index].clone(),
                    text: texts[index].clone(),
                    sampling_params: Some(sampling_params.clone()),
                    disaggregated_params: disaggregated_params[index].clone(),
                    stream,
                    data_parallel_rank: data_parallel_ranks[index],
                    ..Default::default()
                });
            }

            Ok(HttpCompletionRequest::Batch(requests))
        }
    }
}

pub(crate) fn http_chat_payload_to_router_request(
    payload: Value,
    served_model_name: &str,
) -> Result<HttpChatRequest, String> {
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing model".to_string())?;
    if model != served_model_name {
        return Err(format!(
            "model {model} is not served by this worker ({served_model_name})"
        ));
    }

    let mut sampling_params = payload
        .get("sampling_params")
        .map(json_to_sampling_params)
        .transpose()?
        .unwrap_or_default();
    if let Some(max_tokens) = optional_i32(&payload, "max_tokens")? {
        sampling_params.max_new_tokens = Some(max_tokens);
    }
    let stream = optional_bool(&payload, "stream")?.unwrap_or(false);
    let n = optional_i32(&payload, "n")?.unwrap_or(1);
    if n < 1 {
        return Err("n must be at least 1".to_string());
    }
    let text = chat_messages_to_prompt_text(&payload)?;

    if n == 1 {
        return Ok(HttpChatRequest::Single(RouterTextGenerateRequest {
            request_id: payload
                .get("request_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            text,
            sampling_params: Some(sampling_params),
            disaggregated_params: json_to_disaggregated_params(&payload)?,
            stream,
            data_parallel_rank: optional_i32(&payload, "data_parallel_rank")?.unwrap_or_default(),
            ..Default::default()
        }));
    }

    let batch_size = usize::try_from(n).map_err(|_| "n is out of range".to_string())?;
    let request_ids = optional_string_values(&payload, "request_id", batch_size)?;
    let data_parallel_ranks = optional_i32_values(&payload, "data_parallel_rank", batch_size)?;
    let disaggregated_params = json_to_disaggregated_params_values(&payload, batch_size)?;
    let mut requests = Vec::with_capacity(batch_size);

    for index in 0..batch_size {
        requests.push(RouterTextGenerateRequest {
            request_id: request_ids[index].clone(),
            text: text.clone(),
            sampling_params: Some(sampling_params.clone()),
            disaggregated_params: disaggregated_params[index].clone(),
            stream,
            data_parallel_rank: data_parallel_ranks[index],
            ..Default::default()
        });
    }

    Ok(HttpChatRequest::Batch(requests))
}

enum CompletionPrompts {
    Single(String),
    Batch(Vec<String>),
}

fn completion_prompt_to_texts(payload: &Value) -> Result<CompletionPrompts, String> {
    let prompt = payload
        .get("prompt")
        .ok_or_else(|| "missing prompt".to_string())?;
    if let Some(text) = prompt.as_str() {
        return Ok(CompletionPrompts::Single(text.to_string()));
    }
    if let Some(prompts) = prompt.as_array() {
        return prompts
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(ToString::to_string)
                    .ok_or_else(|| "prompt array entries must be strings".to_string())
            })
            .collect::<Result<Vec<_>, _>>()
            .map(CompletionPrompts::Batch);
    }
    Err("prompt must be a string or array of strings".to_string())
}

fn chat_messages_to_prompt_text(payload: &Value) -> Result<String, String> {
    let messages = payload
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| "messages must be an array".to_string())?;
    if messages.is_empty() {
        return Err("messages must not be empty".to_string());
    }

    messages
        .iter()
        .map(chat_message_content_text)
        .collect::<Result<Vec<_>, _>>()
        .map(|contents| contents.join("\n"))
}

fn chat_message_content_text(message: &Value) -> Result<String, String> {
    let content = message
        .get("content")
        .ok_or_else(|| "chat message content is required".to_string())?;
    if let Some(text) = content.as_str() {
        return Ok(text.to_string());
    }
    let Some(parts) = content.as_array() else {
        return Err("chat message content must be a string or array".to_string());
    };

    let mut text = String::new();
    for part in parts {
        if part.get("type").and_then(Value::as_str) == Some("text") {
            let part_text = part
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| "chat text content part requires text".to_string())?;
            text.push_str(part_text);
        }
    }
    Ok(text)
}

fn json_to_disaggregated_params(
    payload: &Value,
) -> Result<Option<RouterDisaggregatedParams>, String> {
    if payload.get("bootstrap_host").is_some()
        || payload.get("bootstrap_port").is_some()
        || payload.get("bootstrap_room").is_some()
    {
        return Ok(Some(RouterDisaggregatedParams {
            bootstrap_host: required_string(payload, "bootstrap_host")?.to_string(),
            bootstrap_port: required_u16(payload, "bootstrap_port")?,
            bootstrap_room: required_bootstrap_room(payload, "bootstrap_room")?,
        }));
    }

    let Some(value) = payload.get("disaggregated_params") else {
        return Ok(None);
    };
    if !value.is_object() {
        return Err("disaggregated_params must be an object".to_string());
    }

    Ok(Some(RouterDisaggregatedParams {
        bootstrap_host: required_string(value, "bootstrap_host")?.to_string(),
        bootstrap_port: required_u16(value, "bootstrap_port")?,
        bootstrap_room: required_bootstrap_room(value, "bootstrap_room")?,
    }))
}

fn json_to_disaggregated_params_values(
    payload: &Value,
    batch_size: usize,
) -> Result<Vec<Option<RouterDisaggregatedParams>>, String> {
    if payload.get("bootstrap_host").is_some()
        || payload.get("bootstrap_port").is_some()
        || payload.get("bootstrap_room").is_some()
    {
        let hosts = required_string_values(payload, "bootstrap_host", batch_size)?;
        let ports = required_u16_values(payload, "bootstrap_port", batch_size)?;
        let rooms = required_bootstrap_room_values(payload, "bootstrap_room", batch_size)?;
        return Ok((0..batch_size)
            .map(|index| {
                Some(RouterDisaggregatedParams {
                    bootstrap_host: hosts[index].clone(),
                    bootstrap_port: ports[index],
                    bootstrap_room: rooms[index],
                })
            })
            .collect());
    }

    if payload.get("disaggregated_params").is_some() {
        return Ok(vec![json_to_disaggregated_params(payload)?; batch_size]);
    }

    Ok(vec![None; batch_size])
}

fn json_to_sampling_params(value: &Value) -> Result<RouterSamplingParams, String> {
    if !value.is_object() {
        return Err("sampling_params must be an object".to_string());
    }

    Ok(RouterSamplingParams {
        max_new_tokens: optional_i32(value, "max_new_tokens")?,
        temperature: optional_f32(value, "temperature")?,
        top_p: optional_f32(value, "top_p")?,
        top_k: optional_i32(value, "top_k")?,
        min_p: optional_f32(value, "min_p")?,
        frequency_penalty: optional_f32(value, "frequency_penalty")?,
        presence_penalty: optional_f32(value, "presence_penalty")?,
        repetition_penalty: optional_f32(value, "repetition_penalty")?,
        stop_token_id: optional_i32(value, "stop_token_id")?,
        stop_token_ids: optional_i32_array(value, "stop_token_ids")?.unwrap_or_default(),
        ignore_eos: optional_bool(value, "ignore_eos")?,
        n: optional_i32(value, "n")?,
        best_of: optional_i32(value, "best_of")?,
    })
}

fn required_string<'a>(value: &'a Value, field: &'static str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{field} must be a string"))
}

fn required_string_values(
    value: &Value,
    field: &'static str,
    batch_size: usize,
) -> Result<Vec<String>, String> {
    let Some(value) = value.get(field) else {
        return Err(format!("{field} is required"));
    };
    if let Some(text) = value.as_str() {
        return Ok(vec![text.to_string(); batch_size]);
    }
    let values = value
        .as_array()
        .ok_or_else(|| format!("{field} must be a string or array of strings"))?;
    if values.len() != batch_size {
        return Err(format!(
            "{field} length {} does not match batch size {batch_size}",
            values.len()
        ));
    }
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(ToString::to_string)
                .ok_or_else(|| format!("{field} entries must be strings"))
        })
        .collect()
}

fn required_bootstrap_room(value: &Value, field: &'static str) -> Result<BootstrapRoom, String> {
    let value = value
        .get(field)
        .ok_or_else(|| format!("{field} must be an unsigned integer"))?;
    required_bootstrap_room_value(value, field)
}

fn required_bootstrap_room_value(
    value: &Value,
    field: &'static str,
) -> Result<BootstrapRoom, String> {
    let raw = value
        .as_u64()
        .ok_or_else(|| format!("{field} must be an unsigned integer"))?;
    if raw > i64::MAX as u64 {
        return Err(format!("{field} must fit in signed 63-bit range"));
    }

    Ok(raw)
}

fn required_bootstrap_room_values(
    value: &Value,
    field: &'static str,
    batch_size: usize,
) -> Result<Vec<BootstrapRoom>, String> {
    let Some(value) = value.get(field) else {
        return Err(format!("{field} is required"));
    };
    if value.as_u64().is_some() {
        return Ok(vec![
            required_bootstrap_room_value(value, field)?;
            batch_size
        ]);
    }

    let values = value
        .as_array()
        .ok_or_else(|| format!("{field} must be an unsigned integer or array"))?;
    if values.len() != batch_size {
        return Err(format!(
            "{field} length {} does not match batch size {batch_size}",
            values.len()
        ));
    }
    values
        .iter()
        .map(|value| required_bootstrap_room_value(value, field))
        .collect()
}

fn required_u16(value: &Value, field: &'static str) -> Result<u16, String> {
    let value = value
        .get(field)
        .ok_or_else(|| format!("{field} must be an unsigned integer"))?;
    required_u16_value(value, field)
}

fn required_u16_value(value: &Value, field: &'static str) -> Result<u16, String> {
    let raw = value
        .as_u64()
        .ok_or_else(|| format!("{field} must be an unsigned integer"))?;
    u16::try_from(raw).map_err(|_| format!("{field} is too large for u16"))
}

fn required_u16_values(
    value: &Value,
    field: &'static str,
    batch_size: usize,
) -> Result<Vec<u16>, String> {
    let Some(value) = value.get(field) else {
        return Err(format!("{field} is required"));
    };
    if value.as_u64().is_some() {
        return Ok(vec![required_u16_value(value, field)?; batch_size]);
    }

    let values = value
        .as_array()
        .ok_or_else(|| format!("{field} must be an unsigned integer or array"))?;
    if values.len() != batch_size {
        return Err(format!(
            "{field} length {} does not match batch size {batch_size}",
            values.len()
        ));
    }
    values
        .iter()
        .map(|value| required_u16_value(value, field))
        .collect()
}

fn optional_i32(value: &Value, field: &'static str) -> Result<Option<i32>, String> {
    let Some(raw) = value.get(field) else {
        return Ok(None);
    };
    let raw = raw
        .as_i64()
        .ok_or_else(|| format!("{field} must be an integer"))?;
    i32::try_from(raw)
        .map(Some)
        .map_err(|_| format!("{field} is too large for i32"))
}

fn optional_f32(value: &Value, field: &'static str) -> Result<Option<f32>, String> {
    let Some(raw) = value.get(field) else {
        return Ok(None);
    };
    raw.as_f64()
        .map(|value| Some(value as f32))
        .ok_or_else(|| format!("{field} must be a number"))
}

fn optional_bool(value: &Value, field: &'static str) -> Result<Option<bool>, String> {
    let Some(raw) = value.get(field) else {
        return Ok(None);
    };
    raw.as_bool()
        .map(Some)
        .ok_or_else(|| format!("{field} must be a boolean"))
}

fn optional_i32_array(value: &Value, field: &'static str) -> Result<Option<Vec<i32>>, String> {
    let Some(raw) = value.get(field) else {
        return Ok(None);
    };
    let raw = raw
        .as_array()
        .ok_or_else(|| format!("{field} must be an integer array"))?;
    raw.iter()
        .map(|value| {
            let value = value
                .as_i64()
                .ok_or_else(|| format!("{field} must be an integer array"))?;
            i32::try_from(value)
                .map_err(|_| format!("{field} contains an integer too large for i32"))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn unix_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
