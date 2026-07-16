use std::collections::HashMap;
use std::convert::Infallible;
use std::fmt;
use std::future::Future;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::extract::{Query, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_stream::wrappers::ReceiverStream;

use crate::engine_info_bootstrap::EngineInfoBootstrapService;
use crate::openai_classify::{classify_response_json, parse_classify_request};
use crate::openai_embedding::{embeddings_response_json, parse_embedding_request};
use crate::openai_id::openai_response_id;
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
use crate::serving_metrics::{
    RequestObservation, ServingMetrics as HttpMetrics,
    observe_responses as observe_router_responses,
};
use crate::tokenizer::{ChatTemplateInput, Tokenizer};
use crate::types::BootstrapRoom;
use crate::weight_update::{get_weights_by_name_from_disk, update_model_info_from_disk};
use crate::worker::WorkerExecutor;

pub struct HttpRouterService<T, W> {
    runtime: Arc<Mutex<RouterRuntime<T, W>>>,
    model_info: Arc<Mutex<RouterGetModelInfoResponse>>,
    profile: Arc<Mutex<Option<ProfileSession>>>,
    profile_attributes: HashMap<String, String>,
    server_info: HttpServerInfo,
    metrics: Arc<HttpMetrics>,
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
            metrics: Arc::clone(&self.metrics),
            engine_info_bootstrap: self.engine_info_bootstrap.clone(),
            allow_disaggregated_requests: self.allow_disaggregated_requests,
            max_transfer_polls: self.max_transfer_polls,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpServerInfo {
    pub tp_size: usize,
    pub dp_size: usize,
    pub load_balance_method: String,
    pub max_running_requests: Option<usize>,
    pub max_prefill_tokens: Option<usize>,
    pub max_total_tokens: Option<usize>,
    pub disaggregation_mode: String,
    pub disaggregation_bootstrap_port: Option<u16>,
    pub enable_metrics: bool,
    pub kv_events: Option<HttpKvEventsInfo>,
    pub kv_cache: Option<HttpKvCacheInfo>,
}

impl Default for HttpServerInfo {
    fn default() -> Self {
        Self {
            tp_size: 1,
            dp_size: 1,
            load_balance_method: "round_robin".to_string(),
            max_running_requests: None,
            max_prefill_tokens: None,
            max_total_tokens: None,
            disaggregation_mode: "null".to_string(),
            disaggregation_bootstrap_port: None,
            enable_metrics: false,
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
            metrics: Arc::new(HttpMetrics::default()),
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
        let enable_metrics = self.server_info.enable_metrics;
        let router = Router::new()
            .route("/health", get(health))
            .route("/health_generate", get(health_generate::<T, W>))
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
                "/poll_transfers",
                get(poll_transfers::<T, W>).post(poll_transfers::<T, W>),
            )
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
            .route("/v1/responses", post(responses::<T, W>))
            .route("/v1/rerank", post(rerank::<T, W>))
            .route("/rerank", post(rerank::<T, W>))
            .route("/v1/score", post(score::<T, W>))
            .route("/v1/embeddings", post(embeddings::<T, W>))
            .route("/v1/classify", post(classify::<T, W>))
            .route("/generate", post(generate::<T, W>));
        let router = if enable_metrics {
            router.route("/metrics", get(metrics::<T, W>))
        } else {
            router
        };
        router.with_state(self)
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

async fn metrics<T, W>(State(service): State<HttpRouterService<T, W>>) -> Response
where
    T: Send + 'static,
    W: Send + 'static,
{
    let mut response = Response::new(Body::from(service.metrics.render()));
    response.headers_mut().insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
    );
    response
}

async fn health_generate<T, W>(State(service): State<HttpRouterService<T, W>>) -> Response
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let mut runtime = match service.runtime.try_lock() {
        Ok(runtime) => runtime,
        Err(std::sync::TryLockError::WouldBlock) => {
            return (
                StatusCode::OK,
                Json(json!({
                    "healthy": true,
                    "message": "runtime is actively serving a request"
                })),
            )
                .into_response();
        }
        Err(std::sync::TryLockError::Poisoned(_)) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "healthy": false,
                    "message": "router runtime mutex poisoned"
                })),
            )
                .into_response();
        }
    };

    if service.allow_disaggregated_requests {
        let load = runtime.load();
        return (
            StatusCode::OK,
            Json(json!({
                "healthy": true,
                "message": "PD runtime control plane is ready",
                "waiting_queue_depth": load.waiting_queue_depth,
                "decode_queue_depth": load.decode_queue_depth,
            })),
        )
            .into_response();
    }

    let probe = RouterGenerateRequest {
        request_id: String::new(),
        tokenized: Some(RouterTokenizedInput {
            original_text: String::new(),
            input_ids: vec![0],
        }),
        sampling_params: Some(RouterSamplingParams {
            max_new_tokens: Some(1),
            temperature: Some(0.0),
            ..RouterSamplingParams::default()
        }),
        stream: false,
        ..RouterGenerateRequest::default()
    };
    match runtime.generate_stream(probe) {
        Ok(_) => (
            StatusCode::OK,
            Json(json!({ "healthy": true, "message": "ready" })),
        )
            .into_response(),
        Err(error) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "healthy": false,
                "message": error.to_string(),
            })),
        )
            .into_response(),
    }
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
        "model_path": info.model_path,
        "tp_size": service.server_info.tp_size,
        "dp_size": service.server_info.dp_size,
        "load_balance_method": service.server_info.load_balance_method,
        "disaggregation_mode": service.server_info.disaggregation_mode,
        "enable_metrics": service.server_info.enable_metrics,
    });

    if let Some(max_running_requests) = service.server_info.max_running_requests {
        body["max_running_requests"] = json!(max_running_requests);
        body["max_num_reqs"] = json!(max_running_requests);
    }
    if let Some(max_prefill_tokens) = service.server_info.max_prefill_tokens {
        body["max_prefill_tokens"] = json!(max_prefill_tokens);
    }
    if let Some(max_total_tokens) = service.server_info.max_total_tokens {
        body["max_total_tokens"] = json!(max_total_tokens);
    }
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
    let total_tokens = num_running_reqs + num_waiting_reqs;

    Json(json!({
        "timestamp": unix_timestamp_secs(),
        "version": env!("CARGO_PKG_VERSION"),
        "aggregate": {
            "total_tokens": total_tokens,
        },
        "loads": [{
            "dp_rank": 0,
            "num_running_reqs": num_running_reqs,
            "num_waiting_reqs": num_waiting_reqs,
            "num_reqs": total_tokens,
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

async fn poll_transfers<T, W>(State(service): State<HttpRouterService<T, W>>) -> Response
where
    T: Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let response = match service.runtime.lock() {
        Ok(mut runtime) => match runtime.poll_transfers() {
            Ok(response) => response,
            Err(error) => return router_runtime_error_json(error),
        },
        Err(_) => return internal_error_json("router runtime mutex poisoned"),
    };

    Json(json!({
        "completed_batches": response.completed_batches,
        "pending_batches": response.pending_batches,
        "completed_descriptor_checksums": response.completed_descriptor_checksums,
        "pending_descriptor_checksums": response.pending_descriptor_checksums,
    }))
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
        Err(message) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": { "message": message } })),
            )
                .into_response();
        }
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

fn profile_output_dir_from_body(body: &[u8]) -> Result<Option<String>, &'static str> {
    if body.is_empty() {
        return Ok(None);
    }
    let value: Value =
        serde_json::from_slice(body).map_err(|_| "request body must be a JSON object")?;
    let object = value
        .as_object()
        .ok_or("request body must be a JSON object")?;
    match object.get("output_dir") {
        Some(Value::String(output_dir)) => Ok(Some(output_dir.clone())),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err("output_dir must be a string when provided"),
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
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Response
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let request = match http_generate_payload_to_router_request_with_headers(payload, &headers) {
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
    if stream {
        return match request {
            HttpGenerateRequest::Text(request) => {
                http_text_stream_response(service, request, |response| {
                    generate_stream_frame(response).map(|frame| vec![frame])
                })
            }
            HttpGenerateRequest::Tokenized(request) => http_token_stream_response(service, request),
            HttpGenerateRequest::BatchText(_) | HttpGenerateRequest::BatchTokenized(_) => {
                bad_request_json("streaming batched /generate is not supported yet")
            }
        };
    }
    let mut observation = RequestObservation::new(Arc::clone(&service.metrics));
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

    match &response {
        Ok(HttpGenerateResponse::Single(responses)) => {
            observe_router_responses(&mut observation, responses)
        }
        Ok(HttpGenerateResponse::Batch(batch)) => {
            for responses in batch {
                observe_router_responses(&mut observation, responses);
            }
        }
        Err(_) => {}
    }
    observation.finish(response.is_ok());
    match response {
        Ok(HttpGenerateResponse::Single(mut responses)) => {
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

fn generate_stream_frame(response: RouterGenerateResponse) -> Result<Bytes, String> {
    let payload = match response.body {
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
        RouterGenerateResponseBody::Error(error) => {
            json!({ "error": { "message": error.message } })
        }
    };
    let payload = serde_json::to_string(&payload)
        .map_err(|error| format!("serialize SGLang generate stream JSON: {error}"))?;
    Ok(Bytes::from(format!("data: {payload}\n\n")))
}

fn http_token_stream_response<T, W>(
    service: HttpRouterService<T, W>,
    request: RouterGenerateRequest,
) -> Response
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let (sender, receiver) = tokio::sync::mpsc::channel::<Result<Bytes, Infallible>>(16);
    let runtime = Arc::clone(&service.runtime);
    let metrics = Arc::clone(&service.metrics);
    let max_transfer_polls = service.max_transfer_polls;
    tokio::task::spawn_blocking(move || {
        let mut observation = RequestObservation::new(metrics);
        let mut connected = true;
        let result = match runtime.lock() {
            Ok(mut runtime) => {
                let mut send = |response: RouterGenerateResponse| {
                    observation.observe(&response);
                    let sent = match generate_stream_frame(response) {
                        Ok(frame) => sender.blocking_send(Ok(frame)).is_ok(),
                        Err(message) => {
                            let _ = sender.blocking_send(Ok(sse_error_frame(message)));
                            false
                        }
                    };
                    connected &= sent;
                    sent
                };
                if max_transfer_polls == 0 {
                    runtime.generate_stream_with_sink(request, &mut send)
                } else {
                    runtime.generate_stream_with_transfer_polling_sink(
                        request,
                        max_transfer_polls,
                        &mut send,
                    )
                }
            }
            Err(_) => Err(RouterRuntimeError::Runtime(
                crate::engine::RuntimeError::InvalidState(
                    "HTTP router runtime mutex poisoned".to_string(),
                ),
            )),
        };
        if let Err(error) = &result {
            let _ = sender.blocking_send(Ok(sse_error_frame(error.to_string())));
        }
        let _ = sender.blocking_send(Ok(Bytes::from_static(b"data: [DONE]\n\n")));
        observation.finish(result.is_ok() && connected);
    });

    let mut response = Response::new(Body::from_stream(ReceiverStream::new(receiver)));
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
    headers: HeaderMap,
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
    let chat_template_input = match chat_template_input_from_payload(&payload) {
        Ok(input) => input,
        Err(error) => return bad_request_json(error),
    };
    let prompt = match service.runtime.lock() {
        Ok(runtime) => match runtime.apply_chat_template(&chat_template_input) {
            Ok(prompt) => prompt,
            Err(error) => return bad_request_json(error.to_string()),
        },
        Err(_) => return internal_error_json("HTTP router runtime mutex poisoned"),
    };
    let request =
        match http_chat_payload_to_router_request_with_headers(payload, &model, &headers, prompt) {
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
    if stream {
        return match request {
            HttpChatRequest::Single(request) => http_chat_stream_response(service, *request, model),
            HttpChatRequest::Batch(_) => {
                bad_request_json("streaming batched /v1/chat/completions is not supported yet")
            }
        };
    }

    let mut observation = RequestObservation::new(Arc::clone(&service.metrics));
    let response = {
        let mut runtime = service
            .runtime
            .lock()
            .expect("HTTP router runtime lock should be held");
        match request {
            HttpChatRequest::Single(request) => {
                if service.max_transfer_polls == 0 {
                    runtime
                        .generate_text_stream(*request)
                        .map(HttpChatResponse::Single)
                } else {
                    runtime
                        .generate_text_stream_with_transfer_polling(
                            *request,
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

    match &response {
        Ok(HttpChatResponse::Single(responses)) => {
            observe_router_responses(&mut observation, responses)
        }
        Ok(HttpChatResponse::Batch(batch)) => {
            for responses in batch {
                observe_router_responses(&mut observation, responses);
            }
        }
        Err(_) => {}
    }
    observation.finish(response.is_ok());
    match response {
        Ok(HttpChatResponse::Single(mut responses)) => {
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
                        "id": openai_response_id("chatcmpl-", &response.request_id),
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
    Single(Box<RouterTextGenerateRequest>),
    Batch(Vec<RouterTextGenerateRequest>),
}

fn http_chat_payload_to_router_request_with_headers(
    payload: Value,
    served_model_name: &str,
    headers: &HeaderMap,
    prompt: String,
) -> Result<HttpChatRequest, String> {
    http_chat_payload_to_router_request(
        payload_with_routed_dp_rank_header(payload, headers)?,
        served_model_name,
        prompt,
    )
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
            "id": openai_response_id("chatcmpl-", response_id.as_deref().unwrap_or_default()),
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

fn http_chat_stream_response<T, W>(
    service: HttpRouterService<T, W>,
    request: RouterTextGenerateRequest,
    model: String,
) -> Response
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    http_text_stream_response(service, request, move |response| {
        chat_stream_frame(response, &model).map(|frame| vec![frame])
    })
}

fn http_text_stream_response<T, W, F>(
    service: HttpRouterService<T, W>,
    request: RouterTextGenerateRequest,
    mut encode: F,
) -> Response
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
    F: FnMut(RouterGenerateResponse) -> Result<Vec<Bytes>, String> + Send + 'static,
{
    let (sender, receiver) = tokio::sync::mpsc::channel::<Result<Bytes, Infallible>>(16);
    let runtime = Arc::clone(&service.runtime);
    let metrics = Arc::clone(&service.metrics);
    let max_transfer_polls = service.max_transfer_polls;
    tokio::task::spawn_blocking(move || {
        let mut observation = RequestObservation::new(metrics);
        let mut connected = true;
        let result = match runtime.lock() {
            Ok(mut runtime) => {
                let mut send = |response: RouterGenerateResponse| {
                    observation.observe(&response);
                    let sent = match encode(response) {
                        Ok(frames) => frames
                            .into_iter()
                            .all(|frame| sender.blocking_send(Ok(frame)).is_ok()),
                        Err(message) => {
                            let _ = sender.blocking_send(Ok(sse_error_frame(message)));
                            false
                        }
                    };
                    connected &= sent;
                    sent
                };
                if max_transfer_polls == 0 {
                    runtime.generate_text_stream_with_sink(request, &mut send)
                } else {
                    runtime.generate_text_stream_with_transfer_polling_sink(
                        request,
                        max_transfer_polls,
                        &mut send,
                    )
                }
            }
            Err(_) => Err(RouterRuntimeError::Runtime(
                crate::engine::RuntimeError::InvalidState(
                    "HTTP router runtime mutex poisoned".to_string(),
                ),
            )),
        };
        if let Err(error) = &result {
            let _ = sender.blocking_send(Ok(sse_error_frame(error.to_string())));
        }
        let _ = sender.blocking_send(Ok(Bytes::from_static(b"data: [DONE]\n\n")));
        observation.finish(result.is_ok() && connected);
    });

    let mut response = Response::new(Body::from_stream(ReceiverStream::new(receiver)));
    response.headers_mut().insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("text/event-stream"),
    );
    response
}

fn chat_stream_frame(response: RouterGenerateResponse, model: &str) -> Result<Bytes, String> {
    let payload = match response.body {
        RouterGenerateResponseBody::Chunk(chunk) => json!({
            "id": openai_response_id("chatcmpl-", &response.request_id),
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{
                "index": chunk.index,
                "delta": { "content": chunk.text },
                "finish_reason": Value::Null,
            }],
        }),
        RouterGenerateResponseBody::Complete(complete) => json!({
            "id": openai_response_id("chatcmpl-", &response.request_id),
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
            json!({ "error": { "message": error.message } })
        }
    };
    let payload = serde_json::to_string(&payload)
        .map_err(|error| format!("serialize OpenAI chat stream JSON: {error}"))?;
    Ok(Bytes::from(format!("data: {payload}\n\n")))
}

fn sse_error_frame(message: impl Into<String>) -> Bytes {
    let payload = json!({ "error": { "message": message.into() } });
    Bytes::from(format!("data: {payload}\n\n"))
}

async fn responses<T, W>(
    State(service): State<HttpRouterService<T, W>>,
    headers: HeaderMap,
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
    let request =
        match http_responses_payload_to_router_request_with_headers(payload, &model, &headers) {
            Ok(request) => request,
            Err(error) => return bad_request_json(error),
        };
    let stream = request.request.stream;
    if request.request.disaggregated_params.is_some() && !service.allow_disaggregated_requests {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({
                "error": {
                    "message": "disaggregated HTTP responses require a PD transfer-enabled runtime"
                }
            })),
        )
            .into_response();
    }

    let response_model = request.model;
    let router_request = request.request;
    if stream {
        let mut encoder = ResponsesStreamEncoder::new(response_model);
        return http_text_stream_response(service, router_request, move |response| {
            encoder.encode(response)
        });
    }
    let mut observation = RequestObservation::new(Arc::clone(&service.metrics));
    let responses = {
        let mut runtime = service
            .runtime
            .lock()
            .expect("HTTP router runtime lock should be held");
        if service.max_transfer_polls == 0 {
            runtime.generate_text_stream(router_request)
        } else {
            runtime.generate_text_stream_with_transfer_polling(
                router_request,
                service.max_transfer_polls,
            )
        }
    };

    if let Ok(responses) = &responses {
        observe_router_responses(&mut observation, responses);
    }
    observation.finish(responses.is_ok());
    match responses {
        Ok(responses) => http_responses_response_from_router_responses(responses, &response_model),
        Err(error) => router_runtime_error_json(error),
    }
}

pub(crate) struct HttpResponsesRequest {
    pub(crate) model: String,
    pub(crate) request: RouterTextGenerateRequest,
}

fn http_responses_payload_to_router_request_with_headers(
    payload: Value,
    served_model_name: &str,
    headers: &HeaderMap,
) -> Result<HttpResponsesRequest, String> {
    http_responses_payload_to_router_request(
        payload_with_routed_dp_rank_header(payload, headers)?,
        served_model_name,
    )
}

pub(crate) fn http_responses_payload_to_router_request(
    payload: Value,
    served_model_name: &str,
) -> Result<HttpResponsesRequest, String> {
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
    apply_openai_sampling_params(&payload, &mut sampling_params)?;
    if let Some(max_output_tokens) = optional_i32(&payload, "max_output_tokens")? {
        sampling_params.max_new_tokens = Some(max_output_tokens);
    }

    let mut text = responses_input_to_text(&payload)?;
    if let Some(instructions) = optional_string(&payload, "instructions")? {
        text = if text.is_empty() {
            instructions.to_string()
        } else {
            format!("{instructions}\n{text}")
        };
    }

    Ok(HttpResponsesRequest {
        model: model.to_string(),
        request: RouterTextGenerateRequest {
            request_id: payload
                .get("request_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            text,
            sampling_params: Some(sampling_params),
            disaggregated_params: json_to_disaggregated_params(&payload)?,
            stream: optional_bool(&payload, "stream")?.unwrap_or(false),
            data_parallel_rank: optional_routed_dp_rank(&payload)?.unwrap_or_default(),
            ..Default::default()
        },
    })
}

fn responses_input_to_text(payload: &Value) -> Result<String, String> {
    let input = payload
        .get("input")
        .ok_or_else(|| "missing input".to_string())?;
    responses_input_value_to_text(input)
}

fn responses_input_value_to_text(value: &Value) -> Result<String, String> {
    if let Some(text) = value.as_str() {
        return Ok(text.to_string());
    }
    let Some(items) = value.as_array() else {
        return Err("input must be a string or array".to_string());
    };
    if items.is_empty() {
        return Err("input array must not be empty".to_string());
    }

    items
        .iter()
        .map(responses_input_item_to_text)
        .collect::<Result<Vec<_>, _>>()
        .map(|texts| texts.join("\n"))
}

fn responses_input_item_to_text(value: &Value) -> Result<String, String> {
    if let Some(text) = value.as_str() {
        return Ok(text.to_string());
    }

    let content = value
        .get("content")
        .ok_or_else(|| "responses input item content is required".to_string())?;
    responses_content_to_text(content)
}

fn responses_content_to_text(value: &Value) -> Result<String, String> {
    if let Some(text) = value.as_str() {
        return Ok(text.to_string());
    }
    let Some(parts) = value.as_array() else {
        return Err("responses content must be a string or array".to_string());
    };

    let mut text = String::new();
    for part in parts {
        let part_type = part.get("type").and_then(Value::as_str);
        if matches!(part_type, Some("input_text" | "text")) {
            let part_text = part
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| "responses text content part requires text".to_string())?;
            text.push_str(part_text);
        }
    }
    Ok(text)
}

fn http_responses_response_from_router_responses(
    mut responses: Vec<RouterGenerateResponse>,
    model: &str,
) -> Response {
    let Some(response) = responses.pop() else {
        return internal_error_json("generation produced no response");
    };

    match response.body {
        RouterGenerateResponseBody::Complete(complete) => (
            StatusCode::OK,
            Json(responses_complete_json(
                model,
                &response.request_id,
                complete,
            )),
        )
            .into_response(),
        RouterGenerateResponseBody::Chunk(_) => {
            internal_error_json("non-stream HTTP responses returned a stream chunk")
        }
        RouterGenerateResponseBody::Error(error) => internal_error_json(error.message),
    }
}

struct ResponsesStreamEncoder {
    model: String,
    sequence_number: i32,
    message_open: bool,
    response_id: String,
    item_id: String,
}

impl ResponsesStreamEncoder {
    fn new(model: String) -> Self {
        Self {
            model,
            sequence_number: 0,
            message_open: false,
            response_id: String::new(),
            item_id: String::new(),
        }
    }

    fn encode(&mut self, response: RouterGenerateResponse) -> Result<Vec<Bytes>, String> {
        let mut frames = Vec::new();
        if self.response_id.is_empty() {
            self.response_id = openai_response_id("resp-", &response.request_id);
            self.item_id = responses_message_id(&response.request_id);
            frames.push(self.event(json!({
                "type": "response.created",
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "created_at": unix_timestamp_secs(),
                    "status": "in_progress",
                    "model": self.model,
                    "output": [],
                    "output_text": "",
                },
            }))?);
        }

        match response.body {
            RouterGenerateResponseBody::Chunk(chunk) => {
                self.open_message(&mut frames)?;
                if !chunk.text.is_empty() {
                    frames.push(self.event(json!({
                        "type": "response.output_text.delta",
                        "output_index": chunk.index,
                        "content_index": 0,
                        "item_id": self.item_id,
                        "delta": chunk.text,
                        "logprobs": [],
                    }))?);
                }
            }
            RouterGenerateResponseBody::Complete(complete) => {
                self.open_message(&mut frames)?;
                let output_index = complete.index;
                let output_text = complete.text.clone();
                frames.push(self.event(json!({
                    "type": "response.output_text.done",
                    "output_index": output_index,
                    "content_index": 0,
                    "item_id": self.item_id,
                    "text": output_text,
                    "logprobs": [],
                }))?);
                let text_part = json!({
                    "type": "output_text",
                    "text": output_text,
                    "annotations": [],
                });
                frames.push(self.event(json!({
                    "type": "response.content_part.done",
                    "item_id": self.item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": text_part,
                }))?);
                frames.push(self.event(json!({
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": {
                        "id": self.item_id,
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [text_part],
                    },
                }))?);
                frames.push(self.event(json!({
                    "type": "response.completed",
                    "response": responses_complete_json(
                        &self.model,
                        &response.request_id,
                        complete,
                    ),
                }))?);
            }
            RouterGenerateResponseBody::Error(error) => {
                frames.push(self.event(json!({
                    "type": "response.failed",
                    "error": { "message": error.message },
                }))?);
            }
        }
        Ok(frames)
    }

    fn open_message(&mut self, frames: &mut Vec<Bytes>) -> Result<(), String> {
        if self.message_open {
            return Ok(());
        }
        self.message_open = true;
        frames.push(self.event(json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": self.item_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        }))?);
        frames.push(self.event(json!({
            "type": "response.content_part.added",
            "item_id": self.item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": "",
                "annotations": [],
                "logprobs": Value::Null,
            },
        }))?);
        Ok(())
    }

    fn event(&mut self, mut event: Value) -> Result<Bytes, String> {
        if let Some(object) = event.as_object_mut() {
            object.insert("sequence_number".to_string(), json!(self.sequence_number));
        }
        self.sequence_number += 1;
        let event = serde_json::to_string(&event)
            .map_err(|error| format!("serialize OpenAI responses stream JSON: {error}"))?;
        Ok(Bytes::from(format!("data: {event}\n\n")))
    }
}

pub(crate) fn responses_complete_json(
    model: &str,
    request_id: &str,
    complete: RouterGenerateComplete,
) -> Value {
    let response_id = openai_response_id("resp-", request_id);
    let item_id = responses_message_id(request_id);
    let output_text = complete.text;
    json!({
        "id": response_id,
        "object": "response",
        "created_at": unix_timestamp_secs(),
        "status": "completed",
        "model": model,
        "output": [{
            "id": item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": output_text.clone(),
                "annotations": [],
            }],
        }],
        "output_text": output_text,
        "usage": {
            "input_tokens": complete.prompt_tokens,
            "output_tokens": complete.completion_tokens,
            "total_tokens": complete.prompt_tokens + complete.completion_tokens,
            "cached_tokens": complete.cached_tokens,
        },
    })
}

fn responses_message_id(request_id: &str) -> String {
    format!("msg_{request_id}")
}

async fn completions<T, W>(
    State(service): State<HttpRouterService<T, W>>,
    headers: HeaderMap,
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
    let request =
        match http_completion_payload_to_router_request_with_headers(payload, &model, &headers) {
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
    if stream {
        return match request {
            HttpCompletionRequest::Single(request) => {
                let stream_model = model.clone();
                http_text_stream_response(service, *request, move |response| {
                    completion_stream_frame(response, &stream_model).map(|frame| vec![frame])
                })
            }
            HttpCompletionRequest::Batch(_) => {
                bad_request_json("streaming batched /v1/completions is not supported yet")
            }
        };
    }

    let mut observation = RequestObservation::new(Arc::clone(&service.metrics));
    let response = {
        let mut runtime = service
            .runtime
            .lock()
            .expect("HTTP router runtime lock should be held");
        match request {
            HttpCompletionRequest::Single(request) => {
                if service.max_transfer_polls == 0 {
                    runtime
                        .generate_text_stream(*request)
                        .map(HttpCompletionResponse::Single)
                } else {
                    runtime
                        .generate_text_stream_with_transfer_polling(
                            *request,
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

    match &response {
        Ok(HttpCompletionResponse::Single(responses)) => {
            observe_router_responses(&mut observation, responses)
        }
        Ok(HttpCompletionResponse::Batch(batch)) => {
            for responses in batch {
                observe_router_responses(&mut observation, responses);
            }
        }
        Err(_) => {}
    }
    observation.finish(response.is_ok());
    match response {
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
    Single(Box<RouterTextGenerateRequest>),
    Batch(Vec<RouterTextGenerateRequest>),
}

fn http_completion_payload_to_router_request_with_headers(
    payload: Value,
    served_model_name: &str,
    headers: &HeaderMap,
) -> Result<HttpCompletionRequest, String> {
    http_completion_payload_to_router_request(
        payload_with_routed_dp_rank_header(payload, headers)?,
        served_model_name,
    )
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
                "id": openai_response_id("cmpl-", &response.request_id),
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
            "id": openai_response_id("cmpl-", response_id.as_deref().unwrap_or_default()),
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

fn completion_stream_frame(response: RouterGenerateResponse, model: &str) -> Result<Bytes, String> {
    let payload = match response.body {
        RouterGenerateResponseBody::Chunk(chunk) => json!({
            "id": openai_response_id("cmpl-", &response.request_id),
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
            "id": openai_response_id("cmpl-", &response.request_id),
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
            json!({ "error": { "message": error.message } })
        }
    };
    let payload = serde_json::to_string(&payload)
        .map_err(|error| format!("serialize OpenAI completion stream JSON: {error}"))?;
    Ok(Bytes::from(format!("data: {payload}\n\n")))
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

fn http_generate_payload_to_router_request_with_headers(
    payload: Value,
    headers: &HeaderMap,
) -> Result<HttpGenerateRequest, String> {
    http_generate_payload_to_router_request(payload_with_routed_dp_rank_header(payload, headers)?)
}

fn http_generate_payload_to_router_request(payload: Value) -> Result<HttpGenerateRequest, String> {
    if payload.get("input_ids").is_some() {
        return match http_generate_payload_to_router_token_requests(payload)? {
            HttpTokenGenerateRequests::Single(request) => {
                Ok(HttpGenerateRequest::Tokenized(*request))
            }
            HttpTokenGenerateRequests::Batch(requests) => {
                Ok(HttpGenerateRequest::BatchTokenized(requests))
            }
        };
    }
    match http_generate_payload_to_router_text_requests(payload)? {
        HttpTextGenerateRequests::Single(request) => Ok(HttpGenerateRequest::Text(*request)),
        HttpTextGenerateRequests::Batch(requests) => Ok(HttpGenerateRequest::BatchText(requests)),
    }
}

fn payload_with_routed_dp_rank_header(
    mut payload: Value,
    headers: &HeaderMap,
) -> Result<Value, String> {
    let Some(rank) = routed_dp_rank_header(headers)? else {
        return Ok(payload);
    };
    let Some(object) = payload.as_object_mut() else {
        return Err("request body must be a JSON object".to_string());
    };
    object.insert("routed_dp_rank".to_string(), json!(rank));
    Ok(payload)
}

fn routed_dp_rank_header(headers: &HeaderMap) -> Result<Option<i32>, String> {
    let Some(value) = headers.get("x-data-parallel-rank") else {
        return Ok(None);
    };
    let value = value
        .to_str()
        .map_err(|_| "Invalid X-Data-Parallel-Rank header: must be an integer".to_string())?;
    value.parse::<i32>().map(Some).map_err(|_| {
        format!("Invalid X-Data-Parallel-Rank header: must be an integer, got '{value}'")
    })
}

enum HttpTextGenerateRequests {
    Single(Box<RouterTextGenerateRequest>),
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
        let data_parallel_ranks = optional_routed_dp_rank_values(&payload, batch_size)?;
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
    let data_parallel_rank = optional_routed_dp_rank(&payload)?.unwrap_or_default();

    Ok(HttpTextGenerateRequests::Single(Box::new(
        RouterTextGenerateRequest {
            request_id,
            text,
            sampling_params,
            disaggregated_params,
            stream,
            data_parallel_rank,
            ..Default::default()
        },
    )))
}

enum HttpTokenGenerateRequests {
    Single(Box<RouterGenerateRequest>),
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
        let data_parallel_ranks = optional_routed_dp_rank_values(&payload, batch_size)?;
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
    let data_parallel_rank = optional_routed_dp_rank(&payload)?.unwrap_or_default();

    Ok(HttpTokenGenerateRequests::Single(Box::new(
        RouterGenerateRequest {
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
        },
    )))
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

fn optional_routed_dp_rank(payload: &Value) -> Result<Option<i32>, String> {
    match optional_i32(payload, "routed_dp_rank")? {
        Some(rank) => Ok(Some(rank)),
        None => optional_i32(payload, "data_parallel_rank"),
    }
}

fn optional_routed_dp_rank_values(payload: &Value, batch_size: usize) -> Result<Vec<i32>, String> {
    if payload.get("routed_dp_rank").is_some() {
        return optional_i32_values(payload, "routed_dp_rank", batch_size);
    }
    optional_i32_values(payload, "data_parallel_rank", batch_size)
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
    apply_openai_sampling_params(&payload, &mut sampling_params)?;
    if let Some(max_tokens) = optional_i32(&payload, "max_tokens")? {
        sampling_params.max_new_tokens = Some(max_tokens);
    }
    let stream = optional_bool(&payload, "stream")?.unwrap_or(false);

    match completion_prompt_to_texts(&payload)? {
        CompletionPrompts::Single(text) => Ok(HttpCompletionRequest::Single(Box::new(
            RouterTextGenerateRequest {
                request_id: payload
                    .get("request_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                text,
                sampling_params: Some(sampling_params),
                disaggregated_params: json_to_disaggregated_params(&payload)?,
                stream,
                data_parallel_rank: optional_routed_dp_rank(&payload)?.unwrap_or_default(),
                ..Default::default()
            },
        ))),
        CompletionPrompts::Batch(texts) => {
            let batch_size = texts.len();
            if batch_size == 0 {
                return Err("prompt array must not be empty".to_string());
            }
            let request_ids = optional_string_values(&payload, "request_id", batch_size)?;
            let data_parallel_ranks = optional_routed_dp_rank_values(&payload, batch_size)?;
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
    prompt: String,
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
    apply_openai_sampling_params(&payload, &mut sampling_params)?;
    if let Some(max_tokens) =
        optional_i32(&payload, "max_completion_tokens")?.or(optional_i32(&payload, "max_tokens")?)
    {
        sampling_params.max_new_tokens = Some(max_tokens);
    }
    let stream = optional_bool(&payload, "stream")?.unwrap_or(false);
    let n = optional_i32(&payload, "n")?.unwrap_or(1);
    if n < 1 {
        return Err("n must be at least 1".to_string());
    }

    if n == 1 {
        return Ok(HttpChatRequest::Single(Box::new(
            RouterTextGenerateRequest {
                request_id: payload
                    .get("request_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                text: prompt,
                sampling_params: Some(sampling_params),
                disaggregated_params: json_to_disaggregated_params(&payload)?,
                stream,
                data_parallel_rank: optional_routed_dp_rank(&payload)?.unwrap_or_default(),
                ..Default::default()
            },
        )));
    }

    let batch_size = usize::try_from(n).map_err(|_| "n is out of range".to_string())?;
    let request_ids = optional_string_values(&payload, "request_id", batch_size)?;
    let data_parallel_ranks = optional_routed_dp_rank_values(&payload, batch_size)?;
    let disaggregated_params = json_to_disaggregated_params_values(&payload, batch_size)?;
    let mut requests = Vec::with_capacity(batch_size);

    for index in 0..batch_size {
        requests.push(RouterTextGenerateRequest {
            request_id: request_ids[index].clone(),
            text: prompt.clone(),
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

pub(crate) fn chat_template_input_from_payload(
    payload: &Value,
) -> Result<ChatTemplateInput, String> {
    let messages = payload
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| "messages must be an array".to_string())?;
    if messages.is_empty() {
        return Err("messages must not be empty".to_string());
    }

    let messages = messages
        .iter()
        .map(normalize_chat_message)
        .collect::<Result<Vec<_>, _>>()?;
    let tools = match payload.get("tools") {
        None | Some(Value::Null) => None,
        Some(Value::Array(tools)) => Some(tools.clone()),
        Some(_) => return Err("tools must be an array".to_string()),
    };
    let template_kwargs = match payload.get("chat_template_kwargs") {
        None | Some(Value::Null) => serde_json::Map::new(),
        Some(Value::Object(kwargs)) => kwargs.clone(),
        Some(_) => return Err("chat_template_kwargs must be an object".to_string()),
    };
    for reserved in ["messages", "tools", "add_generation_prompt"] {
        if template_kwargs.contains_key(reserved) {
            return Err(format!(
                "chat_template_kwargs.{reserved} is controlled by the serving runtime"
            ));
        }
    }

    Ok(ChatTemplateInput {
        messages,
        tools,
        template_kwargs,
    })
}

fn normalize_chat_message(message: &Value) -> Result<Value, String> {
    let Some(message) = message.as_object() else {
        return Err("chat message entries must be objects".to_string());
    };
    let role = message
        .get("role")
        .and_then(Value::as_str)
        .ok_or_else(|| "chat message role is required".to_string())?;
    if !matches!(role, "system" | "developer" | "user" | "assistant" | "tool") {
        return Err(format!("unsupported chat message role: {role}"));
    }
    let mut normalized = message.clone();
    let Some(content) = message.get("content") else {
        if role == "assistant" && message.get("tool_calls").is_some() {
            normalized.insert("content".to_string(), Value::Null);
            return Ok(Value::Object(normalized));
        }
        return Err("chat message content is required".to_string());
    };
    if let Some(text) = content.as_str() {
        normalized.insert("content".to_string(), Value::String(text.to_string()));
        return Ok(Value::Object(normalized));
    }
    if content.is_null() && role == "assistant" {
        return Ok(Value::Object(normalized));
    }
    let Some(parts) = content.as_array() else {
        return Err("chat message content must be a string or array".to_string());
    };

    let mut text = String::new();
    for part in parts {
        if part.get("type").and_then(Value::as_str) != Some("text") {
            return Err(
                "multimodal chat content is not supported by this text runtime".to_string(),
            );
        }
        let part_text = part
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| "chat text content part requires text".to_string())?;
        text.push_str(part_text);
    }
    normalized.insert("content".to_string(), Value::String(text));
    Ok(Value::Object(normalized))
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

fn apply_openai_sampling_params(
    payload: &Value,
    sampling: &mut RouterSamplingParams,
) -> Result<(), String> {
    if let Some(value) = optional_f32(payload, "temperature")? {
        sampling.temperature = Some(value);
    }
    if let Some(value) = optional_f32(payload, "top_p")? {
        sampling.top_p = Some(value);
    }
    if let Some(value) = optional_i32(payload, "top_k")? {
        sampling.top_k = Some(value);
    }
    if let Some(value) = optional_f32(payload, "min_p")? {
        sampling.min_p = Some(value);
    }
    if let Some(value) = optional_f32(payload, "frequency_penalty")? {
        sampling.frequency_penalty = Some(value);
    }
    if let Some(value) = optional_f32(payload, "presence_penalty")? {
        sampling.presence_penalty = Some(value);
    }
    if let Some(value) = optional_f32(payload, "repetition_penalty")? {
        sampling.repetition_penalty = Some(value);
    }
    if let Some(value) = optional_bool(payload, "ignore_eos")? {
        sampling.ignore_eos = Some(value);
    }
    if let Some(value) = optional_i32_array(payload, "stop_token_ids")? {
        sampling.stop_token_ids = value;
    }

    for field in ["stop", "seed", "logit_bias", "response_format", "best_of"] {
        if payload.get(field).is_some_and(|value| !value.is_null()) {
            return Err(format!(
                "{field} is not supported by the current sampling runtime"
            ));
        }
    }
    if payload.get("logprobs").is_some_and(|value| value != false) {
        return Err("logprobs is not supported by the current sampling runtime".to_string());
    }
    Ok(())
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

fn optional_string<'a>(value: &'a Value, field: &'static str) -> Result<Option<&'a str>, String> {
    let Some(raw) = value.get(field) else {
        return Ok(None);
    };
    raw.as_str()
        .map(Some)
        .ok_or_else(|| format!("{field} must be a string"))
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn generate_payload_accepts_routed_dp_rank_for_text_request() {
        let request = http_generate_payload_to_router_request(json!({
            "text": "hello",
            "routed_dp_rank": 3
        }))
        .expect("generate payload should parse");

        match request {
            HttpGenerateRequest::Text(request) => {
                assert_eq!(request.data_parallel_rank, 3);
            }
            _ => panic!("expected text request"),
        }
    }

    #[test]
    fn token_batch_payload_accepts_routed_dp_rank_array() {
        let request = http_generate_payload_to_router_request(json!({
            "input_ids": [[1, 2], [3, 4]],
            "routed_dp_rank": [1, 2]
        }))
        .expect("batch token payload should parse");

        match request {
            HttpGenerateRequest::BatchTokenized(requests) => {
                assert_eq!(
                    requests
                        .iter()
                        .map(|request| request.data_parallel_rank)
                        .collect::<Vec<_>>(),
                    vec![1, 2]
                );
            }
            _ => panic!("expected token batch request"),
        }
    }

    #[test]
    fn chat_payload_prefers_routed_dp_rank_over_deprecated_data_parallel_rank() {
        let request = http_chat_payload_to_router_request(
            json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hello"}],
                "data_parallel_rank": 1,
                "routed_dp_rank": 4
            }),
            "tiny",
            "hello".to_string(),
        )
        .expect("chat payload should parse");

        match request {
            HttpChatRequest::Single(request) => {
                assert_eq!(request.data_parallel_rank, 4);
            }
            _ => panic!("expected single chat request"),
        }
    }

    #[test]
    fn chat_payload_maps_openai_sampling_parameters() {
        let request = http_chat_payload_to_router_request(
            json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 3,
                "max_completion_tokens": 5,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.05,
                "frequency_penalty": 0.2,
                "presence_penalty": -0.3,
                "repetition_penalty": 1.1,
                "ignore_eos": true,
                "stop_token_ids": [1, 2]
            }),
            "tiny",
            "rendered prompt".to_string(),
        )
        .expect("chat payload should parse");

        let HttpChatRequest::Single(request) = request else {
            panic!("expected single chat request");
        };
        let sampling = request
            .sampling_params
            .expect("sampling params should be populated");
        assert_eq!(sampling.max_new_tokens, Some(5));
        assert_eq!(sampling.temperature, Some(0.7));
        assert_eq!(sampling.top_p, Some(0.8));
        assert_eq!(sampling.top_k, Some(20));
        assert_eq!(sampling.min_p, Some(0.05));
        assert_eq!(sampling.frequency_penalty, Some(0.2));
        assert_eq!(sampling.presence_penalty, Some(-0.3));
        assert_eq!(sampling.repetition_penalty, Some(1.1));
        assert_eq!(sampling.ignore_eos, Some(true));
        assert_eq!(sampling.stop_token_ids, vec![1, 2]);
    }

    #[test]
    fn completion_payload_accepts_routed_dp_rank() {
        let request = http_completion_payload_to_router_request(
            json!({
                "model": "tiny",
                "prompt": "hello",
                "routed_dp_rank": 5
            }),
            "tiny",
        )
        .expect("completion payload should parse");

        match request {
            HttpCompletionRequest::Single(request) => {
                assert_eq!(request.data_parallel_rank, 5);
            }
            _ => panic!("expected single completion request"),
        }
    }

    #[test]
    fn chat_payload_prefers_routed_dp_rank_header_over_body() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            "x-data-parallel-rank",
            axum::http::HeaderValue::from_static("6"),
        );
        let request = http_chat_payload_to_router_request_with_headers(
            json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hello"}],
                "routed_dp_rank": 2
            }),
            "tiny",
            &headers,
            "hello".to_string(),
        )
        .expect("chat payload should parse");

        match request {
            HttpChatRequest::Single(request) => {
                assert_eq!(request.data_parallel_rank, 6);
            }
            _ => panic!("expected single chat request"),
        }
    }

    #[test]
    fn generate_batch_payload_uses_routed_dp_rank_header_for_each_item() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            "x-data-parallel-rank",
            axum::http::HeaderValue::from_static("7"),
        );
        let request = http_generate_payload_to_router_request_with_headers(
            json!({
                "text": ["hello", "there"],
                "routed_dp_rank": [1, 2]
            }),
            &headers,
        )
        .expect("batch generate payload should parse");

        match request {
            HttpGenerateRequest::BatchText(requests) => {
                assert_eq!(
                    requests
                        .iter()
                        .map(|request| request.data_parallel_rank)
                        .collect::<Vec<_>>(),
                    vec![7, 7]
                );
            }
            _ => panic!("expected text batch request"),
        }
    }

    #[test]
    fn completion_payload_rejects_invalid_routed_dp_rank_header() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            "x-data-parallel-rank",
            axum::http::HeaderValue::from_static("not-a-rank"),
        );
        let result = http_completion_payload_to_router_request_with_headers(
            json!({
                "model": "tiny",
                "prompt": "hello"
            }),
            "tiny",
            &headers,
        );
        let error = match result {
            Ok(_) => panic!("invalid header should reject request"),
            Err(error) => error,
        };

        assert_eq!(
            error,
            "Invalid X-Data-Parallel-Rank header: must be an integer, got 'not-a-rank'"
        );
    }
}
