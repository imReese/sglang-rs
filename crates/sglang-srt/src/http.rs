use std::fmt;
use std::future::Future;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};

use crate::router::{
    RouterDisaggregatedParams, RouterGenerateResponseBody, RouterGetModelInfoResponse,
    RouterRuntime, RouterSamplingParams, RouterTextGenerateRequest,
};
use crate::tokenizer::Tokenizer;
use crate::types::BootstrapRoom;
use crate::worker::WorkerExecutor;

pub struct HttpRouterService<T, W> {
    runtime: Arc<Mutex<RouterRuntime<T, W>>>,
    model_info: RouterGetModelInfoResponse,
    server_info: HttpServerInfo,
    allow_disaggregated_requests: bool,
    max_transfer_polls: usize,
}

impl<T, W> Clone for HttpRouterService<T, W> {
    fn clone(&self) -> Self {
        Self {
            runtime: Arc::clone(&self.runtime),
            model_info: self.model_info.clone(),
            server_info: self.server_info.clone(),
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
}

impl Default for HttpServerInfo {
    fn default() -> Self {
        Self {
            disaggregation_mode: "null".to_string(),
            disaggregation_bootstrap_port: None,
            kv_events: None,
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

impl<T, W> HttpRouterService<T, W> {
    pub fn new(runtime: RouterRuntime<T, W>, model_info: RouterGetModelInfoResponse) -> Self {
        Self {
            runtime: Arc::new(Mutex::new(runtime)),
            model_info,
            server_info: HttpServerInfo::default(),
            allow_disaggregated_requests: false,
            max_transfer_polls: 0,
        }
    }

    pub fn runtime(&self) -> &Arc<Mutex<RouterRuntime<T, W>>> {
        &self.runtime
    }

    pub fn with_disaggregated_requests(mut self) -> Self {
        self.allow_disaggregated_requests = true;
        self
    }

    pub fn with_server_info(mut self, server_info: HttpServerInfo) -> Self {
        self.server_info = server_info;
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
            .route("/server_info", get(server_info::<T, W>))
            .route("/v1/chat/completions", post(chat_completions::<T, W>))
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

async fn list_models<T, W>(State(service): State<HttpRouterService<T, W>>) -> Json<Value>
where
    T: Send + 'static,
    W: Send + 'static,
{
    Json(json!({
        "object": "list",
        "data": [{
            "id": service.model_info.served_model_name,
            "object": "model",
            "owned_by": "sglang-rs",
            "root": service.model_info.model_path,
        }]
    }))
}

async fn server_info<T, W>(State(service): State<HttpRouterService<T, W>>) -> Json<Value>
where
    T: Send + 'static,
    W: Send + 'static,
{
    let mut body = json!({
        "served_model_name": service.model_info.served_model_name,
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

    Json(body)
}

async fn generate<T, W>(
    State(service): State<HttpRouterService<T, W>>,
    Json(payload): Json<Value>,
) -> impl IntoResponse
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
            );
        }
    };
    if request.disaggregated_params.is_some() && !service.allow_disaggregated_requests {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({
                "error": {
                    "message": "disaggregated HTTP generate requires a PD transfer-enabled runtime"
                }
            })),
        );
    }

    let response = {
        let mut runtime = service
            .runtime
            .lock()
            .expect("HTTP router runtime lock should be held");
        if service.max_transfer_polls == 0 {
            runtime.generate_text_stream(request)
        } else {
            runtime.generate_text_stream_with_transfer_polling(request, service.max_transfer_polls)
        }
    };

    match response {
        Ok(mut responses) => {
            let Some(response) = responses.pop() else {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": { "message": "generation produced no response" } })),
                );
            };
            match response.body {
                RouterGenerateResponseBody::Complete(complete) => (
                    StatusCode::OK,
                    Json(json!({
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
                ),
                RouterGenerateResponseBody::Chunk(_) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(
                        json!({ "error": { "message": "non-stream HTTP generate returned a stream chunk" } }),
                    ),
                ),
                RouterGenerateResponseBody::Error(error) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": { "message": error.message } })),
                ),
            }
        }
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": { "message": error.to_string() } })),
        ),
    }
}

async fn chat_completions<T, W>(
    State(service): State<HttpRouterService<T, W>>,
    Json(payload): Json<Value>,
) -> impl IntoResponse
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let model = service.model_info.served_model_name.clone();
    let request = match http_chat_payload_to_router_request(payload, &model) {
        Ok(request) => request,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": { "message": error } })),
            );
        }
    };
    if request.stream {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({
                "error": {
                    "message": "streaming chat completions are not implemented by the Rust worker HTTP endpoint"
                }
            })),
        );
    }
    if request.disaggregated_params.is_some() && !service.allow_disaggregated_requests {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({
                "error": {
                    "message": "disaggregated HTTP chat completions require a PD transfer-enabled runtime"
                }
            })),
        );
    }

    let response = {
        let mut runtime = service
            .runtime
            .lock()
            .expect("HTTP router runtime lock should be held");
        if service.max_transfer_polls == 0 {
            runtime.generate_text_stream(request)
        } else {
            runtime.generate_text_stream_with_transfer_polling(request, service.max_transfer_polls)
        }
    };

    match response {
        Ok(mut responses) => {
            let Some(response) = responses.pop() else {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": { "message": "generation produced no response" } })),
                );
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
                ),
                RouterGenerateResponseBody::Chunk(_) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({
                        "error": {
                            "message": "non-stream HTTP chat completion returned a stream chunk"
                        }
                    })),
                ),
                RouterGenerateResponseBody::Error(error) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": { "message": error.message } })),
                ),
            }
        }
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": { "message": error.to_string() } })),
        ),
    }
}

fn http_generate_payload_to_router_request(
    payload: Value,
) -> Result<RouterTextGenerateRequest, String> {
    let text = payload
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing text".to_string())?
        .to_string();
    let request_id = payload
        .get("request_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let sampling_params = payload
        .get("sampling_params")
        .map(json_to_sampling_params)
        .transpose()?;
    let disaggregated_params = json_to_disaggregated_params(&payload)?;
    let data_parallel_rank = optional_i32(&payload, "data_parallel_rank")?.unwrap_or_default();

    Ok(RouterTextGenerateRequest {
        request_id,
        text,
        sampling_params,
        disaggregated_params,
        stream: false,
        data_parallel_rank,
        ..Default::default()
    })
}

fn http_chat_payload_to_router_request(
    payload: Value,
    served_model_name: &str,
) -> Result<RouterTextGenerateRequest, String> {
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

    Ok(RouterTextGenerateRequest {
        request_id: payload
            .get("request_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        text: chat_messages_to_prompt_text(&payload)?,
        sampling_params: Some(sampling_params),
        disaggregated_params: json_to_disaggregated_params(&payload)?,
        stream: optional_bool(&payload, "stream")?.unwrap_or(false),
        data_parallel_rank: optional_i32(&payload, "data_parallel_rank")?.unwrap_or_default(),
        ..Default::default()
    })
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

fn required_bootstrap_room(value: &Value, field: &'static str) -> Result<BootstrapRoom, String> {
    let raw = value
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{field} must be an unsigned integer"))?;
    if raw > i64::MAX as u64 {
        return Err(format!("{field} must fit in signed 63-bit range"));
    }

    Ok(raw)
}

fn required_u16(value: &Value, field: &'static str) -> Result<u16, String> {
    let raw = value
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{field} must be an unsigned integer"))?;
    u16::try_from(raw).map_err(|_| format!("{field} is too large for u16"))
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
