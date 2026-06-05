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
use crate::worker::WorkerExecutor;

pub struct HttpRouterService<T, W> {
    runtime: Arc<Mutex<RouterRuntime<T, W>>>,
    model_info: RouterGetModelInfoResponse,
    allow_disaggregated_requests: bool,
    max_transfer_polls: usize,
}

impl<T, W> Clone for HttpRouterService<T, W> {
    fn clone(&self) -> Self {
        Self {
            runtime: Arc::clone(&self.runtime),
            model_info: self.model_info.clone(),
            allow_disaggregated_requests: self.allow_disaggregated_requests,
            max_transfer_polls: self.max_transfer_polls,
        }
    }
}

impl<T, W> HttpRouterService<T, W> {
    pub fn new(runtime: RouterRuntime<T, W>, model_info: RouterGetModelInfoResponse) -> Self {
        Self {
            runtime: Arc::new(Mutex::new(runtime)),
            model_info,
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
            bootstrap_room: required_i32(payload, "bootstrap_room")?,
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
        bootstrap_room: required_i32(value, "bootstrap_room")?,
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

fn required_i32(value: &Value, field: &'static str) -> Result<i32, String> {
    let raw = value
        .get(field)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("{field} must be an integer"))?;
    i32::try_from(raw).map_err(|_| format!("{field} is too large for i32"))
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
