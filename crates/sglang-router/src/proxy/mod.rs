// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! HTTP proxy — forwards requests to the upstream SGLang worker.

pub mod sse;

use crate::health::circuit_breaker::CircuitBreaker;
use crate::server::error::ApiError;
use crate::server::header_utils::should_forward_request_header;
use anyhow::Context;
use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response};
use bytes::Bytes;
use futures::StreamExt;
use reqwest::{Client, Url};
use serde_json::json;
use serde_json::Value;
use sglang_srt::proto::sglang::runtime::v1::generate_response::Body as ProtoGenerateResponseBody;
use sglang_srt::proto::sglang::runtime::v1::sglang_service_client::SglangServiceClient;
use sglang_srt::proto::sglang::runtime::v1::{
    DisaggregatedParams as ProtoDisaggregatedParams, GenerateResponse as ProtoGenerateResponse,
    OpenAiJsonRequest, OpenAiJsonResponse, RequestOptions as ProtoRequestOptions,
    SamplingParams as ProtoSamplingParams, TextGenerateRequest as ProtoTextGenerateRequest,
};
use std::sync::Arc;
use std::time::Duration;
use tonic::Code;
use tonic::Request as GrpcRequest;

/// Parse a worker URL emitted by discovery.  On failure, trip the worker's
/// circuit breaker so the malformed worker drops out of subsequent
/// `healthy_workers_for(...)` selection, then surface the error as
/// `ApiError::WorkerMisconfigured`.
fn parse_worker_url(worker_url: &str, breaker: &CircuitBreaker) -> Result<Url, ApiError> {
    Url::parse(worker_url).map_err(|e| {
        breaker.record_failure();
        ApiError::WorkerMisconfigured {
            worker: worker_url.to_string(),
            source: anyhow::Error::new(e).context("parse worker URL"),
        }
    })
}

fn is_grpc_worker_url(worker_url: &str) -> bool {
    worker_url.starts_with("grpc://") || worker_url.starts_with("grpcs://")
}

fn grpc_endpoint_from_worker_url(worker_url: &Url) -> Option<String> {
    match worker_url.scheme() {
        "grpc" => Some(format!(
            "http://{}",
            worker_url.as_str().trim_start_matches("grpc://")
        )),
        "grpcs" => Some(format!(
            "https://{}",
            worker_url.as_str().trim_start_matches("grpcs://")
        )),
        _ => None,
    }
}

enum GrpcForwardError {
    Status(tonic::Status),
    Transport(anyhow::Error),
}

fn grpc_status_to_http_status(status: Code) -> axum::http::StatusCode {
    match status {
        Code::Ok => axum::http::StatusCode::OK,
        Code::InvalidArgument | Code::FailedPrecondition | Code::OutOfRange => {
            axum::http::StatusCode::BAD_REQUEST
        }
        Code::Unauthenticated => axum::http::StatusCode::UNAUTHORIZED,
        Code::PermissionDenied => axum::http::StatusCode::FORBIDDEN,
        Code::NotFound => axum::http::StatusCode::NOT_FOUND,
        Code::AlreadyExists | Code::Aborted => axum::http::StatusCode::CONFLICT,
        Code::ResourceExhausted => axum::http::StatusCode::TOO_MANY_REQUESTS,
        Code::Cancelled => axum::http::StatusCode::BAD_GATEWAY,
        Code::Unknown | Code::Internal | Code::DataLoss => {
            axum::http::StatusCode::INTERNAL_SERVER_ERROR
        }
        Code::Unavailable => axum::http::StatusCode::SERVICE_UNAVAILABLE,
        Code::DeadlineExceeded => axum::http::StatusCode::GATEWAY_TIMEOUT,
        Code::Unimplemented => axum::http::StatusCode::NOT_IMPLEMENTED,
    }
}

fn grpc_status_response(status: tonic::Status) -> Response<Body> {
    let http_status = grpc_status_to_http_status(status.code());
    let body = json!({
        "error": {
            "message": status.message(),
            "code": status.code().to_string(),
        }
    });
    let mut out = Response::new(Body::from(
        serde_json::to_vec(&body).expect("static gRPC status JSON should serialize"),
    ));
    *out.status_mut() = http_status;
    out.headers_mut().insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("application/json"),
    );
    out
}

fn grpc_generate_status(message: impl Into<String>) -> tonic::Status {
    tonic::Status::invalid_argument(message.into())
}

fn value_as_i32(value: &Value, field: &'static str) -> Result<i32, tonic::Status> {
    let Some(raw) = value.as_i64() else {
        return Err(grpc_generate_status(format!("{field} must be an integer")));
    };
    i32::try_from(raw).map_err(|_| grpc_generate_status(format!("{field} is out of i32 range")))
}

fn value_as_u32(value: &Value, field: &'static str) -> Result<u32, tonic::Status> {
    let Some(raw) = value.as_u64() else {
        return Err(grpc_generate_status(format!(
            "{field} must be an unsigned integer"
        )));
    };
    u32::try_from(raw).map_err(|_| grpc_generate_status(format!("{field} is out of u32 range")))
}

fn optional_i32_field(
    object: &serde_json::Map<String, Value>,
    field: &'static str,
) -> Result<Option<i32>, tonic::Status> {
    object
        .get(field)
        .map(|value| value_as_i32(value, field))
        .transpose()
}

fn optional_f32_field(
    object: &serde_json::Map<String, Value>,
    field: &'static str,
) -> Result<Option<f32>, tonic::Status> {
    object
        .get(field)
        .map(|value| {
            value
                .as_f64()
                .map(|raw| raw as f32)
                .ok_or_else(|| grpc_generate_status(format!("{field} must be a number")))
        })
        .transpose()
}

fn optional_bool_field(
    object: &serde_json::Map<String, Value>,
    field: &'static str,
) -> Result<Option<bool>, tonic::Status> {
    object
        .get(field)
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| grpc_generate_status(format!("{field} must be a boolean")))
        })
        .transpose()
}

fn optional_string_field(
    object: &serde_json::Map<String, Value>,
    field: &'static str,
) -> Result<Option<String>, tonic::Status> {
    object
        .get(field)
        .map(|value| {
            value
                .as_str()
                .map(ToString::to_string)
                .ok_or_else(|| grpc_generate_status(format!("{field} must be a string")))
        })
        .transpose()
}

fn proto_sampling_params_from_json(
    payload: &serde_json::Map<String, Value>,
) -> Result<Option<ProtoSamplingParams>, tonic::Status> {
    let Some(value) = payload.get("sampling_params") else {
        return Ok(None);
    };
    let Some(object) = value.as_object() else {
        return Err(grpc_generate_status("sampling_params must be an object"));
    };

    let stop = match object.get("stop") {
        Some(Value::String(item)) => vec![item.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(ToString::to_string)
                    .ok_or_else(|| grpc_generate_status("stop entries must be strings"))
            })
            .collect::<Result<Vec<_>, _>>()?,
        Some(_) => {
            return Err(grpc_generate_status(
                "stop must be a string or string array",
            ))
        }
        None => Vec::new(),
    };
    let stop_token_ids = match object.get("stop_token_ids") {
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| value_as_u32(item, "stop_token_ids"))
            .collect::<Result<Vec<_>, _>>()?,
        Some(_) => return Err(grpc_generate_status("stop_token_ids must be an array")),
        None => Vec::new(),
    };
    let seed = object
        .get("seed")
        .map(|value| {
            value
                .as_i64()
                .ok_or_else(|| grpc_generate_status("seed must be an integer"))
        })
        .transpose()?;

    Ok(Some(ProtoSamplingParams {
        max_new_tokens: optional_i32_field(object, "max_new_tokens")?,
        temperature: optional_f32_field(object, "temperature")?,
        top_p: optional_f32_field(object, "top_p")?,
        top_k: optional_i32_field(object, "top_k")?,
        min_p: optional_f32_field(object, "min_p")?,
        frequency_penalty: optional_f32_field(object, "frequency_penalty")?,
        presence_penalty: optional_f32_field(object, "presence_penalty")?,
        repetition_penalty: optional_f32_field(object, "repetition_penalty")?,
        stop_token_id: optional_i32_field(object, "stop_token_id")?,
        stop,
        n: optional_i32_field(object, "n")?,
        best_of: optional_i32_field(object, "best_of")?,
        seed,
        stop_token_ids,
        ignore_eos: optional_bool_field(object, "ignore_eos")?,
    }))
}

fn proto_disaggregated_params_from_json(
    payload: &serde_json::Map<String, Value>,
) -> Result<Option<ProtoDisaggregatedParams>, tonic::Status> {
    let has_flat = payload.contains_key("bootstrap_host")
        || payload.contains_key("bootstrap_port")
        || payload.contains_key("bootstrap_room");
    let object = if has_flat {
        payload
    } else {
        let Some(value) = payload.get("disaggregated_params") else {
            return Ok(None);
        };
        value
            .as_object()
            .ok_or_else(|| grpc_generate_status("disaggregated_params must be an object"))?
    };

    let bootstrap_host = object
        .get("bootstrap_host")
        .and_then(Value::as_str)
        .ok_or_else(|| grpc_generate_status("bootstrap_host must be a string"))?
        .to_string();
    let bootstrap_port = object
        .get("bootstrap_port")
        .ok_or_else(|| grpc_generate_status("bootstrap_port is required"))
        .and_then(|value| value_as_u32(value, "bootstrap_port"))?;
    let bootstrap_room = object
        .get("bootstrap_room")
        .and_then(Value::as_u64)
        .ok_or_else(|| grpc_generate_status("bootstrap_room must be an unsigned integer"))?;

    Ok(Some(ProtoDisaggregatedParams {
        bootstrap_host,
        bootstrap_port,
        bootstrap_room,
    }))
}

fn proto_text_generate_request_from_json(
    body: &[u8],
) -> Result<ProtoTextGenerateRequest, tonic::Status> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|e| grpc_generate_status(format!("invalid /generate JSON: {e}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| grpc_generate_status("/generate body must be a JSON object"))?;
    let text = object
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| grpc_generate_status("missing text"))?
        .to_string();

    Ok(ProtoTextGenerateRequest {
        text,
        sampling_params: proto_sampling_params_from_json(object)?,
        options: Some(ProtoRequestOptions {
            request_id: optional_string_field(object, "request_id")?,
            stream: optional_bool_field(object, "stream")?.unwrap_or(false),
            data_parallel_rank: optional_i32_field(object, "data_parallel_rank")?
                .unwrap_or_default(),
            trace_headers: Default::default(),
        }),
        disaggregated_params: proto_disaggregated_params_from_json(object)?,
    })
}

fn proto_generate_response_to_json(
    response: ProtoGenerateResponse,
) -> Result<Vec<u8>, anyhow::Error> {
    let body = match response.body {
        Some(ProtoGenerateResponseBody::Chunk(chunk)) => json!({
            "request_id": response.request_id,
            "text": chunk.text,
            "output_ids": chunk.token_ids,
            "usage": {
                "prompt_tokens": chunk.prompt_tokens,
                "completion_tokens": chunk.completion_tokens,
                "cached_tokens": chunk.cached_tokens,
            }
        }),
        Some(ProtoGenerateResponseBody::Complete(complete)) => json!({
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
        Some(ProtoGenerateResponseBody::Error(error)) => json!({
            "error": {
                "message": error.message,
                "code": error.code,
            }
        }),
        None => json!({
            "error": {
                "message": "gRPC GenerateResponse returned no body",
            }
        }),
    };

    serde_json::to_vec(&body).context("serialize generate JSON")
}

#[derive(Debug)]
pub struct Proxy {
    pub client: Client,
    /// Wall-clock timeout applied to non-streaming upstream requests. Streaming
    /// requests deliberately do not use this (long generations are valid).
    pub request_timeout: Duration,
}

impl Proxy {
    /// Build a proxy. `request_timeout` is the per-request wall-clock budget for
    /// non-streaming forwards. Connect timeout is hard-coded to 5 s — even a
    /// streaming request fails fast at TCP setup if the worker is unreachable.
    pub fn new(request_timeout: Duration) -> Result<Self, anyhow::Error> {
        let client = Client::builder()
            .pool_max_idle_per_host(64)
            .connect_timeout(Duration::from_secs(5))
            .build()
            .context("build reqwest client")?;
        Ok(Self {
            client,
            request_timeout,
        })
    }

    /// Classify a reqwest error into the right `ApiError` variant, given an
    /// explicit worker URL. Called from the breaker-gated `forward_*_to`
    /// methods, which carry per-request worker URLs (not a single proxy-level
    /// URL).
    ///
    /// Walks the full source chain to detect timeouts, because reqwest wraps
    /// hyper which wraps `std::io::Error` — a top-level `is_timeout()` check
    /// misses both the wrapped reqwest timeout and the `io::ErrorKind::TimedOut`
    /// cases.
    fn classify_reqwest_error_for(worker: Url, e: reqwest::Error, path: &str) -> ApiError {
        let source = anyhow::Error::new(e).context(format!("worker {worker}: post {path}"));
        let is_timeout = source.chain().any(|c| {
            c.downcast_ref::<reqwest::Error>()
                .is_some_and(|r| r.is_timeout())
        }) || source.chain().any(|c| {
            c.downcast_ref::<std::io::Error>()
                .is_some_and(|io| io.kind() == std::io::ErrorKind::TimedOut)
        });
        if is_timeout {
            ApiError::UpstreamTimeout { worker }
        } else {
            ApiError::UpstreamUnreachable { worker, source }
        }
    }

    /// Breaker-gated JSON POST: checks `breaker.allow()` first, records
    /// success/failure based on response status, and returns
    /// `ApiError::BreakerOpen` immediately when the breaker is Open.
    ///
    /// `worker_url` is the discovery-emitted worker URL string. It's parsed
    /// to [`reqwest::Url`] internally so we can use [`Url::join`] for clean
    /// path concatenation (no double-slash) and pass a typed URL to the
    /// split error variants (`UpstreamUnreachable` / `UpstreamTimeout` /
    /// `UpstreamStatus`).
    pub async fn forward_json_to(
        &self,
        worker_url: &str,
        breaker: &CircuitBreaker,
        path: &str,
        headers: &HeaderMap,
        body: Bytes,
    ) -> Result<Response<Body>, ApiError> {
        if !breaker.allow() {
            return Err(ApiError::BreakerOpen {
                worker: worker_url.to_string(),
            });
        }
        if is_grpc_worker_url(worker_url) {
            return self
                .forward_grpc_json_to(worker_url, breaker, path, body)
                .await;
        }
        let worker_url = parse_worker_url(worker_url, breaker)?;
        let url = worker_url.join(path).map_err(|e| {
            ApiError::Internal(anyhow::Error::new(e).context(format!("join worker path {path}")))
        })?;
        let mut req = self.client.post(url.clone()).body(body);
        for (k, v) in headers {
            if should_forward_request_header(k) {
                req = req.header(k, v);
            }
        }
        req = req
            .header("content-type", "application/json")
            .timeout(self.request_timeout);
        let resp = req.send().await.map_err(|e| {
            breaker.record_failure();
            Self::classify_reqwest_error_for(worker_url.clone(), e, path)
        })?;
        let status = resp.status();
        // Defer breaker recording until after the body completes — a
        // worker that returns 2xx headers and then drops mid-body is
        // still failing the request, and crediting it as healthy lets
        // a misbehaving worker stay eligible. For 5xx the early bail is
        // safe (no body to consume meaningfully), but we still wait
        // until after the read attempt to record exactly once.
        let bytes = match resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    upstream = %url,
                    status = %status,
                    error = ?e,
                    "upstream dropped connection mid-body",
                );
                breaker.record_failure();
                return Err(ApiError::UpstreamStatus { status });
            }
        };
        if status.is_server_error() {
            breaker.record_failure();
        } else {
            breaker.record_success();
        }
        let mut out = Response::new(Body::from(bytes));
        *out.status_mut() = status;
        out.headers_mut().insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("application/json"),
        );
        Ok(out)
    }

    async fn forward_grpc_json_to(
        &self,
        worker_url: &str,
        breaker: &CircuitBreaker,
        path: &str,
        body: Bytes,
    ) -> Result<Response<Body>, ApiError> {
        if path == "/generate" {
            return self
                .forward_grpc_generate_json_to(worker_url, breaker, body)
                .await;
        }
        if path != "/v1/chat/completions" && path != "/v1/completions" {
            breaker.record_failure();
            return Err(ApiError::WorkerMisconfigured {
                worker: worker_url.to_string(),
                source: anyhow::anyhow!("gRPC worker transport does not support path {path}"),
            });
        }

        let parsed = parse_worker_url(worker_url, breaker)?;
        let endpoint = grpc_endpoint_from_worker_url(&parsed).ok_or_else(|| {
            breaker.record_failure();
            ApiError::WorkerMisconfigured {
                worker: worker_url.to_string(),
                source: anyhow::anyhow!("unsupported gRPC worker URL scheme {}", parsed.scheme()),
            }
        })?;

        let fut = async {
            let channel = tonic::transport::Endpoint::from_shared(endpoint.clone())
                .map_err(|e| {
                    GrpcForwardError::Transport(anyhow::anyhow!(
                        "invalid gRPC endpoint {endpoint}: {e}"
                    ))
                })?
                .connect_timeout(self.request_timeout)
                .timeout(self.request_timeout)
                .connect()
                .await
                .map_err(|e| {
                    GrpcForwardError::Transport(anyhow::anyhow!(
                        "connect gRPC worker {endpoint}: {e}"
                    ))
                })?;
            let mut client = SglangServiceClient::new(channel);
            let request = OpenAiJsonRequest {
                json: body.to_vec(),
                options: None,
            };
            let mut stream = match path {
                "/v1/chat/completions" => client
                    .chat_complete(GrpcRequest::new(request))
                    .await
                    .map_err(GrpcForwardError::Status)?
                    .into_inner(),
                "/v1/completions" => client
                    .complete(GrpcRequest::new(request))
                    .await
                    .map_err(GrpcForwardError::Status)?
                    .into_inner(),
                _ => unreachable!("validated gRPC JSON path"),
            };
            let first = stream
                .message()
                .await
                .map_err(GrpcForwardError::Status)?
                .ok_or_else(|| {
                    GrpcForwardError::Status(tonic::Status::internal(
                        "gRPC OpenAI JSON RPC returned no response",
                    ))
                })?;
            Ok::<_, GrpcForwardError>(first.json)
        };

        let json = match tokio::time::timeout(self.request_timeout, fut).await {
            Ok(Ok(json)) => json,
            Ok(Err(GrpcForwardError::Status(status))) => {
                let response = grpc_status_response(status);
                if response.status().is_server_error() {
                    breaker.record_failure();
                } else {
                    breaker.record_success();
                }
                return Ok(response);
            }
            Ok(Err(GrpcForwardError::Transport(source))) => {
                breaker.record_failure();
                return Err(ApiError::UpstreamUnreachable {
                    worker: parsed,
                    source,
                });
            }
            Err(_) => {
                breaker.record_failure();
                return Err(ApiError::UpstreamTimeout { worker: parsed });
            }
        };

        breaker.record_success();
        let mut out = Response::new(Body::from(json));
        out.headers_mut().insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("application/json"),
        );
        Ok(out)
    }

    async fn forward_grpc_generate_json_to(
        &self,
        worker_url: &str,
        breaker: &CircuitBreaker,
        body: Bytes,
    ) -> Result<Response<Body>, ApiError> {
        let parsed = parse_worker_url(worker_url, breaker)?;
        let endpoint = grpc_endpoint_from_worker_url(&parsed).ok_or_else(|| {
            breaker.record_failure();
            ApiError::WorkerMisconfigured {
                worker: worker_url.to_string(),
                source: anyhow::anyhow!("unsupported gRPC worker URL scheme {}", parsed.scheme()),
            }
        })?;

        let fut = async {
            let request =
                proto_text_generate_request_from_json(&body).map_err(GrpcForwardError::Status)?;
            let channel = tonic::transport::Endpoint::from_shared(endpoint.clone())
                .map_err(|e| {
                    GrpcForwardError::Transport(anyhow::anyhow!(
                        "invalid gRPC endpoint {endpoint}: {e}"
                    ))
                })?
                .connect_timeout(self.request_timeout)
                .timeout(self.request_timeout)
                .connect()
                .await
                .map_err(|e| {
                    GrpcForwardError::Transport(anyhow::anyhow!(
                        "connect gRPC worker {endpoint}: {e}"
                    ))
                })?;
            let mut client = SglangServiceClient::new(channel);
            let mut stream = client
                .text_generate(GrpcRequest::new(request))
                .await
                .map_err(GrpcForwardError::Status)?
                .into_inner();
            let first = stream
                .message()
                .await
                .map_err(GrpcForwardError::Status)?
                .ok_or_else(|| {
                    GrpcForwardError::Status(tonic::Status::internal(
                        "gRPC TextGenerate returned no response",
                    ))
                })?;
            proto_generate_response_to_json(first).map_err(GrpcForwardError::Transport)
        };

        let json = match tokio::time::timeout(self.request_timeout, fut).await {
            Ok(Ok(json)) => json,
            Ok(Err(GrpcForwardError::Status(status))) => {
                let response = grpc_status_response(status);
                if response.status().is_server_error() {
                    breaker.record_failure();
                } else {
                    breaker.record_success();
                }
                return Ok(response);
            }
            Ok(Err(GrpcForwardError::Transport(source))) => {
                breaker.record_failure();
                return Err(ApiError::UpstreamUnreachable {
                    worker: parsed,
                    source,
                });
            }
            Err(_) => {
                breaker.record_failure();
                return Err(ApiError::UpstreamTimeout { worker: parsed });
            }
        };

        breaker.record_success();
        let mut out = Response::new(Body::from(json));
        out.headers_mut().insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("application/json"),
        );
        Ok(out)
    }

    /// Breaker-gated streaming POST: checks `breaker.allow()` first, records
    /// success/failure, and returns `ApiError::BreakerOpen` when Open.
    ///
    /// `stream_guards` — when `Some`, the value is threaded into the SSE
    /// pump task and held for the entire body lifetime (headers → last byte
    /// / client disconnect).  The proxy does not inspect the boxed value; it
    /// relies entirely on `Drop` semantics, so callers typically pack
    /// `(LoadGuard, ActiveLoadGuard)` here. This keeps both the per-worker
    /// `active_requests` counter and the per-request active-load entry alive
    /// for the full streaming lifetime — without which a long-running SSE
    /// response would under-report load.
    pub async fn forward_streaming_to(
        &self,
        worker_url: &str,
        breaker: &Arc<CircuitBreaker>,
        path: &str,
        headers: &HeaderMap,
        body: Bytes,
        stream_guards: Option<Box<dyn Send + 'static>>,
    ) -> Result<Response<Body>, ApiError> {
        if !breaker.allow() {
            return Err(ApiError::BreakerOpen {
                worker: worker_url.to_string(),
            });
        }
        if is_grpc_worker_url(worker_url) {
            return self
                .forward_grpc_streaming_to(worker_url, breaker, path, body, stream_guards)
                .await;
        }
        let worker_url = parse_worker_url(worker_url, breaker)?;
        let url = worker_url.join(path).map_err(|e| {
            ApiError::Internal(anyhow::Error::new(e).context(format!("join worker path {path}")))
        })?;
        let mut req = self.client.post(url.clone()).body(body);
        for (k, v) in headers {
            if should_forward_request_header(k) {
                req = req.header(k, v);
            }
        }
        req = req
            .header("content-type", "application/json")
            .header("accept", "text/event-stream");
        let resp = req.send().await.map_err(|e| {
            breaker.record_failure();
            Self::classify_reqwest_error_for(worker_url.clone(), e, path)
        })?;
        let status = resp.status();
        let upstream_ct = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/json")
            .to_string();
        let content_type = if status.is_success() {
            "text/event-stream".to_string()
        } else {
            upstream_ct
        };
        // Breaker recording is deferred to the pump's completion hook so
        // an upstream that returns 2xx headers and then drops mid-stream
        // is recorded as a failure. For 5xx headers we record_failure
        // up front and skip the pump hook (the body we surface is the
        // error response — its stream completing is not a worker win).
        let on_complete: Option<Box<dyn FnOnce(bool) + Send + 'static>> =
            if status.is_server_error() {
                breaker.record_failure();
                None
            } else {
                let breaker_for_hook = Arc::clone(breaker);
                Some(Box::new(move |ok| {
                    if ok {
                        breaker_for_hook.record_success();
                    } else {
                        breaker_for_hook.record_failure();
                    }
                }))
            };
        let body = sse::bytes_stream_to_body(resp.bytes_stream(), stream_guards, on_complete);
        let mut out = Response::new(body);
        *out.status_mut() = status;
        out.headers_mut().insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_str(&content_type)
                .unwrap_or_else(|_| HeaderValue::from_static("application/json")),
        );
        Ok(out)
    }

    async fn forward_grpc_streaming_to(
        &self,
        worker_url: &str,
        breaker: &Arc<CircuitBreaker>,
        path: &str,
        body: Bytes,
        stream_guards: Option<Box<dyn Send + 'static>>,
    ) -> Result<Response<Body>, ApiError> {
        if path == "/generate" {
            return self
                .forward_grpc_generate_streaming_to(worker_url, breaker, body, stream_guards)
                .await;
        }
        if path != "/v1/chat/completions" && path != "/v1/completions" {
            breaker.record_failure();
            return Err(ApiError::WorkerMisconfigured {
                worker: worker_url.to_string(),
                source: anyhow::anyhow!("gRPC worker transport does not support path {path}"),
            });
        }

        let parsed = parse_worker_url(worker_url, breaker)?;
        let endpoint = grpc_endpoint_from_worker_url(&parsed).ok_or_else(|| {
            breaker.record_failure();
            ApiError::WorkerMisconfigured {
                worker: worker_url.to_string(),
                source: anyhow::anyhow!("unsupported gRPC worker URL scheme {}", parsed.scheme()),
            }
        })?;

        let fut = async {
            let channel = tonic::transport::Endpoint::from_shared(endpoint.clone())
                .map_err(|e| {
                    GrpcForwardError::Transport(anyhow::anyhow!(
                        "invalid gRPC endpoint {endpoint}: {e}"
                    ))
                })?
                .connect_timeout(self.request_timeout)
                .connect()
                .await
                .map_err(|e| {
                    GrpcForwardError::Transport(anyhow::anyhow!(
                        "connect gRPC worker {endpoint}: {e}"
                    ))
                })?;
            let mut client = SglangServiceClient::new(channel);
            let request = OpenAiJsonRequest {
                json: body.to_vec(),
                options: None,
            };
            match path {
                "/v1/chat/completions" => client
                    .chat_complete(GrpcRequest::new(request))
                    .await
                    .map_err(GrpcForwardError::Status)
                    .map(|response| response.into_inner()),
                "/v1/completions" => client
                    .complete(GrpcRequest::new(request))
                    .await
                    .map_err(GrpcForwardError::Status)
                    .map(|response| response.into_inner()),
                _ => unreachable!("validated gRPC streaming path"),
            }
        };

        let stream = match tokio::time::timeout(self.request_timeout, fut).await {
            Ok(Ok(stream)) => stream,
            Ok(Err(GrpcForwardError::Status(status))) => {
                let response = grpc_status_response(status);
                if response.status().is_server_error() {
                    breaker.record_failure();
                } else {
                    breaker.record_success();
                }
                return Ok(response);
            }
            Ok(Err(GrpcForwardError::Transport(source))) => {
                breaker.record_failure();
                return Err(ApiError::UpstreamUnreachable {
                    worker: parsed,
                    source,
                });
            }
            Err(_) => {
                breaker.record_failure();
                return Err(ApiError::UpstreamTimeout { worker: parsed });
            }
        };

        let breaker_for_hook = Arc::clone(breaker);
        let on_complete: Option<Box<dyn FnOnce(bool) + Send + 'static>> =
            Some(Box::new(move |ok| {
                if ok {
                    breaker_for_hook.record_success();
                } else {
                    breaker_for_hook.record_failure();
                }
            }));
        let body = sse::json_bytes_stream_to_sse_body(
            stream.map(|item: Result<OpenAiJsonResponse, tonic::Status>| {
                item.map(|response| response.json)
            }),
            stream_guards,
            on_complete,
        );
        let mut out = Response::new(body);
        out.headers_mut().insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("text/event-stream"),
        );
        Ok(out)
    }

    async fn forward_grpc_generate_streaming_to(
        &self,
        worker_url: &str,
        breaker: &Arc<CircuitBreaker>,
        body: Bytes,
        stream_guards: Option<Box<dyn Send + 'static>>,
    ) -> Result<Response<Body>, ApiError> {
        let parsed = parse_worker_url(worker_url, breaker)?;
        let endpoint = grpc_endpoint_from_worker_url(&parsed).ok_or_else(|| {
            breaker.record_failure();
            ApiError::WorkerMisconfigured {
                worker: worker_url.to_string(),
                source: anyhow::anyhow!("unsupported gRPC worker URL scheme {}", parsed.scheme()),
            }
        })?;

        let fut = async {
            let request =
                proto_text_generate_request_from_json(&body).map_err(GrpcForwardError::Status)?;
            let channel = tonic::transport::Endpoint::from_shared(endpoint.clone())
                .map_err(|e| {
                    GrpcForwardError::Transport(anyhow::anyhow!(
                        "invalid gRPC endpoint {endpoint}: {e}"
                    ))
                })?
                .connect_timeout(self.request_timeout)
                .connect()
                .await
                .map_err(|e| {
                    GrpcForwardError::Transport(anyhow::anyhow!(
                        "connect gRPC worker {endpoint}: {e}"
                    ))
                })?;
            let mut client = SglangServiceClient::new(channel);
            client
                .text_generate(GrpcRequest::new(request))
                .await
                .map_err(GrpcForwardError::Status)
                .map(|response| response.into_inner())
        };

        let stream = match tokio::time::timeout(self.request_timeout, fut).await {
            Ok(Ok(stream)) => stream,
            Ok(Err(GrpcForwardError::Status(status))) => {
                let response = grpc_status_response(status);
                if response.status().is_server_error() {
                    breaker.record_failure();
                } else {
                    breaker.record_success();
                }
                return Ok(response);
            }
            Ok(Err(GrpcForwardError::Transport(source))) => {
                breaker.record_failure();
                return Err(ApiError::UpstreamUnreachable {
                    worker: parsed,
                    source,
                });
            }
            Err(_) => {
                breaker.record_failure();
                return Err(ApiError::UpstreamTimeout { worker: parsed });
            }
        };

        let breaker_for_hook = Arc::clone(breaker);
        let on_complete: Option<Box<dyn FnOnce(bool) + Send + 'static>> =
            Some(Box::new(move |ok| {
                if ok {
                    breaker_for_hook.record_success();
                } else {
                    breaker_for_hook.record_failure();
                }
            }));
        let body = sse::json_bytes_stream_to_sse_body(
            stream.map(|item: Result<ProtoGenerateResponse, tonic::Status>| {
                item.and_then(|response| {
                    proto_generate_response_to_json(response)
                        .map_err(|e| tonic::Status::internal(e.to_string()))
                })
            }),
            stream_guards,
            on_complete,
        );
        let mut out = Response::new(body);
        out.headers_mut().insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("text/event-stream"),
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn new_returns_result_not_panic() {
        let p = Proxy::new(Duration::from_secs(5)).unwrap();
        assert_eq!(p.request_timeout, Duration::from_secs(5));
    }
}
