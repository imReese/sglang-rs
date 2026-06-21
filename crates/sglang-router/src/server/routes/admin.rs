// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::{ModelId, WorkerMode};
use crate::policies::registry::{PdPoolResolver, PdPools, PdResolveError};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::workers::Worker;
use axum::body::{to_bytes, Body};
use axum::extract::State;
use axum::http::{HeaderMap, Response};
use axum::response::IntoResponse;
use axum::Json;
use bytes::Bytes;
use futures::future;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Debug, Deserialize)]
struct UpdateWeightsProbe {
    #[serde(default)]
    model: Option<String>,
    model_path: String,
    #[serde(default)]
    load_format: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UpdateWeightVersionProbe {
    #[serde(default)]
    model: Option<String>,
    new_version: String,
}

#[derive(Debug, Default, Deserialize)]
struct AdminModelProbe {
    #[serde(default)]
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AbortProbe {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    rid: Option<String>,
    #[serde(default)]
    request_id: Option<String>,
    #[serde(default)]
    abort_all: bool,
}

#[derive(Debug, Deserialize)]
struct ControlResponseBody {
    success: bool,
    #[serde(default)]
    message: String,
}

pub async fn flush_cache(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    let probe = parse_admin_model_probe(&body)?;
    let model = select_admin_model(&ctx, probe.model)?;
    let model_id = ModelId(model.clone());
    let workers = admin_control_workers(&ctx, &model_id, &model)?;
    let mut flushed_workers = 0usize;

    for worker in workers {
        let response = ctx
            .proxy
            .forward_json_to(
                &worker.url,
                &worker.breaker,
                "/flush_cache",
                &headers,
                body.clone(),
            )
            .await?;
        if !response.status().is_success() {
            return Ok(response);
        }
        flushed_workers += 1;
    }

    Ok(Json(serde_json::json!({
        "success": true,
        "message": format!("flushed cache on {flushed_workers} worker(s)"),
        "flushed_workers": flushed_workers,
        "model": model,
    }))
    .into_response())
}

pub async fn get_loads(State(ctx): State<Arc<AppContext>>) -> Result<Response<Body>, ApiError> {
    let workers = ctx.registry.workers();
    let total_workers = workers.len();
    let futures = workers.iter().map(|worker| {
        let proxy = Arc::clone(&ctx.proxy);
        let worker = Arc::clone(worker);
        async move {
            let load = match proxy.load_for_worker(&worker.url, &worker.breaker).await {
                Ok(load) => load,
                Err(error) => {
                    tracing::warn!(
                        worker = %worker.url,
                        error = %error,
                        "failed to read worker load",
                    );
                    -1
                }
            };
            serde_json::json!({
                "worker": worker.url,
                "worker_type": worker_type_json(worker.mode()),
                "load": load,
            })
        }
    });
    let loads = future::join_all(futures).await;
    let successful = loads
        .iter()
        .filter(|load| {
            load.get("load")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(-1)
                >= 0
        })
        .count();

    Ok(Json(serde_json::json!({
        "loads": loads,
        "total_workers": total_workers,
        "successful": successful,
        "failed": total_workers.saturating_sub(successful),
    }))
    .into_response())
}

pub async fn pause_generation(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    forward_generation_control(
        ctx,
        headers,
        body,
        "/pause_generation",
        "generation paused",
        "paused generation",
    )
    .await
}

pub async fn continue_generation(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    forward_generation_control(
        ctx,
        headers,
        body,
        "/continue_generation",
        "generation continued",
        "continued generation",
    )
    .await
}

pub async fn abort_request(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    let probe: AbortProbe = serde_json::from_slice(&body)
        .map_err(|_| ApiError::BadRequest("request body must be a JSON object".to_string()))?;
    let request_id = probe
        .rid
        .as_deref()
        .or(probe.request_id.as_deref())
        .unwrap_or_default();
    if !probe.abort_all && request_id.is_empty() {
        return Err(ApiError::BadRequest("rid must be non-empty".to_string()));
    }

    let model = select_admin_model(&ctx, probe.model)?;
    let model_id = ModelId(model.clone());
    let workers = admin_control_workers(&ctx, &model_id, &model)?;
    let mut affected_workers = 0usize;
    let mut aborted_workers = 0usize;
    let mut messages = Vec::new();

    for worker in workers {
        let response = ctx
            .proxy
            .forward_json_to(
                &worker.url,
                &worker.breaker,
                "/abort_request",
                &headers,
                body.clone(),
            )
            .await?;
        if !response.status().is_success() {
            return Ok(response);
        }

        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .map_err(|error| {
                ApiError::Internal(anyhow::anyhow!("read abort response body: {error}"))
            })?;
        let control: ControlResponseBody = serde_json::from_slice(&bytes).map_err(|error| {
            ApiError::Internal(anyhow::anyhow!("decode abort response body: {error}"))
        })?;
        affected_workers += 1;
        if control.success {
            aborted_workers += 1;
        }
        if !control.message.is_empty()
            && !messages.iter().any(|message| message == &control.message)
        {
            messages.push(control.message);
        }
    }

    let success = aborted_workers > 0;
    let message = if success {
        "request aborted".to_string()
    } else {
        messages
            .into_iter()
            .next()
            .unwrap_or_else(|| "request not found".to_string())
    };

    Ok(Json(serde_json::json!({
        "success": success,
        "message": message,
        "affected_workers": affected_workers,
        "aborted_workers": aborted_workers,
        "model": model,
        "abort_all": probe.abort_all,
        "rid": request_id,
    }))
    .into_response())
}

pub async fn start_profile(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    forward_generation_control(
        ctx,
        headers,
        body,
        "/start_profile",
        "profile started",
        "started profile",
    )
    .await
}

pub async fn stop_profile(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    forward_generation_control(
        ctx,
        headers,
        body,
        "/stop_profile",
        "profile stopped",
        "stopped profile",
    )
    .await
}

pub async fn update_weights_from_disk(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    let probe: UpdateWeightsProbe = serde_json::from_slice(&body)
        .map_err(|_| ApiError::BadRequest("request body must be a JSON object".to_string()))?;
    if probe.model_path.is_empty() {
        return Err(ApiError::BadRequest(
            "model_path must be non-empty".to_string(),
        ));
    }

    let model = select_admin_model(&ctx, probe.model)?;
    let model_id = ModelId(model.clone());
    let workers = admin_control_workers(&ctx, &model_id, &model)?;
    let mut updated_workers = 0usize;

    for worker in workers {
        let response = ctx
            .proxy
            .forward_json_to(
                &worker.url,
                &worker.breaker,
                "/update_weights_from_disk",
                &headers,
                body.clone(),
            )
            .await?;
        if !response.status().is_success() {
            return Ok(response);
        }
        updated_workers += 1;
    }

    Ok(Json(serde_json::json!({
        "success": true,
        "message": format!(
            "updated weights from {} on {updated_workers} worker(s)",
            probe.model_path
        ),
        "num_paused_requests": 0,
        "updated_workers": updated_workers,
        "model": model,
        "load_format": probe.load_format,
    }))
    .into_response())
}

pub async fn update_weight_version(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    let probe: UpdateWeightVersionProbe = serde_json::from_slice(&body)
        .map_err(|_| ApiError::BadRequest("request body must be a JSON object".to_string()))?;
    if probe.new_version.is_empty() {
        return Err(ApiError::BadRequest(
            "new_version must be non-empty".to_string(),
        ));
    }

    let model = select_admin_model(&ctx, probe.model)?;
    let model_id = ModelId(model.clone());
    let workers = admin_control_workers(&ctx, &model_id, &model)?;
    let mut updated_workers = 0usize;

    for worker in workers {
        let response = ctx
            .proxy
            .forward_json_to(
                &worker.url,
                &worker.breaker,
                "/update_weight_version",
                &headers,
                body.clone(),
            )
            .await?;
        if !response.status().is_success() {
            return Ok(response);
        }
        updated_workers += 1;
    }

    Ok(Json(serde_json::json!({
        "success": true,
        "message": format!("updated weight version to {} on {updated_workers} worker(s)", probe.new_version),
        "new_version": probe.new_version,
        "updated_workers": updated_workers,
        "model": model,
    }))
    .into_response())
}

async fn forward_generation_control(
    ctx: Arc<AppContext>,
    headers: HeaderMap,
    body: Bytes,
    upstream_path: &'static str,
    message: &'static str,
    action: &'static str,
) -> Result<Response<Body>, ApiError> {
    let probe = parse_admin_model_probe(&body)?;
    let model = select_admin_model(&ctx, probe.model)?;
    let model_id = ModelId(model.clone());
    let workers = admin_control_workers(&ctx, &model_id, &model)?;
    let mut affected_workers = 0usize;

    for worker in workers {
        let response = ctx
            .proxy
            .forward_json_to(
                &worker.url,
                &worker.breaker,
                upstream_path,
                &headers,
                body.clone(),
            )
            .await?;
        if !response.status().is_success() {
            return Ok(response);
        }
        affected_workers += 1;
    }

    Ok(Json(serde_json::json!({
        "success": true,
        "message": message,
        "affected_workers": affected_workers,
        "model": model,
        "action": action,
    }))
    .into_response())
}

fn worker_type_json(mode: WorkerMode) -> Option<&'static str> {
    match mode {
        WorkerMode::Plain => None,
        WorkerMode::Prefill => Some("prefill"),
        WorkerMode::Decode => Some("decode"),
    }
}

fn parse_admin_model_probe(body: &[u8]) -> Result<AdminModelProbe, ApiError> {
    if body.is_empty() {
        return Ok(AdminModelProbe::default());
    }
    serde_json::from_slice(body)
        .map_err(|_| ApiError::BadRequest("request body must be a JSON object".to_string()))
}

fn select_admin_model(ctx: &AppContext, requested: Option<String>) -> Result<String, ApiError> {
    if let Some(model) = requested {
        if ctx
            .config
            .models
            .iter()
            .any(|configured| configured.id == model)
        {
            return Ok(model);
        }
        return Err(ApiError::ModelNotFound(model));
    }

    match ctx.config.models.as_slice() {
        [model] => Ok(model.id.clone()),
        [] => Err(ApiError::BadRequest(
            "model is required because no models are configured".to_string(),
        )),
        _ => Err(ApiError::BadRequest(
            "model is required when multiple models are configured".to_string(),
        )),
    }
}

fn admin_control_workers(
    ctx: &AppContext,
    model_id: &ModelId,
    model: &str,
) -> Result<Vec<Arc<Worker>>, ApiError> {
    let resolver = PdPoolResolver::new(Arc::clone(&ctx.registry));
    match resolver.resolve(model_id) {
        Ok(PdPools::Plain { workers }) => Ok(workers),
        Ok(PdPools::Pd { prefill, decode }) => {
            if prefill.is_empty() {
                return Err(ApiError::NoPrefillWorkersAvailable {
                    model: model.to_string(),
                });
            }
            if decode.is_empty() {
                return Err(ApiError::NoDecodeWorkersAvailable {
                    model: model.to_string(),
                });
            }
            Ok(prefill.into_iter().chain(decode).collect())
        }
        Err(PdResolveError::NoHealthyWorkers) => Err(ApiError::NoHealthyWorkers {
            model: model.to_string(),
        }),
        Err(PdResolveError::NoPrefillWorkersAvailable) => {
            Err(ApiError::NoPrefillWorkersAvailable {
                model: model.to_string(),
            })
        }
        Err(PdResolveError::NoDecodeWorkersAvailable) => Err(ApiError::NoDecodeWorkersAvailable {
            model: model.to_string(),
        }),
    }
}
