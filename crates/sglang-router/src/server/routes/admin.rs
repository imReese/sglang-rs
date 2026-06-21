// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::ModelId;
use crate::policies::registry::{PdPoolResolver, PdPools, PdResolveError};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::workers::Worker;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Response};
use axum::response::IntoResponse;
use axum::Json;
use bytes::Bytes;
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
    let workers = admin_update_workers(&ctx, &model_id, &model)?;
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

fn admin_update_workers(
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
