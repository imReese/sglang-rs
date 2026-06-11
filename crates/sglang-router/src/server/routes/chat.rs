// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::{ModelId, WorkerMode};
use crate::policies::registry::{PdPoolResolver, PdPools, PdResolveError};
use crate::policies::SelectionContext;
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::metrics::{RequestOutcome, StaleRequestOutcome, WorkerModeLabel};
use crate::server::routes::pd::{PdDispatchPlan, RouteDispatchPlan, RouteWorkerSelection};
use crate::tokenizer::adapter;
use crate::workers::LoadGuard;
use axum::body::{to_bytes, Body};
use axum::extract::State;
use axum::http::{HeaderMap, Response};
use axum::response::IntoResponse;
use axum::Json;
use bytes::Bytes;
use serde::de::IgnoredAny;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Coarse char-count → token-count divisor used to estimate prefill load
/// from the request body when no real tokenizer count is available. Four
/// bytes per token is the standard SGLang upstream estimate; it
/// overcounts ASCII and undercounts CJK but stays within an order of
/// magnitude of the real token count, which is plenty for load
/// scoring. The active-load counters' role is relative ordering across
/// workers — not absolute accuracy — so the estimate is fit for
/// purpose.
const CHARS_PER_TOKEN_ESTIMATE: usize = 4;

/// Per-route body-size cap on `/v1/chat/completions`. 1 MiB is comfortable
/// for normal chat traffic (a 200 k-token context tokenized as JSON is well
/// under this) while preventing a hostile client from forcing the router to
/// heap-allocate hundreds of MiB before forwarding. The cap is wired in
/// `crate::server::app::build_router` as a route-level `DefaultBodyLimit`
/// layer; axum's `Bytes` extractor enforces it and returns 413
/// PAYLOAD_TOO_LARGE before this handler runs.
pub const MAX_CHAT_BODY_BYTES: usize = 1 << 20;

/// Minimal probe over the request body — we only need the `stream` field
/// and the `model` field to decide between buffered vs SSE forwarding and
/// to select a worker. Deserializing into this struct (vs `serde_json::Value`)
/// does two things:
///
/// 1. Avoids the per-field heap allocation of `Value` for a 1 MiB body.
/// 2. Pins the contract: the body MUST be a JSON object. Degenerate
///    shapes (`null`, `[]`, `"hi"`) fail at this step rather than being
///    silently forwarded with `stream=false`.
///
/// All other fields are ignored — the worker is authoritative for the
/// full request schema.
#[derive(Debug, Deserialize)]
struct RequestProbe {
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    model: Option<String>,
}

/// POST /v1/chat/completions — parse model from body, select a healthy
/// worker via the per-model policy, then proxy the request. If the
/// request opts into streaming (`stream: true`), we pipe SSE bytes back;
/// otherwise buffer.
pub async fn chat_completions(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    routed_generation(
        ctx,
        headers,
        body,
        "/v1/chat/completions",
        ModelSelection::RequireBodyModel,
        ResponsePostprocess::None,
    )
    .await
}

/// POST /v1/completions — same OpenAI-compatible routing path as chat
/// completions, but forwarded to the worker's completions endpoint/RPC.
pub async fn completions(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    routed_generation(
        ctx,
        headers,
        body,
        "/v1/completions",
        ModelSelection::RequireBodyModel,
        ResponsePostprocess::None,
    )
    .await
}

/// POST /generate — SGLang-native text generation. The native endpoint
/// historically omits `model`, so single-model router configs infer the
/// target model from config while multi-model configs require an explicit
/// `model` field to avoid ambiguous dispatch.
pub async fn generate(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    routed_generation(
        ctx,
        headers,
        body,
        "/generate",
        ModelSelection::BodyModelOrSingleConfigured,
        ResponsePostprocess::None,
    )
    .await
}

/// POST /v1/rerank — route reranker requests through the same worker
/// selection and PD dual-dispatch path as sgl-model-gateway's HTTP
/// PDRouter. Rerank requests are non-streaming and carry `model` in the
/// request body.
pub async fn rerank(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    let rerank_options = parse_rerank_response_options(&body)?;
    routed_generation(
        ctx,
        headers,
        body,
        "/v1/rerank",
        ModelSelection::RequireBodyModel,
        ResponsePostprocess::Rerank(rerank_options),
    )
    .await
}

/// POST /v1/embeddings — plain-mode gateway-compatible proxy. SGLang's
/// HTTP PD router rejects embeddings in PD mode, so this route only
/// dispatches to plain workers.
pub async fn embeddings(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    routed_plain_json(
        ctx,
        headers,
        body,
        "/v1/embeddings",
        "PD mode does not support /v1/embeddings",
    )
    .await
}

/// POST /v1/classify — plain-mode gateway-compatible proxy. As with
/// embeddings, the upstream HTTP PD router treats classify as
/// unsupported for PD deployments.
pub async fn classify(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    routed_plain_json(
        ctx,
        headers,
        body,
        "/v1/classify",
        "PD mode does not support /v1/classify",
    )
    .await
}

#[derive(Clone, Copy)]
enum ModelSelection {
    RequireBodyModel,
    BodyModelOrSingleConfigured,
}

#[derive(Clone, Debug)]
enum ResponsePostprocess {
    None,
    Rerank(RerankResponseOptions),
}

#[derive(Clone, Debug, Deserialize)]
struct RerankResponseOptions {
    #[serde(default = "unknown_model")]
    model: String,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default = "default_true")]
    return_documents: bool,
    #[serde(default)]
    rid: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
struct RerankResult {
    score: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    document: Option<String>,
    index: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    meta_info: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Serialize)]
struct RerankResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<serde_json::Value>,
    object: &'static str,
    created: u64,
    model: String,
    results: Vec<RerankResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<serde_json::Value>,
}

async fn routed_plain_json(
    ctx: Arc<AppContext>,
    headers: HeaderMap,
    body: Bytes,
    upstream_path: &'static str,
    pd_unsupported_message: &'static str,
) -> Result<Response<Body>, ApiError> {
    let probe = parse_probe(&body)?;
    let model_str = select_model(&ctx, probe.model, ModelSelection::RequireBodyModel)?;
    let model_id = ModelId(model_str.clone());

    let resolver = PdPoolResolver::new(Arc::clone(&ctx.registry));
    let workers = match resolver.resolve(&model_id) {
        Ok(PdPools::Plain { workers }) => workers,
        Ok(PdPools::Pd { .. }) => {
            return Err(ApiError::BadRequest(pd_unsupported_message.to_string()));
        }
        Err(PdResolveError::NoHealthyWorkers) => {
            return Err(ApiError::NoHealthyWorkers { model: model_str });
        }
        Err(PdResolveError::NoPrefillWorkersAvailable) => {
            return Err(ApiError::NoPrefillWorkersAvailable { model: model_str });
        }
        Err(PdResolveError::NoDecodeWorkersAvailable) => {
            return Err(ApiError::NoDecodeWorkersAvailable { model: model_str });
        }
    };

    let policy = ctx
        .policies
        .get(&model_id)
        .ok_or_else(|| ApiError::ModelNotFound(model_str.clone()))?;
    let selection_ctx = SelectionContext::new(&model_id, Some(&body));
    let worker =
        policy
            .select(&workers, &selection_ctx)
            .ok_or_else(|| ApiError::PolicySelectionFailed {
                model: model_str.clone(),
            })?;

    let guard = worker.load_guard();
    let active_load = estimate_prefill_tokens(&ctx, &model_id, &body);
    let active_guard =
        ctx.active_load
            .register(worker.id.clone(), worker.url.clone(), active_load, 0);
    let stale_token = active_guard.cancel_token().clone();
    let metrics_worker_url = worker.url.clone();
    let metrics_model = model_str.clone();

    let result = {
        let _holds: (LoadGuard, _) = (guard, active_guard);
        let fetch =
            ctx.proxy
                .forward_json_to(&worker.url, &worker.breaker, upstream_path, &headers, body);
        tokio::select! {
            biased;
            r = fetch => r,
            _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
        }
    };

    let outcome = match &result {
        Ok(_) => RequestOutcome::Success,
        Err(ApiError::StaleRequestExpired { .. }) => {
            ctx.metrics
                .record_stale_request(StaleRequestOutcome::Expired);
            RequestOutcome::Cancelled
        }
        Err(_) => RequestOutcome::Error,
    };
    ctx.metrics.record_request(
        &metrics_worker_url,
        &metrics_model,
        WorkerModeLabel::Plain,
        outcome,
    );

    result
}

async fn routed_generation(
    ctx: Arc<AppContext>,
    headers: HeaderMap,
    body: Bytes,
    upstream_path: &'static str,
    model_selection: ModelSelection,
    postprocess: ResponsePostprocess,
) -> Result<Response<Body>, ApiError> {
    let probe = parse_probe(&body)?;
    let streaming = probe.stream.unwrap_or(false);
    let model_str = select_model(&ctx, probe.model, model_selection)?;
    let model_id = ModelId(model_str.clone());

    // PD pool isolation: for PD-mode deployments, prefill traffic
    // selects from the prefill pool only. Plain-mode deployments fall
    // through to the full candidate set. Partial-failure errors
    // (`no_prefill_workers_available`) are surfaced as 503 with a
    // distinct error code so operators can alert independently.
    let resolver = PdPoolResolver::new(Arc::clone(&ctx.registry));
    let workers = resolver
        .prefill_candidates(&model_id)
        .map_err(|e| match e {
            PdResolveError::NoHealthyWorkers => ApiError::NoHealthyWorkers {
                model: model_str.clone(),
            },
            PdResolveError::NoPrefillWorkersAvailable => ApiError::NoPrefillWorkersAvailable {
                model: model_str.clone(),
            },
            PdResolveError::NoDecodeWorkersAvailable => ApiError::NoDecodeWorkersAvailable {
                model: model_str.clone(),
            },
        })?;

    let policy = ctx
        .policies
        .get(&model_id)
        .ok_or_else(|| ApiError::ModelNotFound(model_str.clone()))?;
    let selection_ctx = SelectionContext::new(&model_id, Some(&body));
    let worker =
        policy
            .select(&workers, &selection_ctx)
            .ok_or_else(|| ApiError::PolicySelectionFailed {
                model: model_str.clone(),
            })?;

    // Route-local worker selection. This mirrors sgl-model-gateway's
    // `WorkerSelection::{Single, Dual}` boundary: the policy picks a
    // primary worker, then PD prefill selections are expanded into a
    // prefill/decode pair before request execution.
    //
    // Plain-mode workers skip pair resolution entirely (no decode peer
    // to find). PD-mode requests that fail to resolve a
    // decode peer (`NoDecodeWorkersAvailable`) bubble up as 503 so
    // operators can alert on prefill-vs-decode pool imbalance.
    let selection = RouteWorkerSelection::from_selected_worker(&resolver, &model_id, worker)
        .map_err(|e| match e {
            PdResolveError::NoHealthyWorkers => ApiError::NoHealthyWorkers {
                model: model_str.clone(),
            },
            PdResolveError::NoDecodeWorkersAvailable => ApiError::NoDecodeWorkersAvailable {
                model: model_str.clone(),
            },
            PdResolveError::NoPrefillWorkersAvailable => ApiError::NoPrefillWorkersAvailable {
                model: model_str.clone(),
            },
        })?;
    let worker = Arc::clone(selection.primary_worker());
    let mut pd_dispatch_plan: Option<PdDispatchPlan> = None;

    // Per-worker `active_requests` guard. The `ActiveLoadGuard` below
    // sits beside this one: both track in-flight load, but the
    // ActiveLoadGuard entry is per-request (with timeout-based janitor)
    // while the worker-scoped counter is what the cache-aware policy
    // reads. Both must drop at the same time — when the response stream
    // ends, the client disconnects, or the handler returns an error. In
    // PD mode the pair moves into the spawned prefill task so prefill
    // load is tracked for the full duration of the KV transfer; in plain
    // mode the pair stays in this handler. Decode-load contribution is
    // 0 here: the active-load registry's decode axis is reserved for a
    // future decode-side scheduler — current decode selection is
    // host-affinity only.
    let guard = worker.load_guard();
    let prefill_load = estimate_prefill_tokens(&ctx, &model_id, &body);
    let active_guard =
        ctx.active_load
            .register(worker.id.clone(), worker.url.clone(), prefill_load, 0);
    // Snapshot the stale-request cancel token BEFORE moving the guard
    // into the spawned prefill task / streaming pump / response future.
    // The token is cheap to clone (it's an `Arc<...>` internally) and
    // the chat handler races the client-facing fetch against
    // `token.cancelled()` to surface a 504 `stale_request_expired` if
    // the janitor expires the request mid-flight.
    let stale_token = active_guard.cancel_token().clone();

    // Snapshot the labels we need for metrics BEFORE moving the worker
    // / model_str values into the per-branch fetch futures.
    let metrics_worker_url = worker.url.clone();
    let metrics_mode = match worker.mode() {
        WorkerMode::Prefill => WorkerModeLabel::Prefill,
        WorkerMode::Decode => WorkerModeLabel::Decode,
        WorkerMode::Plain => WorkerModeLabel::Plain,
    };
    let metrics_model = model_str.clone();
    let dispatch_plan = RouteDispatchPlan::prepare(selection, upstream_path, headers, body)?;

    let result = match dispatch_plan {
        RouteDispatchPlan::Pd {
            request: execution_request,
            headers,
        } => {
            // PD-disagg dispatch (Pattern B — spawn prefill, await decode).
            //
            // SGLang's HTTP-mode disagg-prefill requires three flat
            // top-level fields on the request body: `bootstrap_host`,
            // `bootstrap_port` (the prefill worker's bootstrap-server
            // address) and `bootstrap_room` (a per-request 63-bit u64 ID
            // used by both sides to pair up the KV transfer). We inject
            // these here and fan the same modified body to both the
            // prefill and decode workers concurrently.
            //
            // **Why spawn-and-forget for prefill instead of
            // `tokio::join!`?** All three peer SGLang-HTTP-PD routers
            // (Dynamo / llm-d / aibrix) converged on this shape: the
            // prefill request must outlive the client connection because
            // tying prefill to the client future opens a cancel-race
            // window where the engine's NIXL RPC teardown can leak KV
            // block refs (NVBugs 5969206 in Dynamo). The detached task
            // also keeps the LoadGuard + ActiveLoadGuard alive for the full
            // prefill duration — KV transfer can run for tens of seconds
            // even when the client gave up.
            //
            // No watchdog for fail-fast on prefill 5xx: llm-d / aibrix both
            // ship without one. On prefill failure the client experiences
            // the SGLang decode-side bootstrap_room timeout (~30–60 s by
            // default) instead of an immediate 502. A follow-up can wire a
            // `tokio::sync::watch` channel if telemetry shows it matters.
            //
            // **Scope of the "detached" guarantee.** The spawn protects
            // against client disconnect — the handler future being dropped
            // does NOT cancel the prefill HTTP request. It does NOT protect
            // against router shutdown: when `AppContext` tears down, the
            // tokio runtime cancels all unfinished tasks including this
            // one. A future follow-up could thread a `TaskTracker` /
            // `JoinSet` through `AppContext` for graceful shutdown drain;
            // the current implementation ships without one (matching SMG's
            // shutdown behaviour).
            let bootstrap_room = execution_request.bootstrap_room();
            let injected_body = execution_request.body().clone();
            pd_dispatch_plan = Some(execution_request.dispatch_plan().clone());

            let prefill_worker = execution_request.prefill();
            let prefill_url = prefill_worker.url.clone();
            let prefill_breaker = Arc::clone(&prefill_worker.breaker);
            let prefill_headers = headers.clone();
            let prefill_body = injected_body.clone();
            let prefill_proxy = Arc::clone(&ctx.proxy);
            let prefill_holds: (LoadGuard, _) = (guard, active_guard);
            tokio::spawn(async move {
                // The tuple binding extends both guards' lifetime to the
                // end of this async block, which lasts until the prefill
                // HTTP request returns (success / error / engine-side
                // bootstrap_room timeout). The result is logged and
                // swallowed — no channel back to the client. See the big
                // comment above for the rationale.
                let _hold = prefill_holds;
                match prefill_proxy
                    .forward_json_to(
                        &prefill_url,
                        &prefill_breaker,
                        upstream_path,
                        &prefill_headers,
                        prefill_body,
                    )
                    .await
                {
                    Ok(_) => tracing::debug!(
                        prefill_url = %prefill_url,
                        bootstrap_room,
                        "prefill side completed",
                    ),
                    Err(e) => tracing::warn!(
                        prefill_url = %prefill_url,
                        bootstrap_room,
                        error = %e,
                        "prefill request failed; decode will time out on bootstrap_room",
                    ),
                }
            });

            // Synchronously await the decode worker. Its response is what
            // the client sees. The decode side gets its own LoadGuard so
            // per-worker `active_requests` reflects decode-pool load for
            // cache-aware-zmq decisions on the decode side.
            let decode_worker = execution_request.decode();
            let decode_guard = decode_worker.load_guard();
            if streaming {
                let stream_guards: Box<dyn Send + 'static> = Box::new(decode_guard);
                let fetch = ctx.proxy.forward_streaming_to(
                    &decode_worker.url,
                    &decode_worker.breaker,
                    upstream_path,
                    &headers,
                    injected_body,
                    Some(stream_guards),
                );
                tokio::select! {
                    biased;
                    r = fetch => r,
                    _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
                }
            } else {
                let _decode_hold = decode_guard;
                let fetch = ctx.proxy.forward_json_to(
                    &decode_worker.url,
                    &decode_worker.breaker,
                    upstream_path,
                    &headers,
                    injected_body,
                );
                tokio::select! {
                    biased;
                    r = fetch => r,
                    _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
                }
            }
        }
        RouteDispatchPlan::Plain {
            worker,
            headers,
            body,
        } if streaming => {
            // Plain mode, streaming. Both guards ride the SSE pump until
            // the body completes — see the matching comment in the
            // non-streaming arm.
            let stream_guards: Box<dyn Send + 'static> = Box::new((guard, active_guard));
            let fetch = ctx.proxy.forward_streaming_to(
                &worker.url,
                &worker.breaker,
                upstream_path,
                &headers,
                body,
                Some(stream_guards),
            );
            // Bias `fetch` over the cancellation branch: a successful
            // response that completes in the same poll as the token firing
            // MUST win (returning 504 for a request that already has
            // headers is a correctness regression). The cancellation
            // branch only matters when fetch is still pending — at that
            // point biasing the order is a wash.
            tokio::select! {
                biased;
                r = fetch => r,
                _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
            }
        }
        RouteDispatchPlan::Plain {
            worker,
            headers,
            body,
        } => {
            // Plain mode, non-streaming. The handler awaits the full
            // buffered response, so both guards live correctly in this
            // scope. The tuple binding exists only to extend the guards'
            // lifetime to the end of the function — the `forward_json_to`
            // future does not need them (it does not return until the
            // body is buffered).
            let _holds: (LoadGuard, _) = (guard, active_guard);
            let fetch = ctx.proxy.forward_json_to(
                &worker.url,
                &worker.breaker,
                upstream_path,
                &headers,
                body,
            );
            // Same `biased` order as the streaming arm.
            tokio::select! {
                biased;
                r = fetch => r,
                _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
            }
        }
    };

    // Record the dispatch outcome AFTER we know whether the upstream
    // accepted the request. A 504 from the stale-request branch counts as
    // `cancelled` — semantically distinct from upstream errors that bubble
    // through as `error`. The metric is per-worker so convergence tests
    // can scrape `/metrics` and assert that ≥N requests landed on a
    // single prefill worker.
    let outcome = match &result {
        Ok(_) => RequestOutcome::Success,
        Err(ApiError::StaleRequestExpired { .. }) => {
            // The janitor fired the stale-cancel and we observed it
            // user-side; record both the per-request `cancelled` outcome
            // AND the global `expired` count. The two views are useful for
            // different alerts: per-worker request_total{cancelled} flags a
            // worker that's hanging, while stale_requests_total{expired}
            // tracks the global health of the janitor.
            ctx.metrics
                .record_stale_request(StaleRequestOutcome::Expired);
            RequestOutcome::Cancelled
        }
        Err(_) => RequestOutcome::Error,
    };
    ctx.metrics
        .record_request(&metrics_worker_url, &metrics_model, metrics_mode, outcome);

    // Mirror PD-dispatch metadata onto successful upstream responses so
    // external tests / sidecars can observe decode affinity and the
    // bootstrap room without sniffing the proxy hop. Plain-mode requests
    // skip this (no decode peer was resolved). A malformed decode URL
    // was already rejected at the request-side parse — we only reach
    // this branch when the URL was header-valid, so the second parse is
    // defensive only.
    match result {
        Ok(response) => {
            let mut response = postprocess_response(response, postprocess).await?;
            if let Some(plan) = pd_dispatch_plan {
                plan.insert_response_headers(response.headers_mut());
            }
            Ok(response)
        }
        Err(error) => Err(error),
    }
}

async fn postprocess_response(
    response: Response<Body>,
    postprocess: ResponsePostprocess,
) -> Result<Response<Body>, ApiError> {
    match postprocess {
        ResponsePostprocess::None => Ok(response),
        ResponsePostprocess::Rerank(options) => build_rerank_response(options, response).await,
    }
}

async fn build_rerank_response(
    options: RerankResponseOptions,
    response: Response<Body>,
) -> Result<Response<Body>, ApiError> {
    if !response.status().is_success() {
        return Ok(response);
    }

    let body_bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .map_err(|e| ApiError::Internal(anyhow::Error::new(e).context("read rerank body")))?;
    let mut results: Vec<RerankResult> = serde_json::from_slice(&body_bytes).map_err(|e| {
        tracing::error!(error = %e, "failed to decode upstream rerank results");
        ApiError::Internal(
            anyhow::Error::new(e).context("decode upstream rerank results for gateway response"),
        )
    })?;

    if let Some(top_k) = options.top_k {
        results.truncate(top_k);
    }
    if !options.return_documents {
        for result in &mut results {
            result.document = None;
        }
    }

    let response = RerankResponse {
        id: options.rid,
        object: "rerank",
        created: unix_timestamp_secs(),
        model: options.model,
        results,
        usage: None,
    };
    Ok(Json(response).into_response())
}

/// Estimate prefill-token count from the raw request body for use as the
/// active-load `prefill_load` counter. Prefer the model tokenizer so
/// cache-aware and load-aware policies see token units; fall back to the
/// upstream char-count heuristic for non-generation request shapes.
fn estimate_prefill_tokens(ctx: &AppContext, model_id: &ModelId, body: &Bytes) -> usize {
    if let Some(text) = extract_prefill_text(body) {
        if let Some(tokenizer) = ctx.tokenizers.get(&model_id.0) {
            match adapter::encode(&tokenizer, &text) {
                Ok(token_ids) if !token_ids.is_empty() => return token_ids.len(),
                Ok(_) => {}
                Err(error) => tracing::debug!(
                    model = %model_id,
                    error = %error,
                    "active-load tokenizer count failed; falling back to body-size estimate",
                ),
            }
        }
    }

    (body.len() / CHARS_PER_TOKEN_ESTIMATE).max(1)
}

fn extract_prefill_text(body: &[u8]) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    if let Some(prompt) = value.get("prompt").and_then(|prompt| prompt.as_str()) {
        return Some(prompt.to_string());
    }
    if let Some(prompts) = value.get("prompt").and_then(|prompt| prompt.as_array()) {
        let parts = prompts
            .iter()
            .filter_map(|prompt| prompt.as_str())
            .collect::<Vec<_>>();
        if !parts.is_empty() {
            return Some(parts.join("\n"));
        }
    }
    if let Some(messages) = value
        .get("messages")
        .and_then(|messages| messages.as_array())
    {
        let mut text = String::new();
        for message in messages {
            match message.get("content") {
                Some(serde_json::Value::String(content)) => {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(content);
                }
                Some(serde_json::Value::Array(parts)) => {
                    for part in parts {
                        if let Some(content) = part.get("text").and_then(|text| text.as_str()) {
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(content);
                        }
                    }
                }
                _ => {}
            }
        }
        if !text.is_empty() {
            return Some(text);
        }
    }
    value
        .get("text")
        .and_then(|text| text.as_str())
        .map(ToString::to_string)
}

fn parse_rerank_response_options(body: &Bytes) -> Result<RerankResponseOptions, ApiError> {
    serde_json::from_slice(body).map_err(|e| {
        tracing::debug!(error = %e, "rerank request-probe deserialize failed");
        ApiError::BadRequest("invalid request: body must be a JSON object".to_string())
    })
}

fn unknown_model() -> String {
    "unknown".to_string()
}

fn default_true() -> bool {
    true
}

fn unix_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn parse_probe(body: &Bytes) -> Result<RequestProbe, ApiError> {
    // We deliberately do NOT echo the serde error into the client-visible
    // message — that risks leaking field-level detail and is also of little
    // help to a real client (which already has its own JSON validator).
    // Server-side, the full error is logged with `tracing::debug!` for
    // operator triage.
    //
    // Two-step deserialize:
    //   1. `Map<String, IgnoredAny>` *anchors* the shape to a JSON object.
    //      This rejects `null` / `[]` / `"hi"` (all valid JSON but not
    //      request shape) without walking the full value into a
    //      `serde_json::Value` per field.
    //   2. `RequestProbe` (struct of `Option<bool>` + `Option<String>`)
    //      lifts out only the fields we care about — `stream` and `model`.
    //      Other fields are ignored; the worker is authoritative for the
    //      rest of the schema.
    let _: HashMap<String, IgnoredAny> = serde_json::from_slice(body).map_err(|e| {
        tracing::debug!(error = %e, "chat-completions body rejected as non-object JSON");
        ApiError::BadRequest("invalid request: body must be a JSON object".to_string())
    })?;
    let probe: RequestProbe = serde_json::from_slice(body).map_err(|e| {
        tracing::debug!(error = %e, "chat-completions request-probe deserialize failed");
        ApiError::BadRequest("invalid request: body must be a JSON object".to_string())
    })?;
    Ok(probe)
}

fn select_model(
    ctx: &AppContext,
    model: Option<String>,
    selection: ModelSelection,
) -> Result<String, ApiError> {
    if let Some(model) = model {
        return Ok(model);
    }

    match selection {
        ModelSelection::RequireBodyModel => {
            Err(ApiError::BadRequest("missing `model` field".into()))
        }
        ModelSelection::BodyModelOrSingleConfigured => {
            if ctx.config.models.len() == 1 {
                Ok(ctx.config.models[0].id.clone())
            } else {
                Err(ApiError::BadRequest(
                    "missing `model` field for multi-model native generate".into(),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_probe_reads_stream_bool_from_object() {
        let b = Bytes::from_static(br#"{"stream": true, "model": "tiny"}"#);
        assert_eq!(parse_probe(&b).unwrap().stream, Some(true));
        let b = Bytes::from_static(br#"{"stream": false, "model": "tiny"}"#);
        assert_eq!(parse_probe(&b).unwrap().stream, Some(false));
    }

    #[test]
    fn parse_probe_defaults_when_stream_absent() {
        // Existing happy-path contract: well-formed object missing `stream`
        // must default to None (caller picks false). The minimal `RequestProbe`
        // (Option<bool> + #[serde(default)]) must NOT break this.
        let b = Bytes::from_static(br#"{"model": "tiny", "messages": []}"#);
        let p = parse_probe(&b).unwrap();
        assert_eq!(p.stream, None);
        assert_eq!(p.model.as_deref(), Some("tiny"));
    }

    #[test]
    fn parse_probe_rejects_non_object_shapes() {
        // Pin the contract: degenerate JSON (valid JSON but wrong shape)
        // must be rejected, not silently forwarded with `stream=false`.
        for bad in [&b"null"[..], &b"[]"[..], &b"\"hi\""[..], &b"42"[..]] {
            let b = Bytes::copy_from_slice(bad);
            let err = parse_probe(&b).unwrap_err();
            match err {
                ApiError::BadRequest(_) => {}
                other => panic!("expected BadRequest for {bad:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn parse_probe_rejects_malformed_json() {
        let b = Bytes::from_static(b"{not json}");
        let err = parse_probe(&b).unwrap_err();
        assert!(matches!(err, ApiError::BadRequest(_)));
    }

    #[test]
    fn parse_probe_handles_nested_messages_with_stream_true() {
        // Well-formed object with nested arrays/objects (real chat-completions
        // payloads carry `messages: [{role, content: [{type, text}]}]`). The
        // two-step deserialize must not balk on this — only the top-level
        // object shape and the `stream`/`model` fields matter.
        let b = Bytes::from_static(
            br#"{
              "model": "x",
              "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
              "stream": true
            }"#,
        );
        assert_eq!(parse_probe(&b).unwrap().stream, Some(true));
    }

    #[test]
    fn parse_probe_handles_nested_messages_with_stream_false() {
        let b = Bytes::from_static(
            br#"{
              "model": "x",
              "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
              "stream": false
            }"#,
        );
        assert_eq!(parse_probe(&b).unwrap().stream, Some(false));
    }

    #[test]
    fn parse_probe_handles_duplicate_stream_keys() {
        // RFC 8259 says "names within an object SHOULD be unique" but a
        // parser MAY accept duplicates. Step 1 (HashMap) silently
        // last-wins, but step 2 deserializes into the typed `RequestProbe`
        // struct, and `serde_json`'s `#[derive(Deserialize)]` REJECTS
        // duplicate fields with a `duplicate field` error.
        //
        // We map that to `BadRequest` (same path as other malformed input).
        // Pinning "reject" rather than "last-wins" is intentional —
        // ambiguous bodies should fail loudly at the edge, not silently
        // route based on which copy serde happened to see last.
        let b = Bytes::from_static(br#"{"stream": true, "stream": false}"#);
        let err = parse_probe(&b).unwrap_err();
        match err {
            ApiError::BadRequest(_) => {}
            other => panic!("expected BadRequest on duplicate `stream` key, got {other:?}"),
        }
    }

    #[test]
    fn parse_probe_bad_request_message_does_not_leak_serde_detail() {
        // Info-leak guard: the client-visible message must be a fixed
        // string, not the serde error (which can contain line/column
        // detail or hint at field shape).
        let b = Bytes::from_static(br#"{"stream": "not-a-bool"}"#);
        let err = parse_probe(&b).unwrap_err();
        match err {
            ApiError::BadRequest(msg) => assert_eq!(
                msg, "invalid request: body must be a JSON object",
                "client-visible message must be fixed; got: {msg}"
            ),
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }
}
