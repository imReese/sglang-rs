// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::{ModelId, WorkerMode};
use crate::policies::registry::{PdPoolResolver, PdResolveError};
use crate::server::error::ApiError;
use crate::workers::Worker;
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use bytes::Bytes;
use std::sync::Arc;

/// Observability header carrying the decode-pool URL selected via host
/// affinity for a PD-disaggregated request.
pub(crate) const X_SGL_DECODE_URL: HeaderName = HeaderName::from_static("x-sgl-decode-url");

/// Observability header carrying the bootstrap room shared by the
/// prefill and decode side of a PD-disaggregated request.
pub(crate) const X_SGL_BOOTSTRAP_ROOM: HeaderName = HeaderName::from_static("x-sgl-bootstrap-room");

/// Selected PD worker pair, matching sgl-model-gateway's
/// `WorkerSelection::Dual { prefill, decode }` shape for the current
/// HTTP route path.
#[derive(Debug, Clone)]
pub(crate) struct PdWorkerPair {
    prefill: Arc<Worker>,
    decode: Arc<Worker>,
}

impl PdWorkerPair {
    pub(crate) fn new(prefill: Arc<Worker>, decode: Arc<Worker>) -> Self {
        Self { prefill, decode }
    }

    pub(crate) fn resolve_for_prefill(
        resolver: &PdPoolResolver,
        model: &ModelId,
        prefill: Arc<Worker>,
    ) -> Result<Self, PdResolveError> {
        let decode = resolver.decode_with_affinity_for_prefill(model, &prefill)?;
        Ok(Self::new(prefill, decode))
    }

    pub(crate) fn decode(&self) -> &Arc<Worker> {
        &self.decode
    }

    pub(crate) fn prefill(&self) -> &Arc<Worker> {
        &self.prefill
    }

    pub(crate) fn prepare_plan(&self, body: &Bytes) -> Result<PdDispatchPlan, ApiError> {
        let bootstrap_port =
            self.prefill
                .bootstrap_port()
                .ok_or_else(|| ApiError::WorkerMisconfigured {
                    worker: self.prefill.url.clone(),
                    source: anyhow::anyhow!(
                        "PD prefill worker is missing disaggregation bootstrap_port"
                    ),
                })?;
        PdDispatchPlan::new(
            self.decode.url.clone(),
            self.prefill.bootstrap_host(),
            bootstrap_port,
            body,
        )
    }
}

/// Route-local worker selection boundary. This mirrors
/// sgl-model-gateway's `WorkerSelection::{Single, Dual}` shape without
/// forcing the HTTP route path to adopt the full stage pipeline in one
/// jump.
#[derive(Debug, Clone)]
pub(crate) enum RouteWorkerSelection {
    Single { worker: Arc<Worker> },
    Dual { pair: PdWorkerPair },
}

impl RouteWorkerSelection {
    pub(crate) fn from_selected_worker(
        resolver: &PdPoolResolver,
        model: &ModelId,
        worker: Arc<Worker>,
    ) -> Result<Self, PdResolveError> {
        if worker.mode() == WorkerMode::Prefill {
            Ok(Self::Dual {
                pair: PdWorkerPair::resolve_for_prefill(resolver, model, worker)?,
            })
        } else {
            Ok(Self::Single { worker })
        }
    }

    pub(crate) fn primary_worker(&self) -> &Arc<Worker> {
        match self {
            Self::Single { worker } => worker,
            Self::Dual { pair } => pair.prefill(),
        }
    }
}

/// Prepared HTTP PD execution request. This is the small request
/// execution boundary for the current route path: it keeps the selected
/// worker pair beside the bootstrap-injected dispatch plan so the
/// handler can execute prefill/decode without rebuilding either side.
#[derive(Debug, Clone)]
pub(crate) struct PdExecutionRequest {
    pair: PdWorkerPair,
    plan: PdDispatchPlan,
}

impl PdExecutionRequest {
    pub(crate) fn prepare(pair: PdWorkerPair, body: &Bytes) -> Result<Self, ApiError> {
        let plan = pair.prepare_plan(body)?;
        Ok(Self { pair, plan })
    }

    pub(crate) fn prefill(&self) -> &Arc<Worker> {
        self.pair.prefill()
    }

    pub(crate) fn decode(&self) -> &Arc<Worker> {
        self.pair.decode()
    }

    pub(crate) fn body(&self) -> &Bytes {
        self.plan.body()
    }

    pub(crate) fn bootstrap_room(&self) -> u64 {
        self.plan.bootstrap_room()
    }

    pub(crate) fn dispatch_plan(&self) -> &PdDispatchPlan {
        &self.plan
    }

    pub(crate) fn insert_request_headers(&self, headers: &mut HeaderMap) {
        self.plan.insert_request_headers(headers);
    }
}

/// Request-scoped PD dispatch metadata. This mirrors the role of
/// sgl-model-gateway's dispatch metadata stage for the current HTTP
/// router path: one object owns the decode affinity hint and the
/// bootstrap metadata that must be observable on the response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PdDispatchMetadata {
    decode_url: String,
    bootstrap: Option<BootstrapMetadata>,
}

impl PdDispatchMetadata {
    pub(crate) fn new(decode_url: impl Into<String>) -> Self {
        Self {
            decode_url: decode_url.into(),
            bootstrap: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_bootstrap(
        decode_url: impl Into<String>,
        bootstrap: BootstrapMetadata,
    ) -> Self {
        Self {
            decode_url: decode_url.into(),
            bootstrap: Some(bootstrap),
        }
    }

    pub(crate) fn set_bootstrap(&mut self, bootstrap: BootstrapMetadata) {
        self.bootstrap = Some(bootstrap);
    }

    pub(crate) fn insert_request_headers(&self, headers: &mut HeaderMap) {
        self.insert_decode_header(headers, "request");
    }

    pub(crate) fn insert_response_headers(&self, headers: &mut HeaderMap) {
        self.insert_decode_header(headers, "response");
        if let Some(bootstrap) = &self.bootstrap {
            bootstrap.insert_response_header(headers);
        }
    }

    fn insert_decode_header(&self, headers: &mut HeaderMap, direction: &'static str) {
        match HeaderValue::from_str(&self.decode_url) {
            Ok(value) => {
                headers.insert(X_SGL_DECODE_URL, value);
            }
            Err(error) => {
                tracing::warn!(
                    decode_url = %self.decode_url,
                    %direction,
                    %error,
                    "decode worker URL rejected by header parser; omitting decode hint",
                );
            }
        }
    }
}

/// Prepared PD dispatch payload. This is the HTTP router's equivalent
/// of the request-building output in sgl-model-gateway's pipeline: the
/// body is already bootstrap-injected and the dispatch metadata is ready
/// to decorate request/response headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PdDispatchPlan {
    metadata: PdDispatchMetadata,
    body: Bytes,
    bootstrap_room: u64,
}

impl PdDispatchPlan {
    pub(crate) fn new(
        decode_url: impl Into<String>,
        bootstrap_host: impl Into<String>,
        bootstrap_port: u16,
        body: &Bytes,
    ) -> Result<Self, ApiError> {
        Self::from_bootstrap(
            decode_url,
            BootstrapMetadata::new(bootstrap_host, bootstrap_port),
            body,
        )
    }

    fn from_bootstrap(
        decode_url: impl Into<String>,
        bootstrap: BootstrapMetadata,
        body: &Bytes,
    ) -> Result<Self, ApiError> {
        let injected_body = bootstrap.inject_into_body(body)?;
        let bootstrap_room = bootstrap.room();
        let mut metadata = PdDispatchMetadata::new(decode_url);
        metadata.set_bootstrap(bootstrap);
        Ok(Self {
            metadata,
            body: injected_body,
            bootstrap_room,
        })
    }

    #[cfg(test)]
    pub(crate) fn with_bootstrap(
        decode_url: impl Into<String>,
        bootstrap: BootstrapMetadata,
        body: &Bytes,
    ) -> Result<Self, ApiError> {
        Self::from_bootstrap(decode_url, bootstrap, body)
    }

    pub(crate) fn body(&self) -> &Bytes {
        &self.body
    }

    pub(crate) fn bootstrap_room(&self) -> u64 {
        self.bootstrap_room
    }

    pub(crate) fn insert_request_headers(&self, headers: &mut HeaderMap) {
        self.metadata.insert_request_headers(headers);
    }

    pub(crate) fn insert_response_headers(&self, headers: &mut HeaderMap) {
        self.metadata.insert_response_headers(headers);
    }
}

/// Per-request metadata injected into both sides of an SGLang
/// disaggregated prefill/decode dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BootstrapMetadata {
    host: String,
    port: u16,
    room: u64,
}

impl BootstrapMetadata {
    pub(crate) fn new(host: impl Into<String>, port: u16) -> Self {
        Self {
            host: host.into(),
            port,
            room: generate_room_id(),
        }
    }

    #[cfg(test)]
    pub(crate) fn with_room(host: impl Into<String>, port: u16, room: u64) -> Self {
        Self {
            host: host.into(),
            port,
            room,
        }
    }

    pub(crate) fn room(&self) -> u64 {
        self.room
    }

    /// Inject the three flat top-level fields SGLang's HTTP
    /// disagg-prefill validator requires:
    ///
    /// * `bootstrap_host` — the prefill worker's hostname; decode
    ///   connects to this address for the KV transfer.
    /// * `bootstrap_port` — the prefill worker's bootstrap server port.
    /// * `bootstrap_room` — a 63-bit random `u64` identifying this
    ///   request on both prefill and decode sides.
    pub(crate) fn inject_into_body(&self, body: &Bytes) -> Result<Bytes, ApiError> {
        let mut obj: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(body)
            .map_err(|e| {
                tracing::debug!(error = %e, "re-parse for bootstrap injection failed");
                ApiError::BadRequest("invalid request: body must be a JSON object".to_string())
            })?;
        obj.insert(
            "bootstrap_host".to_string(),
            serde_json::Value::String(self.host.clone()),
        );
        obj.insert(
            "bootstrap_port".to_string(),
            serde_json::Value::Number(self.port.into()),
        );
        obj.insert(
            "bootstrap_room".to_string(),
            serde_json::Value::Number(self.room.into()),
        );
        let bytes = serde_json::to_vec(&obj).map_err(|e| {
            ApiError::Internal(
                anyhow::Error::new(e).context("re-serialize bootstrap-injected body"),
            )
        })?;
        Ok(Bytes::from(bytes))
    }

    pub(crate) fn insert_response_header(&self, headers: &mut HeaderMap) {
        if let Ok(value) = HeaderValue::from_str(&self.room.to_string()) {
            headers.insert(X_SGL_BOOTSTRAP_ROOM, value);
        }
    }
}

/// Mint a fresh `bootstrap_room` for a PD-disagg request.
///
/// SGLang's disagg-prefill stores the room as a signed `i64` internally
/// (see `python/sglang/srt/disaggregation/utils.py` — `bootstrap_room`
/// metadata buffer is allocated as `torch.int64`). Generating in
/// `[0, i64::MAX]` keeps the value safely positive when reinterpreted
/// signed. Mirrors SMG's `pd_types::generate_room_id`, Dynamo's
/// `rand::random_range(0..=i64::MAX.cast_unsigned())`, and SGLang's
/// own Python-side `random.randint(0, 2**63 - 1)`.
pub(crate) fn generate_room_id() -> u64 {
    rand::random::<u64>() & (i64::MAX as u64)
}

pub(crate) fn reject_batched_request(upstream_path: &str, body: &Bytes) -> Result<(), ApiError> {
    let obj: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(body).map_err(|e| {
            tracing::debug!(error = %e, "re-parse for PD batch detection failed");
            ApiError::BadRequest("invalid request: body must be a JSON object".to_string())
        })?;

    if request_batch_size(&obj).is_some() {
        return Err(ApiError::BadRequest(format!(
            "PD mode does not support batched {upstream_path} requests"
        )));
    }

    Ok(())
}

fn request_batch_size(obj: &serde_json::Map<String, serde_json::Value>) -> Option<usize> {
    if let Some(items) = obj.get("text").and_then(serde_json::Value::as_array) {
        return Some(items.len());
    }
    if let Some(items) = obj.get("prompt").and_then(serde_json::Value::as_array) {
        if items.iter().all(serde_json::Value::is_string) {
            return Some(items.len());
        }
    }
    if let Some(items) = obj.get("input_ids").and_then(serde_json::Value::as_array) {
        if items.first().map_or(true, serde_json::Value::is_array) {
            return Some(items.len());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::registry::PdPoolResolver;
    use crate::workers::registry::WorkerRegistry;
    use crate::workers::Worker;
    use axum::http::HeaderMap;
    use bytes::Bytes;
    use std::sync::Arc;

    fn worker(id: &str, url: &str, mode: WorkerMode, bootstrap_port: Option<u16>) -> Worker {
        Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: url.into(),
            mode,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port,
        })
    }

    fn worker_spec(
        id: &str,
        url: &str,
        mode: WorkerMode,
        bootstrap_port: Option<u16>,
    ) -> WorkerSpec {
        WorkerSpec {
            id: WorkerId(id.into()),
            url: url.into(),
            mode,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port,
        }
    }

    #[test]
    fn bootstrap_metadata_injects_body_and_response_header_from_one_room() {
        let body = Bytes::from_static(br#"{"model":"x","messages":[]}"#);
        let metadata = BootstrapMetadata::with_room("prefill.local", 8200, 42);

        let injected = metadata.inject_into_body(&body).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&injected).unwrap();
        assert_eq!(parsed["bootstrap_host"], "prefill.local");
        assert_eq!(parsed["bootstrap_port"], 8200);
        assert_eq!(parsed["bootstrap_room"], 42);

        let mut headers = HeaderMap::new();
        metadata.insert_response_header(&mut headers);
        assert_eq!(
            headers
                .get(X_SGL_BOOTSTRAP_ROOM)
                .and_then(|v| v.to_str().ok()),
            Some("42")
        );
    }

    #[test]
    fn dispatch_metadata_inserts_request_and_response_observability_headers() {
        let bootstrap = BootstrapMetadata::with_room("prefill.local", 8200, 42);
        let dispatch = PdDispatchMetadata::with_bootstrap("grpc://decode.local:30000", bootstrap);

        let mut request_headers = HeaderMap::new();
        dispatch.insert_request_headers(&mut request_headers);
        assert_eq!(
            request_headers
                .get(X_SGL_DECODE_URL)
                .and_then(|v| v.to_str().ok()),
            Some("grpc://decode.local:30000")
        );
        assert!(request_headers.get(X_SGL_BOOTSTRAP_ROOM).is_none());

        let mut response_headers = HeaderMap::new();
        dispatch.insert_response_headers(&mut response_headers);
        assert_eq!(
            response_headers
                .get(X_SGL_DECODE_URL)
                .and_then(|v| v.to_str().ok()),
            Some("grpc://decode.local:30000")
        );
        assert_eq!(
            response_headers
                .get(X_SGL_BOOTSTRAP_ROOM)
                .and_then(|v| v.to_str().ok()),
            Some("42")
        );
    }

    #[test]
    fn dispatch_plan_prepares_injected_body_and_observability_headers() {
        let body = Bytes::from_static(br#"{"model":"x","messages":[]}"#);
        let bootstrap = BootstrapMetadata::with_room("prefill.local", 8200, 42);
        let plan =
            PdDispatchPlan::with_bootstrap("grpc://decode.local:30000", bootstrap, &body).unwrap();

        assert_eq!(plan.bootstrap_room(), 42);
        let parsed: serde_json::Value = serde_json::from_slice(plan.body()).unwrap();
        assert_eq!(parsed["bootstrap_host"], "prefill.local");
        assert_eq!(parsed["bootstrap_port"], 8200);
        assert_eq!(parsed["bootstrap_room"], 42);

        let mut request_headers = HeaderMap::new();
        plan.insert_request_headers(&mut request_headers);
        assert_eq!(
            request_headers
                .get(X_SGL_DECODE_URL)
                .and_then(|v| v.to_str().ok()),
            Some("grpc://decode.local:30000")
        );

        let mut response_headers = HeaderMap::new();
        plan.insert_response_headers(&mut response_headers);
        assert_eq!(
            response_headers
                .get(X_SGL_BOOTSTRAP_ROOM)
                .and_then(|v| v.to_str().ok()),
            Some("42")
        );
    }

    #[test]
    fn worker_pair_prepares_plan_from_prefill_bootstrap_and_decode_url() {
        let prefill = Arc::new(worker(
            "prefill",
            "http://prefill.local:30000",
            WorkerMode::Prefill,
            Some(8200),
        ));
        let decode = Arc::new(worker(
            "decode",
            "grpc://decode.local:30000",
            WorkerMode::Decode,
            None,
        ));
        let body = Bytes::from_static(br#"{"model":"x","messages":[]}"#);

        let plan = PdWorkerPair::new(prefill, decode)
            .prepare_plan(&body)
            .unwrap();

        let parsed: serde_json::Value = serde_json::from_slice(plan.body()).unwrap();
        assert_eq!(parsed["bootstrap_host"], "prefill.local");
        assert_eq!(parsed["bootstrap_port"], 8200);

        let mut headers = HeaderMap::new();
        plan.insert_request_headers(&mut headers);
        assert_eq!(
            headers.get(X_SGL_DECODE_URL).and_then(|v| v.to_str().ok()),
            Some("grpc://decode.local:30000")
        );
    }

    #[test]
    fn execution_request_keeps_pair_and_prepared_dispatch_body_together() {
        let prefill = Arc::new(worker(
            "prefill",
            "http://prefill.local:30000",
            WorkerMode::Prefill,
            Some(8200),
        ));
        let decode = Arc::new(worker(
            "decode",
            "grpc://decode.local:30000",
            WorkerMode::Decode,
            None,
        ));
        let body = Bytes::from_static(br#"{"model":"x","messages":[]}"#);
        let pair = PdWorkerPair::new(prefill, decode);

        let request = PdExecutionRequest::prepare(pair, &body).unwrap();

        assert_eq!(request.prefill().id, WorkerId("prefill".into()));
        assert_eq!(request.decode().id, WorkerId("decode".into()));
        let parsed: serde_json::Value = serde_json::from_slice(request.body()).unwrap();
        assert_eq!(parsed["bootstrap_host"], "prefill.local");
        assert_eq!(parsed["bootstrap_port"], 8200);

        let mut headers = HeaderMap::new();
        request.insert_request_headers(&mut headers);
        assert_eq!(
            headers.get(X_SGL_DECODE_URL).and_then(|v| v.to_str().ok()),
            Some("grpc://decode.local:30000")
        );
        assert!(request.bootstrap_room() <= i64::MAX as u64);
    }

    #[test]
    fn worker_pair_resolves_decode_peer_for_selected_prefill() {
        let model = ModelId("tiny".into());
        let registry = Arc::new(WorkerRegistry::default());
        registry
            .add(worker_spec(
                "prefill",
                "grpc://host-a.local:30000",
                WorkerMode::Prefill,
                Some(8200),
            ))
            .unwrap();
        registry
            .add(worker_spec(
                "decode-same-host",
                "grpc://host-a.local:30001",
                WorkerMode::Decode,
                None,
            ))
            .unwrap();
        registry
            .add(worker_spec(
                "decode-other-host",
                "grpc://host-b.local:30000",
                WorkerMode::Decode,
                None,
            ))
            .unwrap();

        let resolver = PdPoolResolver::new(Arc::clone(&registry));
        let selected_prefill = registry
            .workers_for_mode(&model, WorkerMode::Prefill)
            .pop()
            .unwrap();

        let pair = PdWorkerPair::resolve_for_prefill(&resolver, &model, selected_prefill).unwrap();

        assert_eq!(pair.decode().id, WorkerId("decode-same-host".into()));
    }

    #[test]
    fn route_worker_selection_keeps_plain_workers_single() {
        let model = ModelId("tiny".into());
        let registry = Arc::new(WorkerRegistry::default());
        let resolver = PdPoolResolver::new(Arc::clone(&registry));
        let plain = Arc::new(worker(
            "plain",
            "grpc://plain.local:30000",
            WorkerMode::Plain,
            None,
        ));

        let selection =
            RouteWorkerSelection::from_selected_worker(&resolver, &model, Arc::clone(&plain))
                .unwrap();

        assert_eq!(selection.primary_worker().id, WorkerId("plain".into()));
        assert!(matches!(selection, RouteWorkerSelection::Single { .. }));
    }

    #[test]
    fn route_worker_selection_turns_prefill_into_dual_pair() {
        let model = ModelId("tiny".into());
        let registry = Arc::new(WorkerRegistry::default());
        registry
            .add(worker_spec(
                "prefill",
                "grpc://host-a.local:30000",
                WorkerMode::Prefill,
                Some(8200),
            ))
            .unwrap();
        registry
            .add(worker_spec(
                "decode",
                "grpc://host-a.local:30001",
                WorkerMode::Decode,
                None,
            ))
            .unwrap();

        let resolver = PdPoolResolver::new(Arc::clone(&registry));
        let selected_prefill = registry
            .workers_for_mode(&model, WorkerMode::Prefill)
            .pop()
            .unwrap();

        let selection =
            RouteWorkerSelection::from_selected_worker(&resolver, &model, selected_prefill)
                .unwrap();

        assert_eq!(selection.primary_worker().id, WorkerId("prefill".into()));
        match selection {
            RouteWorkerSelection::Dual { pair } => {
                assert_eq!(pair.decode().id, WorkerId("decode".into()));
            }
            RouteWorkerSelection::Single { .. } => panic!("expected PD worker pair"),
        }
    }

    /// `generate_room_id` MUST return values in `[0, i64::MAX]`. The
    /// SGLang prefill stores `bootstrap_room` as `torch.int64`; a u64
    /// with the top bit set would wrap negative on the engine side.
    /// Sample many times to defend against future refactors of the
    /// mask (e.g. someone "simplifying" to plain `rand::random::<u64>()`).
    #[test]
    fn generate_room_id_stays_in_63_bit_range() {
        for _ in 0..10_000 {
            let r = generate_room_id();
            assert!(
                r <= i64::MAX as u64,
                "generate_room_id() returned {r} > i64::MAX; would wrap negative as torch.int64",
            );
        }
    }

    #[test]
    fn inject_bootstrap_fields_includes_required_port() {
        let body = Bytes::from_static(br#"{"model":"x","messages":[]}"#);
        let metadata = BootstrapMetadata::with_room("host", 8997, 42);
        let injected = metadata.inject_into_body(&body).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&injected).unwrap();
        assert_eq!(
            parsed.get("bootstrap_port"),
            Some(&serde_json::Value::Number(8997.into()))
        );
        assert_eq!(
            parsed.get("bootstrap_host"),
            Some(&serde_json::Value::String("host".into()))
        );
        assert_eq!(
            parsed.get("bootstrap_room"),
            Some(&serde_json::Value::Number(42.into()))
        );
    }
}
