// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::server::error::ApiError;
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use bytes::Bytes;

/// Observability header carrying the decode-pool URL selected via host
/// affinity for a PD-disaggregated request.
pub(crate) const X_SGL_DECODE_URL: HeaderName = HeaderName::from_static("x-sgl-decode-url");

/// Observability header carrying the bootstrap room shared by the
/// prefill and decode side of a PD-disaggregated request.
pub(crate) const X_SGL_BOOTSTRAP_ROOM: HeaderName = HeaderName::from_static("x-sgl-bootstrap-room");

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
    use axum::http::HeaderMap;
    use bytes::Bytes;

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
