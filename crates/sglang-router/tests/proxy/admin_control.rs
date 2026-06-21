// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Gateway admin/control-plane routes.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use sglang_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, DiscoveryConfig, ModelConfig, ObservabilityConfig,
    PolicyKind, ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sglang_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sglang_router::policies::factory::build_registry_with_defaults;
use sglang_router::proxy::Proxy;
use sglang_router::server::app::build_router;
use sglang_router::server::app_context::AppContext;
use sglang_router::tokenizer::TokenizerRegistry;
use sglang_router::workers::WorkerRegistry;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

fn config(model_ids: &[&str]) -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        models: model_ids
            .iter()
            .map(|model_id| ModelConfig {
                id: (*model_id).to_string(),
                tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
                policy: PolicyKind::RoundRobin,
                circuit_breaker: None,
                cache_aware: None,
            })
            .collect(),
        discovery: DiscoveryConfig {
            backend: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
                urls: vec!["http://placeholder:0".into()],
            }),
        },
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
    }
}

fn build_ctx(cfg: Config, specs: Vec<WorkerSpec>) -> Arc<AppContext> {
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    for spec in specs {
        registry.add(spec).expect("worker spec should register");
    }
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

fn update_weights_request(body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/update_weights_from_disk")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

fn flush_cache_request(body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/flush_cache")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

fn admin_request(path: &str, body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

#[tokio::test]
async fn pause_and_continue_generation_proxy_to_single_plain_worker() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let cfg = config(&["tiny"]);
    let ctx = build_ctx(
        cfg,
        vec![WorkerSpec {
            id: WorkerId("plain-1".into()),
            url: worker.url.clone(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        }],
    );
    let app = build_router(ctx);

    let pause = app
        .clone()
        .oneshot(admin_request(
            "/pause_generation",
            serde_json::json!({"mode": "in_place"}),
        ))
        .await
        .unwrap();
    assert_eq!(pause.status(), StatusCode::OK);
    let body: serde_json::Value =
        serde_json::from_slice(&pause.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["success"], true);
    assert_eq!(body["affected_workers"], 1);

    let captured_pause = worker
        .captured
        .lock()
        .unwrap()
        .last_body
        .clone()
        .expect("plain worker should receive pause request");
    let forwarded_pause: serde_json::Value = serde_json::from_slice(&captured_pause).unwrap();
    assert_eq!(forwarded_pause["mode"], "in_place");

    let cont = app
        .oneshot(admin_request(
            "/continue_generation",
            serde_json::json!({"torch_empty_cache": false}),
        ))
        .await
        .unwrap();
    assert_eq!(cont.status(), StatusCode::OK);
    let body: serde_json::Value =
        serde_json::from_slice(&cont.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["success"], true);
    assert_eq!(body["affected_workers"], 1);

    let captured_continue = worker
        .captured
        .lock()
        .unwrap()
        .last_body
        .clone()
        .expect("plain worker should receive continue request");
    let forwarded_continue: serde_json::Value = serde_json::from_slice(&captured_continue).unwrap();
    assert_eq!(forwarded_continue["torch_empty_cache"], false);
}

#[tokio::test]
async fn pause_generation_proxies_to_prefill_and_decode_pd_workers() {
    let prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let cfg = config(&["tiny"]);
    let ctx = build_ctx(
        cfg,
        vec![
            WorkerSpec {
                id: WorkerId("prefill-1".into()),
                url: prefill.url.clone(),
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: Some(8997),
            },
            WorkerSpec {
                id: WorkerId("decode-1".into()),
                url: decode.url.clone(),
                mode: WorkerMode::Decode,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: None,
            },
        ],
    );
    let app = build_router(ctx);

    let pause = app
        .oneshot(admin_request(
            "/pause_generation",
            serde_json::json!({"model": "tiny", "mode": "retract"}),
        ))
        .await
        .unwrap();

    assert_eq!(pause.status(), StatusCode::OK);
    let body: serde_json::Value =
        serde_json::from_slice(&pause.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["success"], true);
    assert_eq!(body["affected_workers"], 2);

    for worker in [&prefill, &decode] {
        let captured = worker
            .captured
            .lock()
            .unwrap()
            .last_body
            .clone()
            .expect("PD worker should receive pause request");
        let forwarded: serde_json::Value = serde_json::from_slice(&captured).unwrap();
        assert_eq!(forwarded["model"], "tiny");
        assert_eq!(forwarded["mode"], "retract");
    }
}

#[tokio::test]
async fn abort_request_proxies_to_single_plain_worker() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let cfg = config(&["tiny"]);
    let ctx = build_ctx(
        cfg,
        vec![WorkerSpec {
            id: WorkerId("plain-1".into()),
            url: worker.url.clone(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        }],
    );
    let app = build_router(ctx);

    let response = app
        .oneshot(admin_request(
            "/abort_request",
            serde_json::json!({"model": "tiny", "rid": "plain-abort"}),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["success"], true);
    assert_eq!(body["affected_workers"], 1);

    let captured = worker
        .captured
        .lock()
        .unwrap()
        .last_body
        .clone()
        .expect("plain worker should receive abort request");
    let forwarded: serde_json::Value = serde_json::from_slice(&captured).unwrap();
    assert_eq!(forwarded["rid"], "plain-abort");
    assert_eq!(forwarded["model"], "tiny");
}

#[tokio::test]
async fn abort_request_proxies_to_prefill_and_decode_pd_workers() {
    let prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let cfg = config(&["tiny"]);
    let ctx = build_ctx(
        cfg,
        vec![
            WorkerSpec {
                id: WorkerId("prefill-1".into()),
                url: prefill.url.clone(),
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: Some(8997),
            },
            WorkerSpec {
                id: WorkerId("decode-1".into()),
                url: decode.url.clone(),
                mode: WorkerMode::Decode,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: None,
            },
        ],
    );
    let app = build_router(ctx);

    let response = app
        .oneshot(admin_request(
            "/abort_request",
            serde_json::json!({"model": "tiny", "rid": "pd-abort"}),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["success"], true);
    assert_eq!(body["affected_workers"], 2);

    for worker in [&prefill, &decode] {
        let captured = worker
            .captured
            .lock()
            .unwrap()
            .last_body
            .clone()
            .expect("PD worker should receive abort request");
        let forwarded: serde_json::Value = serde_json::from_slice(&captured).unwrap();
        assert_eq!(forwarded["rid"], "pd-abort");
        assert_eq!(forwarded["model"], "tiny");
    }
}

#[tokio::test]
async fn start_and_stop_profile_proxy_to_single_plain_worker() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let cfg = config(&["tiny"]);
    let ctx = build_ctx(
        cfg,
        vec![WorkerSpec {
            id: WorkerId("plain-1".into()),
            url: worker.url.clone(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        }],
    );
    let app = build_router(ctx);

    let start = app
        .clone()
        .oneshot(admin_request(
            "/start_profile",
            serde_json::json!({"model": "tiny", "output_dir": "/tmp/sglang-profile"}),
        ))
        .await
        .unwrap();

    assert_eq!(start.status(), StatusCode::OK);
    let body: serde_json::Value =
        serde_json::from_slice(&start.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["success"], true);
    assert_eq!(body["affected_workers"], 1);

    let captured_start = worker
        .captured
        .lock()
        .unwrap()
        .last_body
        .clone()
        .expect("plain worker should receive start_profile request");
    let forwarded_start: serde_json::Value = serde_json::from_slice(&captured_start).unwrap();
    assert_eq!(forwarded_start["model"], "tiny");
    assert_eq!(forwarded_start["output_dir"], "/tmp/sglang-profile");

    let stop = app
        .oneshot(admin_request(
            "/stop_profile",
            serde_json::json!({"model": "tiny"}),
        ))
        .await
        .unwrap();
    assert_eq!(stop.status(), StatusCode::OK);
    let body: serde_json::Value =
        serde_json::from_slice(&stop.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["success"], true);
    assert_eq!(body["affected_workers"], 1);
}

#[tokio::test]
async fn start_profile_proxies_to_prefill_and_decode_pd_workers() {
    let prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let cfg = config(&["tiny"]);
    let ctx = build_ctx(
        cfg,
        vec![
            WorkerSpec {
                id: WorkerId("prefill-1".into()),
                url: prefill.url.clone(),
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: Some(8997),
            },
            WorkerSpec {
                id: WorkerId("decode-1".into()),
                url: decode.url.clone(),
                mode: WorkerMode::Decode,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: None,
            },
        ],
    );
    let app = build_router(ctx);

    let start = app
        .oneshot(admin_request(
            "/start_profile",
            serde_json::json!({"model": "tiny", "output_dir": "/tmp/sglang-pd-profile"}),
        ))
        .await
        .unwrap();

    assert_eq!(start.status(), StatusCode::OK);
    let body: serde_json::Value =
        serde_json::from_slice(&start.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["success"], true);
    assert_eq!(body["affected_workers"], 2);

    for worker in [&prefill, &decode] {
        let captured = worker
            .captured
            .lock()
            .unwrap()
            .last_body
            .clone()
            .expect("PD worker should receive start_profile request");
        let forwarded: serde_json::Value = serde_json::from_slice(&captured).unwrap();
        assert_eq!(forwarded["model"], "tiny");
        assert_eq!(forwarded["output_dir"], "/tmp/sglang-pd-profile");
    }
}

#[tokio::test]
async fn flush_cache_proxies_to_single_plain_worker() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let cfg = config(&["tiny"]);
    let ctx = build_ctx(
        cfg,
        vec![WorkerSpec {
            id: WorkerId("plain-1".into()),
            url: worker.url.clone(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        }],
    );
    let app = build_router(ctx);

    let res = app
        .oneshot(flush_cache_request(serde_json::json!({})))
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let body: serde_json::Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["success"], true);
    assert_eq!(body["flushed_workers"], 1);

    let captured = worker
        .captured
        .lock()
        .unwrap()
        .last_body
        .clone()
        .expect("plain worker should receive flush request");
    let forwarded: serde_json::Value = serde_json::from_slice(&captured).unwrap();
    assert_eq!(forwarded, serde_json::json!({}));
}

#[tokio::test]
async fn flush_cache_proxies_to_prefill_and_decode_pd_workers() {
    let prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let cfg = config(&["tiny"]);
    let ctx = build_ctx(
        cfg,
        vec![
            WorkerSpec {
                id: WorkerId("prefill-1".into()),
                url: prefill.url.clone(),
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: Some(8997),
            },
            WorkerSpec {
                id: WorkerId("decode-1".into()),
                url: decode.url.clone(),
                mode: WorkerMode::Decode,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: None,
            },
        ],
    );
    let app = build_router(ctx);

    let res = app
        .oneshot(flush_cache_request(serde_json::json!({"model": "tiny"})))
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let body: serde_json::Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["success"], true);
    assert_eq!(body["flushed_workers"], 2);

    for worker in [&prefill, &decode] {
        let captured = worker
            .captured
            .lock()
            .unwrap()
            .last_body
            .clone()
            .expect("PD worker should receive flush request");
        let forwarded: serde_json::Value = serde_json::from_slice(&captured).unwrap();
        assert_eq!(forwarded["model"], "tiny");
    }
}

#[tokio::test]
async fn update_weights_from_disk_proxies_to_single_plain_worker() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let cfg = config(&["tiny"]);
    let ctx = build_ctx(
        cfg,
        vec![WorkerSpec {
            id: WorkerId("plain-1".into()),
            url: worker.url.clone(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        }],
    );
    let app = build_router(ctx);

    let res = app
        .oneshot(update_weights_request(serde_json::json!({
            "model_path": "/models/tiny-v2",
            "load_format": "safetensors"
        })))
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let body: serde_json::Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["success"], true);

    let captured = worker
        .captured
        .lock()
        .unwrap()
        .last_body
        .clone()
        .expect("plain worker should receive update request");
    let forwarded: serde_json::Value = serde_json::from_slice(&captured).unwrap();
    assert_eq!(forwarded["model_path"], "/models/tiny-v2");
    assert_eq!(forwarded["load_format"], "safetensors");
}

#[tokio::test]
async fn update_weights_from_disk_proxies_to_prefill_and_decode_pd_workers() {
    let prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let cfg = config(&["tiny"]);
    let ctx = build_ctx(
        cfg,
        vec![
            WorkerSpec {
                id: WorkerId("prefill-1".into()),
                url: prefill.url.clone(),
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: Some(8997),
            },
            WorkerSpec {
                id: WorkerId("decode-1".into()),
                url: decode.url.clone(),
                mode: WorkerMode::Decode,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: None,
            },
        ],
    );
    let app = build_router(ctx);

    let res = app
        .oneshot(update_weights_request(serde_json::json!({
            "model": "tiny",
            "model_path": "/models/tiny-v3",
            "load_format": "safetensors"
        })))
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let body: serde_json::Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["success"], true);
    assert_eq!(body["updated_workers"], 2);

    for worker in [&prefill, &decode] {
        let captured = worker
            .captured
            .lock()
            .unwrap()
            .last_body
            .clone()
            .expect("PD worker should receive update request");
        let forwarded: serde_json::Value = serde_json::from_slice(&captured).unwrap();
        assert_eq!(forwarded["model"], "tiny");
        assert_eq!(forwarded["model_path"], "/models/tiny-v3");
    }
}
