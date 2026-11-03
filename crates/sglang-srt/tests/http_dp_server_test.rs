// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::Duration;

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use serde_json::Value;
use tower::ServiceExt;

use sglang_srt::cli::ServerArgs;
use sglang_srt::http::serve_http_router_with_shutdown;
use sglang_srt::server::try_build_bootstrap_http_router_service;

mod support {
    pub mod qwen3_nonzero;
}

fn temp_model_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("sglang-rs-{name}-{}", std::process::id()))
}

fn can_connect_loopback_tcp() -> bool {
    let listener = match TcpListener::bind(("127.0.0.1", 0)) {
        Ok(listener) => listener,
        Err(_) => return false,
    };
    let addr = match listener.local_addr() {
        Ok(addr) => addr,
        Err(_) => return false,
    };
    TcpStream::connect_timeout(&addr, Duration::from_millis(50)).is_ok()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_multi_dp_http_router_service_in_memory() {
    let model_dir = temp_model_dir("multi-dp-http-test");
    support::qwen3_nonzero::write_model(&model_dir);

    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model path"),
        "--device",
        "cpu",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--dp-size",
        "2",
        "--page-size",
        "4",
        "--max-running-requests",
        "8",
    ])
    .expect("args parse");

    let service = try_build_bootstrap_http_router_service(&args)
        .expect("bootstrap http router service should initialize with DP size 2");

    let router = service.into_router();

    // 1. Health check
    let res = router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .expect("request build"),
        )
        .await
        .expect("oneshot call");
    assert_eq!(res.status(), StatusCode::OK);

    // 2. Model info
    let res = router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/get_model_info")
                .body(Body::empty())
                .expect("request build"),
        )
        .await
        .expect("oneshot call");
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = to_bytes(res.into_body(), usize::MAX)
        .await
        .expect("body read");
    let model_info: Value = serde_json::from_slice(&bytes).expect("json parse");
    assert_eq!(model_info["model_type"], "qwen3");

    // 3. Server info (verify dp_size = 2 is exposed)
    let res = router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/get_server_info")
                .body(Body::empty())
                .expect("request build"),
        )
        .await
        .expect("oneshot call");
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = to_bytes(res.into_body(), usize::MAX)
        .await
        .expect("body read");
    let server_info: Value = serde_json::from_slice(&bytes).expect("json parse");
    assert_eq!(server_info["dp_size"], 2);

    // 4. Load info (available cache pages across 2 DP workers)
    let res = router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/get_loads")
                .body(Body::empty())
                .expect("request build"),
        )
        .await
        .expect("oneshot call");
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = to_bytes(res.into_body(), usize::MAX)
        .await
        .expect("body read");
    let load_info: Value = serde_json::from_slice(&bytes).expect("json parse");
    assert_eq!(load_info["loads"][0]["waiting_queue_depth"], 0);
    assert_eq!(load_info["loads"][0]["decode_queue_depth"], 0);
    assert!(
        load_info["loads"][0]["available_cache_pages"]
            .as_u64()
            .unwrap()
            > 0
    );

    // 5. Generate request
    let generate_payload = serde_json::json!({
        "text": "hello",
        "sampling_params": {
            "max_new_tokens": 2,
            "temperature": 0.0
        }
    });
    let res = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&generate_payload).unwrap()))
                .expect("request build"),
        )
        .await
        .expect("oneshot call");
    let status = res.status();
    assert_eq!(status, StatusCode::OK);
    let bytes = to_bytes(res.into_body(), usize::MAX)
        .await
        .expect("body read");
    let gen_res: Value = serde_json::from_slice(&bytes).expect("json parse");
    assert!(gen_res.get("text").is_some());
    assert!(gen_res["usage"]["prompt_tokens"].as_i64().unwrap() > 0);
    assert_eq!(gen_res["usage"]["completion_tokens"], 2);

    // 6. Streaming generate request
    let stream_payload = serde_json::json!({
        "text": "hello",
        "stream": true,
        "sampling_params": {
            "max_new_tokens": 2,
            "temperature": 0.0
        }
    });
    let res = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&stream_payload).unwrap()))
                .expect("request build"),
        )
        .await
        .expect("oneshot call");
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = to_bytes(res.into_body(), usize::MAX)
        .await
        .expect("body read");
    let stream_text = String::from_utf8_lossy(&bytes);
    assert!(stream_text.contains("data: "));

    // 7. Flush cache
    let res = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/flush_cache")
                .body(Body::empty())
                .expect("request build"),
        )
        .await
        .expect("oneshot call");
    assert_eq!(res.status(), StatusCode::OK);

    let _ = fs::remove_dir_all(model_dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_multi_dp_http_server_real_tcp() {
    if !can_connect_loopback_tcp() {
        eprintln!("Skipping test_multi_dp_http_server_real_tcp: loopback TCP blocked by sandbox");
        return;
    }

    let model_dir = temp_model_dir("multi-dp-http-tcp-test");
    support::qwen3_nonzero::write_model(&model_dir);

    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model path"),
        "--device",
        "cpu",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--dp-size",
        "2",
        "--page-size",
        "4",
    ])
    .expect("args parse");

    let service = try_build_bootstrap_http_router_service(&args).expect("service should build");

    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("listener bind");
    let addr = listener.local_addr().expect("local addr");
    drop(listener);

    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let server_task = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    // Make real HTTP GET request over TCP socket
    let mut stream = TcpStream::connect(addr).expect("TCP connect to server");
    use std::io::{Read, Write};
    stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .expect("write request");
    let mut resp = String::new();
    stream.read_to_string(&mut resp).expect("read response");
    assert!(resp.starts_with("HTTP/1.1 200 OK"));

    let _ = shutdown_tx.send(());
    let _ = server_task.await;
    let _ = fs::remove_dir_all(model_dir);
}
