#![cfg(feature = "test-support")]

use std::fs;
use std::net::{SocketAddr, TcpListener};
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::oneshot;
use tonic::transport::Channel;

use sglang_srt::cli::ServerArgs;
use sglang_srt::grpc::serve_grpc_router_with_shutdown;
use sglang_srt::grpc_sidecar::serve_grpc_http_sidecar_with_shutdown;
use sglang_srt::proto::sglang::runtime::v1::sglang_service_client::SglangServiceClient;
use sglang_srt::proto::sglang::runtime::v1::{
    GetModelInfoRequest, GetServerInfoRequest, HealthCheckRequest, RequestOptions, SamplingParams,
    TextGenerateRequest,
};
use sglang_srt::server::test_support::build_reference_grpc_router_service;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grpc_server_accepts_generated_client_requests() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "meta-llama/Llama-3.1-8B-Instruct",
        "--served-model-name",
        "llama3",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--grpc-mode",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_grpc_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_grpc_router_with_shutdown(addr, service, true, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let mut client = connect_with_retry(addr).await;
    let health = client
        .health_check(HealthCheckRequest {})
        .await
        .expect("health check should execute")
        .into_inner();
    let model_info = client
        .get_model_info(GetModelInfoRequest {})
        .await
        .expect("model info should execute")
        .into_inner();

    assert!(health.healthy);
    assert_eq!(health.message, "ready");
    assert_eq!(model_info.model_path, "meta-llama/Llama-3.1-8B-Instruct");
    assert_eq!(model_info.served_model_name, "llama3");
    let server_info = client
        .get_server_info(GetServerInfoRequest {})
        .await
        .expect("server info should execute")
        .into_inner();
    assert!(
        server_info
            .attributes
            .keys()
            .all(|key| !key.starts_with("kv_cache.")),
        "reference service without an active KV pool must not advertise KV geometry"
    );

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grpc_http_sidecar_reports_stream_metrics_and_controls_shared_profile() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny",
        "--smg-grpc-mode",
        "--enable-metrics",
    ])
    .expect("args should parse");
    let grpc_addr = unused_local_addr();
    let sidecar_addr = unused_local_addr();
    let service = build_reference_grpc_router_service(&args);
    let sidecar_service = service.clone();
    let (grpc_shutdown_tx, grpc_shutdown_rx) = oneshot::channel();
    let (sidecar_shutdown_tx, sidecar_shutdown_rx) = oneshot::channel();

    let grpc_server = tokio::spawn(async move {
        serve_grpc_router_with_shutdown(grpc_addr, service, true, async move {
            let _ = grpc_shutdown_rx.await;
        })
        .await
    });
    let sidecar_server = tokio::spawn(async move {
        serve_grpc_http_sidecar_with_shutdown(sidecar_addr, sidecar_service, true, async move {
            let _ = sidecar_shutdown_rx.await;
        })
        .await
    });

    let mut client = connect_with_retry(grpc_addr).await;
    let mut stream = client
        .text_generate(TextGenerateRequest {
            text: "hi".to_string(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(1),
                ..SamplingParams::default()
            }),
            options: Some(RequestOptions {
                request_id: Some("sidecar-metrics".to_string()),
                stream: true,
                data_parallel_rank: 0,
                trace_headers: Default::default(),
            }),
            disaggregated_params: None,
        })
        .await
        .expect("streaming generation should start")
        .into_inner();
    while stream
        .message()
        .await
        .expect("stream response should decode")
        .is_some()
    {}

    let metrics = http_request_with_retry(sidecar_addr, "GET", "/metrics", "").await;
    assert!(metrics.starts_with("HTTP/1.1 200"), "{metrics}");
    assert!(metrics.contains("sglang_requests_total 1\n"), "{metrics}");
    assert!(
        metrics.contains("sglang_requests_in_flight 0\n"),
        "{metrics}"
    );
    assert!(
        metrics.contains("sglang_prompt_tokens_total 2\n"),
        "{metrics}"
    );
    assert!(
        metrics.contains("sglang_generation_tokens_total 1\n"),
        "{metrics}"
    );
    assert!(
        metrics.contains("sglang_time_to_first_token_seconds_count 1\n"),
        "{metrics}"
    );

    let profile_dir = unique_profile_dir();
    let profile_body = serde_json::json!({
        "output_dir": profile_dir.to_string_lossy(),
    })
    .to_string();
    let start =
        http_request_with_retry(sidecar_addr, "POST", "/start_profile", &profile_body).await;
    assert!(start.starts_with("HTTP/1.1 200"), "{start}");
    let stop = http_request_with_retry(sidecar_addr, "POST", "/stop_profile", "").await;
    assert!(stop.starts_with("HTTP/1.1 200"), "{stop}");
    assert_eq!(
        fs::read_dir(&profile_dir)
            .expect("profile directory should exist")
            .count(),
        1
    );

    grpc_shutdown_tx
        .send(())
        .expect("gRPC server should still run");
    sidecar_shutdown_tx
        .send(())
        .expect("sidecar server should still run");
    grpc_server
        .await
        .expect("gRPC task should join")
        .expect("gRPC server should stop cleanly");
    sidecar_server
        .await
        .expect("sidecar task should join")
        .expect("sidecar should stop cleanly");
    fs::remove_dir_all(profile_dir).expect("profile directory should be removed");
}

async fn connect_with_retry(addr: SocketAddr) -> SglangServiceClient<Channel> {
    let endpoint = format!("http://{addr}");
    let mut last_error = None;

    for _ in 0..20 {
        match SglangServiceClient::connect(endpoint.clone()).await {
            Ok(client) => return client,
            Err(error) => {
                last_error = Some(error);
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        }
    }

    panic!(
        "client should connect to test server: {}",
        last_error.expect("at least one connection attempt should run")
    );
}

async fn http_request_with_retry(addr: SocketAddr, method: &str, path: &str, body: &str) -> String {
    let mut last_error = None;
    for _ in 0..100 {
        match tokio::net::TcpStream::connect(addr).await {
            Ok(mut stream) => {
                let request = format!(
                    "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                if let Err(error) = stream.write_all(request.as_bytes()).await {
                    last_error = Some(error);
                } else {
                    let mut response = Vec::new();
                    match stream.read_to_end(&mut response).await {
                        Ok(_) => {
                            return String::from_utf8(response)
                                .expect("sidecar response should be UTF-8");
                        }
                        Err(error) => last_error = Some(error),
                    }
                }
            }
            Err(error) => last_error = Some(error),
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    panic!(
        "HTTP client should connect to gRPC sidecar: {}",
        last_error.expect("at least one connection attempt should run")
    );
}

fn unique_profile_dir() -> std::path::PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "sglang-rs-grpc-sidecar-profile-{}-{suffix}",
        std::process::id()
    ))
}

fn unused_local_addr() -> SocketAddr {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("ephemeral port should bind");
    listener
        .local_addr()
        .expect("ephemeral listener should have local addr")
}
