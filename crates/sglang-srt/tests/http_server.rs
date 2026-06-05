use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};

use serde_json::Value;
use tokio::sync::oneshot;

use sglang_srt::cli::ServerArgs;
use sglang_srt::http::serve_http_router_with_shutdown;
use sglang_srt::server::{
    build_bootstrap_http_router_service, build_bootstrap_prefill_http_router_service,
};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_server_accepts_model_and_generate_requests() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-http",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_bootstrap_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let models = get_json_with_retry(addr, "/v1/models").await;
    let generated = post_json_with_retry(
        addr,
        "/generate",
        r#"{"text":"hello","sampling_params":{"max_new_tokens":1}}"#,
    )
    .await;

    assert_eq!(models["data"][0]["id"], "glm-http");
    assert_eq!(generated["text"], " ");
    assert_eq!(generated["usage"]["prompt_tokens"], 5);
    assert_eq!(generated["usage"]["completion_tokens"], 1);

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_prefill_server_accepts_disaggregated_generate_requests() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-prefill-http",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "fake",
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_bootstrap_prefill_http_router_service(&args);
    let inspected_service = service.clone();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let generated = post_json_with_retry(
        addr,
        "/generate",
        r#"{"request_id":"http-pd-prefill","text":"hello","sampling_params":{"max_new_tokens":1},"bootstrap_host":"10.0.0.8","bootstrap_port":8200,"bootstrap_room":77}"#,
    )
    .await;

    assert_eq!(generated["request_id"], "http-pd-prefill");
    assert_eq!(generated["text"], " ");

    let runtime = inspected_service
        .runtime()
        .lock()
        .expect("runtime lock should be held");
    let worker = runtime.engine().scheduler().worker();
    let summary = worker
        .last_transfer_summary()
        .expect("PD prefill request should record transfer summary");
    assert_eq!(summary.submitted_spans(), 1);
    assert_eq!(worker.transfer_executor().transferred_rooms(), &[77]);

    drop(runtime);
    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

fn unused_local_addr() -> SocketAddr {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("ephemeral port should bind");
    listener
        .local_addr()
        .expect("ephemeral listener should have local addr")
}

async fn get_json_with_retry(addr: SocketAddr, path: &str) -> Value {
    request_json_with_retry(addr, "GET", path, None).await
}

async fn post_json_with_retry(addr: SocketAddr, path: &str, body: &'static str) -> Value {
    request_json_with_retry(addr, "POST", path, Some(body)).await
}

async fn request_json_with_retry(
    addr: SocketAddr,
    method: &'static str,
    path: &str,
    body: Option<&'static str>,
) -> Value {
    let mut last_error = None;

    for _ in 0..20 {
        match request_json(addr, method, path, body).await {
            Ok(value) => return value,
            Err(error) => {
                last_error = Some(error);
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        }
    }

    panic!(
        "HTTP client should connect to test server: {}",
        last_error.expect("at least one connection attempt should run")
    );
}

async fn request_json(
    addr: SocketAddr,
    method: &'static str,
    path: &str,
    body: Option<&'static str>,
) -> Result<Value, std::io::Error> {
    let path = path.to_string();
    tokio::task::spawn_blocking(move || {
        let mut stream = TcpStream::connect(addr)?;
        let body = body.unwrap_or_default();
        let request = format!(
            "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(request.as_bytes())?;
        let mut response = String::new();
        stream.read_to_string(&mut response)?;
        let (_, body) = response
            .split_once("\r\n\r\n")
            .expect("HTTP response should include headers");
        serde_json::from_str(body).map_err(std::io::Error::other)
    })
    .await
    .expect("blocking HTTP request should join")
}
