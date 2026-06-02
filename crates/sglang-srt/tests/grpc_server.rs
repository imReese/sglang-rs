use std::net::{SocketAddr, TcpListener};

use tokio::sync::oneshot;
use tonic::transport::Channel;

use sglang_srt::cli::ServerArgs;
use sglang_srt::grpc::serve_grpc_router_with_shutdown;
use sglang_srt::proto::sglang::runtime::v1::sglang_service_client::SglangServiceClient;
use sglang_srt::proto::sglang::runtime::v1::{GetModelInfoRequest, HealthCheckRequest};
use sglang_srt::server::build_bootstrap_grpc_router_service;

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
    let service = build_bootstrap_grpc_router_service(&args);
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

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
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

fn unused_local_addr() -> SocketAddr {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("ephemeral port should bind");
    listener
        .local_addr()
        .expect("ephemeral listener should have local addr")
}
