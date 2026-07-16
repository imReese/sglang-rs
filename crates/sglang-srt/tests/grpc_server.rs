use std::net::{SocketAddr, TcpListener};

use tokio::sync::oneshot;
use tonic::transport::Channel;

use sglang_srt::cli::ServerArgs;
use sglang_srt::grpc::serve_grpc_router_with_shutdown;
use sglang_srt::proto::sglang::runtime::v1::sglang_service_client::SglangServiceClient;
use sglang_srt::proto::sglang::runtime::v1::{
    GetModelInfoRequest, GetServerInfoRequest, HealthCheckRequest,
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
        "--kv-cache-dtype",
        "bfloat16",
        "--kv-cache-num-layers",
        "78",
        "--kv-cache-kv-heads",
        "64",
        "--kv-cache-head-dim",
        "64",
        "--page-size",
        "64",
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
    assert_eq!(
        server_info.attributes.get("kv_cache.dtype"),
        Some(&"bfloat16".to_string())
    );
    assert_eq!(
        server_info.attributes.get("kv_cache.page_size"),
        Some(&"64".to_string())
    );
    assert_eq!(
        server_info.attributes.get("kv_cache.num_layers"),
        Some(&"78".to_string())
    );
    assert_eq!(
        server_info.attributes.get("kv_cache.kv_heads"),
        Some(&"64".to_string())
    );
    assert_eq!(
        server_info.attributes.get("kv_cache.head_dim"),
        Some(&"64".to_string())
    );
    assert_eq!(
        server_info.attributes.get("kv_cache.kv_tensors_per_token"),
        Some(&"2".to_string())
    );
    assert_eq!(
        server_info.attributes.get("kv_cache.bytes_per_token"),
        Some(&(78 * 2 * 64 * 64 * 2).to_string())
    );
    assert_eq!(
        server_info.attributes.get("kv_cache.page_size_bytes"),
        Some(&(64 * 78 * 2 * 64 * 64 * 2).to_string())
    );

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
