use tonic::Request;

use sglang_srt::cli::ServerArgs;
use sglang_srt::proto::sglang::runtime::v1::GetModelInfoRequest;
use sglang_srt::proto::sglang::runtime::v1::sglang_service_server::SglangService;
use sglang_srt::server::{build_bootstrap_grpc_router_service, grpc_listen_addr};

#[test]
fn grpc_listen_addr_uses_server_host_and_port() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--host",
        "127.0.0.1",
        "--port",
        "30001",
        "--grpc-mode",
    ])
    .expect("args should parse");

    let addr = grpc_listen_addr(&args).expect("listen address should resolve");

    assert_eq!(addr.ip().to_string(), "127.0.0.1");
    assert_eq!(addr.port(), 30001);
}

#[tokio::test]
async fn bootstrap_grpc_router_service_carries_model_metadata() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "meta-llama/Llama-3.1-8B-Instruct",
        "--served-model-name",
        "llama3",
        "--tokenizer-path",
        "hf-tokenizer",
        "--grpc-mode",
    ])
    .expect("args should parse");
    let service = build_bootstrap_grpc_router_service(&args);

    let response = service
        .get_model_info(Request::new(GetModelInfoRequest {}))
        .await
        .expect("model info should execute")
        .into_inner();

    assert_eq!(response.model_path, "meta-llama/Llama-3.1-8B-Instruct");
    assert_eq!(response.tokenizer_path, "hf-tokenizer");
    assert_eq!(response.served_model_name, "llama3");
}
