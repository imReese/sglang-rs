use tonic::Request;

use sglang_srt::cli::ServerArgs;
use sglang_srt::proto::sglang::runtime::v1::generate_response::Body;
use sglang_srt::proto::sglang::runtime::v1::sglang_service_server::SglangService;
use sglang_srt::proto::sglang::runtime::v1::{
    GetModelInfoRequest, RequestOptions, SamplingParams, TextGenerateRequest,
};
use sglang_srt::server::{
    ServerLaunchError, build_bootstrap_fake_pd_grpc_router_service,
    build_bootstrap_grpc_router_service, build_bootstrap_pd_grpc_router_service, grpc_listen_addr,
    launch_grpc_server,
};
use sglang_srt::transfer::{
    DecodeBootstrapRegistry, DisaggregationMode, MooncakeBatchId, MooncakeBatchReleaser,
    MooncakeError, MooncakeKvCacheLayout, MooncakeKvCacheTransferExecutor, MooncakeTransferRequest,
    MooncakeTransferStatus, MooncakeTransferStatusCode, MooncakeTransferStatusReader,
    MooncakeTransferSubmitter, MooncakeTransferTarget, PdConfigError, TransferBackend,
};

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

#[tokio::test]
async fn bootstrap_grpc_router_service_generates_through_model_runner() {
    let args = ServerArgs::parse_from(["serve", "--model-path", "dummy", "--grpc-mode"])
        .expect("args should parse");
    let service = build_bootstrap_grpc_router_service(&args);

    let mut stream = service
        .text_generate(Request::new(TextGenerateRequest {
            text: "hello".to_string(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(1),
                ..Default::default()
            }),
            options: Some(RequestOptions {
                request_id: Some("bootstrap-generate".to_string()),
                stream: true,
                data_parallel_rank: 0,
                trace_headers: Default::default(),
            }),
            disaggregated_params: None,
        }))
        .await
        .expect("text generate should execute")
        .into_inner();

    let response = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("one response")
        .expect("response should be ok");

    assert_eq!(response.request_id, "bootstrap-generate");
    assert_eq!(
        response.body,
        Some(Body::Complete(
            sglang_srt::proto::sglang::runtime::v1::GenerateComplete {
                output_ids: vec![b' ' as u32],
                text: " ".to_string(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 5,
                completion_tokens: 1,
                cached_tokens: 0,
                index: 0,
            }
        ))
    );
    assert!(
        tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
            .await
            .is_none()
    );
}

#[tokio::test]
async fn bootstrap_pd_grpc_router_service_polls_transfer_before_decode() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--grpc-mode",
        "--disaggregation-mode",
        "decode",
        "--disaggregation-decode-polling-interval",
        "1",
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("args should parse");
    let transfer_executor = MooncakeKvCacheTransferExecutor::new(
        RecordingMooncakeBackend::completed(),
        MooncakeKvCacheLayout {
            source_base_addr: 0x3000,
            page_size_bytes: 64,
            target_base_offset: 0,
        },
        MooncakeTransferTarget { target_id: 17 },
    );
    let service = build_bootstrap_pd_grpc_router_service(
        &args,
        DecodeBootstrapRegistry::default(),
        transfer_executor,
    );

    let mut stream = service
        .text_generate(Request::new(TextGenerateRequest {
            text: "hi".to_string(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            options: Some(RequestOptions {
                request_id: Some("bootstrap-pd".to_string()),
                stream: true,
                data_parallel_rank: 0,
                trace_headers: Default::default(),
            }),
            disaggregated_params: Some(
                sglang_srt::proto::sglang::runtime::v1::DisaggregatedParams {
                    bootstrap_host: "10.0.0.9".to_string(),
                    bootstrap_port: 8998,
                    bootstrap_room: 41,
                },
            ),
        }))
        .await
        .expect("PD bootstrap service should poll transfer and generate")
        .into_inner();

    let first = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("first response")
        .expect("first response should be ok");
    let second = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("second response")
        .expect("second response should be ok");

    assert_eq!(first.request_id, "bootstrap-pd");
    assert!(matches!(first.body, Some(Body::Chunk(_))));
    assert_eq!(second.request_id, "bootstrap-pd");
    assert!(matches!(second.body, Some(Body::Complete(_))));
    assert!(
        tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
            .await
            .is_none()
    );
}

#[tokio::test]
async fn bootstrap_fake_pd_grpc_router_service_uses_decode_transfer_path() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--grpc-mode",
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "fake",
        "--disaggregation-decode-polling-interval",
        "1",
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("args should parse");
    let service = build_bootstrap_fake_pd_grpc_router_service(&args);

    let mut stream = service
        .text_generate(Request::new(TextGenerateRequest {
            text: "hi".to_string(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            options: Some(RequestOptions {
                request_id: Some("bootstrap-fake-pd".to_string()),
                stream: true,
                data_parallel_rank: 1,
                trace_headers: Default::default(),
            }),
            disaggregated_params: Some(
                sglang_srt::proto::sglang::runtime::v1::DisaggregatedParams {
                    bootstrap_host: "10.0.0.9".to_string(),
                    bootstrap_port: 8998,
                    bootstrap_room: 42,
                },
            ),
        }))
        .await
        .expect("fake PD bootstrap service should generate")
        .into_inner();

    let first = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("first response")
        .expect("first response should be ok");
    let second = tonic::codegen::tokio_stream::StreamExt::next(&mut stream)
        .await
        .expect("second response")
        .expect("second response should be ok");

    assert_eq!(first.request_id, "bootstrap-fake-pd");
    assert!(matches!(first.body, Some(Body::Chunk(_))));
    assert_eq!(second.request_id, "bootstrap-fake-pd");
    assert!(matches!(second.body, Some(Body::Complete(_))));
}

#[tokio::test]
async fn bootstrap_pd_grpc_router_service_applies_max_running_requests() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--grpc-mode",
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "fake",
        "--max-running-requests",
        "1",
    ])
    .expect("args should parse");
    let service = build_bootstrap_fake_pd_grpc_router_service(&args);
    let runtime = service
        .runtime()
        .lock()
        .expect("runtime lock should be held");

    assert_eq!(runtime.engine().scheduler().max_running_requests(), Some(1));
}

#[tokio::test]
async fn launch_grpc_server_rejects_unsupported_bootstrap_pd_backend() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--grpc-mode",
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--kv-cache-dtype",
        "bfloat16",
        "--kv-cache-num-layers",
        "61",
        "--kv-cache-kv-heads",
        "1",
        "--kv-cache-head-dim",
        "512",
    ])
    .expect("args should parse");

    let error = launch_grpc_server(args)
        .await
        .expect_err("unsupported PD backend should fail before serving");

    assert_eq!(
        error,
        ServerLaunchError::UnsupportedBootstrapPdRuntime {
            mode: DisaggregationMode::Decode,
            transfer_backend: TransferBackend::Mooncake,
        }
    );
}

#[tokio::test]
async fn launch_grpc_server_requires_kv_model_layout_for_mooncake_decode() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--grpc-mode",
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--kv-cache-dtype",
        "bfloat16",
    ])
    .expect("args should parse");

    let error = launch_grpc_server(args)
        .await
        .expect_err("missing Mooncake KV layout should fail before serving");

    assert_eq!(
        error,
        ServerLaunchError::PdConfig(PdConfigError::MissingMooncakeKvCacheModelLayout)
    );
}

#[derive(Default)]
struct RecordingMooncakeBackend {
    submitted_batches: usize,
    statuses: Vec<MooncakeTransferStatusCode>,
    freed_batches: Vec<MooncakeBatchId>,
}

impl RecordingMooncakeBackend {
    fn completed() -> Self {
        Self {
            submitted_batches: 0,
            statuses: vec![MooncakeTransferStatusCode::Completed],
            freed_batches: Vec::new(),
        }
    }
}

impl MooncakeTransferSubmitter for RecordingMooncakeBackend {
    fn submit_transfer(
        &mut self,
        requests: &mut [MooncakeTransferRequest],
    ) -> Result<MooncakeBatchId, MooncakeError> {
        assert!(!requests.is_empty());
        self.submitted_batches += 1;
        Ok(700 + self.submitted_batches as MooncakeBatchId - 1)
    }
}

impl MooncakeTransferStatusReader for RecordingMooncakeBackend {
    fn transfer_status(
        &mut self,
        _batch_id: MooncakeBatchId,
        task_id: usize,
    ) -> Result<MooncakeTransferStatus, MooncakeError> {
        let status = self
            .statuses
            .get(task_id)
            .or_else(|| self.statuses.last())
            .copied()
            .expect("recording Mooncake backend needs at least one status");
        Ok(MooncakeTransferStatus {
            status: status as i32,
            transferred_bytes: 0,
        })
    }
}

impl MooncakeBatchReleaser for RecordingMooncakeBackend {
    fn free_batch(&mut self, batch_id: MooncakeBatchId) -> Result<(), MooncakeError> {
        self.freed_batches.push(batch_id);
        Ok(())
    }
}
