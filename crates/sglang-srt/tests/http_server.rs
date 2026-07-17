#![cfg(feature = "test-support")]

use std::ffi::c_void;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{
    fs,
    path::{Path, PathBuf},
};

use serde_json::Value;
use tokio::sync::oneshot;

use sglang_srt::cache::{CachePageAllocator, RadixCache};
use sglang_srt::cli::ServerArgs;
use sglang_srt::engine::Engine;
use sglang_srt::engine_info_bootstrap::{
    EngineInfoBootstrapService, TransferEngineInfo, TransferEngineInfoRegistration,
};
use sglang_srt::http::{HttpRouterService, HttpServerInfo, serve_http_router_with_shutdown};
use sglang_srt::pd_bootstrap::{PrefillBootstrapService, serve_prefill_bootstrap_with_shutdown};
use sglang_srt::router::{RouterGetModelInfoResponse, RouterRuntime};
use sglang_srt::scheduler::{ScheduleBatch, ScheduledRequest, Scheduler};
use sglang_srt::server::test_support::{
    build_reference_http_router_service, build_reference_mooncake_prefill_http_router_service,
    build_reference_pd_http_router_service, build_reference_prefill_http_router_service,
};
use sglang_srt::server::{
    ServerLaunchError, launch_http_server_with_shutdown, register_prefill_mooncake_routes_from_args,
};
use sglang_srt::tokenizer::ByteTokenizer;
use sglang_srt::transfer::{
    DecodeBootstrapRegistry, KvCacheMemoryLocation, KvTransferBackend, MooncakeBatchId,
    MooncakeBatchReleaser, MooncakeBufferEntry, MooncakeError, MooncakeKvCacheLayout,
    MooncakeKvCacheTransferExecutor, MooncakeMemoryRegistrar, MooncakeTransferRequest,
    MooncakeTransferStatus, MooncakeTransferStatusCode, MooncakeTransferStatusReader,
    MooncakeTransferSubmitter, MooncakeTransferTarget, TransferableKvCacheMemory,
    TransferableKvCacheRegion,
};
use sglang_srt::types::{BootstrapRoom, RequestId, SamplingParams as RuntimeSamplingParams};
use sglang_srt::worker::{
    BatchGeneratedTokens, FallibleModelWorker, GeneratedToken, ModelWorker, WorkerExecutionError,
    WorkerWeightUpdateRequest,
};

#[derive(Default)]
struct HttpTwoStepWorker;

impl ModelWorker for HttpTwoStepWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        let token = match batch.forward_mode() {
            sglang_srt::scheduler::ForwardMode::Prefill => GeneratedToken::unfinished(vec![42]),
            sglang_srt::scheduler::ForwardMode::Decode => GeneratedToken::finished(vec![43]),
        };

        BatchGeneratedTokens::from_batch(batch, vec![token])
            .expect("output shape should match batch")
    }
}

struct SlowHttpWorker {
    decode_started: Arc<AtomicBool>,
}

impl ModelWorker for SlowHttpWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        let token = match batch.forward_mode() {
            sglang_srt::scheduler::ForwardMode::Prefill => GeneratedToken::unfinished(vec![42]),
            sglang_srt::scheduler::ForwardMode::Decode => {
                self.decode_started.store(true, Ordering::Release);
                std::thread::sleep(Duration::from_secs(1));
                GeneratedToken::finished(vec![43])
            }
        };

        BatchGeneratedTokens::from_batch(batch, vec![token])
            .expect("output shape should match batch")
    }
}

fn unique_profile_dir(prefix: &str) -> std::path::PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "sglang-rs-{prefix}-{}-{suffix}",
        std::process::id()
    ))
}

#[derive(Default)]
struct HttpReloadingWorker {
    updates: Vec<WorkerWeightUpdateRequest>,
}

impl FallibleModelWorker for HttpReloadingWorker {
    fn try_generate_batch(
        &mut self,
        batch: &ScheduleBatch,
    ) -> Result<BatchGeneratedTokens, WorkerExecutionError> {
        let token = match batch.forward_mode() {
            sglang_srt::scheduler::ForwardMode::Prefill => GeneratedToken::unfinished(vec![42]),
            sglang_srt::scheduler::ForwardMode::Decode => GeneratedToken::finished(vec![43]),
        };

        Ok(BatchGeneratedTokens::from_batch(batch, vec![token])
            .expect("output shape should match batch"))
    }

    fn update_weights_from_disk(
        &mut self,
        request: &WorkerWeightUpdateRequest,
    ) -> Result<(), WorkerExecutionError> {
        self.updates.push(request.clone());
        Ok(())
    }
}

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
    let service = build_reference_http_router_service(&args);
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
async fn http_server_accepts_tokenized_generate_requests() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-http-tokenized",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_http_router_service(&args);
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
        r#"{"input_ids":[71,72],"original_text":"hi","sampling_params":{"max_new_tokens":2}}"#,
    )
    .await;

    assert_eq!(generated["output_ids"], serde_json::json!([32, 32]));
    assert_eq!(generated["finish_reason"], "stop");
    assert_eq!(generated["usage"]["prompt_tokens"], 2);
    assert_eq!(generated["usage"]["completion_tokens"], 2);

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_server_accepts_rerank_requests() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny-reranker",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let reranked = post_json_with_retry(
        addr,
        "/v1/rerank",
        r#"{
            "model": "tiny-reranker",
            "query": "rust pd router",
            "documents": [
                "python gateway only",
                "rust pd router transfers kv cache",
                "router"
            ]
        }"#,
    )
    .await;

    let results = reranked.as_array().expect("worker should return raw list");
    assert_eq!(results.len(), 3);
    assert_eq!(results[0]["index"], 1);
    assert_eq!(results[0]["document"], "rust pd router transfers kv cache");
    assert_eq!(results[1]["index"], 2);
    assert_eq!(results[1]["document"], "router");
    assert_eq!(results[2]["index"], 0);
    assert_eq!(results[2]["document"], "python gateway only");
    assert!(
        results[0]["score"].as_f64().unwrap() > results[1]["score"].as_f64().unwrap(),
        "more overlapping tokens should score higher: {results:?}"
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
async fn http_server_accepts_score_requests() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny-scorer",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let scored = post_json_with_retry(
        addr,
        "/v1/score",
        r#"{
            "model": "tiny-scorer",
            "query": "rust pd router",
            "items": ["rust pd router transfers kv cache", "python gateway"],
            "label_token_ids": [1, 2, 3],
            "apply_softmax": true
        }"#,
    )
    .await;

    assert_eq!(scored["object"], "scoring");
    assert_eq!(scored["model"], "tiny-scorer");
    let scores = scored["scores"].as_array().expect("scores");
    assert_eq!(scores.len(), 2);
    for row in scores {
        let row = row.as_array().expect("score row");
        assert_eq!(row.len(), 3);
        let sum = row.iter().map(|value| value.as_f64().unwrap()).sum::<f64>();
        assert!((sum - 1.0).abs() < 1e-6);
    }
    assert!(scored["usage"]["prompt_tokens"].as_i64().unwrap() > 0);
    assert_eq!(
        scored["usage"]["total_tokens"],
        scored["usage"]["prompt_tokens"]
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
async fn http_server_accepts_embedding_requests() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny-embedding",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let embeddings = post_json_with_retry(
        addr,
        "/v1/embeddings",
        r#"{
            "model": "tiny-embedding",
            "input": ["rust pd router", "python gateway"],
            "dimensions": 4
        }"#,
    )
    .await;

    assert_eq!(embeddings["object"], "list");
    assert_eq!(embeddings["model"], "tiny-embedding");
    let data = embeddings["data"]
        .as_array()
        .expect("data should be an array");
    assert_eq!(data.len(), 2);
    assert_eq!(data[0]["object"], "embedding");
    assert_eq!(data[0]["index"], 0);
    assert_eq!(data[1]["index"], 1);
    assert_eq!(data[0]["embedding"].as_array().unwrap().len(), 4);
    assert_eq!(data[1]["embedding"].as_array().unwrap().len(), 4);
    assert_ne!(data[0]["embedding"], data[1]["embedding"]);
    assert!(embeddings["usage"]["prompt_tokens"].as_i64().unwrap() > 0);
    assert_eq!(
        embeddings["usage"]["total_tokens"],
        embeddings["usage"]["prompt_tokens"]
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
async fn http_server_accepts_classify_requests() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny-classifier",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let classified = post_json_with_retry(
        addr,
        "/v1/classify",
        r#"{
            "model": "tiny-classifier",
            "input": ["rust pd router", "python gateway"]
        }"#,
    )
    .await;

    assert_eq!(classified["object"], "list");
    assert_eq!(classified["model"], "tiny-classifier");
    assert!(classified["id"].as_str().unwrap().starts_with("classify-"));
    assert!(classified["created"].as_u64().unwrap() > 0);
    let data = classified["data"]
        .as_array()
        .expect("data should be an array");
    assert_eq!(data.len(), 2);
    assert_eq!(data[0]["index"], 0);
    assert_eq!(data[1]["index"], 1);
    assert!(data[0]["label"].as_str().unwrap().starts_with("LABEL_"));
    assert_eq!(data[0]["num_classes"], 3);
    assert_eq!(data[0]["probs"].as_array().unwrap().len(), 3);
    assert_ne!(data[0]["probs"], data[1]["probs"]);
    let prob_sum = data[0]["probs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .sum::<f64>();
    assert!((prob_sum - 1.0).abs() < 1.0e-6);
    assert!(classified["usage"]["prompt_tokens"].as_i64().unwrap() > 0);
    assert_eq!(
        classified["usage"]["total_tokens"],
        classified["usage"]["prompt_tokens"]
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
async fn http_server_accepts_tokenize_and_detokenize_requests_for_sglang_clients() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-http-tokenize",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let tokenized = post_json_with_retry(
        addr,
        "/v1/tokenize",
        r#"{"model":"glm-http-tokenize","prompt":"Hello","add_special_tokens":true}"#,
    )
    .await;
    assert_eq!(
        tokenized["tokens"],
        serde_json::json!([72, 101, 108, 108, 111])
    );
    assert_eq!(tokenized["count"], 5);
    assert_eq!(tokenized["max_model_len"], -1);

    let detokenized = post_json_with_retry(
        addr,
        "/detokenize",
        r#"{"model":"glm-http-tokenize","tokens":[72,101,108,108,111],"skip_special_tokens":true}"#,
    )
    .await;
    assert_eq!(detokenized["text"], "Hello");

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_server_rejects_invalid_detokenize_tokens() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-http-detokenize-errors",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let response = request_raw_with_retry(
        addr,
        "POST",
        "/v1/detokenize",
        Some(r#"{"model":"glm-http-detokenize-errors","tokens":[256]}"#),
    )
    .await;

    assert!(response.starts_with("HTTP/1.1 400"), "{response}");
    assert!(
        response.contains("Error decoding tokens"),
        "invalid tokens should return an SGLang-style decode error: {response}"
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
async fn http_server_accepts_streaming_generate_requests() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-http-stream-generate",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let response = request_raw_with_retry(
        addr,
        "POST",
        "/generate",
        Some(r#"{"text":"hello","sampling_params":{"max_new_tokens":2},"stream":true}"#),
    )
    .await;

    assert!(
        response.starts_with("HTTP/1.1 200"),
        "streaming generate should succeed, got response: {response}"
    );
    assert!(
        response
            .to_ascii_lowercase()
            .contains("content-type: text/event-stream"),
        "streaming generate must return SSE content-type, got response: {response}"
    );
    let events = parse_sse_data(&response);
    assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
    let chunks = events
        .iter()
        .filter(|event| event.as_str() != "[DONE]")
        .map(|event| serde_json::from_str::<Value>(event))
        .collect::<Result<Vec<_>, _>>()
        .expect("SSE data chunks should be JSON");
    assert!(
        chunks
            .iter()
            .any(|chunk| chunk["request_id"].is_string() && chunk["text"].is_string()),
        "expected native generate stream chunks, got {chunks:?}"
    );
    assert!(
        chunks.iter().any(|chunk| chunk["finish_reason"] == "stop"),
        "expected final stop chunk, got {chunks:?}"
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
async fn http_streams_before_decode_finishes_and_keeps_health_lock_free() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "slow-http-stream",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let decode_started = Arc::new(AtomicBool::new(false));
    let runtime = RouterRuntime::new(Engine::new(
        ByteTokenizer,
        Scheduler::new(SlowHttpWorker {
            decode_started: Arc::clone(&decode_started),
        }),
    ));
    let service =
        HttpRouterService::new(runtime, RouterGetModelInfoResponse::from_server_args(&args))
            .with_server_info(HttpServerInfo {
                enable_metrics: true,
                ..Default::default()
            });
    let addr = unused_local_addr();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let ready = request_raw_with_retry(addr, "GET", "/health", None).await;
    assert!(ready.starts_with("HTTP/1.1 200"), "{ready}");

    let started = Instant::now();
    let first_event = read_first_sse_event(
        addr,
        "/generate",
        r#"{"text":"hello","sampling_params":{"max_new_tokens":2},"stream":true}"#,
    )
    .await
    .expect("first SSE event should arrive");
    assert!(first_event.contains("data: "), "{first_event}");
    assert!(
        started.elapsed() < Duration::from_millis(800),
        "the first SSE event waited for decode completion"
    );

    for _ in 0..100 {
        if decode_started.load(Ordering::Acquire) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(decode_started.load(Ordering::Acquire));

    let health = request_raw_with_retry(addr, "GET", "/health", None).await;
    let deep_health = request_raw_with_retry(addr, "GET", "/health_generate", None).await;
    let metrics = request_raw_with_retry(addr, "GET", "/metrics", None).await;
    assert!(health.starts_with("HTTP/1.1 200"), "{health}");
    assert!(deep_health.starts_with("HTTP/1.1 200"), "{deep_health}");
    assert!(
        deep_health.contains("runtime is actively serving a request"),
        "{deep_health}"
    );
    assert!(metrics.contains("sglang_requests_total 1"), "{metrics}");
    assert!(metrics.contains("sglang_requests_in_flight 1"), "{metrics}");
    assert!(
        metrics.contains("sglang_time_to_first_token_seconds_count 1"),
        "{metrics}"
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
async fn http_server_accepts_non_streaming_chat_completions_for_sgl_router() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-chat-http",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let response = post_json_with_retry(
        addr,
        "/v1/chat/completions",
        r#"{"model":"glm-chat-http","messages":[{"role":"user","content":"hi"}],"max_tokens":2}"#,
    )
    .await;

    assert_eq!(response["object"], "chat.completion");
    assert_eq!(response["model"], "glm-chat-http");
    assert_eq!(response["choices"][0]["index"], 0);
    assert_eq!(response["choices"][0]["message"]["role"], "assistant");
    assert_eq!(response["choices"][0]["message"]["content"], "  ");
    assert_eq!(response["choices"][0]["finish_reason"], "stop");
    assert_eq!(response["usage"]["prompt_tokens"], 2);
    assert_eq!(response["usage"]["completion_tokens"], 2);

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_server_accepts_non_streaming_responses_for_openai_clients() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-responses-http",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let response = post_json_with_retry(
        addr,
        "/v1/responses",
        r#"{"model":"glm-responses-http","input":"hi","max_output_tokens":2}"#,
    )
    .await;

    assert_eq!(response["object"], "response");
    assert_eq!(response["model"], "glm-responses-http");
    assert_eq!(response["status"], "completed");
    assert_eq!(response["output"][0]["type"], "message");
    assert_eq!(response["output"][0]["role"], "assistant");
    assert_eq!(response["output"][0]["content"][0]["type"], "output_text");
    assert_eq!(response["output"][0]["content"][0]["text"], "  ");
    assert_eq!(response["usage"]["input_tokens"], 2);
    assert_eq!(response["usage"]["output_tokens"], 2);

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_server_accepts_streaming_responses_for_openai_clients() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-responses-http-stream",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let response = request_raw_with_retry(
        addr,
        "POST",
        "/v1/responses",
        Some(
            r#"{"model":"glm-responses-http-stream","input":"hi","max_output_tokens":2,"stream":true}"#,
        ),
    )
    .await;

    assert!(
        response.starts_with("HTTP/1.1 200"),
        "streaming responses should succeed, got response: {response}"
    );
    assert!(
        response
            .to_ascii_lowercase()
            .contains("content-type: text/event-stream"),
        "streaming responses must return SSE content-type, got response: {response}"
    );
    let events = parse_sse_data(&response);
    assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
    let chunks = events
        .iter()
        .filter(|event| event.as_str() != "[DONE]")
        .map(|event| serde_json::from_str::<Value>(event))
        .collect::<Result<Vec<_>, _>>()
        .expect("SSE data chunks should be JSON");
    assert!(
        chunks
            .iter()
            .any(|chunk| chunk["type"] == "response.output_text.delta"
                && chunk["delta"]
                    .as_str()
                    .is_some_and(|delta| !delta.is_empty())),
        "expected output text deltas, got {chunks:?}"
    );
    assert!(
        chunks
            .iter()
            .any(|chunk| chunk["type"] == "response.output_text.done" && chunk["text"] == "  "),
        "expected output text done event, got {chunks:?}"
    );
    assert!(
        chunks
            .iter()
            .any(|chunk| chunk["type"] == "response.completed"
                && chunk["response"]["status"] == "completed"
                && chunk["response"]["output_text"] == "  "),
        "expected completed response event, got {chunks:?}"
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
async fn http_server_preserves_prefixed_openai_chat_request_id() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-chat-prefixed-id",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let response = post_json_with_retry(
        addr,
        "/v1/chat/completions",
        r#"{"model":"glm-chat-prefixed-id","messages":[{"role":"user","content":"hi"}],"request_id":"chatcmpl-http-prefixed","max_tokens":2}"#,
    )
    .await;

    assert_eq!(response["id"], "chatcmpl-http-prefixed");

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_server_accepts_streaming_chat_completions_for_openai_clients() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-chat-http-stream",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let response = request_raw_with_retry(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(
            r#"{"model":"glm-chat-http-stream","messages":[{"role":"user","content":"hi"}],"max_tokens":2,"stream":true}"#,
        ),
    )
    .await;

    assert!(
        response.starts_with("HTTP/1.1 200"),
        "streaming chat should succeed, got response: {response}"
    );
    assert!(
        response
            .to_ascii_lowercase()
            .contains("content-type: text/event-stream"),
        "streaming chat must return SSE content-type, got response: {response}"
    );
    let events = parse_sse_data(&response);
    assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
    let chunks = events
        .iter()
        .filter(|event| event.as_str() != "[DONE]")
        .map(|event| serde_json::from_str::<Value>(event))
        .collect::<Result<Vec<_>, _>>()
        .expect("SSE data chunks should be JSON");
    assert!(
        chunks
            .iter()
            .any(|chunk| chunk["object"] == "chat.completion.chunk"),
        "expected OpenAI chat completion chunks, got {chunks:?}"
    );
    assert!(
        chunks
            .iter()
            .any(|chunk| chunk["choices"][0]["finish_reason"] == "stop"),
        "expected final stop chunk, got {chunks:?}"
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
async fn http_server_accepts_openai_completions() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-completions-http",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let response = request_raw_with_retry(
        addr,
        "POST",
        "/v1/completions",
        Some(r#"{"model":"glm-completions-http","prompt":"hi","max_tokens":2}"#),
    )
    .await;

    assert!(
        response.starts_with("HTTP/1.1 200"),
        "OpenAI completions should succeed, got response: {response}"
    );
    let (_, body) = response
        .split_once("\r\n\r\n")
        .expect("HTTP response should include headers");
    let response: Value = serde_json::from_str(body).expect("response should be JSON");
    assert_eq!(response["object"], "text_completion");
    assert_eq!(response["model"], "glm-completions-http");
    assert_eq!(response["choices"][0]["index"], 0);
    assert_eq!(response["choices"][0]["text"], "  ");
    assert_eq!(response["choices"][0]["finish_reason"], "stop");
    assert_eq!(response["usage"]["prompt_tokens"], 2);
    assert_eq!(response["usage"]["completion_tokens"], 2);

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_server_accepts_streaming_openai_completions() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-completions-http-stream",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let response = request_raw_with_retry(
        addr,
        "POST",
        "/v1/completions",
        Some(
            r#"{"model":"glm-completions-http-stream","prompt":"hi","max_tokens":2,"stream":true}"#,
        ),
    )
    .await;

    assert!(
        response.starts_with("HTTP/1.1 200"),
        "streaming completions should succeed, got response: {response}"
    );
    assert!(
        response
            .to_ascii_lowercase()
            .contains("content-type: text/event-stream"),
        "streaming completions must return SSE content-type, got response: {response}"
    );
    let events = parse_sse_data(&response);
    assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
    let chunks = events
        .iter()
        .filter(|event| event.as_str() != "[DONE]")
        .map(|event| serde_json::from_str::<Value>(event))
        .collect::<Result<Vec<_>, _>>()
        .expect("SSE data chunks should be JSON");
    assert!(
        chunks
            .iter()
            .any(|chunk| chunk["object"] == "text_completion"),
        "expected OpenAI completion chunks, got {chunks:?}"
    );
    assert!(
        chunks.iter().any(|chunk| chunk["finish_reason"] == "stop"
            || chunk["choices"][0]["finish_reason"] == "stop"),
        "expected final stop chunk, got {chunks:?}"
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
async fn http_server_reports_plain_worker_server_info_for_sgl_router() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-router-plain",
        "--tp-size",
        "2",
        "--dp-size",
        "3",
        "--max-running-requests",
        "17",
        "--max-prefill-tokens",
        "2048",
        "--max-total-tokens",
        "4096",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let server_info = get_json_with_retry(addr, "/server_info").await;

    assert_eq!(server_info["served_model_name"], "glm-router-plain");
    assert_eq!(server_info["model_path"], "dummy");
    assert_eq!(server_info["tp_size"], 2);
    assert_eq!(server_info["dp_size"], 3);
    assert_eq!(server_info["load_balance_method"], "round_robin");
    assert_eq!(server_info["max_running_requests"], 17);
    assert_eq!(server_info["max_num_reqs"], 17);
    assert_eq!(server_info["max_prefill_tokens"], 2048);
    assert_eq!(server_info["max_total_tokens"], 4096);
    assert_eq!(server_info["disaggregation_mode"], "null");
    assert!(server_info.get("disaggregation_bootstrap_port").is_none());
    assert!(server_info.get("kv_events").is_none());

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_prefill_server_info_reports_follow_bootstrap_room_for_gateway_policy() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-prefill",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-bootstrap-port",
        "8999",
        "--tp-size",
        "2",
        "--dp-size",
        "4",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let server_info = get_json_with_retry(addr, "/server_info").await;

    assert_eq!(server_info["served_model_name"], "glm-prefill");
    assert_eq!(server_info["disaggregation_mode"], "prefill");
    assert_eq!(server_info["load_balance_method"], "follow_bootstrap_room");
    assert_eq!(server_info["tp_size"], 2);
    assert_eq!(server_info["dp_size"], 4);

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_server_reports_model_info_and_legacy_aliases_for_sgl_gateway_discovery() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy-model",
        "--tokenizer-path",
        "dummy-tokenizer",
        "--served-model-name",
        "glm-router-discovery",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let model_info = get_json_with_retry(addr, "/model_info").await;
    let legacy_model_info = get_json_with_retry(addr, "/get_model_info").await;
    let legacy_server_info = get_json_with_retry(addr, "/get_server_info").await;

    assert_eq!(model_info["model_path"], "dummy-model");
    assert_eq!(model_info["tokenizer_path"], "dummy-tokenizer");
    assert_eq!(model_info["is_generation"], true);
    assert_eq!(model_info["model_type"], "");
    assert_eq!(
        model_info["architectures"].as_array().unwrap().len(),
        0,
        "gateway discovery should be able to deserialize architectures as an array"
    );
    assert_eq!(legacy_model_info, model_info);
    assert_eq!(
        legacy_server_info["served_model_name"],
        "glm-router-discovery"
    );
    assert_eq!(legacy_server_info["disaggregation_mode"], "null");

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_server_model_info_reports_local_model_architectures_for_gateway_discovery() {
    let model_dir = temp_model_dir("http-model-info-architectures");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "model_type": "glm_moe_dsa",
  "architectures": ["GlmMoEDSAModel"]
}"#,
    )
    .expect("config should be written");
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model dir should be utf-8"),
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let model_info = get_json_with_retry(addr, "/model_info").await;

    assert_eq!(model_info["model_type"], "glm_moe_dsa");
    assert_eq!(
        model_info["architectures"],
        serde_json::json!(["GlmMoEDSAModel"])
    );

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_server_update_weights_from_disk_validates_artifacts_and_updates_model_info() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "old-model",
        "--served-model-name",
        "tiny-http",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let runtime = RouterRuntime::new(Engine::new(
        ByteTokenizer,
        Scheduler::new(HttpReloadingWorker::default()),
    ));
    let service =
        HttpRouterService::new(runtime, RouterGetModelInfoResponse::from_server_args(&args));
    let model_dir = temp_model_dir("http-update-weights");
    write_minimal_generic_model_artifacts(&model_dir);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let update = request_json_dynamic_with_retry(
        addr,
        "POST",
        "/update_weights_from_disk",
        serde_json::json!({
            "model_path": model_dir.to_string_lossy(),
            "load_format": "safetensors"
        })
        .to_string(),
    )
    .await;
    assert_eq!(update["success"], true, "update response: {update}");
    assert!(update["message"].as_str().unwrap().contains("registered"));
    assert_eq!(update["num_paused_requests"], 0);

    let model_info = get_json_with_retry(addr, "/model_info").await;
    let model_path = model_dir.to_string_lossy().to_string();
    assert_eq!(model_info["model_path"], model_path);
    assert_eq!(model_info["tokenizer_path"], model_path);
    assert_eq!(model_info["model_type"], "qwen3");
    assert_eq!(
        model_info["architectures"],
        serde_json::json!(["Qwen3ForCausalLM"])
    );
    assert_eq!(model_info["vocab_size"], 3);
    assert_eq!(model_info["max_context_length"], 32);
    assert!(
        model_info["weight_version"]
            .as_str()
            .unwrap()
            .starts_with("safetensors-sha256:")
    );

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_server_update_weights_from_disk_rejects_invalid_requests() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "old-model",
        "--served-model-name",
        "tiny-http",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let empty_path = request_raw_with_retry(
        addr,
        "POST",
        "/update_weights_from_disk",
        Some(r#"{"model_path":"  "}"#),
    )
    .await;
    assert!(empty_path.starts_with("HTTP/1.1 400"));
    assert!(empty_path.contains("model_path"));

    let unsupported_format = request_raw_with_retry(
        addr,
        "POST",
        "/update_weights_from_disk",
        Some(r#"{"model_path":"some-model","load_format":"gguf"}"#),
    )
    .await;
    assert!(unsupported_format.starts_with("HTTP/1.1 400"));
    assert!(unsupported_format.contains("load_format"));

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_server_update_weight_version_updates_model_info() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "old-model",
        "--served-model-name",
        "tiny-http",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let update = request_json_dynamic_with_retry(
        addr,
        "POST",
        "/update_weight_version",
        serde_json::json!({
            "new_version": "checkpoint-42",
            "abort_all_requests": false
        })
        .to_string(),
    )
    .await;
    assert_eq!(update["success"], true, "update response: {update}");
    assert_eq!(update["new_version"], "checkpoint-42");
    assert!(
        update["message"]
            .as_str()
            .unwrap()
            .contains("checkpoint-42")
    );

    let model_info = get_json_with_retry(addr, "/model_info").await;
    let legacy_model_info = get_json_with_retry(addr, "/get_model_info").await;
    assert_eq!(model_info["weight_version"], "checkpoint-42");
    assert_eq!(legacy_model_info["weight_version"], "checkpoint-42");

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_server_reports_remote_instance_transfer_engine_info() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "tiny-http",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let engine_info = EngineInfoBootstrapService::default();
    engine_info
        .state()
        .lock()
        .expect("engine info state should lock")
        .register_transfer_engine_info(TransferEngineInfoRegistration {
            tp_rank: 0,
            transfer_engine_info: TransferEngineInfo {
                session_id: "session-a".to_string(),
                weights_info_dict: serde_json::json!({
                    "layer.0": {
                        "addr": 4096,
                        "length": 8192,
                    }
                }),
            },
        });
    let addr = unused_local_addr();
    let service =
        build_reference_http_router_service(&args).with_engine_info_bootstrap_service(engine_info);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let response = get_json_with_retry(addr, "/remote_instance_transfer_engine_info?rank=0").await;
    let legacy_response =
        get_json_with_retry(addr, "/get_remote_instance_transfer_engine_info?rank=0").await;

    assert_eq!(response["rank"], 0);
    assert_eq!(
        response["remote_instance_transfer_engine_info"][0],
        "session-a"
    );
    assert_eq!(
        response["remote_instance_transfer_engine_info"][1]["layer.0"]["addr"],
        4096
    );
    assert_eq!(legacy_response, response);

    let missing_rank =
        request_raw_with_retry(addr, "GET", "/remote_instance_transfer_engine_info", None).await;
    assert!(missing_rank.starts_with("HTTP/1.1 400"));
    assert!(missing_rank.contains("Missing or invalid rank parameter"));

    let missing_info = request_raw_with_retry(
        addr,
        "GET",
        "/remote_instance_transfer_engine_info?rank=1",
        None,
    )
    .await;
    assert!(missing_info.starts_with("HTTP/1.1 400"));
    assert!(missing_info.contains("Failed to get transfer engine info for rank 1"));

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_server_get_weights_by_name_reads_safetensors_parameter() {
    let model_dir = temp_model_dir("http-get-weights");
    write_minimal_generic_model_artifacts_with_weight_values(&model_dir, &[1.5, 2.5, 3.5]);
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model dir should be utf-8"),
        "--device",
        "cpu",
        "--served-model-name",
        "tiny-http",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let response = request_json_dynamic_with_retry(
        addr,
        "POST",
        "/get_weights_by_name",
        serde_json::json!({
            "name": "model.embed_tokens.weight",
            "truncate_size": 2
        })
        .to_string(),
    )
    .await;
    assert_eq!(response["parameter"], serde_json::json!([1.5, 2.5]));

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_server_reports_runtime_loads_for_sglang_control_plane() {
    let mut scheduler = Scheduler::with_cache_resources(
        HttpTwoStepWorker,
        RadixCache::default(),
        CachePageAllocator::new(4),
    );
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("http-load-waiting"),
        vec![1, 2, 3],
        RuntimeSamplingParams::new(1),
    ));
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-loads",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let runtime = RouterRuntime::new(Engine::new(ByteTokenizer, scheduler));
    let service =
        HttpRouterService::new(runtime, RouterGetModelInfoResponse::from_server_args(&args));
    let addr = unused_local_addr();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let loads = get_json_with_retry(addr, "/v1/loads?include=core").await;
    let legacy_load = get_json_with_retry(addr, "/get_load").await;

    assert_eq!(loads["version"], env!("CARGO_PKG_VERSION"));
    assert!(loads["timestamp"].as_u64().is_some());
    assert_eq!(
        loads["aggregate"]["total_tokens"], 1,
        "sgl-model-gateway reads aggregate.total_tokens from /v1/loads?include=core"
    );
    assert_eq!(loads["loads"][0]["dp_rank"], 0);
    assert_eq!(loads["loads"][0]["num_waiting_reqs"], 1);
    assert_eq!(loads["loads"][0]["num_running_reqs"], 0);
    assert_eq!(loads["loads"][0]["waiting_queue_depth"], 1);
    assert_eq!(loads["loads"][0]["decode_queue_depth"], 0);
    assert_eq!(loads["loads"][0]["available_cache_pages"], 4);
    assert_eq!(legacy_load[0]["num_reqs"], 1);
    assert_eq!(legacy_load[0]["num_waiting_reqs"], 1);

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_server_abort_request_removes_queued_request() {
    let mut scheduler = Scheduler::with_cache_resources(
        HttpTwoStepWorker,
        RadixCache::default(),
        CachePageAllocator::new(4),
    );
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("http-abort"),
        vec![1, 2, 3],
        RuntimeSamplingParams::new(1),
    ));
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-abort",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let runtime = RouterRuntime::new(Engine::new(ByteTokenizer, scheduler));
    let service =
        HttpRouterService::new(runtime, RouterGetModelInfoResponse::from_server_args(&args));
    let addr = unused_local_addr();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let aborted = post_json_with_retry(addr, "/abort_request", r#"{"rid":"http-abort"}"#).await;
    assert_eq!(aborted["success"], true);
    assert_eq!(aborted["message"], "request aborted");

    let loads = get_json_with_retry(addr, "/get_loads").await;
    assert_eq!(loads["loads"][0]["waiting_queue_depth"], 0);

    let missing = post_json_with_retry(addr, "/abort_request", r#"{"rid":"missing"}"#).await;
    assert_eq!(missing["success"], false);
    assert_eq!(missing["message"], "request not found");

    let empty = request_raw_with_retry(addr, "POST", "/abort_request", Some(r#"{"rid":""}"#)).await;
    assert!(
        empty.starts_with("HTTP/1.1 400"),
        "empty rid should be rejected, got {empty}"
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
async fn http_server_abort_all_removes_all_queued_requests() {
    let mut scheduler = Scheduler::with_cache_resources(
        HttpTwoStepWorker,
        RadixCache::default(),
        CachePageAllocator::new(4),
    );
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("http-abort-a"),
        vec![1],
        RuntimeSamplingParams::new(1),
    ));
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("http-abort-b"),
        vec![2],
        RuntimeSamplingParams::new(1),
    ));
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-abort-all",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let runtime = RouterRuntime::new(Engine::new(ByteTokenizer, scheduler));
    let service =
        HttpRouterService::new(runtime, RouterGetModelInfoResponse::from_server_args(&args));
    let addr = unused_local_addr();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let aborted = post_json_with_retry(addr, "/abort_request", r#"{"abort_all":true}"#).await;
    assert_eq!(aborted["success"], true);
    assert_eq!(aborted["message"], "aborted 2 request(s)");

    let loads = get_json_with_retry(addr, "/get_loads").await;
    assert_eq!(loads["loads"][0]["waiting_queue_depth"], 0);

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_server_flush_cache_uses_router_runtime_state() {
    let mut busy_scheduler = Scheduler::new(HttpTwoStepWorker);
    busy_scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("http-flush-waiting"),
        vec![1, 2, 3],
        RuntimeSamplingParams::new(1),
    ));
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let busy_runtime = RouterRuntime::new(Engine::new(ByteTokenizer, busy_scheduler));
    let busy_service = HttpRouterService::new(
        busy_runtime,
        RouterGetModelInfoResponse::from_server_args(&args),
    );
    let busy_addr = unused_local_addr();
    let (busy_shutdown_tx, busy_shutdown_rx) = oneshot::channel();
    let busy_server = tokio::spawn(async move {
        serve_http_router_with_shutdown(busy_addr, busy_service, async move {
            let _ = busy_shutdown_rx.await;
        })
        .await
    });

    let busy_response = request_raw_with_retry(busy_addr, "POST", "/flush_cache", None).await;
    assert!(
        busy_response.starts_with("HTTP/1.1 400"),
        "flush_cache must reject when requests are queued, got {busy_response}"
    );

    busy_shutdown_tx
        .send(())
        .expect("server should still be running");
    busy_server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");

    let idle_service = build_reference_http_router_service(&args);
    let idle_addr = unused_local_addr();
    let (idle_shutdown_tx, idle_shutdown_rx) = oneshot::channel();
    let idle_server = tokio::spawn(async move {
        serve_http_router_with_shutdown(idle_addr, idle_service, async move {
            let _ = idle_shutdown_rx.await;
        })
        .await
    });

    let idle_response = request_raw_with_retry(idle_addr, "POST", "/flush_cache", None).await;
    assert!(
        idle_response.starts_with("HTTP/1.1 200"),
        "flush_cache should succeed when runtime is idle, got {idle_response}"
    );
    assert!(idle_response.contains("Cache flushed."));

    idle_shutdown_tx
        .send(())
        .expect("server should still be running");
    idle_server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_server_pause_and_continue_generation_use_router_runtime_state() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let service = build_reference_http_router_service(&args);
    let addr = unused_local_addr();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let pause_response = post_json_with_retry(addr, "/pause_generation", "{}").await;
    assert_eq!(pause_response["success"], true);
    assert_eq!(pause_response["message"], "generation paused");

    let paused_generate =
        request_raw_with_retry(addr, "POST", "/generate", Some(r#"{"text":"hello"}"#)).await;
    assert!(
        paused_generate.starts_with("HTTP/1.1 412"),
        "paused generate should be rejected, got {paused_generate}"
    );

    let continue_response = post_json_with_retry(addr, "/continue_generation", "{}").await;
    assert_eq!(continue_response["success"], true);
    assert_eq!(continue_response["message"], "generation continued");

    let generated = post_json_with_retry(
        addr,
        "/generate",
        r#"{"text":"hello","sampling_params":{"max_new_tokens":1}}"#,
    )
    .await;
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
async fn http_server_start_and_stop_profile_writes_trace_file() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let output_dir = unique_profile_dir("http-profile");
    let service = build_reference_http_router_service(&args);
    let addr = unused_local_addr();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let start_body = Box::leak(
        serde_json::json!({"output_dir": output_dir.to_string_lossy()})
            .to_string()
            .into_boxed_str(),
    );
    let start = post_json_with_retry(addr, "/start_profile", start_body).await;
    assert_eq!(start["success"], true);
    assert!(
        start["message"]
            .as_str()
            .expect("start message should be a string")
            .contains("profile started")
    );

    let stop = post_json_with_retry(addr, "/stop_profile", "{}").await;
    assert_eq!(stop["success"], true);
    assert!(
        stop["message"]
            .as_str()
            .expect("stop message should be a string")
            .contains("profile stopped")
    );

    let entries = fs::read_dir(&output_dir)
        .expect("profile output directory should exist")
        .collect::<Result<Vec<_>, _>>()
        .expect("profile directory should be readable");
    assert_eq!(entries.len(), 1);
    let profile: serde_json::Value = serde_json::from_slice(
        &fs::read(entries[0].path()).expect("profile file should be readable"),
    )
    .expect("profile file should contain JSON");
    assert_eq!(profile["profile"]["transport"], "axum-http");
    assert!(profile["profile"]["duration_ms"].as_u64().is_some());

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
    fs::remove_dir_all(output_dir).expect("profile temp directory should clean up");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_server_rejects_disaggregated_generate_without_transfer_runtime() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let response = request_raw_with_retry(
        addr,
        "POST",
        "/generate",
        Some(r#"{"text":"hello","bootstrap_host":"10.0.0.8","bootstrap_port":8200,"bootstrap_room":77}"#),
    )
    .await;

    assert!(response.starts_with("HTTP/1.1 501"));
    assert!(response.contains("PD transfer-enabled runtime"));

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_prefill_server_reports_router_server_info_with_kv_events() {
    let bootstrap_addr = unused_local_addr();
    let zmq_ports = unused_contiguous_local_ports_excluding(2, &[bootstrap_addr.port()]);
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-router-prefill",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--tp-size",
        "2",
        "--dp-size",
        "1",
        "--page-size",
        "64",
        "--kv-cache-dtype",
        "bfloat16",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-bootstrap-port",
        &bootstrap_addr.port().to_string(),
        "--disaggregation-zmq-ports",
        &format!("{}-{}", zmq_ports[0], zmq_ports[1]),
        "--num-reserved-decode-tokens",
        "64",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let service = build_reference_prefill_http_router_service(&args);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let server_info = get_json_with_retry(addr, "/server_info").await;

    assert_eq!(server_info["served_model_name"], "glm-router-prefill");
    assert_eq!(server_info["disaggregation_mode"], "prefill");
    assert_eq!(
        server_info["disaggregation_bootstrap_port"],
        bootstrap_addr.port()
    );
    assert_eq!(server_info["kv_events"]["publisher"], "zmq");
    assert_eq!(server_info["kv_events"]["endpoint_host"], "127.0.0.1");
    assert_eq!(server_info["kv_events"]["endpoint_port_base"], zmq_ports[0]);
    assert_eq!(server_info["kv_events"]["topic"], "");
    assert_eq!(server_info["kv_events"]["block_size"], 64);
    assert_eq!(server_info["kv_events"]["dp_size"], 1);
    assert!(
        server_info.get("kv_cache").is_none(),
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
    let service = build_reference_prefill_http_router_service(&args);
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

    {
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
    }
    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_prefill_server_accepts_batched_disaggregated_token_generate_requests() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-prefill-http-batch",
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
    let service = build_reference_prefill_http_router_service(&args);
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
        r#"{"request_id":["http-pd-batch-a","http-pd-batch-b"],"input_ids":[[1,2,3],[4,5]],"sampling_params":{"max_new_tokens":1},"bootstrap_host":["10.0.0.8","10.0.0.8"],"bootstrap_port":[8200,8200],"bootstrap_room":[77,78]}"#,
    )
    .await;

    let generated = generated
        .as_array()
        .expect("batched /generate should return an array of results");
    assert_eq!(generated.len(), 2);
    assert_eq!(generated[0]["request_id"], "http-pd-batch-a");
    assert_eq!(generated[1]["request_id"], "http-pd-batch-b");
    assert_eq!(generated[0]["usage"]["prompt_tokens"], 3);
    assert_eq!(generated[1]["usage"]["prompt_tokens"], 2);

    {
        let runtime = inspected_service
            .runtime()
            .lock()
            .expect("runtime lock should be held");
        let worker = runtime.engine().scheduler().worker();
        let summary = worker
            .last_transfer_summary()
            .expect("PD prefill batch should record transfer summary");
        assert_eq!(summary.submitted_spans(), 2);
        assert_eq!(worker.transfer_executor().transferred_rooms(), &[77, 78]);
    }
    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_prefill_server_accepts_batched_disaggregated_text_generate_requests() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-prefill-http-text-batch",
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
    let service = build_reference_prefill_http_router_service(&args);
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
        r#"{"request_id":["http-pd-text-batch-a","http-pd-text-batch-b"],"text":["hello","hi"],"sampling_params":{"max_new_tokens":1},"bootstrap_host":["10.0.0.8","10.0.0.8"],"bootstrap_port":[8200,8200],"bootstrap_room":[87,88]}"#,
    )
    .await;

    let generated = generated
        .as_array()
        .expect("batched text /generate should return an array of results");
    assert_eq!(generated.len(), 2);
    assert_eq!(generated[0]["request_id"], "http-pd-text-batch-a");
    assert_eq!(generated[1]["request_id"], "http-pd-text-batch-b");
    assert_eq!(generated[0]["text"], " ");
    assert_eq!(generated[1]["text"], " ");

    {
        let runtime = inspected_service
            .runtime()
            .lock()
            .expect("runtime lock should be held");
        let worker = runtime.engine().scheduler().worker();
        let summary = worker
            .last_transfer_summary()
            .expect("PD prefill text batch should record transfer summary");
        assert_eq!(summary.submitted_spans(), 2);
        assert_eq!(worker.transfer_executor().transferred_rooms(), &[87, 88]);
    }
    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_prefill_server_accepts_batched_disaggregated_openai_completions() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-prefill-completion-batch",
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
    let service = build_reference_prefill_http_router_service(&args);
    let inspected_service = service.clone();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let completion = post_json_with_retry(
        addr,
        "/v1/completions",
        r#"{"request_id":["http-pd-completion-batch-a","http-pd-completion-batch-b"],"model":"glm-prefill-completion-batch","prompt":["hi","hey"],"max_tokens":1,"bootstrap_host":["10.0.0.8","10.0.0.8"],"bootstrap_port":[8200,8200],"bootstrap_room":[97,98]}"#,
    )
    .await;

    assert_eq!(completion["object"], "text_completion");
    assert_eq!(completion["model"], "glm-prefill-completion-batch");
    let choices = completion["choices"]
        .as_array()
        .expect("batched completions should return one choice per prompt");
    assert_eq!(choices.len(), 2);
    assert_eq!(choices[0]["index"], 0);
    assert_eq!(choices[0]["text"], " ");
    assert_eq!(choices[0]["finish_reason"], "stop");
    assert_eq!(choices[1]["index"], 1);
    assert_eq!(choices[1]["text"], " ");
    assert_eq!(choices[1]["finish_reason"], "stop");
    assert_eq!(completion["usage"]["prompt_tokens"], 5);
    assert_eq!(completion["usage"]["completion_tokens"], 2);

    {
        let runtime = inspected_service
            .runtime()
            .lock()
            .expect("runtime lock should be held");
        let worker = runtime.engine().scheduler().worker();
        let summary = worker
            .last_transfer_summary()
            .expect("PD prefill completion batch should record transfer summary");
        assert_eq!(summary.submitted_spans(), 2);
        assert_eq!(worker.transfer_executor().transferred_rooms(), &[97, 98]);
    }
    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_prefill_server_accepts_batched_disaggregated_chat_completions() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-prefill-chat-batch",
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
    let service = build_reference_prefill_http_router_service(&args);
    let inspected_service = service.clone();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let completion = post_json_with_retry(
        addr,
        "/v1/chat/completions",
        r#"{"request_id":["http-pd-chat-batch-a","http-pd-chat-batch-b"],"model":"glm-prefill-chat-batch","messages":[{"role":"user","content":"hi"}],"n":2,"max_tokens":1,"bootstrap_host":["10.0.0.8","10.0.0.8"],"bootstrap_port":[8200,8200],"bootstrap_room":[107,108]}"#,
    )
    .await;

    assert_eq!(completion["object"], "chat.completion");
    assert_eq!(completion["model"], "glm-prefill-chat-batch");
    let choices = completion["choices"]
        .as_array()
        .expect("batched chat completions should return one choice per n");
    assert_eq!(choices.len(), 2);
    assert_eq!(choices[0]["index"], 0);
    assert_eq!(choices[0]["message"]["role"], "assistant");
    assert_eq!(choices[0]["message"]["content"], " ");
    assert_eq!(choices[0]["finish_reason"], "stop");
    assert_eq!(choices[1]["index"], 1);
    assert_eq!(choices[1]["message"]["role"], "assistant");
    assert_eq!(choices[1]["message"]["content"], " ");
    assert_eq!(choices[1]["finish_reason"], "stop");
    assert_eq!(completion["usage"]["prompt_tokens"], 4);
    assert_eq!(completion["usage"]["completion_tokens"], 2);

    {
        let runtime = inspected_service
            .runtime()
            .lock()
            .expect("runtime lock should be held");
        let worker = runtime.engine().scheduler().worker();
        let summary = worker
            .last_transfer_summary()
            .expect("PD prefill chat batch should record transfer summary");
        assert_eq!(summary.submitted_spans(), 2);
        assert_eq!(worker.transfer_executor().transferred_rooms(), &[107, 108]);
    }
    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_pd_server_polls_async_transfer_before_decode() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-pd-http",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-decode-polling-interval",
        "1",
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let transfer_executor = registered_recording_mooncake_executor(0x3000, 64, 0, 17);
    let service = build_reference_pd_http_router_service(
        &args,
        DecodeBootstrapRegistry::default(),
        transfer_executor,
    );
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
        r#"{"request_id":"http-pd-mooncake","text":"hi","sampling_params":{"max_new_tokens":2},"bootstrap_host":"10.0.0.8","bootstrap_port":8200,"bootstrap_room":41}"#,
    )
    .await;

    assert_eq!(generated["request_id"], "http-pd-mooncake");
    assert_eq!(generated["usage"]["completion_tokens"], 2);

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_server_poll_transfers_advances_async_pd_batches() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-pd-http-poll",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-decode-polling-interval",
        "0",
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let transfer_executor = registered_recording_mooncake_executor(0x4000, 64, 0, 18);
    let service = build_reference_pd_http_router_service(
        &args,
        DecodeBootstrapRegistry::default(),
        transfer_executor,
    );
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_http_router_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let waiting = request_raw_with_retry(
        addr,
        "POST",
        "/generate",
        Some(
            r#"{"request_id":"http-pd-poll","text":"hi","sampling_params":{"max_new_tokens":2},"bootstrap_host":"10.0.0.8","bootstrap_port":8200,"bootstrap_room":42}"#,
        ),
    )
    .await;
    assert!(
        waiting.contains("500 Internal Server Error"),
        "first response should wait on async KV transfer, got {waiting}"
    );
    assert!(
        waiting.contains("decode request http-pd-poll is not ready"),
        "first response should expose decode wait, got {waiting}"
    );

    let poll = post_json_with_retry(addr, "/poll_transfers", "{}").await;

    assert_eq!(poll["completed_batches"], 1);
    assert_eq!(poll["pending_batches"], 0);
    assert_eq!(
        poll["completed_descriptor_checksums"],
        serde_json::json!(["e37f348404de723244ca004c09d05fccfdcad793f9013bd1f882c332d1c50b5d"])
    );
    assert_eq!(poll["pending_descriptor_checksums"], serde_json::json!([]));

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mooncake_prefill_http_uses_bootstrap_kv_layout_for_transfer() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-mooncake-prefill-http",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-decode-polling-interval",
        "1",
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("args should parse");
    let addr = unused_local_addr();
    let bootstrap_service = PrefillBootstrapService::default();
    {
        let mut state = bootstrap_service
            .state()
            .lock()
            .expect("bootstrap state lock should be held");
        state
            .ingest_mooncake_bootstrap_frame(&kv_args_frame("session-a", &[0x9000], 128))
            .expect("KVArgs frame should parse");
        state
            .ingest_mooncake_bootstrap_frame(&transfer_metadata_frame(34, "session-a", &[4, 5]))
            .expect("transfer metadata frame should parse");
    }
    let transfer_executor = registered_recording_mooncake_executor(0x2000, 128, 0xdead_0000, 7);
    let service = build_reference_mooncake_prefill_http_router_service(
        &args,
        bootstrap_service,
        transfer_executor,
    );
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
        r#"{"request_id":"http-pd-mooncake-prefill","text":"hi","sampling_params":{"max_new_tokens":2},"bootstrap_host":"10.0.0.8","bootstrap_port":8200,"bootstrap_room":34}"#,
    )
    .await;

    assert_eq!(generated["request_id"], "http-pd-mooncake-prefill");
    assert_eq!(generated["usage"]["completion_tokens"], 2);

    {
        let runtime = inspected_service
            .runtime()
            .lock()
            .expect("runtime lock should be held");
        let worker = runtime.engine().scheduler().worker();
        let submitted_requests = &worker
            .transfer_executor()
            .inner()
            .submitter()
            .submitted_requests;
        assert_eq!(submitted_requests.len(), 1);
        assert_eq!(submitted_requests[0].len(), 2);
        assert_eq!(submitted_requests[0][0].target_id, 7);
        assert_eq!(submitted_requests[0][0].target_offset, 0x9000 + 4 * 128);
        assert_eq!(submitted_requests[0][1].target_id, 7);
        assert_eq!(submitted_requests[0][1].target_offset, 0x9000 + 5 * 128);
    }
    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prefill_http_launch_rejects_unstartable_dummy_mooncake_runtime() {
    let http_addr = unused_local_addr();
    let bootstrap_addr = unused_local_addr();
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--host",
        "127.0.0.1",
        "--port",
        &http_addr.port().to_string(),
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-bootstrap-port",
        &bootstrap_addr.port().to_string(),
    ])
    .expect("args should parse");

    let error = launch_http_server_with_shutdown(args, async {}).await;
    let error = error.expect_err("dummy prefill worker should reject Mooncake PD startup");

    assert_dummy_mooncake_startup_error(&error);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_launch_starts_engine_info_bootstrap_and_serves_remote_info() {
    let http_addr = unused_local_addr();
    let engine_info_addr = unused_local_addr();
    let model_dir = temp_model_dir("http-launch-engine-info");
    write_minimal_generic_model_artifacts(&model_dir);
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_dir.to_str().expect("temp model dir should be utf-8"),
        "--device",
        "cpu",
        "--served-model-name",
        "tiny-http",
        "--host",
        "127.0.0.1",
        "--port",
        &http_addr.port().to_string(),
        "--engine-info-bootstrap-port",
        &engine_info_addr.port().to_string(),
    ])
    .expect("args should parse");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(launch_http_server_with_shutdown(args, async move {
        let _ = shutdown_rx.await;
    }));

    let registration = request_raw_with_retry(
        engine_info_addr,
        "PUT",
        "/register_transfer_engine_info",
        Some(
            r#"{"tp_rank":0,"transfer_engine_info":{"session_id":"session-launch","weights_info_dict":{"layer.0":{"addr":4096,"length":8192}}}}"#,
        ),
    )
    .await;
    assert!(registration.starts_with("HTTP/1.1 200"), "{registration}");

    let response =
        get_json_with_retry(http_addr, "/remote_instance_transfer_engine_info?rank=0").await;
    assert_eq!(response["rank"], 0);
    assert_eq!(
        response["remote_instance_transfer_engine_info"][0],
        "session-launch"
    );
    assert_eq!(
        response["remote_instance_transfer_engine_info"][1]["layer.0"]["length"],
        8192
    );

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prefill_http_launch_routes_reject_unstartable_dummy_mooncake_runtime() {
    let http_addr = unused_local_addr();
    let bootstrap_addr = unused_local_addr();
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-prefill-launch-chat",
        "--host",
        "127.0.0.1",
        "--port",
        &http_addr.port().to_string(),
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-bootstrap-port",
        &bootstrap_addr.port().to_string(),
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("args should parse");

    let error = launch_http_server_with_shutdown(args, async {}).await;
    let error = error.expect_err("dummy prefill worker should reject Mooncake PD startup");

    assert_dummy_mooncake_startup_error(&error);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn decode_http_launch_rejects_unstartable_dummy_mooncake_runtime() {
    let http_addr = unused_local_addr();
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--served-model-name",
        "glm-decode-launch-chat",
        "--host",
        "127.0.0.1",
        "--port",
        &http_addr.port().to_string(),
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--kv-cache-dtype",
        "bfloat16",
        "--num-reserved-decode-tokens",
        "8",
    ])
    .expect("args should parse");

    let error = launch_http_server_with_shutdown(args, async {}).await;
    let error = error.expect_err("dummy decode worker should reject Mooncake PD startup");

    assert_dummy_mooncake_startup_error(&error);
}

fn assert_dummy_mooncake_startup_error(error: &ServerLaunchError) {
    #[cfg(not(feature = "mooncake-link"))]
    assert!(
        error
            .to_string()
            .contains("requires building sglang-srt with the mooncake-link feature"),
        "{error}"
    );
    #[cfg(feature = "mooncake-link")]
    assert!(
        error.to_string().contains("not a local directory"),
        "{error}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prefill_http_launch_registers_mooncake_zmq_routes() {
    let bootstrap_addr = unused_local_addr();
    let zmq_ports = unused_contiguous_local_ports_excluding(2, &[bootstrap_addr.port()]);
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--host",
        "127.0.0.1",
        "--tp-size",
        "2",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-bootstrap-port",
        &bootstrap_addr.port().to_string(),
        "--disaggregation-zmq-ports",
        &format!("{}-{}", zmq_ports[0], zmq_ports[1]),
    ])
    .expect("args should parse");
    let service = PrefillBootstrapService::default();
    register_prefill_mooncake_routes_from_args(&service, &args)
        .expect("prefill ZMQ routes should register");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_prefill_bootstrap_with_shutdown(bootstrap_addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let topology = get_json_with_retry(
        bootstrap_addr,
        "/route?prefill_dp_rank=-1&prefill_cp_rank=-1&target_tp_rank=-1&target_pp_rank=-1",
    )
    .await;
    let tp0 = get_json_with_retry(
        bootstrap_addr,
        "/route?prefill_dp_rank=0&prefill_cp_rank=0&target_tp_rank=0&target_pp_rank=0",
    )
    .await;
    let tp1 = get_json_with_retry(
        bootstrap_addr,
        "/route?prefill_dp_rank=0&prefill_cp_rank=0&target_tp_rank=1&target_pp_rank=0",
    )
    .await;

    assert_eq!(topology["attn_tp_size"], 2);
    assert_eq!(topology["dp_size"], 1);
    assert_eq!(tp0["rank_ip"], "127.0.0.1");
    assert_eq!(tp0["rank_port"], zmq_ports[0]);
    assert_eq!(tp1["rank_ip"], "127.0.0.1");
    assert_eq!(tp1["rank_port"], zmq_ports[1]);

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("servers should stop cleanly");
}

fn kv_args_frame(session_id: &str, dst_kv_ptrs: &[u64], dst_kv_item_len: usize) -> Vec<Vec<u8>> {
    vec![
        b"None".to_vec(),
        b"10.0.0.9".to_vec(),
        b"41001".to_vec(),
        session_id.as_bytes().to_vec(),
        pack_u64s(dst_kv_ptrs),
        pack_u64s(&[]),
        pack_list_of_buffers(&[]),
        b"1".to_vec(),
        b"8".to_vec(),
        dst_kv_item_len.to_string().into_bytes(),
    ]
}

fn transfer_metadata_frame(
    room: BootstrapRoom,
    session_id: &str,
    dst_kv_indices: &[i32],
) -> Vec<Vec<u8>> {
    vec![
        room.to_string().into_bytes(),
        b"10.0.0.9".to_vec(),
        b"41001".to_vec(),
        session_id.as_bytes().to_vec(),
        pack_i32s(dst_kv_indices),
        b"11".to_vec(),
        pack_list_of_buffers(&[]),
        b"1".to_vec(),
        b"64".to_vec(),
    ]
}

fn pack_u64s(values: &[u64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn pack_i32s(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn pack_list_of_buffers(buffers: &[Vec<u8>]) -> Vec<u8> {
    let mut packed = Vec::new();
    packed.extend_from_slice(&(buffers.len() as u64).to_le_bytes());
    for buffer in buffers {
        packed.extend_from_slice(&(buffer.len() as u64).to_le_bytes());
        packed.extend_from_slice(buffer);
    }
    packed
}

fn unused_local_addr() -> SocketAddr {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("ephemeral port should bind");
    listener
        .local_addr()
        .expect("ephemeral listener should have local addr")
}

fn temp_model_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("sglang-rs-{name}-{}", std::process::id()))
}

fn write_minimal_generic_model_artifacts(model_dir: &Path) {
    write_qwen3_model_artifacts(model_dir, &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
}

fn write_minimal_generic_model_artifacts_with_weight_values(model_dir: &Path, values: &[f32]) {
    assert!(values.len() <= 6, "Qwen3 embedding fixture has six values");
    let mut token_embeddings = [0.0_f32; 6];
    token_embeddings[..values.len()].copy_from_slice(values);
    write_qwen3_model_artifacts(model_dir, &token_embeddings);
}

fn write_qwen3_model_artifacts(model_dir: &Path, token_embeddings: &[f32]) {
    assert_eq!(token_embeddings.len(), 6);
    fs::create_dir_all(model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("config.json"),
        r#"{
  "architectures": ["Qwen3ForCausalLM"],
  "model_type": "qwen3",
  "vocab_size": 3,
  "num_hidden_layers": 1,
  "hidden_size": 2,
  "intermediate_size": 2,
  "num_attention_heads": 1,
  "num_key_value_heads": 1,
  "head_dim": 2,
  "hidden_act": "silu",
  "attention_bias": false,
  "rms_norm_eps": 0.000001,
  "rope_theta": 1000000.0,
  "max_position_embeddings": 32,
  "tie_word_embeddings": false,
  "eos_token_id": [2, 3]
}"#,
    )
    .expect("config should be written");
    fs::write(
        model_dir.join("tokenizer.json"),
        r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {"[UNK]": 0, "hello": 1, "world": 2},
    "unk_token": "[UNK]"
  }
}"#,
    )
    .expect("tokenizer should be written");

    let descriptors: Vec<(&str, Vec<usize>, Vec<f32>)> = vec![
        (
            "model.embed_tokens.weight",
            vec![3, 2],
            token_embeddings.to_vec(),
        ),
        ("model.norm.weight", vec![2], vec![1.0; 2]),
        (
            "lm_head.weight",
            vec![3, 2],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.self_attn.q_norm.weight",
            vec![2],
            vec![1.0; 2],
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.self_attn.k_norm.weight",
            vec![2],
            vec![1.0; 2],
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.input_layernorm.weight",
            vec![2],
            vec![1.0; 2],
        ),
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![2],
            vec![1.0; 2],
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            vec![2, 2],
            vec![0.0; 4],
        ),
    ];
    let mut payload = Vec::new();
    let mut fields = Vec::new();
    for (name, shape, values) in descriptors {
        let start = payload.len();
        payload.extend(values.into_iter().flat_map(f32::to_le_bytes));
        let shape = shape
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        fields.push(format!(
            r#""{name}":{{"dtype":"F32","shape":[{shape}],"data_offsets":[{start},{}]}}"#,
            payload.len()
        ));
    }
    let header = format!("{{{}}}", fields.join(","));
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&payload);
    fs::write(model_dir.join("model.safetensors"), bytes)
        .expect("safetensors shard should be written");
}

fn unused_contiguous_local_ports_excluding(count: u16, excluded_ports: &[u16]) -> Vec<u16> {
    for _ in 0..100 {
        let first = unused_local_addr().port();
        let Some(last) = first.checked_add(count - 1) else {
            continue;
        };
        if (first..=last).any(|port| excluded_ports.contains(&port)) {
            continue;
        }
        let listeners = (first..=last)
            .map(|port| TcpListener::bind(("127.0.0.1", port)))
            .collect::<Result<Vec<_>, _>>();
        if let Ok(listeners) = listeners {
            drop(listeners);
            return (first..=last).collect();
        }
    }
    panic!("contiguous local ports should be available");
}

struct RecordingMooncakeBackend {
    status: MooncakeTransferStatusCode,
    submitted_requests: Vec<Vec<MooncakeTransferRequest>>,
    freed_batches: Vec<MooncakeBatchId>,
}

impl RecordingMooncakeBackend {
    fn completed() -> Self {
        Self {
            status: MooncakeTransferStatusCode::Completed,
            submitted_requests: Vec::new(),
            freed_batches: Vec::new(),
        }
    }
}

impl MooncakeTransferSubmitter for RecordingMooncakeBackend {
    fn submit_transfer(
        &mut self,
        requests: &mut [MooncakeTransferRequest],
    ) -> Result<MooncakeBatchId, MooncakeError> {
        self.submitted_requests.push(requests.to_vec());
        Ok(500 + self.submitted_requests.len() as MooncakeBatchId - 1)
    }
}

impl MooncakeTransferStatusReader for RecordingMooncakeBackend {
    fn transfer_status(
        &mut self,
        _batch_id: MooncakeBatchId,
        _task_id: usize,
    ) -> Result<MooncakeTransferStatus, MooncakeError> {
        Ok(MooncakeTransferStatus {
            status: self.status as i32,
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

impl MooncakeMemoryRegistrar for RecordingMooncakeBackend {
    fn register_memory_batch(
        &mut self,
        _buffers: &mut [MooncakeBufferEntry],
        _location: &str,
    ) -> Result<(), MooncakeError> {
        Ok(())
    }

    fn unregister_memory_batch(&mut self, _addrs: &mut [*mut c_void]) -> Result<(), MooncakeError> {
        Ok(())
    }
}

fn registered_recording_mooncake_executor(
    source_base_addr: usize,
    page_size_bytes: usize,
    target_base_offset: u64,
    target_id: i32,
) -> MooncakeKvCacheTransferExecutor<RecordingMooncakeBackend> {
    let mut executor = MooncakeKvCacheTransferExecutor::new(
        RecordingMooncakeBackend::completed(),
        MooncakeKvCacheLayout {
            source_base_addr,
            page_size_bytes,
            target_base_offset,
        },
        MooncakeTransferTarget { target_id },
    )
    .with_memory_registrar(RecordingMooncakeBackend::completed());
    executor
        .register(
            TransferableKvCacheMemory::new(
                vec![TransferableKvCacheRegion {
                    base_addr: source_base_addr,
                    byte_len: page_size_bytes * 8,
                    page_size_bytes,
                }],
                page_size_bytes,
                KvCacheMemoryLocation::Cpu { numa_node: 0 },
            )
            .expect("test NexusKV descriptor should be valid"),
        )
        .expect("test Mooncake executor should register NexusKV memory");
    executor
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

    for _ in 0..100 {
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

async fn request_raw_with_retry(
    addr: SocketAddr,
    method: &'static str,
    path: &str,
    body: Option<&'static str>,
) -> String {
    let mut last_error = None;

    for _ in 0..100 {
        match request_raw(addr, method, path, body).await {
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

async fn request_json_dynamic_with_retry(
    addr: SocketAddr,
    method: &'static str,
    path: &str,
    body: String,
) -> Value {
    let raw = request_raw_dynamic_with_retry(addr, method, path, body).await;
    let (_, body) = raw
        .split_once("\r\n\r\n")
        .expect("HTTP response should include headers");
    serde_json::from_str(body).expect("HTTP response should contain JSON")
}

async fn request_raw_dynamic_with_retry(
    addr: SocketAddr,
    method: &'static str,
    path: &str,
    body: String,
) -> String {
    let mut last_error = None;

    for _ in 0..100 {
        match request_raw_dynamic(addr, method, path, body.clone()).await {
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
    let response = request_raw(addr, method, &path, body).await?;
    let (_, body) = response
        .split_once("\r\n\r\n")
        .expect("HTTP response should include headers");
    serde_json::from_str(body).map_err(std::io::Error::other)
}

async fn request_raw_dynamic(
    addr: SocketAddr,
    method: &'static str,
    path: &str,
    body: String,
) -> Result<String, std::io::Error> {
    let path = path.to_string();
    tokio::task::spawn_blocking(move || {
        let mut stream = TcpStream::connect(addr)?;
        let request = format!(
            "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(request.as_bytes())?;
        let mut response = String::new();
        stream.read_to_string(&mut response)?;
        Ok(response)
    })
    .await
    .expect("blocking HTTP request should join")
}

async fn request_raw(
    addr: SocketAddr,
    method: &'static str,
    path: &str,
    body: Option<&'static str>,
) -> Result<String, std::io::Error> {
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
        Ok(response)
    })
    .await
    .expect("blocking HTTP request should join")
}

async fn read_first_sse_event(
    addr: SocketAddr,
    path: &'static str,
    body: &'static str,
) -> Result<String, std::io::Error> {
    tokio::task::spawn_blocking(move || {
        let mut stream = TcpStream::connect(addr)?;
        stream.set_read_timeout(Some(Duration::from_secs(2)))?;
        let request = format!(
            "POST {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(request.as_bytes())?;

        let mut response = Vec::new();
        let mut buffer = [0_u8; 1024];
        loop {
            let read = stream.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            response.extend_from_slice(&buffer[..read]);
            if response
                .windows(b"\n\n".len())
                .any(|window| window == b"\n\n")
                && response.windows(b"data: ".len()).any(|window| window == b"data: ")
            {
                break;
            }
        }
        String::from_utf8(response).map_err(std::io::Error::other)
    })
    .await
    .expect("blocking streaming HTTP request should join")
}

fn parse_sse_data(raw_response: &str) -> Vec<String> {
    raw_response
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(ToString::to_string)
        .collect()
}
