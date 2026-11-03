// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use sglang_srt::cache::{CachePageAllocator, RadixCache};
use sglang_srt::distributed::dp_scheduler::DpScheduler;
use sglang_srt::distributed::dp_worker_pool::DpWorkerPool;
use sglang_srt::model_executor::ModelWorkerBatch;
use sglang_srt::scheduler::{
    ForwardMode, RequestStage, ScheduleBatch, ScheduledRequest, Scheduler,
};
use sglang_srt::transfer::RecordingKvStateEventPublisher;
use sglang_srt::types::{RequestId, SamplingParams};
use sglang_srt::worker::{BatchGeneratedTokens, GeneratedToken, ModelWorker};

#[derive(Clone, Default)]
struct MockModelWorker {
    seen_batches: Vec<ScheduleBatch>,
}

impl ModelWorker for MockModelWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        self.seen_batches.push(batch.clone());
        let tokens = batch
            .requests()
            .iter()
            .map(|req| {
                if req.stage() == RequestStage::DecodeForward {
                    GeneratedToken::unfinished(vec![200 + req.output_ids().len() as u32])
                } else {
                    GeneratedToken::unfinished(vec![100])
                }
            })
            .collect();
        BatchGeneratedTokens::from_batch(batch, tokens).expect("token count matches batch")
    }
}

fn create_request(id: &str, prompt_len: usize, max_new_tokens: usize) -> ScheduledRequest {
    let tokens = (0..prompt_len as u32).map(|x| x + 10).collect::<Vec<_>>();
    ScheduledRequest::new(
        RequestId::from(id),
        tokens,
        SamplingParams::new(max_new_tokens),
    )
}

#[test]
fn chunked_prefill_slices_prompt_and_emits_incremental_events() {
    let recorder = Arc::new(RecordingKvStateEventPublisher::new());
    let mut scheduler = Scheduler::with_cache_resources(
        MockModelWorker::default(),
        RadixCache::default(),
        CachePageAllocator::new(1024),
    )
    .with_chunked_prefill_size(Some(4))
    .with_event_publisher(recorder.clone());

    // Prompt of 10 tokens: chunks will be [0..4, 4..8, 8..10]
    let request = create_request("req-chunked-1", 10, 2);
    scheduler.enqueue(request);

    // Chunk 1: tokens 0..4
    let batch1 = scheduler.next_prefill_batch(1).expect("batch 1");
    assert_eq!(batch1.forward_mode(), ForwardMode::Prefill);
    assert_eq!(batch1.requests()[0].current_chunk_len(), 4);
    assert_eq!(batch1.requests()[0].prompt_tokens_processed(), 0);

    let outputs1 = scheduler
        .complete_batch(
            batch1,
            BatchGeneratedTokens::from_batch(
                &ScheduleBatch::prefill(vec![create_request("req-chunked-1", 10, 2)]),
                vec![GeneratedToken::unfinished(vec![999])],
            )
            .unwrap(),
        )
        .expect("complete batch 1");

    // Intermediate chunk should NOT produce output tokens to client
    assert!(outputs1.is_empty());
    // Request stays in waiting queue for next chunk
    assert_eq!(scheduler.waiting_queue_depth(), 1);
    assert_eq!(scheduler.decode_queue_depth(), 0);

    // Verify 1 event was published
    let events = recorder.stored_events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].prompt_tokens_processed, 4);
    assert_eq!(events[0].total_prompt_tokens, 10);
    assert!(!events[0].is_final_chunk);
    assert_eq!(events[0].boundary.start_token, 0);
    assert_eq!(events[0].boundary.end_token, 4);

    // Chunk 2: tokens 4..8
    let batch2 = scheduler.next_prefill_batch(1).expect("batch 2");
    assert_eq!(batch2.requests()[0].current_chunk_len(), 4);
    assert_eq!(batch2.requests()[0].prompt_tokens_processed(), 4);

    let outputs2 = scheduler
        .complete_batch(
            batch2,
            BatchGeneratedTokens::from_batch(
                &ScheduleBatch::prefill(vec![create_request("req-chunked-1", 10, 2)]),
                vec![GeneratedToken::unfinished(vec![999])],
            )
            .unwrap(),
        )
        .expect("complete batch 2");
    assert!(outputs2.is_empty());
    assert_eq!(recorder.stored_events().len(), 2);
    let events = recorder.stored_events();
    assert_eq!(events[1].prompt_tokens_processed, 8);
    assert_eq!(events[1].boundary.start_token, 4);
    assert_eq!(events[1].boundary.end_token, 8);
    assert!(!events[1].is_final_chunk);

    // Chunk 3 (final): tokens 8..10 (2 remaining tokens)
    let batch3 = scheduler.next_prefill_batch(1).expect("batch 3");
    assert_eq!(batch3.requests()[0].current_chunk_len(), 2);
    assert_eq!(batch3.requests()[0].prompt_tokens_processed(), 8);

    let outputs3 = scheduler
        .complete_batch(
            batch3,
            BatchGeneratedTokens::from_batch(
                &ScheduleBatch::prefill(vec![create_request("req-chunked-1", 10, 2)]),
                vec![GeneratedToken::unfinished(vec![42])],
            )
            .unwrap(),
        )
        .expect("complete batch 3");

    // Final chunk completes prefill and emits first decoded token!
    assert_eq!(outputs3.len(), 1);
    assert_eq!(outputs3[0].token_ids, vec![42]);
    assert!(!outputs3[0].finished);
    assert_eq!(scheduler.waiting_queue_depth(), 0);
    assert_eq!(scheduler.decode_queue_depth(), 1);

    let events = recorder.stored_events();
    assert_eq!(events.len(), 3);
    assert_eq!(events[2].prompt_tokens_processed, 10);
    assert_eq!(events[2].boundary.start_token, 8);
    assert_eq!(events[2].boundary.end_token, 10);
    assert!(events[2].is_final_chunk);

    // Verify RadixCache now contains the full 10-token prompt
    let test_req = create_request("test", 10, 1);
    let matched = scheduler.prefix_cache().match_prefix(test_req.input_ids());
    assert_eq!(matched.matched_token_count, 10);
}

#[test]
fn mixed_batch_schedules_decodes_and_prefill_chunks_together() {
    let mut scheduler = Scheduler::with_cache_resources(
        MockModelWorker::default(),
        RadixCache::default(),
        CachePageAllocator::new(1024),
    )
    .with_chunked_prefill_size(Some(4));

    // First run decode request 1 to decode stage
    scheduler.enqueue(create_request("req-dec-1", 2, 5));
    let _ = scheduler.dispatch_prefill_batch(1).unwrap();
    assert_eq!(scheduler.decode_queue_depth(), 1);

    // Now enqueue a large prompt to waiting queue
    scheduler.enqueue(create_request("req-prefill-large", 8, 3));
    assert_eq!(scheduler.waiting_queue_depth(), 1);

    // Schedule a mixed batch (batch_size 4, token_budget 16)
    let mixed_batch = scheduler.next_mixed_batch(4, 16).expect("mixed batch");
    assert_eq!(mixed_batch.forward_mode(), ForwardMode::Mixed);
    assert_eq!(mixed_batch.batch_size(), 2);

    let reqs = mixed_batch.requests();
    // First request is the decode request
    assert_eq!(reqs[0].request_id(), &RequestId::from("req-dec-1"));
    assert_eq!(reqs[0].stage(), RequestStage::DecodeForward);

    // Second request is the chunked prefill request (chunk size 4)
    assert_eq!(reqs[1].request_id(), &RequestId::from("req-prefill-large"));
    assert_eq!(reqs[1].stage(), RequestStage::PrefillForward);
    assert_eq!(reqs[1].current_chunk_len(), 4);

    // Verify ModelWorkerBatch converts this mixed batch without panic
    let worker_batch = ModelWorkerBatch::from_schedule_batch(&mixed_batch);
    assert_eq!(worker_batch.forward_mode(), ForwardMode::Mixed);
    assert_eq!(worker_batch.request_ids().len(), 2);
    // Request 0 (decode): 1 input token
    assert_eq!(worker_batch.input_token_counts()[0], 1);
    // Request 1 (chunked prefill): 4 input tokens
    assert_eq!(worker_batch.input_token_counts()[1], 4);
    assert_eq!(worker_batch.input_ids().len(), 5);
}

#[test]
fn dp_scheduler_with_chunked_prefill_and_shared_prefix_cache() {
    let recorder = Arc::new(RecordingKvStateEventPublisher::new());
    let worker_pool = DpWorkerPool::launch(2, |_rank| Ok::<_, String>(MockModelWorker::default()))
        .expect("launch mock worker pool");
    let allocators = vec![CachePageAllocator::new(512), CachePageAllocator::new(512)];

    let mut dp_scheduler = DpScheduler::new(worker_pool, allocators, 4, 1)
        .expect("dp scheduler")
        .with_chunked_prefill_size(Some(4))
        .with_event_publisher(recorder.clone());

    // Enqueue an 8-token request
    let req = create_request("dp-chunked-req", 8, 2);
    dp_scheduler.enqueue(req);

    // Dispatch and wait for chunk 1 (tokens 0..4)
    dp_scheduler.step().expect("step 1 dispatch");
    dp_scheduler
        .step_blocking()
        .expect("step 1 await completion");

    // Intermediate chunk should NOT have finished
    assert_eq!(dp_scheduler.drain_finished_outputs().len(), 0);

    // Event for chunk 1 was emitted
    let events = recorder.stored_events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].prompt_tokens_processed, 4);
    assert!(!events[0].is_final_chunk);

    // Dispatch and wait for chunk 2 (tokens 4..8)
    dp_scheduler.step().expect("step 2 dispatch");
    dp_scheduler
        .step_blocking()
        .expect("step 2 await completion");

    // Chunk 2 was final chunk: produced first token output!
    let step_outputs = dp_scheduler.drain_step_outputs();
    assert_eq!(step_outputs.len(), 1);
    assert_eq!(
        step_outputs[0].request_id,
        RequestId::from("dp-chunked-req")
    );

    let events = recorder.stored_events();
    assert_eq!(events.len(), 2);
    assert_eq!(events[1].prompt_tokens_processed, 8);
    assert!(events[1].is_final_chunk);

    // Verify shared prefix cache has all 8 tokens
    let test_req = create_request("test", 8, 1);
    assert_eq!(
        dp_scheduler
            .prefix_cache()
            .match_prefix(test_req.input_ids())
            .matched_token_count,
        8
    );
}

#[test]
fn prefix_cache_reuse_during_chunked_prefill() {
    let mut scheduler = Scheduler::with_cache_resources(
        MockModelWorker::default(),
        RadixCache::default(),
        CachePageAllocator::new(1024),
    )
    .with_chunked_prefill_size(Some(4));

    // Request A: 8 tokens [10, 11, 12, 13, 14, 15, 16, 17]
    scheduler.enqueue(create_request("req-A", 8, 2));

    // Run chunk 1 of req-A (tokens 0..4)
    let batch1 = scheduler.next_prefill_batch(1).unwrap();
    let _ = scheduler
        .complete_batch(
            batch1,
            BatchGeneratedTokens::from_batch(
                &ScheduleBatch::prefill(vec![create_request("req-A", 8, 2)]),
                vec![GeneratedToken::unfinished(vec![100])],
            )
            .unwrap(),
        )
        .unwrap();

    // Now request B arrives sharing the first 4 tokens: [10, 11, 12, 13, 99, 100]
    let mut req_b_tokens = (0..4u32).map(|x| x + 10).collect::<Vec<_>>();
    req_b_tokens.extend_from_slice(&[99, 100]);
    let req_b = ScheduledRequest::new(
        RequestId::from("req-B"),
        req_b_tokens,
        SamplingParams::new(1),
    );
    scheduler.enqueue(req_b);

    // Schedule next prefill batch: req-A's next chunk or req-B
    // req-B matches prefix cache from req-A's first chunk!
    let matched_b = scheduler
        .prefix_cache()
        .match_prefix(&[10, 11, 12, 13, 99, 100]);
    assert_eq!(matched_b.matched_token_count, 4);
    assert_eq!(matched_b.remaining_input_ids, vec![99, 100]);
}

#[test]
fn hash_chain_parity_with_locus_router() {
    use sglang_srt::cache::compute_sglang_block_hashes;

    let recorder = Arc::new(RecordingKvStateEventPublisher::new());
    let mut scheduler = Scheduler::with_cache_resources(
        MockModelWorker::default(),
        RadixCache::default(),
        CachePageAllocator::new(1024),
    )
    .with_chunked_prefill_size(Some(4))
    .with_event_publisher(recorder.clone());

    let req = create_request("req-hash-test", 8, 1);
    let prompt_tokens = req.input_ids().to_vec();
    scheduler.enqueue(req);

    // Run both chunks to completion
    let batch1 = scheduler.next_prefill_batch(1).unwrap();
    let _ = scheduler
        .complete_batch(
            batch1,
            BatchGeneratedTokens::from_batch(
                &ScheduleBatch::prefill(vec![create_request("req-hash-test", 8, 1)]),
                vec![GeneratedToken::unfinished(vec![100])],
            )
            .unwrap(),
        )
        .unwrap();

    let batch2 = scheduler.next_prefill_batch(1).unwrap();
    let _ = scheduler
        .complete_batch(
            batch2,
            BatchGeneratedTokens::from_batch(
                &ScheduleBatch::prefill(vec![create_request("req-hash-test", 8, 1)]),
                vec![GeneratedToken::finished(vec![101])],
            )
            .unwrap(),
        )
        .unwrap();

    let events = recorder.stored_events();
    assert_eq!(events.len(), 2);

    // Expected block hashes for the full prompt with block_size = 1
    let page_size = scheduler.available_cache_pages().map(|_| 1).unwrap_or(1);
    let expected_hashes = compute_sglang_block_hashes(&prompt_tokens, page_size);

    // Verify chunk 1 hashes
    assert_eq!(events[0].block_hashes, expected_hashes[0..4]);
    assert_eq!(events[0].parent_block_hash, None);

    // Verify chunk 2 hashes
    assert_eq!(events[1].block_hashes, expected_hashes[4..8]);
    assert_eq!(events[1].parent_block_hash, Some(expected_hashes[3]));
}

#[tokio::test]
async fn real_qwen35_0_8b_chunked_prefill_execution() {
    use sglang_srt::cli::ServerArgs;
    use sglang_srt::proto::sglang::runtime::v1::sglang_service_server::SglangService;
    use sglang_srt::proto::sglang::runtime::v1::{
        RequestOptions, SamplingParams, TextGenerateRequest,
    };
    use sglang_srt::server::try_build_bootstrap_grpc_router_service;
    use std::path::Path;
    use tonic::Request;

    let model_path = Path::new("/Users/reese/.cache/sglang-rs/Qwen3.5-0.8B");
    if !model_path.exists() {
        eprintln!("Qwen3.5-0.8B weights not present, skipping real model run");
        return;
    }

    let chunk_size: usize = 4;
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        model_path.to_str().unwrap(),
        "--device",
        "cpu",
        "--grpc-mode",
        "--max-total-tokens",
        "512",
        "--chunked-prefill-size",
        &chunk_size.to_string(),
    ])
    .expect("server args parse successfully");

    let service = try_build_bootstrap_grpc_router_service(&args)
        .expect("bootstrap grpc router service should build with Qwen3.5 on CPU");

    // Prompt with ~18 tokens (> chunk_size 4)
    let prompt =
        "Artificial intelligence has evolved rapidly over recent years, transforming computing.";
    let mut stream = service
        .text_generate(Request::new(TextGenerateRequest {
            text: prompt.to_string(),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(3),
                top_k: Some(1),
                ..Default::default()
            }),
            options: Some(RequestOptions {
                request_id: Some("real-qwen35-chunked-test".to_string()),
                stream: true,
                data_parallel_rank: 0,
                trace_headers: Default::default(),
            }),
            disaggregated_params: None,
        }))
        .await
        .expect("text generate request should succeed")
        .into_inner();

    let mut generated_tokens = Vec::new();
    let mut generated_text = String::new();
    let mut prompt_tokens_reported = 0;
    while let Some(resp) = tonic::codegen::tokio_stream::StreamExt::next(&mut stream).await {
        let resp = resp.expect("stream response ok");
        if let Some(body) = resp.body {
            match body {
                sglang_srt::proto::sglang::runtime::v1::generate_response::Body::Chunk(chunk) => {
                    generated_tokens.extend(chunk.token_ids);
                    generated_text.push_str(&chunk.text);
                }
                sglang_srt::proto::sglang::runtime::v1::generate_response::Body::Complete(
                    complete,
                ) => {
                    prompt_tokens_reported = complete.prompt_tokens;
                    if generated_tokens.is_empty() {
                        generated_tokens.extend(complete.output_ids);
                    }
                    if generated_text.is_empty() {
                        generated_text = complete.text;
                    }
                    break;
                }
                sglang_srt::proto::sglang::runtime::v1::generate_response::Body::Error(err) => {
                    panic!("Generation stream error: {err:?}");
                }
            }
        }
    }

    assert!(
        prompt_tokens_reported as usize > chunk_size,
        "Prompt tokens ({prompt_tokens_reported}) must exceed chunk size ({chunk_size})"
    );
    assert!(
        !generated_tokens.is_empty(),
        "Should generate at least one token from Qwen3.5"
    );
    println!(
        "Successfully ran real Qwen3.5-0.8B with chunked prefill!\n  Prompt tokens: {}\n  Chunk size: {}\n  Generated tokens: {:?}\n  Generated text: {:?}",
        prompt_tokens_reported, chunk_size, generated_tokens, generated_text
    );
}
