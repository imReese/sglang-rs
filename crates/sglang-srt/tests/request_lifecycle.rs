use sglang_srt::engine::Engine;
use sglang_srt::scheduler::{ForwardMode, ScheduleBatch, Scheduler};
use sglang_srt::tokenizer::ByteTokenizer;
use sglang_srt::types::{GenerateRequest, RequestId, SamplingParams, TokenGenerateRequest};
use sglang_srt::worker::{BatchGeneratedTokens, GeneratedToken, ModelWorker};

#[derive(Default)]
struct EchoWorker {
    seen_prompt_tokens: Vec<u32>,
}

impl ModelWorker for EchoWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        let request = &batch.requests()[0];
        self.seen_prompt_tokens = request.input_ids().to_vec();
        BatchGeneratedTokens::from_batch(
            batch,
            vec![GeneratedToken::finished(vec![b'o' as u32, b'k' as u32])],
        )
        .expect("output shape should match batch")
    }
}

#[derive(Default)]
struct TwoStepTextWorker {
    seen_modes: Vec<ForwardMode>,
}

impl ModelWorker for TwoStepTextWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        self.seen_modes.push(batch.forward_mode());
        let token = match batch.forward_mode() {
            ForwardMode::Prefill => GeneratedToken::unfinished(vec![b'o' as u32]),
            ForwardMode::Decode => GeneratedToken::finished(vec![b'k' as u32]),
        };

        BatchGeneratedTokens::from_batch(batch, vec![token])
            .expect("output shape should match batch")
    }
}

#[test]
fn generation_request_runs_through_tokenizer_scheduler_worker_and_detokenizer() {
    let tokenizer = ByteTokenizer::default();
    let scheduler = Scheduler::new(EchoWorker::default());
    let mut engine = Engine::new(tokenizer, scheduler);

    let output = engine
        .generate(GenerateRequest {
            request_id: RequestId::from("req-1"),
            prompt: "hello".to_string(),
            sampling: SamplingParams::new(2),
        })
        .expect("request should run");

    assert_eq!(output.request_id, RequestId::from("req-1"));
    assert_eq!(output.text, "ok");
    assert!(output.finished);
    assert_eq!(
        engine.scheduler().worker().seen_prompt_tokens,
        vec![
            b'h' as u32,
            b'e' as u32,
            b'l' as u32,
            b'l' as u32,
            b'o' as u32
        ]
    );
}

#[test]
fn generation_request_drives_decode_until_the_request_finishes() {
    let tokenizer = ByteTokenizer::default();
    let scheduler = Scheduler::new(TwoStepTextWorker::default());
    let mut engine = Engine::new(tokenizer, scheduler);

    let output = engine
        .generate(GenerateRequest {
            request_id: RequestId::from("req-2"),
            prompt: "hello".to_string(),
            sampling: SamplingParams::new(2),
        })
        .expect("request should run");

    assert_eq!(output.request_id, RequestId::from("req-2"));
    assert_eq!(output.text, "ok");
    assert!(output.finished);
    assert_eq!(
        engine.scheduler().worker().seen_modes,
        vec![ForwardMode::Prefill, ForwardMode::Decode]
    );
}

#[test]
fn tokenized_generation_request_bypasses_tokenizer_for_router_generate_rpc() {
    let tokenizer = ByteTokenizer::default();
    let scheduler = Scheduler::new(EchoWorker::default());
    let mut engine = Engine::new(tokenizer, scheduler);

    let output = engine
        .generate_tokens(TokenGenerateRequest {
            request_id: RequestId::from("router-req"),
            input_ids: vec![11, 22, 33],
            sampling: SamplingParams::new(2),
            disaggregated_params: None,
            data_parallel_rank: 0,
        })
        .expect("token request should run");

    assert_eq!(output.request_id, RequestId::from("router-req"));
    assert_eq!(output.output_ids, vec![b'o' as u32, b'k' as u32]);
    assert!(output.finished);
    assert_eq!(
        engine.scheduler().worker().seen_prompt_tokens,
        vec![11, 22, 33]
    );
}

#[test]
fn scheduler_exposes_waiting_queue_depth_before_dispatch() {
    let mut scheduler = Scheduler::new(EchoWorker::default());

    scheduler.enqueue(sglang_srt::scheduler::ScheduledRequest::new(
        RequestId::from("req-queued"),
        vec![1, 2, 3],
        SamplingParams::new(4),
    ));

    assert_eq!(scheduler.waiting_queue_depth(), 1);
}
