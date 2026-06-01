use sglang_srt::scheduler::{
    ForwardMode, RequestStage, ScheduleBatch, ScheduledRequest, Scheduler,
};
use sglang_srt::types::{RequestId, SamplingParams};
use sglang_srt::worker::{BatchGeneratedTokens, GeneratedToken, ModelWorker};

#[derive(Default)]
struct TwoStepWorker {
    seen_modes: Vec<ForwardMode>,
}

impl ModelWorker for TwoStepWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        self.seen_modes.push(batch.forward_mode());

        let tokens = batch
            .requests()
            .iter()
            .map(|request| match batch.forward_mode() {
                ForwardMode::Prefill => {
                    assert_eq!(request.stage(), RequestStage::PrefillForward);
                    GeneratedToken::unfinished(vec![101])
                }
                ForwardMode::Decode => {
                    assert_eq!(request.stage(), RequestStage::DecodeForward);
                    assert_eq!(request.output_ids(), &[101]);
                    GeneratedToken::finished(vec![102])
                }
            })
            .collect();

        BatchGeneratedTokens::from_batch(batch, tokens).expect("output shape should match batch")
    }
}

#[test]
fn unfinished_prefill_output_is_requeued_for_decode_until_finished() {
    let mut scheduler = Scheduler::new(TwoStepWorker::default());
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("req-decode"),
        vec![1, 2, 3],
        SamplingParams { max_new_tokens: 2 },
    ));

    let prefill_outputs = scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should dispatch");

    assert_eq!(prefill_outputs.len(), 1);
    assert_eq!(prefill_outputs[0].request_id, RequestId::from("req-decode"));
    assert_eq!(prefill_outputs[0].token_ids, vec![101]);
    assert!(!prefill_outputs[0].finished);
    assert_eq!(scheduler.decode_queue_depth(), 1);

    let decode_outputs = scheduler
        .dispatch_decode_batch(4)
        .expect("decode should dispatch");

    assert_eq!(
        scheduler.worker().seen_modes,
        vec![ForwardMode::Prefill, ForwardMode::Decode]
    );
    assert_eq!(decode_outputs.len(), 1);
    assert_eq!(decode_outputs[0].request_id, RequestId::from("req-decode"));
    assert_eq!(decode_outputs[0].token_ids, vec![102]);
    assert!(decode_outputs[0].finished);
    assert_eq!(scheduler.decode_queue_depth(), 0);
}

#[derive(Default)]
struct AlwaysUnfinishedWorker;

impl ModelWorker for AlwaysUnfinishedWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        BatchGeneratedTokens::from_batch(
            batch,
            batch
                .requests()
                .iter()
                .map(|_| GeneratedToken::unfinished(vec![201]))
                .collect(),
        )
        .expect("output shape should match batch")
    }
}

#[test]
fn scheduler_finishes_request_when_max_new_tokens_is_reached() {
    let mut scheduler = Scheduler::new(AlwaysUnfinishedWorker);
    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("req-max-new-tokens"),
        vec![1, 2, 3],
        SamplingParams { max_new_tokens: 1 },
    ));

    let outputs = scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill should dispatch");

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].token_ids, vec![201]);
    assert!(outputs[0].finished);
    assert_eq!(scheduler.decode_queue_depth(), 0);
}
