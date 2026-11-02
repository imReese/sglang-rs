use std::sync::{Arc, Condvar, Mutex};

use super::*;
use crate::backend_model::paged_kv_cache_layout_for_model;
use crate::cache::{CachePageAllocator, RadixCache};
use crate::model_executor::{ForwardModel, KvCacheAllocationConfig, ModelRunner};
use crate::model_registry::ModelRegistry;
use crate::mooncake_control::{MooncakeDecodeReceiver, completion_frames, send_frames};
use crate::scheduler::{ForwardMode, PrefixCachePolicy, ScheduledRequest, Scheduler};
use crate::transfer::{
    DecodeBootstrapMetadataPublishSummary, DecodeBootstrapPublisher, DecodeBootstrapRegistry,
    KvCacheTransferPlan, KvCacheTransferSpan, KvTransferBackend, KvTransferCompletionEvent,
    KvTransferModelWorker, KvTransferPoll,
};
use crate::types::{DisaggregatedParams, RequestId, SamplingParams};

#[path = "../tests/support/qwen3_nonzero.rs"]
mod fixture;

type Resources = Arc<Mutex<RuntimeKvCache<CpuReferenceKvCache>>>;
type Logits = Arc<Mutex<Vec<(ForwardMode, Vec<Vec<f32>>)>>>;

struct ReferenceModel {
    decoder: CpuReferenceDenseDecoder,
    cache: Resources,
    logits: Logits,
}

impl ForwardModel for ReferenceModel {
    fn forward(
        &mut self,
        batch: &ModelWorkerBatch,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        let output = self
            .decoder
            .forward(batch, &mut *self.cache.lock().expect("cache"))?;
        self.logits
            .lock()
            .expect("logits")
            .push((batch.forward_mode(), output.logits().to_vec()));
        Ok(output)
    }
}

fn model(path: &std::path::Path) -> (ModelRunner<ReferenceModel>, Resources, Logits) {
    let artifacts = LocalModelArtifacts::from_model_path(path).expect("load fixture");
    let definition = ModelRegistry
        .definition(path, artifacts.config())
        .expect("Qwen adapter definition");
    let layout = paged_kv_cache_layout_for_model(
        definition.kv_cache_layout().expect("KV geometry"),
        KvCacheDtype::Float32,
        KvCacheAllocationConfig {
            slot_capacity: 64,
            page_size: 1,
        },
    )
    .expect("layout");
    let cache = Arc::new(Mutex::new(RuntimeKvCache::new(
        CpuReferenceKvCache::new(layout).expect("KV storage"),
    )));
    let logits = Arc::new(Mutex::new(Vec::new()));
    let model = ReferenceModel {
        decoder: CpuReferenceDenseDecoder::load(&definition, &artifacts).expect("full Transformer"),
        cache: Arc::clone(&cache),
        logits: Arc::clone(&logits),
    };
    (ModelRunner::new(model), cache, logits)
}

struct Destination {
    slots: Vec<CachePageId>,
    endpoint: String,
}
type Rendezvous = Arc<(Mutex<Option<Destination>>, Condvar)>;

// A test-only host payload copy exercises the real reference decoder. It does
// not implement or claim Mooncake's CUDA/RDMA data plane.
struct ReferencePageCopy {
    source: Resources,
    destination: Resources,
    rendezvous: Rendezvous,
    corrupt: bool,
    pending: Vec<KvTransferCompletionEvent>,
}

impl KvTransferBackend for ReferencePageCopy {
    fn register(&mut self, _: TransferableKvCacheMemory) -> Result<(), KvCacheTransferError> {
        Err(KvCacheTransferError::Runtime(
            "reference HashMap storage has no production memory descriptor".into(),
        ))
    }
    fn submit(&mut self, span: &KvCacheTransferSpan) -> Result<(), KvCacheTransferError> {
        let (lock, ready) = &*self.rendezvous;
        let destination = ready
            .wait_while(lock.lock().expect("rendezvous"), |value| value.is_none())
            .expect("destination ready");
        let destination = destination.as_ref().expect("destination");
        assert_ne!(span.cache_pages(), destination.slots);
        let mut source = self.source.lock().expect("source KV");
        let source = source.allocation_mut();
        let mut target_resources = self.destination.lock().expect("destination KV");
        let target = target_resources.allocation_mut();
        for layer in 0..source.layout.runtime().num_layers {
            for (src, dst) in span.cache_pages().iter().zip(&destination.slots) {
                let (mut key, mut value) = source.gather(layer, &[*src]).expect("real prefill KV");
                assert!(key.iter().any(|v| *v != 0.0));
                assert!(value.iter().any(|v| *v != 0.0));
                if self.corrupt {
                    key.fill(0.0);
                    value.fill(0.0);
                }
                target
                    .write(layer, *dst, key, value)
                    .expect("copy reference KV");
            }
        }
        drop(target_resources);
        let frames = completion_frames(span.bootstrap_room(), span.first_token(), None, 0)
            .expect("community control frames");
        send_frames(destination.endpoint.clone(), frames).map_err(KvCacheTransferError::Runtime)?;
        self.pending.push(KvTransferCompletionEvent::completed(
            span.bootstrap_room(),
            span.descriptor_checksum(),
        ));
        Ok(())
    }
    fn poll(&mut self) -> Result<KvTransferPoll, KvCacheTransferError> {
        Ok(KvTransferPoll::new(std::mem::take(&mut self.pending), 0))
    }
    fn cancel(&mut self, room: u64) -> Result<(), KvCacheTransferError> {
        self.pending.retain(|event| event.bootstrap_room() != room);
        Ok(())
    }
    fn shutdown(&mut self) -> Result<(), KvCacheTransferError> {
        self.pending.clear();
        Ok(())
    }
}

struct ReferenceReceive {
    receiver: MooncakeDecodeReceiver,
    rendezvous: Rendezvous,
}

impl DecodeBootstrapPublisher for ReferenceReceive {
    fn publish_decode_bootstrap_metadata(
        &mut self,
        plan: &KvCacheTransferPlan,
    ) -> Result<DecodeBootstrapMetadataPublishSummary, String> {
        assert_eq!(plan.len(), 1);
        let span = &plan.spans()[0];
        self.receiver.register(span.bootstrap_room())?;
        *self.rendezvous.0.lock().expect("rendezvous") = Some(Destination {
            slots: span.cache_pages().to_vec(),
            endpoint: format!("tcp://127.0.0.1:{}", self.receiver.port()),
        });
        self.rendezvous.1.notify_all();
        Ok(DecodeBootstrapMetadataPublishSummary { published_spans: 1 })
    }
    fn receive_prefill_token(&mut self, room: u64) -> Result<u32, String> {
        self.receiver.receive(room)
    }
    fn cancel_receive(&mut self, room: u64) -> Result<(), String> {
        self.receiver.wait_drained(room)
    }
}

struct ReceiveOnly;
impl KvTransferBackend for ReceiveOnly {
    fn register(&mut self, _: TransferableKvCacheMemory) -> Result<(), KvCacheTransferError> {
        unreachable!("reference receive has no native registration")
    }
    fn submit(&mut self, _: &KvCacheTransferSpan) -> Result<(), KvCacheTransferError> {
        panic!("decode must not submit a local prefill transfer")
    }
    fn poll(&mut self) -> Result<KvTransferPoll, KvCacheTransferError> {
        panic!("decode receives sender completion")
    }
    fn cancel(&mut self, _: u64) -> Result<(), KvCacheTransferError> {
        Ok(())
    }
    fn shutdown(&mut self) -> Result<(), KvCacheTransferError> {
        Ok(())
    }
}

fn request(pd: bool, id: &str) -> ScheduledRequest {
    let mut sampling = SamplingParams::new(4);
    sampling.ignore_eos = true;
    ScheduledRequest::new(RequestId::from(id), vec![1, 2, 3], sampling).with_disaggregated_params(
        pd.then(|| DisaggregatedParams {
            bootstrap_host: "127.0.0.1".into(),
            bootstrap_port: 8998,
            bootstrap_room: 51,
        }),
    )
}

fn run_split(path: &std::path::Path, corrupt: bool) -> (Vec<u32>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let (prefill, source, prefill_logits) = model(path);
    let (decode, destination, decode_logits) = model(path);
    let rendezvous = Arc::new((Mutex::new(None), Condvar::new()));
    let copy = ReferencePageCopy {
        source,
        destination,
        rendezvous: Arc::clone(&rendezvous),
        corrupt,
        pending: Vec::new(),
    };
    let prefill = KvTransferModelWorker::new(prefill, DecodeBootstrapRegistry::default(), copy)
        .with_prefill_only();
    let mut sender = Scheduler::with_cache_resources_and_policy(
        prefill,
        RadixCache::default(),
        CachePageAllocator::new(64),
        PrefixCachePolicy::Disabled,
    );
    sender.enqueue(request(true, "P"));
    let sender = std::thread::spawn(move || {
        let output = sender
            .dispatch_prefill_batch(1)
            .expect("prefill completes after copy");
        assert!(output[0].finished);
        assert_eq!(sender.available_cache_pages(), Some(64));
    });
    let receive = ReferenceReceive {
        receiver: MooncakeDecodeReceiver::start("127.0.0.1").expect("receiver"),
        rendezvous,
    };
    let decode =
        KvTransferModelWorker::new(decode, DecodeBootstrapRegistry::default(), ReceiveOnly)
            .with_decode_bootstrap_publisher(receive)
            .with_decode_side_bootstrap_only();
    let mut allocator = CachePageAllocator::new(64);
    let _held = allocator
        .allocate(7)
        .expect("offset destination physical pages");
    let mut receiver = Scheduler::with_cache_resources_and_policy(
        decode,
        RadixCache::default(),
        allocator,
        PrefixCachePolicy::Disabled,
    );
    receiver.enqueue(request(true, "D"));
    let first = receiver
        .dispatch_prefill_batch(1)
        .expect("receive first token");
    assert!(
        decode_logits.lock().expect("logits").is_empty(),
        "decode did not execute local prefill"
    );
    let mut tokens = first[0].token_ids.clone();
    for _ in 0..3 {
        tokens.extend(
            receiver
                .dispatch_decode_batch(1)
                .expect("real cached decode")[0]
                .token_ids
                .clone(),
        );
    }
    sender.join().expect("sender joins");
    assert_eq!(receiver.available_cache_pages(), Some(57));
    let p = prefill_logits
        .lock()
        .expect("prefill logits")
        .iter()
        .map(|(_, rows)| rows[0].clone())
        .collect();
    let d = decode_logits
        .lock()
        .expect("decode logits")
        .iter()
        .map(|(mode, rows)| {
            assert_eq!(*mode, ForwardMode::Decode);
            rows[0].clone()
        })
        .collect();
    (tokens, p, d)
}

#[test]
fn nonzero_qwen_pd_matches_centralized_logits_and_detects_corrupt_kv() {
    let path = std::env::temp_dir().join(format!("sglang-pd-logits-{}", std::process::id()));
    fixture::write_model(&path);
    let (central, _, central_logits) = model(&path);
    let mut scheduler = Scheduler::with_cache_resources_and_policy(
        central,
        RadixCache::default(),
        CachePageAllocator::new(64),
        PrefixCachePolicy::Disabled,
    );
    scheduler.enqueue(request(false, "central"));
    let mut expected = scheduler
        .dispatch_prefill_batch(1)
        .expect("central prefill")[0]
        .token_ids
        .clone();
    for _ in 0..3 {
        expected.extend(
            scheduler.dispatch_decode_batch(1).expect("central decode")[0]
                .token_ids
                .clone(),
        );
    }
    let expected_logits = central_logits
        .lock()
        .expect("logits")
        .iter()
        .map(|(_, rows)| rows[0].clone())
        .collect::<Vec<_>>();
    let (actual, p, d) = run_split(&path, false);
    assert_eq!(actual, expected);
    for (actual, expected) in p.iter().chain(&d).zip(&expected_logits) {
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-6, "PD logit {a} != central {b}");
        }
    }
    let (_, _, corrupt) = run_split(&path, true);
    assert!(
        corrupt[0]
            .iter()
            .zip(&expected_logits[1])
            .any(|(a, b)| (a - b).abs() > 1e-4),
        "the oracle must detect incorrect transferred KV"
    );
    std::fs::remove_dir_all(path).expect("remove fixture");
}
