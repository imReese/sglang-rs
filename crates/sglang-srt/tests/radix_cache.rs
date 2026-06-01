use sglang_srt::cache::{CachePageId, RadixCache};
use sglang_srt::scheduler::{ScheduleBatch, ScheduledRequest, Scheduler};
use sglang_srt::types::{RequestId, SamplingParams};
use sglang_srt::worker::{BatchGeneratedTokens, GeneratedToken, ModelWorker};

#[derive(Default)]
struct PrefixAwareWorker {
    seen_cache_pages: Vec<CachePageId>,
    seen_uncached_input_ids: Vec<u32>,
}

impl ModelWorker for PrefixAwareWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        let request = &batch.requests()[0];
        self.seen_cache_pages = request.prefix_cache_pages().to_vec();
        self.seen_uncached_input_ids = request.uncached_input_ids().to_vec();
        BatchGeneratedTokens::from_batch(batch, vec![GeneratedToken::finished(vec![1])])
            .expect("output shape should match batch")
    }
}

#[test]
fn match_prefix_returns_longest_cached_prefix_and_remaining_tokens() {
    let mut cache = RadixCache::default();
    cache
        .insert(
            &[10, 11, 12],
            &[
                CachePageId::from(100),
                CachePageId::from(101),
                CachePageId::from(102),
            ],
        )
        .expect("insert should succeed");
    cache
        .insert(
            &[10, 11, 13],
            &[
                CachePageId::from(100),
                CachePageId::from(101),
                CachePageId::from(103),
            ],
        )
        .expect("insert should succeed");

    let matched = cache.match_prefix(&[10, 11, 12, 14, 15]);

    assert_eq!(matched.matched_token_count, 3);
    assert_eq!(
        matched.cache_pages,
        vec![
            CachePageId::from(100),
            CachePageId::from(101),
            CachePageId::from(102)
        ]
    );
    assert_eq!(matched.remaining_input_ids, vec![14, 15]);
}

#[test]
fn match_prefix_stops_before_uncached_branch() {
    let mut cache = RadixCache::default();
    cache
        .insert(
            &[20, 21, 22],
            &[
                CachePageId::from(200),
                CachePageId::from(201),
                CachePageId::from(202),
            ],
        )
        .expect("insert should succeed");

    let matched = cache.match_prefix(&[20, 21, 99]);

    assert_eq!(matched.matched_token_count, 2);
    assert_eq!(
        matched.cache_pages,
        vec![CachePageId::from(200), CachePageId::from(201)]
    );
    assert_eq!(matched.remaining_input_ids, vec![99]);
}

#[test]
fn insert_rejects_page_sequences_that_do_not_match_token_count() {
    let mut cache = RadixCache::default();

    let result = cache.insert(&[1, 2, 3], &[CachePageId::from(1), CachePageId::from(2)]);

    assert!(result.is_err());
}

#[test]
fn scheduler_applies_radix_cache_match_before_dispatching_to_worker() {
    let mut cache = RadixCache::default();
    cache
        .insert(&[30, 31], &[CachePageId::from(300), CachePageId::from(301)])
        .expect("insert should succeed");
    let mut scheduler = Scheduler::with_prefix_cache(PrefixAwareWorker::default(), cache);

    scheduler.enqueue(ScheduledRequest::new(
        RequestId::from("req-with-prefix"),
        vec![30, 31, 32, 33],
        SamplingParams { max_new_tokens: 1 },
    ));

    scheduler.dispatch_next().expect("dispatch should succeed");

    assert_eq!(
        scheduler.worker().seen_cache_pages,
        vec![CachePageId::from(300), CachePageId::from(301)]
    );
    assert_eq!(scheduler.worker().seen_uncached_input_ids, vec![32, 33]);
}
