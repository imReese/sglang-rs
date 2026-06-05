use sglang_srt::cache::{
    KvBlockPrefixIndex, KvCacheWorkerId, KvCacheWorkerSnapshot, compute_sglang_block_hashes,
    sglang_sha256_digest_to_i64,
};

#[test]
fn sglang_block_hash_matches_upstream_single_block_golden() {
    let hashes = compute_sglang_block_hashes(&[1, 2, 3, 4], 4);

    assert_eq!(hashes, vec![-3488128144981237669_i64]);
}

#[test]
fn sglang_block_hash_chains_partial_last_block_with_full_parent_digest() {
    let hashes = compute_sglang_block_hashes(&[1, 2, 3, 4, 5], 4);

    assert_eq!(
        hashes,
        vec![-3488128144981237669_i64, -3787494577174227566_i64]
    );
}

#[test]
fn sglang_block_hash_matches_upstream_multi_block_golden() {
    let hashes = compute_sglang_block_hashes(&[10, 20, 30, 40, 50, 60, 70, 80], 2);

    assert_eq!(
        hashes,
        vec![
            978178666101069530_i64,
            -895308556211281782_i64,
            -8033692805846017938_i64,
            835415944263129316_i64,
        ]
    );
}

#[test]
fn sglang_sha256_digest_to_i64_reinterprets_top_64_bits_as_signed() {
    let empty_sha256_digest = [
        0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9,
        0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52,
        0xb8, 0x55,
    ];

    assert_eq!(
        sglang_sha256_digest_to_i64(&empty_sha256_digest),
        -2039914840885289964_i64
    );
}

#[test]
#[should_panic(expected = "block_size must be positive")]
fn sglang_block_hash_rejects_zero_block_size() {
    let _ = compute_sglang_block_hashes(&[1, 2, 3], 0);
}

#[test]
fn kv_block_prefix_index_returns_workers_at_longest_matching_prefix() {
    let mut index = KvBlockPrefixIndex::default();
    let worker_a = worker("http://prefill-a:30000", 0);
    let worker_b = worker("http://prefill-b:30000", 0);
    let block_hashes = compute_sglang_block_hashes(&[1, 2, 3, 4, 5, 6], 2);
    index.insert(&worker_a, &block_hashes[..2]);
    index.insert(&worker_b, &block_hashes);

    let full_match = index.match_prefix(&block_hashes);
    assert_eq!(full_match.matched_blocks, 3);
    assert_eq!(
        full_match.workers.into_iter().collect::<Vec<_>>(),
        vec![worker_b.clone()]
    );

    let mut partial_probe = block_hashes[..2].to_vec();
    partial_probe.push(123456789);
    let partial_match = index.match_prefix(&partial_probe);
    assert_eq!(partial_match.matched_blocks, 2);
    assert_eq!(
        partial_match.workers.into_iter().collect::<Vec<_>>(),
        vec![worker_a, worker_b]
    );
}

#[test]
fn kv_block_prefix_index_removes_worker_from_cached_prefixes() {
    let mut index = KvBlockPrefixIndex::default();
    let worker_a = worker("http://prefill-a:30000", 0);
    let worker_b = worker("http://prefill-b:30000", 0);
    let block_hashes = compute_sglang_block_hashes(&[10, 20, 30, 40, 50, 60], 2);
    index.insert(&worker_a, &block_hashes);
    index.insert(&worker_b, &block_hashes);

    index.remove(&worker_b, &block_hashes);

    let matched = index.match_prefix(&block_hashes);
    assert_eq!(matched.matched_blocks, 3);
    assert_eq!(
        matched.workers.into_iter().collect::<Vec<_>>(),
        vec![worker_a]
    );
}

#[test]
fn kv_block_prefix_index_clear_worker_drops_all_worker_prefixes() {
    let mut index = KvBlockPrefixIndex::default();
    let worker_a = worker("http://prefill-a:30000", 0);
    let worker_b = worker("http://prefill-b:30000", 0);
    let first_chain = compute_sglang_block_hashes(&[1, 2, 3, 4], 2);
    let second_chain = compute_sglang_block_hashes(&[7, 8, 9, 10], 2);
    index.insert(&worker_a, &first_chain);
    index.insert(&worker_a, &second_chain);
    index.insert(&worker_b, &second_chain);

    index.clear_worker(&worker_a);

    let first_match = index.match_prefix(&first_chain);
    assert_eq!(first_match.matched_blocks, 0);
    assert!(first_match.workers.is_empty());

    let second_match = index.match_prefix(&second_chain);
    assert_eq!(second_match.matched_blocks, 2);
    assert_eq!(
        second_match.workers.into_iter().collect::<Vec<_>>(),
        vec![worker_b]
    );
}

#[test]
fn kv_block_prefix_index_selects_lowest_load_worker_among_cache_matches() {
    let mut index = KvBlockPrefixIndex::default();
    let worker_a = worker("http://prefill-a:30000", 0);
    let worker_b = worker("http://prefill-b:30000", 0);
    let worker_c = worker("http://prefill-c:30000", 0);
    let block_hashes = compute_sglang_block_hashes(&[1, 2, 3, 4, 5, 6], 2);
    index.insert(&worker_a, &block_hashes);
    index.insert(&worker_b, &block_hashes);

    let selected = index
        .select_cache_aware_worker(
            &workers_with_loads([(&worker_a, 7), (&worker_b, 2), (&worker_c, 0)]),
            &block_hashes,
            0.5,
        )
        .expect("cache-aware selector should choose a candidate");

    assert_eq!(selected, worker_b);
}

#[test]
fn kv_block_prefix_index_selects_lowest_load_worker_when_cache_misses() {
    let index = KvBlockPrefixIndex::default();
    let worker_a = worker("http://prefill-a:30000", 0);
    let worker_b = worker("http://prefill-b:30000", 0);
    let block_hashes = compute_sglang_block_hashes(&[1, 2, 3, 4], 2);

    let selected = index
        .select_cache_aware_worker(
            &workers_with_loads([(&worker_a, 7), (&worker_b, 2)]),
            &block_hashes,
            0.5,
        )
        .expect("cache-aware selector should fall back to load");

    assert_eq!(selected, worker_b);
}

#[test]
fn kv_block_prefix_index_selects_lowest_load_worker_below_match_threshold() {
    let mut index = KvBlockPrefixIndex::default();
    let worker_a = worker("http://prefill-a:30000", 0);
    let worker_b = worker("http://prefill-b:30000", 0);
    let block_hashes = compute_sglang_block_hashes(&[1, 2, 3, 4, 5, 6], 2);
    index.insert(&worker_a, &block_hashes[..1]);

    let selected = index
        .select_cache_aware_worker(
            &workers_with_loads([(&worker_a, 7), (&worker_b, 2)]),
            &block_hashes,
            0.5,
        )
        .expect("cache-aware selector should fall back to load");

    assert_eq!(selected, worker_b);
}

#[test]
fn kv_block_prefix_index_selects_cache_aware_worker_from_token_ids() {
    let mut index = KvBlockPrefixIndex::default();
    let worker_a = worker("http://prefill-a:30000", 0);
    let worker_b = worker("http://prefill-b:30000", 0);
    let worker_c = worker("http://prefill-c:30000", 0);
    let input_ids = [11, 22, 33, 44, 55, 66];
    let block_hashes = compute_sglang_block_hashes(&input_ids, 2);
    index.insert(&worker_b, &block_hashes);

    let selected = index
        .select_cache_aware_worker_for_tokens(
            &workers_with_loads([(&worker_a, 0), (&worker_b, 9), (&worker_c, 2)]),
            &input_ids,
            2,
            0.5,
        )
        .expect("token-id selector should choose a candidate");

    assert_eq!(selected, worker_b);
}

#[test]
fn kv_block_prefix_index_token_selector_falls_back_for_empty_tokens() {
    let index = KvBlockPrefixIndex::default();
    let worker_a = worker("http://prefill-a:30000", 0);
    let worker_b = worker("http://prefill-b:30000", 0);

    let selected = index
        .select_cache_aware_worker_for_tokens(
            &workers_with_loads([(&worker_a, 5), (&worker_b, 1)]),
            &[],
            2,
            0.5,
        )
        .expect("empty request selector should fall back to load");

    assert_eq!(selected, worker_b);
}

fn worker(endpoint: &str, dp_rank: u32) -> KvCacheWorkerId {
    KvCacheWorkerId {
        endpoint: endpoint.to_string(),
        dp_rank,
    }
}

fn workers_with_loads<const N: usize>(
    workers: [(&KvCacheWorkerId, usize); N],
) -> Vec<KvCacheWorkerSnapshot> {
    workers
        .into_iter()
        .map(|(id, active_load)| KvCacheWorkerSnapshot {
            id: id.clone(),
            active_load,
        })
        .collect()
}
