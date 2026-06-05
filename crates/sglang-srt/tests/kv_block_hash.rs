use sglang_srt::cache::{compute_sglang_block_hashes, sglang_sha256_digest_to_i64};

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
