use sglang_kernel::cpu::{apply_token_bitmask_inplace, rms_norm, top_k_renorm_probs};
use sglang_kernel::{KernelError, TopK};

#[test]
fn cpu_rms_norm_matches_reference_values() {
    let output =
        rms_norm(&[1.0, 2.0, 3.0, 4.0], &[1.0, 0.5], 2, 2, 1.0e-6).expect("rms norm should run");

    let row0_scale = ((1.0_f32 + 4.0) / 2.0 + 1.0e-6).sqrt();
    let row1_scale = ((9.0_f32 + 16.0) / 2.0 + 1.0e-6).sqrt();
    let expected = [
        1.0 / row0_scale,
        2.0 / row0_scale * 0.5,
        3.0 / row1_scale,
        4.0 / row1_scale * 0.5,
    ];

    assert_close(&output, &expected);
}

#[test]
fn cpu_top_k_renorm_probs_keeps_largest_probs_per_row() {
    let output = top_k_renorm_probs(
        &[0.1, 0.2, 0.7, 0.4, 0.3, 0.2, 0.1, 0.0],
        2,
        4,
        TopK::Fixed(2),
    )
    .expect("top-k renorm should run");

    assert_close(
        &output,
        &[
            0.0,
            0.0,
            7.0 / 11.0,
            4.0 / 11.0,
            3.0 / 5.0,
            2.0 / 5.0,
            0.0,
            0.0,
        ],
    );
}

#[test]
fn cpu_top_k_renorm_probs_supports_per_row_k() {
    let output = top_k_renorm_probs(
        &[0.1, 0.2, 0.7, 0.4, 0.3, 0.2, 0.1, 0.0],
        2,
        4,
        TopK::PerRow(vec![1, 3]),
    )
    .expect("top-k renorm should run");

    assert_close(
        &output,
        &[0.0, 0.0, 1.0, 0.0, 3.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0, 0.0],
    );
}

#[test]
fn cpu_token_bitmask_sets_disallowed_logits_to_negative_infinity() {
    let mut logits = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let bitmask = vec![
        0b0000_0101, // row 0: keep token 0 and 2
        0b0000_1010, // row 1: keep token 1 and 3
    ];

    apply_token_bitmask_inplace(&mut logits, 2, 4, &bitmask, None).expect("bitmask should apply");

    assert_eq!(logits[0], 1.0);
    assert!(logits[1].is_infinite() && logits[1].is_sign_negative());
    assert_eq!(logits[2], 3.0);
    assert!(logits[3].is_infinite() && logits[3].is_sign_negative());
    assert!(logits[4].is_infinite() && logits[4].is_sign_negative());
    assert_eq!(logits[5], 20.0);
    assert!(logits[6].is_infinite() && logits[6].is_sign_negative());
    assert_eq!(logits[7], 40.0);
}

#[test]
fn cpu_token_bitmask_can_target_selected_rows() {
    let mut logits = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let bitmask = vec![
        0b0000_1111, // row 0: keep all
        0b0000_0011, // row 1: keep token 0 and 1
    ];

    apply_token_bitmask_inplace(&mut logits, 2, 4, &bitmask, Some(&[1]))
        .expect("bitmask should apply to selected rows");

    assert_eq!(&logits[..4], &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(logits[4], 10.0);
    assert_eq!(logits[5], 20.0);
    assert!(logits[6].is_infinite() && logits[6].is_sign_negative());
    assert!(logits[7].is_infinite() && logits[7].is_sign_negative());
}

#[test]
fn cpu_kernels_report_shape_errors() {
    let error = rms_norm(&[1.0, 2.0, 3.0], &[1.0, 1.0], 2, 2, 1.0e-6)
        .expect_err("input length should be validated");
    assert_eq!(
        error,
        KernelError::Shape("input length 3 does not match rows * cols 4".to_string())
    );

    let error = top_k_renorm_probs(&[0.5, 0.5], 1, 2, TopK::Fixed(0))
        .expect_err("top_k should be positive");
    assert_eq!(
        error,
        KernelError::InvalidArgument("top_k must be at least 1".to_string())
    );
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1.0e-6,
            "index {index}: expected {expected}, got {actual}"
        );
    }
}
