use std::mem::size_of;

use sglang_kernel::cuda::{CudaContext, CudaDeviceAllocation, CudaDriver};
use sglang_kernel::cuda_hybrid_kernels::{CudaBf16HybridKernels, CudaKdaDecayLaunch};

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

fn upload_bf16(context: &CudaContext, values: &[f32]) -> CudaDeviceAllocation {
    let bytes = values
        .iter()
        .flat_map(|value| bf16_bits(*value).to_ne_bytes())
        .collect::<Vec<_>>();
    upload(context, &bytes)
}

fn upload_f32(context: &CudaContext, values: &[f32]) -> CudaDeviceAllocation {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    upload(context, &bytes)
}

fn upload(context: &CudaContext, bytes: &[u8]) -> CudaDeviceAllocation {
    let mut allocation = context.allocate(bytes.len()).expect("CUDA allocation");
    allocation.copy_from_host(0, bytes).expect("CUDA upload");
    allocation
}

fn zeroed(context: &CudaContext, byte_len: usize) -> CudaDeviceAllocation {
    let mut allocation = context.allocate(byte_len).expect("CUDA allocation");
    allocation.fill(0).expect("CUDA zero fill");
    allocation
}

fn download_bf16(allocation: &CudaDeviceAllocation, count: usize) -> Vec<f32> {
    let mut bytes = vec![0_u8; count * size_of::<u16>()];
    allocation
        .copy_to_host(0, &mut bytes)
        .expect("CUDA download");
    bytes
        .chunks_exact(size_of::<u16>())
        .map(|chunk| {
            let bits = u16::from_ne_bytes([chunk[0], chunk[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

fn download_f32(allocation: &CudaDeviceAllocation, count: usize) -> Vec<f32> {
    let mut bytes = vec![0_u8; count * size_of::<f32>()];
    allocation
        .copy_to_host(0, &mut bytes)
        .expect("CUDA download");
    bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("f32 bytes")))
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }
}

#[test]
#[ignore = "requires a BF16-capable CUDA device, NVIDIA driver, and NVRTC"]
fn cuda_hybrid_primitives_match_reference_formulas() {
    let driver = CudaDriver::load().expect("CUDA driver");
    let ordinal = std::env::var("SGLANG_CUDA_TEST_DEVICE")
        .unwrap_or_else(|_| "0".to_string())
        .parse()
        .expect("CUDA device ordinal");
    let device = driver.device_info(ordinal).expect("CUDA device");
    let context = driver
        .retain_primary_context(ordinal)
        .expect("CUDA context");
    let kernels = CudaBf16HybridKernels::compile(&context, device.compute_capability)
        .expect("hybrid kernels");

    let mut activations = upload_bf16(&context, &[1.0, -2.0, 3.0, 4.0]);
    kernels
        .silu_inplace(&mut activations, 4)
        .expect("SiLU launch");
    assert_close(
        &download_bf16(&activations, 4),
        &[0.731_058_6, -0.238_405_84, 2.857_722_3, 3.928_055],
        0.02,
    );

    let mut heads = upload_bf16(&context, &[3.0, 4.0, 0.0, 5.0]);
    kernels
        .l2_normalize_heads_inplace(&mut heads, 0, 2, 2, 1.0, 1e-6)
        .expect("L2 normalization launch");
    assert_close(&download_bf16(&heads, 4), &[0.6, 0.8, 0.0, 1.0], 0.01);

    let raw_forget = upload_bf16(&context, &[0.0, 1.0, -1.0, 0.5]);
    let dt_bias = upload_f32(&context, &[0.1, -0.2, 0.3, -0.4]);
    let a_log = upload_f32(&context, &[0.0, 0.5]);
    let mut decay = zeroed(&context, 4 * size_of::<f32>());
    kernels
        .kda_decay(CudaKdaDecayLaunch {
            raw_forget: &raw_forget,
            dt_bias: &dt_bias,
            a_log: &a_log,
            decay: &mut decay,
            batch_size: 1,
            head_count: 2,
            key_head_dim: 2,
        })
        .expect("KDA decay launch");
    let expected_decay = [0.1_f32, 0.8, -0.7, 0.1]
        .into_iter()
        .enumerate()
        .map(|(index, raw)| {
            let head_scale = if index < 2 { 1.0 } else { 0.5_f32.exp() };
            (-head_scale * (1.0 + raw.exp()).ln()).exp()
        })
        .collect::<Vec<_>>();
    assert_close(&download_f32(&decay, 4), &expected_decay, 1e-5);

    let sigmoid_input = upload_bf16(&context, &[0.0, 1.0]);
    let mut sigmoid_output = zeroed(&context, 2 * size_of::<f32>());
    kernels
        .sigmoid_to_f32(&sigmoid_input, &mut sigmoid_output, 2)
        .expect("sigmoid launch");
    assert_close(&download_f32(&sigmoid_output, 2), &[0.5, 0.731_058_6], 1e-5);

    let mut values = upload_bf16(&context, &[2.0, -3.0]);
    kernels
        .sigmoid_mul_inplace(&mut values, &sigmoid_input, 2)
        .expect("sigmoid multiply launch");
    assert_close(&download_bf16(&values, 2), &[1.0, -2.193_175_8], 0.02);
}
