use std::mem::size_of;

use sglang_kernel::cuda::{CudaContext, CudaDeviceAllocation, CudaDriver};
use sglang_kernel::cuda_mla::{
    CudaBf16MlaKernels, CudaBf16MlaShape, CudaMlaExpandOutput, CudaMlaPrepareCache,
    CudaMlaPrepareQuery,
};

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

fn upload_u64(context: &CudaContext, values: &[u64]) -> CudaDeviceAllocation {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
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

fn zeroed_bf16(context: &CudaContext, element_count: usize) -> CudaDeviceAllocation {
    let mut allocation = context
        .allocate(element_count * size_of::<u16>())
        .expect("CUDA allocation");
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
fn cuda_absorbed_mla_primitives_match_reference_geometry() {
    let driver = CudaDriver::load().expect("CUDA driver");
    let ordinal = std::env::var("SGLANG_CUDA_TEST_DEVICE")
        .unwrap_or_else(|_| "0".to_string())
        .parse()
        .expect("CUDA device ordinal");
    let device = driver.device_info(ordinal).expect("CUDA device");
    let context = driver
        .retain_primary_context(ordinal)
        .expect("CUDA context");
    let kernels =
        CudaBf16MlaKernels::compile(&context, device.compute_capability).expect("MLA kernels");
    let shape = CudaBf16MlaShape {
        row_count: 1,
        head_count: 1,
        kv_lora_rank: 2,
        qk_nope_head_dim: 2,
        qk_rope_head_dim: 2,
        value_head_dim: 2,
    };
    let kv_b = upload_bf16(
        &context,
        &[
            1.0, 0.0, // absorbed key row 0
            0.0, 1.0, // absorbed key row 1
            1.0, 0.0, // expanded value row 0
            0.0, 1.0, // expanded value row 1
        ],
    );
    let positions = upload_u64(&context, &[7]);
    let rope_inverse_frequencies = upload_f32(&context, &[1.0]);

    let query = upload_bf16(&context, &[2.0, 3.0, 4.0, 5.0]);
    let mut prepared_query = zeroed_bf16(&context, 4);
    kernels
        .prepare_query(CudaMlaPrepareQuery {
            query: &query,
            kv_b_weight: &kv_b,
            positions: &positions,
            rope_inverse_frequencies: &rope_inverse_frequencies,
            output: &mut prepared_query,
            shape,
            rope_magnitude_scale: 1.0,
            rope_interleaved: true,
            skip_rope: true,
        })
        .expect("prepare MLA query");
    assert_close(
        &download_bf16(&prepared_query, 4),
        &[2.0, 3.0, 4.0, 5.0],
        0.01,
    );
    kernels
        .prepare_query(CudaMlaPrepareQuery {
            query: &query,
            kv_b_weight: &kv_b,
            positions: &positions,
            rope_inverse_frequencies: &rope_inverse_frequencies,
            output: &mut prepared_query,
            shape,
            rope_magnitude_scale: 0.5,
            rope_interleaved: true,
            skip_rope: false,
        })
        .expect("prepare interleaved MLA query");
    let (sine, cosine) = 7.0_f32.sin_cos();
    assert_close(
        &download_bf16(&prepared_query, 4),
        &[
            2.0,
            3.0,
            (4.0 * cosine - 5.0 * sine) * 0.5,
            (5.0 * cosine + 4.0 * sine) * 0.5,
        ],
        0.03,
    );

    let compressed = upload_bf16(&context, &[1.0, 1.0, 6.0, 7.0]);
    let norm = upload_bf16(&context, &[1.0, 1.0]);
    let mut cache_key = zeroed_bf16(&context, 4);
    let mut cache_value = zeroed_bf16(&context, 2);
    kernels
        .prepare_cache(CudaMlaPrepareCache {
            compressed_kv: &compressed,
            kv_norm_weight: &norm,
            positions: &positions,
            rope_inverse_frequencies: &rope_inverse_frequencies,
            cache_key: &mut cache_key,
            cache_value: &mut cache_value,
            shape,
            rms_norm_epsilon: 1e-6,
            rope_magnitude_scale: 1.0,
            rope_interleaved: true,
            skip_rope: true,
        })
        .expect("prepare compressed MLA cache");
    assert_close(&download_bf16(&cache_key, 4), &[1.0, 1.0, 6.0, 7.0], 0.01);
    assert_close(&download_bf16(&cache_value, 2), &[1.0, 1.0], 0.01);

    let latent_attention = upload_bf16(&context, &[8.0, 9.0]);
    let mut expanded = zeroed_bf16(&context, 2);
    kernels
        .expand_output(CudaMlaExpandOutput {
            latent_attention: &latent_attention,
            kv_b_weight: &kv_b,
            output: &mut expanded,
            shape,
        })
        .expect("expand MLA output");
    assert_close(&download_bf16(&expanded, 2), &[8.0, 9.0], 0.01);
}
