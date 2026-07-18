use std::mem::size_of;

use sglang_kernel::cuda::{CudaContext, CudaDeviceAllocation, CudaDriver};
use sglang_kernel::cuda_bf16_kernels::CudaBf16DenseKernels;

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
    let mut allocation = context.allocate(bytes.len()).expect("CUDA allocation");
    allocation.copy_from_host(0, &bytes).expect("CUDA upload");
    allocation
}

fn download_bf16(allocation: &CudaDeviceAllocation, count: usize) -> Vec<f32> {
    let mut bytes = vec![0_u8; count * size_of::<u16>()];
    allocation
        .copy_to_host(0, &mut bytes)
        .expect("CUDA download");
    bytes
        .chunks_exact(size_of::<u16>())
        .map(|chunk| f32::from_bits((u16::from_ne_bytes([chunk[0], chunk[1]]) as u32) << 16))
        .collect()
}

#[test]
#[ignore = "requires a BF16-capable CUDA device, NVIDIA driver, and NVRTC"]
fn cuda_weighted_expert_accumulation_matches_reference() {
    let driver = CudaDriver::load().expect("CUDA driver");
    let ordinal = std::env::var("SGLANG_CUDA_TEST_DEVICE")
        .unwrap_or_else(|_| "0".to_string())
        .parse()
        .expect("CUDA device ordinal");
    let device = driver.device_info(ordinal).expect("CUDA device");
    let context = driver
        .retain_primary_context(ordinal)
        .expect("CUDA context");
    let kernels = CudaBf16DenseKernels::compile(&context, device.compute_capability)
        .expect("BF16 dense kernels");
    let mut accumulator = upload_bf16(&context, &[0.0, 0.0, 0.0]);
    let first = upload_bf16(&context, &[2.0, 4.0, 6.0]);
    let second = upload_bf16(&context, &[1.0, 2.0, 3.0]);

    kernels
        .weighted_accumulate(&mut accumulator, &first, 3, 0.25)
        .expect("first expert accumulation");
    kernels
        .weighted_accumulate(&mut accumulator, &second, 3, 0.5)
        .expect("second expert accumulation");

    assert_eq!(download_bf16(&accumulator, 3), vec![1.0, 2.0, 3.0]);
}
