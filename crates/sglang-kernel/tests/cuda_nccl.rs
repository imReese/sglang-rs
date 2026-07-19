use std::sync::{Arc, Barrier};

use sglang_kernel::cuda::CudaDriver;
use sglang_kernel::nccl::{NcclCommunicator, NcclLibrary, NcclRank};

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    bits.wrapping_add(rounding_bias) as u16
}

fn bf16_value(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn test_device_ordinals() -> [usize; 2] {
    if let Ok(value) = std::env::var("SGLANG_CUDA_TEST_DEVICES") {
        let devices = value
            .split(',')
            .map(str::trim)
            .map(|ordinal| {
                ordinal
                    .parse::<usize>()
                    .expect("SGLANG_CUDA_TEST_DEVICES must contain CUDA ordinals")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            devices.len(),
            2,
            "SGLANG_CUDA_TEST_DEVICES must contain exactly two ordinals"
        );
        assert_ne!(devices[0], devices[1], "NCCL requires two distinct devices");
        return [devices[0], devices[1]];
    }

    let second = std::env::var("SGLANG_CUDA_TEST_DEVICE")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("SGLANG_CUDA_TEST_DEVICE must be a CUDA ordinal")
        })
        .unwrap_or(1);
    assert_ne!(
        second, 0,
        "NCCL acceptance requires a nonzero second device"
    );
    [0, second]
}

#[test]
#[ignore = "requires two CUDA devices, NVIDIA drivers, and NCCL with BF16 support"]
fn cuda_nccl_bf16_all_reduce_runs_across_two_device_ordinals() {
    let device_ordinals = test_device_ordinals();
    let driver = CudaDriver::load().expect("CUDA driver should load");
    let device_count = driver
        .device_count()
        .expect("CUDA devices should enumerate");
    for ordinal in device_ordinals {
        assert!(
            ordinal < device_count,
            "CUDA ordinal {ordinal} is unavailable on a {device_count}-device runner"
        );
    }
    let unique_id = NcclLibrary::load()
        .expect("NCCL should load")
        .unique_id()
        .expect("NCCL unique ID should be generated");
    let launch_barrier = Arc::new(Barrier::new(device_ordinals.len()));

    let ranks = device_ordinals
        .into_iter()
        .enumerate()
        .map(|(rank, device_ordinal)| {
            let launch_barrier = Arc::clone(&launch_barrier);
            std::thread::spawn(move || -> Result<Vec<f32>, String> {
                let driver = CudaDriver::load().map_err(|error| error.to_string())?;
                let context = driver
                    .retain_primary_context(device_ordinal)
                    .map_err(|error| error.to_string())?;
                let library = NcclLibrary::load().map_err(|error| error.to_string())?;
                let communicator = NcclCommunicator::initialize(
                    library,
                    &context,
                    unique_id,
                    NcclRank::new(device_ordinals.len(), rank)
                        .map_err(|error| error.to_string())?,
                )
                .map_err(|error| error.to_string())?;
                let input = if rank == 0 { [1.0, 2.0] } else { [3.0, 4.0] };
                let bytes = input
                    .into_iter()
                    .flat_map(|value| bf16_bits(value).to_ne_bytes())
                    .collect::<Vec<_>>();
                let mut allocation = context
                    .allocate(bytes.len())
                    .map_err(|error| error.to_string())?;
                allocation
                    .copy_from_host(0, &bytes)
                    .map_err(|error| error.to_string())?;

                launch_barrier.wait();
                communicator
                    .all_reduce_bf16_sum_in_place(&mut allocation, input.len())
                    .map_err(|error| error.to_string())?;

                let mut output = vec![0_u8; bytes.len()];
                allocation
                    .copy_to_host(0, &mut output)
                    .map_err(|error| error.to_string())?;
                Ok(output
                    .chunks_exact(2)
                    .map(|bytes| bf16_value(u16::from_ne_bytes([bytes[0], bytes[1]])))
                    .collect())
            })
        })
        .collect::<Vec<_>>();

    for rank in ranks {
        let output = rank
            .join()
            .expect("NCCL rank thread should not panic")
            .expect("NCCL rank should complete");
        assert_eq!(output, vec![4.0, 6.0]);
    }
}
