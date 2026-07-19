use std::fmt;

use sglang_kernel::cuda::{CudaContext, CudaDeviceAllocation, CudaDriver};
use sglang_kernel::nccl::{NcclCommunicator, NcclLibrary, NcclRank, NcclUniqueId};
use sglang_srt::parallel::{RankWorker, TensorParallelRank, TensorParallelTopology, WorkerGroup};

const BF16_BYTES: usize = std::mem::size_of::<u16>();

struct NcclAllReduceWorker {
    rank: TensorParallelRank,
    context: CudaContext,
    communicator: Option<NcclCommunicator>,
}

impl RankWorker for NcclAllReduceWorker {
    type Command = Vec<f32>;
    type Output = Vec<f32>;
    type Error = CudaWorkerError;

    fn execute(&mut self, values: Self::Command) -> Result<Self::Output, Self::Error> {
        let rank_scale = (self.rank.global_rank() + 1) as f32;
        let values = values
            .into_iter()
            .map(|value| value * rank_scale)
            .collect::<Vec<_>>();
        let bytes = f32_values_to_bf16_bytes(&values);
        let mut allocation = self.context.allocate(bytes.len())?;
        allocation.copy_from_host(0, &bytes)?;
        self.communicator
            .as_ref()
            .ok_or(CudaWorkerError::CommunicatorAlreadyShutdown)?
            .all_reduce_bf16_sum_in_place(&mut allocation, values.len())?;
        download_bf16(&allocation, values.len()).map_err(CudaWorkerError::from)
    }

    fn shutdown(&mut self) -> Result<(), Self::Error> {
        self.communicator.take();
        Ok(())
    }
}

#[derive(Debug)]
enum CudaWorkerError {
    Cuda(sglang_kernel::cuda::CudaError),
    Nccl(sglang_kernel::nccl::NcclError),
    CommunicatorAlreadyShutdown,
}

impl fmt::Display for CudaWorkerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(error) => write!(formatter, "CUDA error: {error}"),
            Self::Nccl(error) => write!(formatter, "NCCL error: {error}"),
            Self::CommunicatorAlreadyShutdown => {
                formatter.write_str("NCCL communicator is already shut down")
            }
        }
    }
}

impl From<sglang_kernel::cuda::CudaError> for CudaWorkerError {
    fn from(value: sglang_kernel::cuda::CudaError) -> Self {
        Self::Cuda(value)
    }
}

impl From<sglang_kernel::nccl::NcclError> for CudaWorkerError {
    fn from(value: sglang_kernel::nccl::NcclError) -> Self {
        Self::Nccl(value)
    }
}

#[test]
#[ignore = "requires two CUDA devices, the NVIDIA driver, and NCCL with BF16 support"]
fn cuda_worker_group_owns_two_rank_nccl_lifecycle() {
    let [first_device, second_device] = cuda_test_devices();
    assert!(
        second_device > first_device,
        "SGLANG_CUDA_TEST_DEVICES must contain two ascending ordinals"
    );
    let topology = TensorParallelTopology::new(2, 1, 0, first_device, second_device - first_device)
        .expect("two-rank topology");
    let unique_id = NcclLibrary::load()
        .and_then(|library| library.unique_id())
        .expect("load NCCL and create a unique communicator ID");
    let mut group = launch_nccl_group(&topology, unique_id);

    let outputs = group
        .execute_all(vec![1.0, 2.0])
        .expect("two-rank all-reduce");
    assert_eq!(outputs.len(), 2);
    for output in outputs {
        assert_eq!(output.output(), &[3.0, 6.0]);
    }
    group.shutdown().expect("deterministic NCCL shutdown");
}

fn launch_nccl_group(
    topology: &TensorParallelTopology,
    unique_id: NcclUniqueId,
) -> WorkerGroup<NcclAllReduceWorker> {
    WorkerGroup::launch(topology, move |rank| {
        let driver = CudaDriver::load()?;
        let context = driver.retain_primary_context(rank.device_ordinal())?;
        let communicator = NcclCommunicator::initialize(
            NcclLibrary::load()?,
            &context,
            unique_id,
            NcclRank::new(rank.world_size(), rank.global_rank())?,
        )?;
        Ok::<_, CudaWorkerError>(NcclAllReduceWorker {
            rank,
            context,
            communicator: Some(communicator),
        })
    })
    .expect("launch NCCL worker group")
}

fn cuda_test_devices() -> [usize; 2] {
    if let Ok(devices) = std::env::var("SGLANG_CUDA_TEST_DEVICES") {
        let parsed = devices
            .split(',')
            .map(str::trim)
            .map(|value| value.parse::<usize>().expect("CUDA device ordinal"))
            .collect::<Vec<_>>();
        return parsed
            .try_into()
            .expect("SGLANG_CUDA_TEST_DEVICES must contain exactly two ordinals");
    }
    [
        0,
        std::env::var("SGLANG_CUDA_TEST_DEVICE")
            .ok()
            .map(|value| value.parse().expect("SGLANG_CUDA_TEST_DEVICE ordinal"))
            .unwrap_or(1),
    ]
}

fn f32_values_to_bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| {
            let bits = value.to_bits();
            let rounding_bias = 0x7fff + ((bits >> 16) & 1);
            ((bits.wrapping_add(rounding_bias) >> 16) as u16).to_ne_bytes()
        })
        .collect()
}

fn download_bf16(
    allocation: &CudaDeviceAllocation,
    element_count: usize,
) -> Result<Vec<f32>, sglang_kernel::cuda::CudaError> {
    let mut bytes = vec![0_u8; element_count * BF16_BYTES];
    allocation.copy_to_host(0, &mut bytes)?;
    Ok(bytes
        .chunks_exact(BF16_BYTES)
        .map(|chunk| {
            let bits = u16::from_ne_bytes([chunk[0], chunk[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect())
}
