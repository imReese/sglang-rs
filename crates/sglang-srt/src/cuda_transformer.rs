use std::fmt;

use sglang_kernel::cublas::{CudaBlas, CudaBlasError};
use sglang_kernel::cuda::{CudaContext, CudaDeviceAllocation, CudaError};
use sglang_kernel::cuda_bf16_kernels::{CudaBf16DenseKernels, CudaBf16KernelError};
use sglang_kernel::cuda_hybrid_kernels::CudaHybridKernelError;
use sglang_kernel::cuda_kv_kernels::CudaKvPairCopyError;
use sglang_kernel::cuda_linear_attention::CudaLinearAttentionError;
use sglang_kernel::cuda_mla::CudaMlaKernelError;

use crate::cuda_attention::CudaPagedAttentionError;
use crate::cuda_kv_cache::CudaKvStorageError;
use crate::cuda_recurrent_state::CudaRecurrentStateError;
use crate::model_artifacts::{
    LocalModelArtifacts, ModelArtifactError, SafetensorsTensorDecodeError,
};
use crate::models::DenseFeedForwardWeightNames;

const BF16_BYTES: usize = 2;

#[derive(Debug)]
pub(crate) enum CudaExecutorError {
    Unsupported(String),
    Execution(String),
    Shape(String),
    MissingTensor(String),
    ModelArtifact(ModelArtifactError),
    TensorDecode(SafetensorsTensorDecodeError),
    Cuda(CudaError),
    CudaBlas(CudaBlasError),
    Kernel(CudaBf16KernelError),
    HybridKernel(CudaHybridKernelError),
    LinearAttention(CudaLinearAttentionError),
    MultiLatentAttention(CudaMlaKernelError),
    Attention(CudaPagedAttentionError),
    KvCache(CudaKvStorageError),
    KvCopy(CudaKvPairCopyError),
    RecurrentState(CudaRecurrentStateError),
}

impl fmt::Display for CudaExecutorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(message) => write!(formatter, "unsupported CUDA executor: {message}"),
            Self::Execution(message) => write!(formatter, "CUDA executor failed: {message}"),
            Self::Shape(message) => write!(formatter, "CUDA executor shape error: {message}"),
            Self::MissingTensor(name) => {
                write!(formatter, "CUDA executor tensor {name} is missing")
            }
            Self::ModelArtifact(error) => {
                write!(formatter, "CUDA executor artifact error: {error}")
            }
            Self::TensorDecode(error) => {
                write!(formatter, "CUDA executor tensor decode failed: {error}")
            }
            Self::Cuda(error) => {
                write!(formatter, "CUDA executor device operation failed: {error}")
            }
            Self::CudaBlas(error) => write!(formatter, "CUDA executor cuBLAS failed: {error}"),
            Self::Kernel(error) => write!(formatter, "CUDA executor kernel failed: {error}"),
            Self::HybridKernel(error) => {
                write!(formatter, "CUDA executor hybrid kernel failed: {error}")
            }
            Self::LinearAttention(error) => {
                write!(formatter, "CUDA executor linear attention failed: {error}")
            }
            Self::MultiLatentAttention(error) => {
                write!(
                    formatter,
                    "CUDA executor multi-latent attention failed: {error}"
                )
            }
            Self::Attention(error) => write!(formatter, "CUDA executor attention failed: {error}"),
            Self::KvCache(error) => write!(formatter, "CUDA executor KV cache failed: {error}"),
            Self::KvCopy(error) => write!(formatter, "CUDA executor KV scatter failed: {error}"),
            Self::RecurrentState(error) => {
                write!(formatter, "CUDA executor recurrent state failed: {error}")
            }
        }
    }
}

impl std::error::Error for CudaExecutorError {}

macro_rules! error_conversion {
    ($source:ty, $variant:ident) => {
        impl From<$source> for CudaExecutorError {
            fn from(value: $source) -> Self {
                Self::$variant(value)
            }
        }
    };
}

error_conversion!(ModelArtifactError, ModelArtifact);
error_conversion!(SafetensorsTensorDecodeError, TensorDecode);
error_conversion!(CudaError, Cuda);
error_conversion!(CudaBlasError, CudaBlas);
error_conversion!(CudaBf16KernelError, Kernel);
error_conversion!(CudaHybridKernelError, HybridKernel);
error_conversion!(CudaLinearAttentionError, LinearAttention);
error_conversion!(CudaMlaKernelError, MultiLatentAttention);
error_conversion!(CudaPagedAttentionError, Attention);
error_conversion!(CudaKvStorageError, KvCache);
error_conversion!(CudaKvPairCopyError, KvCopy);
error_conversion!(CudaRecurrentStateError, RecurrentState);

pub(crate) struct CudaBf16Matrix {
    allocation: CudaDeviceAllocation,
    rows: usize,
    columns: usize,
}

impl CudaBf16Matrix {
    pub(crate) fn load(
        artifacts: &LocalModelArtifacts,
        context: &CudaContext,
        name: &str,
        rows: usize,
        columns: usize,
    ) -> Result<Self, CudaExecutorError> {
        let element_count = checked_product(rows, columns, name)?;
        Ok(Self {
            allocation: upload_required_bf16(artifacts, context, name, element_count)?,
            rows,
            columns,
        })
    }

    pub(crate) fn allocation(&self) -> &CudaDeviceAllocation {
        &self.allocation
    }
}

pub(crate) struct CudaBf16DenseFeedForward {
    intermediate_size: usize,
    gate: CudaBf16Matrix,
    up: CudaBf16Matrix,
    down: CudaBf16Matrix,
}

impl CudaBf16DenseFeedForward {
    pub(crate) fn load(
        artifacts: &LocalModelArtifacts,
        context: &CudaContext,
        names: &DenseFeedForwardWeightNames,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self, CudaExecutorError> {
        if hidden_size == 0 || intermediate_size == 0 {
            return Err(CudaExecutorError::Shape(format!(
                "dense feed-forward sizes must be non-zero, got hidden={hidden_size}, intermediate={intermediate_size}"
            )));
        }
        Ok(Self {
            intermediate_size,
            gate: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.gate_weight,
                intermediate_size,
                hidden_size,
            )?,
            up: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.up_weight,
                intermediate_size,
                hidden_size,
            )?,
            down: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.down_weight,
                hidden_size,
                intermediate_size,
            )?,
        })
    }

    pub(crate) fn forward(
        &self,
        blas: &CudaBlas,
        kernels: &CudaBf16DenseKernels,
        context: &CudaContext,
        hidden: &CudaDeviceAllocation,
        row_count: usize,
        hidden_size: usize,
    ) -> Result<CudaDeviceAllocation, CudaExecutorError> {
        let gate = linear(blas, context, hidden, row_count, hidden_size, &self.gate)?;
        let up = linear(blas, context, hidden, row_count, hidden_size, &self.up)?;
        let element_count = checked_product(
            row_count,
            self.intermediate_size,
            "feed-forward intermediate",
        )?;
        let mut activated = allocate_bf16(context, element_count)?;
        kernels.silu_mul(&gate, &up, &mut activated, element_count)?;
        linear(
            blas,
            context,
            &activated,
            row_count,
            self.intermediate_size,
            &self.down,
        )
    }
}

pub(crate) fn linear(
    blas: &CudaBlas,
    context: &CudaContext,
    input: &CudaDeviceAllocation,
    rows: usize,
    input_columns: usize,
    weight: &CudaBf16Matrix,
) -> Result<CudaDeviceAllocation, CudaExecutorError> {
    if weight.columns != input_columns {
        return Err(CudaExecutorError::Shape(format!(
            "linear input width {input_columns} does not match weight width {}",
            weight.columns
        )));
    }
    let mut output = allocate_bf16(
        context,
        checked_product(rows, weight.rows, "linear output")?,
    )?;
    blas.bf16_gemm_row_major(
        input,
        rows,
        input_columns,
        &weight.allocation,
        weight.rows,
        &mut output,
    )?;
    Ok(output)
}

pub(crate) fn rms_norm(
    kernels: &CudaBf16DenseKernels,
    context: &CudaContext,
    input: &CudaDeviceAllocation,
    weight: &CudaDeviceAllocation,
    rows: usize,
    width: usize,
    epsilon: f32,
) -> Result<CudaDeviceAllocation, CudaExecutorError> {
    let mut output = allocate_bf16(context, checked_product(rows, width, "RMSNorm output")?)?;
    kernels.rms_norm(input, weight, &mut output, rows, width, epsilon)?;
    Ok(output)
}

pub(crate) fn add_optional_bias(
    kernels: &CudaBf16DenseKernels,
    values: &mut CudaDeviceAllocation,
    bias: Option<&CudaDeviceAllocation>,
    rows: usize,
    width: usize,
) -> Result<(), CudaExecutorError> {
    if let Some(bias) = bias {
        kernels.add_bias(values, bias, rows, width)?;
    }
    Ok(())
}

pub(crate) fn add(
    kernels: &CudaBf16DenseKernels,
    context: &CudaContext,
    left: &CudaDeviceAllocation,
    right: &CudaDeviceAllocation,
    element_count: usize,
) -> Result<CudaDeviceAllocation, CudaExecutorError> {
    let mut output = allocate_bf16(context, element_count)?;
    kernels.add(left, right, &mut output, element_count)?;
    Ok(output)
}

pub(crate) fn allocate_bf16(
    context: &CudaContext,
    element_count: usize,
) -> Result<CudaDeviceAllocation, CudaExecutorError> {
    let byte_len = checked_product(element_count, BF16_BYTES, "BF16 allocation")?;
    Ok(context.allocate(byte_len)?)
}

pub(crate) fn allocate_f32(
    context: &CudaContext,
    element_count: usize,
) -> Result<CudaDeviceAllocation, CudaExecutorError> {
    let byte_len = checked_product(element_count, 4, "F32 allocation")?;
    Ok(context.allocate(byte_len)?)
}

pub(crate) fn upload_required_bf16(
    artifacts: &LocalModelArtifacts,
    context: &CudaContext,
    name: &str,
    expected_elements: usize,
) -> Result<CudaDeviceAllocation, CudaExecutorError> {
    let tensor = artifacts
        .safetensors()
        .read_tensor(name)?
        .ok_or_else(|| CudaExecutorError::MissingTensor(name.to_string()))?;
    if !matches!(tensor.metadata.dtype.as_str(), "F32" | "F16" | "BF16") {
        return Err(CudaExecutorError::Unsupported(format!(
            "tensor {name} uses checkpoint dtype {}; the BF16 executor currently accepts unquantized F32, F16, or BF16 weights and does not apply quantized weight scales",
            tensor.metadata.dtype
        )));
    }
    if tensor.element_count() != expected_elements {
        return Err(CudaExecutorError::Shape(format!(
            "tensor {name} has {} elements, expected {expected_elements}",
            tensor.element_count()
        )));
    }
    let values = tensor.decode_f32_values()?;
    let bytes = f32_values_to_bf16_bytes(&values);
    let mut allocation = context.allocate(bytes.len())?;
    allocation.copy_from_host(0, &bytes)?;
    Ok(allocation)
}

pub(crate) fn upload_optional_bf16(
    artifacts: &LocalModelArtifacts,
    context: &CudaContext,
    name: Option<&str>,
    expected_elements: usize,
) -> Result<Option<CudaDeviceAllocation>, CudaExecutorError> {
    name.map(|name| upload_required_bf16(artifacts, context, name, expected_elements))
        .transpose()
}

pub(crate) fn upload_required_f32(
    artifacts: &LocalModelArtifacts,
    context: &CudaContext,
    name: &str,
    expected_elements: usize,
) -> Result<CudaDeviceAllocation, CudaExecutorError> {
    let values = read_required_f32_values(artifacts, name, expected_elements)?;
    upload_f32_values(context, &values)
}

pub(crate) fn upload_u32(
    context: &CudaContext,
    values: &[u32],
) -> Result<CudaDeviceAllocation, CudaExecutorError> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    upload_bytes(context, &bytes)
}

pub(crate) fn upload_usize_as_u64(
    context: &CudaContext,
    field: &'static str,
    values: &[usize],
) -> Result<CudaDeviceAllocation, CudaExecutorError> {
    let bytes = values
        .iter()
        .map(|value| {
            u64::try_from(*value).map_err(|_| {
                CudaExecutorError::Shape(format!(
                    "{field} value {value} cannot be represented as u64"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flat_map(u64::to_ne_bytes)
        .collect::<Vec<_>>();
    upload_bytes(context, &bytes)
}

pub(crate) fn upload_bytes(
    context: &CudaContext,
    bytes: &[u8],
) -> Result<CudaDeviceAllocation, CudaExecutorError> {
    let mut allocation = context.allocate(bytes.len())?;
    allocation.copy_from_host(0, bytes)?;
    Ok(allocation)
}

pub(crate) fn download_bf16(
    allocation: &CudaDeviceAllocation,
    element_count: usize,
) -> Result<Vec<f32>, CudaExecutorError> {
    let mut bytes = vec![0_u8; checked_product(element_count, BF16_BYTES, "BF16 download")?];
    allocation.copy_to_host(0, &mut bytes)?;
    Ok(bytes
        .chunks_exact(BF16_BYTES)
        .map(|chunk| {
            let bits = u16::from_ne_bytes([chunk[0], chunk[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect())
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

pub(crate) fn read_required_f32_values(
    artifacts: &LocalModelArtifacts,
    name: &str,
    expected_elements: usize,
) -> Result<Vec<f32>, CudaExecutorError> {
    let tensor = artifacts
        .safetensors()
        .read_tensor(name)?
        .ok_or_else(|| CudaExecutorError::MissingTensor(name.to_string()))?;
    if !matches!(tensor.metadata.dtype.as_str(), "F32" | "F16" | "BF16") {
        return Err(CudaExecutorError::Unsupported(format!(
            "tensor {name} uses checkpoint dtype {}; the BF16 executor currently accepts unquantized F32, F16, or BF16 weights and does not apply quantized weight scales",
            tensor.metadata.dtype
        )));
    }
    if tensor.element_count() != expected_elements {
        return Err(CudaExecutorError::Shape(format!(
            "tensor {name} has {} elements, expected {expected_elements}",
            tensor.element_count()
        )));
    }
    Ok(tensor.decode_f32_values()?)
}

fn upload_f32_values(
    context: &CudaContext,
    values: &[f32],
) -> Result<CudaDeviceAllocation, CudaExecutorError> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    upload_bytes(context, &bytes)
}

pub(crate) fn checked_product(
    left: usize,
    right: usize,
    name: impl fmt::Display,
) -> Result<usize, CudaExecutorError> {
    left.checked_mul(right)
        .ok_or_else(|| CudaExecutorError::Shape(format!("{name} size overflowed")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_conversion_rounds_to_nearest_even() {
        let values = [0.0_f32, 1.0, -2.5, f32::INFINITY];
        let bytes = f32_values_to_bf16_bytes(&values);
        let decoded = bytes
            .chunks_exact(2)
            .map(|chunk| f32::from_bits((u16::from_ne_bytes([chunk[0], chunk[1]]) as u32) << 16))
            .collect::<Vec<_>>();
        assert_eq!(decoded, values);
    }
}
