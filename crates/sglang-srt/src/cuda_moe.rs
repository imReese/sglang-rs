use sglang_kernel::cublas::CudaBlas;
use sglang_kernel::cuda::{CudaContext, CudaDeviceAllocation};
use sglang_kernel::cuda_bf16_kernels::CudaBf16DenseKernels;
use sglang_kernel::cuda_w4a16::{CudaW4A16GemvLaunch, CudaW4A16Kernels};

use crate::compressed_tensors::CompressedTensorsInt4Weight;
use crate::cuda_transformer::{
    CudaBf16DenseFeedForward, CudaBf16Matrix, CudaExecutorError, add, allocate_bf16, download_bf16,
    linear, read_required_f32_values, upload_bytes,
};
use crate::model_artifacts::LocalModelArtifacts;
use crate::models::{
    DenseFeedForwardWeightNames, MoeFeedForwardConfig, MoeFeedForwardWeightNames,
    RoutedExpertWeightFormat,
};
use crate::moe::MoeRouter;

pub(crate) struct CudaBf16MixtureOfExperts {
    hidden_size: usize,
    router: MoeRouter,
    gate: CudaBf16Matrix,
    experts: Vec<CudaRoutedExpert>,
    shared_expert: Option<CudaBf16DenseFeedForward>,
}

impl CudaBf16MixtureOfExperts {
    pub(crate) fn load(
        artifacts: &LocalModelArtifacts,
        context: &CudaContext,
        config: &MoeFeedForwardConfig,
        names: &MoeFeedForwardWeightNames,
        hidden_size: usize,
    ) -> Result<Self, CudaExecutorError> {
        if hidden_size == 0 {
            return Err(CudaExecutorError::Shape(
                "CUDA MoE hidden size must be non-zero".to_string(),
            ));
        }
        if names.experts.len() != config.routed_expert_count {
            return Err(CudaExecutorError::Shape(format!(
                "CUDA MoE weight map has {} routed experts, expected {}",
                names.experts.len(),
                config.routed_expert_count
            )));
        }
        let correction_bias = names
            .correction_bias
            .as_deref()
            .map(|name| read_required_f32_values(artifacts, name, config.routed_expert_count))
            .transpose()?;
        let router = MoeRouter::new(config.clone(), correction_bias)
            .map_err(|error| CudaExecutorError::Unsupported(error.to_string()))?;
        let experts = names
            .experts
            .iter()
            .map(|expert| -> Result<CudaRoutedExpert, CudaExecutorError> {
                match config.routed_expert_weight_format {
                    RoutedExpertWeightFormat::Unquantized => {
                        Ok(CudaRoutedExpert::Bf16(CudaBf16DenseFeedForward::load(
                            artifacts,
                            context,
                            expert,
                            hidden_size,
                            config.expert_intermediate_size,
                        )?))
                    }
                    RoutedExpertWeightFormat::CompressedTensorsInt4 { group_size } => Ok(
                        CudaRoutedExpert::CompressedInt4(CudaW4A16DenseFeedForward::load(
                            artifacts,
                            context,
                            expert,
                            hidden_size,
                            config.expert_intermediate_size,
                            group_size,
                        )?),
                    ),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let shared_intermediate_size = config
            .expert_intermediate_size
            .checked_mul(config.shared_expert_count)
            .ok_or_else(|| {
                CudaExecutorError::Shape("CUDA shared expert size overflowed".to_string())
            })?;
        let shared_expert = match (config.shared_expert_count, &names.shared_expert) {
            (0, None) => None,
            (0, Some(_)) => {
                return Err(CudaExecutorError::Shape(
                    "CUDA MoE weight map provides a shared expert while the config disables it"
                        .to_string(),
                ));
            }
            (_, None) => {
                return Err(CudaExecutorError::Shape(
                    "CUDA MoE config requires shared experts but the weight map has none"
                        .to_string(),
                ));
            }
            (_, Some(shared)) => Some(CudaBf16DenseFeedForward::load(
                artifacts,
                context,
                shared,
                hidden_size,
                shared_intermediate_size,
            )?),
        };
        Ok(Self {
            hidden_size,
            gate: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.gate_weight,
                config.routed_expert_count,
                hidden_size,
            )?,
            router,
            experts,
            shared_expert,
        })
    }

    pub(crate) fn forward_single(
        &self,
        blas: &CudaBlas,
        kernels: &CudaBf16DenseKernels,
        w4a16_kernels: Option<&CudaW4A16Kernels>,
        context: &CudaContext,
        hidden: &CudaDeviceAllocation,
    ) -> Result<CudaDeviceAllocation, CudaExecutorError> {
        let logits = linear(blas, context, hidden, 1, self.hidden_size, &self.gate)?;
        let logits = download_bf16(&logits, self.router.config().routed_expert_count)?;
        let routed = self
            .router
            .route(&logits)
            .map_err(|error| CudaExecutorError::Execution(error.to_string()))?;

        let mut output = allocate_bf16(context, self.hidden_size)?;
        output.fill(0)?;
        for expert in routed {
            let expert_output = self.experts[expert.index].forward(
                blas,
                kernels,
                w4a16_kernels,
                context,
                hidden,
                self.hidden_size,
            )?;
            kernels.weighted_accumulate(
                &mut output,
                &expert_output,
                self.hidden_size,
                expert.weight,
            )?;
        }
        if let Some(shared_expert) = &self.shared_expert {
            let shared =
                shared_expert.forward(blas, kernels, context, hidden, 1, self.hidden_size)?;
            output = add(kernels, context, &output, &shared, self.hidden_size)?;
        }
        Ok(output)
    }
}

enum CudaRoutedExpert {
    Bf16(CudaBf16DenseFeedForward),
    CompressedInt4(CudaW4A16DenseFeedForward),
}

impl CudaRoutedExpert {
    fn forward(
        &self,
        blas: &CudaBlas,
        dense_kernels: &CudaBf16DenseKernels,
        w4a16_kernels: Option<&CudaW4A16Kernels>,
        context: &CudaContext,
        hidden: &CudaDeviceAllocation,
        hidden_size: usize,
    ) -> Result<CudaDeviceAllocation, CudaExecutorError> {
        match self {
            Self::Bf16(expert) => {
                expert.forward(blas, dense_kernels, context, hidden, 1, hidden_size)
            }
            Self::CompressedInt4(expert) => {
                let kernels = w4a16_kernels.ok_or_else(|| {
                    CudaExecutorError::Unsupported(
                        "CUDA W4A16 kernels were not initialized for compressed routed experts"
                            .to_string(),
                    )
                })?;
                expert.forward(dense_kernels, kernels, context, hidden, hidden_size)
            }
        }
    }
}

struct CudaW4A16DenseFeedForward {
    intermediate_size: usize,
    gate: CudaW4A16Matrix,
    up: CudaW4A16Matrix,
    down: CudaW4A16Matrix,
}

impl CudaW4A16DenseFeedForward {
    fn load(
        artifacts: &LocalModelArtifacts,
        context: &CudaContext,
        names: &DenseFeedForwardWeightNames,
        hidden_size: usize,
        intermediate_size: usize,
        group_size: usize,
    ) -> Result<Self, CudaExecutorError> {
        Ok(Self {
            intermediate_size,
            gate: CudaW4A16Matrix::load(
                artifacts,
                context,
                &names.gate_weight,
                intermediate_size,
                hidden_size,
                group_size,
            )?,
            up: CudaW4A16Matrix::load(
                artifacts,
                context,
                &names.up_weight,
                intermediate_size,
                hidden_size,
                group_size,
            )?,
            down: CudaW4A16Matrix::load(
                artifacts,
                context,
                &names.down_weight,
                hidden_size,
                intermediate_size,
                group_size,
            )?,
        })
    }

    fn forward(
        &self,
        dense_kernels: &CudaBf16DenseKernels,
        w4a16_kernels: &CudaW4A16Kernels,
        context: &CudaContext,
        hidden: &CudaDeviceAllocation,
        hidden_size: usize,
    ) -> Result<CudaDeviceAllocation, CudaExecutorError> {
        let gate = self.gate.forward(w4a16_kernels, context, hidden)?;
        let up = self.up.forward(w4a16_kernels, context, hidden)?;
        let mut activated = allocate_bf16(context, self.intermediate_size)?;
        dense_kernels.silu_mul(&gate, &up, &mut activated, self.intermediate_size)?;
        if self.down.columns != self.intermediate_size || self.down.rows != hidden_size {
            return Err(CudaExecutorError::Shape(
                "CUDA compressed expert down projection does not match hidden geometry".to_string(),
            ));
        }
        self.down.forward(w4a16_kernels, context, &activated)
    }
}

struct CudaW4A16Matrix {
    rows: usize,
    columns: usize,
    group_size: usize,
    packed: CudaDeviceAllocation,
    scales: CudaDeviceAllocation,
}

impl CudaW4A16Matrix {
    fn load(
        artifacts: &LocalModelArtifacts,
        context: &CudaContext,
        logical_weight_name: &str,
        rows: usize,
        columns: usize,
        group_size: usize,
    ) -> Result<Self, CudaExecutorError> {
        let weight = CompressedTensorsInt4Weight::load(
            artifacts,
            logical_weight_name,
            rows,
            columns,
            group_size,
        )
        .map_err(|error| CudaExecutorError::Unsupported(error.to_string()))?;
        Ok(Self {
            rows: weight.rows(),
            columns: weight.columns(),
            group_size: weight.group_size(),
            packed: upload_bytes(context, &weight.packed_i32_bytes())?,
            scales: upload_bytes(context, &f32_values_to_bf16_bytes(weight.scales()))?,
        })
    }

    fn forward(
        &self,
        kernels: &CudaW4A16Kernels,
        context: &CudaContext,
        input: &CudaDeviceAllocation,
    ) -> Result<CudaDeviceAllocation, CudaExecutorError> {
        let mut output = allocate_bf16(context, self.rows)?;
        kernels
            .gemv(CudaW4A16GemvLaunch {
                input,
                packed_weight: &self.packed,
                scales: &self.scales,
                output: &mut output,
                rows: self.rows,
                columns: self.columns,
                group_size: self.group_size,
            })
            .map_err(|error| CudaExecutorError::Execution(error.to_string()))?;
        Ok(output)
    }
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
