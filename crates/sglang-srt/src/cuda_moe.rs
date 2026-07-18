use sglang_kernel::cublas::CudaBlas;
use sglang_kernel::cuda::{CudaContext, CudaDeviceAllocation};
use sglang_kernel::cuda_bf16_kernels::CudaBf16DenseKernels;

use crate::cuda_transformer::{
    CudaBf16DenseFeedForward, CudaBf16Matrix, CudaExecutorError, add, allocate_bf16, download_bf16,
    linear, read_required_f32_values,
};
use crate::model_artifacts::LocalModelArtifacts;
use crate::models::{MoeFeedForwardConfig, MoeFeedForwardWeightNames};
use crate::moe::MoeRouter;

pub(crate) struct CudaBf16MixtureOfExperts {
    hidden_size: usize,
    router: MoeRouter,
    gate: CudaBf16Matrix,
    experts: Vec<CudaBf16DenseFeedForward>,
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
            .map(|expert| {
                CudaBf16DenseFeedForward::load(
                    artifacts,
                    context,
                    expert,
                    hidden_size,
                    config.expert_intermediate_size,
                )
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
                context,
                hidden,
                1,
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
