use sglang_kernel::cublas::CudaBlas;
use sglang_kernel::cuda::{CudaContext, CudaDeviceAllocation};
use sglang_kernel::cuda_mla::{
    CudaBf16MlaKernels, CudaBf16MlaShape, CudaMlaExpandOutput, CudaMlaPrepareCache,
    CudaMlaPrepareQuery,
};

use crate::cache::CachePageId;
use crate::cuda_attention::{
    CudaBf16PagedAttentionExecutor, CudaPagedAttentionForward, CudaPagedAttentionMetadata,
};
use crate::cuda_kv_cache::CudaKvStorage;
use crate::cuda_transformer::{
    CudaBf16Matrix, CudaExecutorError, allocate_bf16, checked_product, linear,
    upload_required_bf16, upload_usize_as_u64,
};
use crate::kv_cache::PagedKvCacheLayout;
use crate::model_artifacts::LocalModelArtifacts;
use crate::models::{HybridFullAttentionConfig, HybridMultiLatentAttentionWeightNames};

pub(crate) struct CudaBf16MultiLatentAttention {
    shape: CudaMultiLatentAttentionShape,
    query: CudaBf16Matrix,
    kv_a: CudaBf16Matrix,
    kv_a_norm: CudaDeviceAllocation,
    kv_b: CudaBf16Matrix,
    output: CudaBf16Matrix,
}

pub(crate) struct CudaMultiLatentAttentionForward<'a> {
    pub(crate) context: &'a CudaContext,
    pub(crate) blas: &'a CudaBlas,
    pub(crate) kernels: &'a CudaBf16MlaKernels,
    pub(crate) attention: &'a mut CudaBf16PagedAttentionExecutor,
    pub(crate) hidden: &'a CudaDeviceAllocation,
    pub(crate) position: usize,
    pub(crate) output_slot: CachePageId,
    pub(crate) sequence_slots: &'a [CachePageId],
    pub(crate) cache_layer_index: usize,
    pub(crate) kv_layout: PagedKvCacheLayout,
    pub(crate) kv_storage: &'a mut CudaKvStorage,
    pub(crate) rms_norm_epsilon: f32,
    pub(crate) rope_theta: f32,
}

impl CudaBf16MultiLatentAttention {
    pub(crate) fn load(
        artifacts: &LocalModelArtifacts,
        context: &CudaContext,
        names: &HybridMultiLatentAttentionWeightNames,
        hidden_size: usize,
        config: HybridFullAttentionConfig,
    ) -> Result<Self, CudaExecutorError> {
        let shape = CudaMultiLatentAttentionShape::new(hidden_size, config)?;
        Ok(Self {
            query: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.query_weight,
                shape.query_size,
                hidden_size,
            )?,
            kv_a: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.kv_a_weight,
                shape.compressed_size,
                hidden_size,
            )?,
            kv_a_norm: upload_required_bf16(
                artifacts,
                context,
                &names.kv_a_norm,
                shape.kv_lora_rank,
            )?,
            kv_b: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.kv_b_weight,
                shape.expanded_size,
                shape.kv_lora_rank,
            )?,
            output: CudaBf16Matrix::load(
                artifacts,
                context,
                &names.output_weight,
                hidden_size,
                shape.output_size,
            )?,
            shape,
        })
    }

    pub(crate) fn forward(
        &self,
        launch: CudaMultiLatentAttentionForward<'_>,
    ) -> Result<CudaDeviceAllocation, CudaExecutorError> {
        let CudaMultiLatentAttentionForward {
            context,
            blas,
            kernels,
            attention,
            hidden,
            position,
            output_slot,
            sequence_slots,
            cache_layer_index,
            kv_layout,
            kv_storage,
            rms_norm_epsilon,
            rope_theta,
        } = launch;
        if sequence_slots.get(position) != Some(&output_slot) {
            return Err(CudaExecutorError::Shape(format!(
                "MLA output KV slot {} does not match sequence slot at position {position}",
                output_slot.as_usize()
            )));
        }
        let active_sequence = &sequence_slots[..=position];
        let query = linear(
            blas,
            context,
            hidden,
            1,
            self.shape.hidden_size,
            &self.query,
        )?;
        let compressed_kv = linear(blas, context, hidden, 1, self.shape.hidden_size, &self.kv_a)?;
        let positions = upload_usize_as_u64(context, "MLA positions", &[position])?;
        let kernel_shape = self.shape.kernel_shape();
        let mut prepared_query = allocate_bf16(
            context,
            checked_product(
                self.shape.head_count,
                self.shape.prepared_head_dim,
                "MLA prepared query",
            )?,
        )?;
        let mut cache_key = allocate_bf16(context, self.shape.compressed_size)?;
        let mut cache_value = allocate_bf16(context, self.shape.kv_lora_rank)?;
        kernels.prepare_query(CudaMlaPrepareQuery {
            query: &query,
            kv_b_weight: self.kv_b.allocation(),
            positions: &positions,
            output: &mut prepared_query,
            shape: kernel_shape,
            rope_theta,
            skip_rope: self.shape.skip_rope,
        })?;
        kernels.prepare_cache(CudaMlaPrepareCache {
            compressed_kv: &compressed_kv,
            kv_norm_weight: &self.kv_a_norm,
            positions: &positions,
            cache_key: &mut cache_key,
            cache_value: &mut cache_value,
            shape: kernel_shape,
            rms_norm_epsilon,
            rope_theta,
            skip_rope: self.shape.skip_rope,
        })?;
        kv_storage.write_tensor_slot_from_device(
            kv_layout,
            cache_layer_index,
            0,
            output_slot.as_usize(),
            &cache_key,
            0,
        )?;
        kv_storage.write_tensor_slot_from_device(
            kv_layout,
            cache_layer_index,
            1,
            output_slot.as_usize(),
            &cache_value,
            0,
        )?;

        let metadata = CudaPagedAttentionMetadata::for_single_query(active_sequence, kv_layout)?;
        let device_metadata = metadata.upload(context)?;
        let mut latent_attention = allocate_bf16(
            context,
            checked_product(
                self.shape.head_count,
                self.shape.kv_lora_rank,
                "MLA latent attention output",
            )?,
        )?;
        attention.forward(CudaPagedAttentionForward {
            kv_layout,
            kv_storage,
            layer_index: cache_layer_index,
            metadata: &device_metadata,
            queries: &prepared_query,
            queries_offset: 0,
            query_head_count: self.shape.head_count,
            scale: (self.shape.query_head_dim as f32).sqrt().recip(),
            output: &mut latent_attention,
            output_offset: 0,
        })?;
        let mut expanded_attention = allocate_bf16(context, self.shape.output_size)?;
        kernels.expand_output(CudaMlaExpandOutput {
            latent_attention: &latent_attention,
            kv_b_weight: self.kv_b.allocation(),
            output: &mut expanded_attention,
            shape: kernel_shape,
        })?;
        linear(
            blas,
            context,
            &expanded_attention,
            1,
            self.shape.output_size,
            &self.output,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CudaMultiLatentAttentionShape {
    hidden_size: usize,
    head_count: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    value_head_dim: usize,
    query_head_dim: usize,
    prepared_head_dim: usize,
    query_size: usize,
    compressed_size: usize,
    expanded_size: usize,
    output_size: usize,
    skip_rope: bool,
}

impl CudaMultiLatentAttentionShape {
    fn new(
        hidden_size: usize,
        config: HybridFullAttentionConfig,
    ) -> Result<Self, CudaExecutorError> {
        let HybridFullAttentionConfig::MultiLatent {
            num_attention_heads,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            value_head_dim,
            skip_rope,
        } = config
        else {
            return Err(CudaExecutorError::Unsupported(
                "CUDA MLA component requires a multi-latent attention execution plan".to_string(),
            ));
        };
        if hidden_size == 0
            || num_attention_heads == 0
            || kv_lora_rank == 0
            || qk_nope_head_dim == 0
            || qk_rope_head_dim == 0
            || value_head_dim == 0
            || !qk_rope_head_dim.is_multiple_of(2)
        {
            return Err(CudaExecutorError::Shape(format!(
                "invalid CUDA MLA geometry: hidden={hidden_size}, heads={num_attention_heads}, rank={kv_lora_rank}, nope={qk_nope_head_dim}, rope={qk_rope_head_dim}, value={value_head_dim}"
            )));
        }
        let query_head_dim = qk_nope_head_dim
            .checked_add(qk_rope_head_dim)
            .ok_or_else(|| CudaExecutorError::Shape("MLA query head overflowed".to_string()))?;
        let prepared_head_dim = kv_lora_rank
            .checked_add(qk_rope_head_dim)
            .ok_or_else(|| CudaExecutorError::Shape("MLA prepared head overflowed".to_string()))?;
        let expanded_head_dim = qk_nope_head_dim
            .checked_add(value_head_dim)
            .ok_or_else(|| CudaExecutorError::Shape("MLA expanded head overflowed".to_string()))?;
        Ok(Self {
            hidden_size,
            head_count: num_attention_heads,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            value_head_dim,
            query_head_dim,
            prepared_head_dim,
            query_size: checked_product(num_attention_heads, query_head_dim, "MLA query size")?,
            compressed_size: prepared_head_dim,
            expanded_size: checked_product(
                num_attention_heads,
                expanded_head_dim,
                "MLA expanded KV size",
            )?,
            output_size: checked_product(num_attention_heads, value_head_dim, "MLA output size")?,
            skip_rope,
        })
    }

    fn kernel_shape(self) -> CudaBf16MlaShape {
        CudaBf16MlaShape {
            row_count: 1,
            head_count: self.head_count,
            kv_lora_rank: self.kv_lora_rank,
            qk_nope_head_dim: self.qk_nope_head_dim,
            qk_rope_head_dim: self.qk_rope_head_dim,
            value_head_dim: self.value_head_dim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_mla_shape_uses_compressed_cache_and_expanded_output() {
        let shape = CudaMultiLatentAttentionShape::new(
            64,
            HybridFullAttentionConfig::MultiLatent {
                num_attention_heads: 4,
                kv_lora_rank: 8,
                qk_nope_head_dim: 6,
                qk_rope_head_dim: 4,
                value_head_dim: 5,
                skip_rope: false,
            },
        )
        .expect("valid MLA shape");
        assert_eq!(shape.query_size, 40);
        assert_eq!(shape.compressed_size, 12);
        assert_eq!(shape.expanded_size, 44);
        assert_eq!(shape.output_size, 20);
    }
}
