use crate::model_artifacts::{
    LocalModelArtifacts, LocalModelCheckpointCatalog, ModelArtifactError,
    SafetensorsLayerTensorSpan, SafetensorsTensorSpan,
};

pub(super) fn validate_deepseek_checkpoint(
    artifacts: &LocalModelArtifacts,
) -> Result<(), ModelArtifactError> {
    let catalog = artifacts.checkpoint_catalog()?;
    let num_hidden_layers = required_num_hidden_layers(&catalog, "DeepSeek")?;

    required_root_tensor(&catalog, "DeepSeek", "model.embed_tokens.weight")?;
    required_root_tensor(&catalog, "DeepSeek", "model.norm.weight")?;
    required_root_tensor(&catalog, "DeepSeek", "lm_head.weight")?;
    let hc_head_fn = required_root_tensor(&catalog, "DeepSeek", "model.hc_head_fn")?;
    let hc_head_base = required_root_tensor(&catalog, "DeepSeek", "model.hc_head_base")?;
    let hc_head_scale = required_root_tensor(&catalog, "DeepSeek", "model.hc_head_scale")?;
    validate_deepseek_hc_head_shapes(&catalog, &hc_head_fn, &hc_head_base, &hc_head_scale)?;

    for layer_id in 0..num_hidden_layers {
        for suffix in [
            "self_attn.wq_a.weight",
            "self_attn.wq_b.weight",
            "self_attn.wkv.weight",
            "self_attn.q_norm.weight",
            "self_attn.kv_norm.weight",
            "self_attn.wo_a.weight",
            "self_attn.wo_b.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "hc_attn_fn",
            "hc_attn_base",
            "hc_attn_scale",
            "hc_ffn_fn",
            "hc_ffn_base",
            "hc_ffn_scale",
        ] {
            required_layer_tensor(&catalog, "DeepSeek", layer_id, suffix)?;
        }

        if catalog.config().is_moe_layer(layer_id) {
            required_layer_tensor(&catalog, "DeepSeek", layer_id, "mlp.gate.weight")?;
            required_routed_experts(&catalog, "DeepSeek", layer_id)?;
        } else {
            required_layer_tensor(&catalog, "DeepSeek", layer_id, "mlp.gate_up_proj.weight")?;
            required_layer_tensor(&catalog, "DeepSeek", layer_id, "mlp.down_proj.weight")?;
        }
    }

    Ok(())
}

pub(super) fn validate_glm_moe_dsa_checkpoint(
    artifacts: &LocalModelArtifacts,
) -> Result<(), ModelArtifactError> {
    let catalog = artifacts.checkpoint_catalog()?;
    let num_hidden_layers = required_num_hidden_layers(&catalog, "GLM-DSA")?;

    required_root_tensor(&catalog, "GLM-DSA", "model.embed_tokens.weight")?;
    required_root_tensor(&catalog, "GLM-DSA", "model.norm.weight")?;
    required_root_tensor(&catalog, "GLM-DSA", "lm_head.weight")?;

    for layer_id in 0..num_hidden_layers {
        for suffix in [
            "self_attn.q_a_proj.weight",
            "self_attn.q_a_layernorm.weight",
            "self_attn.q_b_proj.weight",
            "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_a_layernorm.weight",
            "self_attn.kv_b_proj.weight",
            "self_attn.o_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ] {
            required_layer_tensor(&catalog, "GLM-DSA", layer_id, suffix)?;
        }

        if catalog.config().is_moe_layer(layer_id) {
            required_layer_tensor(&catalog, "GLM-DSA", layer_id, "mlp.gate.weight")?;
            required_routed_experts(&catalog, "GLM-DSA", layer_id)?;
        } else {
            for suffix in [
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ] {
                required_layer_tensor(&catalog, "GLM-DSA", layer_id, suffix)?;
            }
        }
    }

    Ok(())
}

fn required_num_hidden_layers(
    catalog: &LocalModelCheckpointCatalog,
    family: &str,
) -> Result<usize, ModelArtifactError> {
    catalog.config().num_hidden_layers.ok_or_else(|| {
        invalid_checkpoint(
            catalog,
            format!("missing {family} model num_hidden_layers config"),
        )
    })
}

fn required_root_tensor(
    catalog: &LocalModelCheckpointCatalog,
    family: &str,
    tensor_name: &str,
) -> Result<SafetensorsTensorSpan, ModelArtifactError> {
    catalog
        .safetensors()
        .tensor_span(tensor_name)?
        .ok_or_else(|| {
            invalid_checkpoint(
                catalog,
                format!("missing {family} model tensor {tensor_name}"),
            )
        })
}

fn required_layer_tensor<'a>(
    catalog: &'a LocalModelCheckpointCatalog,
    family: &str,
    layer_id: usize,
    suffix: &str,
) -> Result<&'a SafetensorsLayerTensorSpan, ModelArtifactError> {
    catalog
        .layer_tensors()
        .span(layer_id, suffix)
        .ok_or_else(|| {
            invalid_checkpoint(
                catalog,
                format!("missing {family} layer {layer_id} tensor {suffix}"),
            )
        })
}

fn required_routed_experts(
    catalog: &LocalModelCheckpointCatalog,
    family: &str,
    layer_id: usize,
) -> Result<(), ModelArtifactError> {
    catalog
        .routed_experts()
        .layer(layer_id)
        .map(|_| ())
        .ok_or_else(|| {
            invalid_checkpoint(
                catalog,
                format!("missing {family} layer {layer_id} routed expert weights"),
            )
        })
}

fn validate_deepseek_hc_head_shapes(
    catalog: &LocalModelCheckpointCatalog,
    hc_head_fn: &SafetensorsTensorSpan,
    hc_head_base: &SafetensorsTensorSpan,
    hc_head_scale: &SafetensorsTensorSpan,
) -> Result<(), ModelArtifactError> {
    let hidden_size = catalog.config().hidden_size.ok_or_else(|| {
        invalid_checkpoint(
            catalog,
            "missing DeepSeek model hidden_size config for HC head validation",
        )
    })?;
    let hc_mult = catalog.config().hc_mult.ok_or_else(|| {
        invalid_checkpoint(
            catalog,
            "missing DeepSeek model hc_mult config for HC head validation",
        )
    })?;
    let hc_dim = hc_mult
        .checked_mul(hidden_size)
        .ok_or_else(|| invalid_checkpoint(catalog, "DeepSeek HC head dimension overflowed"))?;

    validate_tensor_shape(catalog, "model.hc_head_fn", hc_head_fn, &[hc_mult, hc_dim])?;
    validate_tensor_shape(catalog, "model.hc_head_base", hc_head_base, &[hc_mult])?;
    validate_tensor_shape(catalog, "model.hc_head_scale", hc_head_scale, &[1])?;
    Ok(())
}

fn validate_tensor_shape(
    catalog: &LocalModelCheckpointCatalog,
    tensor_name: &str,
    tensor: &SafetensorsTensorSpan,
    expected_shape: &[usize],
) -> Result<(), ModelArtifactError> {
    if tensor.metadata.shape == expected_shape {
        return Ok(());
    }

    Err(invalid_checkpoint(
        catalog,
        format!(
            "DeepSeek model tensor {tensor_name} shape {:?} does not match expected {:?}",
            tensor.metadata.shape, expected_shape
        ),
    ))
}

fn invalid_checkpoint(
    catalog: &LocalModelCheckpointCatalog,
    message: impl Into<String>,
) -> ModelArtifactError {
    ModelArtifactError::InvalidSafetensorsData {
        path: catalog.model_path().to_path_buf(),
        message: message.into(),
    }
}
