use crate::{KernelError, KernelResult, TopK};

const BITMASK_BITS_PER_WORD: usize = u32::BITS as usize;

pub fn rms_norm(
    input: &[f32],
    weight: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
) -> KernelResult<Vec<f32>> {
    validate_matrix_len("input", input.len(), rows, cols)?;
    if weight.len() != cols {
        return Err(KernelError::Shape(format!(
            "weight length {} does not match cols {cols}",
            weight.len()
        )));
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(KernelError::InvalidArgument(
            "eps must be finite and non-negative".to_string(),
        ));
    }

    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let offset = row * cols;
        let row_values = &input[offset..offset + cols];
        let mean_square = row_values.iter().map(|value| value * value).sum::<f32>() / cols as f32;
        let scale = (mean_square + eps).sqrt();
        for col in 0..cols {
            output[offset + col] = row_values[col] / scale * weight[col];
        }
    }

    Ok(output)
}

pub fn linear(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    rows: usize,
    input_features: usize,
    output_features: usize,
) -> KernelResult<Vec<f32>> {
    validate_matrix_len("input", input.len(), rows, input_features)?;
    validate_matrix_len("weight", weight.len(), output_features, input_features)?;
    if let Some(bias) = bias
        && bias.len() != output_features
    {
        return Err(KernelError::Shape(format!(
            "bias length {} does not match output features {output_features}",
            bias.len()
        )));
    }

    let mut output = vec![0.0; rows * output_features];
    for row in 0..rows {
        let input_row = &input[row * input_features..(row + 1) * input_features];
        for output_feature in 0..output_features {
            let weight_row =
                &weight[output_feature * input_features..(output_feature + 1) * input_features];
            let value = input_row
                .iter()
                .zip(weight_row)
                .map(|(input, weight)| input * weight)
                .sum::<f32>();
            output[row * output_features + output_feature] =
                value + bias.map_or(0.0, |bias| bias[output_feature]);
        }
    }
    Ok(output)
}

pub fn silu_and_mul(gate: &[f32], up: &[f32]) -> KernelResult<Vec<f32>> {
    if gate.len() != up.len() {
        return Err(KernelError::Shape(format!(
            "SiLU gate length {} does not match up length {}",
            gate.len(),
            up.len()
        )));
    }

    Ok(gate
        .iter()
        .zip(up)
        .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
        .collect())
}

pub fn apply_neox_rope_inplace(
    values: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    position: usize,
    theta: f32,
) -> KernelResult<()> {
    validate_matrix_len("RoPE values", values.len(), num_heads, head_dim)?;
    if !head_dim.is_multiple_of(2) {
        return Err(KernelError::InvalidArgument(format!(
            "NeoX RoPE head dimension {head_dim} must be even"
        )));
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(KernelError::InvalidArgument(
            "RoPE theta must be finite and positive".to_string(),
        ));
    }

    let half_dim = head_dim / 2;
    for head in 0..num_heads {
        let offset = head * head_dim;
        for index in 0..half_dim {
            let inverse_frequency = theta.powf(-((2 * index) as f32) / head_dim as f32);
            let angle = position as f32 * inverse_frequency;
            let (cos, sin) = (angle.cos(), angle.sin());
            let first = values[offset + index];
            let second = values[offset + half_dim + index];
            values[offset + index] = first * cos - second * sin;
            values[offset + half_dim + index] = second * cos + first * sin;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupedQueryAttentionShape {
    pub token_count: usize,
    pub query_head_count: usize,
    pub kv_head_count: usize,
    pub head_dim: usize,
    pub scale: f32,
}

pub fn grouped_query_attention(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    shape: GroupedQueryAttentionShape,
) -> KernelResult<Vec<f32>> {
    let GroupedQueryAttentionShape {
        token_count,
        query_head_count,
        kv_head_count,
        head_dim,
        scale,
    } = shape;
    validate_matrix_len("query", query.len(), query_head_count, head_dim)?;
    let kv_token_width = kv_head_count
        .checked_mul(head_dim)
        .ok_or_else(|| KernelError::InvalidArgument("KV token width overflowed".to_string()))?;
    validate_matrix_len("keys", keys.len(), token_count, kv_token_width)?;
    validate_matrix_len("values", values.len(), token_count, kv_token_width)?;
    if !query_head_count.is_multiple_of(kv_head_count) {
        return Err(KernelError::InvalidArgument(format!(
            "query head count {query_head_count} must be divisible by KV head count {kv_head_count}"
        )));
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(KernelError::InvalidArgument(
            "attention scale must be finite and positive".to_string(),
        ));
    }

    let queries_per_kv_head = query_head_count / kv_head_count;
    let mut output = vec![0.0; query.len()];
    let mut scores = vec![0.0; token_count];
    for query_head in 0..query_head_count {
        let kv_head = query_head / queries_per_kv_head;
        let query_offset = query_head * head_dim;
        let query_row = &query[query_offset..query_offset + head_dim];
        for (token, score) in scores.iter_mut().enumerate() {
            let key_offset = token * kv_token_width + kv_head * head_dim;
            let key_row = &keys[key_offset..key_offset + head_dim];
            *score = query_row
                .iter()
                .zip(key_row)
                .map(|(query, key)| query * key)
                .sum::<f32>()
                * scale;
        }

        let max_score = scores
            .iter()
            .copied()
            .max_by(f32::total_cmp)
            .ok_or_else(|| KernelError::InvalidArgument("attention has no tokens".to_string()))?;
        let normalizer = scores
            .iter_mut()
            .map(|score| {
                *score = (*score - max_score).exp();
                *score
            })
            .sum::<f32>();
        if !normalizer.is_finite() || normalizer <= 0.0 {
            return Err(KernelError::InvalidArgument(
                "attention softmax normalization is invalid".to_string(),
            ));
        }

        for (token, probability) in scores.iter().enumerate() {
            let value_offset = token * kv_token_width + kv_head * head_dim;
            for dimension in 0..head_dim {
                output[query_offset + dimension] +=
                    probability / normalizer * values[value_offset + dimension];
            }
        }
    }
    Ok(output)
}

pub fn top_k_renorm_probs(
    probs: &[f32],
    rows: usize,
    cols: usize,
    top_k: TopK,
) -> KernelResult<Vec<f32>> {
    validate_matrix_len("probs", probs.len(), rows, cols)?;
    let per_row_top_k = top_k_values(top_k, rows, cols)?;
    let mut output = vec![0.0; probs.len()];

    for (row, k) in per_row_top_k.into_iter().enumerate() {
        let offset = row * cols;
        let row_probs = &probs[offset..offset + cols];
        validate_probabilities(row_probs)?;

        let mut indexed = row_probs
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<(usize, f32)>>();
        indexed.sort_by(|(left_index, left), (right_index, right)| {
            right
                .total_cmp(left)
                .then_with(|| left_index.cmp(right_index))
        });

        let sum = indexed.iter().take(k).map(|(_, value)| *value).sum::<f32>();
        if sum == 0.0 {
            continue;
        }

        for (col, value) in indexed.into_iter().take(k) {
            output[offset + col] = value / sum;
        }
    }

    Ok(output)
}

pub fn apply_token_bitmask_inplace(
    logits: &mut [f32],
    rows: usize,
    cols: usize,
    bitmask: &[u32],
    indices: Option<&[usize]>,
) -> KernelResult<()> {
    validate_matrix_len("logits", logits.len(), rows, cols)?;
    let bitmask_stride = cols.div_ceil(BITMASK_BITS_PER_WORD);
    if bitmask.len() != rows * bitmask_stride {
        return Err(KernelError::Shape(format!(
            "bitmask length {} does not match rows * ceil(cols / 32) {}",
            bitmask.len(),
            rows * bitmask_stride
        )));
    }

    let row_indices = match indices {
        Some(indices) => indices.to_vec(),
        None => (0..rows).collect::<Vec<_>>(),
    };
    for row in &row_indices {
        if *row >= rows {
            return Err(KernelError::InvalidArgument(format!(
                "row index {row} is out of range for {rows} rows"
            )));
        }
    }

    for row in row_indices {
        let logit_offset = row * cols;
        let bitmask_offset = row * bitmask_stride;
        for col in 0..cols {
            let word = bitmask[bitmask_offset + col / BITMASK_BITS_PER_WORD];
            let allowed = ((word >> (col % BITMASK_BITS_PER_WORD)) & 1) != 0;
            if !allowed {
                logits[logit_offset + col] = f32::NEG_INFINITY;
            }
        }
    }

    Ok(())
}

fn validate_matrix_len(
    name: &'static str,
    len: usize,
    rows: usize,
    cols: usize,
) -> KernelResult<()> {
    if rows == 0 {
        return Err(KernelError::InvalidArgument(
            "rows must be at least 1".to_string(),
        ));
    }
    if cols == 0 {
        return Err(KernelError::InvalidArgument(
            "cols must be at least 1".to_string(),
        ));
    }
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| KernelError::InvalidArgument("rows * cols overflowed".to_string()))?;
    if len != expected {
        return Err(KernelError::Shape(format!(
            "{name} length {len} does not match rows * cols {expected}"
        )));
    }
    Ok(())
}

fn top_k_values(top_k: TopK, rows: usize, cols: usize) -> KernelResult<Vec<usize>> {
    match top_k {
        TopK::Fixed(k) => {
            validate_top_k(k, cols)?;
            Ok(vec![k; rows])
        }
        TopK::PerRow(values) => {
            if values.len() != rows {
                return Err(KernelError::Shape(format!(
                    "top_k length {} does not match rows {rows}",
                    values.len()
                )));
            }
            for value in &values {
                validate_top_k(*value, cols)?;
            }
            Ok(values)
        }
    }
}

fn validate_top_k(top_k: usize, cols: usize) -> KernelResult<()> {
    if top_k == 0 {
        return Err(KernelError::InvalidArgument(
            "top_k must be at least 1".to_string(),
        ));
    }
    if top_k > cols {
        return Err(KernelError::InvalidArgument(format!(
            "top_k {top_k} must not exceed cols {cols}"
        )));
    }
    Ok(())
}

fn validate_probabilities(probs: &[f32]) -> KernelResult<()> {
    for prob in probs {
        if !prob.is_finite() || *prob < 0.0 {
            return Err(KernelError::InvalidArgument(
                "probabilities must be finite and non-negative".to_string(),
            ));
        }
    }
    Ok(())
}
