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
