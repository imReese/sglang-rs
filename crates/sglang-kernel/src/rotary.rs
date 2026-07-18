use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RotaryEmbeddingStyle {
    Neox,
    Interleaved,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YarnScalingConfig {
    pub factor: f32,
    pub original_max_position_embeddings: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub extrapolation_factor: f32,
    pub attention_factor: f32,
    pub mscale: f32,
    pub mscale_all_dim: f32,
    pub apply_attention_scale: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RotaryEmbeddingScaling {
    Default,
    Yarn(YarnScalingConfig),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotaryEmbeddingConfig {
    theta: f32,
    style: RotaryEmbeddingStyle,
    scaling: RotaryEmbeddingScaling,
}

impl RotaryEmbeddingConfig {
    pub fn standard(theta: f32, style: RotaryEmbeddingStyle) -> Result<Self, RotaryEmbeddingError> {
        let config = Self {
            theta,
            style,
            scaling: RotaryEmbeddingScaling::Default,
        };
        config.validate_scalars()?;
        Ok(config)
    }

    pub fn yarn(
        theta: f32,
        style: RotaryEmbeddingStyle,
        scaling: YarnScalingConfig,
    ) -> Result<Self, RotaryEmbeddingError> {
        let config = Self {
            theta,
            style,
            scaling: RotaryEmbeddingScaling::Yarn(scaling),
        };
        config.validate_scalars()?;
        Ok(config)
    }

    pub fn theta(self) -> f32 {
        self.theta
    }

    pub fn style(self) -> RotaryEmbeddingStyle {
        self.style
    }

    pub fn scaling(self) -> RotaryEmbeddingScaling {
        self.scaling
    }

    pub fn inverse_frequencies(self, rotary_dim: usize) -> Result<Vec<f32>, RotaryEmbeddingError> {
        self.validate(rotary_dim)?;
        let half_dim = rotary_dim / 2;
        let mut frequencies = Vec::with_capacity(half_dim);
        for pair_index in 0..half_dim {
            let exponent = -2.0 * pair_index as f64 / rotary_dim as f64;
            let extrapolated = (self.theta as f64).powf(exponent);
            let frequency = match self.scaling {
                RotaryEmbeddingScaling::Default => extrapolated,
                RotaryEmbeddingScaling::Yarn(scaling) => {
                    let interpolated = extrapolated / scaling.factor as f64;
                    let (low, high) = yarn_correction_range(
                        scaling.beta_fast as f64,
                        scaling.beta_slow as f64,
                        rotary_dim,
                        self.theta as f64,
                        scaling.original_max_position_embeddings,
                    );
                    let ramp = yarn_linear_ramp(pair_index as f64, low, high);
                    let mask = (1.0 - ramp) * scaling.extrapolation_factor as f64;
                    interpolated * (1.0 - mask) + extrapolated * mask
                }
            };
            frequencies.push(frequency as f32);
        }
        Ok(frequencies)
    }

    pub fn magnitude_scale(self) -> f32 {
        match self.scaling {
            RotaryEmbeddingScaling::Default => 1.0,
            RotaryEmbeddingScaling::Yarn(scaling) => {
                yarn_mscale(scaling.factor, scaling.mscale)
                    / yarn_mscale(scaling.factor, scaling.mscale_all_dim)
                    * scaling.attention_factor
            }
        }
    }

    pub fn mla_attention_scale(self, base_scale: f32) -> f32 {
        match self.scaling {
            RotaryEmbeddingScaling::Yarn(scaling) if scaling.apply_attention_scale => {
                let scale = yarn_mscale(scaling.factor, scaling.mscale_all_dim);
                base_scale * scale * scale
            }
            RotaryEmbeddingScaling::Default | RotaryEmbeddingScaling::Yarn(_) => base_scale,
        }
    }

    pub fn validate(self, rotary_dim: usize) -> Result<(), RotaryEmbeddingError> {
        self.validate_scalars()?;
        if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
            return Err(RotaryEmbeddingError::Invalid(format!(
                "rotary dimension {rotary_dim} must be non-zero and even"
            )));
        }
        Ok(())
    }

    fn validate_scalars(self) -> Result<(), RotaryEmbeddingError> {
        if !self.theta.is_finite() || self.theta <= 0.0 {
            return Err(RotaryEmbeddingError::Invalid(
                "RoPE theta must be finite and positive".to_string(),
            ));
        }
        let RotaryEmbeddingScaling::Yarn(scaling) = self.scaling else {
            return Ok(());
        };
        if self.theta <= 1.0 {
            return Err(RotaryEmbeddingError::Invalid(
                "YaRN RoPE theta must be greater than one".to_string(),
            ));
        }
        if !scaling.factor.is_finite() || scaling.factor < 1.0 {
            return Err(RotaryEmbeddingError::Invalid(
                "YaRN factor must be finite and at least one".to_string(),
            ));
        }
        if scaling.original_max_position_embeddings == 0 {
            return Err(RotaryEmbeddingError::Invalid(
                "YaRN original_max_position_embeddings must be non-zero".to_string(),
            ));
        }
        if !scaling.beta_fast.is_finite()
            || !scaling.beta_slow.is_finite()
            || scaling.beta_slow <= 0.0
            || scaling.beta_fast <= scaling.beta_slow
        {
            return Err(RotaryEmbeddingError::Invalid(
                "YaRN beta_fast must be finite and greater than positive beta_slow".to_string(),
            ));
        }
        if !scaling.extrapolation_factor.is_finite() || scaling.extrapolation_factor < 0.0 {
            return Err(RotaryEmbeddingError::Invalid(
                "YaRN extrapolation_factor must be finite and non-negative".to_string(),
            ));
        }
        if !scaling.attention_factor.is_finite() || scaling.attention_factor <= 0.0 {
            return Err(RotaryEmbeddingError::Invalid(
                "YaRN attention_factor must be finite and positive".to_string(),
            ));
        }
        if !scaling.mscale.is_finite()
            || !scaling.mscale_all_dim.is_finite()
            || scaling.mscale < 0.0
            || scaling.mscale_all_dim < 0.0
        {
            return Err(RotaryEmbeddingError::Invalid(
                "YaRN mscale values must be finite and non-negative".to_string(),
            ));
        }
        Ok(())
    }
}

pub fn apply_rotary_embedding_inplace(
    values: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    position: usize,
    config: RotaryEmbeddingConfig,
) -> Result<(), RotaryEmbeddingError> {
    if num_heads == 0 || head_dim == 0 {
        return Err(RotaryEmbeddingError::Invalid(
            "RoPE head count and head dimension must be non-zero".to_string(),
        ));
    }
    let expected = num_heads
        .checked_mul(head_dim)
        .ok_or(RotaryEmbeddingError::ShapeOverflow)?;
    if values.len() != expected {
        return Err(RotaryEmbeddingError::Invalid(format!(
            "RoPE values length {} does not match shape [{num_heads}, {head_dim}]",
            values.len()
        )));
    }
    if rotary_dim > head_dim {
        return Err(RotaryEmbeddingError::Invalid(format!(
            "rotary dimension {rotary_dim} exceeds head dimension {head_dim}"
        )));
    }
    let frequencies = config.inverse_frequencies(rotary_dim)?;
    let magnitude_scale = config.magnitude_scale();
    for head in 0..num_heads {
        let head_offset = head * head_dim;
        for (pair_index, frequency) in frequencies.iter().copied().enumerate() {
            let (first_index, second_index) = match config.style() {
                RotaryEmbeddingStyle::Neox => (
                    head_offset + pair_index,
                    head_offset + rotary_dim / 2 + pair_index,
                ),
                RotaryEmbeddingStyle::Interleaved => (
                    head_offset + pair_index * 2,
                    head_offset + pair_index * 2 + 1,
                ),
            };
            let angle = position as f32 * frequency;
            let (sin, cos) = angle.sin_cos();
            let first = values[first_index];
            let second = values[second_index];
            values[first_index] = (first * cos - second * sin) * magnitude_scale;
            values[second_index] = (second * cos + first * sin) * magnitude_scale;
        }
    }
    Ok(())
}

fn yarn_mscale(factor: f32, multiplier: f32) -> f32 {
    if factor <= 1.0 {
        1.0
    } else {
        0.1 * multiplier * factor.ln() + 1.0
    }
}

fn yarn_correction_range(
    beta_fast: f64,
    beta_slow: f64,
    rotary_dim: usize,
    theta: f64,
    original_max_position_embeddings: usize,
) -> (f64, f64) {
    let correction_dim = |rotations: f64| {
        rotary_dim as f64
            * (original_max_position_embeddings as f64 / (rotations * 2.0 * std::f64::consts::PI))
                .ln()
            / (2.0 * theta.ln())
    };
    let maximum = rotary_dim.saturating_sub(1) as f64;
    (
        correction_dim(beta_fast).floor().clamp(0.0, maximum),
        correction_dim(beta_slow).ceil().clamp(0.0, maximum),
    )
}

fn yarn_linear_ramp(index: f64, low: f64, high: f64) -> f64 {
    let high = if low == high { high + 0.001 } else { high };
    ((index - low) / (high - low)).clamp(0.0, 1.0)
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum RotaryEmbeddingError {
    Invalid(String),
    ShapeOverflow,
}

impl fmt::Display for RotaryEmbeddingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => formatter.write_str(message),
            Self::ShapeOverflow => formatter.write_str("RoPE tensor shape overflowed"),
        }
    }
}

impl std::error::Error for RotaryEmbeddingError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn official_profile() -> RotaryEmbeddingConfig {
        RotaryEmbeddingConfig::yarn(
            50_000.0,
            RotaryEmbeddingStyle::Interleaved,
            YarnScalingConfig {
                factor: 64.0,
                original_max_position_embeddings: 4096,
                beta_fast: 32.0,
                beta_slow: 1.0,
                extrapolation_factor: 1.0,
                attention_factor: 1.0,
                mscale: 1.0,
                mscale_all_dim: 1.0,
                apply_attention_scale: true,
            },
        )
        .expect("official profile should be valid")
    }

    #[test]
    fn interleaved_style_rotates_adjacent_dimensions() {
        let config = RotaryEmbeddingConfig::standard(10_000.0, RotaryEmbeddingStyle::Interleaved)
            .expect("default RoPE should be valid");
        let mut values = vec![1.0, 2.0, 3.0, 4.0];

        apply_rotary_embedding_inplace(&mut values, 1, 4, 4, 1, config)
            .expect("interleaved RoPE should run");

        let (sin_0, cos_0) = 1.0_f32.sin_cos();
        let (sin_1, cos_1) = 0.01_f32.sin_cos();
        assert!((values[0] - (cos_0 - 2.0 * sin_0)).abs() < 1e-6);
        assert!((values[1] - (2.0 * cos_0 + sin_0)).abs() < 1e-6);
        assert!((values[2] - (3.0 * cos_1 - 4.0 * sin_1)).abs() < 1e-6);
        assert!((values[3] - (4.0 * cos_1 + 3.0 * sin_1)).abs() < 1e-6);
    }

    #[test]
    fn yarn_frequency_and_mla_scale_match_community_formula() {
        let config = official_profile();
        let frequencies = config
            .inverse_frequencies(64)
            .expect("official frequencies should build");
        let base = 192.0_f32.sqrt().recip();
        let expected_mscale = 0.1 * 64.0_f32.ln() + 1.0;

        assert_eq!(frequencies.len(), 32);
        assert!(
            frequencies
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
        );
        for (index, expected) in [
            (0, 1.0_f32),
            (8, 0.066_874_03),
            (16, 0.001_537_296_7),
            (24, 0.000_004_672_965),
            (31, 0.000_000_438_220_65),
        ] {
            assert!((frequencies[index] - expected).abs() < expected * 1e-5 + 1e-9);
        }
        assert!((config.magnitude_scale() - 1.0).abs() < 1e-6);
        assert!((config.mla_attention_scale(base) - base * expected_mscale.powi(2)).abs() < 1e-6);
    }

    #[test]
    fn yarn_rejects_invalid_scaling_before_execution() {
        let error = RotaryEmbeddingConfig::yarn(
            50_000.0,
            RotaryEmbeddingStyle::Interleaved,
            YarnScalingConfig {
                factor: 0.5,
                original_max_position_embeddings: 4096,
                beta_fast: 32.0,
                beta_slow: 1.0,
                extrapolation_factor: 1.0,
                attention_factor: 1.0,
                mscale: 1.0,
                mscale_all_dim: 1.0,
                apply_attention_scale: true,
            },
        )
        .expect_err("invalid factor must fail fast");

        assert!(error.to_string().contains("factor"));
    }
}
