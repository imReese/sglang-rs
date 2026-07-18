use std::fmt;

use crate::models::{MoeFeedForwardConfig, RouterActivation};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RoutedExpert {
    pub(crate) index: usize,
    pub(crate) weight: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct MoeRouter {
    config: MoeFeedForwardConfig,
    correction_bias: Option<Vec<f32>>,
}

impl MoeRouter {
    pub(crate) fn new(
        config: MoeFeedForwardConfig,
        correction_bias: Option<Vec<f32>>,
    ) -> Result<Self, MoeRoutingError> {
        if config.routed_expert_count == 0 {
            return Err(MoeRoutingError::ZeroDimension("routed_expert_count"));
        }
        if config.experts_per_token == 0 {
            return Err(MoeRoutingError::ZeroDimension("experts_per_token"));
        }
        if config.experts_per_token > config.routed_expert_count {
            return Err(MoeRoutingError::ExpertsPerTokenOutOfRange {
                experts_per_token: config.experts_per_token,
                routed_expert_count: config.routed_expert_count,
            });
        }
        if config.expert_group_count == 0 {
            return Err(MoeRoutingError::ZeroDimension("expert_group_count"));
        }
        if !config
            .routed_expert_count
            .is_multiple_of(config.expert_group_count)
        {
            return Err(MoeRoutingError::UnevenExpertGroups {
                routed_expert_count: config.routed_expert_count,
                expert_group_count: config.expert_group_count,
            });
        }
        if config.selected_expert_group_count == 0
            || config.selected_expert_group_count > config.expert_group_count
        {
            return Err(MoeRoutingError::SelectedGroupCountOutOfRange {
                selected: config.selected_expert_group_count,
                available: config.expert_group_count,
            });
        }
        let experts_per_group = config.routed_expert_count / config.expert_group_count;
        let selected_capacity = experts_per_group
            .checked_mul(config.selected_expert_group_count)
            .ok_or(MoeRoutingError::ShapeOverflow)?;
        if config.experts_per_token > selected_capacity {
            return Err(MoeRoutingError::SelectedGroupsTooSmall {
                experts_per_token: config.experts_per_token,
                selected_capacity,
            });
        }
        if config.expert_intermediate_size == 0 {
            return Err(MoeRoutingError::ZeroDimension("expert_intermediate_size"));
        }
        if !config.routed_scaling_factor.is_finite() || config.routed_scaling_factor <= 0.0 {
            return Err(MoeRoutingError::InvalidScalingFactor(
                config.routed_scaling_factor,
            ));
        }
        if let Some(bias) = &correction_bias
            && bias.len() != config.routed_expert_count
        {
            return Err(MoeRoutingError::CorrectionBiasCountMismatch {
                expected: config.routed_expert_count,
                actual: bias.len(),
            });
        }
        Ok(Self {
            config,
            correction_bias,
        })
    }

    pub(crate) fn config(&self) -> &MoeFeedForwardConfig {
        &self.config
    }

    pub(crate) fn route(&self, logits: &[f32]) -> Result<Vec<RoutedExpert>, MoeRoutingError> {
        if logits.len() != self.config.routed_expert_count {
            return Err(MoeRoutingError::LogitCountMismatch {
                expected: self.config.routed_expert_count,
                actual: logits.len(),
            });
        }
        if let Some((index, value)) = logits
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(MoeRoutingError::NonFiniteLogit { index, value });
        }

        let probabilities = match self.config.router_activation {
            RouterActivation::Sigmoid => logits
                .iter()
                .copied()
                .map(stable_sigmoid)
                .collect::<Vec<_>>(),
            RouterActivation::Softmax => normalized_exponentials(logits)?,
        };
        let selection_scores = probabilities
            .iter()
            .enumerate()
            .map(|(expert, probability)| {
                probability
                    + self
                        .correction_bias
                        .as_ref()
                        .map_or(0.0, |bias| bias[expert])
            })
            .collect::<Vec<_>>();
        if let Some((index, value)) = selection_scores
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(MoeRoutingError::NonFiniteSelectionScore { index, value });
        }

        let experts_per_group = self.config.routed_expert_count / self.config.expert_group_count;
        let mut group_scores = (0..self.config.expert_group_count)
            .map(|group| {
                let start = group * experts_per_group;
                let mut scores = selection_scores[start..start + experts_per_group].to_vec();
                scores.sort_by(|left, right| right.total_cmp(left));
                (group, scores.into_iter().take(2).sum::<f32>())
            })
            .collect::<Vec<_>>();
        group_scores.sort_by(|(left_group, left), (right_group, right)| {
            right
                .total_cmp(left)
                .then_with(|| left_group.cmp(right_group))
        });
        let mut selected_groups = vec![false; self.config.expert_group_count];
        for (group, _) in group_scores
            .into_iter()
            .take(self.config.selected_expert_group_count)
        {
            selected_groups[group] = true;
        }

        let mut selected = selection_scores
            .iter()
            .copied()
            .enumerate()
            .filter(|(expert, _)| selected_groups[expert / experts_per_group])
            .collect::<Vec<_>>();
        selected.sort_by(|(left_expert, left), (right_expert, right)| {
            right
                .total_cmp(left)
                .then_with(|| left_expert.cmp(right_expert))
        });
        selected.truncate(self.config.experts_per_token);

        let normalizer = if self.config.renormalize {
            let sum = selected
                .iter()
                .map(|(expert, _)| probabilities[*expert])
                .sum::<f32>();
            if !sum.is_finite() || sum <= 0.0 {
                return Err(MoeRoutingError::InvalidNormalization(sum));
            }
            sum
        } else {
            1.0
        };
        selected
            .into_iter()
            .map(|(index, _)| {
                let weight = probabilities[index] / normalizer * self.config.routed_scaling_factor;
                if !weight.is_finite() || weight < 0.0 {
                    return Err(MoeRoutingError::InvalidRoutedWeight { index, weight });
                }
                Ok(RoutedExpert { index, weight })
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum MoeRoutingError {
    ZeroDimension(&'static str),
    ExpertsPerTokenOutOfRange {
        experts_per_token: usize,
        routed_expert_count: usize,
    },
    UnevenExpertGroups {
        routed_expert_count: usize,
        expert_group_count: usize,
    },
    SelectedGroupCountOutOfRange {
        selected: usize,
        available: usize,
    },
    SelectedGroupsTooSmall {
        experts_per_token: usize,
        selected_capacity: usize,
    },
    InvalidScalingFactor(f32),
    CorrectionBiasCountMismatch {
        expected: usize,
        actual: usize,
    },
    LogitCountMismatch {
        expected: usize,
        actual: usize,
    },
    NonFiniteLogit {
        index: usize,
        value: f32,
    },
    NonFiniteSelectionScore {
        index: usize,
        value: f32,
    },
    InvalidNormalization(f32),
    InvalidRoutedWeight {
        index: usize,
        weight: f32,
    },
    ShapeOverflow,
}

impl fmt::Display for MoeRoutingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension(dimension) => {
                write!(formatter, "MoE dimension {dimension} must be non-zero")
            }
            Self::ExpertsPerTokenOutOfRange {
                experts_per_token,
                routed_expert_count,
            } => write!(
                formatter,
                "MoE experts_per_token {experts_per_token} exceeds routed expert count {routed_expert_count}"
            ),
            Self::UnevenExpertGroups {
                routed_expert_count,
                expert_group_count,
            } => write!(
                formatter,
                "MoE routed expert count {routed_expert_count} is not divisible by {expert_group_count} groups"
            ),
            Self::SelectedGroupCountOutOfRange {
                selected,
                available,
            } => write!(
                formatter,
                "MoE selected expert group count {selected} must be within 1..={available}"
            ),
            Self::SelectedGroupsTooSmall {
                experts_per_token,
                selected_capacity,
            } => write!(
                formatter,
                "MoE selected groups contain {selected_capacity} experts, fewer than top-k {experts_per_token}"
            ),
            Self::InvalidScalingFactor(value) => write!(
                formatter,
                "MoE routed scaling factor must be finite and positive, got {value}"
            ),
            Self::CorrectionBiasCountMismatch { expected, actual } => write!(
                formatter,
                "MoE correction bias has {actual} values, expected {expected}"
            ),
            Self::LogitCountMismatch { expected, actual } => write!(
                formatter,
                "MoE router produced {actual} logits, expected {expected}"
            ),
            Self::NonFiniteLogit { index, value } => {
                write!(formatter, "MoE router logit {index} is not finite: {value}")
            }
            Self::NonFiniteSelectionScore { index, value } => write!(
                formatter,
                "MoE router selection score {index} is not finite: {value}"
            ),
            Self::InvalidNormalization(value) => write!(
                formatter,
                "MoE routed probability normalization is invalid: {value}"
            ),
            Self::InvalidRoutedWeight { index, weight } => write!(
                formatter,
                "MoE routed expert {index} has invalid output weight {weight}"
            ),
            Self::ShapeOverflow => formatter.write_str("MoE routing shape overflowed"),
        }
    }
}

impl std::error::Error for MoeRoutingError {}

fn stable_sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponent = value.exp();
        exponent / (1.0 + exponent)
    }
}

fn normalized_exponentials(values: &[f32]) -> Result<Vec<f32>, MoeRoutingError> {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials = values
        .iter()
        .map(|value| (value - maximum).exp())
        .collect::<Vec<_>>();
    let sum = exponentials.iter().sum::<f32>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(MoeRoutingError::InvalidNormalization(sum));
    }
    Ok(exponentials.into_iter().map(|value| value / sum).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> MoeFeedForwardConfig {
        MoeFeedForwardConfig {
            routed_expert_count: 4,
            experts_per_token: 1,
            expert_group_count: 2,
            selected_expert_group_count: 1,
            shared_expert_count: 1,
            expert_intermediate_size: 8,
            renormalize: true,
            router_activation: RouterActivation::Sigmoid,
            routed_scaling_factor: 2.0,
        }
    }

    #[test]
    fn grouped_routing_uses_correction_bias_only_for_selection() {
        let router =
            MoeRouter::new(test_config(), Some(vec![0.0, 0.0, 1.0, 0.0])).expect("valid router");
        let routed = router.route(&[0.0; 4]).expect("routing should succeed");

        assert_eq!(
            routed,
            vec![RoutedExpert {
                index: 2,
                weight: 2.0
            }]
        );
    }

    #[test]
    fn softmax_routing_preserves_unrenormalized_probabilities() {
        let router = MoeRouter::new(
            MoeFeedForwardConfig {
                experts_per_token: 2,
                expert_group_count: 1,
                selected_expert_group_count: 1,
                renormalize: false,
                router_activation: RouterActivation::Softmax,
                routed_scaling_factor: 1.0,
                ..test_config()
            },
            None,
        )
        .expect("valid router");
        let routed = router
            .route(&[2.0, 1.0, 0.0, -1.0])
            .expect("routing should succeed");

        assert_eq!(routed.len(), 2);
        assert_eq!(routed[0].index, 0);
        assert_eq!(routed[1].index, 1);
        assert!(routed.iter().map(|expert| expert.weight).sum::<f32>() < 1.0);
    }

    #[test]
    fn router_rejects_group_capacity_smaller_than_top_k() {
        assert!(matches!(
            MoeRouter::new(
                MoeFeedForwardConfig {
                    experts_per_token: 3,
                    ..test_config()
                },
                None,
            ),
            Err(MoeRoutingError::SelectedGroupsTooSmall {
                experts_per_token: 3,
                selected_capacity: 2,
            })
        ));
    }

    #[test]
    fn router_rejects_non_finite_logits() {
        let router = MoeRouter::new(test_config(), None).expect("valid router");
        assert!(matches!(
            router.route(&[0.0, f32::NAN, 0.0, 0.0]),
            Err(MoeRoutingError::NonFiniteLogit { index: 1, .. })
        ));
    }
}
