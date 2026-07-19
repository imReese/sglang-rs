use std::fmt;

use crate::models::{AttentionArchitecture, FeedForwardArchitecture, ModelExecutionArchitecture};
use crate::parallel::{TensorParallelRank, TensorPartition, TensorPartitionError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DenseTensorParallelPlan {
    query_partition: TensorPartition,
    kv_partition: TensorPartition,
    intermediate_partition: TensorPartition,
    local_query_head_count: usize,
    local_kv_head_count: usize,
    head_dim: usize,
    local_intermediate_size: usize,
}

impl DenseTensorParallelPlan {
    pub(crate) fn from_execution(
        execution: ModelExecutionArchitecture,
        rank: TensorParallelRank,
    ) -> Result<Self, DenseTensorParallelError> {
        let ModelExecutionArchitecture::Transformer {
            attention:
                AttentionArchitecture::MultiHead {
                    num_attention_heads,
                    num_key_value_heads,
                    head_dim,
                },
            feed_forward: FeedForwardArchitecture::Dense { intermediate_size },
        } = execution
        else {
            return Err(DenseTensorParallelError::UnsupportedArchitecture);
        };
        let world_size = rank.world_size();
        if !num_attention_heads.is_multiple_of(world_size) {
            return Err(DenseTensorParallelError::QueryHeadsNotDivisible {
                query_heads: num_attention_heads,
                world_size,
            });
        }
        if !intermediate_size.is_multiple_of(world_size) {
            return Err(DenseTensorParallelError::IntermediateNotDivisible {
                intermediate_size,
                world_size,
            });
        }

        let query_partition = TensorPartition::for_rank(rank);
        let intermediate_partition = TensorPartition::for_rank(rank);
        let (kv_partition, local_kv_head_count) = if num_key_value_heads >= world_size {
            if !num_key_value_heads.is_multiple_of(world_size) {
                return Err(DenseTensorParallelError::KvHeadsNotDivisible {
                    kv_heads: num_key_value_heads,
                    world_size,
                });
            }
            (
                TensorPartition::for_rank(rank),
                num_key_value_heads / world_size,
            )
        } else {
            (
                TensorPartition::for_replicated_shards(rank, num_key_value_heads)?,
                1,
            )
        };

        Ok(Self {
            query_partition,
            kv_partition,
            intermediate_partition,
            local_query_head_count: num_attention_heads / world_size,
            local_kv_head_count,
            head_dim,
            local_intermediate_size: intermediate_size / world_size,
        })
    }

    pub(crate) fn query_partition(self) -> TensorPartition {
        self.query_partition
    }

    pub(crate) fn kv_partition(self) -> TensorPartition {
        self.kv_partition
    }

    pub(crate) fn intermediate_partition(self) -> TensorPartition {
        self.intermediate_partition
    }

    pub(crate) fn local_query_head_count(self) -> usize {
        self.local_query_head_count
    }

    pub(crate) fn local_kv_head_count(self) -> usize {
        self.local_kv_head_count
    }

    pub(crate) fn local_query_size(self) -> Result<usize, DenseTensorParallelError> {
        checked_product(
            self.local_query_head_count,
            self.head_dim,
            "local query width",
        )
    }

    pub(crate) fn local_kv_size(self) -> Result<usize, DenseTensorParallelError> {
        checked_product(self.local_kv_head_count, self.head_dim, "local KV width")
    }

    pub(crate) fn local_intermediate_size(self) -> usize {
        self.local_intermediate_size
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum DenseTensorParallelError {
    UnsupportedArchitecture,
    QueryHeadsNotDivisible {
        query_heads: usize,
        world_size: usize,
    },
    KvHeadsNotDivisible {
        kv_heads: usize,
        world_size: usize,
    },
    IntermediateNotDivisible {
        intermediate_size: usize,
        world_size: usize,
    },
    SizeOverflow {
        name: &'static str,
    },
    Partition(TensorPartitionError),
}

impl fmt::Display for DenseTensorParallelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedArchitecture => formatter.write_str(
                "dense tensor parallel execution requires multi-head attention and dense feed-forward components",
            ),
            Self::QueryHeadsNotDivisible {
                query_heads,
                world_size,
            } => write!(
                formatter,
                "query head count {query_heads} must be divisible by tensor parallel world size {world_size}"
            ),
            Self::KvHeadsNotDivisible {
                kv_heads,
                world_size,
            } => write!(
                formatter,
                "KV head count {kv_heads} must shard evenly across tensor parallel world size {world_size}"
            ),
            Self::IntermediateNotDivisible {
                intermediate_size,
                world_size,
            } => write!(
                formatter,
                "feed-forward intermediate size {intermediate_size} must be divisible by tensor parallel world size {world_size}"
            ),
            Self::SizeOverflow { name } => write!(formatter, "{name} overflowed"),
            Self::Partition(error) => write!(formatter, "tensor partition is invalid: {error}"),
        }
    }
}

impl std::error::Error for DenseTensorParallelError {}

impl From<TensorPartitionError> for DenseTensorParallelError {
    fn from(value: TensorPartitionError) -> Self {
        Self::Partition(value)
    }
}

fn checked_product(
    left: usize,
    right: usize,
    name: &'static str,
) -> Result<usize, DenseTensorParallelError> {
    left.checked_mul(right)
        .ok_or(DenseTensorParallelError::SizeOverflow { name })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::TensorParallelTopology;

    #[test]
    fn shards_dense_attention_and_feed_forward_per_rank() {
        let topology = TensorParallelTopology::new(2, 1, 0, 0, 1).expect("topology");
        let execution = dense_execution(8, 4, 2, 16);
        let rank_one =
            DenseTensorParallelPlan::from_execution(execution, topology.local_ranks()[1])
                .expect("rank plan");

        assert_eq!(rank_one.local_query_head_count(), 4);
        assert_eq!(rank_one.local_kv_head_count(), 2);
        assert_eq!(rank_one.local_query_size().expect("query size"), 8);
        assert_eq!(rank_one.local_kv_size().expect("KV size"), 4);
        assert_eq!(rank_one.local_intermediate_size(), 8);
        assert_eq!(rank_one.query_partition().range(16).expect("query"), 8..16);
        assert_eq!(rank_one.kv_partition().range(8).expect("KV"), 4..8);
        assert_eq!(
            rank_one
                .intermediate_partition()
                .range(16)
                .expect("intermediate"),
            8..16
        );
    }

    #[test]
    fn replicates_kv_heads_when_world_size_exceeds_kv_heads() {
        let topology = TensorParallelTopology::new(8, 1, 0, 0, 1).expect("topology");
        let execution = dense_execution(8, 2, 4, 16);
        let rank_five =
            DenseTensorParallelPlan::from_execution(execution, topology.local_ranks()[5])
                .expect("rank plan");

        assert_eq!(rank_five.local_query_head_count(), 1);
        assert_eq!(rank_five.local_kv_head_count(), 1);
        assert_eq!(rank_five.kv_partition().partition_count(), 2);
        assert_eq!(rank_five.kv_partition().partition_index(), 1);
        assert_eq!(rank_five.kv_partition().range(8).expect("KV"), 4..8);
    }

    fn dense_execution(
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
    ) -> ModelExecutionArchitecture {
        ModelExecutionArchitecture::Transformer {
            attention: AttentionArchitecture::MultiHead {
                num_attention_heads: query_heads,
                num_key_value_heads: kv_heads,
                head_dim,
            },
            feed_forward: FeedForwardArchitecture::Dense { intermediate_size },
        }
    }
}
