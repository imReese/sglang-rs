// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver, Sender};

use crate::parallel_topology::EpRank;

/// Error conditions arising during Expert Parallel (EP) communication or routing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MoeCommError {
    ZeroDimension(&'static str),
    UnevenSharding {
        total_experts: usize,
        ep_size: usize,
    },
    InvalidExpertId {
        expert_id: usize,
        total_experts: usize,
    },
    RankOutOfRange {
        rank: usize,
        ep_size: usize,
    },
    ChannelClosed,
    HiddenDimensionMismatch {
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for MoeCommError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension(name) => write!(f, "MoE dimension '{name}' must be non-zero"),
            Self::UnevenSharding {
                total_experts,
                ep_size,
            } => write!(
                f,
                "total experts ({total_experts}) must be divisible by ep_size ({ep_size})"
            ),
            Self::InvalidExpertId {
                expert_id,
                total_experts,
            } => write!(
                f,
                "expert_id {expert_id} out of bounds for total_experts {total_experts}"
            ),
            Self::RankOutOfRange { rank, ep_size } => {
                write!(f, "rank {rank} out of bounds for ep_size {ep_size}")
            }
            Self::ChannelClosed => write!(f, "MoE communication channel closed unexpectedly"),
            Self::HiddenDimensionMismatch { expected, actual } => write!(
                f,
                "hidden dimension mismatch: expected {expected}, actual {actual}"
            ),
        }
    }
}

impl std::error::Error for MoeCommError {}

/// Maps global MoE expert IDs to specific EP ranks and local expert indices.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertMapping {
    total_experts: usize,
    ep_size: usize,
    experts_per_rank: usize,
}

impl ExpertMapping {
    pub fn new(total_experts: usize, ep_size: usize) -> Result<Self, MoeCommError> {
        if total_experts == 0 {
            return Err(MoeCommError::ZeroDimension("total_experts"));
        }
        if ep_size == 0 {
            return Err(MoeCommError::ZeroDimension("ep_size"));
        }
        if !total_experts.is_multiple_of(ep_size) {
            return Err(MoeCommError::UnevenSharding {
                total_experts,
                ep_size,
            });
        }
        let experts_per_rank = total_experts / ep_size;
        Ok(Self {
            total_experts,
            ep_size,
            experts_per_rank,
        })
    }

    pub fn total_experts(&self) -> usize {
        self.total_experts
    }

    pub fn ep_size(&self) -> usize {
        self.ep_size
    }

    pub fn experts_per_rank(&self) -> usize {
        self.experts_per_rank
    }

    /// Determines which EP rank hosts the given global expert.
    pub fn rank_for_expert(&self, expert_id: usize) -> Result<EpRank, MoeCommError> {
        if expert_id >= self.total_experts {
            return Err(MoeCommError::InvalidExpertId {
                expert_id,
                total_experts: self.total_experts,
            });
        }
        Ok(EpRank(expert_id / self.experts_per_rank))
    }

    /// Determines the local index within the host rank for the given global expert.
    pub fn local_expert_index(&self, expert_id: usize) -> Result<usize, MoeCommError> {
        if expert_id >= self.total_experts {
            return Err(MoeCommError::InvalidExpertId {
                expert_id,
                total_experts: self.total_experts,
            });
        }
        Ok(expert_id % self.experts_per_rank)
    }

    /// Converts a local expert index on a specific EP rank back to its global expert ID.
    pub fn global_expert_id(
        &self,
        rank: EpRank,
        local_index: usize,
    ) -> Result<usize, MoeCommError> {
        if rank.0 >= self.ep_size {
            return Err(MoeCommError::RankOutOfRange {
                rank: rank.0,
                ep_size: self.ep_size,
            });
        }
        if local_index >= self.experts_per_rank {
            return Err(MoeCommError::InvalidExpertId {
                expert_id: local_index,
                total_experts: self.experts_per_rank,
            });
        }
        Ok(rank.0 * self.experts_per_rank + local_index)
    }

    /// Returns the range of global expert IDs hosted on the given EP rank.
    pub fn global_expert_range(
        &self,
        rank: EpRank,
    ) -> Result<std::ops::Range<usize>, MoeCommError> {
        if rank.0 >= self.ep_size {
            return Err(MoeCommError::RankOutOfRange {
                rank: rank.0,
                ep_size: self.ep_size,
            });
        }
        let start = rank.0 * self.experts_per_rank;
        Ok(start..(start + self.experts_per_rank))
    }
}

/// Token payload dispatched from a source rank to the EP rank hosting the routed expert.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenDispatchPacket {
    pub token_idx: usize,
    pub source_rank: EpRank,
    pub target_rank: EpRank,
    pub expert_id: usize,
    pub local_expert_id: usize,
    pub routing_weight: f32,
    pub hidden_state: Vec<f32>,
}

/// Token computation output returned from the EP rank hosting the expert back to the source rank.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenCombinePacket {
    pub token_idx: usize,
    pub source_rank: EpRank,
    pub expert_id: usize,
    pub routing_weight: f32,
    pub output_state: Vec<f32>,
}

/// Communicator abstraction for executing MoE token dispatch and combine All-to-All operations.
pub trait ExpertParallelCommunicator: Send + Sync {
    fn ep_rank(&self) -> EpRank;
    fn ep_world_size(&self) -> usize;

    /// Dispatches outbound token hidden states to the EP ranks hosting their selected experts,
    /// and collects inbound tokens destined for local experts on this rank.
    fn dispatch_all_to_all(
        &self,
        outbound: Vec<TokenDispatchPacket>,
    ) -> Result<Vec<TokenDispatchPacket>, MoeCommError>;

    /// Returns computed expert outputs back to their origin ranks, and receives inbound
    /// expert outputs for tokens originally owned by this rank.
    fn combine_all_to_all(
        &self,
        outbound: Vec<TokenCombinePacket>,
    ) -> Result<Vec<TokenCombinePacket>, MoeCommError>;
}

/// Intra-process channel-based MoE communicator providing deadlock-free All-to-All collectives.
pub struct ChannelMoeCommunicator {
    rank: EpRank,
    ep_size: usize,
    dispatch_senders: Vec<Sender<Vec<TokenDispatchPacket>>>,
    dispatch_receiver: Mutex<Receiver<Vec<TokenDispatchPacket>>>,
    combine_senders: Vec<Sender<Vec<TokenCombinePacket>>>,
    combine_receiver: Mutex<Receiver<Vec<TokenCombinePacket>>>,
}

impl ChannelMoeCommunicator {
    /// Creates a fully-connected communication group of $N$ channel communicators.
    pub fn create_group(ep_size: usize) -> Result<Vec<Self>, MoeCommError> {
        if ep_size == 0 {
            return Err(MoeCommError::ZeroDimension("ep_size"));
        }

        let mut dispatch_txs = Vec::with_capacity(ep_size);
        let mut dispatch_rxs = Vec::with_capacity(ep_size);
        let mut combine_txs = Vec::with_capacity(ep_size);
        let mut combine_rxs = Vec::with_capacity(ep_size);

        for _ in 0..ep_size {
            let (dtx, drx) = mpsc::channel();
            let (ctx, crx) = mpsc::channel();
            dispatch_txs.push(dtx);
            dispatch_rxs.push(drx);
            combine_txs.push(ctx);
            combine_rxs.push(crx);
        }

        let mut communicators = Vec::with_capacity(ep_size);
        for (rank_idx, (drx, crx)) in dispatch_rxs.into_iter().zip(combine_rxs).enumerate() {
            communicators.push(Self {
                rank: EpRank(rank_idx),
                ep_size,
                dispatch_senders: dispatch_txs.clone(),
                dispatch_receiver: Mutex::new(drx),
                combine_senders: combine_txs.clone(),
                combine_receiver: Mutex::new(crx),
            });
        }

        Ok(communicators)
    }
}

impl ExpertParallelCommunicator for ChannelMoeCommunicator {
    fn ep_rank(&self) -> EpRank {
        self.rank
    }

    fn ep_world_size(&self) -> usize {
        self.ep_size
    }

    fn dispatch_all_to_all(
        &self,
        outbound: Vec<TokenDispatchPacket>,
    ) -> Result<Vec<TokenDispatchPacket>, MoeCommError> {
        // 1. Partition outbound tokens by target rank
        let mut partitioned: Vec<Vec<TokenDispatchPacket>> =
            (0..self.ep_size).map(|_| Vec::new()).collect();

        for packet in outbound {
            if packet.target_rank.0 >= self.ep_size {
                return Err(MoeCommError::RankOutOfRange {
                    rank: packet.target_rank.0,
                    ep_size: self.ep_size,
                });
            }
            partitioned[packet.target_rank.0].push(packet);
        }

        // 2. Dispatch batches to all target ranks (including self)
        for (target_rank, batch) in partitioned.into_iter().enumerate() {
            self.dispatch_senders[target_rank]
                .send(batch)
                .map_err(|_| MoeCommError::ChannelClosed)?;
        }

        // 3. Receive exactly ep_size batches from all ranks
        let rx = self
            .dispatch_receiver
            .lock()
            .map_err(|_| MoeCommError::ChannelClosed)?;
        let mut inbound = Vec::new();
        for _ in 0..self.ep_size {
            let mut batch = rx.recv().map_err(|_| MoeCommError::ChannelClosed)?;
            inbound.append(&mut batch);
        }

        Ok(inbound)
    }

    fn combine_all_to_all(
        &self,
        outbound: Vec<TokenCombinePacket>,
    ) -> Result<Vec<TokenCombinePacket>, MoeCommError> {
        // 1. Partition outbound results by source rank (where the token originated)
        let mut partitioned: Vec<Vec<TokenCombinePacket>> =
            (0..self.ep_size).map(|_| Vec::new()).collect();

        for packet in outbound {
            if packet.source_rank.0 >= self.ep_size {
                return Err(MoeCommError::RankOutOfRange {
                    rank: packet.source_rank.0,
                    ep_size: self.ep_size,
                });
            }
            partitioned[packet.source_rank.0].push(packet);
        }

        // 2. Send batches back to origin ranks
        for (target_rank, batch) in partitioned.into_iter().enumerate() {
            self.combine_senders[target_rank]
                .send(batch)
                .map_err(|_| MoeCommError::ChannelClosed)?;
        }

        // 3. Receive exactly ep_size batches
        let rx = self
            .combine_receiver
            .lock()
            .map_err(|_| MoeCommError::ChannelClosed)?;
        let mut inbound = Vec::new();
        for _ in 0..self.ep_size {
            let mut batch = rx.recv().map_err(|_| MoeCommError::ChannelClosed)?;
            inbound.append(&mut batch);
        }

        Ok(inbound)
    }
}

/// Recombines received expert output states according to their routing weights:
/// $$y_{\text{token}} = \sum_{e \in \text{selected}} w_e \cdot y_e$$
pub fn recombine_token_states(
    packets: &[TokenCombinePacket],
    num_tokens: usize,
    hidden_dim: usize,
) -> Result<Vec<Vec<f32>>, MoeCommError> {
    let mut combined: Vec<Vec<f32>> = vec![vec![0.0; hidden_dim]; num_tokens];

    for packet in packets {
        if packet.token_idx >= num_tokens {
            return Err(MoeCommError::InvalidExpertId {
                expert_id: packet.token_idx,
                total_experts: num_tokens,
            });
        }
        if packet.output_state.len() != hidden_dim {
            return Err(MoeCommError::HiddenDimensionMismatch {
                expected: hidden_dim,
                actual: packet.output_state.len(),
            });
        }
        let token_slice = &mut combined[packet.token_idx];
        let weight = packet.routing_weight;
        for (dst, src) in token_slice.iter_mut().zip(&packet.output_state) {
            *dst += weight * *src;
        }
    }

    Ok(combined)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_expert_mapping() {
        let mapping = ExpertMapping::new(16, 4).expect("valid mapping");
        assert_eq!(mapping.total_experts(), 16);
        assert_eq!(mapping.ep_size(), 4);
        assert_eq!(mapping.experts_per_rank(), 4);

        // Check rank and local index queries
        assert_eq!(mapping.rank_for_expert(0).unwrap(), EpRank(0));
        assert_eq!(mapping.local_expert_index(0).unwrap(), 0);

        assert_eq!(mapping.rank_for_expert(7).unwrap(), EpRank(1));
        assert_eq!(mapping.local_expert_index(7).unwrap(), 3);

        assert_eq!(mapping.rank_for_expert(15).unwrap(), EpRank(3));
        assert_eq!(mapping.local_expert_index(15).unwrap(), 3);

        // Check reverse conversion
        assert_eq!(mapping.global_expert_id(EpRank(2), 1).unwrap(), 9);

        // Check range
        assert_eq!(mapping.global_expert_range(EpRank(1)).unwrap(), 4..8);

        // Out of bounds
        assert!(mapping.rank_for_expert(16).is_err());
        assert!(ExpertMapping::new(15, 4).is_err()); // uneven
    }

    #[test]
    fn test_moe_all_to_all_round_trip() {
        let ep_size = 4;
        let total_experts = 16;
        let mapping = ExpertMapping::new(total_experts, ep_size).unwrap();
        let communicators = ChannelMoeCommunicator::create_group(ep_size).unwrap();
        assert_eq!(communicators.len(), ep_size);

        let hidden_dim = 4;

        // Spawn 4 threads representing 4 EP ranks
        let handles: Vec<_> = communicators
            .into_iter()
            .map(|comm| {
                let rank = comm.ep_rank();
                thread::spawn(move || {
                    // Each rank generates 1 token with hidden_state = [rank as f32, ...]
                    // Token on rank 0 selects expert 5 (hosted on Rank 1) with weight 0.5
                    // Token on rank 1 selects expert 10 (hosted on Rank 2) with weight 0.5
                    // Token on rank 2 selects expert 15 (hosted on Rank 3) with weight 0.5
                    // Token on rank 3 selects expert 0 (hosted on Rank 0) with weight 0.5
                    let target_expert = match rank.0 {
                        0 => 5,
                        1 => 10,
                        2 => 15,
                        3 => 0,
                        _ => unreachable!(),
                    };
                    let target_rank = mapping.rank_for_expert(target_expert).unwrap();
                    let local_expert = mapping.local_expert_index(target_expert).unwrap();

                    let outbound_dispatch = vec![TokenDispatchPacket {
                        token_idx: 0,
                        source_rank: rank,
                        target_rank,
                        expert_id: target_expert,
                        local_expert_id: local_expert,
                        routing_weight: 0.5,
                        hidden_state: vec![(rank.0 + 1) as f32; hidden_dim],
                    }];

                    // 1. Dispatch All-to-All
                    let received_tokens = comm.dispatch_all_to_all(outbound_dispatch).unwrap();
                    assert_eq!(
                        received_tokens.len(),
                        1,
                        "Rank {:?} should receive 1 token",
                        rank
                    );

                    // 2. Mock expert computation: scale hidden_state by 2.0
                    let outbound_combine: Vec<TokenCombinePacket> = received_tokens
                        .into_iter()
                        .map(|token| {
                            let computed_output: Vec<f32> =
                                token.hidden_state.iter().map(|v| v * 2.0).collect();
                            TokenCombinePacket {
                                token_idx: token.token_idx,
                                source_rank: token.source_rank,
                                expert_id: token.expert_id,
                                routing_weight: token.routing_weight,
                                output_state: computed_output,
                            }
                        })
                        .collect();

                    // 3. Combine All-to-All (send back to origin)
                    let combined_tokens = comm.combine_all_to_all(outbound_combine).unwrap();
                    assert_eq!(
                        combined_tokens.len(),
                        1,
                        "Rank {:?} should receive back 1 result",
                        rank
                    );

                    // 4. Recombine states: weight 0.5 * (val * 2.0) = val
                    let final_states =
                        recombine_token_states(&combined_tokens, 1, hidden_dim).unwrap();
                    let expected_val = (rank.0 + 1) as f32;
                    assert_eq!(final_states[0], vec![expected_val; hidden_dim]);
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("thread completed successfully");
        }
    }
}
