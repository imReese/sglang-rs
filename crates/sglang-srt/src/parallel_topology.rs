// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

/// Unique global device / rank identifier within the distributed world [0, world_size).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct GlobalRank(pub usize);

impl fmt::Display for GlobalRank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GlobalRank({})", self.0)
    }
}

/// Pipeline Parallelism (PP) stage index [0, pp_size).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PpStage(pub usize);

impl fmt::Display for PpStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PpStage({})", self.0)
    }
}

/// Attention Data Parallelism (Attn DP) rank index [0, attn_dp_size).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct AttnDpRank(pub usize);

impl fmt::Display for AttnDpRank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AttnDpRank({})", self.0)
    }
}

/// Attention Tensor Parallelism (Attn TP) rank index [0, attn_tp_size).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct AttnTpRank(pub usize);

impl fmt::Display for AttnTpRank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AttnTpRank({})", self.0)
    }
}

/// Expert Parallelism (EP) rank index [0, ep_size).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct EpRank(pub usize);

impl fmt::Display for EpRank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EpRank({})", self.0)
    }
}

/// MoE Dense Linear Tensor Parallelism rank index [0, moe_dense_tp_size).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct DenseTpRank(pub usize);

impl fmt::Display for DenseTpRank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DenseTpRank({})", self.0)
    }
}

/// 4D distributed coordinate representing a device's precise multi-dimensional identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Parallel4DCoordinate {
    pub global_rank: GlobalRank,
    pub pp_stage: PpStage,
    pub attn_dp_rank: AttnDpRank,
    pub attn_tp_rank: AttnTpRank,
    pub ep_rank: EpRank,
    pub dense_tp_rank: DenseTpRank,
}

impl Parallel4DCoordinate {
    pub fn is_first_pipeline_stage(&self) -> bool {
        self.pp_stage.0 == 0
    }

    pub fn is_last_pipeline_stage(&self, pp_size: usize) -> bool {
        self.pp_stage.0 + 1 == pp_size
    }
}

/// Sizing specification for the 4D hybrid parallel groups.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Parallel4DConfig {
    pub pp_size: usize,
    pub attn_dp_size: usize,
    pub attn_tp_size: usize,
    pub ep_size: usize,
    pub moe_dense_tp_size: usize,
}

impl Parallel4DConfig {
    pub fn new(
        pp_size: usize,
        attn_dp_size: usize,
        attn_tp_size: usize,
        ep_size: usize,
        moe_dense_tp_size: usize,
    ) -> Result<Self, ParallelTopologyError> {
        if pp_size == 0 {
            return Err(ParallelTopologyError::ZeroDimension("pp_size"));
        }
        if attn_dp_size == 0 {
            return Err(ParallelTopologyError::ZeroDimension("attn_dp_size"));
        }
        if attn_tp_size == 0 {
            return Err(ParallelTopologyError::ZeroDimension("attn_tp_size"));
        }
        if ep_size == 0 {
            return Err(ParallelTopologyError::ZeroDimension("ep_size"));
        }
        if moe_dense_tp_size == 0 {
            return Err(ParallelTopologyError::ZeroDimension("moe_dense_tp_size"));
        }

        let stage_size_attn = attn_dp_size
            .checked_mul(attn_tp_size)
            .ok_or(ParallelTopologyError::Overflow("stage_size_attn"))?;
        let stage_size_moe = ep_size
            .checked_mul(moe_dense_tp_size)
            .ok_or(ParallelTopologyError::Overflow("stage_size_moe"))?;

        if stage_size_attn != stage_size_moe {
            return Err(ParallelTopologyError::StageSizingMismatch {
                stage_size_attn,
                stage_size_moe,
            });
        }

        Ok(Self {
            pp_size,
            attn_dp_size,
            attn_tp_size,
            ep_size,
            moe_dense_tp_size,
        })
    }

    pub fn stage_size(&self) -> usize {
        self.attn_dp_size * self.attn_tp_size
    }

    pub fn world_size(&self) -> usize {
        self.pp_size * self.stage_size()
    }
}

/// Concrete 4D Distributed Topology for hybrid TP x Attn DP x EP x PP orchestration.
#[derive(Clone, Debug)]
pub struct Parallel4DTopology {
    config: Parallel4DConfig,
    world_size: usize,
    ranks: Vec<Parallel4DCoordinate>,
}

impl Parallel4DTopology {
    pub fn new(config: Parallel4DConfig) -> Result<Self, ParallelTopologyError> {
        let world_size = config.world_size();
        let stage_size = config.stage_size();
        let mut ranks = Vec::with_capacity(world_size);

        for g in 0..world_size {
            let pp = g / stage_size;
            let intra_stage = g % stage_size;

            let attn_dp = intra_stage / config.attn_tp_size;
            let attn_tp = intra_stage % config.attn_tp_size;

            let ep = intra_stage / config.moe_dense_tp_size;
            let dense_tp = intra_stage % config.moe_dense_tp_size;

            ranks.push(Parallel4DCoordinate {
                global_rank: GlobalRank(g),
                pp_stage: PpStage(pp),
                attn_dp_rank: AttnDpRank(attn_dp),
                attn_tp_rank: AttnTpRank(attn_tp),
                ep_rank: EpRank(ep),
                dense_tp_rank: DenseTpRank(dense_tp),
            });
        }

        Ok(Self {
            config,
            world_size,
            ranks,
        })
    }

    pub fn config(&self) -> &Parallel4DConfig {
        &self.config
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    pub fn coordinate_of(
        &self,
        rank: GlobalRank,
    ) -> Result<Parallel4DCoordinate, ParallelTopologyError> {
        self.ranks
            .get(rank.0)
            .copied()
            .ok_or(ParallelTopologyError::RankOutOfBounds {
                rank: rank.0,
                world_size: self.world_size,
            })
    }

    /// Ranks sharing the same (pp_stage, attn_dp_rank) -> forms the Attention Tensor Parallel Group (NVLink All-Reduce).
    pub fn attn_tp_group_for(
        &self,
        rank: GlobalRank,
    ) -> Result<Vec<GlobalRank>, ParallelTopologyError> {
        let coord = self.coordinate_of(rank)?;
        let group = self
            .ranks
            .iter()
            .filter(|c| c.pp_stage == coord.pp_stage && c.attn_dp_rank == coord.attn_dp_rank)
            .map(|c| c.global_rank)
            .collect();
        Ok(group)
    }

    /// Ranks sharing the same (pp_stage, attn_tp_rank) -> forms the Attention Data Parallel Group.
    pub fn attn_dp_group_for(
        &self,
        rank: GlobalRank,
    ) -> Result<Vec<GlobalRank>, ParallelTopologyError> {
        let coord = self.coordinate_of(rank)?;
        let group = self
            .ranks
            .iter()
            .filter(|c| c.pp_stage == coord.pp_stage && c.attn_tp_rank == coord.attn_tp_rank)
            .map(|c| c.global_rank)
            .collect();
        Ok(group)
    }

    /// Ranks sharing the same (pp_stage, dense_tp_rank) -> forms the MoE All-to-All Dispatch/Combine Group.
    pub fn ep_group_for(&self, rank: GlobalRank) -> Result<Vec<GlobalRank>, ParallelTopologyError> {
        let coord = self.coordinate_of(rank)?;
        let group = self
            .ranks
            .iter()
            .filter(|c| c.pp_stage == coord.pp_stage && c.dense_tp_rank == coord.dense_tp_rank)
            .map(|c| c.global_rank)
            .collect();
        Ok(group)
    }

    /// Ranks forming the Pipeline Parallelism chain for this token slot (same intra-stage position across stages).
    pub fn pp_group_for(&self, rank: GlobalRank) -> Result<Vec<GlobalRank>, ParallelTopologyError> {
        let coord = self.coordinate_of(rank)?;
        let group = self
            .ranks
            .iter()
            .filter(|c| {
                c.attn_dp_rank == coord.attn_dp_rank && c.attn_tp_rank == coord.attn_tp_rank
            })
            .map(|c| c.global_rank)
            .collect();
        Ok(group)
    }

    /// Previous stage rank for Point-to-Point activation passing.
    pub fn pp_prev_rank(
        &self,
        rank: GlobalRank,
    ) -> Result<Option<GlobalRank>, ParallelTopologyError> {
        let coord = self.coordinate_of(rank)?;
        if coord.pp_stage.0 == 0 {
            Ok(None)
        } else {
            let prev_stage = coord.pp_stage.0 - 1;
            let intra_stage = rank.0 % self.config.stage_size();
            Ok(Some(GlobalRank(
                prev_stage * self.config.stage_size() + intra_stage,
            )))
        }
    }

    /// Next stage rank for Point-to-Point activation passing.
    pub fn pp_next_rank(
        &self,
        rank: GlobalRank,
    ) -> Result<Option<GlobalRank>, ParallelTopologyError> {
        let coord = self.coordinate_of(rank)?;
        if coord.pp_stage.0 + 1 >= self.config.pp_size {
            Ok(None)
        } else {
            let next_stage = coord.pp_stage.0 + 1;
            let intra_stage = rank.0 % self.config.stage_size();
            Ok(Some(GlobalRank(
                next_stage * self.config.stage_size() + intra_stage,
            )))
        }
    }
}

/// Errors occurring during topology creation or coordinate queries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParallelTopologyError {
    ZeroDimension(&'static str),
    Overflow(&'static str),
    StageSizingMismatch {
        stage_size_attn: usize,
        stage_size_moe: usize,
    },
    RankOutOfBounds {
        rank: usize,
        world_size: usize,
    },
}

impl fmt::Display for ParallelTopologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension(name) => write!(f, "parallel dimension '{name}' must be positive"),
            Self::Overflow(name) => {
                write!(f, "integer overflow while computing dimension '{name}'")
            }
            Self::StageSizingMismatch {
                stage_size_attn,
                stage_size_moe,
            } => write!(
                f,
                "stage size mismatch between attention ({stage_size_attn}) and MoE ({stage_size_moe})"
            ),
            Self::RankOutOfBounds { rank, world_size } => {
                write!(
                    f,
                    "rank {rank} is out of bounds for world size {world_size}"
                )
            }
        }
    }
}

impl std::error::Error for ParallelTopologyError {}

/// Abstract communicator for Tensor Parallel collective operations.
pub trait TensorParallelCommunicator: Send + Sync {
    fn all_reduce(&self, buffer: &mut [f32]) -> Result<(), String>;
    fn all_gather(&self, send_buf: &[f32], recv_buf: &mut [f32]) -> Result<(), String>;
}

/// Abstract communicator for MoE Expert Parallel All-to-All operations.
pub trait ExpertParallelCommunicator: Send + Sync {
    fn all_to_all_dispatch(
        &self,
        token_payloads: &[u8],
        token_splits_per_expert: &[usize],
    ) -> Result<Vec<u8>, String>;

    fn all_to_all_combine(
        &self,
        expert_payloads: &[u8],
        combine_splits_per_rank: &[usize],
    ) -> Result<Vec<u8>, String>;
}

/// Abstract communicator for Pipeline Parallel Point-to-Point activation streaming.
pub trait PipelineParallelCommunicator: Send + Sync {
    fn send_forward_activation(&self, tensor_payload: &[u8]) -> Result<(), String>;
    fn recv_forward_activation(&self, buffer: &mut [u8]) -> Result<(), String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_gpu_topology() {
        let config = Parallel4DConfig::new(1, 1, 1, 1, 1).expect("valid config");
        let topo = Parallel4DTopology::new(config).expect("valid topo");
        assert_eq!(topo.world_size(), 1);

        let coord = topo.coordinate_of(GlobalRank(0)).expect("coord 0");
        assert_eq!(coord.pp_stage, PpStage(0));
        assert_eq!(coord.attn_dp_rank, AttnDpRank(0));
        assert_eq!(coord.attn_tp_rank, AttnTpRank(0));
        assert_eq!(coord.ep_rank, EpRank(0));
        assert_eq!(coord.dense_tp_rank, DenseTpRank(0));

        assert!(coord.is_first_pipeline_stage());
        assert!(coord.is_last_pipeline_stage(1));
        assert_eq!(topo.pp_prev_rank(GlobalRank(0)).unwrap(), None);
        assert_eq!(topo.pp_next_rank(GlobalRank(0)).unwrap(), None);
    }

    #[test]
    fn test_stage_mismatch_rejected() {
        // attn: 2 * 2 = 4, moe: 3 * 1 = 3 -> mismatch!
        let err = Parallel4DConfig::new(1, 2, 2, 3, 1).unwrap_err();
        assert_eq!(
            err,
            ParallelTopologyError::StageSizingMismatch {
                stage_size_attn: 4,
                stage_size_moe: 3,
            }
        );
    }

    #[test]
    fn test_deepseek_v3_style_64_gpu_cluster() {
        // 64 GPUs total:
        // PP = 4 stages (16 GPUs per stage)
        // Per stage:
        //   Attention View: Attn DP = 8, Attn TP = 2 (8 * 2 = 16)
        //   MoE View:       EP = 16, Dense TP = 1   (16 * 1 = 16)
        let config = Parallel4DConfig::new(4, 8, 2, 16, 1).expect("valid 64-card config");
        assert_eq!(config.world_size(), 64);
        assert_eq!(config.stage_size(), 16);

        let topo = Parallel4DTopology::new(config).expect("topology");

        // Verify Rank 0 (PP Stage 0, Intra 0)
        let r0 = topo.coordinate_of(GlobalRank(0)).expect("r0");
        assert_eq!(r0.pp_stage, PpStage(0));
        assert_eq!(r0.attn_dp_rank, AttnDpRank(0));
        assert_eq!(r0.attn_tp_rank, AttnTpRank(0));
        assert_eq!(r0.ep_rank, EpRank(0));
        assert_eq!(r0.dense_tp_rank, DenseTpRank(0));

        // Verify Rank 17 (PP Stage 1, Intra 1: g = 1 * 16 + 1 = 17)
        let r17 = topo.coordinate_of(GlobalRank(17)).expect("r17");
        assert_eq!(r17.pp_stage, PpStage(1));
        assert_eq!(r17.attn_dp_rank, AttnDpRank(0));
        assert_eq!(r17.attn_tp_rank, AttnTpRank(1));
        assert_eq!(r17.ep_rank, EpRank(1));
        assert_eq!(r17.dense_tp_rank, DenseTpRank(0));

        // Group checks:
        // Attn TP group for Rank 0 -> ranks in Stage 0 with Attn DP = 0: Ranks 0 and 1
        let attn_tp_grp = topo.attn_tp_group_for(GlobalRank(0)).unwrap();
        assert_eq!(attn_tp_grp, vec![GlobalRank(0), GlobalRank(1)]);

        // Attn DP group for Rank 0 -> ranks in Stage 0 with Attn TP = 0: Ranks 0, 2, 4, 6, 8, 10, 12, 14
        let attn_dp_grp = topo.attn_dp_group_for(GlobalRank(0)).unwrap();
        assert_eq!(
            attn_dp_grp,
            vec![
                GlobalRank(0),
                GlobalRank(2),
                GlobalRank(4),
                GlobalRank(6),
                GlobalRank(8),
                GlobalRank(10),
                GlobalRank(12),
                GlobalRank(14)
            ]
        );

        // EP group for Rank 0 -> all 16 EP ranks in Stage 0 (Dense TP = 0)
        let ep_grp = topo.ep_group_for(GlobalRank(0)).unwrap();
        assert_eq!(ep_grp.len(), 16);
        assert_eq!(ep_grp[0], GlobalRank(0));
        assert_eq!(ep_grp[15], GlobalRank(15));

        // PP chain for Rank 0 (Intra stage 0 across PP 0..4) -> Ranks 0, 16, 32, 48
        let pp_grp = topo.pp_group_for(GlobalRank(0)).unwrap();
        assert_eq!(
            pp_grp,
            vec![
                GlobalRank(0),
                GlobalRank(16),
                GlobalRank(32),
                GlobalRank(48)
            ]
        );

        // Pipeline P2P neighbors:
        assert_eq!(topo.pp_prev_rank(GlobalRank(0)).unwrap(), None);
        assert_eq!(
            topo.pp_next_rank(GlobalRank(0)).unwrap(),
            Some(GlobalRank(16))
        );

        assert_eq!(
            topo.pp_prev_rank(GlobalRank(16)).unwrap(),
            Some(GlobalRank(0))
        );
        assert_eq!(
            topo.pp_next_rank(GlobalRank(16)).unwrap(),
            Some(GlobalRank(32))
        );

        assert_eq!(
            topo.pp_prev_rank(GlobalRank(48)).unwrap(),
            Some(GlobalRank(32))
        );
        assert_eq!(topo.pp_next_rank(GlobalRank(48)).unwrap(), None);
    }
}
