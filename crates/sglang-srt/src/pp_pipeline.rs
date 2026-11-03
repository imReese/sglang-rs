// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::ops::Range;
use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver, Sender};

use crate::parallel_topology::PpStage;

/// Error conditions arising during Pipeline Parallelism (PP) communication or execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PipelineCommError {
    ZeroDimension(&'static str),
    InvalidStage { stage: usize, pp_size: usize },
    TotalLayersTooSmall { total_layers: usize, pp_size: usize },
    ChannelClosed,
    StageMismatch { expected: usize, actual: usize },
    FirstStageHasNoPredecessor,
    LastStageHasNoSuccessor,
    ShapeMismatch { expected: usize, actual: usize },
}

impl fmt::Display for PipelineCommError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension(name) => write!(f, "PP dimension '{name}' must be non-zero"),
            Self::InvalidStage { stage, pp_size } => {
                write!(f, "PP stage {stage} out of range for pp_size {pp_size}")
            }
            Self::TotalLayersTooSmall {
                total_layers,
                pp_size,
            } => write!(
                f,
                "total layers ({total_layers}) must be >= pp_size ({pp_size})"
            ),
            Self::ChannelClosed => write!(f, "pipeline communication channel closed unexpectedly"),
            Self::StageMismatch { expected, actual } => write!(
                f,
                "pipeline stage mismatch: expected stage {expected}, got stage {actual}"
            ),
            Self::FirstStageHasNoPredecessor => {
                write!(
                    f,
                    "first pipeline stage (0) does not receive from a predecessor"
                )
            }
            Self::LastStageHasNoSuccessor => {
                write!(f, "last pipeline stage does not send to a successor")
            }
            Self::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "activation shape mismatch: expected {expected}, actual {actual}"
                )
            }
        }
    }
}

impl std::error::Error for PipelineCommError {}

/// Configuration specifying layer partitioning across pipeline stages.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PipelineStageConfig {
    stage: PpStage,
    pp_size: usize,
    total_layers: usize,
}

impl PipelineStageConfig {
    pub fn new(
        stage: PpStage,
        pp_size: usize,
        total_layers: usize,
    ) -> Result<Self, PipelineCommError> {
        if pp_size == 0 {
            return Err(PipelineCommError::ZeroDimension("pp_size"));
        }
        if total_layers == 0 {
            return Err(PipelineCommError::ZeroDimension("total_layers"));
        }
        if total_layers < pp_size {
            return Err(PipelineCommError::TotalLayersTooSmall {
                total_layers,
                pp_size,
            });
        }
        if stage.0 >= pp_size {
            return Err(PipelineCommError::InvalidStage {
                stage: stage.0,
                pp_size,
            });
        }
        Ok(Self {
            stage,
            pp_size,
            total_layers,
        })
    }

    pub fn stage(&self) -> PpStage {
        self.stage
    }

    pub fn pp_size(&self) -> usize {
        self.pp_size
    }

    pub fn total_layers(&self) -> usize {
        self.total_layers
    }

    pub fn is_first_stage(&self) -> bool {
        self.stage.0 == 0
    }

    pub fn is_last_stage(&self) -> bool {
        self.stage.0 + 1 == self.pp_size
    }

    /// Computes the layer range $[start, end)$ assigned to a given stage.
    /// Distributes remaining layers across initial stages evenly.
    pub fn layer_range_for_stage(
        stage: PpStage,
        pp_size: usize,
        total_layers: usize,
    ) -> Result<Range<usize>, PipelineCommError> {
        if pp_size == 0 {
            return Err(PipelineCommError::ZeroDimension("pp_size"));
        }
        if stage.0 >= pp_size {
            return Err(PipelineCommError::InvalidStage {
                stage: stage.0,
                pp_size,
            });
        }
        let base_layers = total_layers / pp_size;
        let remainder = total_layers % pp_size;

        let start = if stage.0 < remainder {
            stage.0 * (base_layers + 1)
        } else {
            remainder * (base_layers + 1) + (stage.0 - remainder) * base_layers
        };

        let count = if stage.0 < remainder {
            base_layers + 1
        } else {
            base_layers
        };

        Ok(start..(start + count))
    }

    /// Returns the layer range assigned to this stage.
    pub fn assigned_layers(&self) -> Range<usize> {
        Self::layer_range_for_stage(self.stage, self.pp_size, self.total_layers).unwrap_or(0..0)
    }

    /// Returns the number of transformer layers assigned to this stage.
    pub fn num_assigned_layers(&self) -> usize {
        let range = self.assigned_layers();
        range.end - range.start
    }
}

/// Activation tensor packet passed between consecutive pipeline stages.
#[derive(Clone, Debug, PartialEq)]
pub struct PipelineActivationPacket {
    pub batch_id: u64,
    pub stage_from: PpStage,
    pub stage_to: PpStage,
    pub request_ids: Vec<String>,
    pub sequence_lengths: Vec<usize>,
    pub hidden_dim: usize,
    pub hidden_states: Vec<f32>,
}

/// Abstract communicator for pipeline stage handoff.
pub trait PipelineParallelCommunicator: Send + Sync {
    fn stage(&self) -> PpStage;
    fn pp_size(&self) -> usize;

    fn is_first_stage(&self) -> bool {
        self.stage().0 == 0
    }

    fn is_last_stage(&self) -> bool {
        self.stage().0 + 1 == self.pp_size()
    }

    /// Sends activation tensor forward to the next stage ($stage + 1$).
    fn send_forward(&self, packet: PipelineActivationPacket) -> Result<(), PipelineCommError>;

    /// Receives activation tensor forward from the preceding stage ($stage - 1$).
    fn recv_forward(&self) -> Result<PipelineActivationPacket, PipelineCommError>;
}

/// Intra-process channel-based pipeline communicator connecting adjacent stages.
pub struct ChannelPipelineCommunicator {
    stage: PpStage,
    pp_size: usize,
    forward_sender: Option<Sender<PipelineActivationPacket>>,
    forward_receiver: Option<Mutex<Receiver<PipelineActivationPacket>>>,
}

impl ChannelPipelineCommunicator {
    /// Creates a complete interconnected pipeline of $K$ stages.
    pub fn create_pipeline(pp_size: usize) -> Result<Vec<Self>, PipelineCommError> {
        if pp_size == 0 {
            return Err(PipelineCommError::ZeroDimension("pp_size"));
        }

        if pp_size == 1 {
            return Ok(vec![Self {
                stage: PpStage(0),
                pp_size: 1,
                forward_sender: None,
                forward_receiver: None,
            }]);
        }

        // Create pp_size - 1 channels: Channel i connects Stage i to Stage i + 1
        let mut senders = Vec::with_capacity(pp_size - 1);
        let mut receivers = Vec::with_capacity(pp_size - 1);
        for _ in 0..(pp_size - 1) {
            let (tx, rx) = mpsc::channel();
            senders.push(tx);
            receivers.push(rx);
        }

        let mut communicators = Vec::with_capacity(pp_size);
        for s in 0..pp_size {
            let forward_sender = senders.get(s).cloned();

            let forward_receiver = if s > 0 {
                // Takes ownership from receivers[s - 1]
                Some(Mutex::new(receivers.remove(0)))
            } else {
                None
            };

            communicators.push(Self {
                stage: PpStage(s),
                pp_size,
                forward_sender,
                forward_receiver,
            });
        }

        Ok(communicators)
    }
}

impl PipelineParallelCommunicator for ChannelPipelineCommunicator {
    fn stage(&self) -> PpStage {
        self.stage
    }

    fn pp_size(&self) -> usize {
        self.pp_size
    }

    fn send_forward(&self, packet: PipelineActivationPacket) -> Result<(), PipelineCommError> {
        let sender = self
            .forward_sender
            .as_ref()
            .ok_or(PipelineCommError::LastStageHasNoSuccessor)?;

        if packet.stage_from != self.stage {
            return Err(PipelineCommError::StageMismatch {
                expected: self.stage.0,
                actual: packet.stage_from.0,
            });
        }
        if packet.stage_to.0 != self.stage.0 + 1 {
            return Err(PipelineCommError::StageMismatch {
                expected: self.stage.0 + 1,
                actual: packet.stage_to.0,
            });
        }

        sender
            .send(packet)
            .map_err(|_| PipelineCommError::ChannelClosed)
    }

    fn recv_forward(&self) -> Result<PipelineActivationPacket, PipelineCommError> {
        let rx_mutex = self
            .forward_receiver
            .as_ref()
            .ok_or(PipelineCommError::FirstStageHasNoPredecessor)?;

        let rx = rx_mutex
            .lock()
            .map_err(|_| PipelineCommError::ChannelClosed)?;

        let packet = rx.recv().map_err(|_| PipelineCommError::ChannelClosed)?;

        if packet.stage_to != self.stage {
            return Err(PipelineCommError::StageMismatch {
                expected: self.stage.0,
                actual: packet.stage_to.0,
            });
        }

        Ok(packet)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_pipeline_layer_partitioning() {
        // DeepSeek-V3 style 61 transformer layers across 4 PP stages:
        // 61 = 16 + 15 + 15 + 15
        let pp_size = 4;
        let total_layers = 61;

        let r0 =
            PipelineStageConfig::layer_range_for_stage(PpStage(0), pp_size, total_layers).unwrap();
        assert_eq!(r0, 0..16); // 16 layers

        let r1 =
            PipelineStageConfig::layer_range_for_stage(PpStage(1), pp_size, total_layers).unwrap();
        assert_eq!(r1, 16..31); // 15 layers

        let r2 =
            PipelineStageConfig::layer_range_for_stage(PpStage(2), pp_size, total_layers).unwrap();
        assert_eq!(r2, 31..46); // 15 layers

        let r3 =
            PipelineStageConfig::layer_range_for_stage(PpStage(3), pp_size, total_layers).unwrap();
        assert_eq!(r3, 46..61); // 15 layers

        // Even 32 layers across 4 stages: 8 each
        let cfg = PipelineStageConfig::new(PpStage(2), 4, 32).unwrap();
        assert_eq!(cfg.assigned_layers(), 16..24);
        assert_eq!(cfg.num_assigned_layers(), 8);
        assert!(!cfg.is_first_stage());
        assert!(!cfg.is_last_stage());
    }

    #[test]
    fn test_pipeline_multi_stage_forward() {
        let pp_size = 4;
        let communicators = ChannelPipelineCommunicator::create_pipeline(pp_size).unwrap();
        assert_eq!(communicators.len(), pp_size);

        let hidden_dim = 4;

        let handles: Vec<_> = communicators
            .into_iter()
            .map(|comm| {
                thread::spawn(move || {
                    let stage = comm.stage();
                    if comm.is_first_stage() {
                        // Stage 0: creates initial activations and sends forward to Stage 1
                        let initial_hidden = vec![1.0; hidden_dim];
                        comm.send_forward(PipelineActivationPacket {
                            batch_id: 100,
                            stage_from: stage,
                            stage_to: PpStage(1),
                            request_ids: vec!["req_1".to_string()],
                            sequence_lengths: vec![1],
                            hidden_dim,
                            hidden_states: initial_hidden,
                        })
                        .unwrap();
                        1.0f32
                    } else if comm.is_last_stage() {
                        // Last stage: receives from stage - 1, computes final value
                        let packet = comm.recv_forward().unwrap();
                        assert_eq!(packet.batch_id, 100);
                        assert_eq!(packet.stage_from, PpStage(2));
                        // Cumulative: 1.0 + 1 (stage 1) + 2 (stage 2) = 4.0
                        packet.hidden_states[0] + (stage.0 as f32) // 4.0 + 3.0 = 7.0
                    } else {
                        // Intermediate stages: receive from stage - 1, add stage index, send to stage + 1
                        let mut packet = comm.recv_forward().unwrap();
                        assert_eq!(packet.stage_from.0 + 1, stage.0);
                        for val in packet.hidden_states.iter_mut() {
                            *val += stage.0 as f32;
                        }
                        packet.stage_from = stage;
                        packet.stage_to = PpStage(stage.0 + 1);
                        comm.send_forward(packet).unwrap();
                        0.0f32
                    }
                })
            })
            .collect();

        let mut results = Vec::new();
        for handle in handles {
            results.push(handle.join().unwrap());
        }

        // Result from last stage (index 3)
        assert_eq!(results[3], 7.0);
    }
}
