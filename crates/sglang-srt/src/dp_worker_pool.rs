// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};

use crate::scheduler::ScheduleBatch;
use crate::worker::{
    BatchGeneratedTokens, FallibleModelWorker, WorkerExecutionError, WorkerWeightUpdateRequest,
};

/// Commands sent from the Unified Scheduler to a specific DP Worker rank.
#[derive(Debug)]
pub enum DpWorkerCommand {
    ExecuteBatch { sequence: u64, batch: ScheduleBatch },
    UpdateWeights { request: WorkerWeightUpdateRequest },
    Shutdown,
}

/// Completion events emitted by a DP Worker rank back to the Unified Scheduler.
#[derive(Debug)]
pub enum DpWorkerEvent {
    BatchCompleted {
        sequence: u64,
        rank: usize,
        result: Result<BatchGeneratedTokens, WorkerExecutionError>,
    },
    WeightsUpdated {
        rank: usize,
        result: Result<(), WorkerExecutionError>,
    },
    FatalError {
        rank: usize,
        error: String,
    },
}

/// Handle to a single running worker thread.
struct DpRankHandle {
    rank_id: usize,
    commands: Sender<DpWorkerCommand>,
    is_idle: bool,
}

/// Thread-safe concurrent worker pool for multi-DP inference execution.
///
/// Driven by a single centralized scheduler without inter-process IPC or multiple network ports.
pub struct DpWorkerPool {
    ranks: Vec<DpRankHandle>,
    events_rx: Receiver<DpWorkerEvent>,
    joins: Vec<Option<JoinHandle<()>>>,
    next_sequence: u64,
    is_shutdown: bool,
}

impl fmt::Debug for DpWorkerPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DpWorkerPool")
            .field("dp_size", &self.ranks.len())
            .field("is_shutdown", &self.is_shutdown)
            .finish()
    }
}

impl DpWorkerPool {
    /// Launches a pool of worker threads, each running an independent `W: FallibleModelWorker`.
    pub fn launch<W, F, E>(dp_size: usize, initializer: F) -> Result<Self, DpWorkerPoolError>
    where
        W: FallibleModelWorker + Send + 'static,
        F: Fn(usize) -> Result<W, E> + Send + Sync + 'static,
        E: fmt::Display + Send + 'static,
    {
        if dp_size == 0 {
            return Err(DpWorkerPoolError::ZeroSize);
        }

        let (events_tx, events_rx) = mpsc::channel();
        let mut ranks = Vec::with_capacity(dp_size);
        let mut joins = Vec::with_capacity(dp_size);

        for rank_id in 0..dp_size {
            let (commands_tx, commands_rx) = mpsc::channel();
            let rank_events_tx = events_tx.clone();
            let worker = match initializer(rank_id) {
                Ok(w) => w,
                Err(err) => {
                    return Err(DpWorkerPoolError::Initialization {
                        rank: rank_id,
                        reason: err.to_string(),
                    });
                }
            };

            let join = thread::Builder::new()
                .name(format!("sglang-dp-worker-{}", rank_id))
                .spawn(move || {
                    Self::worker_thread_loop(rank_id, worker, commands_rx, rank_events_tx);
                })
                .map_err(|err| DpWorkerPoolError::Spawn {
                    rank: rank_id,
                    reason: err.to_string(),
                })?;

            ranks.push(DpRankHandle {
                rank_id,
                commands: commands_tx,
                is_idle: true,
            });
            joins.push(Some(join));
        }

        Ok(Self {
            ranks,
            events_rx,
            joins,
            next_sequence: 1,
            is_shutdown: false,
        })
    }

    pub fn dp_size(&self) -> usize {
        self.ranks.len()
    }

    pub fn is_idle(&self, rank: usize) -> bool {
        self.ranks.get(rank).is_some_and(|h| h.is_idle)
    }

    /// Returns a list of all currently idle ranks available to accept a new batch.
    pub fn idle_ranks(&self) -> Vec<usize> {
        self.ranks
            .iter()
            .filter(|h| h.is_idle)
            .map(|h| h.rank_id)
            .collect()
    }

    /// Submits a batch for non-blocking asynchronous execution on the specified rank.
    pub fn submit_batch(
        &mut self,
        rank: usize,
        batch: ScheduleBatch,
    ) -> Result<u64, DpWorkerPoolError> {
        if self.is_shutdown {
            return Err(DpWorkerPoolError::AlreadyShutdown);
        }

        let dp_size = self.ranks.len();
        let handle = self
            .ranks
            .get_mut(rank)
            .ok_or(DpWorkerPoolError::InvalidRank { rank, dp_size })?;

        if !handle.is_idle {
            return Err(DpWorkerPoolError::RankBusy { rank });
        }

        let seq = self.next_sequence;
        self.next_sequence += 1;

        handle
            .commands
            .send(DpWorkerCommand::ExecuteBatch {
                sequence: seq,
                batch,
            })
            .map_err(|_| DpWorkerPoolError::ChannelDisconnected { rank })?;

        handle.is_idle = false;
        Ok(seq)
    }

    /// Broadcasts a weight reload request to all worker ranks.
    pub fn broadcast_weight_update(
        &mut self,
        request: WorkerWeightUpdateRequest,
    ) -> Result<(), DpWorkerPoolError> {
        if self.is_shutdown {
            return Err(DpWorkerPoolError::AlreadyShutdown);
        }

        for handle in self.ranks.iter_mut() {
            handle
                .commands
                .send(DpWorkerCommand::UpdateWeights {
                    request: request.clone(),
                })
                .map_err(|_| DpWorkerPoolError::ChannelDisconnected {
                    rank: handle.rank_id,
                })?;
            handle.is_idle = false;
        }

        Ok(())
    }

    /// Non-blocking drain of all pending worker completion events.
    ///
    /// Updates worker idle states and returns all collected completion events.
    pub fn poll_events(&mut self) -> Vec<DpWorkerEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.events_rx.try_recv() {
            match &event {
                DpWorkerEvent::BatchCompleted { rank, .. }
                | DpWorkerEvent::WeightsUpdated { rank, .. }
                | DpWorkerEvent::FatalError { rank, .. } => {
                    if let Some(handle) = self.ranks.get_mut(*rank) {
                        handle.is_idle = true;
                    }
                }
            }
            events.push(event);
        }
        events
    }

    /// Blocks until at least one event arrives, or returns immediately if events are waiting.
    pub fn wait_for_event(&mut self) -> Result<DpWorkerEvent, DpWorkerPoolError> {
        let event = self
            .events_rx
            .recv()
            .map_err(|_| DpWorkerPoolError::EventReceiverClosed)?;

        match &event {
            DpWorkerEvent::BatchCompleted { rank, .. }
            | DpWorkerEvent::WeightsUpdated { rank, .. }
            | DpWorkerEvent::FatalError { rank, .. } => {
                if let Some(handle) = self.ranks.get_mut(*rank) {
                    handle.is_idle = true;
                }
            }
        }
        Ok(event)
    }

    pub fn shutdown(&mut self) -> Result<(), DpWorkerPoolError> {
        if self.is_shutdown {
            return Ok(());
        }
        self.is_shutdown = true;

        for handle in &self.ranks {
            let _ = handle.commands.send(DpWorkerCommand::Shutdown);
        }

        for join in self.joins.iter_mut() {
            if let Some(handle) = join.take() {
                let _ = handle.join();
            }
        }

        Ok(())
    }

    fn worker_thread_loop<W: FallibleModelWorker>(
        rank: usize,
        mut worker: W,
        commands: Receiver<DpWorkerCommand>,
        events: Sender<DpWorkerEvent>,
    ) {
        while let Ok(cmd) = commands.recv() {
            match cmd {
                DpWorkerCommand::ExecuteBatch { sequence, batch } => {
                    let result = worker.try_generate_batch(&batch);
                    let _ = events.send(DpWorkerEvent::BatchCompleted {
                        sequence,
                        rank,
                        result,
                    });
                }
                DpWorkerCommand::UpdateWeights { request } => {
                    let result = worker.update_weights_from_disk(&request);
                    let _ = events.send(DpWorkerEvent::WeightsUpdated { rank, result });
                }
                DpWorkerCommand::Shutdown => break,
            }
        }
    }
}

impl Drop for DpWorkerPool {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

/// Errors occurring during worker pool operations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DpWorkerPoolError {
    ZeroSize,
    AlreadyShutdown,
    InvalidRank { rank: usize, dp_size: usize },
    RankBusy { rank: usize },
    ChannelDisconnected { rank: usize },
    EventReceiverClosed,
    Initialization { rank: usize, reason: String },
    Spawn { rank: usize, reason: String },
}

impl fmt::Display for DpWorkerPoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSize => f.write_str("DP worker pool size must be positive"),
            Self::AlreadyShutdown => f.write_str("DP worker pool is already shut down"),
            Self::InvalidRank { rank, dp_size } => {
                write!(f, "invalid rank {rank} for pool of size {dp_size}")
            }
            Self::RankBusy { rank } => write!(f, "rank {rank} is currently busy executing a batch"),
            Self::ChannelDisconnected { rank } => {
                write!(f, "communication channel to rank {rank} is disconnected")
            }
            Self::EventReceiverClosed => f.write_str("worker event receiver channel closed"),
            Self::Initialization { rank, reason } => {
                write!(f, "failed to initialize worker for rank {rank}: {reason}")
            }
            Self::Spawn { rank, reason } => {
                write!(f, "failed to spawn thread for rank {rank}: {reason}")
            }
        }
    }
}

impl std::error::Error for DpWorkerPoolError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::ScheduledRequest;
    use crate::types::{RequestId, SamplingParams};
    use crate::worker::GeneratedToken;

    struct EchoMockWorker {
        rank_id: usize,
    }

    impl FallibleModelWorker for EchoMockWorker {
        fn try_generate_batch(
            &mut self,
            batch: &ScheduleBatch,
        ) -> Result<BatchGeneratedTokens, WorkerExecutionError> {
            let tokens = batch
                .requests()
                .iter()
                .map(|_| GeneratedToken::finished(vec![self.rank_id as u32 + 100]))
                .collect();
            BatchGeneratedTokens::from_batch(batch, tokens).map_err(WorkerExecutionError::from)
        }

        fn update_weights_from_disk(
            &mut self,
            _request: &WorkerWeightUpdateRequest,
        ) -> Result<(), WorkerExecutionError> {
            Ok(())
        }
    }

    #[test]
    fn test_dp_worker_pool_launch_and_execute_across_ranks() {
        let mut pool =
            DpWorkerPool::launch(4, |rank_id| Ok::<_, String>(EchoMockWorker { rank_id }))
                .expect("launch pool of 4 ranks");

        assert_eq!(pool.dp_size(), 4);
        assert_eq!(pool.idle_ranks(), vec![0, 1, 2, 3]);

        // Submit batch to Rank 1
        let req1 = ScheduledRequest::new(RequestId::from("r1"), vec![1, 2], SamplingParams::new(1));
        let batch1 = ScheduleBatch::prefill(vec![req1]);
        let seq1 = pool.submit_batch(1, batch1).expect("submit rank 1");
        assert!(!pool.is_idle(1));
        assert_eq!(pool.idle_ranks(), vec![0, 2, 3]);

        // Submit batch to Rank 3
        let req3 = ScheduledRequest::new(RequestId::from("r3"), vec![3, 4], SamplingParams::new(1));
        let batch3 = ScheduleBatch::prefill(vec![req3]);
        let seq3 = pool.submit_batch(3, batch3).expect("submit rank 3");
        assert!(!pool.is_idle(3));
        assert_eq!(pool.idle_ranks(), vec![0, 2]);

        // Wait for results
        let mut completed = 0;
        while completed < 2 {
            let event = pool.wait_for_event().expect("event");
            match event {
                DpWorkerEvent::BatchCompleted {
                    sequence,
                    rank,
                    result,
                } => {
                    let out = result.expect("batch success");
                    if sequence == seq1 {
                        assert_eq!(rank, 1);
                        assert_eq!(out.tokens()[0].token_ids(), &[101]);
                    } else if sequence == seq3 {
                        assert_eq!(rank, 3);
                        assert_eq!(out.tokens()[0].token_ids(), &[103]);
                    } else {
                        panic!("unexpected sequence: {sequence}");
                    }
                    completed += 1;
                }
                other => panic!("unexpected event: {other:?}"),
            }
        }

        assert!(pool.is_idle(1));
        assert!(pool.is_idle(3));
        assert_eq!(pool.idle_ranks(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_dp_worker_pool_rejects_busy_rank() {
        let mut pool =
            DpWorkerPool::launch(2, |rank_id| Ok::<_, String>(EchoMockWorker { rank_id }))
                .expect("launch pool");

        let req = ScheduledRequest::new(RequestId::from("busy"), vec![1], SamplingParams::new(1));
        let batch = ScheduleBatch::prefill(vec![req.clone()]);
        pool.submit_batch(0, batch.clone()).expect("first submit");

        // Rank 0 is busy, second submit must fail with RankBusy
        let err = pool.submit_batch(0, batch).unwrap_err();
        assert_eq!(err, DpWorkerPoolError::RankBusy { rank: 0 });
    }
}
