use std::any::Any;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorParallelRank {
    global_rank: usize,
    local_rank: usize,
    world_size: usize,
    device_ordinal: usize,
}

impl TensorParallelRank {
    pub fn global_rank(self) -> usize {
        self.global_rank
    }

    pub fn local_rank(self) -> usize {
        self.local_rank
    }

    pub fn world_size(self) -> usize {
        self.world_size
    }

    pub fn device_ordinal(self) -> usize {
        self.device_ordinal
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorParallelTopology {
    world_size: usize,
    node_count: usize,
    node_rank: usize,
    local_ranks: Vec<TensorParallelRank>,
}

impl TensorParallelTopology {
    pub fn new(
        world_size: usize,
        node_count: usize,
        node_rank: usize,
        base_device_ordinal: usize,
        device_ordinal_step: usize,
    ) -> Result<Self, ParallelTopologyError> {
        if world_size == 0 {
            return Err(ParallelTopologyError::ZeroWorldSize);
        }
        if node_count == 0 {
            return Err(ParallelTopologyError::ZeroNodeCount);
        }
        if node_rank >= node_count {
            return Err(ParallelTopologyError::InvalidNodeRank {
                node_rank,
                node_count,
            });
        }
        if !world_size.is_multiple_of(node_count) {
            return Err(ParallelTopologyError::WorldSizeNotDivisible {
                world_size,
                node_count,
            });
        }
        if device_ordinal_step == 0 {
            return Err(ParallelTopologyError::ZeroDeviceOrdinalStep);
        }

        let local_world_size = world_size / node_count;
        let first_global_rank = node_rank.checked_mul(local_world_size).ok_or(
            ParallelTopologyError::GlobalRankOverflow {
                node_rank,
                local_world_size,
            },
        )?;
        let local_ranks = (0..local_world_size)
            .map(|local_rank| {
                let global_rank = first_global_rank.checked_add(local_rank).ok_or(
                    ParallelTopologyError::GlobalRankOverflow {
                        node_rank,
                        local_world_size,
                    },
                )?;
                let device_ordinal = local_rank
                    .checked_mul(device_ordinal_step)
                    .and_then(|offset| base_device_ordinal.checked_add(offset))
                    .ok_or(ParallelTopologyError::DeviceOrdinalOverflow {
                        base_device_ordinal,
                        device_ordinal_step,
                        local_rank,
                    })?;
                Ok(TensorParallelRank {
                    global_rank,
                    local_rank,
                    world_size,
                    device_ordinal,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            world_size,
            node_count,
            node_rank,
            local_ranks,
        })
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn node_rank(&self) -> usize {
        self.node_rank
    }

    pub fn local_ranks(&self) -> &[TensorParallelRank] {
        &self.local_ranks
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParallelTopologyError {
    ZeroWorldSize,
    ZeroNodeCount,
    InvalidNodeRank {
        node_rank: usize,
        node_count: usize,
    },
    WorldSizeNotDivisible {
        world_size: usize,
        node_count: usize,
    },
    ZeroDeviceOrdinalStep,
    GlobalRankOverflow {
        node_rank: usize,
        local_world_size: usize,
    },
    DeviceOrdinalOverflow {
        base_device_ordinal: usize,
        device_ordinal_step: usize,
        local_rank: usize,
    },
}

impl fmt::Display for ParallelTopologyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroWorldSize => {
                formatter.write_str("tensor parallel world size must be positive")
            }
            Self::ZeroNodeCount => {
                formatter.write_str("tensor parallel node count must be positive")
            }
            Self::InvalidNodeRank {
                node_rank,
                node_count,
            } => write!(
                formatter,
                "node rank {node_rank} must be smaller than node count {node_count}"
            ),
            Self::WorldSizeNotDivisible {
                world_size,
                node_count,
            } => write!(
                formatter,
                "tensor parallel world size {world_size} must be divisible by node count {node_count}"
            ),
            Self::ZeroDeviceOrdinalStep => {
                formatter.write_str("device ordinal step must be positive")
            }
            Self::GlobalRankOverflow {
                node_rank,
                local_world_size,
            } => write!(
                formatter,
                "global tensor parallel rank overflowed for node rank {node_rank} and local world size {local_world_size}"
            ),
            Self::DeviceOrdinalOverflow {
                base_device_ordinal,
                device_ordinal_step,
                local_rank,
            } => write!(
                formatter,
                "device ordinal overflowed for base {base_device_ordinal}, step {device_ordinal_step}, and local rank {local_rank}"
            ),
        }
    }
}

impl std::error::Error for ParallelTopologyError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorPartition {
    partition_count: usize,
    partition_index: usize,
}

impl TensorPartition {
    pub fn new(
        partition_count: usize,
        partition_index: usize,
    ) -> Result<Self, TensorPartitionError> {
        if partition_count == 0 {
            return Err(TensorPartitionError::ZeroPartitionCount);
        }
        if partition_index >= partition_count {
            return Err(TensorPartitionError::InvalidPartitionIndex {
                partition_index,
                partition_count,
            });
        }
        Ok(Self {
            partition_count,
            partition_index,
        })
    }

    pub fn for_rank(rank: TensorParallelRank) -> Self {
        Self {
            partition_count: rank.world_size,
            partition_index: rank.global_rank,
        }
    }

    pub fn for_replicated_shards(
        rank: TensorParallelRank,
        shard_count: usize,
    ) -> Result<Self, TensorPartitionError> {
        if shard_count == 0 {
            return Err(TensorPartitionError::ZeroPartitionCount);
        }
        if shard_count > rank.world_size {
            return Err(TensorPartitionError::ShardCountExceedsWorldSize {
                shard_count,
                world_size: rank.world_size,
            });
        }
        if !rank.world_size.is_multiple_of(shard_count) {
            return Err(TensorPartitionError::WorldSizeNotDivisibleByShardCount {
                world_size: rank.world_size,
                shard_count,
            });
        }
        let replicas_per_shard = rank.world_size / shard_count;
        Self::new(shard_count, rank.global_rank / replicas_per_shard)
    }

    pub fn partition_count(self) -> usize {
        self.partition_count
    }

    pub fn partition_index(self) -> usize {
        self.partition_index
    }

    pub fn range(self, extent: usize) -> Result<std::ops::Range<usize>, TensorPartitionError> {
        if !extent.is_multiple_of(self.partition_count) {
            return Err(TensorPartitionError::ExtentNotDivisible {
                extent,
                partition_count: self.partition_count,
            });
        }
        let partition_len = extent / self.partition_count;
        let start = self.partition_index.checked_mul(partition_len).ok_or(
            TensorPartitionError::PartitionRangeOverflow {
                extent,
                partition_count: self.partition_count,
                partition_index: self.partition_index,
            },
        )?;
        let end = start.checked_add(partition_len).ok_or(
            TensorPartitionError::PartitionRangeOverflow {
                extent,
                partition_count: self.partition_count,
                partition_index: self.partition_index,
            },
        )?;
        Ok(start..end)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorPartitionError {
    ZeroPartitionCount,
    InvalidPartitionIndex {
        partition_index: usize,
        partition_count: usize,
    },
    ShardCountExceedsWorldSize {
        shard_count: usize,
        world_size: usize,
    },
    WorldSizeNotDivisibleByShardCount {
        world_size: usize,
        shard_count: usize,
    },
    ExtentNotDivisible {
        extent: usize,
        partition_count: usize,
    },
    PartitionRangeOverflow {
        extent: usize,
        partition_count: usize,
        partition_index: usize,
    },
}

impl fmt::Display for TensorPartitionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroPartitionCount => formatter.write_str("partition count must be positive"),
            Self::InvalidPartitionIndex {
                partition_index,
                partition_count,
            } => write!(
                formatter,
                "partition index {partition_index} must be smaller than partition count {partition_count}"
            ),
            Self::ShardCountExceedsWorldSize {
                shard_count,
                world_size,
            } => write!(
                formatter,
                "replicated shard count {shard_count} exceeds tensor parallel world size {world_size}"
            ),
            Self::WorldSizeNotDivisibleByShardCount {
                world_size,
                shard_count,
            } => write!(
                formatter,
                "tensor parallel world size {world_size} must be divisible by replicated shard count {shard_count}"
            ),
            Self::ExtentNotDivisible {
                extent,
                partition_count,
            } => write!(
                formatter,
                "tensor extent {extent} must be divisible by partition count {partition_count}"
            ),
            Self::PartitionRangeOverflow {
                extent,
                partition_count,
                partition_index,
            } => write!(
                formatter,
                "partition range overflowed for extent {extent}, partition count {partition_count}, and partition index {partition_index}"
            ),
        }
    }
}

impl std::error::Error for TensorPartitionError {}

pub trait RankWorker: Send + 'static {
    type Command: Clone + Send + 'static;
    type Output: Send + 'static;
    type Error: fmt::Display + Send + 'static;

    fn execute(&mut self, command: Self::Command) -> Result<Self::Output, Self::Error>;
    fn shutdown(&mut self) -> Result<(), Self::Error>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RankOutput<O> {
    rank: TensorParallelRank,
    output: O,
}

impl<O> RankOutput<O> {
    pub fn rank(&self) -> TensorParallelRank {
        self.rank
    }

    pub fn output(&self) -> &O {
        &self.output
    }

    pub fn into_output(self) -> O {
        self.output
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RankFailure {
    pub rank: TensorParallelRank,
    pub message: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WorkerGroupError {
    ThreadSpawn {
        rank: TensorParallelRank,
        message: String,
    },
    RankFailures {
        stage: &'static str,
        failures: Vec<RankFailure>,
    },
    EventChannelClosed {
        stage: &'static str,
    },
    UnexpectedEvent {
        stage: &'static str,
        detail: String,
    },
    AlreadyShutdown,
}

impl fmt::Display for WorkerGroupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ThreadSpawn { rank, message } => write!(
                formatter,
                "failed to spawn tensor parallel rank {} on device {}: {message}",
                rank.global_rank(),
                rank.device_ordinal()
            ),
            Self::RankFailures { stage, failures } => {
                write!(formatter, "worker group {stage} failed")?;
                for failure in failures {
                    write!(
                        formatter,
                        "; rank {} on device {}: {}",
                        failure.rank.global_rank(),
                        failure.rank.device_ordinal(),
                        failure.message
                    )?;
                }
                Ok(())
            }
            Self::EventChannelClosed { stage } => {
                write!(
                    formatter,
                    "worker group event channel closed during {stage}"
                )
            }
            Self::UnexpectedEvent { stage, detail } => {
                write!(
                    formatter,
                    "worker group received an unexpected event during {stage}: {detail}"
                )
            }
            Self::AlreadyShutdown => formatter.write_str("worker group is already shut down"),
        }
    }
}

impl std::error::Error for WorkerGroupError {}

pub struct WorkerGroup<W>
where
    W: RankWorker,
{
    ranks: Vec<RankHandle<W::Command>>,
    events: Receiver<RankEvent<W::Output>>,
    joins: Vec<Option<JoinHandle<()>>>,
    next_sequence: u64,
    shutdown: bool,
}

impl<W> fmt::Debug for WorkerGroup<W>
where
    W: RankWorker,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WorkerGroup")
            .field(
                "ranks",
                &self
                    .ranks
                    .iter()
                    .map(|handle| handle.rank)
                    .collect::<Vec<_>>(),
            )
            .field("shutdown", &self.shutdown)
            .finish_non_exhaustive()
    }
}

impl<W> WorkerGroup<W>
where
    W: RankWorker,
{
    pub fn launch<F, E>(
        topology: &TensorParallelTopology,
        initializer: F,
    ) -> Result<Self, WorkerGroupError>
    where
        F: Fn(TensorParallelRank) -> Result<W, E> + Send + Sync + 'static,
        E: fmt::Display + Send + 'static,
    {
        let initializer = Arc::new(initializer);
        let (event_sender, events) = mpsc::channel();
        let mut ranks: Vec<RankHandle<W::Command>> =
            Vec::with_capacity(topology.local_ranks().len());
        let mut joins: Vec<Option<JoinHandle<()>>> =
            Vec::with_capacity(topology.local_ranks().len());

        for rank in topology.local_ranks().iter().copied() {
            let (command_sender, commands) = mpsc::channel();
            let rank_event_sender = event_sender.clone();
            let rank_initializer = Arc::clone(&initializer);
            let join = thread::Builder::new()
                .name(format!("sglang-tp-rank-{}", rank.global_rank()))
                .spawn(move || run_rank_worker(rank, rank_initializer, commands, rank_event_sender))
                .map_err(|error| WorkerGroupError::ThreadSpawn {
                    rank,
                    message: error.to_string(),
                });
            let join = match join {
                Ok(join) => join,
                Err(error) => {
                    for handle in &ranks {
                        let _ = handle.commands.send(RankCommand::Shutdown { sequence: 0 });
                    }
                    for join in joins.into_iter().flatten() {
                        let _ = join.join();
                    }
                    return Err(error);
                }
            };
            ranks.push(RankHandle {
                rank,
                commands: command_sender,
            });
            joins.push(Some(join));
        }
        drop(event_sender);

        let mut group = Self {
            ranks,
            events,
            joins,
            next_sequence: 1,
            shutdown: false,
        };
        let mut failures = Vec::new();
        for _ in 0..group.ranks.len() {
            match group.events.recv() {
                Ok(RankEvent::Ready { .. }) => {}
                Ok(RankEvent::StartupFailed { rank, message }) => {
                    failures.push(RankFailure { rank, message });
                }
                Ok(event) => failures.push(RankFailure {
                    rank: event.rank(),
                    message: format!("unexpected startup event: {}", event.name()),
                }),
                Err(_) => {
                    let _ = group.shutdown_inner();
                    return Err(WorkerGroupError::EventChannelClosed { stage: "startup" });
                }
            }
        }
        if failures.is_empty() {
            Ok(group)
        } else {
            let _ = group.shutdown_inner();
            Err(WorkerGroupError::RankFailures {
                stage: "startup",
                failures,
            })
        }
    }

    pub fn ranks(&self) -> impl ExactSizeIterator<Item = TensorParallelRank> + '_ {
        self.ranks.iter().map(|handle| handle.rank)
    }

    pub fn execute_all(
        &mut self,
        command: W::Command,
    ) -> Result<Vec<RankOutput<W::Output>>, WorkerGroupError> {
        if self.shutdown {
            return Err(WorkerGroupError::AlreadyShutdown);
        }
        let sequence = self.take_sequence();
        let mut failures = Vec::new();
        let mut expected_responses = 0;
        for handle in &self.ranks {
            match handle.commands.send(RankCommand::Execute {
                sequence,
                command: command.clone(),
            }) {
                Ok(()) => expected_responses += 1,
                Err(error) => failures.push(RankFailure {
                    rank: handle.rank,
                    message: error.to_string(),
                }),
            }
        }

        let mut outputs = Vec::with_capacity(expected_responses);
        for _ in 0..expected_responses {
            match self.events.recv() {
                Ok(RankEvent::Executed {
                    rank,
                    sequence: response_sequence,
                    result,
                }) if response_sequence == sequence => match result {
                    Ok(output) => outputs.push(RankOutput { rank, output }),
                    Err(message) => failures.push(RankFailure { rank, message }),
                },
                Ok(event) => {
                    return Err(WorkerGroupError::UnexpectedEvent {
                        stage: "execution",
                        detail: format!(
                            "{} from rank {} for sequence {:?}, expected {sequence}",
                            event.name(),
                            event.rank().global_rank(),
                            event.sequence()
                        ),
                    });
                }
                Err(_) => {
                    return Err(WorkerGroupError::EventChannelClosed { stage: "execution" });
                }
            }
        }
        if !failures.is_empty() {
            return Err(WorkerGroupError::RankFailures {
                stage: "execution",
                failures,
            });
        }
        outputs.sort_by_key(|output| output.rank.global_rank());
        Ok(outputs)
    }

    pub fn shutdown(&mut self) -> Result<(), WorkerGroupError> {
        if self.shutdown {
            return Ok(());
        }
        self.shutdown_inner()
    }

    fn shutdown_inner(&mut self) -> Result<(), WorkerGroupError> {
        if self.shutdown {
            return Ok(());
        }
        self.shutdown = true;
        let sequence = self.take_sequence();
        let mut failures = Vec::new();
        let mut expected_responses = 0;
        for handle in &self.ranks {
            match handle.commands.send(RankCommand::Shutdown { sequence }) {
                Ok(()) => expected_responses += 1,
                Err(error) => failures.push(RankFailure {
                    rank: handle.rank,
                    message: error.to_string(),
                }),
            }
        }
        for _ in 0..expected_responses {
            match self.events.recv() {
                Ok(RankEvent::Stopped {
                    rank,
                    sequence: response_sequence,
                    result,
                }) if response_sequence == sequence => {
                    if let Err(message) = result {
                        failures.push(RankFailure { rank, message });
                    }
                }
                Ok(event) => failures.push(RankFailure {
                    rank: event.rank(),
                    message: format!(
                        "unexpected {} event for sequence {:?} during shutdown sequence {sequence}",
                        event.name(),
                        event.sequence()
                    ),
                }),
                Err(_) => break,
            }
        }
        for (handle, join) in self.ranks.iter().zip(&mut self.joins) {
            if join.take().is_some_and(|join| join.join().is_err()) {
                failures.push(RankFailure {
                    rank: handle.rank,
                    message: "worker thread panicked".to_string(),
                });
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(WorkerGroupError::RankFailures {
                stage: "shutdown",
                failures,
            })
        }
    }

    fn take_sequence(&mut self) -> u64 {
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1);
        sequence
    }
}

impl<W> Drop for WorkerGroup<W>
where
    W: RankWorker,
{
    fn drop(&mut self) {
        let _ = self.shutdown_inner();
    }
}

struct RankHandle<C> {
    rank: TensorParallelRank,
    commands: Sender<RankCommand<C>>,
}

enum RankCommand<C> {
    Execute { sequence: u64, command: C },
    Shutdown { sequence: u64 },
}

enum RankEvent<O> {
    Ready {
        rank: TensorParallelRank,
    },
    StartupFailed {
        rank: TensorParallelRank,
        message: String,
    },
    Executed {
        rank: TensorParallelRank,
        sequence: u64,
        result: Result<O, String>,
    },
    Stopped {
        rank: TensorParallelRank,
        sequence: u64,
        result: Result<(), String>,
    },
}

impl<O> RankEvent<O> {
    fn rank(&self) -> TensorParallelRank {
        match self {
            Self::Ready { rank }
            | Self::StartupFailed { rank, .. }
            | Self::Executed { rank, .. }
            | Self::Stopped { rank, .. } => *rank,
        }
    }

    fn sequence(&self) -> Option<u64> {
        match self {
            Self::Executed { sequence, .. } | Self::Stopped { sequence, .. } => Some(*sequence),
            Self::Ready { .. } | Self::StartupFailed { .. } => None,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Ready { .. } => "ready",
            Self::StartupFailed { .. } => "startup-failed",
            Self::Executed { .. } => "executed",
            Self::Stopped { .. } => "stopped",
        }
    }
}

fn run_rank_worker<W, F, E>(
    rank: TensorParallelRank,
    initializer: Arc<F>,
    commands: Receiver<RankCommand<W::Command>>,
    events: Sender<RankEvent<W::Output>>,
) where
    W: RankWorker,
    F: Fn(TensorParallelRank) -> Result<W, E> + Send + Sync + 'static,
    E: fmt::Display + Send + 'static,
{
    let initialized = catch_unwind(AssertUnwindSafe(|| initializer(rank)));
    let mut worker = match initialized {
        Ok(Ok(worker)) => worker,
        Ok(Err(error)) => {
            let _ = events.send(RankEvent::StartupFailed {
                rank,
                message: error.to_string(),
            });
            return;
        }
        Err(panic) => {
            let _ = events.send(RankEvent::StartupFailed {
                rank,
                message: format!("initializer panicked: {}", panic_message(panic)),
            });
            return;
        }
    };
    if events.send(RankEvent::Ready { rank }).is_err() {
        let _ = worker.shutdown();
        return;
    }

    while let Ok(command) = commands.recv() {
        match command {
            RankCommand::Execute { sequence, command } => {
                let result = catch_unwind(AssertUnwindSafe(|| worker.execute(command)))
                    .map_err(|panic| format!("worker panicked: {}", panic_message(panic)))
                    .and_then(|result| result.map_err(|error| error.to_string()));
                if events
                    .send(RankEvent::Executed {
                        rank,
                        sequence,
                        result,
                    })
                    .is_err()
                {
                    let _ = worker.shutdown();
                    return;
                }
            }
            RankCommand::Shutdown { sequence } => {
                let result = catch_unwind(AssertUnwindSafe(|| worker.shutdown()))
                    .map_err(|panic| format!("worker panicked: {}", panic_message(panic)))
                    .and_then(|result| result.map_err(|error| error.to_string()));
                let _ = events.send(RankEvent::Stopped {
                    rank,
                    sequence,
                    result,
                });
                return;
            }
        }
    }
    let _ = worker.shutdown();
}

fn panic_message(panic: Box<dyn Any + Send>) -> String {
    if let Some(message) = panic.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct ArithmeticWorker {
        rank: TensorParallelRank,
        shutdown_count: Arc<AtomicUsize>,
    }

    impl RankWorker for ArithmeticWorker {
        type Command = usize;
        type Output = usize;
        type Error = Infallible;

        fn execute(&mut self, command: Self::Command) -> Result<Self::Output, Self::Error> {
            Ok(command + self.rank.global_rank())
        }

        fn shutdown(&mut self) -> Result<(), Self::Error> {
            self.shutdown_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[test]
    fn topology_maps_global_local_rank_and_device_without_platform_names() {
        let topology = TensorParallelTopology::new(8, 2, 1, 2, 3).expect("topology");
        let ranks = topology.local_ranks();

        assert_eq!(topology.world_size(), 8);
        assert_eq!(topology.node_count(), 2);
        assert_eq!(topology.node_rank(), 1);
        assert_eq!(ranks.len(), 4);
        assert_eq!(ranks[0].global_rank(), 4);
        assert_eq!(ranks[0].local_rank(), 0);
        assert_eq!(ranks[0].device_ordinal(), 2);
        assert_eq!(ranks[3].global_rank(), 7);
        assert_eq!(ranks[3].local_rank(), 3);
        assert_eq!(ranks[3].device_ordinal(), 11);
    }

    #[test]
    fn topology_fails_fast_on_invalid_geometry() {
        assert_eq!(
            TensorParallelTopology::new(0, 1, 0, 0, 1),
            Err(ParallelTopologyError::ZeroWorldSize)
        );
        assert_eq!(
            TensorParallelTopology::new(4, 0, 0, 0, 1),
            Err(ParallelTopologyError::ZeroNodeCount)
        );
        assert_eq!(
            TensorParallelTopology::new(4, 2, 2, 0, 1),
            Err(ParallelTopologyError::InvalidNodeRank {
                node_rank: 2,
                node_count: 2,
            })
        );
        assert_eq!(
            TensorParallelTopology::new(3, 2, 0, 0, 1),
            Err(ParallelTopologyError::WorldSizeNotDivisible {
                world_size: 3,
                node_count: 2,
            })
        );
        assert_eq!(
            TensorParallelTopology::new(2, 1, 0, 0, 0),
            Err(ParallelTopologyError::ZeroDeviceOrdinalStep)
        );
    }

    #[test]
    fn tensor_partition_maps_standard_and_replicated_rank_slices() {
        let topology = TensorParallelTopology::new(8, 1, 0, 0, 1).expect("topology");
        let rank_five = topology.local_ranks()[5];

        let standard = TensorPartition::for_rank(rank_five);
        assert_eq!(standard.range(64).expect("standard range"), 40..48);

        let replicated =
            TensorPartition::for_replicated_shards(rank_five, 2).expect("replicated KV shard");
        assert_eq!(replicated.partition_count(), 2);
        assert_eq!(replicated.partition_index(), 1);
        assert_eq!(replicated.range(16).expect("replicated range"), 8..16);
    }

    #[test]
    fn tensor_partition_rejects_uneven_or_invalid_shards() {
        let rank = TensorParallelTopology::new(4, 1, 0, 0, 1)
            .expect("topology")
            .local_ranks()[3];
        assert_eq!(
            TensorPartition::for_rank(rank).range(10),
            Err(TensorPartitionError::ExtentNotDivisible {
                extent: 10,
                partition_count: 4,
            })
        );
        assert_eq!(
            TensorPartition::for_replicated_shards(rank, 3),
            Err(TensorPartitionError::WorldSizeNotDivisibleByShardCount {
                world_size: 4,
                shard_count: 3,
            })
        );
    }

    #[test]
    fn worker_group_broadcasts_in_rank_order_and_shuts_down_each_rank() {
        let topology = TensorParallelTopology::new(3, 1, 0, 0, 1).expect("topology");
        let shutdown_count = Arc::new(AtomicUsize::new(0));
        let initializer_count = Arc::clone(&shutdown_count);
        let mut group = WorkerGroup::launch(&topology, move |rank| {
            Ok::<_, Infallible>(ArithmeticWorker {
                rank,
                shutdown_count: Arc::clone(&initializer_count),
            })
        })
        .expect("worker group");

        let outputs = group.execute_all(10).expect("broadcast");
        assert_eq!(
            outputs
                .iter()
                .map(|output| (output.rank().global_rank(), *output.output()))
                .collect::<Vec<_>>(),
            vec![(0, 10), (1, 11), (2, 12)]
        );
        group.shutdown().expect("shutdown");
        assert_eq!(shutdown_count.load(Ordering::SeqCst), 3);
        group.shutdown().expect("idempotent shutdown");
        assert_eq!(shutdown_count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn worker_group_reports_the_failing_startup_rank() {
        let topology = TensorParallelTopology::new(2, 1, 0, 4, 2).expect("topology");
        let error = WorkerGroup::<ArithmeticWorker>::launch(&topology, move |rank| {
            if rank.global_rank() == 1 {
                return Err("rank initialization rejected");
            }
            Ok(ArithmeticWorker {
                rank,
                shutdown_count: Arc::new(AtomicUsize::new(0)),
            })
        })
        .expect_err("rank one must fail");

        let WorkerGroupError::RankFailures { stage, failures } = error else {
            panic!("unexpected error: {error}");
        };
        assert_eq!(stage, "startup");
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].rank.global_rank(), 1);
        assert_eq!(failures[0].rank.device_ordinal(), 6);
        assert_eq!(failures[0].message, "rank initialization rejected");
    }
}
