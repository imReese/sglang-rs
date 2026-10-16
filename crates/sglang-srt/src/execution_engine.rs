use std::fmt;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread::{self, JoinHandle};

use crate::scheduler::ScheduleBatch;
use crate::worker::{BatchGeneratedTokens, WorkerExecutionError, WorkerExecutor};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExecutionId(u64);

impl ExecutionId {
    pub fn get(self) -> u64 {
        self.0
    }
}

pub struct ExecutionCommand {
    id: ExecutionId,
    batch: ScheduleBatch,
}

impl fmt::Debug for ExecutionCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecutionCommand")
            .field("id", &self.id)
            .field("batch", &self.batch)
            .finish()
    }
}

#[derive(Debug)]
pub struct ExecutionCompletion {
    pub id: ExecutionId,
    pub result: Result<BatchGeneratedTokens, WorkerExecutionError>,
}

pub struct ExecutionEngine {
    command_tx: Option<Sender<ExecutionCommand>>,
    completion_rx: Receiver<ExecutionCompletion>,
    next_id: u64,
    in_flight: usize,
    join_handle: Option<JoinHandle<()>>,
}

impl fmt::Debug for ExecutionEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecutionEngine")
            .field("in_flight", &self.in_flight)
            .finish_non_exhaustive()
    }
}

impl ExecutionEngine {
    pub fn spawn<W>(mut worker: W) -> Self
    where
        W: WorkerExecutor + Send + 'static,
    {
        let (command_tx, command_rx) = mpsc::channel::<ExecutionCommand>();
        let (completion_tx, completion_rx) = mpsc::channel::<ExecutionCompletion>();

        let join_handle = thread::Builder::new()
            .name("sglang-execution-engine".to_string())
            .spawn(move || {
                while let Ok(command) = command_rx.recv() {
                    let result = worker.execute_batch(&command.batch);
                    let _ = completion_tx.send(ExecutionCompletion {
                        id: command.id,
                        result,
                    });
                }
                let _ = worker.shutdown();
            })
            .expect("failed to spawn sglang execution engine");

        Self {
            command_tx: Some(command_tx),
            completion_rx,
            next_id: 0,
            in_flight: 0,
            join_handle: Some(join_handle),
        }
    }

    pub fn submit(
        &mut self,
        batch: ScheduleBatch,
    ) -> Result<ExecutionId, ExecutionEngineSubmitError> {
        let id = ExecutionId(self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        self.command_tx
            .as_ref()
            .ok_or(ExecutionEngineSubmitError::WorkerStopped)?
            .send(ExecutionCommand { id, batch })
            .map_err(|_| ExecutionEngineSubmitError::WorkerStopped)?;
        self.in_flight += 1;
        Ok(id)
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight
    }

    pub fn try_recv(&mut self) -> Result<Option<ExecutionCompletion>, ExecutionEnginePollError> {
        match self.completion_rx.try_recv() {
            Ok(completion) => {
                self.in_flight = self.in_flight.saturating_sub(1);
                Ok(Some(completion))
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(ExecutionEnginePollError::WorkerStopped),
        }
    }

    pub fn recv(&mut self) -> Result<ExecutionCompletion, ExecutionEnginePollError> {
        match self.completion_rx.recv() {
            Ok(completion) => {
                self.in_flight = self.in_flight.saturating_sub(1);
                Ok(completion)
            }
            Err(_) => Err(ExecutionEnginePollError::WorkerStopped),
        }
    }
}

impl Drop for ExecutionEngine {
    fn drop(&mut self) {
        self.command_tx.take();
        if let Some(join_handle) = self.join_handle.take() {
            let _ = join_handle.join();
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum ExecutionEngineSubmitError {
    WorkerStopped,
}

impl fmt::Display for ExecutionEngineSubmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkerStopped => f.write_str("execution engine has stopped"),
        }
    }
}
impl std::error::Error for ExecutionEngineSubmitError {}

#[derive(Debug, Eq, PartialEq)]
pub enum ExecutionEnginePollError {
    Execution(WorkerExecutionError),
    WorkerStopped,
}

impl fmt::Display for ExecutionEnginePollError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Execution(error) => write!(f, "execution error: {error}"),
            Self::WorkerStopped => f.write_str("execution engine has stopped"),
        }
    }
}
impl std::error::Error for ExecutionEnginePollError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::ScheduleBatch;
    use crate::worker::{BatchGeneratedTokens, GeneratedToken, ModelWorker};

    struct TestWorker;

    impl ModelWorker for TestWorker {
        fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
            let tokens = (0..batch.batch_size())
                .map(|_| GeneratedToken::finished(vec![42]))
                .collect();
            BatchGeneratedTokens::from_batch(batch, tokens).expect("test output matches batch")
        }
    }

    #[test]
    fn submit_and_receive_are_separate_operations() {
        let mut engine = ExecutionEngine::spawn(TestWorker);
        let id = engine
            .submit(ScheduleBatch::decode(Vec::new()))
            .expect("engine accepts command");
        assert_eq!(id.get(), 0);
        assert_eq!(engine.in_flight(), 1);
        let completion = engine.recv().expect("engine completes command");
        assert_eq!(completion.id, id);
        assert_eq!(engine.in_flight(), 0);
        assert_eq!(
            completion
                .result
                .expect("execution succeeds")
                .into_tokens()
                .len(),
            0
        );
    }

    #[test]
    fn multiple_submissions_have_stable_ids() {
        let mut engine = ExecutionEngine::spawn(TestWorker);
        let first = engine
            .submit(ScheduleBatch::prefill(Vec::new()))
            .expect("first accepted");
        let second = engine
            .submit(ScheduleBatch::decode(Vec::new()))
            .expect("second accepted");
        assert_eq!([first.get(), second.get()], [0, 1]);
        assert_eq!(engine.in_flight(), 2);
        assert_eq!(engine.recv().expect("first completion").id, first);
        assert_eq!(engine.recv().expect("second completion").id, second);
        assert_eq!(engine.in_flight(), 0);
    }
}
