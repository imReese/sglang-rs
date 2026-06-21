use std::collections::VecDeque;
use std::fmt;

use crate::cache::{
    CacheAllocationError, CachePageAllocator, CachePageId, PrefixMatch, RadixCache,
};
use crate::types::{DisaggregatedParams, FAKE_BOOTSTRAP_HOST, RequestId, SamplingParams};
use crate::worker::{DecodeRequestState, GeneratedToken, WorkerExecutionError, WorkerExecutor};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestStage {
    PrefillWaiting,
    PrefillForward,
    DecodeWaiting,
    DecodeForward,
    Finished,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ForwardMode {
    Prefill,
    Decode,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScheduledRequest {
    request_id: RequestId,
    input_ids: Vec<u32>,
    output_ids: Vec<u32>,
    allocated_cache_pages: Vec<CachePageId>,
    sequence_cache_pages: Vec<CachePageId>,
    sampling: SamplingParams,
    disaggregated_params: Option<DisaggregatedParams>,
    data_parallel_rank: i32,
    prefix_match: PrefixMatch,
    stage: RequestStage,
}

impl ScheduledRequest {
    pub fn new(request_id: RequestId, input_ids: Vec<u32>, sampling: SamplingParams) -> Self {
        let remaining_input_ids = input_ids.clone();
        Self {
            request_id,
            input_ids,
            output_ids: Vec::new(),
            allocated_cache_pages: Vec::new(),
            sequence_cache_pages: Vec::new(),
            sampling,
            disaggregated_params: None,
            data_parallel_rank: 0,
            prefix_match: PrefixMatch {
                matched_token_count: 0,
                cache_pages: Vec::new(),
                remaining_input_ids,
            },
            stage: RequestStage::PrefillWaiting,
        }
    }

    pub fn with_disaggregated_params(
        mut self,
        disaggregated_params: Option<DisaggregatedParams>,
    ) -> Self {
        self.disaggregated_params = disaggregated_params;
        self
    }

    pub fn with_data_parallel_rank(mut self, data_parallel_rank: i32) -> Self {
        self.data_parallel_rank = data_parallel_rank;
        self
    }

    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    pub(crate) fn into_request_id(self) -> RequestId {
        self.request_id
    }

    pub fn input_ids(&self) -> &[u32] {
        &self.input_ids
    }

    pub fn output_ids(&self) -> &[u32] {
        &self.output_ids
    }

    pub fn sampling(&self) -> &SamplingParams {
        &self.sampling
    }

    pub fn disaggregated_params(&self) -> Option<&DisaggregatedParams> {
        self.disaggregated_params.as_ref()
    }

    pub fn data_parallel_rank(&self) -> i32 {
        self.data_parallel_rank
    }

    pub fn skips_radix_cache_insert(&self) -> bool {
        self.disaggregated_params
            .as_ref()
            .is_some_and(|params| params.bootstrap_host == FAKE_BOOTSTRAP_HOST)
    }

    pub fn stage(&self) -> RequestStage {
        self.stage
    }

    pub fn prefix_cache_pages(&self) -> &[crate::cache::CachePageId] {
        &self.prefix_match.cache_pages
    }

    pub fn uncached_input_ids(&self) -> &[u32] {
        &self.prefix_match.remaining_input_ids
    }

    pub fn allocated_cache_pages(&self) -> &[CachePageId] {
        &self.allocated_cache_pages
    }

    pub fn sequence_cache_pages(&self) -> &[CachePageId] {
        &self.sequence_cache_pages
    }

    pub fn cached_token_count(&self) -> usize {
        self.prefix_match.matched_token_count
    }

    fn apply_prefix_match(&mut self, prefix_match: PrefixMatch) {
        self.prefix_match = prefix_match;
    }

    fn set_stage(&mut self, stage: RequestStage) {
        self.stage = stage;
    }

    fn set_allocated_cache_pages(&mut self, cache_pages: Vec<CachePageId>) {
        self.allocated_cache_pages = cache_pages;
    }

    fn set_sequence_cache_pages(&mut self, cache_pages: Vec<CachePageId>) {
        self.sequence_cache_pages = cache_pages;
    }

    fn append_sequence_cache_pages(&mut self, cache_pages: &[CachePageId]) {
        self.sequence_cache_pages.extend_from_slice(cache_pages);
    }

    fn clear_forward_cache_allocation(&mut self) {
        let allocated_page_count = self.allocated_cache_pages.len();
        if allocated_page_count > 0 {
            self.sequence_cache_pages.truncate(
                self.sequence_cache_pages
                    .len()
                    .saturating_sub(allocated_page_count),
            );
        }
        self.allocated_cache_pages.clear();
    }

    fn append_output_ids(&mut self, output_ids: &[u32]) {
        self.output_ids.extend_from_slice(output_ids);
    }

    fn reached_max_new_tokens(&self) -> bool {
        self.output_ids.len() >= self.sampling.max_new_tokens
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum SchedulerError {
    EmptyQueue,
    RunningRequestLimitReached { max_running_requests: usize },
    DecodeNotReady { request_id: RequestId },
    CacheAllocation(CacheAllocationError),
    Worker(WorkerExecutionError),
}

impl fmt::Display for SchedulerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyQueue => formatter.write_str("scheduler queue is empty"),
            Self::RunningRequestLimitReached {
                max_running_requests,
            } => write!(
                formatter,
                "running request limit reached: {max_running_requests}"
            ),
            Self::DecodeNotReady { request_id } => {
                write!(
                    formatter,
                    "decode request {} is not ready",
                    request_id.as_str()
                )
            }
            Self::CacheAllocation(error) => write!(formatter, "cache allocation error: {error}"),
            Self::Worker(error) => write!(formatter, "worker execution error: {error}"),
        }
    }
}

impl std::error::Error for SchedulerError {}

impl From<CacheAllocationError> for SchedulerError {
    fn from(value: CacheAllocationError) -> Self {
        Self::CacheAllocation(value)
    }
}

impl From<WorkerExecutionError> for SchedulerError {
    fn from(value: WorkerExecutionError) -> Self {
        Self::Worker(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduledOutput {
    pub request_id: RequestId,
    pub token_ids: Vec<u32>,
    pub cached_tokens: usize,
    pub finished: bool,
}

pub struct Scheduler<W> {
    waiting_queue: VecDeque<ScheduledRequest>,
    decode_queue: VecDeque<ScheduledRequest>,
    prefix_cache: RadixCache,
    cache_page_allocator: Option<CachePageAllocator>,
    max_running_requests: Option<usize>,
    worker: W,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScheduleBatch {
    forward_mode: ForwardMode,
    requests: Vec<ScheduledRequest>,
}

impl ScheduleBatch {
    fn prefill(requests: Vec<ScheduledRequest>) -> Self {
        Self {
            forward_mode: ForwardMode::Prefill,
            requests,
        }
    }

    fn decode(requests: Vec<ScheduledRequest>) -> Self {
        Self {
            forward_mode: ForwardMode::Decode,
            requests,
        }
    }

    pub fn forward_mode(&self) -> ForwardMode {
        self.forward_mode
    }

    pub fn batch_size(&self) -> usize {
        self.requests.len()
    }

    pub fn requests(&self) -> &[ScheduledRequest] {
        &self.requests
    }

    pub fn total_uncached_tokens(&self) -> usize {
        self.requests
            .iter()
            .map(|request| request.uncached_input_ids().len())
            .sum()
    }

    pub(crate) fn into_requests(self) -> Vec<ScheduledRequest> {
        self.requests
    }
}

impl<W> Scheduler<W> {
    pub fn new(worker: W) -> Self {
        Self::with_prefix_cache(worker, RadixCache::default())
    }

    pub fn with_prefix_cache(worker: W, prefix_cache: RadixCache) -> Self {
        Self {
            waiting_queue: VecDeque::new(),
            decode_queue: VecDeque::new(),
            prefix_cache,
            cache_page_allocator: None,
            max_running_requests: None,
            worker,
        }
    }

    pub fn with_cache_resources(
        worker: W,
        prefix_cache: RadixCache,
        cache_page_allocator: CachePageAllocator,
    ) -> Self {
        Self {
            waiting_queue: VecDeque::new(),
            decode_queue: VecDeque::new(),
            prefix_cache,
            cache_page_allocator: Some(cache_page_allocator),
            max_running_requests: None,
            worker,
        }
    }

    pub fn with_max_running_requests(mut self, max_running_requests: Option<usize>) -> Self {
        self.max_running_requests = max_running_requests;
        self
    }

    pub fn max_running_requests(&self) -> Option<usize> {
        self.max_running_requests
    }

    pub fn enqueue(&mut self, request: ScheduledRequest) {
        self.waiting_queue.push_back(request);
    }

    pub fn waiting_queue_depth(&self) -> usize {
        self.waiting_queue.len()
    }

    pub fn decode_queue_depth(&self) -> usize {
        self.decode_queue.len()
    }

    pub fn available_cache_pages(&self) -> Option<usize> {
        self.cache_page_allocator
            .as_ref()
            .map(CachePageAllocator::available_pages)
    }

    pub fn abort_request(&mut self, request_id: &RequestId) -> bool
    where
        W: WorkerExecutor,
    {
        if let Some(request) = remove_request_from_queue(&mut self.waiting_queue, request_id) {
            self.worker.complete_request(&request);
            return true;
        }

        if let Some(request) = remove_request_from_queue(&mut self.decode_queue, request_id) {
            self.worker.complete_request(&request);
            return true;
        }

        false
    }

    pub fn abort_all_requests(&mut self) -> usize
    where
        W: WorkerExecutor,
    {
        let mut aborted = 0;
        while let Some(request) = self.waiting_queue.pop_front() {
            self.worker.complete_request(&request);
            aborted += 1;
        }
        while let Some(request) = self.decode_queue.pop_front() {
            self.worker.complete_request(&request);
            aborted += 1;
        }
        aborted
    }

    pub fn worker(&self) -> &W {
        &self.worker
    }

    pub fn worker_mut(&mut self) -> &mut W {
        &mut self.worker
    }

    pub fn flush_cache(&mut self) -> bool {
        if !self.waiting_queue.is_empty() || !self.decode_queue.is_empty() {
            return false;
        }

        self.prefix_cache.clear();
        if let Some(allocator) = self.cache_page_allocator.as_mut() {
            allocator.reset();
        }

        true
    }

    pub fn next_prefill_batch(
        &mut self,
        max_batch_size: usize,
    ) -> Result<ScheduleBatch, SchedulerError> {
        self.next_prefill_batch_with_token_budget(max_batch_size, usize::MAX)
    }

    pub fn next_prefill_batch_with_token_budget(
        &mut self,
        max_batch_size: usize,
        max_uncached_tokens: usize,
    ) -> Result<ScheduleBatch, SchedulerError> {
        if self.waiting_queue.is_empty() || max_batch_size == 0 {
            return Err(SchedulerError::EmptyQueue);
        }

        let max_batch_size = max_batch_size.min(self.remaining_running_request_capacity()?);
        let mut requests = Vec::with_capacity(max_batch_size.min(self.waiting_queue.len()));
        let mut uncached_token_count = 0;

        while requests.len() < max_batch_size {
            let Some(mut request) = self.waiting_queue.pop_front() else {
                break;
            };
            if let Err(error) = self.prepare_prefill_request(&mut request) {
                self.restore_failed_prefill_attempt(requests, request);
                return Err(error);
            }
            let request_uncached_tokens = request.uncached_input_ids().len();

            if !requests.is_empty()
                && uncached_token_count + request_uncached_tokens > max_uncached_tokens
            {
                self.waiting_queue.push_front(request);
                break;
            }

            uncached_token_count += request_uncached_tokens;
            requests.push(request);
        }

        if requests.is_empty() {
            let mut request = self
                .waiting_queue
                .pop_front()
                .ok_or(SchedulerError::EmptyQueue)?;
            if let Err(error) = self.prepare_prefill_request(&mut request) {
                self.waiting_queue.push_front(request);
                return Err(error);
            }
            requests.push(request);
        }

        Ok(ScheduleBatch::prefill(requests))
    }

    fn remaining_running_request_capacity(&self) -> Result<usize, SchedulerError> {
        let Some(max_running_requests) = self.max_running_requests else {
            return Ok(usize::MAX);
        };

        let remaining = max_running_requests.saturating_sub(self.decode_queue.len());
        if remaining == 0 {
            return Err(SchedulerError::RunningRequestLimitReached {
                max_running_requests,
            });
        }

        Ok(remaining)
    }

    fn prepare_prefill_request(
        &mut self,
        request: &mut ScheduledRequest,
    ) -> Result<(), SchedulerError> {
        let prefix_match = self.prefix_cache.match_prefix(request.input_ids());
        let uncached_token_count = prefix_match.remaining_input_ids.len();
        let allocated_cache_pages = match self.cache_page_allocator.as_mut() {
            Some(allocator) => allocator.allocate(uncached_token_count)?,
            None => Vec::new(),
        };
        let mut sequence_cache_pages =
            Vec::with_capacity(prefix_match.cache_pages.len() + allocated_cache_pages.len());
        sequence_cache_pages.extend_from_slice(&prefix_match.cache_pages);
        sequence_cache_pages.extend_from_slice(&allocated_cache_pages);
        request.apply_prefix_match(prefix_match);
        request.set_allocated_cache_pages(allocated_cache_pages);
        request.set_sequence_cache_pages(sequence_cache_pages);
        request.set_stage(RequestStage::PrefillForward);
        Ok(())
    }

    fn restore_failed_prefill_attempt(
        &mut self,
        prepared_requests: Vec<ScheduledRequest>,
        failed_request: ScheduledRequest,
    ) {
        if let Some(allocator) = self.cache_page_allocator.as_mut() {
            for request in prepared_requests.iter() {
                allocator.release(request.allocated_cache_pages());
            }
        }

        self.waiting_queue.push_front(failed_request);
        for mut request in prepared_requests.into_iter().rev() {
            request.set_allocated_cache_pages(Vec::new());
            request.set_sequence_cache_pages(Vec::new());
            request.set_stage(RequestStage::PrefillWaiting);
            self.waiting_queue.push_front(request);
        }
    }

    pub fn next_decode_batch(
        &mut self,
        max_batch_size: usize,
    ) -> Result<ScheduleBatch, SchedulerError> {
        self.next_decode_batch_with_ready_check(max_batch_size, |_| true)
    }

    pub fn next_decode_batch_with_ready_check<F>(
        &mut self,
        max_batch_size: usize,
        mut is_ready: F,
    ) -> Result<ScheduleBatch, SchedulerError>
    where
        F: FnMut(&ScheduledRequest) -> bool,
    {
        if self.decode_queue.is_empty() || max_batch_size == 0 {
            return Err(SchedulerError::EmptyQueue);
        }

        let batch_size = max_batch_size.min(self.decode_queue.len());
        let mut requests = Vec::with_capacity(batch_size);

        while requests.len() < batch_size {
            let Some(next_request) = self.decode_queue.front() else {
                break;
            };

            if !is_ready(next_request) {
                if requests.is_empty() {
                    return Err(SchedulerError::DecodeNotReady {
                        request_id: next_request.request_id().clone(),
                    });
                }
                break;
            }

            let mut request = self
                .decode_queue
                .pop_front()
                .ok_or(SchedulerError::EmptyQueue)?;
            if let Err(error) = self.prepare_decode_request(&mut request) {
                self.restore_failed_decode_attempt(requests, request);
                return Err(error);
            }
            request.set_stage(RequestStage::DecodeForward);
            requests.push(request);
        }

        Ok(ScheduleBatch::decode(requests))
    }

    fn prepare_decode_request(
        &mut self,
        request: &mut ScheduledRequest,
    ) -> Result<(), SchedulerError> {
        let allocated_cache_pages = match self.cache_page_allocator.as_mut() {
            Some(allocator) => allocator.allocate(1)?,
            None => Vec::new(),
        };
        request.set_allocated_cache_pages(allocated_cache_pages);
        let allocated_cache_pages = request.allocated_cache_pages().to_vec();
        request.append_sequence_cache_pages(&allocated_cache_pages);
        Ok(())
    }

    fn restore_failed_decode_attempt(
        &mut self,
        prepared_requests: Vec<ScheduledRequest>,
        failed_request: ScheduledRequest,
    ) {
        if let Some(allocator) = self.cache_page_allocator.as_mut() {
            for request in prepared_requests.iter() {
                allocator.release(request.allocated_cache_pages());
            }
        }

        self.decode_queue.push_front(failed_request);
        for mut request in prepared_requests.into_iter().rev() {
            request.clear_forward_cache_allocation();
            request.set_stage(RequestStage::DecodeWaiting);
            self.decode_queue.push_front(request);
        }
    }
}

impl<W> Scheduler<W>
where
    W: WorkerExecutor,
{
    pub fn dispatch_prefill_batch(
        &mut self,
        max_batch_size: usize,
    ) -> Result<Vec<ScheduledOutput>, SchedulerError> {
        self.dispatch_prefill_batch_with_token_budget(max_batch_size, usize::MAX)
    }

    pub fn dispatch_prefill_batch_with_token_budget(
        &mut self,
        max_batch_size: usize,
        max_uncached_tokens: usize,
    ) -> Result<Vec<ScheduledOutput>, SchedulerError> {
        let batch =
            self.next_prefill_batch_with_token_budget(max_batch_size, max_uncached_tokens)?;
        self.dispatch_batch(batch)
    }

    pub fn dispatch_decode_batch(
        &mut self,
        max_batch_size: usize,
    ) -> Result<Vec<ScheduledOutput>, SchedulerError> {
        let batch = self.next_decode_batch_with_worker_ready_check(max_batch_size)?;
        self.dispatch_batch(batch)
    }

    pub fn dispatch_decode_batch_with_ready_check<F>(
        &mut self,
        max_batch_size: usize,
        is_ready: F,
    ) -> Result<Vec<ScheduledOutput>, SchedulerError>
    where
        F: FnMut(&ScheduledRequest) -> bool,
    {
        let batch = self.next_decode_batch_with_ready_check(max_batch_size, is_ready)?;
        self.dispatch_batch(batch)
    }

    fn next_decode_batch_with_worker_ready_check(
        &mut self,
        max_batch_size: usize,
    ) -> Result<ScheduleBatch, SchedulerError> {
        if self.decode_queue.is_empty() || max_batch_size == 0 {
            return Err(SchedulerError::EmptyQueue);
        }

        let batch_size = max_batch_size.min(self.decode_queue.len());
        let mut requests = Vec::with_capacity(batch_size);

        while requests.len() < batch_size {
            let Some(next_request) = self.decode_queue.front() else {
                break;
            };

            match self.worker.decode_request_state(next_request)? {
                DecodeRequestState::Ready => {}
                DecodeRequestState::Pending => {
                    if requests.is_empty() {
                        return Err(SchedulerError::DecodeNotReady {
                            request_id: next_request.request_id().clone(),
                        });
                    }
                    break;
                }
                DecodeRequestState::Failed(message) => {
                    return Err(SchedulerError::Worker(WorkerExecutionError::Runtime(
                        message,
                    )));
                }
            }

            let mut request = self
                .decode_queue
                .pop_front()
                .ok_or(SchedulerError::EmptyQueue)?;
            if let Err(error) = self.prepare_decode_request(&mut request) {
                self.restore_failed_decode_attempt(requests, request);
                return Err(error);
            }
            request.set_stage(RequestStage::DecodeForward);
            requests.push(request);
        }

        Ok(ScheduleBatch::decode(requests))
    }

    fn dispatch_batch(
        &mut self,
        batch: ScheduleBatch,
    ) -> Result<Vec<ScheduledOutput>, SchedulerError> {
        let generated = match self.worker.execute_batch(&batch) {
            Ok(generated) => generated,
            Err(error) => {
                self.release_failed_batch_cache_pages(&batch);
                return Err(error.into());
            }
        };
        let forward_mode = batch.forward_mode();
        let requests = batch.into_requests();
        let tokens = generated.into_tokens();
        let mut outputs = Vec::with_capacity(requests.len());

        for (mut request, generated_token) in requests.into_iter().zip(tokens.into_iter()) {
            if forward_mode == ForwardMode::Prefill {
                self.publish_prefill_cache_pages(&request);
            }

            request.append_output_ids(generated_token.token_ids());

            let finished = generated_token.is_finished() || request.reached_max_new_tokens();

            if finished {
                request.set_stage(RequestStage::Finished);
                self.worker.complete_request(&request);
            } else {
                request.set_stage(RequestStage::DecodeWaiting);
                self.decode_queue.push_back(request.clone());
            }

            outputs.push(scheduled_output(request, generated_token, finished));
        }

        Ok(outputs)
    }

    fn release_failed_batch_cache_pages(&mut self, batch: &ScheduleBatch) {
        let Some(allocator) = self.cache_page_allocator.as_mut() else {
            return;
        };

        for request in batch.requests() {
            allocator.release(request.allocated_cache_pages());
        }
    }

    fn publish_prefill_cache_pages(&mut self, request: &ScheduledRequest) {
        if request.allocated_cache_pages().is_empty() || request.skips_radix_cache_insert() {
            return;
        }

        let input_ids = request.input_ids();
        let prefix_pages = request.prefix_cache_pages();
        let allocated_pages = request.allocated_cache_pages();
        let mut cache_pages = Vec::with_capacity(prefix_pages.len() + allocated_pages.len());
        cache_pages.extend_from_slice(prefix_pages);
        cache_pages.extend_from_slice(allocated_pages);

        // The scheduler produced the page list from the same request tokens, so
        // length mismatch would indicate an internal invariant violation.
        self.prefix_cache
            .insert(input_ids, &cache_pages)
            .expect("prefill cache pages must match request input length");
    }

    pub fn dispatch_next(&mut self) -> Result<ScheduledOutput, SchedulerError> {
        self.dispatch_prefill_batch(1)?
            .pop()
            .ok_or(SchedulerError::EmptyQueue)
    }
}

fn scheduled_output(
    request: ScheduledRequest,
    generated: GeneratedToken,
    finished: bool,
) -> ScheduledOutput {
    ScheduledOutput {
        cached_tokens: request.cached_token_count(),
        request_id: request.into_request_id(),
        token_ids: generated.token_ids().to_vec(),
        finished,
    }
}

fn remove_request_from_queue(
    queue: &mut VecDeque<ScheduledRequest>,
    request_id: &RequestId,
) -> Option<ScheduledRequest> {
    let Some(index) = queue
        .iter()
        .position(|request| request.request_id() == request_id)
    else {
        return None;
    };

    queue.remove(index)
}
