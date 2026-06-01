use std::collections::VecDeque;
use std::fmt;

use crate::cache::{
    CacheAllocationError, CachePageAllocator, CachePageId, PrefixMatch, RadixCache,
};
use crate::types::{RequestId, SamplingParams};
use crate::worker::{GeneratedToken, WorkerExecutor};

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduledRequest {
    request_id: RequestId,
    input_ids: Vec<u32>,
    output_ids: Vec<u32>,
    allocated_cache_pages: Vec<CachePageId>,
    sampling: SamplingParams,
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
            sampling,
            prefix_match: PrefixMatch {
                matched_token_count: 0,
                cache_pages: Vec::new(),
                remaining_input_ids,
            },
            stage: RequestStage::PrefillWaiting,
        }
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

    fn apply_prefix_match(&mut self, prefix_match: PrefixMatch) {
        self.prefix_match = prefix_match;
    }

    fn set_stage(&mut self, stage: RequestStage) {
        self.stage = stage;
    }

    fn set_allocated_cache_pages(&mut self, cache_pages: Vec<CachePageId>) {
        self.allocated_cache_pages = cache_pages;
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
    CacheAllocation(CacheAllocationError),
}

impl fmt::Display for SchedulerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyQueue => formatter.write_str("scheduler queue is empty"),
            Self::CacheAllocation(error) => write!(formatter, "cache allocation error: {error}"),
        }
    }
}

impl std::error::Error for SchedulerError {}

impl From<CacheAllocationError> for SchedulerError {
    fn from(value: CacheAllocationError) -> Self {
        Self::CacheAllocation(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduledOutput {
    pub request_id: RequestId,
    pub token_ids: Vec<u32>,
    pub finished: bool,
}

pub struct Scheduler<W> {
    waiting_queue: VecDeque<ScheduledRequest>,
    decode_queue: VecDeque<ScheduledRequest>,
    prefix_cache: RadixCache,
    cache_page_allocator: Option<CachePageAllocator>,
    worker: W,
}

#[derive(Clone, Debug, Eq, PartialEq)]
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
            worker,
        }
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

    pub fn worker(&self) -> &W {
        &self.worker
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

    fn prepare_prefill_request(
        &mut self,
        request: &mut ScheduledRequest,
    ) -> Result<(), SchedulerError> {
        let prefix_match = self.prefix_cache.match_prefix(request.input_ids());
        request.apply_prefix_match(prefix_match);
        let uncached_token_count = request.uncached_input_ids().len();
        let allocated_cache_pages = match self.cache_page_allocator.as_mut() {
            Some(allocator) => allocator.allocate(uncached_token_count)?,
            None => Vec::new(),
        };
        request.set_allocated_cache_pages(allocated_cache_pages);
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
            request.set_stage(RequestStage::PrefillWaiting);
            self.waiting_queue.push_front(request);
        }
    }

    pub fn next_decode_batch(
        &mut self,
        max_batch_size: usize,
    ) -> Result<ScheduleBatch, SchedulerError> {
        if self.decode_queue.is_empty() || max_batch_size == 0 {
            return Err(SchedulerError::EmptyQueue);
        }

        let batch_size = max_batch_size.min(self.decode_queue.len());
        let mut requests = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let mut request = self
                .decode_queue
                .pop_front()
                .ok_or(SchedulerError::EmptyQueue)?;
            request.set_stage(RequestStage::DecodeForward);
            requests.push(request);
        }

        Ok(ScheduleBatch::decode(requests))
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
        let batch = self.next_decode_batch(max_batch_size)?;
        self.dispatch_batch(batch)
    }

    fn dispatch_batch(
        &mut self,
        batch: ScheduleBatch,
    ) -> Result<Vec<ScheduledOutput>, SchedulerError> {
        let generated = self.worker.execute_batch(&batch);
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
            } else {
                request.set_stage(RequestStage::DecodeWaiting);
                self.decode_queue.push_back(request.clone());
            }

            outputs.push(scheduled_output(request, generated_token, finished));
        }

        Ok(outputs)
    }

    fn publish_prefill_cache_pages(&mut self, request: &ScheduledRequest) {
        if request.allocated_cache_pages().is_empty() {
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
        request_id: request.into_request_id(),
        token_ids: generated.token_ids().to_vec(),
        finished,
    }
}
