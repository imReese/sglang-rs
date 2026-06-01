use crate::cache::CachePageId;
use crate::scheduler::{ForwardMode, ScheduleBatch, ScheduledRequest};
use crate::types::RequestId;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelWorkerBatch {
    forward_mode: ForwardMode,
    request_ids: Vec<RequestId>,
    input_ids: Vec<u32>,
    positions: Vec<usize>,
    sequence_lengths: Vec<usize>,
    request_offsets: Vec<usize>,
    prefix_cache_pages: Vec<Vec<CachePageId>>,
    out_cache_pages: Vec<CachePageId>,
}

impl ModelWorkerBatch {
    pub fn from_schedule_batch(batch: &ScheduleBatch) -> Self {
        let mut worker_batch = Self {
            forward_mode: batch.forward_mode(),
            request_ids: Vec::with_capacity(batch.batch_size()),
            input_ids: Vec::new(),
            positions: Vec::new(),
            sequence_lengths: Vec::with_capacity(batch.batch_size()),
            request_offsets: Vec::with_capacity(batch.batch_size()),
            prefix_cache_pages: Vec::with_capacity(batch.batch_size()),
            out_cache_pages: Vec::new(),
        };

        for request in batch.requests() {
            worker_batch.push_request(batch.forward_mode(), request);
        }

        worker_batch
    }

    pub fn forward_mode(&self) -> ForwardMode {
        self.forward_mode
    }

    pub fn request_ids(&self) -> &[RequestId] {
        &self.request_ids
    }

    pub fn input_ids(&self) -> &[u32] {
        &self.input_ids
    }

    pub fn positions(&self) -> &[usize] {
        &self.positions
    }

    pub fn sequence_lengths(&self) -> &[usize] {
        &self.sequence_lengths
    }

    pub fn request_offsets(&self) -> &[usize] {
        &self.request_offsets
    }

    pub fn prefix_cache_pages(&self) -> &[Vec<CachePageId>] {
        &self.prefix_cache_pages
    }

    pub fn out_cache_pages(&self) -> &[CachePageId] {
        &self.out_cache_pages
    }

    fn push_request(&mut self, forward_mode: ForwardMode, request: &ScheduledRequest) {
        self.request_ids.push(request.request_id().clone());
        self.request_offsets.push(self.input_ids.len());
        self.prefix_cache_pages
            .push(request.prefix_cache_pages().to_vec());

        match forward_mode {
            ForwardMode::Prefill => self.push_prefill_request(request),
            ForwardMode::Decode => self.push_decode_request(request),
        }
    }

    fn push_prefill_request(&mut self, request: &ScheduledRequest) {
        let prefix_len = request.prefix_cache_pages().len();
        let uncached_input_ids = request.uncached_input_ids();

        self.input_ids.extend_from_slice(uncached_input_ids);
        self.out_cache_pages
            .extend_from_slice(request.allocated_cache_pages());
        self.positions
            .extend(prefix_len..prefix_len + uncached_input_ids.len());
        self.sequence_lengths.push(request.input_ids().len());
    }

    fn push_decode_request(&mut self, request: &ScheduledRequest) {
        let decode_token = request.output_ids().last().copied().unwrap_or_default();

        self.input_ids.push(decode_token);
        self.positions
            .push(request.input_ids().len() + request.output_ids().len() - 1);
        self.sequence_lengths
            .push(request.input_ids().len() + request.output_ids().len());
    }
}
