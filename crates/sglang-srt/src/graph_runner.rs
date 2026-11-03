// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::fmt;

use crate::model_executor::{LogitSamplerLike, ModelForwardError, ModelForwardOutput};
use crate::scheduler::{ForwardMode, ScheduleBatch, ScheduledRequest};
use crate::types::RequestId;
use crate::worker::{
    BatchGeneratedTokens, FallibleModelWorker, GeneratedToken, WorkerExecutionError,
    WorkerWeightUpdateRequest,
};

/// Errors originating in the bucketized graph model runner execution engine.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphRunnerError {
    EmptyBuckets,
    UnsortedBuckets,
    ZeroBucket,
    EmptyBatch,
    ZeroPageSize,
    BucketPaddingOverflow { requested: usize, bucket: usize },
    ModelForward(ModelForwardError),
}

impl fmt::Display for GraphRunnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBuckets => write!(f, "bucket table must not be empty"),
            Self::UnsortedBuckets => write!(f, "bucket sizes must be strictly increasing"),
            Self::ZeroBucket => write!(f, "bucket size must be non-zero"),
            Self::EmptyBatch => write!(f, "batch size must be positive"),
            Self::ZeroPageSize => write!(f, "KV cache page size must be positive"),
            Self::BucketPaddingOverflow { requested, bucket } => write!(
                f,
                "requested batch size {requested} exceeds target bucket size {bucket}"
            ),
            Self::ModelForward(err) => write!(f, "model forward error: {err}"),
        }
    }
}

impl std::error::Error for GraphRunnerError {}

impl From<ModelForwardError> for GraphRunnerError {
    fn from(err: ModelForwardError) -> Self {
        Self::ModelForward(err)
    }
}

/// Bucket sizing configuration for discrete CUDA Graph capture & replay.
/// Standard sizes range from 1 to 256 for sub-microsecond decode launches.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchBucketTable {
    buckets: Vec<usize>,
}

impl Default for BatchBucketTable {
    fn default() -> Self {
        Self {
            buckets: vec![
                1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256,
            ],
        }
    }
}

impl BatchBucketTable {
    pub fn new(buckets: Vec<usize>) -> Result<Self, GraphRunnerError> {
        if buckets.is_empty() {
            return Err(GraphRunnerError::EmptyBuckets);
        }
        let mut prev = 0;
        for &b in &buckets {
            if b == 0 {
                return Err(GraphRunnerError::ZeroBucket);
            }
            if b <= prev {
                return Err(GraphRunnerError::UnsortedBuckets);
            }
            prev = b;
        }
        Ok(Self { buckets })
    }

    /// Finds the smallest pre-captured bucket $B_{\text{bucket}} \ge B$.
    /// Returns `None` if batch size exceeds the maximum bucket, signaling Eager fallback.
    pub fn select_bucket(&self, batch_size: usize) -> Option<usize> {
        self.buckets.iter().copied().find(|&b| b >= batch_size)
    }

    pub fn max_bucket(&self) -> usize {
        *self.buckets.last().unwrap_or(&0)
    }

    pub fn buckets(&self) -> &[usize] {
        &self.buckets
    }
}

/// Hardware-optimized batch descriptor mapping high-level scheduler requests
/// into contiguous memory buffers suitable for pinned GPU transfers and MLA / PagedAttention kernels.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelInputDescriptor {
    forward_mode: ForwardMode,
    batch_size: usize,
    total_tokens: usize,
    input_ids: Vec<u32>,
    positions: Vec<usize>,
    context_lens: Vec<usize>,
    slot_mapping: Vec<usize>,
    block_tables: Vec<Vec<usize>>,
    request_ids: Vec<RequestId>,
    schedule_batch: Option<ScheduleBatch>,
}

impl ModelInputDescriptor {
    pub fn from_schedule_batch(
        batch: &ScheduleBatch,
        page_size: usize,
    ) -> Result<Self, GraphRunnerError> {
        if batch.batch_size() == 0 {
            return Err(GraphRunnerError::EmptyBatch);
        }
        if page_size == 0 {
            return Err(GraphRunnerError::ZeroPageSize);
        }

        let mode = batch.forward_mode();
        let batch_size = batch.batch_size();

        let mut input_ids = Vec::new();
        let mut positions = Vec::new();
        let mut context_lens = Vec::with_capacity(batch_size);
        let mut slot_mapping = Vec::new();
        let mut raw_block_tables = Vec::with_capacity(batch_size);
        let mut request_ids = Vec::with_capacity(batch_size);

        for req in batch.requests() {
            request_ids.push(req.request_id().clone());

            match mode {
                ForwardMode::Prefill => {
                    let uncached = req.uncached_input_ids();
                    let prefix_len = req.forward_prefix_token_count();
                    let total_seq_len = req.input_ids().len();
                    context_lens.push(total_seq_len);

                    for (offset, &token_id) in uncached.iter().enumerate() {
                        input_ids.push(token_id);
                        positions.push(prefix_len + offset);
                    }

                    for slot in req.forward_cache_pages() {
                        slot_mapping.push(slot.as_usize());
                    }
                }
                ForwardMode::Decode => {
                    let last_token = req
                        .output_ids()
                        .last()
                        .copied()
                        .or_else(|| req.input_ids().last().copied())
                        .unwrap_or(0);

                    let seq_len = req.input_ids().len() + req.output_ids().len();
                    context_lens.push(seq_len);
                    input_ids.push(last_token);
                    positions.push(seq_len.saturating_sub(1));

                    if let Some(slot) = req.forward_cache_pages().first() {
                        slot_mapping.push(slot.as_usize());
                    } else {
                        slot_mapping.push(0);
                    }
                }
                ForwardMode::Mixed => {
                    if req.stage() == crate::scheduler::RequestStage::DecodeForward
                        || !req.output_ids().is_empty()
                    {
                        let last_token = req
                            .output_ids()
                            .last()
                            .copied()
                            .or_else(|| req.input_ids().last().copied())
                            .unwrap_or(0);

                        let seq_len = req.input_ids().len() + req.output_ids().len();
                        context_lens.push(seq_len);
                        input_ids.push(last_token);
                        positions.push(seq_len.saturating_sub(1));

                        if let Some(slot) = req.forward_cache_pages().first() {
                            slot_mapping.push(slot.as_usize());
                        } else {
                            slot_mapping.push(0);
                        }
                    } else {
                        let uncached = req.uncached_input_ids();
                        let prefix_len = req.forward_prefix_token_count();
                        let total_seq_len = req.input_ids().len();
                        context_lens.push(total_seq_len);

                        for (offset, &token_id) in uncached.iter().enumerate() {
                            input_ids.push(token_id);
                            positions.push(prefix_len + offset);
                        }

                        for slot in req.forward_cache_pages() {
                            slot_mapping.push(slot.as_usize());
                        }
                    }
                }
            }

            // Extract unique physical block page indices
            let mut unique_pages = BTreeSet::new();
            for slot in req.sequence_cache_pages() {
                unique_pages.insert(slot.page_index(page_size));
            }
            raw_block_tables.push(unique_pages.into_iter().collect::<Vec<_>>());
        }

        // Align block tables to uniform 2D layout with padding
        let max_blocks = raw_block_tables
            .iter()
            .map(|t| t.len())
            .max()
            .unwrap_or(1)
            .max(1);

        let mut block_tables = Vec::with_capacity(batch_size);
        for mut table in raw_block_tables {
            table.resize(max_blocks, 0);
            block_tables.push(table);
        }

        let total_tokens = input_ids.len();

        Ok(Self {
            forward_mode: mode,
            batch_size,
            total_tokens,
            input_ids,
            positions,
            context_lens,
            slot_mapping,
            block_tables,
            request_ids,
            schedule_batch: Some(batch.clone()),
        })
    }

    /// Pads this descriptor up to `bucket_size` for CUDA Graph static replay.
    pub fn pad_to_bucket(
        &self,
        bucket_size: usize,
        pad_token_id: u32,
    ) -> Result<Self, GraphRunnerError> {
        if bucket_size < self.batch_size {
            return Err(GraphRunnerError::BucketPaddingOverflow {
                requested: self.batch_size,
                bucket: bucket_size,
            });
        }
        if bucket_size == self.batch_size {
            return Ok(self.clone());
        }

        let pad_count = bucket_size - self.batch_size;
        let mut padded = self.clone();
        padded.batch_size = bucket_size;
        padded.schedule_batch = None;

        let max_blocks = padded.block_tables.first().map(|t| t.len()).unwrap_or(1);

        for i in 0..pad_count {
            padded.input_ids.push(pad_token_id);
            padded.positions.push(0);
            padded.context_lens.push(1);
            padded.slot_mapping.push(0);
            padded.block_tables.push(vec![0; max_blocks]);
            padded
                .request_ids
                .push(RequestId::from(format!("__pad_{i}").as_str()));
        }

        padded.total_tokens = padded.input_ids.len();
        Ok(padded)
    }

    pub fn forward_mode(&self) -> ForwardMode {
        self.forward_mode
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    pub fn input_ids(&self) -> &[u32] {
        &self.input_ids
    }

    pub fn positions(&self) -> &[usize] {
        &self.positions
    }

    pub fn context_lens(&self) -> &[usize] {
        &self.context_lens
    }

    pub fn slot_mapping(&self) -> &[usize] {
        &self.slot_mapping
    }

    pub fn block_tables(&self) -> &[Vec<usize>] {
        &self.block_tables
    }

    pub fn request_ids(&self) -> &[RequestId] {
        &self.request_ids
    }

    pub fn schedule_batch(&self) -> Option<&ScheduleBatch> {
        self.schedule_batch.as_ref()
    }
}

/// Adapter bridging any existing model implementing `ForwardModel` into the `GraphModelRunner`.
pub struct GraphModelAdapter<M> {
    inner: M,
}

impl<M> GraphModelAdapter<M> {
    pub fn new(inner: M) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &M {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.inner
    }

    pub fn into_inner(self) -> M {
        self.inner
    }
}

impl<M> GraphForwardModel for GraphModelAdapter<M>
where
    M: crate::model_executor::ForwardModel + Send + 'static,
{
    fn forward_eager(
        &mut self,
        descriptor: &ModelInputDescriptor,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        let worker_batch = if let Some(batch) = descriptor.schedule_batch() {
            crate::model_executor::ModelWorkerBatch::from_schedule_batch(batch)
        } else {
            let reqs: Vec<ScheduledRequest> = descriptor
                .request_ids()
                .iter()
                .zip(descriptor.input_ids())
                .map(|(id, &token)| {
                    ScheduledRequest::new(
                        id.clone(),
                        vec![token],
                        crate::types::SamplingParams::new(1),
                    )
                })
                .collect();
            let batch = match descriptor.forward_mode() {
                ForwardMode::Prefill => ScheduleBatch::prefill(reqs),
                ForwardMode::Decode => ScheduleBatch::decode(reqs),
                ForwardMode::Mixed => ScheduleBatch::mixed(reqs),
            };
            crate::model_executor::ModelWorkerBatch::from_schedule_batch(&batch)
        };
        self.inner.forward(&worker_batch)
    }

    fn complete_request(&mut self, request_id: &RequestId) {
        self.inner.complete_request(request_id);
    }

    fn update_weights_from_disk(
        &mut self,
        request: &WorkerWeightUpdateRequest,
    ) -> Result<(), WorkerExecutionError> {
        self.inner
            .update_weights_from_disk(request)
            .map_err(|e| WorkerExecutionError::Runtime(e.to_string()))
    }
}

/// Abstract contract for models executed under `GraphModelRunner`.
pub trait GraphForwardModel: Send + 'static {
    /// Eager execution path (used for Prefill, Chunked-Prefill, or batches > max bucket).
    fn forward_eager(
        &mut self,
        descriptor: &ModelInputDescriptor,
    ) -> Result<ModelForwardOutput, ModelForwardError>;

    /// Captured CUDA Graph replay path (used for discrete decode batch buckets).
    fn forward_graph(
        &mut self,
        _bucket_size: usize,
        descriptor: &ModelInputDescriptor,
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        self.forward_eager(descriptor)
    }

    fn complete_request(&mut self, _request_id: &RequestId) {}

    fn update_weights_from_disk(
        &mut self,
        _request: &WorkerWeightUpdateRequest,
    ) -> Result<(), WorkerExecutionError> {
        Ok(())
    }
}

/// High-performance ModelRunner implementing a dual-path execution strategy:
/// Bucketized CUDA Graph replay for microsecond decode + Eager fallback for prefill.
pub struct GraphModelRunner<M, S> {
    model: M,
    sampler: S,
    bucket_table: BatchBucketTable,
    page_size: usize,
    pad_token_id: u32,
}

impl<M, S> GraphModelRunner<M, S> {
    pub fn new(model: M, sampler: S, page_size: usize) -> Self {
        Self {
            model,
            sampler,
            bucket_table: BatchBucketTable::default(),
            page_size,
            pad_token_id: 0,
        }
    }

    pub fn with_bucket_table(mut self, bucket_table: BatchBucketTable) -> Self {
        self.bucket_table = bucket_table;
        self
    }

    pub fn with_pad_token_id(mut self, pad_token_id: u32) -> Self {
        self.pad_token_id = pad_token_id;
        self
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    pub fn sampler(&self) -> &S {
        &self.sampler
    }

    pub fn sampler_mut(&mut self) -> &mut S {
        &mut self.sampler
    }

    pub fn bucket_table(&self) -> &BatchBucketTable {
        &self.bucket_table
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }
}

impl<M, S> FallibleModelWorker for GraphModelRunner<M, S>
where
    M: GraphForwardModel,
    S: LogitSamplerLike,
{
    fn try_generate_batch(
        &mut self,
        batch: &ScheduleBatch,
    ) -> Result<BatchGeneratedTokens, WorkerExecutionError> {
        if batch.batch_size() == 0 {
            return Err(WorkerExecutionError::Runtime(
                "cannot execute empty schedule batch".to_string(),
            ));
        }

        let input_descriptor = ModelInputDescriptor::from_schedule_batch(batch, self.page_size)
            .map_err(|e| WorkerExecutionError::Runtime(e.to_string()))?;

        let original_batch_size = batch.batch_size();

        // 1. Select execution path: CUDA Graph for Decode with matching bucket; Eager otherwise
        let forward_output = if batch.forward_mode() == ForwardMode::Decode
            && let Some(bucket_size) = self.bucket_table.select_bucket(original_batch_size)
        {
            let padded_descriptor = if input_descriptor.batch_size() < bucket_size {
                input_descriptor
                    .pad_to_bucket(bucket_size, self.pad_token_id)
                    .map_err(|e| WorkerExecutionError::Runtime(e.to_string()))?
            } else {
                input_descriptor
            };

            let full_output = self
                .model
                .forward_graph(bucket_size, &padded_descriptor)
                .map_err(|e| WorkerExecutionError::Runtime(e.to_string()))?;

            // Truncate padded outputs back to original batch size
            full_output
                .truncate_to_batch_size(original_batch_size)
                .map_err(|e| WorkerExecutionError::Runtime(e.to_string()))?
        } else {
            self.model
                .forward_eager(&input_descriptor)
                .map_err(|e| WorkerExecutionError::Runtime(e.to_string()))?
        };

        // 2. Fused Sampling
        let token_ids = self
            .sampler
            .sample(forward_output, batch)
            .map_err(|e| WorkerExecutionError::Runtime(e.to_string()))?;

        if token_ids.len() != original_batch_size {
            return Err(WorkerExecutionError::Runtime(
                ModelForwardError::BatchSizeMismatch {
                    request_count: original_batch_size,
                    output_count: token_ids.len(),
                }
                .to_string(),
            ));
        }

        // 3. Construct GeneratedToken results with stop token / EOS criteria
        let generated_tokens: Vec<GeneratedToken> = token_ids
            .into_iter()
            .zip(batch.requests())
            .map(|(token_id, request)| {
                if !request.sampling().ignore_eos
                    && request.sampling().stop_token_ids.contains(&token_id)
                {
                    GeneratedToken::finished(vec![token_id])
                } else {
                    GeneratedToken::unfinished(vec![token_id])
                }
            })
            .collect();

        Ok(BatchGeneratedTokens::from_batch(batch, generated_tokens)?)
    }

    fn complete_request(&mut self, request: &ScheduledRequest) {
        self.model.complete_request(request.request_id());
    }

    fn update_weights_from_disk(
        &mut self,
        request: &WorkerWeightUpdateRequest,
    ) -> Result<(), WorkerExecutionError> {
        self.model.update_weights_from_disk(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::CachePageId;
    use crate::model_executor::{LogitSampler, SystemRandomSource};
    use crate::types::SamplingParams;

    struct MockGraphModel {
        eager_count: usize,
        graph_count: usize,
        vocab_size: usize,
    }

    impl MockGraphModel {
        fn new(vocab_size: usize) -> Self {
            Self {
                eager_count: 0,
                graph_count: 0,
                vocab_size,
            }
        }
    }

    impl GraphForwardModel for MockGraphModel {
        fn forward_eager(
            &mut self,
            descriptor: &ModelInputDescriptor,
        ) -> Result<ModelForwardOutput, ModelForwardError> {
            self.eager_count += 1;
            let mut logits = Vec::with_capacity(descriptor.batch_size());
            for _ in 0..descriptor.batch_size() {
                let mut row = vec![0.0f32; self.vocab_size];
                row[42] = 10.0; // Token 42 highest prob
                logits.push(row);
            }
            ModelForwardOutput::new(logits)
        }

        fn forward_graph(
            &mut self,
            bucket_size: usize,
            descriptor: &ModelInputDescriptor,
        ) -> Result<ModelForwardOutput, ModelForwardError> {
            assert_eq!(descriptor.batch_size(), bucket_size);
            self.graph_count += 1;
            let mut logits = Vec::with_capacity(bucket_size);
            for _ in 0..bucket_size {
                let mut row = vec![0.0f32; self.vocab_size];
                row[99] = 10.0; // Token 99 highest prob in graph mode
                logits.push(row);
            }
            ModelForwardOutput::new(logits)
        }
    }

    #[test]
    fn test_bucket_table_selection() {
        let table = BatchBucketTable::default();
        assert_eq!(table.select_bucket(1), Some(1));
        assert_eq!(table.select_bucket(3), Some(4));
        assert_eq!(table.select_bucket(4), Some(4));
        assert_eq!(table.select_bucket(5), Some(8));
        assert_eq!(table.select_bucket(60), Some(64));
        assert_eq!(table.select_bucket(256), Some(256));
        assert_eq!(table.select_bucket(257), None); // Fallback to eager
    }

    #[test]
    fn test_descriptor_padding() {
        let req1 =
            ScheduledRequest::new(RequestId::from("r1"), vec![1, 2, 3], SamplingParams::new(1));
        let req2 =
            ScheduledRequest::new(RequestId::from("r2"), vec![4, 5, 6], SamplingParams::new(1));

        let batch = ScheduleBatch::decode(vec![req1, req2]);
        let desc = ModelInputDescriptor::from_schedule_batch(&batch, 4).unwrap();
        assert_eq!(desc.batch_size(), 2);

        let padded = desc.pad_to_bucket(4, 0).unwrap();
        assert_eq!(padded.batch_size(), 4);
        assert_eq!(padded.input_ids().len(), 4);
        assert_eq!(padded.request_ids()[0], RequestId::from("r1"));
        assert_eq!(padded.request_ids()[1], RequestId::from("r2"));
        assert_eq!(padded.request_ids()[2], RequestId::from("__pad_0"));
        assert_eq!(padded.request_ids()[3], RequestId::from("__pad_1"));
    }

    #[test]
    fn test_graph_runner_eager_and_graph_dispatch() {
        let model = MockGraphModel::new(128);
        let sampler = LogitSampler::new(SystemRandomSource);
        let mut runner = GraphModelRunner::new(model, sampler, 4);

        // 1. Prefill request -> should route to eager
        let mut req_prefill = ScheduledRequest::new(
            RequestId::from("prefill_1"),
            vec![10, 11, 12],
            SamplingParams::new(1),
        );
        req_prefill.set_forward_cache_pages(vec![
            CachePageId::from(0),
            CachePageId::from(1),
            CachePageId::from(2),
        ]);

        let prefill_batch = ScheduleBatch::prefill(vec![req_prefill]);
        let tokens = runner.try_generate_batch(&prefill_batch).unwrap();
        assert_eq!(tokens.tokens().len(), 1);
        assert_eq!(tokens.tokens()[0].token_ids(), &[42]);
        assert_eq!(runner.model().eager_count, 1);
        assert_eq!(runner.model().graph_count, 0);

        // 2. Decode batch with size 3 -> should pad to bucket 4 and route to graph
        let mut req_d1 =
            ScheduledRequest::new(RequestId::from("d1"), vec![1], SamplingParams::new(1));
        req_d1.set_forward_cache_pages(vec![CachePageId::from(4)]);
        let mut req_d2 =
            ScheduledRequest::new(RequestId::from("d2"), vec![2], SamplingParams::new(1));
        req_d2.set_forward_cache_pages(vec![CachePageId::from(5)]);
        let mut req_d3 =
            ScheduledRequest::new(RequestId::from("d3"), vec![3], SamplingParams::new(1));
        req_d3.set_forward_cache_pages(vec![CachePageId::from(6)]);

        let decode_batch = ScheduleBatch::decode(vec![req_d1, req_d2, req_d3]);
        let decode_tokens = runner.try_generate_batch(&decode_batch).unwrap();

        // Exactly 3 tokens returned (padded 4th discarded)
        assert_eq!(decode_tokens.tokens().len(), 3);
        assert_eq!(decode_tokens.tokens()[0].token_ids(), &[99]);
        assert_eq!(decode_tokens.tokens()[1].token_ids(), &[99]);
        assert_eq!(decode_tokens.tokens()[2].token_ids(), &[99]);

        assert_eq!(runner.model().graph_count, 1);
        assert_eq!(runner.model().eager_count, 1);
    }

    struct LegacyForwardModel;

    impl crate::model_executor::ForwardModel for LegacyForwardModel {
        fn forward(
            &mut self,
            batch: &crate::model_executor::ModelWorkerBatch,
        ) -> Result<ModelForwardOutput, ModelForwardError> {
            let mut logits = Vec::with_capacity(batch.request_ids().len());
            for _ in 0..batch.request_ids().len() {
                let mut row = vec![0.0f32; 64];
                row[7] = 5.0; // Token 7
                logits.push(row);
            }
            ModelForwardOutput::new(logits)
        }
    }

    #[test]
    fn test_graph_model_adapter() {
        let adapter = GraphModelAdapter::new(LegacyForwardModel);
        let sampler = LogitSampler::new(SystemRandomSource);
        let mut runner = GraphModelRunner::new(adapter, sampler, 4);

        let req = ScheduledRequest::new(
            RequestId::from("legacy_1"),
            vec![1, 2],
            SamplingParams::new(1),
        );
        let batch = ScheduleBatch::prefill(vec![req]);
        let tokens = runner.try_generate_batch(&batch).unwrap();
        assert_eq!(tokens.tokens().len(), 1);
        assert_eq!(tokens.tokens()[0].token_ids(), &[7]);
    }
}
