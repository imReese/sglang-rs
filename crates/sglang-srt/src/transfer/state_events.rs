// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Incremental KV cache state event stream aligned with NexusKV and Locus.
//!
//! When chunked prefill materializes chunks of tokens into physical memory,
//! events are published to notify external routers (e.g., Locus) and state
//! registries (e.g., NexusKV) of newly valid `ReusableBoundary` slices.

use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

use crate::cache::compute_sglang_block_hashes;
use crate::types::RequestId;

/// Describes a contiguous slice of materialized tokens available for prefix reuse,
/// directly matching NexusKV's `ReusableSlice` and Locus's state boundaries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReusableBoundary {
    pub start_token: usize,
    pub end_token: usize,
    pub tier: String,
}

impl ReusableBoundary {
    pub fn new(start_token: usize, end_token: usize, tier: impl Into<String>) -> Self {
        Self {
            start_token,
            end_token,
            tier: tier.into(),
        }
    }

    pub fn token_count(&self) -> usize {
        self.end_token.saturating_sub(self.start_token)
    }
}

/// Event emitted when a chunk or block of KV cache has been computed and materialized
/// into physical device memory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockStoredEvent {
    pub request_id: RequestId,
    pub block_hashes: Vec<i64>,
    pub parent_block_hash: Option<i64>,
    pub token_ids: Vec<u32>,
    pub block_size: usize,
    pub data_parallel_rank: i32,
    pub prompt_tokens_processed: usize,
    pub total_prompt_tokens: usize,
    pub is_final_chunk: bool,
    pub boundary: ReusableBoundary,
}

impl BlockStoredEvent {
    pub fn from_chunk(
        request_id: RequestId,
        full_prompt: &[u32],
        chunk_range: std::ops::Range<usize>,
        block_size: usize,
        data_parallel_rank: i32,
    ) -> Self {
        assert!(block_size > 0, "block_size must be positive");
        let start = chunk_range.start;
        let end = chunk_range.end.min(full_prompt.len());
        let chunk_tokens = full_prompt[start..end].to_vec();
        let total_prompt_tokens = full_prompt.len();
        let is_final_chunk = end >= total_prompt_tokens;

        // Compute hash chain up to the end of this chunk
        let prefix_hashes = compute_sglang_block_hashes(&full_prompt[..end], block_size);
        let start_block_idx = start / block_size;
        let parent_block_hash = if start_block_idx > 0 {
            prefix_hashes.get(start_block_idx - 1).copied()
        } else {
            None
        };
        let block_hashes = prefix_hashes[start_block_idx..].to_vec();

        Self {
            request_id,
            block_hashes,
            parent_block_hash,
            token_ids: chunk_tokens,
            block_size,
            data_parallel_rank,
            prompt_tokens_processed: end,
            total_prompt_tokens,
            is_final_chunk,
            boundary: ReusableBoundary::new(start, end, "GPU"),
        }
    }
}

/// Event emitted when cache blocks are removed/evicted from memory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockRemovedEvent {
    pub block_hashes: Vec<i64>,
    pub data_parallel_rank: i32,
}

/// Unified trait for KV cache state event publication.
pub trait KvStateEventPublisher: Send + Sync {
    fn publish_block_stored(&self, event: &BlockStoredEvent);
    fn publish_block_removed(&self, event: &BlockRemovedEvent);
    fn publish_all_blocks_cleared(&self);
}

/// No-op implementation of [`KvStateEventPublisher`].
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopKvStateEventPublisher;

impl KvStateEventPublisher for NoopKvStateEventPublisher {
    fn publish_block_stored(&self, _event: &BlockStoredEvent) {}
    fn publish_block_removed(&self, _event: &BlockRemovedEvent) {}
    fn publish_all_blocks_cleared(&self) {}
}

/// Thread-safe in-memory recorder for KV state events (useful for tests and metrics).
#[derive(Clone, Debug, Default)]
pub struct RecordingKvStateEventPublisher {
    stored_events: Arc<Mutex<Vec<BlockStoredEvent>>>,
    removed_events: Arc<Mutex<Vec<BlockRemovedEvent>>>,
    cleared_count: Arc<Mutex<usize>>,
}

impl RecordingKvStateEventPublisher {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stored_events(&self) -> Vec<BlockStoredEvent> {
        self.stored_events.lock().unwrap().clone()
    }

    pub fn removed_events(&self) -> Vec<BlockRemovedEvent> {
        self.removed_events.lock().unwrap().clone()
    }

    pub fn cleared_count(&self) -> usize {
        *self.cleared_count.lock().unwrap()
    }

    pub fn clear(&self) {
        self.stored_events.lock().unwrap().clear();
        self.removed_events.lock().unwrap().clear();
        *self.cleared_count.lock().unwrap() = 0;
    }
}

impl KvStateEventPublisher for RecordingKvStateEventPublisher {
    fn publish_block_stored(&self, event: &BlockStoredEvent) {
        self.stored_events.lock().unwrap().push(event.clone());
    }

    fn publish_block_removed(&self, event: &BlockRemovedEvent) {
        self.removed_events.lock().unwrap().push(event.clone());
    }

    fn publish_all_blocks_cleared(&self) {
        *self.cleared_count.lock().unwrap() += 1;
    }
}

/// Broadcast publisher forwarding state events over a tokio broadcast channel.
#[derive(Clone, Debug)]
pub struct BroadcastKvStateEventPublisher {
    sender: broadcast::Sender<BlockStoredEvent>,
}

impl BroadcastKvStateEventPublisher {
    pub fn new(capacity: usize) -> (Self, broadcast::Receiver<BlockStoredEvent>) {
        let (sender, receiver) = broadcast::channel(capacity);
        (Self { sender }, receiver)
    }

    pub fn subscribe(&self) -> broadcast::Receiver<BlockStoredEvent> {
        self.sender.subscribe()
    }
}

impl KvStateEventPublisher for BroadcastKvStateEventPublisher {
    fn publish_block_stored(&self, event: &BlockStoredEvent) {
        let _ = self.sender.send(event.clone());
    }

    fn publish_block_removed(&self, _event: &BlockRemovedEvent) {}

    fn publish_all_blocks_cleared(&self) {}
}
