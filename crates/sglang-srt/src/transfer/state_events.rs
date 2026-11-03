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

/// Composite publisher that broadcasts to multiple inner publishers.
#[derive(Clone, Default)]
pub struct CompositeKvStateEventPublisher {
    publishers: Vec<Arc<dyn KvStateEventPublisher>>,
}

impl CompositeKvStateEventPublisher {
    pub fn new(publishers: Vec<Arc<dyn KvStateEventPublisher>>) -> Self {
        Self { publishers }
    }

    pub fn push(&mut self, publisher: Arc<dyn KvStateEventPublisher>) {
        self.publishers.push(publisher);
    }
}

impl KvStateEventPublisher for CompositeKvStateEventPublisher {
    fn publish_block_stored(&self, event: &BlockStoredEvent) {
        for p in &self.publishers {
            p.publish_block_stored(event);
        }
    }

    fn publish_block_removed(&self, event: &BlockRemovedEvent) {
        for p in &self.publishers {
            p.publish_block_removed(event);
        }
    }

    fn publish_all_blocks_cleared(&self) {
        for p in &self.publishers {
            p.publish_all_blocks_cleared();
        }
    }
}

#[derive(Clone, Debug)]
enum ZmqPublishCommand {
    BlockStored(BlockStoredEvent),
    BlockRemoved(BlockRemovedEvent),
    AllBlocksCleared,
}

/// ZMQ-based KV cache state event publisher that broadcasts events to subscribers
/// (such as SGLang router Locus `HashTree`) over a ZeroMQ PUB socket.
pub struct ZmqKvStateEventPublisher {
    sender: tokio::sync::mpsc::UnboundedSender<ZmqPublishCommand>,
    endpoint: String,
    port: u16,
    topic: String,
}

impl ZmqKvStateEventPublisher {
    pub async fn bind(host: &str, port: u16, topic: impl Into<String>) -> Result<Self, String> {
        use zeromq::{Endpoint, PubSocket, Socket, SocketSend, ZmqMessage};

        let mut socket = PubSocket::new();
        let bind_addr = format!("tcp://{host}:{port}");
        let endpoint = socket
            .bind(&bind_addr)
            .await
            .map_err(|e| format!("failed to bind ZMQ PUB socket on {bind_addr}: {e}"))?;
        let bound_port = match endpoint {
            Endpoint::Tcp(_, p) => p,
            _ => port,
        };
        let topic = topic.into();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ZmqPublishCommand>();
        let task_topic = topic.clone();

        tokio::spawn(async move {
            let mut seq: i64 = 1;
            while let Some(cmd) = rx.recv().await {
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0);
                let payload = match cmd {
                    ZmqPublishCommand::BlockStored(ev) => {
                        let dp = if ev.data_parallel_rank >= 0 {
                            Some(ev.data_parallel_rank as u32)
                        } else {
                            None
                        };
                        encode_single_event_batch(ts, &encode_block_stored(&ev), dp)
                    }
                    ZmqPublishCommand::BlockRemoved(ev) => {
                        let dp = if ev.data_parallel_rank >= 0 {
                            Some(ev.data_parallel_rank as u32)
                        } else {
                            None
                        };
                        encode_single_event_batch(ts, &encode_block_removed(&ev), dp)
                    }
                    ZmqPublishCommand::AllBlocksCleared => {
                        encode_single_event_batch(ts, &encode_all_blocks_cleared(), None)
                    }
                };
                let mut msg = ZmqMessage::from(task_topic.as_bytes().to_vec());
                msg.push_back(seq.to_be_bytes().to_vec().into());
                msg.push_back(payload.into());
                if socket.send(msg).await.is_err() {
                    break;
                }
                seq = seq.wrapping_add(1);
            }
            // Send shutdown sentinel END_SEQ (-1)
            let mut msg = ZmqMessage::from(task_topic.as_bytes().to_vec());
            msg.push_back((-1_i64).to_be_bytes().to_vec().into());
            msg.push_back(Vec::new().into());
            let _ = socket.send(msg).await;
        });

        Ok(Self {
            sender: tx,
            endpoint: format!("tcp://{host}:{bound_port}"),
            port: bound_port,
            topic,
        })
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn topic(&self) -> &str {
        &self.topic
    }
}

impl KvStateEventPublisher for ZmqKvStateEventPublisher {
    fn publish_block_stored(&self, event: &BlockStoredEvent) {
        let _ = self
            .sender
            .send(ZmqPublishCommand::BlockStored(event.clone()));
    }

    fn publish_block_removed(&self, event: &BlockRemovedEvent) {
        let _ = self
            .sender
            .send(ZmqPublishCommand::BlockRemoved(event.clone()));
    }

    fn publish_all_blocks_cleared(&self) {
        let _ = self.sender.send(ZmqPublishCommand::AllBlocksCleared);
    }
}

fn encode_block_stored(event: &BlockStoredEvent) -> Vec<u8> {
    use rmp::encode as mp;

    let mut buf = Vec::new();
    mp::write_array_len(&mut buf, 7).expect("rmp write");
    mp::write_str(&mut buf, "BlockStored").expect("rmp write");
    mp::write_array_len(&mut buf, event.block_hashes.len() as u32).expect("rmp write");
    for hash in &event.block_hashes {
        mp::write_sint(&mut buf, *hash).expect("rmp write");
    }
    match event.parent_block_hash {
        Some(h) => {
            let _ = mp::write_sint(&mut buf, h).expect("rmp write");
        }
        None => {
            mp::write_nil(&mut buf).expect("rmp write");
        }
    }
    mp::write_array_len(&mut buf, event.token_ids.len() as u32).expect("rmp write");
    for id in &event.token_ids {
        mp::write_uint(&mut buf, *id as u64).expect("rmp write");
    }
    mp::write_uint(&mut buf, event.block_size as u64).expect("rmp write");
    mp::write_nil(&mut buf).expect("rmp write");
    let medium = if event.boundary.tier.is_empty() {
        "GPU"
    } else {
        &event.boundary.tier
    };
    mp::write_str(&mut buf, medium).expect("rmp write");
    buf
}

fn encode_block_removed(event: &BlockRemovedEvent) -> Vec<u8> {
    use rmp::encode as mp;

    let mut buf = Vec::new();
    mp::write_array_len(&mut buf, 3).expect("rmp write");
    mp::write_str(&mut buf, "BlockRemoved").expect("rmp write");
    mp::write_array_len(&mut buf, event.block_hashes.len() as u32).expect("rmp write");
    for hash in &event.block_hashes {
        mp::write_sint(&mut buf, *hash).expect("rmp write");
    }
    mp::write_nil(&mut buf).expect("rmp write");
    buf
}

fn encode_all_blocks_cleared() -> Vec<u8> {
    use rmp::encode as mp;

    let mut buf = Vec::new();
    mp::write_array_len(&mut buf, 1).expect("rmp write");
    mp::write_str(&mut buf, "AllBlocksCleared").expect("rmp write");
    buf
}

fn encode_single_event_batch(ts: f64, event_bytes: &[u8], attn_dp_rank: Option<u32>) -> Vec<u8> {
    use rmp::encode as mp;

    let mut buf = Vec::new();
    let total_len = if attn_dp_rank.is_some() { 3 } else { 2 };
    mp::write_array_len(&mut buf, total_len).expect("rmp write");
    mp::write_f64(&mut buf, ts).expect("rmp write");
    mp::write_array_len(&mut buf, 1).expect("rmp write");
    buf.extend_from_slice(event_bytes);
    if let Some(rank) = attn_dp_rank {
        mp::write_uint(&mut buf, rank as u64).expect("rmp write");
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeromq::{Socket, SocketRecv, SubSocket};

    #[tokio::test]
    async fn zmq_publisher_broadcasts_block_stored_events_to_subscriber() {
        let publisher = ZmqKvStateEventPublisher::bind("127.0.0.1", 0, "kv")
            .await
            .expect("bind zmq publisher");
        let port = publisher.port();
        assert!(port > 0);

        let mut sub = SubSocket::new();
        sub.connect(&format!("tcp://127.0.0.1:{port}"))
            .await
            .expect("connect subscriber");
        sub.subscribe("kv").await.expect("subscribe to topic");

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let event = BlockStoredEvent {
            request_id: RequestId::from("test-req"),
            block_hashes: vec![12345, 67890],
            parent_block_hash: Some(11111),
            token_ids: vec![10, 20, 30],
            block_size: 16,
            data_parallel_rank: 0,
            prompt_tokens_processed: 3,
            total_prompt_tokens: 3,
            is_final_chunk: true,
            boundary: ReusableBoundary::new(0, 3, "GPU"),
        };

        publisher.publish_block_stored(&event);

        let msg = tokio::time::timeout(std::time::Duration::from_secs(2), sub.recv())
            .await
            .expect("recv timeout")
            .expect("recv msg");

        assert_eq!(msg.len(), 3);
        assert_eq!(msg.get(0).unwrap().as_ref(), b"kv");
        let seq_bytes: [u8; 8] = msg.get(1).unwrap().as_ref().try_into().unwrap();
        assert_eq!(i64::from_be_bytes(seq_bytes), 1);
        let payload = msg.get(2).unwrap();
        assert!(!payload.is_empty());
    }
}
