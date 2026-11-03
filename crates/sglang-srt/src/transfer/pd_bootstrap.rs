use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::future::Future;
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;
use zeromq::{PullSocket, PushSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

use crate::pd_control::{PdDecodeReceiver, completion_frames, send_frames};
use crate::transfer::{
    DecodeBootstrapMetadataPublishSummary, DecodeBootstrapPublisher, KvBatchReleaser,
    KvCacheTransferError, KvCacheTransferExecutor, KvCacheTransferPlan, KvCacheTransferSpan,
    KvPoll, KvRemoteLayout, KvTransferBackend, KvTransferLayout, KvTransferPoll,
    KvTransferStatusReader, KvTransferSubmitter, KvTransferTargetResolver,
    TransferableKvCacheMemory,
};
use crate::types::BootstrapRoom;

const DP_RANK_ENTRY_CLEANUP_INTERVAL_SECS: u64 = 120;

#[derive(Clone, Debug, Default)]
pub struct PrefillBootstrapService {
    state: Arc<Mutex<PrefillBootstrapState>>,
}

impl PrefillBootstrapService {
    pub fn state(&self) -> &Arc<Mutex<PrefillBootstrapState>> {
        &self.state
    }

    fn into_router(self) -> Router {
        Router::new()
            .route("/health", get(health))
            .route("/route", get(route_get).put(route_put))
            .route("/register_dp_rank", post(register_dp_rank))
            .route("/query_dp_ranks", post(query_dp_ranks))
            .with_state(self)
    }
}

pub struct PdBootstrapKvCacheTransferExecutor<E> {
    bootstrap_service: PrefillBootstrapService,
    inner: E,
    metadata_wait_timeout: Duration,
    metadata_poll_interval: Duration,
    pending_tokens: BTreeMap<BootstrapRoom, u32>,
    pending_states: BTreeMap<BootstrapRoom, Vec<u8>>,
}

pub type MooncakeBootstrapKvCacheTransferExecutor<E> = PdBootstrapKvCacheTransferExecutor<E>;

impl<E> PdBootstrapKvCacheTransferExecutor<E> {
    const DEFAULT_METADATA_WAIT_TIMEOUT: Duration = Duration::from_secs(5);
    const DEFAULT_METADATA_POLL_INTERVAL: Duration = Duration::from_millis(2);

    pub fn new(bootstrap_service: PrefillBootstrapService, inner: E) -> Self {
        Self {
            bootstrap_service,
            inner,
            metadata_wait_timeout: Self::DEFAULT_METADATA_WAIT_TIMEOUT,
            metadata_poll_interval: Self::DEFAULT_METADATA_POLL_INTERVAL,
            pending_tokens: BTreeMap::new(),
            pending_states: BTreeMap::new(),
        }
    }

    pub fn bootstrap_service(&self) -> &PrefillBootstrapService {
        &self.bootstrap_service
    }

    pub fn inner(&self) -> &E {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut E {
        &mut self.inner
    }

    pub fn with_metadata_wait_timeout(mut self, timeout: Duration) -> Self {
        self.metadata_wait_timeout = timeout;
        self
    }

    fn remote_kv_layouts_for_room(
        &self,
        room: BootstrapRoom,
    ) -> Result<Vec<(String, KvRemoteLayout)>, KvCacheTransferError> {
        let deadline = Instant::now() + self.metadata_wait_timeout;
        loop {
            let result = {
                let state = self
                    .bootstrap_service
                    .state()
                    .lock()
                    .expect("prefill bootstrap state lock should be held");
                state.remote_kv_layouts_for_room(room)
            };

            match result {
                Ok(layouts) => return Ok(layouts),
                Err(error) if Instant::now() >= deadline => {
                    return Err(KvCacheTransferError::Runtime(error.to_string()));
                }
                Err(_) => std::thread::sleep(self.metadata_poll_interval),
            }
        }
    }

    fn register_span_dp_rank(&self, span: &KvCacheTransferSpan) {
        self.bootstrap_service
            .state()
            .lock()
            .expect("prefill bootstrap state lock should be held")
            .register_dp_rank(span.bootstrap_room(), span.data_parallel_rank());
    }
}

impl<S, R> KvTransferBackend for PdBootstrapKvCacheTransferExecutor<KvCacheTransferExecutor<S, R>>
where
    S: KvTransferSubmitter + KvTransferStatusReader + KvBatchReleaser,
    R: KvTransferTargetResolver,
{
    fn register(&mut self, memory: TransferableKvCacheMemory) -> Result<(), KvCacheTransferError> {
        self.inner.register(memory)
    }

    fn submit(
        &mut self,
        span: &crate::transfer::KvCacheTransferSpan,
    ) -> Result<(), KvCacheTransferError> {
        self.register_span_dp_rank(span);
        let remote_layouts = self.remote_kv_layouts_for_room(span.bootstrap_room())?;
        {
            let mut state = self
                .bootstrap_service
                .state
                .lock()
                .map_err(|e| KvCacheTransferError::Runtime(e.to_string()))?;
            if state.aborted_rooms.contains(&span.bootstrap_room()) {
                return Err(KvCacheTransferError::Runtime(
                    "decode aborted this transfer room before submission".into(),
                ));
            }
            state.active_rooms.insert(span.bootstrap_room());
        }
        for (session_id, layout) in remote_layouts {
            self.inner
                .insert_remote_kv_session_layout(span.bootstrap_room(), session_id, layout);
        }
        let result = self.inner.submit(span);
        if let Some(token) = span.first_token() {
            self.pending_tokens.insert(span.bootstrap_room(), token);
        }
        if let Some(state) = span.recurrent_state() {
            self.pending_states
                .insert(span.bootstrap_room(), state.to_vec());
        }
        if let Err(error) = &result {
            self.inner.cancel(span.bootstrap_room())?;
            self.notify_room(span.bootstrap_room(), Some(&error.to_string()))?;
        }
        result
    }

    fn poll(&mut self) -> Result<KvTransferPoll, KvCacheTransferError> {
        let aborted = {
            let state = self
                .bootstrap_service
                .state
                .lock()
                .map_err(|e| KvCacheTransferError::Runtime(e.to_string()))?;
            state
                .active_rooms
                .intersection(&state.aborted_rooms)
                .copied()
                .collect::<Vec<_>>()
        };
        let mut canceled = Vec::new();
        for room in aborted {
            self.inner.cancel(room)?;
            self.notify_room(room, Some("decode aborted the transfer"))?;
            canceled.push(crate::transfer::KvTransferCompletionEvent::canceled(
                room,
                String::new(),
            ));
        }
        let poll = self.inner.poll()?;
        for event in poll.events() {
            let error = match event.status() {
                crate::transfer::KvTransferEventStatus::Completed => None,
                crate::transfer::KvTransferEventStatus::Failed(message) => Some(message.as_str()),
                crate::transfer::KvTransferEventStatus::Canceled => {
                    Some("prefill transfer canceled")
                }
            };
            self.notify_room(event.bootstrap_room(), error)?;
        }
        let mut events = poll.events().to_vec();
        events.extend(canceled);
        Ok(KvTransferPoll::new(events, poll.pending_transfers()))
    }

    fn cancel(&mut self, bootstrap_room: BootstrapRoom) -> Result<(), KvCacheTransferError> {
        self.inner.cancel(bootstrap_room)?;
        self.notify_room(bootstrap_room, Some("prefill request canceled"))
    }

    fn shutdown(&mut self) -> Result<(), KvCacheTransferError> {
        for room in self.pending_tokens.keys().copied().collect::<Vec<_>>() {
            self.cancel(room)?;
        }
        self.inner.shutdown()
    }
}

impl<S, R> PdBootstrapKvCacheTransferExecutor<KvCacheTransferExecutor<S, R>>
where
    S: KvTransferStatusReader + KvBatchReleaser,
{
    fn notify_room(
        &mut self,
        room: BootstrapRoom,
        error: Option<&str>,
    ) -> Result<(), KvCacheTransferError> {
        let (targets, abort_targets) = {
            let state = self
                .bootstrap_service
                .state
                .lock()
                .map_err(|e| KvCacheTransferError::Runtime(e.to_string()))?;
            let targets = state
                .transfer_rooms
                .get(&room)
                .map(|state| {
                    state
                        .transfers
                        .values()
                        .map(|transfer| {
                            (
                                transfer.endpoint.clone(),
                                transfer.dst_port,
                                transfer.dst_aux_index.unwrap_or(0),
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            (
                targets,
                state.abort_targets.get(&room).cloned().unwrap_or_default(),
            )
        };
        let token = self.pending_tokens.get(&room).copied();
        let recurrent_state = self.pending_states.get(&room).map(|v| v.as_slice());
        for (host, port, aux) in targets {
            if token.is_none() {
                continue;
            }
            let frames = completion_frames(
                room,
                if error.is_none() { token } else { None },
                recurrent_state,
                error,
                aux,
            )
            .map_err(KvCacheTransferError::Runtime)?;
            send_frames(format!("tcp://{host}:{port}"), frames)
                .map_err(KvCacheTransferError::Runtime)?;
        }
        for endpoint in abort_targets {
            send_frames(
                endpoint,
                vec![vec![
                    b"ABORT_ACK".to_vec(),
                    room.to_string().into_bytes(),
                    b"0".to_vec(),
                ]],
            )
            .map_err(KvCacheTransferError::Runtime)?;
        }
        let mut state = self
            .bootstrap_service
            .state
            .lock()
            .map_err(|e| KvCacheTransferError::Runtime(e.to_string()))?;
        state.active_rooms.remove(&room);
        state.abort_targets.remove(&room);
        state.transfer_rooms.remove(&room);
        self.pending_tokens.remove(&room);
        self.pending_states.remove(&room);
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct PrefillBootstrapState {
    attn_tp_size: Option<usize>,
    attn_cp_size: Option<usize>,
    dp_size: Option<usize>,
    pp_size: Option<usize>,
    page_size: Option<usize>,
    kv_cache_dtype: Option<String>,
    follow_bootstrap_room: Option<bool>,
    registered_count: usize,
    prefill_port_table: PrefillPortTable,
    room_to_dp_rank: BTreeMap<BootstrapRoom, RegisteredDpRank>,
    decode_kv_args_table: BTreeMap<String, PdKvArgsRegisterInfo>,
    decode_kv_args_registration_count: usize,
    transfer_rooms: BTreeMap<BootstrapRoom, PdTransferRoom>,
    active_rooms: BTreeSet<BootstrapRoom>,
    aborted_rooms: BTreeSet<BootstrapRoom>,
    abort_targets: BTreeMap<BootstrapRoom, Vec<String>>,
}

type PrefillPpRankTable = BTreeMap<usize, PrefillRankInfo>;
type PrefillTpRankTable = BTreeMap<usize, PrefillPpRankTable>;
type PrefillCpRankTable = BTreeMap<usize, PrefillTpRankTable>;
type PrefillPortTable = BTreeMap<usize, PrefillCpRankTable>;

impl PrefillBootstrapState {
    pub fn ingest_bootstrap_frame(
        &mut self,
        frame: &[Vec<u8>],
    ) -> Result<(), PdBootstrapFrameError> {
        let room = ascii_field(frame, 0, "room")?;
        match room {
            "WATERMARK" | "STAGING_RSP" => Ok(()),
            "None" => {
                let register = PdKvArgsRegisterInfo::from_frame(frame)?;
                self.decode_kv_args_table
                    .insert(register.session_id.clone(), register);
                self.decode_kv_args_registration_count += 1;
                Ok(())
            }
            _ => {
                let transfer = PdTransferInfo::from_frame(frame)?;
                let room = transfer.room;
                let session = transfer.session_id.clone();
                let required_dst_info_num = transfer.required_dst_info_num;
                let decode_prefix_len = transfer.decode_prefix_len;
                let room_state =
                    self.transfer_rooms
                        .entry(room)
                        .or_insert_with(|| PdTransferRoom {
                            required_dst_info_num,
                            status: KvPoll::Bootstrapping,
                            decode_prefix_len: None,
                            transfers: BTreeMap::new(),
                        });
                room_state.required_dst_info_num = required_dst_info_num;
                if room_state.decode_prefix_len.is_none() && decode_prefix_len.is_some() {
                    room_state.decode_prefix_len = decode_prefix_len;
                }
                room_state.transfers.insert(session, transfer);
                if room_state.transfers.len() == room_state.required_dst_info_num {
                    room_state.status = if self.aborted_rooms.contains(&room) {
                        KvPoll::Failed
                    } else {
                        KvPoll::WaitingForInput
                    };
                }
                Ok(())
            }
        }
    }

    pub fn ingest_mooncake_bootstrap_frame(
        &mut self,
        frame: &[Vec<u8>],
    ) -> Result<(), PdBootstrapFrameError> {
        self.ingest_bootstrap_frame(frame)
    }

    pub fn decode_kv_args(&self, session_id: &str) -> Option<&PdKvArgsRegisterInfo> {
        self.decode_kv_args_table.get(session_id)
    }

    pub fn decode_kv_args_registration_count(&self) -> usize {
        self.decode_kv_args_registration_count
    }

    pub fn transfer_room(&self, room: BootstrapRoom) -> Option<&PdTransferRoom> {
        self.transfer_rooms.get(&room)
    }

    pub fn transfer_status(&self, room: BootstrapRoom) -> Option<KvPoll> {
        self.transfer_rooms.get(&room).map(|room| room.status)
    }

    pub fn remote_kv_layouts_for_room(
        &self,
        room: BootstrapRoom,
    ) -> Result<Vec<(String, KvRemoteLayout)>, PdRemoteKvLayoutError> {
        let room_state = self
            .transfer_rooms
            .get(&room)
            .ok_or(PdRemoteKvLayoutError::MissingTransferRoom(room))?;
        let mut layouts = Vec::with_capacity(room_state.transfers.len());

        for transfer in room_state.transfers.values() {
            if transfer.is_dummy {
                continue;
            }

            let kv_args = self
                .decode_kv_args_table
                .get(&transfer.session_id)
                .ok_or_else(|| PdRemoteKvLayoutError::MissingKvArgsRegistration {
                    room,
                    session_id: transfer.session_id.clone(),
                })?;

            layouts.push((
                transfer.session_id.clone(),
                KvRemoteLayout {
                    dst_kv_ptrs: kv_args.dst_kv_ptrs.clone(),
                    dst_kv_indices: transfer.dst_kv_indices.clone(),
                    dst_kv_item_len: kv_args.dst_kv_item_len,
                },
            ));
        }

        Ok(layouts)
    }

    pub(crate) fn register_route(&mut self, registration: PrefillRouteRegistration) {
        self.attn_tp_size.get_or_insert(registration.attn_tp_size);
        self.attn_cp_size.get_or_insert(registration.attn_cp_size);
        self.dp_size
            .get_or_insert(if registration.system_dp_size == 1 {
                registration.attn_dp_size
            } else {
                registration.system_dp_size
            });
        self.pp_size.get_or_insert(registration.pp_size);
        if let Some(page_size) = registration.page_size {
            self.page_size.get_or_insert(page_size);
        }
        if let Some(kv_cache_dtype) = registration.kv_cache_dtype.clone() {
            self.kv_cache_dtype.get_or_insert(kv_cache_dtype);
        }
        self.follow_bootstrap_room.get_or_insert_with(|| {
            registration
                .load_balance_method
                .as_deref()
                .unwrap_or("follow_bootstrap_room")
                == "follow_bootstrap_room"
        });

        let dp_group = if registration.system_dp_size == 1 {
            registration.attn_dp_rank
        } else {
            registration.system_dp_rank
        };
        self.prefill_port_table
            .entry(dp_group)
            .or_default()
            .entry(registration.attn_cp_rank)
            .or_default()
            .entry(registration.attn_tp_rank)
            .or_default()
            .insert(
                registration.pp_rank,
                PrefillRankInfo {
                    rank_ip: registration.rank_ip,
                    rank_port: registration.rank_port,
                },
            );
        self.registered_count += 1;
    }

    fn is_ready(&self) -> bool {
        let (Some(dp_size), Some(attn_cp_size), Some(attn_tp_size), Some(pp_size)) = (
            self.dp_size,
            self.attn_cp_size,
            self.attn_tp_size,
            self.pp_size,
        ) else {
            return false;
        };
        self.registered_count >= dp_size * attn_cp_size * attn_tp_size * pp_size
    }

    pub fn server_info(&self) -> Option<PrefillServerInfo> {
        if !self.is_ready() {
            return None;
        }
        Some(PrefillServerInfo {
            attn_tp_size: self.attn_tp_size?,
            attn_cp_size: self.attn_cp_size?,
            dp_size: self.dp_size?,
            pp_size: self.pp_size?,
            page_size: self.page_size,
            kv_cache_dtype: self.kv_cache_dtype.clone(),
            follow_bootstrap_room: self.follow_bootstrap_room.unwrap_or(true),
        })
    }

    pub fn rank_info(
        &self,
        prefill_dp_rank: usize,
        prefill_cp_rank: usize,
        target_tp_rank: usize,
        target_pp_rank: usize,
    ) -> Option<PrefillRankInfo> {
        self.prefill_port_table
            .get(&prefill_dp_rank)?
            .get(&prefill_cp_rank)?
            .get(&target_tp_rank)?
            .get(&target_pp_rank)
            .cloned()
    }

    fn register_dp_rank(&mut self, bootstrap_room: BootstrapRoom, dp_rank: i32) {
        self.prune_expired_dp_rank_entries(now_secs());
        self.room_to_dp_rank.insert(
            bootstrap_room,
            RegisteredDpRank {
                dp_rank,
                timestamp_secs: now_secs(),
            },
        );
    }

    fn query_dp_ranks(&mut self, bootstrap_rooms: &[BootstrapRoom]) -> serde_json::Value {
        self.prune_expired_dp_rank_entries(now_secs());
        let mut result = serde_json::Map::new();
        for room in bootstrap_rooms {
            if let Some(entry) = self.room_to_dp_rank.get(room) {
                result.insert(room.to_string(), json!(entry.dp_rank));
            }
        }
        serde_json::Value::Object(result)
    }

    fn prune_expired_dp_rank_entries(&mut self, now_secs: u64) -> usize {
        let before = self.room_to_dp_rank.len();
        self.room_to_dp_rank.retain(|_, entry| {
            now_secs.saturating_sub(entry.timestamp_secs) <= DP_RANK_ENTRY_CLEANUP_INTERVAL_SECS
        });
        before.saturating_sub(self.room_to_dp_rank.len())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PrefillServerInfo {
    pub attn_tp_size: usize,
    pub attn_cp_size: usize,
    pub dp_size: usize,
    pub pp_size: usize,
    pub page_size: Option<usize>,
    pub kv_cache_dtype: Option<String>,
    pub follow_bootstrap_room: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PrefillRankInfo {
    pub rank_ip: String,
    pub rank_port: u16,
}

impl PrefillRankInfo {
    pub fn zmq_endpoint(&self) -> String {
        format!("tcp://{}:{}", self.rank_ip, self.rank_port)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PdKvArgsRegisterInfo {
    pub room: String,
    pub endpoint: String,
    pub dst_port: u16,
    pub session_id: String,
    pub dst_kv_ptrs: Vec<u64>,
    pub dst_aux_ptrs: Vec<u64>,
    pub dst_state_data_ptrs: Vec<Vec<u64>>,
    pub dst_tp_rank: i32,
    pub dst_attn_tp_size: usize,
    pub dst_kv_item_len: usize,
    pub dst_state_item_lens: Vec<Vec<u32>>,
    pub dst_state_dim_per_tensor: Vec<Vec<u32>>,
}

pub type MooncakeKvArgsRegisterInfo = PdKvArgsRegisterInfo;

impl PdKvArgsRegisterInfo {
    fn from_frame(frame: &[Vec<u8>]) -> Result<Self, PdBootstrapFrameError> {
        require_min_fields(frame, 10, "KVArgs registration")?;
        Ok(Self {
            room: ascii_field(frame, 0, "room")?.to_string(),
            endpoint: ascii_field(frame, 1, "endpoint")?.to_string(),
            dst_port: parse_u16(frame, 2, "dst_port")?,
            session_id: ascii_field(frame, 3, "session_id")?.to_string(),
            dst_kv_ptrs: unpack_u64s(field(frame, 4, "dst_kv_ptrs")?)?,
            dst_aux_ptrs: unpack_u64s(field(frame, 5, "dst_aux_ptrs")?)?,
            dst_state_data_ptrs: unpack_u64_lists(field(frame, 6, "dst_state_data_ptrs")?)?,
            dst_tp_rank: parse_i32(frame, 7, "dst_tp_rank")?,
            dst_attn_tp_size: parse_usize(frame, 8, "dst_attn_tp_size")?,
            dst_kv_item_len: parse_usize(frame, 9, "dst_kv_item_len")?,
            dst_state_item_lens: optional_u32_lists(frame, 10)?,
            dst_state_dim_per_tensor: optional_u32_lists(frame, 11)?,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PdTransferInfo {
    pub room: BootstrapRoom,
    pub endpoint: String,
    pub dst_port: u16,
    pub session_id: String,
    pub dst_kv_indices: Vec<i32>,
    pub dst_aux_index: Option<i32>,
    pub dst_state_indices: Vec<Vec<i32>>,
    pub required_dst_info_num: usize,
    pub is_dummy: bool,
    pub decode_prefix_len: Option<usize>,
}

pub type MooncakeTransferInfo = PdTransferInfo;

impl PdTransferInfo {
    fn from_frame(frame: &[Vec<u8>]) -> Result<Self, PdBootstrapFrameError> {
        require_min_fields(frame, 8, "transfer metadata")?;
        let is_dummy = field(frame, 4, "dst_kv_indices")?.is_empty()
            && field(frame, 5, "dst_aux_index")?.is_empty();
        let (dst_kv_indices, dst_aux_index, dst_state_indices) = if is_dummy {
            (Vec::new(), None, Vec::new())
        } else {
            (
                unpack_i32s(field(frame, 4, "dst_kv_indices")?)?,
                Some(parse_i32(frame, 5, "dst_aux_index")?),
                unpack_i32_lists(field(frame, 6, "dst_state_indices")?)?,
            )
        };

        Ok(Self {
            room: parse_bootstrap_room(frame, 0, "room")?,
            endpoint: ascii_field(frame, 1, "endpoint")?.to_string(),
            dst_port: parse_u16(frame, 2, "dst_port")?,
            session_id: ascii_field(frame, 3, "session_id")?.to_string(),
            dst_kv_indices,
            dst_aux_index,
            dst_state_indices,
            required_dst_info_num: parse_usize(frame, 7, "required_dst_info_num")?,
            is_dummy,
            decode_prefix_len: optional_usize(frame, 8, "decode_prefix_len")?,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PdTransferRoom {
    pub required_dst_info_num: usize,
    pub status: KvPoll,
    pub decode_prefix_len: Option<usize>,
    pub transfers: BTreeMap<String, PdTransferInfo>,
}

pub type MooncakeTransferRoom = PdTransferRoom;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PdRemoteKvLayoutError {
    MissingTransferRoom(BootstrapRoom),
    MissingKvArgsRegistration {
        room: BootstrapRoom,
        session_id: String,
    },
}

pub type MooncakeRemoteKvLayoutError = PdRemoteKvLayoutError;

impl fmt::Display for PdRemoteKvLayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingTransferRoom(room) => {
                write!(formatter, "missing transfer room: {room}")
            }
            Self::MissingKvArgsRegistration { room, session_id } => write!(
                formatter,
                "missing KVArgs registration for room {room} session {session_id}"
            ),
        }
    }
}

impl std::error::Error for PdRemoteKvLayoutError {}

#[derive(Clone, Debug)]
struct RegisteredDpRank {
    dp_rank: i32,
    timestamp_secs: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PrefillRouteRegistration {
    pub attn_tp_size: usize,
    pub attn_tp_rank: usize,
    pub attn_cp_size: usize,
    pub attn_cp_rank: usize,
    pub attn_dp_size: usize,
    pub attn_dp_rank: usize,
    pub pp_size: usize,
    pub pp_rank: usize,
    pub system_dp_size: usize,
    pub system_dp_rank: usize,
    pub rank_ip: String,
    pub rank_port: u16,
    pub page_size: Option<usize>,
    pub kv_cache_dtype: Option<String>,
    pub load_balance_method: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct RouteQuery {
    prefill_dp_rank: i32,
    prefill_cp_rank: i32,
    target_tp_rank: i32,
    target_pp_rank: i32,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct RegisterDpRankRequest {
    bootstrap_room: BootstrapRoom,
    dp_rank: i32,
}

#[derive(Clone, Debug, Deserialize)]
struct QueryDpRanksRequest {
    bootstrap_rooms: Vec<BootstrapRoom>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PdBootstrapFrameError {
    MissingField {
        frame: &'static str,
        expected: usize,
        actual: usize,
    },
    Utf8Field {
        field: &'static str,
    },
    IntegerField {
        field: &'static str,
        value: String,
    },
    BinaryLength {
        field: &'static str,
        width: usize,
        actual: usize,
    },
    PackedListHeader {
        field: &'static str,
    },
    PackedListLength {
        field: &'static str,
    },
}

pub type MooncakeBootstrapFrameError = PdBootstrapFrameError;

impl fmt::Display for PdBootstrapFrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField {
                frame,
                expected,
                actual,
            } => write!(
                formatter,
                "{frame} frame requires at least {expected} fields, got {actual}"
            ),
            Self::Utf8Field { field } => write!(formatter, "{field} must be ASCII/UTF-8"),
            Self::IntegerField { field, value } => {
                write!(formatter, "{field} must be an integer, got {value}")
            }
            Self::BinaryLength {
                field,
                width,
                actual,
            } => write!(
                formatter,
                "{field} byte length {actual} must be divisible by {width}"
            ),
            Self::PackedListHeader { field } => {
                write!(formatter, "{field} packed list header is incomplete")
            }
            Self::PackedListLength { field } => {
                write!(formatter, "{field} packed list payload is incomplete")
            }
        }
    }
}

impl std::error::Error for PdBootstrapFrameError {}

#[derive(Debug)]
pub enum PrefillBootstrapServeError {
    Io(std::io::Error),
    BootstrapFrame(PdBootstrapFrameError),
    Zmq(String),
}

impl fmt::Display for PrefillBootstrapServeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "prefill bootstrap server error: {error}"),
            Self::BootstrapFrame(error) => {
                write!(formatter, "prefill bootstrap frame error: {error}")
            }
            Self::Zmq(error) => write!(formatter, "prefill bootstrap ZMQ error: {error}"),
        }
    }
}

impl std::error::Error for PrefillBootstrapServeError {}

impl From<std::io::Error> for PrefillBootstrapServeError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<PdBootstrapFrameError> for PrefillBootstrapServeError {
    fn from(value: PdBootstrapFrameError) -> Self {
        Self::BootstrapFrame(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PdDecodeKvArgsRegistration {
    pub endpoint: String,
    pub dst_port: u16,
    pub session_id: String,
    pub dst_kv_ptrs: Vec<u64>,
    pub dst_aux_ptrs: Vec<u64>,
    pub dst_state_data_ptrs: Vec<Vec<u64>>,
    pub dst_tp_rank: i32,
    pub dst_attn_tp_size: usize,
    pub dst_kv_item_len: usize,
    pub dst_state_item_lens: Vec<Vec<u32>>,
    pub dst_state_dim_per_tensor: Vec<Vec<u32>>,
}

pub type MooncakeDecodeKvArgsRegistration = PdDecodeKvArgsRegistration;

impl PdDecodeKvArgsRegistration {
    fn to_frame(&self) -> Vec<Vec<u8>> {
        vec![
            b"None".to_vec(),
            self.endpoint.as_bytes().to_vec(),
            self.dst_port.to_string().into_bytes(),
            self.session_id.as_bytes().to_vec(),
            pack_u64s(&self.dst_kv_ptrs),
            pack_u64s(&self.dst_aux_ptrs),
            pack_u64_lists(&self.dst_state_data_ptrs),
            self.dst_tp_rank.to_string().into_bytes(),
            self.dst_attn_tp_size.to_string().into_bytes(),
            self.dst_kv_item_len.to_string().into_bytes(),
            pack_u32_lists(&self.dst_state_item_lens),
            pack_u32_lists(&self.dst_state_dim_per_tensor),
            Vec::new(),
            b"0".to_vec(),
        ]
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PdDecodeTransferMetadata {
    pub room: BootstrapRoom,
    pub endpoint: String,
    pub dst_port: u16,
    pub session_id: String,
    pub dst_kv_indices: Vec<i32>,
    pub dst_aux_index: Option<i32>,
    pub dst_state_indices: Vec<Vec<i32>>,
    pub required_dst_info_num: usize,
    pub decode_prefix_len: Option<usize>,
    pub is_dummy: bool,
}

pub type MooncakeDecodeTransferMetadata = PdDecodeTransferMetadata;

impl PdDecodeTransferMetadata {
    fn to_frame(&self) -> Result<Vec<Vec<u8>>, PdDecodeBootstrapError> {
        let (dst_kv_indices, dst_aux_index, dst_state_indices) = if self.is_dummy {
            (Vec::new(), Vec::new(), Vec::new())
        } else {
            let dst_aux_index =
                self.dst_aux_index
                    .ok_or(PdDecodeBootstrapError::MissingAuxIndex {
                        session_id: self.session_id.clone(),
                    })?;
            (
                pack_i32s(&self.dst_kv_indices),
                dst_aux_index.to_string().into_bytes(),
                pack_i32_lists(&self.dst_state_indices),
            )
        };

        Ok(vec![
            self.room.to_string().into_bytes(),
            self.endpoint.as_bytes().to_vec(),
            self.dst_port.to_string().into_bytes(),
            self.session_id.as_bytes().to_vec(),
            dst_kv_indices,
            dst_aux_index,
            dst_state_indices,
            self.required_dst_info_num.to_string().into_bytes(),
            self.decode_prefix_len.unwrap_or(0).to_string().into_bytes(),
        ])
    }
}

#[derive(Debug)]
pub struct PdDecodeBootstrapPublisher {
    endpoint: String,
    dst_port: u16,
    session_id: String,
    kv_cache_layout: Option<KvTransferLayout>,
    target_tp_rank: i32,
    target_pp_rank: i32,
    dst_aux_index: Option<i32>,
    required_dst_info_num: usize,
    registered_kv_args_endpoints: BTreeSet<String>,
    receiver: Option<PdDecodeReceiver>,
    receive_targets: BTreeMap<BootstrapRoom, String>,
}

pub type MooncakeDecodeBootstrapPublisher = PdDecodeBootstrapPublisher;

impl PdDecodeBootstrapPublisher {
    pub fn new(endpoint: impl Into<String>, dst_port: u16, session_id: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            dst_port,
            session_id: session_id.into(),
            kv_cache_layout: None,
            target_tp_rank: 0,
            target_pp_rank: 0,
            dst_aux_index: Some(0),
            required_dst_info_num: 1,
            registered_kv_args_endpoints: BTreeSet::new(),
            receiver: None,
            receive_targets: BTreeMap::new(),
        }
    }

    pub fn with_kv_cache_layout(mut self, kv_cache_layout: KvTransferLayout) -> Self {
        self.kv_cache_layout = Some(kv_cache_layout);
        self
    }

    pub fn with_receiver(mut self) -> Result<Self, String> {
        let receiver = PdDecodeReceiver::start(&self.endpoint)?;
        self.dst_port = receiver.port();
        self.receiver = Some(receiver);
        Ok(self)
    }

    pub fn with_target_ranks(mut self, target_tp_rank: i32, target_pp_rank: i32) -> Self {
        self.target_tp_rank = target_tp_rank;
        self.target_pp_rank = target_pp_rank;
        self
    }

    pub fn with_required_dst_info_num(mut self, required_dst_info_num: usize) -> Self {
        self.required_dst_info_num = required_dst_info_num;
        self
    }

    pub fn with_dst_aux_index(mut self, dst_aux_index: Option<i32>) -> Self {
        self.dst_aux_index = dst_aux_index;
        self
    }

    fn metadata_for_span(
        &self,
        span: &KvCacheTransferSpan,
    ) -> Result<PdDecodeTransferMetadata, String> {
        Ok(PdDecodeTransferMetadata {
            room: span.bootstrap_room(),
            endpoint: self.endpoint.clone(),
            dst_port: self.dst_port,
            session_id: self.session_id.clone(),
            dst_kv_indices: span
                .page_indices()
                .iter()
                .map(|page| {
                    i32::try_from(page.as_usize()).map_err(|_| {
                        format!(
                            "physical KV page {} cannot fit into metadata i32 index",
                            page.as_usize()
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            dst_aux_index: self.dst_aux_index,
            dst_state_indices: Vec::new(),
            required_dst_info_num: self.required_dst_info_num,
            decode_prefix_len: Some(span.token_offset()),
            is_dummy: false,
        })
    }

    fn kv_args_registration(&self) -> Option<PdDecodeKvArgsRegistration> {
        let layout = self.kv_cache_layout?;
        if layout.source_base_addr == 0 {
            return None;
        }
        Some(PdDecodeKvArgsRegistration {
            endpoint: self.endpoint.clone(),
            dst_port: self.dst_port,
            session_id: self.session_id.clone(),
            dst_kv_ptrs: vec![layout.source_base_addr as u64],
            dst_aux_ptrs: Vec::new(),
            dst_state_data_ptrs: Vec::new(),
            dst_tp_rank: self.target_tp_rank,
            dst_attn_tp_size: 1,
            dst_kv_item_len: layout.page_size_bytes,
            dst_state_item_lens: Vec::new(),
            dst_state_dim_per_tensor: Vec::new(),
        })
    }

    #[doc(hidden)]
    pub fn kv_args_registration_for_test(&self) -> Option<PdDecodeKvArgsRegistration> {
        self.kv_args_registration()
    }
}

impl DecodeBootstrapPublisher for PdDecodeBootstrapPublisher {
    fn publish_decode_bootstrap_metadata(
        &mut self,
        plan: &KvCacheTransferPlan,
    ) -> Result<DecodeBootstrapMetadataPublishSummary, String> {
        let mut published_spans = 0;
        for span in plan.spans() {
            let metadata = self.metadata_for_span(span)?;
            let bootstrap_addr = format!(
                "{}:{}",
                span.disaggregated_params().bootstrap_host,
                span.disaggregated_params().bootstrap_port
            );
            let prefill_dp_rank = span.data_parallel_rank();
            let target_tp_rank = self.target_tp_rank;
            let target_pp_rank = self.target_pp_rank;
            let kv_args_registration_key = format!(
                "{}|{}|{}|{}|{}",
                bootstrap_addr, prefill_dp_rank, target_tp_rank, target_pp_rank, self.session_id
            );
            let kv_args_registration = if self
                .registered_kv_args_endpoints
                .contains(&kv_args_registration_key)
            {
                None
            } else {
                self.kv_args_registration()
            };
            let mark_kv_args_registered = kv_args_registration.is_some();
            if let Some(receiver) = &self.receiver {
                receiver.register(span.bootstrap_room())?;
            }
            self.receive_targets
                .insert(span.bootstrap_room(), bootstrap_addr.clone());
            publish_decode_metadata_blocking(
                bootstrap_addr,
                prefill_dp_rank,
                target_tp_rank,
                target_pp_rank,
                kv_args_registration,
                metadata,
            )?;
            if mark_kv_args_registered {
                self.registered_kv_args_endpoints
                    .insert(kv_args_registration_key);
            }
            published_spans += 1;
        }

        Ok(DecodeBootstrapMetadataPublishSummary { published_spans })
    }

    fn receive_prefill_token(&mut self, room: BootstrapRoom) -> Result<u32, String> {
        self.receive_prefill_token_and_state(room)
            .map(|(token, _)| token)
    }

    fn receive_prefill_token_and_state(
        &mut self,
        room: BootstrapRoom,
    ) -> Result<(u32, Option<Vec<u8>>), String> {
        let receiver = self
            .receiver
            .as_ref()
            .ok_or("decode receiver was not started")?;
        let (token, state) = receiver.receive_token_and_state(room)?;
        receiver.remove(room);
        self.receive_targets.remove(&room);
        Ok((token, state))
    }

    fn cancel_receive(&mut self, room: BootstrapRoom) -> Result<(), String> {
        let Some(bootstrap_addr) = self.receive_targets.get(&room).cloned() else {
            return Ok(());
        };
        let receiver = self
            .receiver
            .as_ref()
            .ok_or("decode receiver was not started")?;
        if receiver.is_drained(room) {
            receiver.remove(room);
            self.receive_targets.remove(&room);
            return Ok(());
        }
        let target = query_control_target(bootstrap_addr)?;
        send_frames(
            target,
            vec![vec![
                b"ABORT".to_vec(),
                room.to_string().into_bytes(),
                self.endpoint.as_bytes().to_vec(),
                self.dst_port.to_string().into_bytes(),
            ]],
        )?;
        receiver.wait_drained(room)?;
        receiver.remove(room);
        self.receive_targets.remove(&room);
        Ok(())
    }
}

impl Drop for PdDecodeBootstrapPublisher {
    fn drop(&mut self) {
        for room in self.receive_targets.keys().copied().collect::<Vec<_>>() {
            if self.receiver.is_some()
                && let Err(error) = self.cancel_receive(room)
            {
                eprintln!(
                    "fatal: cannot confirm remote KV writes stopped before receiver destruction: {error}"
                );
                std::process::abort();
            }
        }
    }
}

pub fn query_control_target(bootstrap_addr: String) -> Result<String, String> {
    std::thread::spawn(move || {
        let response = http_get(
            &bootstrap_addr,
            "/route?prefill_dp_rank=0&prefill_cp_rank=0&target_tp_rank=0&target_pp_rank=0",
        )
        .map_err(|e| e.to_string())?;
        let body = http_success_body(response).map_err(|e| e.to_string())?;
        let rank: PrefillRankInfo = serde_json::from_str(&body).map_err(|e| e.to_string())?;
        Ok(rank.zmq_endpoint())
    })
    .join()
    .map_err(|_| "control route lookup panicked".to_string())?
}

pub fn query_mooncake_control_target(bootstrap_addr: String) -> Result<String, String> {
    query_control_target(bootstrap_addr)
}

#[derive(Debug)]
pub enum PdDecodeBootstrapError {
    HttpStatus { status: u16, body: String },
    InvalidHttpResponse,
    Io(std::io::Error),
    Json(serde_json::Error),
    MissingAuxIndex { session_id: String },
    TaskJoin(String),
    Zmq(String),
}

pub type MooncakeDecodeBootstrapError = PdDecodeBootstrapError;

impl fmt::Display for PdDecodeBootstrapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HttpStatus { status, body } => {
                write!(
                    formatter,
                    "bootstrap route query failed with HTTP {status}: {body}"
                )
            }
            Self::InvalidHttpResponse => formatter.write_str("invalid bootstrap HTTP response"),
            Self::Io(error) => write!(formatter, "bootstrap HTTP client I/O error: {error}"),
            Self::Json(error) => write!(formatter, "bootstrap HTTP JSON error: {error}"),
            Self::MissingAuxIndex { session_id } => {
                write!(
                    formatter,
                    "transfer metadata for {session_id} needs aux index"
                )
            }
            Self::TaskJoin(error) => write!(formatter, "bootstrap client task join error: {error}"),
            Self::Zmq(error) => write!(formatter, "bootstrap ZMQ client error: {error}"),
        }
    }
}

impl std::error::Error for PdDecodeBootstrapError {}

impl From<std::io::Error> for PdDecodeBootstrapError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for PdDecodeBootstrapError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

pub async fn serve_prefill_bootstrap(
    addr: SocketAddr,
    service: PrefillBootstrapService,
) -> Result<(), PrefillBootstrapServeError> {
    serve_prefill_bootstrap_with_shutdown(addr, service, std::future::pending::<()>()).await
}

pub async fn serve_prefill_bootstrap_with_shutdown<F>(
    addr: SocketAddr,
    service: PrefillBootstrapService,
    shutdown: F,
) -> Result<(), PrefillBootstrapServeError>
where
    F: Future<Output = ()> + Send + 'static,
{
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, service.into_router())
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

pub async fn serve_bootstrap_zmq_with_shutdown<F>(
    endpoint: impl Into<String>,
    service: PrefillBootstrapService,
    shutdown: F,
) -> Result<(), PrefillBootstrapServeError>
where
    F: Future<Output = ()> + Send + 'static,
{
    let endpoint = endpoint.into();
    let mut socket = PullSocket::new();
    socket
        .bind(&endpoint)
        .await
        .map_err(|error| PrefillBootstrapServeError::Zmq(error.to_string()))?;

    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            _ = &mut shutdown => return Ok(()),
            message = socket.recv() => {
                let message = message
                    .map_err(|error| PrefillBootstrapServeError::Zmq(error.to_string()))?;
                let frames = message
                    .into_vec()
                    .into_iter()
                    .map(|frame| frame.to_vec())
                    .collect::<Vec<_>>();
                if frames.first().is_some_and(|field| field == b"ABORT") {
                    let room = parse_bootstrap_room(&frames, 1, "room")?;
                    let host = ascii_field(&frames, 2, "decode host")?;
                    let port = parse_u16(&frames, 3, "decode port")?;
                    let endpoint = format!("tcp://{host}:{port}");
                    let quiescent = {
                        let mut state = service.state.lock().expect("bootstrap state lock");
                        state.aborted_rooms.insert(room);
                        if state.active_rooms.contains(&room) {
                            state.abort_targets.entry(room).or_default().push(endpoint.clone());
                            false
                        } else { true }
                    };
                    if quiescent {
                        send_bootstrap_frame(&endpoint, vec![b"ABORT_ACK".to_vec(), room.to_string().into_bytes(), b"0".to_vec()]).await.map_err(|e| PrefillBootstrapServeError::Zmq(e.to_string()))?;
                    }
                    continue;
                }
                service
                    .state
                    .lock()
                    .expect("prefill bootstrap state lock should be held")
                    .ingest_bootstrap_frame(&frames)?;
            }
        }
    }
}

pub async fn serve_mooncake_bootstrap_zmq_with_shutdown<F>(
    endpoint: impl Into<String>,
    service: PrefillBootstrapService,
    shutdown: F,
) -> Result<(), PrefillBootstrapServeError>
where
    F: Future<Output = ()> + Send + 'static,
{
    serve_bootstrap_zmq_with_shutdown(endpoint, service, shutdown).await
}

pub async fn serve_bootstrap_zmq_endpoints_with_shutdown<F>(
    endpoints: Vec<String>,
    service: PrefillBootstrapService,
    shutdown: F,
) -> Result<(), PrefillBootstrapServeError>
where
    F: Future<Output = ()> + Send + 'static,
{
    if endpoints.is_empty() {
        shutdown.await;
        return Ok(());
    }

    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let mut tasks = tokio::task::JoinSet::new();
    for endpoint in endpoints {
        let service = service.clone();
        let shutdown_rx = shutdown_rx.clone();
        tasks.spawn(serve_bootstrap_zmq_with_shutdown(
            endpoint,
            service,
            watch_bootstrap_shutdown(shutdown_rx),
        ));
    }

    tokio::pin!(shutdown);
    tokio::select! {
        _ = &mut shutdown => {
            let _ = shutdown_tx.send(true);
            join_bootstrap_zmq_tasks(tasks).await?;
            Ok(())
        }
        result = tasks.join_next(), if !tasks.is_empty() => {
            let _ = shutdown_tx.send(true);
            join_bootstrap_zmq_tasks(tasks).await?;
            match result {
                Some(Ok(Ok(()))) => Ok(()),
                Some(Ok(Err(error))) => Err(error),
                Some(Err(error)) => {
                    Err(PrefillBootstrapServeError::Zmq(error.to_string()))
                }
                None => Ok(()),
            }
        }
    }
}

pub async fn serve_mooncake_bootstrap_zmq_endpoints_with_shutdown<F>(
    endpoints: Vec<String>,
    service: PrefillBootstrapService,
    shutdown: F,
) -> Result<(), PrefillBootstrapServeError>
where
    F: Future<Output = ()> + Send + 'static,
{
    serve_bootstrap_zmq_endpoints_with_shutdown(endpoints, service, shutdown).await
}

pub async fn query_prefill_route(
    bootstrap_addr: &str,
    prefill_dp_rank: i32,
    prefill_cp_rank: i32,
    target_tp_rank: i32,
    target_pp_rank: i32,
) -> Result<PrefillRankInfo, PdDecodeBootstrapError> {
    let bootstrap_addr = bootstrap_addr.to_string();
    tokio::task::spawn_blocking(move || {
        let path = format!(
            "/route?prefill_dp_rank={prefill_dp_rank}&prefill_cp_rank={prefill_cp_rank}&target_tp_rank={target_tp_rank}&target_pp_rank={target_pp_rank}"
        );
        let response = http_get(&bootstrap_addr, &path)?;
        let body = http_success_body(response)?;
        Ok(serde_json::from_str(&body)?)
    })
    .await
    .map_err(|error| PdDecodeBootstrapError::TaskJoin(error.to_string()))?
}

pub async fn send_transfer_metadata(
    endpoint: &str,
    metadata: &PdDecodeTransferMetadata,
) -> Result<(), PdDecodeBootstrapError> {
    send_bootstrap_frame(endpoint, metadata.to_frame()?).await
}

pub async fn send_mooncake_transfer_metadata(
    endpoint: &str,
    metadata: &PdDecodeTransferMetadata,
) -> Result<(), PdDecodeBootstrapError> {
    send_transfer_metadata(endpoint, metadata).await
}

pub async fn send_kv_args_registration(
    endpoint: &str,
    registration: &PdDecodeKvArgsRegistration,
) -> Result<(), PdDecodeBootstrapError> {
    send_bootstrap_frame(endpoint, registration.to_frame()).await
}

pub async fn send_mooncake_kv_args_registration(
    endpoint: &str,
    registration: &PdDecodeKvArgsRegistration,
) -> Result<(), PdDecodeBootstrapError> {
    send_kv_args_registration(endpoint, registration).await
}

fn publish_decode_metadata_blocking(
    bootstrap_addr: String,
    prefill_dp_rank: i32,
    target_tp_rank: i32,
    target_pp_rank: i32,
    kv_args_registration: Option<PdDecodeKvArgsRegistration>,
    metadata: PdDecodeTransferMetadata,
) -> Result<(), String> {
    let run_client = move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| error.to_string())?;
        runtime.block_on(async move {
            let rank = query_prefill_route(
                &bootstrap_addr,
                prefill_dp_rank,
                0,
                target_tp_rank,
                target_pp_rank,
            )
            .await
            .map_err(|error| error.to_string())?;
            if let Some(kv_args_registration) = kv_args_registration.as_ref() {
                send_kv_args_registration(&rank.zmq_endpoint(), kv_args_registration)
                    .await
                    .map_err(|error| error.to_string())?;
            }
            send_transfer_metadata(&rank.zmq_endpoint(), &metadata)
                .await
                .map_err(|error| error.to_string())
        })
    };

    if tokio::runtime::Handle::try_current().is_ok() {
        tokio::task::block_in_place(run_client)
    } else {
        run_client()
    }
}

pub fn publish_mooncake_decode_metadata_blocking(
    bootstrap_addr: String,
    prefill_dp_rank: i32,
    target_tp_rank: i32,
    target_pp_rank: i32,
    kv_args_registration: Option<PdDecodeKvArgsRegistration>,
    metadata: PdDecodeTransferMetadata,
) -> Result<(), String> {
    publish_decode_metadata_blocking(
        bootstrap_addr,
        prefill_dp_rank,
        target_tp_rank,
        target_pp_rank,
        kv_args_registration,
        metadata,
    )
}

async fn send_bootstrap_frame(
    endpoint: &str,
    frame: Vec<Vec<u8>>,
) -> Result<(), PdDecodeBootstrapError> {
    let mut last_error = None;
    for _ in 0..20 {
        let mut socket = PushSocket::new();
        match socket.connect(endpoint).await {
            Ok(()) => {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                match socket.send(zmq_message(frame.clone())).await {
                    Ok(()) => return Ok(()),
                    Err(error) => last_error = Some(error.to_string()),
                }
            }
            Err(error) => last_error = Some(error.to_string()),
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }

    Err(PdDecodeBootstrapError::Zmq(
        last_error.unwrap_or_else(|| "no ZMQ send attempts ran".to_string()),
    ))
}

pub async fn send_mooncake_bootstrap_frame(
    endpoint: &str,
    frame: Vec<Vec<u8>>,
) -> Result<(), PdDecodeBootstrapError> {
    send_bootstrap_frame(endpoint, frame).await
}

fn http_get(addr: &str, path: &str) -> Result<String, PdDecodeBootstrapError> {
    let mut stream = std::net::TcpStream::connect(addr)?;
    let request = format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes())?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    Ok(response)
}

fn http_success_body(response: String) -> Result<String, PdDecodeBootstrapError> {
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .ok_or(PdDecodeBootstrapError::InvalidHttpResponse)?;
    let status = headers
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|status| status.parse::<u16>().ok())
        .ok_or(PdDecodeBootstrapError::InvalidHttpResponse)?;
    if status != 200 {
        return Err(PdDecodeBootstrapError::HttpStatus {
            status,
            body: body.to_string(),
        });
    }
    Ok(body.to_string())
}

fn zmq_message(frames: Vec<Vec<u8>>) -> ZmqMessage {
    let mut frames = frames.into_iter();
    let first = frames
        .next()
        .expect("bootstrap message should have at least one frame");
    let mut message = ZmqMessage::from(first);
    for frame in frames {
        message.push_back(frame.into());
    }
    message
}

async fn join_bootstrap_zmq_tasks(
    mut tasks: tokio::task::JoinSet<Result<(), PrefillBootstrapServeError>>,
) -> Result<(), PrefillBootstrapServeError> {
    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => return Err(error),
            Err(error) => return Err(PrefillBootstrapServeError::Zmq(error.to_string())),
        }
    }
    Ok(())
}

pub async fn join_mooncake_zmq_tasks(
    tasks: tokio::task::JoinSet<Result<(), PrefillBootstrapServeError>>,
) -> Result<(), PrefillBootstrapServeError> {
    join_bootstrap_zmq_tasks(tasks).await
}

fn pack_i32s(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn pack_u64s(values: &[u64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn pack_i32_lists(values: &[Vec<i32>]) -> Vec<u8> {
    pack_list_of_buffers(
        &values
            .iter()
            .map(|values| pack_i32s(values))
            .collect::<Vec<_>>(),
    )
}

fn pack_u64_lists(values: &[Vec<u64>]) -> Vec<u8> {
    pack_list_of_buffers(
        &values
            .iter()
            .map(|values| pack_u64s(values))
            .collect::<Vec<_>>(),
    )
}

fn pack_u32_lists(values: &[Vec<u32>]) -> Vec<u8> {
    pack_list_of_buffers(
        &values
            .iter()
            .map(|values| {
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
    )
}

fn pack_list_of_buffers(buffers: &[Vec<u8>]) -> Vec<u8> {
    if buffers.is_empty() {
        return Vec::new();
    }

    let mut packed = Vec::new();
    packed.extend_from_slice(&(buffers.len() as u32).to_le_bytes());
    for buffer in buffers {
        packed.extend_from_slice(&(buffer.len() as u32).to_le_bytes());
    }
    for buffer in buffers {
        packed.extend_from_slice(buffer);
    }
    packed
}

async fn watch_bootstrap_shutdown(mut shutdown_rx: tokio::sync::watch::Receiver<bool>) {
    loop {
        if *shutdown_rx.borrow() {
            break;
        }
        if shutdown_rx.changed().await.is_err() {
            break;
        }
    }
}

async fn health() -> &'static str {
    "OK"
}

async fn route_put(
    State(service): State<PrefillBootstrapService>,
    Json(registration): Json<PrefillRouteRegistration>,
) -> &'static str {
    service
        .state
        .lock()
        .expect("prefill bootstrap state lock should be held")
        .register_route(registration);
    "OK"
}

async fn route_get(
    State(service): State<PrefillBootstrapService>,
    Query(query): Query<RouteQuery>,
) -> Response {
    let state = service
        .state
        .lock()
        .expect("prefill bootstrap state lock should be held");

    if query.prefill_dp_rank == -1
        && query.prefill_cp_rank == -1
        && query.target_tp_rank == -1
        && query.target_pp_rank == -1
    {
        return match state.server_info() {
            Some(info) => Json(info).into_response(),
            None => (
                StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "Prefill server not fully registered yet ({} workers registered).",
                    state.registered_count
                ),
            )
                .into_response(),
        };
    }

    if !state.is_ready() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            format!(
                "Prefill server not fully registered yet ({} workers registered).",
                state.registered_count
            ),
        )
            .into_response();
    }

    let Ok(prefill_dp_rank) = usize::try_from(query.prefill_dp_rank) else {
        return bad_rank_query();
    };
    let Ok(prefill_cp_rank) = usize::try_from(query.prefill_cp_rank) else {
        return bad_rank_query();
    };
    let Ok(target_tp_rank) = usize::try_from(query.target_tp_rank) else {
        return bad_rank_query();
    };
    let Ok(target_pp_rank) = usize::try_from(query.target_pp_rank) else {
        return bad_rank_query();
    };

    match state.rank_info(
        prefill_dp_rank,
        prefill_cp_rank,
        target_tp_rank,
        target_pp_rank,
    ) {
        Some(info) => Json(info).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            format!(
                "Bootstrap info not found for dp_rank={} cp_rank={} tp_rank={} pp_rank={}",
                query.prefill_dp_rank,
                query.prefill_cp_rank,
                query.target_tp_rank,
                query.target_pp_rank
            ),
        )
            .into_response(),
    }
}

async fn register_dp_rank(
    State(service): State<PrefillBootstrapService>,
    Json(request): Json<RegisterDpRankRequest>,
) -> &'static str {
    service
        .state
        .lock()
        .expect("prefill bootstrap state lock should be held")
        .register_dp_rank(request.bootstrap_room, request.dp_rank);
    "OK"
}

async fn query_dp_ranks(
    State(service): State<PrefillBootstrapService>,
    Json(request): Json<QueryDpRanksRequest>,
) -> Json<serde_json::Value> {
    let mut state = service
        .state
        .lock()
        .expect("prefill bootstrap state lock should be held");
    Json(state.query_dp_ranks(&request.bootstrap_rooms))
}

fn bad_rank_query() -> Response {
    (
        StatusCode::BAD_REQUEST,
        "rank query fields must be non-negative",
    )
        .into_response()
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn require_min_fields(
    frame: &[Vec<u8>],
    expected: usize,
    frame_name: &'static str,
) -> Result<(), PdBootstrapFrameError> {
    if frame.len() < expected {
        return Err(PdBootstrapFrameError::MissingField {
            frame: frame_name,
            expected,
            actual: frame.len(),
        });
    }
    Ok(())
}

fn field<'a>(
    frame: &'a [Vec<u8>],
    index: usize,
    field: &'static str,
) -> Result<&'a [u8], PdBootstrapFrameError> {
    frame
        .get(index)
        .map(Vec::as_slice)
        .ok_or(PdBootstrapFrameError::MissingField {
            frame: field,
            expected: index + 1,
            actual: frame.len(),
        })
}

fn ascii_field<'a>(
    frame: &'a [Vec<u8>],
    index: usize,
    name: &'static str,
) -> Result<&'a str, PdBootstrapFrameError> {
    std::str::from_utf8(field(frame, index, name)?)
        .map_err(|_| PdBootstrapFrameError::Utf8Field { field: name })
}

fn parse_i32(
    frame: &[Vec<u8>],
    index: usize,
    name: &'static str,
) -> Result<i32, PdBootstrapFrameError> {
    let value = ascii_field(frame, index, name)?;
    value
        .parse()
        .map_err(|_| PdBootstrapFrameError::IntegerField {
            field: name,
            value: value.to_string(),
        })
}

fn parse_bootstrap_room(
    frame: &[Vec<u8>],
    index: usize,
    name: &'static str,
) -> Result<BootstrapRoom, PdBootstrapFrameError> {
    let value = ascii_field(frame, index, name)?;
    let room = value
        .parse::<BootstrapRoom>()
        .map_err(|_| PdBootstrapFrameError::IntegerField {
            field: name,
            value: value.to_string(),
        })?;
    if room > i64::MAX as u64 {
        return Err(PdBootstrapFrameError::IntegerField {
            field: name,
            value: value.to_string(),
        });
    }

    Ok(room)
}

fn parse_u16(
    frame: &[Vec<u8>],
    index: usize,
    name: &'static str,
) -> Result<u16, PdBootstrapFrameError> {
    let value = ascii_field(frame, index, name)?;
    value
        .parse()
        .map_err(|_| PdBootstrapFrameError::IntegerField {
            field: name,
            value: value.to_string(),
        })
}

fn parse_usize(
    frame: &[Vec<u8>],
    index: usize,
    name: &'static str,
) -> Result<usize, PdBootstrapFrameError> {
    let value = ascii_field(frame, index, name)?;
    value
        .parse()
        .map_err(|_| PdBootstrapFrameError::IntegerField {
            field: name,
            value: value.to_string(),
        })
}

fn optional_usize(
    frame: &[Vec<u8>],
    index: usize,
    name: &'static str,
) -> Result<Option<usize>, PdBootstrapFrameError> {
    let Some(value) = frame.get(index) else {
        return Ok(None);
    };
    if value.is_empty() {
        return Ok(None);
    }
    parse_usize(frame, index, name).map(Some)
}

fn optional_u32_lists(
    frame: &[Vec<u8>],
    index: usize,
) -> Result<Vec<Vec<u32>>, PdBootstrapFrameError> {
    let Some(value) = frame.get(index) else {
        return Ok(Vec::new());
    };
    unpack_u32_lists(value)
}

fn unpack_u64s(bytes: &[u8]) -> Result<Vec<u64>, PdBootstrapFrameError> {
    unpack_fixed_width(bytes, 8, "u64 values", |chunk| {
        u64::from_le_bytes(chunk.try_into().expect("chunk width is checked"))
    })
}

fn unpack_i32s(bytes: &[u8]) -> Result<Vec<i32>, PdBootstrapFrameError> {
    unpack_fixed_width(bytes, 4, "i32 values", |chunk| {
        i32::from_le_bytes(chunk.try_into().expect("chunk width is checked"))
    })
}

fn unpack_u64_lists(bytes: &[u8]) -> Result<Vec<Vec<u64>>, PdBootstrapFrameError> {
    unpack_list_of_buffers(bytes, "u64 lists")?
        .into_iter()
        .map(unpack_u64s)
        .collect()
}

fn unpack_u32_lists(bytes: &[u8]) -> Result<Vec<Vec<u32>>, PdBootstrapFrameError> {
    unpack_list_of_buffers(bytes, "u32 lists")?
        .into_iter()
        .map(|bytes| {
            unpack_fixed_width(bytes, 4, "u32 values", |chunk| {
                u32::from_le_bytes(chunk.try_into().expect("chunk width is checked"))
            })
        })
        .collect()
}

fn unpack_i32_lists(bytes: &[u8]) -> Result<Vec<Vec<i32>>, PdBootstrapFrameError> {
    unpack_list_of_buffers(bytes, "i32 lists")?
        .into_iter()
        .map(unpack_i32s)
        .collect()
}

fn unpack_fixed_width<T, F>(
    bytes: &[u8],
    width: usize,
    field: &'static str,
    decode: F,
) -> Result<Vec<T>, PdBootstrapFrameError>
where
    F: Fn(&[u8]) -> T,
{
    if !bytes.len().is_multiple_of(width) {
        return Err(PdBootstrapFrameError::BinaryLength {
            field,
            width,
            actual: bytes.len(),
        });
    }
    Ok(bytes.chunks_exact(width).map(decode).collect())
}

fn unpack_list_of_buffers<'a>(
    bytes: &'a [u8],
    field: &'static str,
) -> Result<Vec<&'a [u8]>, PdBootstrapFrameError> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    if bytes.len() < 4 {
        return Err(PdBootstrapFrameError::PackedListHeader { field });
    }
    let count = u32::from_le_bytes(bytes[0..4].try_into().expect("slice has length 4")) as usize;
    let header_len = 4 + count * 4;
    if bytes.len() < header_len {
        return Err(PdBootstrapFrameError::PackedListHeader { field });
    }

    let mut lengths = Vec::with_capacity(count);
    for index in 0..count {
        let offset = 4 + index * 4;
        lengths.push(u32::from_le_bytes(
            bytes[offset..offset + 4]
                .try_into()
                .expect("slice has length 4"),
        ) as usize);
    }

    let mut offset = header_len;
    let mut buffers = Vec::with_capacity(count);
    for length in lengths {
        let end = offset + length;
        if bytes.len() < end {
            return Err(PdBootstrapFrameError::PackedListLength { field });
        }
        buffers.push(&bytes[offset..end]);
        offset = end;
    }
    Ok(buffers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Json;
    use axum::extract::State;

    #[tokio::test]
    async fn query_dp_ranks_prunes_expired_bootstrap_rooms() {
        let service = PrefillBootstrapService::default();
        {
            let mut state = service.state().lock().expect("state lock should be held");
            state.register_dp_rank(41, 3);
            state
                .room_to_dp_rank
                .get_mut(&41)
                .expect("room should be registered")
                .timestamp_secs = now_secs() - 121;
        }

        let Json(response) = query_dp_ranks(
            State(service.clone()),
            Json(QueryDpRanksRequest {
                bootstrap_rooms: vec![41],
            }),
        )
        .await;

        assert_eq!(response, serde_json::json!({}));
        assert!(
            service
                .state()
                .lock()
                .expect("state lock should be held")
                .room_to_dp_rank
                .is_empty(),
            "expired room-to-dp-rank mappings should be removed"
        );
    }
}
