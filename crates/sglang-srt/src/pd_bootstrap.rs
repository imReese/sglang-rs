use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;

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
    prefill_port_table:
        BTreeMap<usize, BTreeMap<usize, BTreeMap<usize, BTreeMap<usize, PrefillRankInfo>>>>,
    room_to_dp_rank: BTreeMap<i32, RegisteredDpRank>,
}

impl PrefillBootstrapState {
    fn register_route(&mut self, registration: PrefillRouteRegistration) {
        self.attn_tp_size.get_or_insert(registration.attn_tp_size);
        self.attn_cp_size.get_or_insert(registration.attn_cp_size);
        self.dp_size
            .get_or_insert(if registration.system_dp_size == 1 {
                registration.attn_dp_size
            } else {
                registration.system_dp_size
            });
        self.pp_size.get_or_insert(registration.pp_size);
        if registration.page_size.is_some() {
            self.page_size
                .get_or_insert(registration.page_size.unwrap());
        }
        if registration.kv_cache_dtype.is_some() {
            self.kv_cache_dtype
                .get_or_insert_with(|| registration.kv_cache_dtype.clone().unwrap());
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

    fn server_info(&self) -> Option<PrefillServerInfo> {
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

    fn rank_info(
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

    fn register_dp_rank(&mut self, bootstrap_room: i32, dp_rank: i32) {
        self.room_to_dp_rank.insert(
            bootstrap_room,
            RegisteredDpRank {
                dp_rank,
                timestamp_secs: now_secs(),
            },
        );
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PrefillRankInfo {
    pub rank_ip: String,
    pub rank_port: u16,
}

#[derive(Clone, Debug)]
struct RegisteredDpRank {
    dp_rank: i32,
    #[allow(dead_code)]
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
    bootstrap_room: i32,
    dp_rank: i32,
}

#[derive(Clone, Debug, Deserialize)]
struct QueryDpRanksRequest {
    bootstrap_rooms: Vec<i32>,
}

#[derive(Debug)]
pub enum PrefillBootstrapServeError {
    Io(std::io::Error),
}

impl fmt::Display for PrefillBootstrapServeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "prefill bootstrap server error: {error}"),
        }
    }
}

impl std::error::Error for PrefillBootstrapServeError {}

impl From<std::io::Error> for PrefillBootstrapServeError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
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
    let state = service
        .state
        .lock()
        .expect("prefill bootstrap state lock should be held");
    let mut result = serde_json::Map::new();
    for room in request.bootstrap_rooms {
        if let Some(entry) = state.room_to_dp_rank.get(&room) {
            result.insert(room.to_string(), json!(entry.dp_rank));
        }
    }
    Json(serde_json::Value::Object(result))
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
