use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, put};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug, Default)]
pub struct EngineInfoBootstrapService {
    state: Arc<Mutex<EngineInfoBootstrapState>>,
}

impl EngineInfoBootstrapService {
    pub fn state(&self) -> &Arc<Mutex<EngineInfoBootstrapState>> {
        &self.state
    }

    pub fn transfer_engine_info(&self, rank: i32) -> Option<TransferEngineInfo> {
        self.state
            .lock()
            .expect("engine info bootstrap state lock should be held")
            .transfer_engine_info(rank)
            .cloned()
    }

    fn into_router(self) -> Router {
        Router::new()
            .route("/health", get(health))
            .route(
                "/register_transfer_engine_info",
                put(register_transfer_engine_info),
            )
            .route("/get_transfer_engine_info", get(get_transfer_engine_info))
            .with_state(self)
    }
}

#[derive(Clone, Debug, Default)]
pub struct EngineInfoBootstrapState {
    transfer_engine_info: BTreeMap<i32, TransferEngineInfo>,
}

impl EngineInfoBootstrapState {
    pub fn register_transfer_engine_info(&mut self, registration: TransferEngineInfoRegistration) {
        self.transfer_engine_info
            .insert(registration.tp_rank, registration.transfer_engine_info);
    }

    pub fn transfer_engine_info(&self, rank: i32) -> Option<&TransferEngineInfo> {
        self.transfer_engine_info.get(&rank)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct TransferEngineInfoRegistration {
    pub tp_rank: i32,
    pub transfer_engine_info: TransferEngineInfo,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TransferEngineInfo {
    pub session_id: String,
    pub weights_info_dict: Value,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct TransferEngineInfoQuery {
    rank: i32,
}

#[derive(Debug)]
pub enum EngineInfoBootstrapServeError {
    Io(std::io::Error),
}

impl fmt::Display for EngineInfoBootstrapServeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "engine info bootstrap server error: {error}"),
        }
    }
}

impl std::error::Error for EngineInfoBootstrapServeError {}

impl From<std::io::Error> for EngineInfoBootstrapServeError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

pub async fn serve_engine_info_bootstrap(
    addr: SocketAddr,
    service: EngineInfoBootstrapService,
) -> Result<(), EngineInfoBootstrapServeError> {
    serve_engine_info_bootstrap_with_shutdown(addr, service, std::future::pending::<()>()).await
}

pub async fn serve_engine_info_bootstrap_with_shutdown<F>(
    addr: SocketAddr,
    service: EngineInfoBootstrapService,
    shutdown: F,
) -> Result<(), EngineInfoBootstrapServeError>
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

async fn register_transfer_engine_info(
    State(service): State<EngineInfoBootstrapService>,
    Json(registration): Json<TransferEngineInfoRegistration>,
) -> Response {
    if registration.tp_rank < 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "detail": "Invalid rank parameter" })),
        )
            .into_response();
    }
    if registration.transfer_engine_info.session_id.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "detail": "session_id cannot be empty" })),
        )
            .into_response();
    }

    service
        .state()
        .lock()
        .expect("engine info bootstrap state lock should be held")
        .register_transfer_engine_info(registration);
    (StatusCode::OK, "OK").into_response()
}

async fn get_transfer_engine_info(
    State(service): State<EngineInfoBootstrapService>,
    Query(query): Query<TransferEngineInfoQuery>,
) -> Response {
    if query.rank < 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "detail": "Invalid rank parameter" })),
        )
            .into_response();
    }

    let info = service
        .state()
        .lock()
        .expect("engine info bootstrap state lock should be held")
        .transfer_engine_info(query.rank)
        .cloned();

    let Some(info) = info else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "detail": format!("No transfer engine info for rank {}", query.rank) })),
        )
            .into_response();
    };

    (
        StatusCode::OK,
        Json(json!({
            "rank": query.rank,
            "remote_instance_transfer_engine_info": [
                info.session_id,
                info.weights_info_dict
            ],
        })),
    )
        .into_response()
}
