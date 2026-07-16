use std::future::Future;
use std::net::SocketAddr;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde::Deserialize;
use tonic::{Code, Request};

use crate::grpc::GrpcRouterService;
use crate::http::HttpServeError;
use crate::proto::sglang::runtime::v1::sglang_service_server::SglangService;
use crate::proto::sglang::runtime::v1::{StartProfileRequest, StopProfileRequest};
use crate::tokenizer::Tokenizer;
use crate::worker::WorkerExecutor;

struct GrpcHttpSidecar<T, W> {
    service: GrpcRouterService<T, W>,
}

impl<T, W> Clone for GrpcHttpSidecar<T, W> {
    fn clone(&self) -> Self {
        Self {
            service: self.service.clone(),
        }
    }
}

#[derive(Default, Deserialize)]
struct ProfilePayload {
    output_dir: Option<String>,
}

pub async fn serve_grpc_http_sidecar_with_shutdown<T, W, F>(
    addr: SocketAddr,
    service: GrpcRouterService<T, W>,
    enable_metrics: bool,
    shutdown: F,
) -> Result<(), HttpServeError>
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
    F: Future<Output = ()> + Send + 'static,
{
    let sidecar = GrpcHttpSidecar { service };
    let router = Router::new()
        .route("/start_profile", post(start_profile::<T, W>))
        .route("/stop_profile", post(stop_profile::<T, W>));
    let router = if enable_metrics {
        router.route("/metrics", get(metrics::<T, W>))
    } else {
        router
    };
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router.with_state(sidecar))
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

async fn metrics<T, W>(State(sidecar): State<GrpcHttpSidecar<T, W>>) -> Response
where
    T: Send + 'static,
    W: Send + 'static,
{
    let mut response = Response::new(Body::from(sidecar.service.metrics_render()));
    response.headers_mut().insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
    );
    response
}

async fn start_profile<T, W>(State(sidecar): State<GrpcHttpSidecar<T, W>>, body: Bytes) -> Response
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    let payload = if body.is_empty() {
        ProfilePayload::default()
    } else {
        match serde_json::from_slice::<ProfilePayload>(&body) {
            Ok(payload) => payload,
            Err(error) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("invalid profile JSON: {error}\n"),
                )
                    .into_response();
            }
        }
    };
    match sidecar
        .service
        .start_profile(Request::new(StartProfileRequest {
            output_dir: payload.output_dir,
        }))
        .await
    {
        Ok(response) => (
            StatusCode::OK,
            format!("{}\n", response.into_inner().message),
        )
            .into_response(),
        Err(error) => tonic_status_response(error),
    }
}

async fn stop_profile<T, W>(State(sidecar): State<GrpcHttpSidecar<T, W>>) -> Response
where
    T: Tokenizer + Send + 'static,
    W: WorkerExecutor + Send + 'static,
{
    match sidecar
        .service
        .stop_profile(Request::new(StopProfileRequest {}))
        .await
    {
        Ok(response) => (
            StatusCode::OK,
            format!("{}\n", response.into_inner().message),
        )
            .into_response(),
        Err(error) => tonic_status_response(error),
    }
}

fn tonic_status_response(error: tonic::Status) -> Response {
    let status = match error.code() {
        Code::InvalidArgument => StatusCode::BAD_REQUEST,
        Code::AlreadyExists | Code::FailedPrecondition => StatusCode::CONFLICT,
        Code::Unavailable | Code::ResourceExhausted => StatusCode::SERVICE_UNAVAILABLE,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, format!("{}\n", error.message())).into_response()
}
