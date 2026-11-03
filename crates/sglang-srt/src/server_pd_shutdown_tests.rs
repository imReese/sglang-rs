use super::*;
use crate::mooncake_control::send_frames;
use crate::scheduler::ScheduleBatch;
use crate::tokenizer::ByteTokenizer;
use crate::worker::{BatchGeneratedTokens, FallibleModelWorker, WorkerExecutionError};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use zeromq::{PullSocket, Socket, SocketRecv};

struct ShutdownGate {
    entered: Option<tokio::sync::oneshot::Sender<()>>,
    release: std::sync::mpsc::Receiver<()>,
}

impl FallibleModelWorker for ShutdownGate {
    fn try_generate_batch(
        &mut self,
        _: &ScheduleBatch,
    ) -> Result<BatchGeneratedTokens, WorkerExecutionError> {
        unreachable!("this test sends only bootstrap control messages")
    }

    fn shutdown(&mut self) -> Result<(), WorkerExecutionError> {
        self.entered
            .take()
            .expect("shutdown once")
            .send(())
            .expect("observer");
        self.release
            .recv_timeout(Duration::from_secs(10))
            .map_err(|e| WorkerExecutionError::Runtime(e.to_string()))
    }
}

fn address() -> SocketAddr {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("port")
        .local_addr()
        .expect("address")
}

async fn bootstrap_health(addr: SocketAddr) {
    for _ in 0..100 {
        if let Ok(mut socket) = tokio::net::TcpStream::connect(addr).await {
            socket
                .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
                .await
                .expect("health request");
            let mut response = String::new();
            socket
                .read_to_string(&mut response)
                .await
                .expect("health response");
            assert!(response.starts_with("HTTP/1.1 200"), "{response}");
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("bootstrap unavailable at {addr}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn http_and_grpc_keep_abort_control_alive_until_runtime_shutdown_finishes() {
    for grpc in [false, true] {
        let service_addr = address();
        let bootstrap_addr = address();
        let zmq_addr = address();
        let (entered, draining) = tokio::sync::oneshot::channel();
        let (release, released) = std::sync::mpsc::channel();
        let worker = ShutdownGate {
            entered: Some(entered),
            release: released,
        };
        let runtime = RouterRuntime::new(Engine::new(ByteTokenizer, Scheduler::new(worker)));
        let args = ServerArgs::parse_from(["serve", "--model-path", "unused-control-only-test"])
            .expect("args");
        let bootstrap = PrefillBootstrapService::default();
        let endpoints = vec![format!("tcp://{zmq_addr}")];
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let shutdown = async move {
                let _ = stopped.await;
            };
            if grpc {
                serve_prefill_grpc_and_bootstrap(
                    service_addr,
                    GrpcRouterService::with_server_args(runtime, &args),
                    PrefillGrpcBootstrapServices {
                        sidecar_addr: address(),
                        enable_metrics: false,
                        prefill_addr: bootstrap_addr,
                        prefill_service: bootstrap,
                        zmq_endpoints: endpoints,
                    },
                    shutdown,
                )
                .await
            } else {
                serve_prefill_http_and_bootstrap(
                    service_addr,
                    HttpRouterService::new(
                        runtime,
                        RouterGetModelInfoResponse::from_server_args(&args),
                    ),
                    PrefillHttpBootstrapServices {
                        prefill_addr: bootstrap_addr,
                        prefill_service: bootstrap,
                        zmq_endpoints: endpoints,
                    },
                    shutdown,
                )
                .await
            }
        });
        bootstrap_health(bootstrap_addr).await;
        stop.send(()).expect("start graceful shutdown");
        tokio::time::timeout(Duration::from_secs(5), draining)
            .await
            .expect("runtime drain starts")
            .expect("worker alive");
        bootstrap_health(bootstrap_addr).await;
        let mut acknowledgements = PullSocket::new();
        let endpoint = acknowledgements
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("ACK receiver")
            .to_string();
        let port = endpoint.rsplit(':').next().expect("port");
        send_frames(
            format!("tcp://{zmq_addr}"),
            vec![vec![
                b"ABORT".to_vec(),
                b"17".to_vec(),
                b"127.0.0.1".to_vec(),
                port.as_bytes().to_vec(),
            ]],
        )
        .expect("ABORT during shutdown");
        let ack = tokio::time::timeout(Duration::from_secs(5), acknowledgements.recv())
            .await
            .expect("control still serves ACK")
            .expect("ACK")
            .into_vec();
        assert_eq!(ack[0].as_ref(), b"ABORT_ACK");
        assert!(!server.is_finished());
        release.send(()).expect("allow runtime teardown");
        server
            .await
            .expect("server joins")
            .expect("shutdown completes");
    }
}
