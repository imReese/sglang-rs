use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};

use serde_json::Value;
use tokio::sync::oneshot;

use sglang_srt::pd_bootstrap::{PrefillBootstrapService, serve_prefill_bootstrap_with_shutdown};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prefill_bootstrap_route_registers_topology_and_rank_endpoint() {
    let addr = unused_local_addr();
    let service = PrefillBootstrapService::default();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_prefill_bootstrap_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let not_ready = get_raw_with_retry(
        addr,
        "/route?prefill_dp_rank=-1&prefill_cp_rank=-1&target_tp_rank=-1&target_pp_rank=-1",
    )
    .await;
    assert!(not_ready.starts_with("HTTP/1.1 503"));

    let register_response = put_json_with_retry(
        addr,
        "/route",
        r#"{"attn_tp_size":2,"attn_tp_rank":1,"attn_cp_size":1,"attn_cp_rank":0,"attn_dp_size":1,"attn_dp_rank":0,"pp_size":1,"pp_rank":0,"system_dp_size":1,"system_dp_rank":0,"rank_ip":"10.0.0.7","rank_port":45678,"page_size":64,"kv_cache_dtype":"auto","load_balance_method":"follow_bootstrap_room"}"#,
    )
    .await;
    assert!(register_response.starts_with("HTTP/1.1 200"));

    put_json_with_retry(
        addr,
        "/route",
        r#"{"attn_tp_size":2,"attn_tp_rank":0,"attn_cp_size":1,"attn_cp_rank":0,"attn_dp_size":1,"attn_dp_rank":0,"pp_size":1,"pp_rank":0,"system_dp_size":1,"system_dp_rank":0,"rank_ip":"10.0.0.6","rank_port":45677,"page_size":64,"kv_cache_dtype":"auto","load_balance_method":"follow_bootstrap_room"}"#,
    )
    .await;

    let topology = get_json_with_retry(
        addr,
        "/route?prefill_dp_rank=-1&prefill_cp_rank=-1&target_tp_rank=-1&target_pp_rank=-1",
    )
    .await;
    assert_eq!(topology["attn_tp_size"], 2);
    assert_eq!(topology["attn_cp_size"], 1);
    assert_eq!(topology["dp_size"], 1);
    assert_eq!(topology["pp_size"], 1);
    assert_eq!(topology["page_size"], 64);
    assert_eq!(topology["kv_cache_dtype"], "auto");
    assert_eq!(topology["follow_bootstrap_room"], true);

    let rank = get_json_with_retry(
        addr,
        "/route?prefill_dp_rank=0&prefill_cp_rank=0&target_tp_rank=1&target_pp_rank=0",
    )
    .await;
    assert_eq!(rank["rank_ip"], "10.0.0.7");
    assert_eq!(rank["rank_port"], 45678);

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prefill_bootstrap_tracks_room_dp_rank_queries() {
    let addr = unused_local_addr();
    let service = PrefillBootstrapService::default();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_prefill_bootstrap_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let register_response = post_json_with_retry(
        addr,
        "/register_dp_rank",
        r#"{"bootstrap_room":41,"dp_rank":3}"#,
    )
    .await;
    assert!(register_response.starts_with("HTTP/1.1 200"));

    let dp_ranks =
        post_json_value_with_retry(addr, "/query_dp_ranks", r#"{"bootstrap_rooms":[40,41]}"#).await;
    assert!(dp_ranks.get("40").is_none());
    assert_eq!(dp_ranks["41"], 3);

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

fn unused_local_addr() -> SocketAddr {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("ephemeral port should bind");
    listener
        .local_addr()
        .expect("ephemeral listener should have local addr")
}

async fn get_json_with_retry(addr: SocketAddr, path: &'static str) -> Value {
    response_body_json(request_with_retry(addr, "GET", path, None).await)
}

async fn get_raw_with_retry(addr: SocketAddr, path: &'static str) -> String {
    request_with_retry(addr, "GET", path, None).await
}

async fn put_json_with_retry(addr: SocketAddr, path: &'static str, body: &'static str) -> String {
    request_with_retry(addr, "PUT", path, Some(body)).await
}

async fn post_json_with_retry(addr: SocketAddr, path: &'static str, body: &'static str) -> String {
    request_with_retry(addr, "POST", path, Some(body)).await
}

async fn post_json_value_with_retry(
    addr: SocketAddr,
    path: &'static str,
    body: &'static str,
) -> Value {
    response_body_json(post_json_with_retry(addr, path, body).await)
}

async fn request_with_retry(
    addr: SocketAddr,
    method: &'static str,
    path: &'static str,
    body: Option<&'static str>,
) -> String {
    let mut last_error = None;

    for _ in 0..20 {
        match request(addr, method, path, body).await {
            Ok(response) => return response,
            Err(error) => {
                last_error = Some(error);
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        }
    }

    panic!(
        "HTTP client should connect to test server: {}",
        last_error.expect("at least one connection attempt should run")
    );
}

async fn request(
    addr: SocketAddr,
    method: &'static str,
    path: &'static str,
    body: Option<&'static str>,
) -> Result<String, std::io::Error> {
    tokio::task::spawn_blocking(move || {
        let mut stream = TcpStream::connect(addr)?;
        let body = body.unwrap_or_default();
        let request = format!(
            "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(request.as_bytes())?;
        let mut response = String::new();
        stream.read_to_string(&mut response)?;
        Ok(response)
    })
    .await
    .expect("blocking HTTP request should join")
}

fn response_body_json(response: String) -> Value {
    let (_, body) = response
        .split_once("\r\n\r\n")
        .expect("HTTP response should include headers");
    serde_json::from_str(body).expect("HTTP response body should be JSON")
}
