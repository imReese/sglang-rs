use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};

use serde_json::Value;
use tokio::sync::oneshot;

use sglang_srt::pd_bootstrap::{PrefillBootstrapService, serve_prefill_bootstrap_with_shutdown};
use sglang_srt::transfer::KvPoll;

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

#[test]
fn prefill_bootstrap_ingests_mooncake_kv_args_register_frame() {
    let service = PrefillBootstrapService::default();
    let frame = vec![
        b"None".to_vec(),
        b"10.0.0.9".to_vec(),
        b"41001".to_vec(),
        b"session-a".to_vec(),
        pack_u64s(&[0x1000, 0x2000]),
        pack_u64s(&[0x3000]),
        pack_u64_lists(&[vec![0x4000, 0x5000], vec![0x6000]]),
        b"1".to_vec(),
        b"8".to_vec(),
        b"128".to_vec(),
        pack_u32_lists(&[vec![16, 32]]),
        pack_u32_lists(&[vec![4, 8]]),
    ];

    service
        .state()
        .lock()
        .expect("state lock should be held")
        .ingest_mooncake_bootstrap_frame(&frame)
        .expect("KVArgs registration frame should parse");

    let state = service.state().lock().expect("state lock should be held");
    let kv_args = state
        .decode_kv_args("session-a")
        .expect("decode KV args should be registered");
    assert_eq!(kv_args.endpoint, "10.0.0.9");
    assert_eq!(kv_args.dst_port, 41001);
    assert_eq!(kv_args.dst_kv_ptrs, vec![0x1000, 0x2000]);
    assert_eq!(kv_args.dst_aux_ptrs, vec![0x3000]);
    assert_eq!(
        kv_args.dst_state_data_ptrs,
        vec![vec![0x4000, 0x5000], vec![0x6000]]
    );
    assert_eq!(kv_args.dst_tp_rank, 1);
    assert_eq!(kv_args.dst_attn_tp_size, 8);
    assert_eq!(kv_args.dst_kv_item_len, 128);
    assert_eq!(kv_args.dst_state_item_lens, vec![vec![16, 32]]);
    assert_eq!(kv_args.dst_state_dim_per_tensor, vec![vec![4, 8]]);
}

#[test]
fn prefill_bootstrap_ingests_mooncake_transfer_frames_and_marks_room_waiting() {
    let service = PrefillBootstrapService::default();
    let first = vec![
        b"77".to_vec(),
        b"10.0.0.9".to_vec(),
        b"41001".to_vec(),
        b"session-a".to_vec(),
        pack_i32s(&[3, 4, 5]),
        b"11".to_vec(),
        pack_i32_lists(&[vec![30, 31], vec![40]]),
        b"2".to_vec(),
        b"64".to_vec(),
    ];
    let second = vec![
        b"77".to_vec(),
        b"10.0.0.10".to_vec(),
        b"41002".to_vec(),
        b"session-b".to_vec(),
        pack_i32s(&[6, 7]),
        b"12".to_vec(),
        pack_i32_lists(&[vec![50]]),
        b"2".to_vec(),
        b"0".to_vec(),
    ];

    {
        let mut state = service.state().lock().expect("state lock should be held");
        state
            .ingest_mooncake_bootstrap_frame(&first)
            .expect("first transfer frame should parse");
        assert_eq!(state.transfer_status(77), Some(KvPoll::Bootstrapping));
        state
            .ingest_mooncake_bootstrap_frame(&second)
            .expect("second transfer frame should parse");
        assert_eq!(state.transfer_status(77), Some(KvPoll::WaitingForInput));
    }

    let state = service.state().lock().expect("state lock should be held");
    let room = state
        .transfer_room(77)
        .expect("transfer room should be tracked");
    assert_eq!(room.required_dst_info_num, 2);
    assert_eq!(room.decode_prefix_len, Some(64));
    assert_eq!(room.transfers.len(), 2);
    assert_eq!(room.transfers["session-a"].dst_kv_indices, vec![3, 4, 5]);
    assert_eq!(room.transfers["session-a"].dst_aux_index, Some(11));
    assert_eq!(
        room.transfers["session-a"].dst_state_indices,
        vec![vec![30, 31], vec![40]]
    );
    assert_eq!(room.transfers["session-b"].endpoint, "10.0.0.10");
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

fn pack_u64s(values: &[u64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn pack_i32s(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
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

fn pack_i32_lists(values: &[Vec<i32>]) -> Vec<u8> {
    pack_list_of_buffers(
        &values
            .iter()
            .map(|values| pack_i32s(values))
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
