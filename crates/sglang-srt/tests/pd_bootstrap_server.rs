use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::Duration;

use serde_json::Value;
use tokio::sync::oneshot;
use zeromq::{PushSocket, Socket, SocketSend, ZmqMessage};

use sglang_srt::cache::{CachePageAllocator, RadixCache};
use sglang_srt::engine_info_bootstrap::{
    EngineInfoBootstrapService, serve_engine_info_bootstrap_with_shutdown,
};
use sglang_srt::pd_bootstrap::{
    MooncakeBootstrapKvCacheTransferExecutor, MooncakeDecodeBootstrapPublisher,
    MooncakeDecodeKvArgsRegistration, MooncakeDecodeTransferMetadata, PrefillBootstrapService,
    query_prefill_route, send_mooncake_kv_args_registration, send_mooncake_transfer_metadata,
    serve_mooncake_bootstrap_zmq_endpoints_with_shutdown,
    serve_mooncake_bootstrap_zmq_with_shutdown, serve_prefill_bootstrap_with_shutdown,
};
use sglang_srt::scheduler::{ScheduleBatch, ScheduledRequest, Scheduler};
use sglang_srt::transfer::{
    DecodeBootstrapRegistry, FakeKvCacheTransferExecutor, KvPoll, KvTransferModelWorker,
    MooncakeBatchId, MooncakeBatchReleaser, MooncakeError, MooncakeKvCacheLayout,
    MooncakeKvCacheTransferExecutor, MooncakeRemoteKvLayout, MooncakeTransferRequest,
    MooncakeTransferStatus, MooncakeTransferStatusCode, MooncakeTransferStatusReader,
    MooncakeTransferSubmitter, MooncakeTransferTarget,
};
use sglang_srt::types::{BootstrapRoom, DisaggregatedParams, RequestId, SamplingParams};
use sglang_srt::worker::{BatchGeneratedTokens, GeneratedToken, ModelWorker};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn engine_info_bootstrap_registers_and_serves_transfer_engine_info() {
    let addr = unused_local_addr();
    let service = EngineInfoBootstrapService::default();
    let observed = service.clone();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_engine_info_bootstrap_with_shutdown(addr, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let health = get_raw_with_retry(addr, "/health").await;
    assert!(health.starts_with("HTTP/1.1 200"));
    assert!(health.ends_with("OK"));

    let missing = get_raw_with_retry(addr, "/get_transfer_engine_info?rank=0").await;
    assert!(missing.starts_with("HTTP/1.1 404"));

    let register = put_json_with_retry(
        addr,
        "/register_transfer_engine_info",
        r#"{"tp_rank":0,"transfer_engine_info":{"session_id":"session-a","weights_info_dict":{"layer0":{"addr":4096,"nbytes":128},"lm_head":[1,2,3]}}}"#,
    )
    .await;
    assert!(register.starts_with("HTTP/1.1 200"));

    let info = get_json_with_retry(addr, "/get_transfer_engine_info?rank=0").await;
    assert_eq!(info["rank"], 0);
    assert_eq!(info["remote_instance_transfer_engine_info"][0], "session-a");
    assert_eq!(
        info["remote_instance_transfer_engine_info"][1]["layer0"]["addr"],
        4096
    );
    assert_eq!(
        observed
            .transfer_engine_info(0)
            .expect("registered info should be available")
            .session_id,
        "session-a"
    );

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prefill_transfer_executor_registers_dp_rank_for_bootstrap_queries() {
    let addr = unused_local_addr();
    let service = PrefillBootstrapService::default();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server_service = service.clone();
    let server = tokio::spawn(async move {
        serve_prefill_bootstrap_with_shutdown(addr, server_service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    {
        let mut state = service.state().lock().expect("state lock should be held");
        state
            .ingest_mooncake_bootstrap_frame(&vec![
                b"None".to_vec(),
                b"127.0.0.1".to_vec(),
                b"41010".to_vec(),
                b"decode-rank-session".to_vec(),
                pack_u64s(&[0x9000]),
                pack_u64s(&[]),
                pack_u64_lists(&[]),
                b"0".to_vec(),
                b"1".to_vec(),
                b"128".to_vec(),
            ])
            .expect("decode KV args should parse");
        state
            .ingest_mooncake_bootstrap_frame(&vec![
                b"99".to_vec(),
                b"127.0.0.1".to_vec(),
                b"41010".to_vec(),
                b"decode-rank-session".to_vec(),
                pack_i32s(&[0, 1, 2]),
                b"0".to_vec(),
                pack_i32_lists(&[]),
                b"1".to_vec(),
                b"3".to_vec(),
            ])
            .expect("decode transfer metadata should parse");
    }

    let inner = MooncakeKvCacheTransferExecutor::new(
        RecordingMooncakeSubmitter::default(),
        MooncakeKvCacheLayout {
            source_base_addr: 0x5000,
            page_size_bytes: 128,
            target_base_offset: 0,
        },
        MooncakeTransferTarget { target_id: 7 },
    );
    let transfer_executor = MooncakeBootstrapKvCacheTransferExecutor::new(service, inner)
        .with_metadata_wait_timeout(Duration::from_millis(50));
    let worker = KvTransferModelWorker::new(
        BootstrapMetadataWorker,
        DecodeBootstrapRegistry::default(),
        transfer_executor,
    );
    let mut scheduler =
        Scheduler::with_cache_resources(worker, RadixCache::default(), CachePageAllocator::new(4));
    scheduler.enqueue(
        ScheduledRequest::new(
            RequestId::from("pd-prefill-rank-register"),
            vec![1, 2, 3],
            SamplingParams::new(1),
        )
        .with_disaggregated_params(Some(DisaggregatedParams {
            bootstrap_host: addr.ip().to_string(),
            bootstrap_port: addr.port(),
            bootstrap_room: 99,
        }))
        .with_data_parallel_rank(5),
    );

    scheduler
        .dispatch_prefill_batch(1)
        .expect("prefill transfer should submit through Mooncake executor");

    let dp_ranks =
        post_json_value_with_retry(addr, "/query_dp_ranks", r#"{"bootstrap_rooms":[99]}"#).await;
    assert_eq!(dp_ranks["99"], 5);

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

#[test]
fn decode_bootstrap_publisher_uses_live_session_id_and_nonzero_kv_layout() {
    let publisher = MooncakeDecodeBootstrapPublisher::new("127.0.0.1", 41009, "127.0.0.1:41011")
        .with_kv_cache_layout(MooncakeKvCacheLayout {
            source_base_addr: 0x7000,
            page_size_bytes: 256,
            target_base_offset: 0,
        });
    let registration = publisher
        .kv_args_registration_for_test()
        .expect("KVArgs registration should exist");

    assert_eq!(registration.endpoint, "127.0.0.1");
    assert_eq!(registration.dst_port, 41009);
    assert_eq!(registration.mooncake_session_id, "127.0.0.1:41011");
    assert_eq!(registration.dst_kv_ptrs, vec![0x7000]);
    assert_eq!(registration.dst_kv_item_len, 256);
}

#[test]
fn prefill_bootstrap_builds_remote_kv_layouts_from_decode_metadata() {
    let service = PrefillBootstrapService::default();
    let kv_args_frame = vec![
        b"None".to_vec(),
        b"10.0.0.9".to_vec(),
        b"41001".to_vec(),
        b"session-a".to_vec(),
        pack_u64s(&[0x1000, 0x2000]),
        pack_u64s(&[]),
        pack_u64_lists(&[]),
        b"1".to_vec(),
        b"8".to_vec(),
        b"128".to_vec(),
    ];
    let transfer_frame = vec![
        b"77".to_vec(),
        b"10.0.0.9".to_vec(),
        b"41001".to_vec(),
        b"session-a".to_vec(),
        pack_i32s(&[3, 4, 5]),
        b"11".to_vec(),
        pack_i32_lists(&[]),
        b"1".to_vec(),
        b"64".to_vec(),
    ];

    let mut state = service.state().lock().expect("state lock should be held");
    state
        .ingest_mooncake_bootstrap_frame(&kv_args_frame)
        .expect("KVArgs registration frame should parse");
    state
        .ingest_mooncake_bootstrap_frame(&transfer_frame)
        .expect("transfer frame should parse");

    let layouts = state
        .remote_kv_layouts_for_room(77)
        .expect("remote KV layouts should build from matching metadata");

    assert_eq!(
        layouts,
        vec![(
            "session-a".to_string(),
            MooncakeRemoteKvLayout {
                dst_kv_ptrs: vec![0x1000, 0x2000],
                dst_kv_indices: vec![3, 4, 5],
                dst_kv_item_len: 128,
            },
        )]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mooncake_bootstrap_zmq_listener_ingests_multipart_transfer_metadata() {
    let addr = unused_local_addr();
    let endpoint = format!("tcp://{}", addr);
    let service = PrefillBootstrapService::default();
    let observed_service = service.clone();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_mooncake_bootstrap_zmq_with_shutdown(endpoint, service, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    send_zmq_multipart_with_retry(
        addr,
        vec![
            b"81".to_vec(),
            b"10.0.0.11".to_vec(),
            b"41003".to_vec(),
            b"session-zmq".to_vec(),
            pack_i32s(&[7, 8, 9]),
            b"13".to_vec(),
            pack_i32_lists(&[vec![60, 61]]),
            b"1".to_vec(),
            b"96".to_vec(),
        ],
    )
    .await;

    wait_until(|| {
        observed_service
            .state()
            .lock()
            .expect("state lock should be held")
            .transfer_status(81)
            == Some(KvPoll::WaitingForInput)
    })
    .await;

    let state = observed_service
        .state()
        .lock()
        .expect("state lock should be held");
    let room = state
        .transfer_room(81)
        .expect("transfer room should be tracked");
    assert_eq!(room.decode_prefix_len, Some(96));
    assert_eq!(room.transfers["session-zmq"].endpoint, "10.0.0.11");
    assert_eq!(room.transfers["session-zmq"].dst_kv_indices, vec![7, 8, 9]);

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mooncake_bootstrap_zmq_endpoint_group_ingests_metadata_from_each_port() {
    let first_addr = unused_local_addr();
    let second_addr = unused_local_addr();
    let service = PrefillBootstrapService::default();
    let observed_service = service.clone();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_mooncake_bootstrap_zmq_endpoints_with_shutdown(
            vec![
                format!("tcp://{}", first_addr),
                format!("tcp://{}", second_addr),
            ],
            service,
            async move {
                let _ = shutdown_rx.await;
            },
        )
        .await
    });

    send_zmq_multipart_with_retry(first_addr, transfer_frame(91, "first-port-session")).await;
    send_zmq_multipart_with_retry(second_addr, transfer_frame(92, "second-port-session")).await;

    wait_until(|| {
        let state = observed_service
            .state()
            .lock()
            .expect("state lock should be held");
        state.transfer_status(91) == Some(KvPoll::WaitingForInput)
            && state.transfer_status(92) == Some(KvPoll::WaitingForInput)
    })
    .await;

    let state = observed_service
        .state()
        .lock()
        .expect("state lock should be held");
    assert!(
        state
            .transfer_room(91)
            .expect("first room should be tracked")
            .transfers
            .contains_key("first-port-session")
    );
    assert!(
        state
            .transfer_room(92)
            .expect("second room should be tracked")
            .transfers
            .contains_key("second-port-session")
    );

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn decode_bootstrap_client_queries_route_and_sends_transfer_metadata() {
    let bootstrap_addr = unused_local_addr();
    let zmq_addr = unused_local_addr();
    let service = PrefillBootstrapService::default();
    let observed_service = service.clone();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        let (shutdown_tx, shutdown_rx_watch) = tokio::sync::watch::channel(false);
        let mut http_task = tokio::spawn(serve_prefill_bootstrap_with_shutdown(
            bootstrap_addr,
            service.clone(),
            watch_shutdown(shutdown_rx_watch.clone()),
        ));
        let mut zmq_task = tokio::spawn(serve_mooncake_bootstrap_zmq_with_shutdown(
            format!("tcp://{}", zmq_addr),
            service,
            watch_shutdown(shutdown_rx_watch),
        ));

        tokio::select! {
            _ = async move { let _ = shutdown_rx.await; } => {
                let _ = shutdown_tx.send(true);
                http_task.await.expect("HTTP bootstrap task should join")?;
                zmq_task.await.expect("ZMQ bootstrap task should join")?;
                Ok(())
            }
            result = &mut http_task => {
                let _ = shutdown_tx.send(true);
                zmq_task.await.expect("ZMQ bootstrap task should join")?;
                result.expect("HTTP bootstrap task should join")
            }
            result = &mut zmq_task => {
                let _ = shutdown_tx.send(true);
                http_task.await.expect("HTTP bootstrap task should join")?;
                result.expect("ZMQ bootstrap task should join")
            }
        }
    });

    let register_response = put_json_with_retry(
        bootstrap_addr,
        "/route",
        Box::leak(format!(
            r#"{{"attn_tp_size":1,"attn_tp_rank":0,"attn_cp_size":1,"attn_cp_rank":0,"attn_dp_size":1,"attn_dp_rank":0,"pp_size":1,"pp_rank":0,"system_dp_size":1,"system_dp_rank":0,"rank_ip":"127.0.0.1","rank_port":{},"page_size":64,"kv_cache_dtype":"auto","load_balance_method":"follow_bootstrap_room"}}"#,
            zmq_addr.port()
        ).into_boxed_str()),
    )
    .await;
    assert!(register_response.starts_with("HTTP/1.1 200"));

    let rank = query_prefill_route(&bootstrap_addr.to_string(), 0, 0, 0, 0)
        .await
        .expect("decode client should fetch prefill route");
    assert_eq!(rank.rank_ip, "127.0.0.1");
    assert_eq!(rank.rank_port, zmq_addr.port());

    send_mooncake_transfer_metadata(
        &rank.zmq_endpoint(),
        &MooncakeDecodeTransferMetadata {
            room: 93,
            endpoint: "127.0.0.1".to_string(),
            dst_port: 41005,
            mooncake_session_id: "decode-client-session".to_string(),
            dst_kv_indices: vec![9, 10, 11],
            dst_aux_index: Some(14),
            dst_state_indices: vec![vec![70, 71]],
            required_dst_info_num: 1,
            decode_prefix_len: Some(128),
            is_dummy: false,
        },
    )
    .await
    .expect("decode client should send metadata over ZMQ");

    wait_until(|| {
        observed_service
            .state()
            .lock()
            .expect("state lock should be held")
            .transfer_status(93)
            == Some(KvPoll::WaitingForInput)
    })
    .await;

    let state = observed_service
        .state()
        .lock()
        .expect("state lock should be held");
    let transfer = &state
        .transfer_room(93)
        .expect("room should be tracked")
        .transfers["decode-client-session"];
    assert_eq!(transfer.dst_kv_indices, vec![9, 10, 11]);
    assert_eq!(transfer.dst_aux_index, Some(14));
    assert_eq!(transfer.decode_prefix_len, Some(128));

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("servers should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn decode_bootstrap_client_sends_kv_args_registration() {
    let zmq_addr = unused_local_addr();
    let service = PrefillBootstrapService::default();
    let observed_service = service.clone();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        serve_mooncake_bootstrap_zmq_with_shutdown(format!("tcp://{}", zmq_addr), service, async {
            let _ = shutdown_rx.await;
        })
        .await
    });

    send_mooncake_kv_args_registration(
        &format!("tcp://{}", zmq_addr),
        &MooncakeDecodeKvArgsRegistration {
            endpoint: "127.0.0.1".to_string(),
            dst_port: 41006,
            mooncake_session_id: "decode-kvargs-session".to_string(),
            dst_kv_ptrs: vec![0x1000, 0x2000],
            dst_aux_ptrs: vec![0x3000],
            dst_state_data_ptrs: vec![vec![0x4000, 0x5000]],
            dst_tp_rank: 1,
            dst_attn_tp_size: 2,
            dst_kv_item_len: 128,
            dst_state_item_lens: vec![vec![16, 32]],
            dst_state_dim_per_tensor: vec![vec![4, 8]],
        },
    )
    .await
    .expect("decode client should send KV args registration over ZMQ");

    wait_until(|| {
        observed_service
            .state()
            .lock()
            .expect("state lock should be held")
            .decode_kv_args("decode-kvargs-session")
            .is_some()
    })
    .await;

    let state = observed_service
        .state()
        .lock()
        .expect("state lock should be held");
    let kv_args = state
        .decode_kv_args("decode-kvargs-session")
        .expect("decode KV args should be registered");
    assert_eq!(kv_args.endpoint, "127.0.0.1");
    assert_eq!(kv_args.dst_port, 41006);
    assert_eq!(kv_args.dst_kv_ptrs, vec![0x1000, 0x2000]);
    assert_eq!(kv_args.dst_state_data_ptrs, vec![vec![0x4000, 0x5000]]);
    assert_eq!(kv_args.dst_tp_rank, 1);
    assert_eq!(kv_args.dst_attn_tp_size, 2);

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("server should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn decode_worker_publishes_bootstrap_metadata_to_prefill_zmq_route() {
    let bootstrap_addr = unused_local_addr();
    let zmq_addr = unused_local_addr();
    let service = PrefillBootstrapService::default();
    let observed_service = service.clone();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        let (shutdown_tx, shutdown_rx_watch) = tokio::sync::watch::channel(false);
        let mut http_task = tokio::spawn(serve_prefill_bootstrap_with_shutdown(
            bootstrap_addr,
            service.clone(),
            watch_shutdown(shutdown_rx_watch.clone()),
        ));
        let mut zmq_task = tokio::spawn(serve_mooncake_bootstrap_zmq_with_shutdown(
            format!("tcp://{}", zmq_addr),
            service,
            watch_shutdown(shutdown_rx_watch),
        ));

        tokio::select! {
            _ = async move { let _ = shutdown_rx.await; } => {
                let _ = shutdown_tx.send(true);
                http_task.await.expect("HTTP bootstrap task should join")?;
                zmq_task.await.expect("ZMQ bootstrap task should join")?;
                Ok(())
            }
            result = &mut http_task => {
                let _ = shutdown_tx.send(true);
                zmq_task.await.expect("ZMQ bootstrap task should join")?;
                result.expect("HTTP bootstrap task should join")
            }
            result = &mut zmq_task => {
                let _ = shutdown_tx.send(true);
                http_task.await.expect("HTTP bootstrap task should join")?;
                result.expect("ZMQ bootstrap task should join")
            }
        }
    });

    let register_response = put_json_with_retry(
        bootstrap_addr,
        "/route",
        Box::leak(format!(
            r#"{{"attn_tp_size":1,"attn_tp_rank":0,"attn_cp_size":1,"attn_cp_rank":0,"attn_dp_size":1,"attn_dp_rank":0,"pp_size":1,"pp_rank":0,"system_dp_size":1,"system_dp_rank":0,"rank_ip":"127.0.0.1","rank_port":{},"page_size":64,"kv_cache_dtype":"auto","load_balance_method":"follow_bootstrap_room"}}"#,
            zmq_addr.port()
        ).into_boxed_str()),
    )
    .await;
    assert!(register_response.starts_with("HTTP/1.1 200"));

    let worker = KvTransferModelWorker::new(
        BootstrapMetadataWorker,
        DecodeBootstrapRegistry::default(),
        FakeKvCacheTransferExecutor::default(),
    )
    .with_decode_bootstrap_publisher(MooncakeDecodeBootstrapPublisher::new(
        "127.0.0.1",
        41007,
        "decode-worker-session",
    ));
    let mut scheduler =
        Scheduler::with_cache_resources(worker, RadixCache::default(), CachePageAllocator::new(4));
    scheduler.enqueue(
        ScheduledRequest::new(
            RequestId::from("pd-decode-zmq-publish"),
            vec![1, 2, 3],
            SamplingParams::new(1),
        )
        .with_disaggregated_params(Some(DisaggregatedParams {
            bootstrap_host: bootstrap_addr.ip().to_string(),
            bootstrap_port: bootstrap_addr.port(),
            bootstrap_room: 95,
        })),
    );

    scheduler
        .dispatch_prefill_batch(1)
        .expect("decode worker should publish bootstrap metadata through ZMQ");

    wait_until(|| {
        observed_service
            .state()
            .lock()
            .expect("state lock should be held")
            .transfer_status(95)
            == Some(KvPoll::WaitingForInput)
    })
    .await;

    let state = observed_service
        .state()
        .lock()
        .expect("state lock should be held");
    let transfer = &state
        .transfer_room(95)
        .expect("room should be tracked")
        .transfers["decode-worker-session"];
    assert_eq!(transfer.endpoint, "127.0.0.1");
    assert_eq!(transfer.dst_port, 41007);
    assert_eq!(transfer.dst_kv_indices, vec![0, 1, 2]);
    assert_eq!(transfer.decode_prefix_len, Some(3));

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("servers should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn decode_worker_registers_kv_args_before_transfer_metadata() {
    let bootstrap_addr = unused_local_addr();
    let zmq_addr = unused_local_addr();
    let service = PrefillBootstrapService::default();
    let observed_service = service.clone();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        let (shutdown_tx, shutdown_rx_watch) = tokio::sync::watch::channel(false);
        let mut http_task = tokio::spawn(serve_prefill_bootstrap_with_shutdown(
            bootstrap_addr,
            service.clone(),
            watch_shutdown(shutdown_rx_watch.clone()),
        ));
        let mut zmq_task = tokio::spawn(serve_mooncake_bootstrap_zmq_with_shutdown(
            format!("tcp://{}", zmq_addr),
            service,
            watch_shutdown(shutdown_rx_watch),
        ));

        tokio::select! {
            _ = async move { let _ = shutdown_rx.await; } => {
                let _ = shutdown_tx.send(true);
                http_task.await.expect("HTTP bootstrap task should join")?;
                zmq_task.await.expect("ZMQ bootstrap task should join")?;
                Ok(())
            }
            result = &mut http_task => {
                let _ = shutdown_tx.send(true);
                zmq_task.await.expect("ZMQ bootstrap task should join")?;
                result.expect("HTTP bootstrap task should join")
            }
            result = &mut zmq_task => {
                let _ = shutdown_tx.send(true);
                http_task.await.expect("HTTP bootstrap task should join")?;
                result.expect("ZMQ bootstrap task should join")
            }
        }
    });

    let register_response = put_json_with_retry(
        bootstrap_addr,
        "/route",
        Box::leak(format!(
            r#"{{"attn_tp_size":1,"attn_tp_rank":0,"attn_cp_size":1,"attn_cp_rank":0,"attn_dp_size":1,"attn_dp_rank":0,"pp_size":1,"pp_rank":0,"system_dp_size":1,"system_dp_rank":0,"rank_ip":"127.0.0.1","rank_port":{},"page_size":64,"kv_cache_dtype":"auto","load_balance_method":"follow_bootstrap_room"}}"#,
            zmq_addr.port()
        ).into_boxed_str()),
    )
    .await;
    assert!(register_response.starts_with("HTTP/1.1 200"));

    let worker = KvTransferModelWorker::new(
        BootstrapMetadataWorker,
        DecodeBootstrapRegistry::default(),
        FakeKvCacheTransferExecutor::default(),
    )
    .with_decode_bootstrap_publisher(
        MooncakeDecodeBootstrapPublisher::new("127.0.0.1", 41008, "decode-kvargs-session")
            .with_kv_cache_layout(MooncakeKvCacheLayout {
                source_base_addr: 0x9000,
                page_size_bytes: 128,
                target_base_offset: 0,
            }),
    );
    let mut scheduler =
        Scheduler::with_cache_resources(worker, RadixCache::default(), CachePageAllocator::new(4));
    scheduler.enqueue(
        ScheduledRequest::new(
            RequestId::from("pd-decode-kvargs"),
            vec![1, 2],
            SamplingParams::new(1),
        )
        .with_disaggregated_params(Some(DisaggregatedParams {
            bootstrap_host: bootstrap_addr.ip().to_string(),
            bootstrap_port: bootstrap_addr.port(),
            bootstrap_room: 96,
        })),
    );

    scheduler
        .dispatch_prefill_batch(1)
        .expect("decode worker should publish KV args and metadata through ZMQ");

    wait_until(|| {
        observed_service
            .state()
            .lock()
            .expect("state lock should be held")
            .remote_kv_layouts_for_room(96)
            .is_ok()
    })
    .await;

    let state = observed_service
        .state()
        .lock()
        .expect("state lock should be held");
    let kv_args = state
        .decode_kv_args("decode-kvargs-session")
        .expect("decode KV args should be registered before metadata is consumed");
    assert_eq!(kv_args.dst_kv_ptrs, vec![0x9000]);
    assert_eq!(kv_args.dst_kv_item_len, 128);
    let layouts = state
        .remote_kv_layouts_for_room(96)
        .expect("remote layout should include decode KV args and metadata");
    assert_eq!(
        layouts,
        vec![(
            "decode-kvargs-session".to_string(),
            MooncakeRemoteKvLayout {
                dst_kv_ptrs: vec![0x9000],
                dst_kv_indices: vec![0, 1],
                dst_kv_item_len: 128,
            }
        )]
    );

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("servers should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn decode_worker_reuses_kv_args_registration_for_multiple_rooms() {
    let bootstrap_addr = unused_local_addr();
    let zmq_addr = unused_local_addr();
    let service = PrefillBootstrapService::default();
    let observed_service = service.clone();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        let (shutdown_tx, shutdown_rx_watch) = tokio::sync::watch::channel(false);
        let mut http_task = tokio::spawn(serve_prefill_bootstrap_with_shutdown(
            bootstrap_addr,
            service.clone(),
            watch_shutdown(shutdown_rx_watch.clone()),
        ));
        let mut zmq_task = tokio::spawn(serve_mooncake_bootstrap_zmq_with_shutdown(
            format!("tcp://{}", zmq_addr),
            service,
            watch_shutdown(shutdown_rx_watch),
        ));

        tokio::select! {
            _ = async move { let _ = shutdown_rx.await; } => {
                let _ = shutdown_tx.send(true);
                http_task.await.expect("HTTP bootstrap task should join")?;
                zmq_task.await.expect("ZMQ bootstrap task should join")?;
                Ok(())
            }
            result = &mut http_task => {
                let _ = shutdown_tx.send(true);
                zmq_task.await.expect("ZMQ bootstrap task should join")?;
                result.expect("HTTP bootstrap task should join")
            }
            result = &mut zmq_task => {
                let _ = shutdown_tx.send(true);
                http_task.await.expect("HTTP bootstrap task should join")?;
                result.expect("ZMQ bootstrap task should join")
            }
        }
    });

    let register_response = put_json_with_retry(
        bootstrap_addr,
        "/route",
        Box::leak(format!(
            r#"{{"attn_tp_size":1,"attn_tp_rank":0,"attn_cp_size":1,"attn_cp_rank":0,"attn_dp_size":1,"attn_dp_rank":0,"pp_size":1,"pp_rank":0,"system_dp_size":1,"system_dp_rank":0,"rank_ip":"127.0.0.1","rank_port":{},"page_size":64,"kv_cache_dtype":"auto","load_balance_method":"follow_bootstrap_room"}}"#,
            zmq_addr.port()
        ).into_boxed_str()),
    )
    .await;
    assert!(register_response.starts_with("HTTP/1.1 200"));

    let worker = KvTransferModelWorker::new(
        BootstrapMetadataWorker,
        DecodeBootstrapRegistry::default(),
        FakeKvCacheTransferExecutor::default(),
    )
    .with_decode_bootstrap_publisher(
        MooncakeDecodeBootstrapPublisher::new("127.0.0.1", 41009, "decode-reuse-session")
            .with_kv_cache_layout(MooncakeKvCacheLayout {
                source_base_addr: 0xa000,
                page_size_bytes: 128,
                target_base_offset: 0,
            }),
    );
    let mut scheduler =
        Scheduler::with_cache_resources(worker, RadixCache::default(), CachePageAllocator::new(4));

    for room in [97, 98] {
        scheduler.enqueue(
            ScheduledRequest::new(
                RequestId::from(if room == 97 {
                    "pd-decode-kvargs-a"
                } else {
                    "pd-decode-kvargs-b"
                }),
                vec![1, 2],
                SamplingParams::new(1),
            )
            .with_disaggregated_params(Some(DisaggregatedParams {
                bootstrap_host: bootstrap_addr.ip().to_string(),
                bootstrap_port: bootstrap_addr.port(),
                bootstrap_room: room,
            })),
        );
        scheduler
            .dispatch_prefill_batch(1)
            .expect("decode worker should publish metadata for each room");
    }

    wait_until(|| {
        let state = observed_service
            .state()
            .lock()
            .expect("state lock should be held");
        state.remote_kv_layouts_for_room(97).is_ok() && state.remote_kv_layouts_for_room(98).is_ok()
    })
    .await;

    let state = observed_service
        .state()
        .lock()
        .expect("state lock should be held");
    assert_eq!(
        state.decode_kv_args_registration_count(),
        1,
        "decode KVArgs should be registered once per session/bootstrap endpoint"
    );
    assert_eq!(
        state
            .transfer_room(97)
            .expect("first room should be tracked")
            .transfers
            .len(),
        1
    );
    assert_eq!(
        state
            .transfer_room(98)
            .expect("second room should be tracked")
            .transfers
            .len(),
        1
    );

    shutdown_tx
        .send(())
        .expect("server should still be running");
    server
        .await
        .expect("server task should join")
        .expect("servers should stop cleanly");
}

struct BootstrapMetadataWorker;

impl ModelWorker for BootstrapMetadataWorker {
    fn generate_batch(&mut self, batch: &ScheduleBatch) -> BatchGeneratedTokens {
        BatchGeneratedTokens::from_batch(
            batch,
            batch
                .requests()
                .iter()
                .map(|_| GeneratedToken::unfinished(vec![1]))
                .collect(),
        )
        .expect("output shape should match batch")
    }
}

#[derive(Default)]
struct RecordingMooncakeSubmitter {
    submitted_requests: Vec<Vec<MooncakeTransferRequest>>,
    freed_batches: Vec<MooncakeBatchId>,
}

impl MooncakeTransferSubmitter for RecordingMooncakeSubmitter {
    fn submit_transfer(
        &mut self,
        requests: &mut [MooncakeTransferRequest],
    ) -> Result<MooncakeBatchId, MooncakeError> {
        self.submitted_requests.push(requests.to_vec());
        Ok(100 + self.submitted_requests.len() as MooncakeBatchId - 1)
    }
}

impl MooncakeTransferStatusReader for RecordingMooncakeSubmitter {
    fn transfer_status(
        &mut self,
        _batch_id: MooncakeBatchId,
        _task_id: usize,
    ) -> Result<MooncakeTransferStatus, MooncakeError> {
        Ok(MooncakeTransferStatus {
            status: MooncakeTransferStatusCode::Completed as i32,
            transferred_bytes: 0,
        })
    }
}

impl MooncakeBatchReleaser for RecordingMooncakeSubmitter {
    fn free_batch(&mut self, batch_id: MooncakeBatchId) -> Result<(), MooncakeError> {
        self.freed_batches.push(batch_id);
        Ok(())
    }
}

fn unused_local_addr() -> SocketAddr {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("ephemeral port should bind");
    listener
        .local_addr()
        .expect("ephemeral listener should have local addr")
}

async fn watch_shutdown(mut shutdown: tokio::sync::watch::Receiver<bool>) {
    while !*shutdown.borrow() {
        if shutdown.changed().await.is_err() {
            break;
        }
    }
}

fn transfer_frame(room: BootstrapRoom, session_id: &str) -> Vec<Vec<u8>> {
    vec![
        room.to_string().into_bytes(),
        b"10.0.0.12".to_vec(),
        b"41004".to_vec(),
        session_id.as_bytes().to_vec(),
        pack_i32s(&[1, 2]),
        b"3".to_vec(),
        pack_i32_lists(&[vec![4]]),
        b"1".to_vec(),
        b"0".to_vec(),
    ]
}

async fn send_zmq_multipart_with_retry(addr: SocketAddr, frames: Vec<Vec<u8>>) {
    let endpoint = format!("tcp://{}", addr);
    let mut last_error = None;

    for _ in 0..20 {
        let mut socket = PushSocket::new();
        match socket.connect(&endpoint).await {
            Ok(()) => {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                match socket.send(zmq_message(frames.clone())).await {
                    Ok(()) => return,
                    Err(error) => last_error = Some(error.to_string()),
                }
            }
            Err(error) => last_error = Some(error.to_string()),
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }

    panic!(
        "ZMQ client should send multipart frame: {}",
        last_error.unwrap_or_else(|| "no attempts ran".to_string())
    );
}

fn zmq_message(frames: Vec<Vec<u8>>) -> ZmqMessage {
    let mut frames = frames.into_iter();
    let first = frames
        .next()
        .expect("ZMQ multipart message should have at least one frame");
    let mut message = ZmqMessage::from(first);
    for frame in frames {
        message.push_back(frame.into());
    }
    message
}

async fn wait_until(mut predicate: impl FnMut() -> bool) {
    for _ in 0..50 {
        if predicate() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    panic!("condition should become true");
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
