// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use sglang_srt::tcp_transfer::SharedTcpTransferEngine;
use sglang_srt::transfer::{
    KvBatchReleaser, KvBufferEntry, KvMemoryRegistrar, KvSegmentOpener, KvTransferRequest,
    KvTransferStatusCode, KvTransferStatusReader, KvTransferSubmitter,
};
use std::ffi::c_void;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SGLang Native TCP KV Cache Transfer Demo (macOS / Linux) ===");

    // 1. Decode side: start engine and register memory buffer
    let mut decode_engine = SharedTcpTransferEngine::new("127.0.0.1", 0)
        .map_err(|e| format!("Failed to create decode engine: {e}"))?;
    let decode_endpoint = decode_engine.local_endpoint()?;
    println!(
        "[Decode] Engine started and listening on: {}",
        decode_endpoint
    );

    let mut decode_kv_buffer = vec![0u8; 1024];
    let mut decode_entries = [KvBufferEntry {
        addr: decode_kv_buffer.as_mut_ptr() as *mut c_void,
        length: decode_kv_buffer.len(),
    }];
    decode_engine.register_memory_batch(&mut decode_entries, "cpu")?;
    println!(
        "[Decode] Registered KV cache buffer: addr={:p}, len={} bytes",
        decode_kv_buffer.as_ptr(),
        decode_kv_buffer.len()
    );

    // 2. Prefill side: connect to Decode and transfer data
    let mut prefill_engine = SharedTcpTransferEngine::new("127.0.0.1", 0)
        .map_err(|e| format!("Failed to create prefill engine: {e}"))?;
    let target_id = prefill_engine.open_segment(&decode_endpoint)?;
    println!(
        "[Prefill] Connected to Decode endpoint '{}', assigned target_id={}",
        decode_endpoint, target_id
    );

    let payload =
        b"Hello from SGLang Prefill! This KV cache chunk was transferred over native TCP on macOS.";
    let target_offset = 128u64;

    println!(
        "[Prefill] Submitting transfer: length={} bytes -> target_offset={}",
        payload.len(),
        target_offset
    );

    let mut requests = [KvTransferRequest {
        opcode: 0,
        source: payload.as_ptr() as *mut c_void,
        target_id,
        target_offset,
        length: payload.len() as u64,
    }];

    let batch_id = prefill_engine.submit_transfer(&mut requests)?;
    let status = prefill_engine.transfer_status(batch_id, 0)?;

    println!(
        "[Prefill] Transfer finished: batch_id={}, status={}, transferred_bytes={}",
        batch_id,
        if status.status == KvTransferStatusCode::Completed as i32 {
            "COMPLETED"
        } else {
            "FAILED"
        },
        status.transferred_bytes
    );

    // 3. Verify Decode buffer
    let received_slice =
        &decode_kv_buffer[target_offset as usize..target_offset as usize + payload.len()];
    let received_str = std::str::from_utf8(received_slice)?;
    println!(
        "[Decode] Verified received payload in memory: \"{}\"",
        received_str
    );

    assert_eq!(received_slice, payload);
    println!("=== Demo Completed Successfully! ===");

    prefill_engine.free_batch(batch_id)?;
    Ok(())
}
