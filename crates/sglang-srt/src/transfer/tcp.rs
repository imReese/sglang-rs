// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::transfer::{
    KvBatchId, KvBatchReleaser, KvBufferEntry, KvMemoryRegistrar, KvSegmentId, KvSegmentOpener,
    KvTransferRequest, KvTransferStatus, KvTransferStatusCode, KvTransferStatusReader,
    KvTransferSubmitter, KvTransportError,
};

pub const TCP_TRANSFER_MAGIC: &[u8; 8] = b"SG_TCPKV";

#[derive(Clone, Copy, Debug)]
pub struct RegisteredBuffer {
    pub base_addr: usize,
    pub length: usize,
}

unsafe impl Send for RegisteredBuffer {}
unsafe impl Sync for RegisteredBuffer {}

pub struct TcpTransferEngine {
    local_endpoint: String,
    registered_buffers: Arc<Mutex<Vec<RegisteredBuffer>>>,
    listener_shutdown: Option<Arc<AtomicBool>>,
    listener_thread: Option<JoinHandle<()>>,
    targets: BTreeMap<i32, String>,
    streams: BTreeMap<i32, TcpStream>,
    next_target_id: i32,
    next_batch_id: u64,
    batches: BTreeMap<u64, Vec<KvTransferStatus>>,
}

impl TcpTransferEngine {
    pub fn new(bind_host: &str, bind_port: u16) -> Result<Self, String> {
        let listener = TcpListener::bind(format!("{bind_host}:{bind_port}")).map_err(|e| {
            format!("failed to bind TCP transfer engine on {bind_host}:{bind_port}: {e}")
        })?;
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;

        let local_addr = listener.local_addr().map_err(|e| e.to_string())?;
        let local_endpoint = format!("{}:{}", bind_host, local_addr.port());

        let registered_buffers = Arc::new(Mutex::new(Vec::new()));
        let shutdown = Arc::new(AtomicBool::new(false));

        let thread_shutdown = Arc::clone(&shutdown);
        let thread_buffers = Arc::clone(&registered_buffers);

        let listener_thread = std::thread::Builder::new()
            .name("tcp-transfer-server".into())
            .spawn(move || {
                Self::run_listener_loop(listener, thread_shutdown, thread_buffers);
            })
            .map_err(|e| e.to_string())?;

        Ok(Self {
            local_endpoint,
            registered_buffers,
            listener_shutdown: Some(shutdown),
            listener_thread: Some(listener_thread),
            targets: BTreeMap::new(),
            streams: BTreeMap::new(),
            next_target_id: 1,
            next_batch_id: 1,
            batches: BTreeMap::new(),
        })
    }

    pub fn local_endpoint(&self) -> &str {
        &self.local_endpoint
    }

    fn run_listener_loop(
        listener: TcpListener,
        shutdown: Arc<AtomicBool>,
        buffers: Arc<Mutex<Vec<RegisteredBuffer>>>,
    ) {
        while !shutdown.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((stream, _peer_addr)) => {
                    let bufs = Arc::clone(&buffers);
                    std::thread::spawn(move || {
                        let _ = Self::handle_connection(stream, bufs);
                    });
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(_) => break,
            }
        }
    }

    fn handle_connection(
        mut stream: TcpStream,
        buffers: Arc<Mutex<Vec<RegisteredBuffer>>>,
    ) -> std::io::Result<()> {
        stream.set_nonblocking(false).ok();
        stream.set_nodelay(true).ok();
        stream.set_read_timeout(Some(Duration::from_secs(10))).ok();
        stream.set_write_timeout(Some(Duration::from_secs(10))).ok();
        loop {
            let mut magic = [0u8; 8];
            match stream.read_exact(&mut magic) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
                Err(e) => return Err(e),
            }
            if &magic != TCP_TRANSFER_MAGIC {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "invalid TCP transfer magic header",
                ));
            }

            let mut offset_bytes = [0u8; 8];
            stream.read_exact(&mut offset_bytes)?;
            let target_offset = u64::from_le_bytes(offset_bytes);

            let mut len_bytes = [0u8; 8];
            stream.read_exact(&mut len_bytes)?;
            let length = u64::from_le_bytes(len_bytes);

            let destination_res = {
                let locked = buffers
                    .lock()
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
                resolve_destination(&locked, target_offset, length)
            };
            let dst = match destination_res {
                Ok(dst) => dst,
                Err(e) => {
                    let _ = stream.write_all(&[1u8]);
                    let _ = stream.flush();
                    return Err(e);
                }
            };

            unsafe {
                let mut bytes_read = 0;
                while bytes_read < length as usize {
                    let chunk = std::slice::from_raw_parts_mut(
                        dst.add(bytes_read),
                        (length as usize) - bytes_read,
                    );
                    let n = stream.read(chunk)?;
                    if n == 0 {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "connection closed unexpectedly while reading payload",
                        ));
                    }
                    bytes_read += n;
                }
            }

            // Acknowledge chunk
            stream.write_all(&[0u8])?;
            stream.flush()?;
        }
    }
}

fn resolve_destination(
    buffers: &[RegisteredBuffer],
    target_offset: u64,
    length: u64,
) -> Result<*mut u8, std::io::Error> {
    if buffers.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no registered KV cache buffer on receiver",
        ));
    }

    // 1. Check if target_offset is an absolute virtual address within any registered buffer
    for buf in buffers {
        let buf_start = buf.base_addr as u64;
        let buf_end = buf_start.saturating_add(buf.length as u64);
        if target_offset >= buf_start && target_offset.saturating_add(length) <= buf_end {
            return Ok(target_offset as *mut u8);
        }
    }

    // 2. Otherwise treat target_offset as a relative offset into the first buffer
    let first = &buffers[0];
    if (target_offset as usize).saturating_add(length as usize) <= first.length {
        let dst = (first.base_addr as *mut u8).wrapping_add(target_offset as usize);
        return Ok(dst);
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        format!(
            "target offset {} + len {} does not fit into any registered buffer (count: {})",
            target_offset,
            length,
            buffers.len()
        ),
    ))
}

impl Drop for TcpTransferEngine {
    fn drop(&mut self) {
        if let Some(flag) = self.listener_shutdown.take() {
            flag.store(true, Ordering::Relaxed);
        }
        if let Some(thread) = self.listener_thread.take() {
            let _ = thread.join();
        }
    }
}

#[derive(Clone)]
pub struct SharedTcpTransferEngine {
    inner: Arc<Mutex<TcpTransferEngine>>,
}

impl SharedTcpTransferEngine {
    pub fn new(bind_host: impl Into<String>, bind_port: u16) -> Result<Self, String> {
        let engine = TcpTransferEngine::new(&bind_host.into(), bind_port)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(engine)),
        })
    }

    pub fn local_endpoint(&self) -> Result<String, KvTransportError> {
        let engine = self
            .inner
            .lock()
            .map_err(|_| KvTransportError::StatusQueryFailed("lock poisoned".to_string()))?;
        Ok(engine.local_endpoint().to_string())
    }

    fn send_transfer_chunk(
        stream: &mut TcpStream,
        target_offset: u64,
        length: u64,
        src: &[u8],
    ) -> std::io::Result<()> {
        stream.write_all(TCP_TRANSFER_MAGIC)?;
        stream.write_all(&target_offset.to_le_bytes())?;
        stream.write_all(&length.to_le_bytes())?;
        stream.write_all(src)?;
        stream.flush()?;

        let mut ack = [0u8; 1];
        stream.read_exact(&mut ack)?;
        if ack[0] != 0 {
            return Err(std::io::Error::other("remote receiver rejected chunk"));
        }
        Ok(())
    }
}

impl KvMemoryRegistrar for SharedTcpTransferEngine {
    fn register_memory_batch(
        &mut self,
        buffers: &mut [KvBufferEntry],
        _location: &str,
    ) -> Result<(), KvTransportError> {
        let engine = self
            .inner
            .lock()
            .map_err(|_| KvTransportError::RegisterMemoryFailed("lock poisoned".to_string()))?;
        let mut bufs = engine.registered_buffers.lock().map_err(|_| {
            KvTransportError::RegisterMemoryFailed("buffers lock poisoned".to_string())
        })?;
        bufs.clear();
        for buffer in buffers.iter() {
            bufs.push(RegisteredBuffer {
                base_addr: buffer.addr as usize,
                length: buffer.length,
            });
        }
        Ok(())
    }

    fn unregister_memory_batch(
        &mut self,
        _addrs: &mut [*mut c_void],
    ) -> Result<(), KvTransportError> {
        let engine = self
            .inner
            .lock()
            .map_err(|_| KvTransportError::UnregisterMemoryFailed("lock poisoned".to_string()))?;
        let mut bufs = engine.registered_buffers.lock().map_err(|_| {
            KvTransportError::UnregisterMemoryFailed("buffers lock poisoned".to_string())
        })?;
        bufs.clear();
        Ok(())
    }
}

impl KvSegmentOpener for SharedTcpTransferEngine {
    fn open_segment(&mut self, segment: &str) -> Result<KvSegmentId, KvTransportError> {
        let mut engine = self
            .inner
            .lock()
            .map_err(|e| KvTransportError::OpenSegmentFailed(e.to_string()))?;
        let target_id = engine.next_target_id;
        engine.next_target_id += 1;
        engine.targets.insert(target_id, segment.to_string());
        Ok(target_id)
    }
}

impl KvTransferSubmitter for SharedTcpTransferEngine {
    fn submit_transfer(
        &mut self,
        requests: &mut [KvTransferRequest],
    ) -> Result<KvBatchId, KvTransportError> {
        let mut engine = self
            .inner
            .lock()
            .map_err(|_| KvTransportError::SubmitTransferFailed("lock poisoned".to_string()))?;

        let batch_id = engine.next_batch_id;
        engine.next_batch_id += 1;

        let mut statuses = Vec::with_capacity(requests.len());

        // Group or execute requests
        for request in requests.iter() {
            let target_addr = engine
                .targets
                .get(&request.target_id)
                .cloned()
                .ok_or_else(|| {
                    KvTransportError::OpenSegmentFailed(format!(
                        "target ID {} not opened",
                        request.target_id
                    ))
                })?;

            let src = unsafe {
                std::slice::from_raw_parts(request.source as *const u8, request.length as usize)
            };

            let mut success = false;
            for attempt in 0..2 {
                if let std::collections::btree_map::Entry::Vacant(entry) =
                    engine.streams.entry(request.target_id)
                {
                    let addrs = target_addr
                        .to_socket_addrs()
                        .map_err(|e| KvTransportError::ConnectionFailed(e.to_string()))?
                        .collect::<Vec<SocketAddr>>();
                    let socket_addr = match addrs.first() {
                        Some(a) => *a,
                        None => break,
                    };

                    match TcpStream::connect_timeout(&socket_addr, Duration::from_secs(5)) {
                        Ok(stream) => {
                            stream.set_nodelay(true).ok();
                            stream.set_read_timeout(Some(Duration::from_secs(10))).ok();
                            stream.set_write_timeout(Some(Duration::from_secs(10))).ok();
                            entry.insert(stream);
                        }
                        Err(_) => break,
                    }
                }

                let Some(stream) = engine.streams.get_mut(&request.target_id) else {
                    break;
                };

                match Self::send_transfer_chunk(stream, request.target_offset, request.length, src)
                {
                    Ok(()) => {
                        success = true;
                        break;
                    }
                    Err(_) => {
                        engine.streams.remove(&request.target_id);
                        if attempt == 1 {
                            break;
                        }
                    }
                }
            }

            if success {
                statuses.push(KvTransferStatus {
                    status: KvTransferStatusCode::Completed as i32,
                    transferred_bytes: request.length,
                });
            } else {
                statuses.push(KvTransferStatus {
                    status: KvTransferStatusCode::Failed as i32,
                    transferred_bytes: 0,
                });
            }
        }

        engine.batches.insert(batch_id, statuses);
        Ok(batch_id)
    }
}

impl KvTransferStatusReader for SharedTcpTransferEngine {
    fn transfer_status(
        &mut self,
        batch_id: KvBatchId,
        task_id: usize,
    ) -> Result<KvTransferStatus, KvTransportError> {
        let engine = self
            .inner
            .lock()
            .map_err(|_| KvTransportError::StatusQueryFailed("lock poisoned".to_string()))?;
        let batch = engine.batches.get(&batch_id).ok_or_else(|| {
            KvTransportError::StatusQueryFailed(format!("batch {batch_id} not found"))
        })?;
        let status = batch.get(task_id).copied().ok_or_else(|| {
            KvTransportError::StatusQueryFailed(format!(
                "task {task_id} not found in batch {batch_id}"
            ))
        })?;
        Ok(status)
    }
}

impl KvBatchReleaser for SharedTcpTransferEngine {
    fn free_batch(&mut self, batch_id: KvBatchId) -> Result<(), KvTransportError> {
        let mut engine = self
            .inner
            .lock()
            .map_err(|_| KvTransportError::FreeBatchFailed("lock poisoned".to_string()))?;
        engine.batches.remove(&batch_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn can_connect_loopback_tcp() -> bool {
        let Ok(listener) = std::net::TcpListener::bind("127.0.0.1:0") else {
            return false;
        };
        let Ok(addr) = listener.local_addr() else {
            return false;
        };
        match std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(100)) {
            Ok(_) => true,
            Err(e) if e.raw_os_error() == Some(1) => false,
            Err(_) => false,
        }
    }

    #[test]
    fn test_tcp_transfer_engine_creation_and_endpoint() {
        let engine = SharedTcpTransferEngine::new("127.0.0.1", 0).expect("engine creation");
        let endpoint = engine.local_endpoint().expect("local endpoint");
        assert!(endpoint.starts_with("127.0.0.1:"));
        let port: u16 = endpoint["127.0.0.1:".len()..].parse().expect("valid port");
        assert!(port > 0);
    }

    #[test]
    fn test_tcp_transfer_single_buffer_relative_offset() {
        if !can_connect_loopback_tcp() {
            eprintln!("skipping test: loopback TCP connect blocked by sandbox environment");
            return;
        }
        let mut rx = SharedTcpTransferEngine::new("127.0.0.1", 0).expect("rx");
        let mut rx_buf = vec![0u8; 1024];
        let mut entries = [KvBufferEntry {
            addr: rx_buf.as_mut_ptr() as *mut c_void,
            length: rx_buf.len(),
        }];
        rx.register_memory_batch(&mut entries, "cpu")
            .expect("register");

        let mut tx = SharedTcpTransferEngine::new("127.0.0.1", 0).expect("tx");
        let endpoint = rx.local_endpoint().expect("endpoint");
        let target_id = tx.open_segment(&endpoint).expect("open segment");

        let message = b"Hello from SGLang TCP KV Cache transfer!";
        let mut requests = [KvTransferRequest {
            opcode: 0,
            source: message.as_ptr() as *mut c_void,
            target_id,
            target_offset: 64,
            length: message.len() as u64,
        }];

        let batch_id = tx.submit_transfer(&mut requests).expect("submit");
        let status = tx.transfer_status(batch_id, 0).expect("status");
        assert_eq!(status.status, KvTransferStatusCode::Completed as i32);
        assert_eq!(status.transferred_bytes, message.len() as u64);

        assert_eq!(&rx_buf[64..64 + message.len()], message);

        tx.free_batch(batch_id).expect("free batch");
    }

    #[test]
    fn test_tcp_transfer_absolute_address_resolution() {
        if !can_connect_loopback_tcp() {
            eprintln!("skipping test: loopback TCP connect blocked by sandbox environment");
            return;
        }
        let mut rx = SharedTcpTransferEngine::new("127.0.0.1", 0).expect("rx");
        let mut rx_buf = vec![0u8; 2048];
        let base_addr = rx_buf.as_mut_ptr() as usize;
        let mut entries = [KvBufferEntry {
            addr: rx_buf.as_mut_ptr() as *mut c_void,
            length: rx_buf.len(),
        }];
        rx.register_memory_batch(&mut entries, "cpu")
            .expect("register");

        let mut tx = SharedTcpTransferEngine::new("127.0.0.1", 0).expect("tx");
        let endpoint = rx.local_endpoint().expect("endpoint");
        let target_id = tx.open_segment(&endpoint).expect("open segment");

        let payload = b"Direct virtual address resolution test";
        let target_offset = (base_addr + 128) as u64;
        let mut requests = [KvTransferRequest {
            opcode: 0,
            source: payload.as_ptr() as *mut c_void,
            target_id,
            target_offset,
            length: payload.len() as u64,
        }];

        let batch_id = tx.submit_transfer(&mut requests).expect("submit");
        let status = tx.transfer_status(batch_id, 0).expect("status");
        assert_eq!(status.status, KvTransferStatusCode::Completed as i32);
        assert_eq!(status.transferred_bytes, payload.len() as u64);

        assert_eq!(&rx_buf[128..128 + payload.len()], payload);
    }

    #[test]
    fn test_tcp_transfer_multiple_pages() {
        if !can_connect_loopback_tcp() {
            eprintln!("skipping test: loopback TCP connect blocked by sandbox environment");
            return;
        }
        let mut rx = SharedTcpTransferEngine::new("127.0.0.1", 0).expect("rx");
        let mut rx_buf = vec![0u8; 4096];
        let mut entries = [KvBufferEntry {
            addr: rx_buf.as_mut_ptr() as *mut c_void,
            length: rx_buf.len(),
        }];
        rx.register_memory_batch(&mut entries, "cpu")
            .expect("register");

        let mut tx = SharedTcpTransferEngine::new("127.0.0.1", 0).expect("tx");
        let endpoint = rx.local_endpoint().expect("endpoint");
        let target_id = tx.open_segment(&endpoint).expect("open segment");

        let page1 = [0xAAu8; 128];
        let page2 = [0xBBu8; 128];
        let page3 = [0xCCu8; 128];

        let mut requests = [
            KvTransferRequest {
                opcode: 0,
                source: page1.as_ptr() as *mut c_void,
                target_id,
                target_offset: 0,
                length: 128,
            },
            KvTransferRequest {
                opcode: 0,
                source: page2.as_ptr() as *mut c_void,
                target_id,
                target_offset: 256,
                length: 128,
            },
            KvTransferRequest {
                opcode: 0,
                source: page3.as_ptr() as *mut c_void,
                target_id,
                target_offset: 512,
                length: 128,
            },
        ];

        let batch_id = tx.submit_transfer(&mut requests).expect("submit");
        for task_id in 0..3 {
            let status = tx.transfer_status(batch_id, task_id).expect("status");
            assert_eq!(status.status, KvTransferStatusCode::Completed as i32);
            assert_eq!(status.transferred_bytes, 128);
        }

        assert_eq!(&rx_buf[0..128], &page1[..]);
        assert_eq!(&rx_buf[128..256], &[0u8; 128]);
        assert_eq!(&rx_buf[256..384], &page2[..]);
        assert_eq!(&rx_buf[384..512], &[0u8; 128]);
        assert_eq!(&rx_buf[512..640], &page3[..]);
    }

    #[test]
    fn test_tcp_transfer_out_of_bounds_fails() {
        let mut rx = SharedTcpTransferEngine::new("127.0.0.1", 0).expect("rx");
        let mut rx_buf = vec![0u8; 256];
        let mut entries = [KvBufferEntry {
            addr: rx_buf.as_mut_ptr() as *mut c_void,
            length: rx_buf.len(),
        }];
        rx.register_memory_batch(&mut entries, "cpu")
            .expect("register");

        let mut tx = SharedTcpTransferEngine::new("127.0.0.1", 0).expect("tx");
        let endpoint = rx.local_endpoint().expect("endpoint");
        let target_id = tx.open_segment(&endpoint).expect("open segment");

        let payload = [1u8; 128];
        let mut requests = [KvTransferRequest {
            opcode: 0,
            source: payload.as_ptr() as *mut c_void,
            target_id,
            target_offset: 200, // 200 + 128 = 328 > 256
            length: 128,
        }];

        let batch_id = tx.submit_transfer(&mut requests).expect("submit");
        let status = tx.transfer_status(batch_id, 0).expect("status");
        assert_eq!(status.status, KvTransferStatusCode::Failed as i32);
    }
}
