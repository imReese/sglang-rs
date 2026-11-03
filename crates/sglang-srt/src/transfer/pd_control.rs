// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use zeromq::{PullSocket, PushSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

use crate::types::BootstrapRoom;

#[derive(Default)]
struct Room {
    token: Option<u32>,
    recurrent_state: Option<Vec<u8>>,
    terminal: Option<Result<(), String>>,
    abort_ack: bool,
}

#[derive(Default)]
struct Inbox {
    rooms: BTreeMap<BootstrapRoom, Room>,
    failure: Option<String>,
}

pub(crate) struct PdDecodeReceiver {
    port: u16,
    inbox: Arc<(Mutex<Inbox>, Condvar)>,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for PdDecodeReceiver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PdDecodeReceiver")
            .field("port", &self.port)
            .finish_non_exhaustive()
    }
}

impl PdDecodeReceiver {
    pub(crate) fn start(host: &str) -> Result<Self, String> {
        let endpoint = format!("tcp://{host}:0");
        let inbox = Arc::new((Mutex::new(Inbox::default()), Condvar::new()));
        let worker_inbox = Arc::clone(&inbox);
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let (stop, stop_rx) = tokio::sync::oneshot::channel();
        let thread = std::thread::Builder::new()
            .name("sglang-pd-receiver".into())
            .spawn(move || {
                let run = || -> Result<(), String> {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| e.to_string())?;
                    runtime.block_on(async {
                        let mut socket = PullSocket::new();
                        let bound = socket.bind(&endpoint).await.map_err(|e| e.to_string())?;
                        let address = bound.to_string();
                        let port = address
                            .rsplit(':')
                            .next()
                            .ok_or("missing receiver port")?
                            .parse::<u16>()
                            .map_err(|e| e.to_string())?;
                        ready_tx.send(Ok(port)).map_err(|e| e.to_string())?;
                        tokio::pin!(stop_rx);
                        loop {
                            tokio::select! {
                                _ = &mut stop_rx => return Ok(()),
                                message = socket.recv() => {
                                    let frames = message
                                        .map_err(|e| e.to_string())?
                                        .into_vec()
                                        .into_iter()
                                        .map(|b| b.to_vec())
                                        .collect::<Vec<_>>();
                                    let (lock, changed) = &*worker_inbox;
                                    let mut state = lock.lock().map_err(|e| e.to_string())?;
                                    if let Err(error) = ingest(&mut state, &frames) {
                                        state.failure = Some(error);
                                    }
                                    changed.notify_all();
                                }
                            }
                        }
                    })
                };
                if let Err(error) = run() {
                    let _ = ready_tx.send(Err(error.clone()));
                    let (lock, changed) = &*worker_inbox;
                    if let Ok(mut state) = lock.lock() {
                        state.failure = Some(error);
                    }
                    changed.notify_all();
                }
            })
            .map_err(|e| e.to_string())?;
        let port = match ready_rx
            .recv()
            .map_err(|e| e.to_string())
            .and_then(|ready| ready)
        {
            Ok(port) => port,
            Err(error) => {
                let _ = thread.join();
                return Err(error);
            }
        };
        Ok(Self {
            port,
            inbox,
            stop: Some(stop),
            thread: Some(thread),
        })
    }

    pub(crate) fn port(&self) -> u16 {
        self.port
    }

    pub(crate) fn register(&self, room: BootstrapRoom) -> Result<(), String> {
        let mut state = self.inbox.0.lock().map_err(|e| e.to_string())?;
        match state.rooms.entry(room) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(Room::default());
            }
            std::collections::btree_map::Entry::Occupied(_) => {
                return Err(format!("duplicate decode receive room {room}"));
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn receive(&self, room: BootstrapRoom) -> Result<u32, String> {
        self.wait(room, false)?
            .0
            .ok_or_else(|| format!("room {room} completed without a first token"))
    }

    pub(crate) fn receive_token_and_state(
        &self,
        room: BootstrapRoom,
    ) -> Result<(u32, Option<Vec<u8>>), String> {
        let (token, state) = self.wait(room, false)?;
        let token = token.ok_or_else(|| format!("room {room} completed without a first token"))?;
        Ok((token, state))
    }

    pub(crate) fn wait_drained(&self, room: BootstrapRoom) -> Result<(), String> {
        self.wait(room, true).map(|_| ())
    }

    pub(crate) fn is_drained(&self, room: BootstrapRoom) -> bool {
        self.inbox.0.lock().is_ok_and(|state| {
            state
                .rooms
                .get(&room)
                .is_some_and(|received| received.abort_ack || received.terminal.is_some())
        })
    }

    fn wait(
        &self,
        room: BootstrapRoom,
        cancel: bool,
    ) -> Result<(Option<u32>, Option<Vec<u8>>), String> {
        let deadline = Instant::now() + Duration::from_secs(300);
        let (lock, changed) = &*self.inbox;
        let mut state = lock.lock().map_err(|e| e.to_string())?;
        loop {
            if let Some(error) = &state.failure {
                return Err(error.clone());
            }
            let received = state
                .rooms
                .get(&room)
                .ok_or_else(|| format!("unregistered decode room {room}"))?;
            if cancel && (received.abort_ack || received.terminal.is_some()) {
                return Ok((received.token, received.recurrent_state.clone()));
            }
            match &received.terminal {
                Some(Err(error)) => return Err(error.clone()),
                Some(Ok(())) if received.token.is_some() => {
                    return Ok((received.token, received.recurrent_state.clone()));
                }
                _ => {}
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(format!(
                    "timed out waiting for PD control room {room}; KV memory remains reserved until sender drain acknowledgement"
                ));
            }
            state = changed
                .wait_timeout(state, remaining)
                .map_err(|e| e.to_string())?
                .0;
        }
    }

    pub(crate) fn remove(&self, room: BootstrapRoom) {
        if let Ok(mut state) = self.inbox.0.lock() {
            state.rooms.remove(&room);
        }
    }
}

impl Drop for PdDecodeReceiver {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn ascii<T: std::str::FromStr>(frame: Option<&Vec<u8>>) -> Result<T, String> {
    std::str::from_utf8(frame.ok_or("missing PD control field")?)
        .map_err(|e| e.to_string())?
        .parse()
        .map_err(|_| "invalid PD control field".into())
}

fn ingest(state: &mut Inbox, frames: &[Vec<u8>]) -> Result<(), String> {
    let header = frames.first().ok_or("empty PD control frame")?;
    let aux = header == b"AUX_DATA";
    let ack = header == b"ABORT_ACK";
    let room_id = ascii::<u64>(frames.get(usize::from(aux || ack)))?;
    let Some(room) = state.rooms.get_mut(&room_id) else {
        return Ok(());
    };
    if ack {
        if frames.len() < 3 || ascii::<u32>(frames.get(2))? != 0 {
            return Err("single-rank PD requires rank-zero drain acknowledgement".into());
        }
        room.abort_ack = true;
    } else if aux {
        if frames.len() != 6 {
            return Err("invalid AUX_DATA frame count".into());
        }
        let buffer = ascii::<usize>(frames.get(2))?;
        let _aux_index = ascii::<usize>(frames.get(3))?;
        let length = u32::from_be_bytes(
            frames[4]
                .as_slice()
                .try_into()
                .map_err(|_| "invalid AUX_DATA length")?,
        ) as usize;
        if frames[5].len() != length {
            return Err("AUX_DATA payload length mismatch".into());
        }
        if buffer == 0 {
            if length != 64 {
                return Err("first-token auxiliary buffer must contain 16 int32 values".into());
            }
            let token = i32::from_ne_bytes(
                frames[5][..4]
                    .try_into()
                    .map_err(|_| "invalid first token")?,
            );
            room.token = Some(u32::try_from(token).map_err(|_| "negative prefill token")?);
        } else if buffer == 1 {
            room.recurrent_state = Some(frames[5].clone());
        }
    } else {
        if frames.len() < 3 || ascii::<u32>(frames.get(2))? != 0 {
            return Err("single-rank PD requires rank-zero completion".into());
        }
        let status = ascii::<u8>(frames.get(1))?;
        room.terminal = Some(match status {
            4 => Ok(()),
            0 => Err(frames
                .get(3)
                .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
                .unwrap_or_else(|| "prefill transfer failed".into())),
            _ => return Err(format!("unexpected nonterminal PD control status {status}")),
        });
    }
    Ok(())
}

pub(crate) fn completion_frames(
    room: BootstrapRoom,
    token: Option<u32>,
    recurrent_state: Option<&[u8]>,
    error: Option<&str>,
    aux_index: i32,
) -> Result<Vec<Vec<Vec<u8>>>, String> {
    let mut frames = Vec::new();
    if let Some(token) = token {
        let token = i32::try_from(token)
            .map_err(|_| "prefill token exceeds community int32 auxiliary field")?;
        let mut data = vec![0; 64];
        data[..4].copy_from_slice(&token.to_ne_bytes());
        frames.push(vec![
            b"AUX_DATA".to_vec(),
            room.to_string().into_bytes(),
            b"0".to_vec(),
            aux_index.to_string().into_bytes(),
            64_u32.to_be_bytes().to_vec(),
            data,
        ]);
    }
    if let Some(state) = recurrent_state {
        let length_u32 =
            u32::try_from(state.len()).map_err(|_| "recurrent state length exceeds u32")?;
        frames.push(vec![
            b"AUX_DATA".to_vec(),
            room.to_string().into_bytes(),
            b"1".to_vec(),
            aux_index.to_string().into_bytes(),
            length_u32.to_be_bytes().to_vec(),
            state.to_vec(),
        ]);
    }
    frames.push(vec![
        room.to_string().into_bytes(),
        if error.is_some() {
            b"0".to_vec()
        } else {
            b"4".to_vec()
        },
        b"0".to_vec(),
        error.unwrap_or("").as_bytes().to_vec(),
    ]);
    Ok(frames)
}

pub(crate) fn send_frames(endpoint: String, frames: Vec<Vec<Vec<u8>>>) -> Result<(), String> {
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| e.to_string())?;
        runtime.block_on(async {
            let mut socket = PushSocket::new();
            tokio::time::timeout(Duration::from_secs(5), socket.connect(&endpoint))
                .await
                .map_err(|_| "PD control connect timed out")?
                .map_err(|e| e.to_string())?;
            for frame in frames {
                let mut parts = frame.into_iter();
                let mut message = ZmqMessage::from(parts.next().ok_or("empty control frame")?);
                for part in parts {
                    message.push_back(part.into());
                }
                tokio::time::timeout(Duration::from_secs(5), socket.send(message))
                    .await
                    .map_err(|_| "PD control send timed out")?
                    .map_err(|e| e.to_string())?;
            }
            Ok(())
        })
    })
    .join()
    .map_err(|_| "PD control sender panicked".to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_waits_for_both_success_and_first_token_over_real_control_socket() {
        let receiver = PdDecodeReceiver::start("127.0.0.1").expect("receiver");
        receiver.register(7).expect("room");
        let endpoint = format!("tcp://127.0.0.1:{}", receiver.port());
        let mut frames = completion_frames(7, Some(3), None, None, 0).expect("frames");
        let status = frames.pop().expect("status");
        send_frames(endpoint.clone(), vec![status]).expect("send completion before auxiliary data");
        let (output, result) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            output.send(receiver.receive(7)).expect("test reader");
        });
        assert!(result.recv_timeout(Duration::from_millis(20)).is_err());
        send_frames(endpoint, frames).expect("send missing first token");
        assert_eq!(
            result
                .recv_timeout(Duration::from_secs(2))
                .expect("receive completes")
                .expect("valid token"),
            3
        );
        thread.join().expect("receiver joins");
    }

    #[test]
    fn decode_receives_token_and_recurrent_state_over_real_control_socket() {
        let receiver = PdDecodeReceiver::start("127.0.0.1").expect("receiver");
        receiver.register(8).expect("room");
        let endpoint = format!("tcp://127.0.0.1:{}", receiver.port());
        let state_bytes = b"sample_recurrent_state_payload";
        let frames = completion_frames(8, Some(42), Some(state_bytes), None, 0).expect("frames");
        send_frames(endpoint, frames).expect("send completion frames");
        let (token, state) = receiver
            .receive_token_and_state(8)
            .expect("receive token and state");
        assert_eq!(token, 42);
        assert_eq!(state.as_deref(), Some(&state_bytes[..]));
    }

    #[test]
    fn malformed_auxiliary_data_and_nonzero_rank_fail_explicitly() {
        let mut state = Inbox::default();
        state.rooms.insert(9, Room::default());
        let mut frames = completion_frames(9, Some(2), None, None, 0).expect("frames");
        frames[0][4] = 63_u32.to_be_bytes().to_vec();
        assert!(ingest(&mut state, &frames[0]).is_err());
        frames[1][2] = b"1".to_vec();
        assert!(ingest(&mut state, &frames[1]).is_err());
        assert!(state.rooms[&9].terminal.is_none());
    }
}
