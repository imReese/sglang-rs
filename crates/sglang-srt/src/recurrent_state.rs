use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::types::RequestId;

#[derive(Debug)]
pub struct RecurrentStateBatchSlots {
    slots: Vec<u32>,
    newly_assigned: Vec<u32>,
}

impl RecurrentStateBatchSlots {
    pub fn slots(&self) -> &[u32] {
        &self.slots
    }

    pub fn newly_assigned(&self) -> &[u32] {
        &self.newly_assigned
    }
}

#[derive(Debug)]
pub struct RecurrentStateSlots {
    capacity: usize,
    free_slots: Vec<u32>,
    assigned_slots: HashMap<String, u32>,
}

impl RecurrentStateSlots {
    pub fn new(capacity: usize) -> Result<Self, RecurrentStateSlotError> {
        if capacity == 0 {
            return Err(RecurrentStateSlotError::ZeroCapacity);
        }
        let capacity_u32 = u32::try_from(capacity)
            .map_err(|_| RecurrentStateSlotError::CapacityTooLarge(capacity))?;
        Ok(Self {
            capacity,
            free_slots: (0..capacity_u32).rev().collect(),
            assigned_slots: HashMap::with_capacity(capacity),
        })
    }

    pub fn assign_batch(
        &mut self,
        request_ids: &[RequestId],
    ) -> Result<RecurrentStateBatchSlots, RecurrentStateSlotError> {
        let mut seen = HashSet::with_capacity(request_ids.len());
        for request_id in request_ids {
            if !seen.insert(request_id.as_str()) {
                return Err(RecurrentStateSlotError::DuplicateRequest(
                    request_id.as_str().to_string(),
                ));
            }
        }

        let required = request_ids
            .iter()
            .filter(|request_id| !self.assigned_slots.contains_key(request_id.as_str()))
            .count();
        if required > self.free_slots.len() {
            return Err(RecurrentStateSlotError::Exhausted {
                capacity: self.capacity,
                assigned: self.assigned_slots.len(),
                requested: required,
            });
        }

        let mut slots = Vec::with_capacity(request_ids.len());
        let mut newly_assigned = Vec::with_capacity(required);
        for request_id in request_ids {
            let slot = match self.assigned_slots.get(request_id.as_str()).copied() {
                Some(slot) => slot,
                None => {
                    let slot = self
                        .free_slots
                        .pop()
                        .ok_or(RecurrentStateSlotError::Exhausted {
                            capacity: self.capacity,
                            assigned: self.assigned_slots.len(),
                            requested: 1,
                        })?;
                    self.assigned_slots
                        .insert(request_id.as_str().to_string(), slot);
                    newly_assigned.push(slot);
                    slot
                }
            };
            slots.push(slot);
        }
        Ok(RecurrentStateBatchSlots {
            slots,
            newly_assigned,
        })
    }

    pub fn release(&mut self, request_id: &RequestId) -> Option<u32> {
        let slot = self.assigned_slots.remove(request_id.as_str())?;
        self.free_slots.push(slot);
        Some(slot)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RecurrentStateSlotError {
    ZeroCapacity,
    CapacityTooLarge(usize),
    DuplicateRequest(String),
    Exhausted {
        capacity: usize,
        assigned: usize,
        requested: usize,
    },
}

impl fmt::Display for RecurrentStateSlotError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCapacity => {
                formatter.write_str("recurrent state slot capacity must be non-zero")
            }
            Self::CapacityTooLarge(capacity) => write!(
                formatter,
                "recurrent state slot capacity {capacity} exceeds the u32 device index range"
            ),
            Self::DuplicateRequest(request_id) => write!(
                formatter,
                "recurrent state batch contains duplicate request {request_id}"
            ),
            Self::Exhausted {
                capacity,
                assigned,
                requested,
            } => write!(
                formatter,
                "recurrent state slots are exhausted: capacity={capacity}, assigned={assigned}, new requests={requested}"
            ),
        }
    }
}

impl std::error::Error for RecurrentStateSlotError {}

#[derive(Clone, Debug, PartialEq)]
pub struct RecurrentLayerStatePayload {
    pub conv: Vec<f32>,
    pub recurrent: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecurrentStatePayload {
    pub next_position: usize,
    pub layer_states: Vec<RecurrentLayerStatePayload>,
}

impl RecurrentStatePayload {
    pub const MAGIC: u32 = 0x5347_4C52; // 'SGLR'

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&Self::MAGIC.to_le_bytes());
        let pos_u64 = self.next_position as u64;
        bytes.extend_from_slice(&pos_u64.to_le_bytes());
        let layer_count_u32 = self.layer_states.len() as u32;
        bytes.extend_from_slice(&layer_count_u32.to_le_bytes());
        for layer in &self.layer_states {
            let conv_len_u32 = layer.conv.len() as u32;
            bytes.extend_from_slice(&conv_len_u32.to_le_bytes());
            for value in &layer.conv {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let rec_len_u32 = layer.recurrent.len() as u32;
            bytes.extend_from_slice(&rec_len_u32.to_le_bytes());
            for value in &layer.recurrent {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 16 {
            return Err("recurrent state payload too short".to_string());
        }
        let magic = u32::from_le_bytes(
            bytes[0..4]
                .try_into()
                .map_err(|_| "failed to read magic".to_string())?,
        );
        if magic != Self::MAGIC {
            return Err(format!(
                "invalid recurrent state magic: expected {:#X}, found {magic:#X}",
                Self::MAGIC
            ));
        }
        let next_position = u64::from_le_bytes(
            bytes[4..12]
                .try_into()
                .map_err(|_| "failed to read next_position".to_string())?,
        ) as usize;
        let layer_count = u32::from_le_bytes(
            bytes[12..16]
                .try_into()
                .map_err(|_| "failed to read layer_count".to_string())?,
        ) as usize;

        let mut offset = 16;
        let mut layer_states = Vec::with_capacity(layer_count);
        for _ in 0..layer_count {
            if offset + 4 > bytes.len() {
                return Err("unexpected EOF reading conv length".to_string());
            }
            let conv_len = u32::from_le_bytes(
                bytes[offset..offset + 4]
                    .try_into()
                    .map_err(|_| "failed to read conv length".to_string())?,
            ) as usize;
            offset += 4;
            let conv_bytes = conv_len
                .checked_mul(4)
                .ok_or_else(|| "conv byte length overflow".to_string())?;
            if offset + conv_bytes > bytes.len() {
                return Err("unexpected EOF reading conv data".to_string());
            }
            let mut conv = Vec::with_capacity(conv_len);
            for i in 0..conv_len {
                let elem_offset = offset + i * 4;
                conv.push(f32::from_le_bytes(
                    bytes[elem_offset..elem_offset + 4]
                        .try_into()
                        .map_err(|_| "failed to read conv float".to_string())?,
                ));
            }
            offset += conv_bytes;

            if offset + 4 > bytes.len() {
                return Err("unexpected EOF reading recurrent length".to_string());
            }
            let rec_len = u32::from_le_bytes(
                bytes[offset..offset + 4]
                    .try_into()
                    .map_err(|_| "failed to read recurrent length".to_string())?,
            ) as usize;
            offset += 4;
            let rec_bytes = rec_len
                .checked_mul(4)
                .ok_or_else(|| "recurrent byte length overflow".to_string())?;
            if offset + rec_bytes > bytes.len() {
                return Err("unexpected EOF reading recurrent data".to_string());
            }
            let mut recurrent = Vec::with_capacity(rec_len);
            for i in 0..rec_len {
                let elem_offset = offset + i * 4;
                recurrent.push(f32::from_le_bytes(
                    bytes[elem_offset..elem_offset + 4]
                        .try_into()
                        .map_err(|_| "failed to read recurrent float".to_string())?,
                ));
            }
            offset += rec_bytes;

            layer_states.push(RecurrentLayerStatePayload { conv, recurrent });
        }

        if offset != bytes.len() {
            return Err(format!(
                "trailing bytes in recurrent state payload: read {offset}, total {}",
                bytes.len()
            ));
        }

        Ok(Self {
            next_position,
            layer_states,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_assignment_reuses_requests_and_released_slots() {
        let mut slots = RecurrentStateSlots::new(2).expect("slot pool should initialize");
        let first = slots
            .assign_batch(&[RequestId::from("request-a"), RequestId::from("request-b")])
            .expect("first batch should fit");
        assert_eq!(first.slots(), &[0, 1]);
        assert_eq!(first.newly_assigned(), &[0, 1]);

        let reused = slots
            .assign_batch(&[RequestId::from("request-b")])
            .expect("existing request should reuse its slot");
        assert_eq!(reused.slots(), &[1]);
        assert!(reused.newly_assigned().is_empty());

        assert_eq!(slots.release(&RequestId::from("request-a")), Some(0));
        let replacement = slots
            .assign_batch(&[RequestId::from("request-c")])
            .expect("released slot should be reusable");
        assert_eq!(replacement.slots(), &[0]);
        assert_eq!(replacement.newly_assigned(), &[0]);
    }

    #[test]
    fn batch_assignment_is_atomic_on_duplicate_or_exhaustion() {
        let mut slots = RecurrentStateSlots::new(1).expect("slot pool should initialize");
        let duplicate = slots
            .assign_batch(&[RequestId::from("same"), RequestId::from("same")])
            .expect_err("duplicate requests must fail");
        assert_eq!(
            duplicate,
            RecurrentStateSlotError::DuplicateRequest("same".to_string())
        );

        let exhausted = slots
            .assign_batch(&[RequestId::from("first"), RequestId::from("second")])
            .expect_err("oversized batch must fail without partial assignment");
        assert_eq!(
            exhausted,
            RecurrentStateSlotError::Exhausted {
                capacity: 1,
                assigned: 0,
                requested: 2,
            }
        );
        let assigned = slots
            .assign_batch(&[RequestId::from("second")])
            .expect("failed batch must leave the only slot available");
        assert_eq!(assigned.slots(), &[0]);
    }

    #[test]
    fn recurrent_state_payload_roundtrips_bytes_cleanly() {
        let payload = RecurrentStatePayload {
            next_position: 42,
            layer_states: vec![
                RecurrentLayerStatePayload {
                    conv: vec![1.0, -2.5, 3.125],
                    recurrent: vec![0.5, 0.25, -0.125, 4.0],
                },
                RecurrentLayerStatePayload {
                    conv: vec![0.0, 1.5, -1.0],
                    recurrent: vec![2.0, -3.0, 0.0, 1.0],
                },
            ],
        };
        let bytes = payload.to_bytes();
        let decoded = RecurrentStatePayload::from_bytes(&bytes).expect("valid bytes decode");
        assert_eq!(decoded, payload);
    }

    #[test]
    fn recurrent_state_payload_rejects_corrupted_or_truncated_bytes() {
        assert!(RecurrentStatePayload::from_bytes(&[]).is_err());
        assert!(RecurrentStatePayload::from_bytes(&[0u8; 15]).is_err());
        assert!(RecurrentStatePayload::from_bytes(&[0u8; 16]).is_err()); // wrong magic

        let mut valid = RecurrentStatePayload {
            next_position: 1,
            layer_states: vec![RecurrentLayerStatePayload {
                conv: vec![1.0],
                recurrent: vec![2.0],
            }],
        }
        .to_bytes();

        valid.push(0); // trailing byte
        assert!(RecurrentStatePayload::from_bytes(&valid).is_err());
    }
}
