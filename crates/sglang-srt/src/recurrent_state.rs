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
}
