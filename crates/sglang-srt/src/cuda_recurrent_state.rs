use std::collections::HashSet;
use std::fmt;
use std::mem::size_of;

use sglang_kernel::cuda::{CudaContext, CudaDeviceAllocation, CudaError};

use crate::models::RecurrentStateLayout;
use crate::recurrent_state::{RecurrentStateSlotError, RecurrentStateSlots};
use crate::types::RequestId;

const BF16_BYTES: usize = 2;
const F32_BYTES: usize = 4;

/// CUDA-owned recurrent state in community-compatible layer/slot layouts:
/// convolution `[layer, slot, history, channels]` and temporal
/// `[layer, slot, value_head, value_dim, key_dim]`.
pub struct CudaRecurrentStateStorage {
    layout: RecurrentStateLayout,
    slot_capacity: usize,
    conv_slot_bytes: usize,
    temporal_slot_bytes: usize,
    conv_state: CudaDeviceAllocation,
    temporal_state: CudaDeviceAllocation,
    state_indices: CudaDeviceAllocation,
    host_state_indices: Vec<u32>,
    slots: RecurrentStateSlots,
}

impl fmt::Debug for CudaRecurrentStateStorage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaRecurrentStateStorage")
            .field("layout", &self.layout)
            .field("slot_capacity", &self.slot_capacity)
            .field("device_ordinal", &self.conv_state.device_ordinal())
            .field("conv_state_bytes", &self.conv_state.byte_len())
            .field("temporal_state_bytes", &self.temporal_state.byte_len())
            .finish_non_exhaustive()
    }
}

impl CudaRecurrentStateStorage {
    pub fn allocate(
        context: &CudaContext,
        layout: RecurrentStateLayout,
        slot_capacity: usize,
    ) -> Result<Self, CudaRecurrentStateError> {
        validate_layout(layout)?;
        let slots = RecurrentStateSlots::new(slot_capacity)?;
        let conv_slot_bytes = checked_bytes(
            layout
                .conv_elements_per_layer()
                .ok_or(CudaRecurrentStateError::ShapeOverflow)?,
            BF16_BYTES,
        )?;
        let temporal_slot_bytes = checked_bytes(
            layout
                .temporal_elements_per_layer()
                .ok_or(CudaRecurrentStateError::ShapeOverflow)?,
            F32_BYTES,
        )?;
        let layer_slots = checked_product(layout.layer_count, slot_capacity)?;
        let mut conv_state = context.allocate(checked_product(layer_slots, conv_slot_bytes)?)?;
        conv_state.fill(0)?;
        let mut temporal_state =
            context.allocate(checked_product(layer_slots, temporal_slot_bytes)?)?;
        temporal_state.fill(0)?;
        let mut state_indices =
            context.allocate(checked_bytes(slot_capacity, size_of::<u32>())?)?;
        state_indices.fill(0)?;
        Ok(Self {
            layout,
            slot_capacity,
            conv_slot_bytes,
            temporal_slot_bytes,
            conv_state,
            temporal_state,
            state_indices,
            host_state_indices: Vec::with_capacity(slot_capacity),
            slots,
        })
    }

    pub fn layout(&self) -> RecurrentStateLayout {
        self.layout
    }

    pub fn prepare_batch(
        &mut self,
        request_ids: &[RequestId],
    ) -> Result<(), CudaRecurrentStateError> {
        let assignment = self.slots.assign_batch(request_ids)?;
        let newly_assigned = assignment
            .newly_assigned()
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let prepared = (|| {
            for slot in assignment.newly_assigned() {
                self.clear_slot(*slot as usize)?;
            }
            self.host_state_indices.clear();
            self.host_state_indices
                .extend_from_slice(assignment.slots());
            let bytes = u32_slice_bytes(&self.host_state_indices);
            self.state_indices.copy_from_host(0, &bytes)?;
            Ok(())
        })();
        if prepared.is_err() {
            for (request_id, slot) in request_ids.iter().zip(assignment.slots()) {
                if newly_assigned.contains(slot) {
                    self.slots.release(request_id);
                }
            }
        }
        prepared
    }

    pub fn release_request(&mut self, request_id: &RequestId) {
        self.slots.release(request_id);
    }

    pub fn layer_state_mut(
        &mut self,
        layer_index: usize,
    ) -> Result<CudaRecurrentLayerState<'_>, CudaRecurrentStateError> {
        if layer_index >= self.layout.layer_count {
            return Err(CudaRecurrentStateError::LayerOutOfRange {
                layer_index,
                layer_count: self.layout.layer_count,
            });
        }
        let layer_slot_offset = checked_product(layer_index, self.slot_capacity)?;
        Ok(CudaRecurrentLayerState {
            conv_state: &mut self.conv_state,
            conv_state_offset: checked_product(layer_slot_offset, self.conv_slot_bytes)?,
            temporal_state: &mut self.temporal_state,
            temporal_state_offset: checked_product(layer_slot_offset, self.temporal_slot_bytes)?,
            state_indices: &self.state_indices,
            batch_size: self.host_state_indices.len(),
            state_slot_count: self.slot_capacity,
        })
    }

    fn clear_slot(&mut self, slot: usize) -> Result<(), CudaRecurrentStateError> {
        for layer_index in 0..self.layout.layer_count {
            let layer_slot = checked_product(layer_index, self.slot_capacity)?
                .checked_add(slot)
                .ok_or(CudaRecurrentStateError::ShapeOverflow)?;
            self.conv_state.fill_range(
                checked_product(layer_slot, self.conv_slot_bytes)?,
                self.conv_slot_bytes,
                0,
            )?;
            self.temporal_state.fill_range(
                checked_product(layer_slot, self.temporal_slot_bytes)?,
                self.temporal_slot_bytes,
                0,
            )?;
        }
        Ok(())
    }
}

pub struct CudaRecurrentLayerState<'a> {
    pub conv_state: &'a mut CudaDeviceAllocation,
    pub conv_state_offset: usize,
    pub temporal_state: &'a mut CudaDeviceAllocation,
    pub temporal_state_offset: usize,
    pub state_indices: &'a CudaDeviceAllocation,
    pub batch_size: usize,
    pub state_slot_count: usize,
}

#[derive(Debug)]
pub enum CudaRecurrentStateError {
    Cuda(CudaError),
    Slot(RecurrentStateSlotError),
    InvalidLayout(String),
    ShapeOverflow,
    LayerOutOfRange {
        layer_index: usize,
        layer_count: usize,
    },
}

impl fmt::Display for CudaRecurrentStateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(error) => {
                write!(formatter, "CUDA recurrent state operation failed: {error}")
            }
            Self::Slot(error) => write!(formatter, "CUDA recurrent state slot error: {error}"),
            Self::InvalidLayout(message) => {
                write!(formatter, "invalid CUDA recurrent state layout: {message}")
            }
            Self::ShapeOverflow => formatter.write_str("CUDA recurrent state shape overflowed"),
            Self::LayerOutOfRange {
                layer_index,
                layer_count,
            } => write!(
                formatter,
                "CUDA recurrent state layer {layer_index} is outside layer count {layer_count}"
            ),
        }
    }
}

impl std::error::Error for CudaRecurrentStateError {}

impl From<CudaError> for CudaRecurrentStateError {
    fn from(value: CudaError) -> Self {
        Self::Cuda(value)
    }
}

impl From<RecurrentStateSlotError> for CudaRecurrentStateError {
    fn from(value: RecurrentStateSlotError) -> Self {
        Self::Slot(value)
    }
}

fn validate_layout(layout: RecurrentStateLayout) -> Result<(), CudaRecurrentStateError> {
    if layout.layer_count == 0 {
        return Err(CudaRecurrentStateError::InvalidLayout(
            "layer count must be non-zero".to_string(),
        ));
    }
    if layout.conv_kernel_dim < 2 {
        return Err(CudaRecurrentStateError::InvalidLayout(format!(
            "convolution kernel dimension must be at least 2, got {}",
            layout.conv_kernel_dim
        )));
    }
    if layout.key_head_count == 0
        || layout.value_head_count == 0
        || layout.key_head_dim == 0
        || layout.value_head_dim == 0
    {
        return Err(CudaRecurrentStateError::InvalidLayout(
            "head counts and dimensions must be non-zero".to_string(),
        ));
    }
    layout
        .elements_per_request()
        .ok_or(CudaRecurrentStateError::ShapeOverflow)?;
    Ok(())
}

fn checked_product(left: usize, right: usize) -> Result<usize, CudaRecurrentStateError> {
    left.checked_mul(right)
        .ok_or(CudaRecurrentStateError::ShapeOverflow)
}

fn checked_bytes(
    element_count: usize,
    element_size: usize,
) -> Result<usize, CudaRecurrentStateError> {
    checked_product(element_count, element_size)
}

fn u32_slice_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> RecurrentStateLayout {
        RecurrentStateLayout {
            layer_count: 2,
            conv_kernel_dim: 3,
            key_head_count: 1,
            value_head_count: 1,
            key_head_dim: 2,
            value_head_dim: 2,
        }
    }

    #[test]
    fn invalid_layout_fails_before_cuda_allocation() {
        let invalid = RecurrentStateLayout {
            layer_count: 0,
            ..layout()
        };
        assert!(matches!(
            validate_layout(invalid),
            Err(CudaRecurrentStateError::InvalidLayout(_))
        ));
    }

    #[test]
    #[ignore = "requires a CUDA device and NVIDIA driver"]
    fn cuda_recurrent_state_reuses_and_clears_request_slots() {
        let driver = sglang_kernel::cuda::CudaDriver::load().expect("CUDA driver should load");
        let ordinal = std::env::var("SGLANG_CUDA_TEST_DEVICE")
            .unwrap_or_else(|_| "0".to_string())
            .parse()
            .expect("SGLANG_CUDA_TEST_DEVICE must be a CUDA device ordinal");
        let context = driver
            .retain_primary_context(ordinal)
            .expect("CUDA primary context should initialize");
        let mut storage = CudaRecurrentStateStorage::allocate(&context, layout(), 2)
            .expect("CUDA recurrent state should allocate");
        storage
            .prepare_batch(&[RequestId::from("request-a"), RequestId::from("request-b")])
            .expect("initial requests should receive slots");
        assert_eq!(storage.host_state_indices, [0, 1]);

        let conv_slot_bytes = storage.conv_slot_bytes;
        let temporal_slot_bytes = storage.temporal_slot_bytes;
        for layer_index in 0..layout().layer_count {
            let layer = storage
                .layer_state_mut(layer_index)
                .expect("layer state should exist");
            layer
                .conv_state
                .fill_range(layer.conv_state_offset, conv_slot_bytes * 2, 0x3f)
                .expect("conv slots should be writable");
            layer
                .temporal_state
                .fill_range(layer.temporal_state_offset, temporal_slot_bytes * 2, 0x3f)
                .expect("temporal slots should be writable");
        }

        storage.release_request(&RequestId::from("request-a"));
        storage
            .prepare_batch(&[RequestId::from("request-c")])
            .expect("replacement request should reuse the released slot");
        assert_eq!(storage.host_state_indices, [0]);

        for layer_index in 0..layout().layer_count {
            let conv_offset = (layer_index * storage.slot_capacity) * storage.conv_slot_bytes;
            let temporal_offset =
                (layer_index * storage.slot_capacity) * storage.temporal_slot_bytes;
            let mut conv = vec![1_u8; storage.conv_slot_bytes];
            let mut adjacent_conv = vec![0_u8; storage.conv_slot_bytes];
            let mut temporal = vec![1_u8; storage.temporal_slot_bytes];
            let mut adjacent_temporal = vec![0_u8; storage.temporal_slot_bytes];
            storage
                .conv_state
                .copy_to_host(conv_offset, &mut conv)
                .expect("conv slot should download");
            storage
                .conv_state
                .copy_to_host(conv_offset + storage.conv_slot_bytes, &mut adjacent_conv)
                .expect("adjacent conv slot should download");
            storage
                .temporal_state
                .copy_to_host(temporal_offset, &mut temporal)
                .expect("temporal slot should download");
            storage
                .temporal_state
                .copy_to_host(
                    temporal_offset + storage.temporal_slot_bytes,
                    &mut adjacent_temporal,
                )
                .expect("adjacent temporal slot should download");
            assert!(conv.iter().all(|byte| *byte == 0));
            assert!(temporal.iter().all(|byte| *byte == 0));
            assert!(adjacent_conv.iter().all(|byte| *byte == 0x3f));
            assert!(adjacent_temporal.iter().all(|byte| *byte == 0x3f));
        }
    }
}
