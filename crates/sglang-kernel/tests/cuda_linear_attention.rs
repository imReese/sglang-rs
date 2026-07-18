use std::mem::size_of;

use sglang_kernel::cpu::{
    KeyGatedDeltaRuleShape, causal_depthwise_conv1d_step, key_gated_delta_rule_step,
};
use sglang_kernel::cuda::{CudaContext, CudaDeviceAllocation, CudaDriver};
use sglang_kernel::cuda_linear_attention::{
    CudaBf16LinearAttentionKernels, CudaCausalConv1dLaunch, CudaCausalConv1dShape,
    CudaKeyGatedDeltaLaunch, CudaKeyGatedDeltaShape, CudaLinearAttentionError,
};

fn cuda_test_device_ordinal() -> usize {
    std::env::var("SGLANG_CUDA_TEST_DEVICE")
        .unwrap_or_else(|_| "0".to_string())
        .parse()
        .expect("SGLANG_CUDA_TEST_DEVICE must be a CUDA device ordinal")
}

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

fn round_bf16(value: f32) -> f32 {
    f32::from_bits((bf16_bits(value) as u32) << 16)
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| bf16_bits(*value).to_ne_bytes())
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn bytes_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(size_of::<u16>())
        .map(|chunk| {
            let bits = u16::from_ne_bytes(chunk.try_into().expect("BF16 chunk should be exact"));
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

fn bytes_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("f32 chunk should be exact")))
        .collect()
}

fn device_allocation(context: &CudaContext, bytes: &[u8]) -> CudaDeviceAllocation {
    let mut allocation = context
        .allocate(bytes.len())
        .expect("CUDA test allocation should succeed");
    allocation
        .copy_from_host(0, bytes)
        .expect("CUDA test upload should succeed");
    allocation
}

fn zeroed_device_allocation(context: &CudaContext, byte_len: usize) -> CudaDeviceAllocation {
    let mut allocation = context
        .allocate(byte_len)
        .expect("CUDA test allocation should succeed");
    allocation.fill(0).expect("CUDA test fill should succeed");
    allocation
}

fn copy_bf16(allocation: &CudaDeviceAllocation, element_count: usize) -> Vec<f32> {
    let mut bytes = vec![0_u8; element_count * size_of::<u16>()];
    allocation
        .copy_to_host(0, &mut bytes)
        .expect("CUDA BF16 download should succeed");
    bytes_bf16(&bytes)
}

fn copy_f32(allocation: &CudaDeviceAllocation, element_count: usize) -> Vec<f32> {
    let mut bytes = vec![0_u8; element_count * size_of::<f32>()];
    allocation
        .copy_to_host(0, &mut bytes)
        .expect("CUDA f32 download should succeed");
    bytes_f32(&bytes)
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index} differs: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

fn reference_conv_step(
    input: &[f32],
    weight: &[f32],
    state: &mut [f32],
    state_indices: &[u32],
    shape: CudaCausalConv1dShape,
) -> Vec<f32> {
    let state_elements_per_slot = (shape.kernel_size - 1) * shape.channels;
    let mut output = Vec::with_capacity(input.len());
    for (batch, slot) in state_indices.iter().enumerate() {
        let input_start = batch * shape.channels;
        let state_start = *slot as usize * state_elements_per_slot;
        let row = causal_depthwise_conv1d_step(
            &input[input_start..input_start + shape.channels],
            weight,
            &mut state[state_start..state_start + state_elements_per_slot],
            shape.channels,
            shape.kernel_size,
        )
        .expect("CPU reference convolution should accept the CUDA test shape");
        output.extend(row.into_iter().map(round_bf16));
    }
    output
}

struct KdaReferenceInputs<'a> {
    query: &'a [f32],
    key: &'a [f32],
    value: &'a [f32],
    decay: &'a [f32],
    beta: &'a [f32],
}

fn reference_kda_step(
    inputs: KdaReferenceInputs<'_>,
    state: &mut [f32],
    state_indices: &[u32],
    shape: CudaKeyGatedDeltaShape,
) -> Vec<f32> {
    let key_elements_per_row = shape.head_count * shape.key_head_dim;
    let value_elements_per_row = shape.head_count * shape.value_head_dim;
    let state_elements_per_slot = shape.head_count * shape.key_head_dim * shape.value_head_dim;
    let beta_elements_per_row = shape.head_count;
    let mut output = Vec::with_capacity(shape.batch_size * value_elements_per_row);
    for (batch, slot) in state_indices.iter().enumerate() {
        let key_start = batch * key_elements_per_row;
        let value_start = batch * value_elements_per_row;
        let beta_start = batch * beta_elements_per_row;
        let state_start = *slot as usize * state_elements_per_slot;
        let mut cpu_state = vec![0.0_f32; state_elements_per_slot];
        for head in 0..shape.head_count {
            for value_index in 0..shape.value_head_dim {
                for key_index in 0..shape.key_head_dim {
                    let gpu_index = state_start
                        + (head * shape.value_head_dim + value_index) * shape.key_head_dim
                        + key_index;
                    let cpu_index = (head * shape.key_head_dim + key_index) * shape.value_head_dim
                        + value_index;
                    cpu_state[cpu_index] = state[gpu_index];
                }
            }
        }
        let row = key_gated_delta_rule_step(
            &inputs.query[key_start..key_start + key_elements_per_row],
            &inputs.key[key_start..key_start + key_elements_per_row],
            &inputs.value[value_start..value_start + value_elements_per_row],
            &inputs.decay[key_start..key_start + key_elements_per_row],
            &inputs.beta[beta_start..beta_start + beta_elements_per_row],
            &mut cpu_state,
            KeyGatedDeltaRuleShape {
                head_count: shape.head_count,
                key_head_dim: shape.key_head_dim,
                value_head_dim: shape.value_head_dim,
            },
        )
        .expect("CPU KDA reference should accept the CUDA test shape");
        for head in 0..shape.head_count {
            for value_index in 0..shape.value_head_dim {
                for key_index in 0..shape.key_head_dim {
                    let gpu_index = state_start
                        + (head * shape.value_head_dim + value_index) * shape.key_head_dim
                        + key_index;
                    let cpu_index = (head * shape.key_head_dim + key_index) * shape.value_head_dim
                        + value_index;
                    state[gpu_index] = cpu_state[cpu_index];
                }
            }
        }
        output.extend(row.into_iter().map(round_bf16));
    }
    output
}

#[test]
#[ignore = "requires a BF16-capable CUDA device, NVIDIA driver, and NVRTC"]
fn cuda_kda_decode_updates_slot_owned_conv_and_temporal_state() {
    let driver = CudaDriver::load().expect("CUDA driver should load");
    let ordinal = cuda_test_device_ordinal();
    let device = driver
        .device_info(ordinal)
        .expect("CUDA test device should exist");
    let context = driver
        .retain_primary_context(ordinal)
        .expect("CUDA primary context should initialize");
    let mut kernels = CudaBf16LinearAttentionKernels::compile(&context, device.compute_capability)
        .expect("linear-attention kernels should compile for the test device");

    let conv_shape = CudaCausalConv1dShape {
        batch_size: 2,
        state_slot_count: 4,
        channels: 3,
        kernel_size: 3,
    };
    let state_indices = [2_u32, 0];
    let state_indices_device = device_allocation(&context, &u32_bytes(&state_indices));
    let conv_weight = [0.25, -0.5, 1.0, 0.5, 0.25, -0.75, -0.25, 0.75, 0.5].map(round_bf16);
    let conv_weight_device = device_allocation(&context, &bf16_bytes(&conv_weight));
    let conv_state_elements =
        conv_shape.state_slot_count * (conv_shape.kernel_size - 1) * conv_shape.channels;
    let mut conv_state = zeroed_device_allocation(&context, conv_state_elements * size_of::<u16>());
    let mut conv_output = zeroed_device_allocation(
        &context,
        conv_shape.batch_size * conv_shape.channels * size_of::<u16>(),
    );
    let mut conv_reference_state = vec![0.0_f32; conv_state_elements];

    for input in [
        [0.5, -1.0, 1.5, 2.0, 0.25, -0.5],
        [1.0, 0.75, -0.25, -1.5, 1.25, 0.5],
    ] {
        let input = input.map(round_bf16);
        let input_device = device_allocation(&context, &bf16_bytes(&input));
        let expected = reference_conv_step(
            &input,
            &conv_weight,
            &mut conv_reference_state,
            &state_indices,
            conv_shape,
        );
        kernels
            .causal_conv1d_update(CudaCausalConv1dLaunch {
                input: &input_device,
                input_offset: 0,
                weight: &conv_weight_device,
                weight_offset: 0,
                state: &mut conv_state,
                state_offset: 0,
                state_indices: &state_indices_device,
                state_indices_offset: 0,
                output: &mut conv_output,
                output_offset: 0,
                shape: conv_shape,
            })
            .expect("CUDA convolution decode should update indexed state");
        assert_close(&copy_bf16(&conv_output, input.len()), &expected, 0.02);
    }
    assert_close(
        &copy_bf16(&conv_state, conv_state_elements),
        &conv_reference_state,
        0.0,
    );

    let kda_shape = CudaKeyGatedDeltaShape {
        batch_size: 2,
        state_slot_count: 4,
        head_count: 2,
        key_head_dim: 3,
        value_head_dim: 2,
    };
    let temporal_state_elements = kda_shape.state_slot_count
        * kda_shape.head_count
        * kda_shape.key_head_dim
        * kda_shape.value_head_dim;
    let mut temporal_state =
        zeroed_device_allocation(&context, temporal_state_elements * size_of::<f32>());
    let mut kda_output = zeroed_device_allocation(
        &context,
        kda_shape.batch_size * kda_shape.head_count * kda_shape.value_head_dim * size_of::<u16>(),
    );
    let mut temporal_reference_state = vec![0.0_f32; temporal_state_elements];
    let decay = [
        0.9, 0.8, 0.7, 0.6, 0.75, 0.95, 0.85, 0.65, 0.55, 0.92, 0.72, 0.82,
    ];
    let beta = [0.25, 0.75, 0.5, 0.4];
    let decay_device = device_allocation(&context, &f32_bytes(&decay));
    let beta_device = device_allocation(&context, &f32_bytes(&beta));

    for (query, key, value) in [
        (
            [
                0.5, -0.25, 0.75, 0.25, 0.5, -0.5, -0.75, 0.25, 0.5, 0.4, -0.6, 0.2,
            ],
            [
                0.25, 0.5, -0.75, -0.5, 0.25, 0.75, 0.6, -0.2, 0.4, -0.3, 0.7, 0.1,
            ],
            [1.0, -0.5, 0.25, 0.75, -1.0, 0.5, 0.4, -0.2],
        ),
        (
            [
                0.3, 0.2, -0.4, -0.6, 0.1, 0.7, 0.8, -0.5, 0.25, 0.15, 0.45, -0.35,
            ],
            [
                -0.2, 0.6, 0.4, 0.3, -0.7, 0.5, 0.1, 0.8, -0.4, 0.55, -0.25, 0.65,
            ],
            [0.5, 0.75, -0.25, 1.0, 0.6, -0.8, -0.3, 0.9],
        ),
    ] {
        let query = query.map(round_bf16);
        let key = key.map(round_bf16);
        let value = value.map(round_bf16);
        let query_device = device_allocation(&context, &bf16_bytes(&query));
        let key_device = device_allocation(&context, &bf16_bytes(&key));
        let value_device = device_allocation(&context, &bf16_bytes(&value));
        let expected = reference_kda_step(
            KdaReferenceInputs {
                query: &query,
                key: &key,
                value: &value,
                decay: &decay,
                beta: &beta,
            },
            &mut temporal_reference_state,
            &state_indices,
            kda_shape,
        );
        kernels
            .key_gated_delta_decode(CudaKeyGatedDeltaLaunch {
                query: &query_device,
                query_offset: 0,
                key: &key_device,
                key_offset: 0,
                value: &value_device,
                value_offset: 0,
                decay: &decay_device,
                decay_offset: 0,
                beta: &beta_device,
                beta_offset: 0,
                state: &mut temporal_state,
                state_offset: 0,
                state_indices: &state_indices_device,
                state_indices_offset: 0,
                output: &mut kda_output,
                output_offset: 0,
                shape: kda_shape,
            })
            .expect("CUDA KDA decode should update indexed temporal state");
        assert_close(
            &copy_bf16(
                &kda_output,
                kda_shape.batch_size * kda_shape.head_count * kda_shape.value_head_dim,
            ),
            &expected,
            0.02,
        );
    }
    assert_close(
        &copy_f32(&temporal_state, temporal_state_elements),
        &temporal_reference_state,
        1e-5,
    );

    let duplicate_indices = device_allocation(&context, &u32_bytes(&[1, 1]));
    let query = device_allocation(&context, &bf16_bytes(&[0.0; 12]));
    let key = device_allocation(&context, &bf16_bytes(&[0.0; 12]));
    let value = device_allocation(&context, &bf16_bytes(&[0.0; 8]));
    let error = kernels
        .key_gated_delta_decode(CudaKeyGatedDeltaLaunch {
            query: &query,
            query_offset: 0,
            key: &key,
            key_offset: 0,
            value: &value,
            value_offset: 0,
            decay: &decay_device,
            decay_offset: 0,
            beta: &beta_device,
            beta_offset: 0,
            state: &mut temporal_state,
            state_offset: 0,
            state_indices: &duplicate_indices,
            state_indices_offset: 0,
            output: &mut kda_output,
            output_offset: 0,
            shape: kda_shape,
        })
        .expect_err("duplicate writable slots must fail before recurrence launch");
    assert!(matches!(
        error,
        CudaLinearAttentionError::DuplicateStateIndex
    ));

    let out_of_range_indices = device_allocation(&context, &u32_bytes(&[4, 1]));
    let error = kernels
        .key_gated_delta_decode(CudaKeyGatedDeltaLaunch {
            query: &query,
            query_offset: 0,
            key: &key,
            key_offset: 0,
            value: &value,
            value_offset: 0,
            decay: &decay_device,
            decay_offset: 0,
            beta: &beta_device,
            beta_offset: 0,
            state: &mut temporal_state,
            state_offset: 0,
            state_indices: &out_of_range_indices,
            state_indices_offset: 0,
            output: &mut kda_output,
            output_offset: 0,
            shape: kda_shape,
        })
        .expect_err("out-of-range writable slots must fail before recurrence launch");
    assert!(matches!(
        error,
        CudaLinearAttentionError::StateIndexOutOfRange
    ));
}
