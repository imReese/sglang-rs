# Prefill/Decode Disaggregation

The production transfer implementation uses native Mooncake FFI and registers
the active runtime KV allocation. Network P/D requires CUDA/Mooncake hardware
acceptance; local CPU tests are not a substitute. Return to the
[project overview](../README.md).

## Build

Build against real Mooncake artifacts:

```bash
MOONCAKE_BUILD_DIR=/path/to/Mooncake/build \
  cargo build --release --locked -p sglang-srt \
  --bin sglang-rs --features mooncake-link
```

`MOONCAKE_HOME` is an alternative when the artifacts are under its `build/`
directory. The native libraries and their dependencies must be installed;
enabling `mooncake-link`, including through `--all-features`, does not work
without them.

Workers use the community-style `--disaggregation-mode prefill|decode` and
`--disaggregation-transfer-backend tcp|mooncake|mooncake_tcp` flags. Bootstrap
metadata carries the room and destination KV layout between workers.

## Native TCP Transfer Backend (`tcp`)

The native TCP transfer backend (`--disaggregation-transfer-backend tcp`) provides
a pure-Rust, cross-platform data plane that does not require Mooncake C++ libraries,
RDMA devices, or CUDA. It runs seamlessly on macOS (CPU/Metal) as well as Linux,
streaming KV cache chunks directly over standard TCP sockets into registered memory
buffers.

When using `tcp`, ephemeral ports can be bound automatically (e.g. `127.0.0.1:0`)
or specified explicitly via `--disaggregation-mooncake-rpc-port <PORT>`. The bound
address is published automatically via the P/D ZeroMQ bootstrap protocol to peer
workers.

## Current Limits

- P/D requires `tp_size=1`, `dp_size=1`, `nnodes=1`, and `node_rank=0`.
  Centralized CUDA dense tensor parallelism is a separate execution path.
- Both workers disable radix reuse and transfer the full prompt. Decode-side
  prefix reuse and KV offload requests are rejected at startup.
- Hybrid models requiring recurrent state, including Qwen3.5 and Kimi Linear,
  are rejected because KV-only transfer is insufficient.
- The Mooncake adapter requires one contiguous registered region. This is a
  Mooncake implementation constraint, not a NexusKV descriptor limitation.
- NIXL, Ascend, and Mori transfer executors are not implemented. Fake/local
  snapshot helpers are test-only and cannot be selected by production CLI.

Prefill sends the sampled first token and terminal status after all native KV
tasks have drained. Decode waits for both before using the received KV; it
does not repeat prefill locally.

## Shutdown and Cancellation

On SIGINT/SIGTERM, the service stops accepting requests, drains or cancels
transfers, frees native batch IDs, unregisters memory, and then releases model
resources. Bootstrap ABORT handling remains available during drain. Stop
decode workers before prefill workers when draining a deployment. Unconfirmed
writes retain their KV pages; destruction that cannot establish safe drain or
unregistration is fatal rather than permitting unsafe memory reuse.

## NexusKV Integration

NexusKV integration is currently the memory contract only. NexusKV
store/offload integration and a NexusKV network transfer executor are not
implemented. A host offload tier does not replace the active accelerator KV
pool.

For numerical checks, lifecycle tests, and linked network acceptance commands,
see [validation](validation.md#pd-correctness-and-lifecycle).
