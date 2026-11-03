# Validation

This guide separates local correctness tests, full-model CPU checks, and
accelerator/network acceptance. Return to the [project overview](../README.md).

## Recorded CPU Demo

The [terminal replay](../README.md#demo) uses the official Qwen3.5-0.8B
checkpoint at revision `2fc06364715b967f1860aea9cf38778875588b17` on macOS
arm64. It covers startup, health checks, a chat response, token-by-token
streaming, and graceful shutdown.

| Recorded request | Input tokens | Output tokens | End-to-end time | First content token |
| --- | ---: | ---: | ---: | ---: |
| Chat: capital of France | 26 | 4 | 12.24 s | Not measured separately |
| Streaming: count from 1 to 5 | 29 | 16 | 30.25 s | 17.77 s |

These are client-observed timings from a functional CPU reference run, not a
performance benchmark or validation of CUDA, Metal, or network P/D transfer.
Idle waits in the GIF are shortened; `MODEL_PATH` and `PORT` replace local
paths and the chosen port.

## Local and Hosted CI

The [Rust CI workflow](../.github/workflows/ci.yml) runs on Ubuntu and macOS.
Run the same checks locally:

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets --locked --features sglang-srt/test-support
cargo clippy --workspace --all-targets --locked --features sglang-srt/test-support -- -D warnings
cargo test --workspace --locked --features sglang-srt/test-support
```

For a self-contained Transformer smoke without a pretrained model download:

```bash
./scripts/run_cpu_reference_smoke.sh
```

This creates a complete one-layer Qwen3 checkpoint, starts a centralized CPU
server, checks its generated token, and stops the server. It is separate from
the full-model demo and is not a network transfer test.

## P/D Correctness and Lifecycle

```bash
cargo test -p sglang-srt --locked --lib nonzero_qwen_pd
cargo test -p sglang-srt --locked --lib pd_shutdown_tests
cargo test -p sglang-srt --locked --features test-support \
  --test core transfer_lifecycle --test server pd_bootstrap_server
```

The numerical oracle runs a complete nonzero-weight Qwen3 Transformer for four
output tokens, compares every logit with centralized execution, and detects
deliberately corrupted KV. Its host payload copy is test-only. Local ZMQ
control tests cover completion and cancellation, but do not validate CUDA
memory registration, RDMA, or NVLink.

## CUDA and P/D Acceptance

The [CUDA Acceptance workflow](../.github/workflows/cuda-acceptance.yml) uses
a Linux self-hosted runner labeled `cuda`, without a GPU product-model label.
It exercises nonzero device placement, allocation, kernels, inference, and
two-rank NCCL execution. Its full matrix requires two visible CUDA devices;
this is an acceptance-test requirement, not a single-rank serving requirement.

Start it manually, or set `ENABLE_CUDA_CI=true` for pushes to `main`. P/D network
tests can be enabled with the `pd_network` manual input or `ENABLE_PD_NETWORK_CI=true`.

For a single-device allocation and kernel check:

```bash
SGLANG_CUDA_TEST_DEVICE=0 cargo test -p sglang-srt --locked \
  --test core cuda_backend:: -- --ignored --nocapture --test-threads=1
```

For P/D network acceptance, using a nonzero CUDA ordinal:

```bash
SGLANG_CUDA_TEST_DEVICE=1 \
  cargo test -p sglang-router --locked --test proxy \
  real_srt_pd::router_pd_chat_completes_with_real_cuda_qwen_workers \
  -- --ignored --nocapture --test-threads=1
```

The linked test compares four generated tokens and finish reasons with
centralized CUDA inference. Hardware tests are ignored by ordinary local
test runs. A passing CPU suite, skipped CUDA job, or successful compilation
does not establish accelerator or network-transfer acceptance.
