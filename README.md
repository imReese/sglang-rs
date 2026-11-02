# sglang-rs

[![Rust CI](https://github.com/imReese/sglang-rs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/imReese/sglang-rs/actions/workflows/ci.yml)

A Rust-native LLM serving runtime with shared model execution, paged KV cache
management, and Mooncake-linked prefill/decode disaggregation.

This is an independent implementation aligned with the serving and routing
contracts of [SGLang](https://github.com/sgl-project/sglang) and
`sgl-model-gateway`. The project is under active development and is not yet
production-ready or a drop-in replacement for community SGLang.

## Demo

The official [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
checkpoint running on macOS arm64 through the Rust CPU reference backend.
Its 24-layer hybrid decoder combines 18 Gated DeltaNet linear-attention
layers with 6 full-attention layers.

![Qwen3.5-0.8B hybrid attention and streaming through the Rust CPU reference server](docs/assets/qwen-cpu-demo.gif)

The replay shows server startup, health checks, an OpenAI-compatible chat
response, token-by-token streaming, and graceful shutdown. Idle waits are
shortened; `MODEL_PATH` and `PORT` replace local paths and the chosen port.

| Recorded request | Input tokens | Output tokens | End-to-end time | First content token |
| --- | ---: | ---: | ---: | ---: |
| Chat: capital of France | 26 | 4 | 12.24 s | Not measured separately |
| Streaming: count from 1 to 5 | 29 | 16 | 30.25 s | 17.77 s |

These are client-observed timings from a functional CPU reference run, not a
performance benchmark or validation of CUDA, Metal, or network P/D transfer.

## Quick Start

### Build

Install Rust through `rustup`; the checkout pins its toolchain in
[`rust-toolchain.toml`](rust-toolchain.toml). The default build does not require
a CUDA SDK or native Mooncake libraries.

```bash
git clone https://github.com/imReese/sglang-rs.git
cd sglang-rs
cargo build --release --locked --bin sglang-rs
```

The crate also provides the `sglang` binary with the same `serve` command shape.

### Run Qwen3.5-0.8B on CPU

Download the official safetensors checkpoint with the Hugging Face `hf` CLI,
or point `MODEL_PATH` at an existing local checkpoint. The revision below is
the one used for the recorded demo; its weight file is approximately 1.75 GB.

```bash
export MODEL_PATH="$HOME/.cache/sglang-rs/Qwen3.5-0.8B"
hf download Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --local-dir "$MODEL_PATH"

./target/release/sglang-rs serve \
  --model-path "$MODEL_PATH" \
  --served-model-name Qwen3.5-0.8B \
  --device cpu \
  --host 127.0.0.1 --port 30000 \
  --engine-info-bootstrap-port 0 \
  --num-reserved-decode-tokens 256 \
  --max-running-requests 1
```

The CPU backend executes in FP32, including conversion from BF16 checkpoint
weights. It is a reference implementation for correctness and local testing;
keep prompts short and do not use its latency to estimate accelerator
performance. This example reserves 256 KV token slots, not the checkpoint's
full advertised context capacity.

### Send a Request

```bash
curl --fail http://127.0.0.1:30000/health

curl --fail-with-body --no-buffer \
  http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.5-0.8B",
    "messages": [{"role": "user", "content": "Count from 1 to 5, separated by commas. Output only the numbers."}],
    "temperature": 0,
    "max_tokens": 16,
    "stream": true,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Use `"stream": false` for a single JSON response. The HTTP service also
provides `/v1/completions`, `/generate`, and `/get_model_info`. Native gRPC
serving is available with `--grpc-mode`, including an HTTP sidecar for worker
discovery and control-plane integration. Protocol coverage is incremental;
unsupported operations return explicit errors.

For a small, self-contained Transformer smoke test without downloading a
pretrained checkpoint:

```bash
./scripts/run_cpu_reference_smoke.sh
```

This script creates a complete one-layer Qwen3 checkpoint, starts a centralized
CPU server, checks its generated token, and stops the server. It is separate
from the full-model demo above and is not a network transfer test.

## Model and Backend Coverage

`ModelRegistry` selects adapters from Hugging Face `architectures`. Adapters
parse configuration, map weights, compose shared Transformer components, and
declare capability requirements. Recognition of an architecture alone does
not establish that an executable backend exists.

| Hugging Face architecture | Shared components | CPU reference | CUDA implementation |
| --- | --- | --- | --- |
| `Qwen2ForCausalLM`, `Qwen3ForCausalLM` | Dense decoder, MHA/GQA | FP32 | BF16; dense tensor-parallel path |
| `Qwen3_5ForConditionalGeneration` | Gated DeltaNet + full attention | FP32 | BF16; single rank |
| `DeepseekV3ForCausalLM`, `KimiK25ForConditionalGeneration` | MLA + MoE | FP32 | BF16; single rank |
| `KimiLinearForCausalLM` | KDA + MLA/MoE | FP32 | BF16; single rank |
| `GlmMoeDsaForCausalLM`, `DeepseekV4ForCausalLM` | Shared MLA/MoE definitions | No executable path | No executable path |

The table describes implemented execution paths, not full-scale checkpoint
acceptance. The full Qwen3.5-0.8B checkpoint has been exercised on CPU; other
paths have focused Transformer fixture tests and, where applicable, opt-in
CUDA hardware tests. Checkpoint variants and quantization formats are validated
per adapter. Vision/audio inputs are not supported by these text paths.

GLM DSA remains a thin adapter over shared DeepSeek/MLA-MoE definitions. Until
its required execution components exist, startup fails with the missing
capabilities rather than substituting another model.

| Device selection | Behavior |
| --- | --- |
| `--device cpu` | Explicit FP32 reference backend; single-rank execution |
| `--device cuda` | CUDA execution with capability checks and BF16 kernels |
| `--device auto` | Probes registered production backends, currently CUDA; never falls back to CPU |
| `--device metal`, `rocm`, or `musa` | No registered execution provider; startup fails explicitly |

CUDA execution requires the NVIDIA driver, NVRTC, and cuBLAS; dense tensor
parallelism additionally requires NCCL. Device placement uses ordinals and
rank configuration, not GPU product names. Dense tensor parallelism is
currently single-node; unavailable topology or component requirements fail
at startup. The current BF16 execution path
requires compute capability 8.0 or newer. B200 is the first production
acceptance target, not a hard-coded runtime or CI requirement.

## Runtime Architecture

```text
HTTP / gRPC serving
        |
Engine / Scheduler / WorkerGroup
        |
ModelRunner + backend-owned execution resources
        |
Shared dense / hybrid / MLA / MoE components
        |
Backend kernels and device memory

ModelRegistry -> model adapter -> ModelDefinition
Active KV storage -> NexusKV descriptor -> transfer backend
```

- **Serving and scheduling:** request lifecycle, batch formation, admission,
  token budgets, completion, and cancellation are independent of model family.
- **Model execution:** adapters describe models; shared executors run their
  components. Models do not own separate schedulers or transfer lifecycles.
- **Backend resources:** providers create compatible executors and execution
  resources. `ModelRunner` owns the active KV pool; backends allocate its
  physical memory.
- **KV and transfer:** scheduling manages logical pages and token slots.
  Runtime KV storage directly implements NexusKV `KvCacheMemoryProvider`;
  transfer consumes its `TransferableKvCacheMemory` descriptor.
- **Kernels:** `sglang-kernel` contains CPU reference primitives and CUDA
  Driver, NVRTC, cuBLAS, and NCCL bindings. Hardware differences are expressed
  through capabilities rather than model-specific execution infrastructure.

`--page-size` is the number of token slots per physical KV page.
`--num-reserved-decode-tokens` is the total slot capacity and must be nonzero
and divisible by the page size. Invalid geometry fails before serving begins.
Prefix reuse operates on complete physical pages and is disabled for recurrent
models until prefix-state restoration is available.

## Prefill/Decode Disaggregation

The production transfer implementation uses native Mooncake FFI and registers
the active runtime KV allocation. Build it against real Mooncake artifacts:

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
`--disaggregation-transfer-backend mooncake|mooncake_tcp` flags. Bootstrap
metadata carries the room and destination KV layout between workers.

Current limits:

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

On SIGINT/SIGTERM, the service stops accepting requests, drains or cancels
transfers, frees native batch IDs, unregisters memory, and then releases model
resources. Bootstrap ABORT handling remains available during drain. Stop
decode workers before prefill workers when draining a deployment. Unconfirmed
writes retain their KV pages; destruction that cannot establish safe drain or
unregistration is fatal rather than permitting unsafe memory reuse.

**NexusKV integration is currently the memory contract only.** NexusKV
store/offload integration and a NexusKV network transfer executor are not
implemented. A host offload tier does not replace the active accelerator KV
pool.

## Validation

### Local and Hosted CI

The [Rust CI workflow](.github/workflows/ci.yml) runs on Ubuntu and macOS.
Run the same checks locally:

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets --locked --features sglang-srt/test-support
cargo clippy --workspace --all-targets --locked --features sglang-srt/test-support -- -D warnings
cargo test --workspace --locked --features sglang-srt/test-support
```

Focused P/D correctness and lifecycle checks:

```bash
cargo test -p sglang-srt --locked --lib nonzero_qwen_pd
cargo test -p sglang-srt --locked --lib pd_shutdown_tests
cargo test -p sglang-srt --locked --features test-support \
  --test mooncake_lifecycle --test pd_bootstrap_server
```

The numerical oracle runs a complete nonzero-weight Qwen3 Transformer for four
output tokens, compares every logit with centralized execution, and detects
deliberately corrupted KV. Its host payload copy is test-only. Local ZMQ
control tests cover completion and cancellation, but do not validate CUDA
memory registration, RDMA, or NVLink.

### CUDA and Native Mooncake

The [CUDA Acceptance workflow](.github/workflows/cuda-acceptance.yml) uses a
Linux self-hosted runner labeled `cuda`, without a GPU product-model label.
It exercises nonzero device placement, allocation, kernels, inference, and
two-rank NCCL execution. Its full matrix requires two visible CUDA devices;
this is an acceptance-test requirement, not a single-rank serving requirement.

Start it manually, or set `ENABLE_CUDA_CI=true` for pushes to `main`. Native
registration and linked P/D tests additionally require `MOONCAKE_BUILD_DIR`
and the `mooncake` manual input or `ENABLE_MOONCAKE_CI=true`.

For a single-device allocation and kernel check:

```bash
SGLANG_CUDA_TEST_DEVICE=0 cargo test -p sglang-srt --locked \
  --test cuda_backend -- --ignored --nocapture --test-threads=1
```

For linked P/D acceptance, using a nonzero CUDA ordinal:

```bash
MOONCAKE_BUILD_DIR=/path/to/Mooncake/build SGLANG_CUDA_TEST_DEVICE=1 \
  cargo test -p sglang-router --locked --features mooncake-link --test proxy \
  real_srt_pd::router_pd_chat_completes_with_real_cuda_qwen_mooncake_workers \
  -- --ignored --nocapture --test-threads=1
```

The linked test compares four generated tokens and finish reasons with
centralized CUDA inference. Hardware tests are ignored by ordinary local
test runs. A passing CPU suite, skipped CUDA job, or successful compilation
does not establish accelerator or network-transfer acceptance.

## Repository Layout

| Path | Responsibility |
| --- | --- |
| [`crates/sglang-srt`](crates/sglang-srt) | Serving, scheduler, model registry, execution, KV cache, and transfer orchestration |
| [`crates/sglang-kernel`](crates/sglang-kernel) | CPU primitives and CUDA/kernel library bindings |
| [`crates/sglang-router`](crates/sglang-router) | HTTP/gRPC routing and P/D integration tests; see its [README](crates/sglang-router/README.md) |
| [`proto`](proto) | Runtime gRPC service definitions |
| [`scripts`](scripts) | Local smoke and launch scripts |
| [`.github/workflows`](.github/workflows) | Hosted Rust checks and opt-in CUDA acceptance |

## License

[MIT](LICENSE). External models, native libraries, and other dependencies
retain their own licenses.
