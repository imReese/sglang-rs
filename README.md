# sglang-rs

Rust runtime work for rewriting SGLang while keeping CUDA kernels as the
execution backend boundary.

The goal is to move Python-heavy runtime work into Rust: request lifecycle
management, scheduling, prefix-cache matching, tokenization/detokenization
boundaries, and eventually the bridge into the existing CUDA kernels.

## Project Layout

This repository is organized as an independent Rust workspace rather than a
mirror of the upstream SGLang source tree. The Rust crates live under
`crates/`, shared protocol contracts live under `proto/`, and non-Rust runtime
surfaces will get their own top-level directories as they become real
integration targets.

The intended long-term layout is:

```text
sglang-rs/
  Cargo.toml
  Cargo.lock
  rust-toolchain.toml

  proto/
    sglang/runtime/v1/
      sglang.proto

  crates/
    sglang-srt/        # Current runtime crate: router, scheduler, engine, gRPC
    sglang-core/       # Future shared config/types/errors crate if boundaries grow
    sglang-cuda/       # Future Rust FFI wrapper around CUDA/C++ kernels
    sglang-python/     # Future PyO3 extension crate for Python integration

  cuda/                # Future CUDA/C++ kernels and headers
    include/
    src/

  python/              # Future pure Python package, CLI glue, and Python tests
    sglang_rs/
    tests/

  docs/                # Future architecture notes and design docs
```

`crates/` is the conventional Rust workspace location for packages. Each
subdirectory is a Rust crate with its own `Cargo.toml`; crates are split only
when the boundary is useful. For now, `crates/sglang-srt` remains the main crate
so runtime work can move quickly without premature crate churn.

CUDA and Python code should not be folded into the current crate by default.
CUDA kernels belong under `cuda/`, with a Rust wrapper crate such as
`crates/sglang-cuda` handling build/link/FFI. Python code belongs under
`python/`, while a PyO3 extension crate belongs under `crates/sglang-python`.
The shared `proto/` directory stays at the workspace root because it is a
cross-language contract.

## Current Scope

This repository currently contains the first `sglang-srt` runtime crate:

- `proto`: the initial `sglang.runtime.v1.SglangService` contract for the
  native Rust gRPC path in the community roadmap, including typed
  text/tokenized generation, embedding, classification, tokenization,
  health/model/load/control-plane RPCs, OpenAI-compatible JSON pass-through
  RPCs, and admin operations. The crate build script compiles this contract
  into Tonic/prost Rust types exposed through `sglang_srt::proto`; the root
  `buf.yaml` enables future proto lint and breaking-change checks once Buf is
  available in CI.
- `cli`: `sglang serve`-style argument parsing for `--model-path`/`--model`,
  `--host`, `--port`, `--tp-size`, `--dp-size`, `--grpc-mode`,
  `--served-model-name`, `--tokenizer-path`, and the upstream PD
  disaggregation flags, with unknown server args preserved for incremental
  upstream compatibility. The crate builds both `sglang` and `sglang-rs`
  binaries so the upstream command shape works.
- `engine`: text and tokenized request lifecycle glue that drives prefill and
  decode until completion, with a token stream path that preserves incremental
  prefill/decode outputs and prefix-cache hit counts for router streaming.
- `router`: SGLang gateway/router protocol boundary types for tokenized
  `Generate` requests/responses using the current `tokenized.input_ids: u32`
  and `chunk`/`complete` stream shape, plus health and model-info responses used
  during worker registration. `RouterRuntime` adapts these requests into the
  engine's tokenized generation and streaming paths, preserves PD bootstrap
  metadata (`bootstrap_host`, `bootstrap_port`, `bootstrap_room`) for downstream
  worker execution, generates request IDs in Rust when the caller omits one,
  validates token budgets before scheduler dispatch, maps protocol errors to
  router status classes for the future gRPC bridge, and exposes a flush-cache
  control operation for gateway control-plane calls.
- `grpc`: gRPC boundary helpers and the initial `GrpcRouterService` adapter for
  the generated Tonic service trait. It wires tokenized `Generate`,
  `HealthCheck`, and `FlushCache` into `RouterRuntime`, converts router
  protocol errors to canonical `tonic::Status` codes, and leaves unsupported
  RPCs as explicit `UNIMPLEMENTED` responses while the runtime surface grows.
- `tokenizer`: tokenizer trait plus a temporary byte tokenizer for tests.
- `transfer`: PD disaggregation mode/backend normalization, including
  SGLang-compatible `mooncake_tcp` handling, decode bootstrap session tracking,
  KV delta transfer planning from prepared prefill worker batches, a transfer
  executor abstraction that drives bootstrap-room status transitions,
  bootstrap-room-aware Mooncake target resolution, Mooncake KV transfer request
  construction, batch status polling, and a `KvTransferModelWorker` wrapper that
  runs transfer as part of scheduler prefill dispatch. It also exposes a
  decode-side KV-ready predicate used by scheduler decode batching to keep PD
  decode requests queued until their bootstrap room reaches `Success`, plus
  engine/router polling hooks and bounded transfer-polling generation entry
  points for advancing asynchronous Mooncake transfer completions from the
  control plane before decode resumes. The gRPC adapter can use the same path
  through a bounded `with_max_transfer_polls` service setting for tokenized and
  text generate RPCs. The
  module also contains the initial Mooncake transfer-engine ABI boundary for
  memory registration and batch transfer.
- `scheduler`: waiting queue, prefill/decode batch formation, request stages,
  uncached-token budgeted prefill batching, decode requeueing,
  `max_new_tokens` stopping, prefix-cache application, and KV cache page
  allocation for uncached prefill tokens. Decode batching consults worker
  readiness before dispatch so PD decode requests do not leave the decode queue
  before KV transfer is complete. Successful prefill dispatches publish
  allocated pages back into RadixCache for future prefix reuse. The dispatch path
  can run in local PD mode by routing prefill and decode batches to separate
  worker executors, matching the split execution boundary used by SGLang
  disaggregation.
- `model_executor`: prepared model-worker batches with flattened input ids,
  positions, sequence lengths, request offsets, and prefix cache pages for the
  future CUDA/model executor boundary. The batch also exposes per-request
  cached/input token counts so PD prefill workers can treat the uncached prefill
  span as the future KV-transfer delta. PD bootstrap metadata is carried per
  request so the later KV transfer implementation has the same context that
  SGLang attaches to disaggregated requests.
- `cache`: RadixCache-style token-prefix matching plus a finite KV cache page
  allocator for page assignment/reuse and safe full reset when no decode request
  is active.
- `worker`: batch model worker trait that will become the CUDA/model executor
  boundary, plus a `PdModelWorkers` wrapper that sends prefill batches to a
  prefill worker and decode batches to a decode worker.
- `types`: generation request and response types.

The implementation is intentionally small while the architecture is being
carved out. The current worker is test-driven and mockable; CUDA integration is
not implemented yet. PD support currently covers the scheduler/router execution
split; network bootstrap metadata and KV transfer are still future integration
work.

## Development

Run all checks:

```bash
cargo fmt --all --check
cargo test --workspace
```
