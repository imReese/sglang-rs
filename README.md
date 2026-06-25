# sglang-rs

Rust runtime work for rewriting SGLang while keeping kernel execution behind a
cross-platform backend boundary.

The goal is to move Python-heavy runtime work into Rust: request lifecycle
management, scheduling, prefix-cache matching, tokenization/detokenization
boundaries, and eventually the bridge into the existing SGLang kernel library.

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
    sglang-router/    # Rust router package aligned with sgl-model-gateway PD shape
    sglang-srt/        # Current runtime crate: router, scheduler, engine, gRPC
    sglang-core/       # Future shared config/types/errors crate if boundaries grow
    sglang-kernel/     # Rust kernel backend boundary with CPU references first
    sglang-python/     # Future PyO3 extension crate for Python integration

  kernels/             # Future native kernel sources and imported upstream pieces
    cuda/
    cpu/
    metal/
    rocm/
    musa/

  python/              # Future pure Python package, CLI glue, and Python tests
    sglang_rs/
    tests/

  docs/                # Future architecture notes and design docs
```

`crates/` is the conventional Rust workspace location for packages. Each
subdirectory is a Rust crate with its own `Cargo.toml`; crates are split only
when the boundary is useful. For now, `crates/sglang-srt` remains the main crate
so runtime work can move quickly without premature crate churn.

Native kernels and Python code should not be folded into the current crate by
default. Kernel backend sources belong under `kernels/`, with
`crates/sglang-kernel` handling the Rust-side backend abstraction, build/link
steps, and FFI. Python code belongs under `python/`, while a PyO3 extension
crate belongs under `crates/sglang-python`. The shared `proto/` directory stays
at the workspace root because it is a cross-language contract.

## Kernel Reuse Direction

The upstream SGLang repository already has a kernel package under
`sgl-kernel/`, published as `sglang-kernel` while keeping the Python import path
as `sgl_kernel`. This repository should align with that name and treat
`sglang-kernel` as the Rust crate boundary for native execution backends rather
than naming the crate after one backend such as CUDA.

The first integration target is not to rewrite every kernel in Rust. Instead,
`crates/sglang-kernel` exposes a small Rust API over kernels needed by the
runtime and PD path, starting with CPU reference implementations and then
binding selected upstream implementations through thin FFI layers. The backend
layout should stay cross-platform:

- CUDA/CUTLASS/FlashInfer kernels for NVIDIA deployments.
- CPU kernels for local correctness tests and non-GPU fallback paths.
- Metal kernels for Apple Silicon development and eventual MLX-backed paths.
- ROCm/HIP and MUSA-oriented sources where upstream support already exists.

For the PD-disaggregation milestone, the highest-value reuse candidates are KV
cache transfer/layout helpers, attention/MLA kernels, GEMM and quantization
kernels, MoE top-k/alignment kernels, elementwise operations such as RMSNorm and
RoPE, and sampling/grammar helpers. The Rust runtime should depend on traits and
typed buffers owned by `sglang-kernel`; backend-specific CUDA, CPU, Metal, ROCm,
or MUSA details should stay behind feature-gated implementations.

Runtime backend selection is an explicit launch-time contract. See
[`docs/backend-capabilities.md`](docs/backend-capabilities.md) for the current
production, CPU reference, and transfer backend boundaries, and
[`docs/backend-implementation-standards.md`](docs/backend-implementation-standards.md)
for the implementation rules that keep B200/CUDA as one backend target rather
than core runtime logic. B200 validation is the first real GPU target and must
use `--device cuda`; local MacBook smoke paths should use `--device cpu`.

## Current Scope

This repository currently contains the first `sglang-srt` runtime crate and the
`sglang-router` package used to exercise the gateway/router boundary:

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
  `--served-model-name`, `--tokenizer-path`, `--device`, and the upstream PD
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
  router status classes for the future gRPC bridge, and exposes `/flush_cache`
  plus `/update_weights_from_disk`, `/update_weight_version`, `/get_weights_by_name`,
  `/remote_instance_transfer_engine_info`, `/poll_transfers` with descriptor
  checksum preservation, `/pause_generation`, `/continue_generation`,
  `/abort_request`, `/start_profile`, and `/stop_profile` forwarding for gateway
  control-plane calls. `/abort_request`
  supports both targeted `rid` aborts and SGLang-compatible `abort_all`.
  The Rust router also exposes gateway-compatible `/v1/loads` and `/get_loads`
  worker-load aggregation across HTTP and gRPC workers.
- `grpc`: gRPC boundary helpers and the initial `GrpcRouterService` adapter for
  the generated Tonic service trait. It wires tokenized `Generate`,
  `HealthCheck`, `FlushCache`, `PauseGeneration`, `ContinueGeneration`,
  `Abort`, `StartProfile`, `StopProfile`, `UpdateWeightsFromDisk`,
  `UpdateWeightVersion`, and `GetWeightsByName` into `RouterRuntime`, converts
  router protocol errors to canonical `tonic::Status` codes, and leaves unsupported RPCs as explicit
  `UNIMPLEMENTED` responses while the runtime surface grows.
- `server`: bootstrap helpers for launching the Rust gRPC router service from
  parsed `ServerArgs`, including model metadata propagation and an injectable PD
  service builder that wraps the bootstrap model runner with
  `KvTransferModelWorker`, a finite decode KV page pool, and bounded transfer
  polling. The HTTP launcher also starts the SGLang-compatible engine-info
  bootstrap service on `--engine-info-bootstrap-port` (default `6789`) and
  shares that state with `/remote_instance_transfer_engine_info`. The bootstrap
  launcher validates the requested runtime backend against the loaded model
  capability before serving, can run the decode-side PD path with the fake
  transfer backend for local/runtime wiring tests, and explicitly rejects
  unsupported real PD backends until Mooncake/model KV memory wiring lands.
- `engine_info_bootstrap`: lightweight HTTP bootstrap service compatible with
  SGLang's transfer-engine info registration flow. It stores per-rank
  `session_id` and `weights_info_dict` payloads via
  `/register_transfer_engine_info` and serves `/get_transfer_engine_info` for
  remote instance weight-transfer discovery. The SRT HTTP service can expose
  the same data through the community `/remote_instance_transfer_engine_info`
  and deprecated `/get_remote_instance_transfer_engine_info` endpoints.
- `tokenizer`: tokenizer trait plus a temporary byte tokenizer for tests.
- `transfer`: PD disaggregation mode/backend normalization, including
  SGLang-compatible `mooncake_tcp` handling, decode bootstrap session tracking,
  KV delta transfer planning from prepared prefill worker batches, a transfer
  executor abstraction that drives bootstrap-room status transitions,
  bootstrap-room-aware Mooncake target resolution, Mooncake KV transfer request
  construction, per-span descriptor checksums carried into submitted Mooncake
  batch records, `/poll_transfers` descriptor checksum reporting, local
  snapshot content checksums for CPU-verifiable KV transfer, batch status
  polling, and a `KvTransferModelWorker` wrapper that registers PD bootstrap
  sessions from prefill request metadata and runs transfer as part of scheduler
  prefill dispatch. It also exposes a
  decode-side KV-ready predicate used by scheduler decode batching to keep PD
  decode requests queued until their bootstrap room reaches `Success`, plus
  engine/router polling hooks and bounded transfer-polling generation entry
  points for advancing asynchronous Mooncake transfer completions from the
  control plane before decode resumes. The gRPC adapter can use the same path
  through a bounded `with_max_transfer_polls` service setting for tokenized and
  text generate RPCs. The
  module also contains the initial Mooncake transfer-engine ABI boundary for
  memory registration and batch transfer. Transfer backend capabilities classify
  Mooncake as the production path, fake as reference-only, and NIXL/Ascend/Mori
  as planned targets until real executors are implemented.
- `scheduler`: waiting queue, prefill/decode batch formation, request stages,
  uncached-token budgeted prefill batching, decode requeueing,
  `max_new_tokens` stopping, prefix-cache application, and KV cache page
  allocation for uncached prefill tokens. Decode batching consults worker
  readiness before dispatch so PD decode requests do not leave the decode queue
  before KV transfer is complete. Successful local/ready prefill dispatches
  publish allocated pages back into RadixCache for future prefix reuse, while
  asynchronous PD prefill pages are staged and only published after transfer
  polling observes KV readiness; failed transfers release unpublished decode
  pages and aborts cancel submitted transfer batches before removing decode
  requests from the running queue. The dispatch path
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
not implemented yet. PD support covers the scheduler/router execution split,
bootstrap metadata propagation, bounded transfer polling, fake/local snapshot
transfer paths, control-plane descriptor checksums, snapshot-content checksums,
and the Mooncake-linked transfer-engine boundary; real device KV memory wiring
remains the next deeper integration layer.

## Development

Run all checks:

```bash
cargo fmt --all --check
cargo test --workspace
```

Run a local process-level PD smoke with a tiny real safetensors model:

```bash
./scripts/run_cpu_pd_smoke.sh
```

The smoke builds `sglang-rs` and `sgl-router` in debug mode, creates a temporary
CPU embedding LM checkpoint, starts prefill/decode workers plus the PD router,
sends an OpenAI chat request, verifies the model-generated `world` token, and
then shuts the stack down. Use `KEEP_RUNNING=1` to leave the services running
after the request. Set `TRANSPORT=grpc` to run the same router smoke against
gRPC SRT workers.
