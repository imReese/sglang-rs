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
- CPU kernels for explicitly selected local reference and correctness tests.
- Metal kernels for Apple Silicon development and eventual MLX-backed paths.
- ROCm/HIP and MUSA-oriented sources where upstream support already exists.

For the PD-disaggregation milestone, the highest-value reuse candidates are KV
cache transfer/layout helpers, attention/MLA kernels, GEMM and quantization
kernels, MoE top-k/alignment kernels, elementwise operations such as RMSNorm and
RoPE, and sampling/grammar helpers. The Rust runtime should depend on traits and
typed buffers owned by `sglang-kernel`; backend-specific CUDA, CPU, Metal, ROCm,
or MUSA details should stay behind feature-gated implementations.

## Current Scope

This repository currently contains the first `sglang-srt` runtime crate and the
`sglang-router` package used to exercise the gateway/router boundary:

- `sglang-kernel`: CPU reference kernels plus a dynamically loaded CUDA Driver
  backend. The CUDA path does not require a CUDA SDK at Rust compile time; on a
  GPU host it performs real device discovery, compute-capability queries,
  primary-context management, memory accounting, and RAII `cuMemAlloc_v2` /
  `cuMemFree_v2` device allocation. Checked host/device copies and dynamically
  loaded cuBLAS provide the first real CUDA weight-execution path without
  requiring a CUDA SDK at Rust compile time. Dynamically compiled BF16 paged
  attention consumes scheduler request metadata and reads GQA K/V rows directly
  from physical slots in the page-major CUDA KV pool.
- `backend`: runtime capability contracts for compute capability, supported
  dtypes, attention implementations, tensor parallelism, KV memory
  registration, Mooncake, RDMA, and NVLink. CUDA dtype support is derived from
  SM capability rather than product names, while unavailable execution or
  transport capabilities remain explicit instead of silently falling back.
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
  transfer backend for local/runtime wiring tests, fails before binding when a
  requested production transfer backend is unavailable, and wires linked
  Mooncake to model-owned transferable KV memory when the runtime exposes it.
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
  module also contains the Mooncake transfer-engine ABI boundary for memory
  registration and batch transfer. Registered KV regions carry an explicit
  device location and are owned by an RAII lease that unregisters them before
  the model allocation is released. Transfer backend capabilities classify
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
carved out. A CUDA/cuBLAS executor now runs the weight-backed embedding LM used
for end-to-end protocol validation. A BF16 GQA paged-attention primitive is
implemented, while MoE CUDA kernels and a production GLM/DeepSeek CUDA executor
are not implemented yet. PD support
covers the scheduler/router execution split, bootstrap metadata propagation,
bounded transfer polling, fake/local snapshot transfer paths, control-plane
descriptor checksums, snapshot-content checksums, and the Mooncake-linked
transfer-engine boundary with managed memory registration. The CUDA KV pool
owns a contiguous allocation laid out as
`page -> layer -> K/V tensor -> token`, matching SGLang's page-major layout.
The pool exposes checked tensor locations for attention kernels, while
Mooncake registers the same physical allocation instead of transfer-owned
staging memory. Dtype-independent CUDA scatter/gather kernels accept strided
K/V rows plus scheduler physical-slot indices and write them directly into that
registered allocation; invalid slot maps fail before launch and are guarded
again on the device. The BF16 paged-attention kernel uses FP32 online-softmax
accumulation and reads that same allocation for variable-length prefill and
decode metadata. It is not yet connected to a complete production GLM/DeepSeek
CUDA executor, so those model/backend pairs still fail at startup instead of
serving through the CPU reference path. The current complete GLM forward
provider remains the explicitly selected CPU reference runtime.

`--device auto` follows the community CLI surface. It selects CUDA when a
working NVIDIA driver and visible device are present, and selects the CPU
reference backend only when CUDA is absent. A broken CUDA installation,
missing cuBLAS, unsupported model/backend pair, or unsupported accelerator
fails at startup instead of falling back to CPU execution.

## KV Page Contract

`--page-size` follows the community SGLang CLI and means the number of token
slots in one physical KV page. `--num-reserved-decode-tokens` is the total token
slot capacity, so it must be nonzero and divisible by `--page-size`; invalid
geometry fails during service construction before a server port is bound.

The scheduler allocates global token slots while the radix cache only reuses
complete physical pages. PD bootstrap metadata and Mooncake transfer requests
carry physical page indices, so a page is transferred exactly once even though
the model executor addresses each token slot independently. The CPU reference
GLM backing store and the CUDA KV pool expose the same physical-page geometry.
Fake and local snapshot transfer remain reference-only backends and are never a
fallback for a requested Mooncake deployment.

## Development

GitHub Actions runs `Rust CI` for every pull request and push to `main` on a
GitHub-hosted Ubuntu runner. It checks formatting, all workspace targets,
Clippy with warnings denied, and the complete default-feature test suite. The
workflow caches Cargo downloads but not `target`, so compiled artifacts remain
on the disposable hosted runner rather than a developer machine or persistent
CI cache. Production Rust source containing lint-suppression attributes such as
`#[allow]` or `#[expect]` also fails this job.

Real CUDA acceptance runs in the separate `CUDA Acceptance` workflow. Register
a Linux self-hosted runner with the `cuda` label, then start the workflow
manually. Set the repository variable `ENABLE_CUDA_CI=true` to run it
automatically after pushes to `main`. The workflow executes the real CUDA
allocation, KV copy, BF16 paged-attention, cuBLAS inference, and HTTP service
tests serially and cleans its per-run Cargo target directory when finished.
The first production runner is expected to be B200, but the workflow has no
product label or product check; A100, H100, and later compatible CUDA devices
run the same capability-gated tests.

To include native Mooncake registration, set `MOONCAKE_BUILD_DIR` to the
Mooncake build directory visible on that runner and select the `mooncake`
manual input. `ENABLE_MOONCAKE_CI=true` enables the same test for automatic
CUDA runs. Missing drivers, NVRTC, cuBLAS, compute capability, Mooncake build
artifacts, or runner labels stop or queue the workflow explicitly; CI never
substitutes the CPU reference backend or a fake transfer backend.

Run all checks:

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
```

On a MacBook, these tests validate allocator geometry, page-aligned prefix
reuse, physical-page transfer planning, decode bootstrap metadata, the CPU
reference backing store, checked CUDA layout arithmetic, batched K/V copy plans,
physical-slot validation, and causal paged-attention metadata construction
without requiring a CUDA driver:

```bash
cargo test -p sglang-srt --test cache_allocator \
  --test scheduler_cache_allocation --test pd_transfer_plan \
  --test glm_runtime --test cuda_kv_cache --test cuda_attention
```

On a CUDA host, validate real page-major allocation, page write/read,
global token-slot addressing, and the transferable-memory view:

```bash
cargo test -p sglang-srt --test cuda_backend \
  cuda_backend_round_trips_page_major_device_kv_memory -- --ignored --nocapture
```

With the CUDA toolkit's NVRTC library available to the dynamic loader, validate
runtime kernel compilation, RMSNorm, SiLU-mul, and device-to-device physical KV
slot writes:

```bash
cargo test -p sglang-srt --test cuda_backend \
  cuda_runtime_kernels_execute_and_write_kv_slots -- --ignored --nocapture
```

Validate a batched, strided K/V scatter across physical page boundaries and
gather it back from the exact allocation exposed to Mooncake:

```bash
cargo test -p sglang-srt --test cuda_backend \
  cuda_kv_kernels_scatter_and_gather_batched_physical_slots -- --ignored --nocapture
```

Validate BF16 GQA paged attention against a CPU numerical reference after K/V
has been scattered across physical page boundaries. This also asserts that
attention reads the exact CUDA allocation exposed for Mooncake registration:

```bash
cargo test -p sglang-srt --test cuda_backend \
  cuda_bf16_paged_attention_reads_mooncake_registered_physical_kv_slots -- --ignored --nocapture
```

Validate real safetensors weight upload, cuBLAS logits, token sampling, and the
HTTP inference service on CUDA:

```bash
cargo test -p sglang-srt --test cuda_inference \
  cuda_auto_selects_cublas_for_weight_backed_http_inference -- --ignored --nocapture
```

With native Mooncake built, validate CUDA KV registration and deregistration:

```bash
MOONCAKE_BUILD_DIR=/path/to/Mooncake/build \
  cargo test -p sglang-srt --features mooncake-link --test cuda_backend \
  cuda_mooncake_registers_real_cuda_kv_memory -- --ignored --nocapture
```

CUDA is the first production backend and B200 is the first hardware acceptance
target; the runtime and kernel interfaces do not encode a B200 product check.
The BF16 attention implementation accepts CUDA compute capability 8.0 or newer,
which leaves A100 and H100 on the same backend path, but those devices are not
yet part of the committed hardware acceptance matrix. Metal, ROCm, and MUSA
execution are not implemented yet. Requesting one of those devices fails at
startup with the unavailable backend/capability instead of running the CPU
reference model.

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
