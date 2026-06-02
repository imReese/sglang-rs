# sglang-rs

Rust runtime work for rewriting SGLang while keeping CUDA kernels as the
execution backend boundary.

The goal is to move Python-heavy runtime work into Rust: request lifecycle
management, scheduling, prefix-cache matching, tokenization/detokenization
boundaries, and eventually the bridge into the existing CUDA kernels.

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
  prefill/decode outputs for router streaming.
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
- `grpc`: gRPC boundary helpers, including router protocol-error conversion to
  `tonic::Status` so generated service implementations can return canonical
  gRPC status codes without duplicating error mapping logic.
- `tokenizer`: tokenizer trait plus a temporary byte tokenizer for tests.
- `transfer`: PD disaggregation mode/backend normalization, including
  SGLang-compatible `mooncake_tcp` handling, plus the initial Mooncake
  transfer-engine ABI boundary for memory registration and batch transfer.
- `scheduler`: waiting queue, prefill/decode batch formation, request stages,
  uncached-token budgeted prefill batching, decode requeueing,
  `max_new_tokens` stopping, prefix-cache application, and KV cache page
  allocation for uncached prefill tokens. Successful prefill dispatches publish
  allocated pages back into RadixCache for future prefix reuse. The dispatch
  path can run in local PD mode by routing prefill and decode batches to
  separate worker executors, matching the split execution boundary used by
  SGLang disaggregation.
- `model_executor`: prepared model-worker batches with flattened input ids,
  positions, sequence lengths, request offsets, and prefix cache pages for the
  future CUDA/model executor boundary. PD bootstrap metadata is carried per
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
