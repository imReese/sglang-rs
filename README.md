# sglang-rs

Rust runtime work for rewriting SGLang while keeping CUDA kernels as the
execution backend boundary.

The goal is to move Python-heavy runtime work into Rust: request lifecycle
management, scheduling, prefix-cache matching, tokenization/detokenization
boundaries, and eventually the bridge into the existing CUDA kernels.

## Current Scope

This repository currently contains the first `sglang-srt` runtime crate:

- `cli`: `sglang serve`-style argument parsing for `--model-path`/`--model`,
  `--host`, `--port`, `--tp-size`, `--dp-size`, `--grpc-mode`,
  `--served-model-name`, and `--tokenizer-path`, with unknown server args
  preserved for incremental upstream compatibility. The crate builds both
  `sglang` and `sglang-rs` binaries so the upstream command shape works.
- `engine`: text and tokenized request lifecycle glue that drives prefill and
  decode until completion.
- `router`: SGLang gateway/router protocol boundary types for tokenized
  `Generate` requests/responses using the current `tokenized.input_ids: u32`
  and `chunk`/`complete` stream shape, plus health and model-info responses used
  during worker registration. `RouterRuntime` adapts these requests into the
  engine's tokenized generation path.
- `tokenizer`: tokenizer trait plus a temporary byte tokenizer for tests.
- `scheduler`: waiting queue, prefill/decode batch formation, request stages,
  uncached-token budgeted prefill batching, decode requeueing,
  `max_new_tokens` stopping, prefix-cache application, and KV cache page
  allocation for uncached prefill tokens. Successful prefill dispatches publish
  allocated pages back into RadixCache for future prefix reuse.
- `model_executor`: prepared model-worker batches with flattened input ids,
  positions, sequence lengths, request offsets, and prefix cache pages for the
  future CUDA/model executor boundary.
- `cache`: RadixCache-style token-prefix matching plus a finite KV cache page
  allocator for page assignment/reuse.
- `worker`: batch model worker trait that will become the CUDA/model executor boundary.
- `types`: generation request and response types.

The implementation is intentionally small while the architecture is being
carved out. The current worker is test-driven and mockable; CUDA integration is
not implemented yet.

## Development

Run all checks:

```bash
cargo fmt --all --check
cargo test --workspace
```
