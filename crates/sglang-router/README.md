# sglang-router

Slim, KV-aware, OpenAI-compatible router for SGLang workers.

**Status:** functional HTTP/gRPC proxy aligned with the SGLang model gateway
shape. Exposes `/v1/tokenize`, `/v1/detokenize`, `/v1/models`,
`/v1/chat/completions` (buffered and SSE), `/generate`, plus `/healthz` /
`/readyz`. Control-plane forwarding covers `/update_weights_from_disk`,
`/flush_cache`, `/pause_generation`, `/continue_generation`, and
`/abort_request` across plain workers and PD prefill/decode pools.

## Building

```bash
cargo build -p sglang-router --release
```

## License

Apache-2.0.
