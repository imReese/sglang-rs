# sglang-router

Slim, KV-aware, OpenAI-compatible router for SGLang workers.

**Status:** functional HTTP/gRPC proxy aligned with the SGLang model gateway
shape. Exposes `/v1/tokenize`, `/v1/detokenize`, `/v1/models`,
`/v1/chat/completions` (buffered and SSE), `/generate`, plus `/healthz` /
`/readyz`, `/v1/loads`, and `/get_loads`. Control-plane forwarding covers
`/update_weights_from_disk`, `/flush_cache`, `/pause_generation`,
`/continue_generation`, `/abort_request`, `/start_profile`, and
`/stop_profile` across plain workers and PD prefill/decode pools.
`/abort_request` supports targeted `rid` aborts and SGLang-compatible
`abort_all`.

## Building

```bash
cargo build -p sglang-router --release
```

## License

Apache-2.0.
