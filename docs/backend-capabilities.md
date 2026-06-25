# Backend and Capability Boundaries

This project keeps compute backends, CPU reference execution, and transfer
backends as separate launch-time capabilities. A worker must fail fast when the
requested backend does not match the runtime that was actually loaded.

## Runtime Backends

`--device` is the community-aligned compute device selector:

- `auto`: compatibility mode for local development and existing smoke tests. It
  accepts the best runtime currently loadable for the model path. This mode is
  not a production acceptance target.
- `cpu`: correctness and local wiring backend. It is allowed to run CPU
  reference models and bootstrap placeholder paths, but it is not a GPU
  production backend.
- `cuda`: first production backend target. B200 validation must launch workers
  with this value and must not fall back to CPU reference execution.
- `musa`, `xpu`, `npu`, `hpu`: community device targets accepted by the Rust
  parser and required to fail fast until an executable runtime is registered.
- `metal`, `rocm`: planned Rust runtime device targets kept behind the same
  internal capability boundary for future non-community platform experiments.

Current runtime capability mapping:

| Runtime | Capability | Forward | Transferable KV |
| --- | --- | --- | --- |
| `space-reference` | `cpu-reference` | yes | no |
| `cpu-embedding-lm` | `cpu-reference` | yes | no |
| `glm-moe-dsa-f32-cpu` | `cpu-reference` | yes | yes |
| `deepseek-v4-metadata` | `metadata-only` | no | no |
| unsupported local model types | `unsupported` | no | no |

When `--device cuda` is requested today, these CPU and metadata-only runtimes
reject startup with `UnsupportedDevice`. The future CUDA
executor should register a `Production(RuntimeBackend::Cuda)` capability for
the model family it implements instead of weakening this check.

## Transfer Backends

Transfer backends describe KV movement, not compute execution:

- `mooncake` and normalized `mooncake_tcp` are production transfer backends.
  `mooncake_tcp` forces TCP transport while retaining the Mooncake backend.
- `fake` is a reference-only transfer backend for local CPU PD smoke tests.
  It must not be used as evidence that the production transfer path works.
- `nixl`, `ascend`, and `mori` are planned backends. They are parsed as explicit
  future targets but remain unsupported by the bootstrap launcher until real
  executors are implemented.

The first real GPU acceptance target is B200 with:

```bash
--device cuda --disaggregation-transfer-backend mooncake
```

The CPU PD smoke script intentionally uses:

```bash
--device cpu --disaggregation-transfer-backend fake
```

That path is valuable for scheduler/router/PD wiring, but it is not a
production backend validation.
