# Agent Instructions

This repository is a Rust rewrite and extension of SGLang runtime pieces. Keep
the public CLI and behavior aligned with community SGLang unless there is a
clear reason to diverge.

## Backend Work

Runtime and backend boundaries must stay explicit. `--device cuda` on B200 is
the first production acceptance target, but B200 and NVIDIA Blackwell are CUDA
backend validation targets, not core runtime assumptions. Core scheduling,
request lifecycle, PD routing, and transfer orchestration must not branch on a
specific GPU SKU.

Device-specific behavior belongs behind backend interfaces and capability
records. A CUDA implementation can specialize internally for B200, H100, A100,
or future devices, but the public SRT logic should ask for capabilities rather
than hard-coding those devices.

The CPU tiny/reference models are correctness and wiring tools only. They may be
used by local tests, smoke scripts, and CPU reference execution, but they must
not silently satisfy a production accelerator request.

### Runtime Capabilities

Every executable runtime backend should publish a capability record that can be
validated at launch. At minimum, the record should express:

- device family, such as `cpu`, `cuda`, `musa`, `xpu`, `npu`, `hpu`, `metal`, or
  `rocm`
- device capability or architecture level, such as CUDA compute capability when
  available
- supported model dtypes, including `fp16`, `bf16`, and `fp8`
- attention backend support, including prefill/decode distinctions when they
  differ
- tensor-parallel support and known TP constraints
- KV cache memory registration support
- supported KV transfer transports, including Mooncake, RDMA, NVLink, TCP-only,
  or none

Capability validation must fail fast when the requested device or feature is not
available. Error messages should say what was requested, what runtime was
loaded, and which capability is missing.

### CUDA Acceptance

CUDA should be implemented as one production backend. B200 is the first real
acceptance target for that backend, but the interfaces must allow H100, A100,
and later CUDA devices to be added by extending capability detection and backend
implementation details.

CUDA code may contain device-specific optimized paths under the CUDA backend,
but feature checks should be driven by capability data. Avoid checks such as
`if b200` in shared SRT logic. Prefer checks such as `supports_fp8`,
`supports_mooncake_rdma`, `supports_nvlink`, or `compute_capability >= ...`
inside backend capability validation.

### Transfer Backends

KV transfer is a separate backend boundary from model compute. Real Mooncake
transfer, fake/local transfer, and local snapshot transfer must stay distinct:

- `mooncake` and normalized `mooncake_tcp` are production transfer backends.
- `fake` is a reference/test transfer backend only.
- local snapshot transfer is a CPU-verifiable test/reference mechanism, not a
  production Mooncake substitute.
- planned transfer targets such as NIXL, Ascend, and Mori must fail fast until
  they have real executors.

The runtime should never satisfy a production transfer request by falling back to
fake or local snapshot transfer.

### Fail-Fast Requirements

Unsupported backend, device, dtype, tensor parallel, attention, KV registration,
or transfer combinations must fail before serving traffic whenever the mismatch
is knowable at launch.

The failure should include:

- requested device and transfer backend
- loaded runtime name and capability class
- missing capability, such as `fp8`, `tensor_parallel`, `registered_kv_memory`,
  `mooncake_rdma`, or a specific attention backend
- actionable next step, such as using `--device cpu` for reference tests or
  enabling a real CUDA runtime implementation

Silent fallback to CPU reference, fake transfer, or placeholder generation is
only allowed in explicit local/reference configurations.

### Feature Completion Checklist

Each backend or transfer feature should document and test three validation
levels:

- Local MacBook/reference validation: what can be checked without production
  GPU hardware, usually parser behavior, capability validation, CPU reference
  semantics, and fail-fast errors.
- B200/CUDA validation: what must run on the first production target, including
  CUDA runtime loading, dtype coverage, tensor parallel behavior, KV memory
  registration, and Mooncake-linked transfer.
- Other GPU/backend status: whether H100, A100, Metal, ROCm, MUSA, XPU, NPU, or
  HPU are supported. If unsupported, the expected fail-fast error must be
  covered by tests or a documented manual check.

Completion notes for a feature should explicitly state whether it is production
ready, reference-only, metadata-only, or planned.

## CLI Compatibility

Use community SGLang CLI names where possible. For device selection, use
`--device`, not a custom public runtime-backend flag. Internal runtime
capability types may be more detailed than the public CLI.
