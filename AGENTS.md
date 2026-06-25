# Agent Instructions

This repository is a Rust rewrite and extension of SGLang runtime pieces. Keep
the public CLI and behavior aligned with community SGLang unless there is a
clear reason to diverge.

## Backend Work

Before implementing runtime, device, kernel, or KV-transfer features, read
`docs/backend-implementation-standards.md` and follow it as a mandatory
engineering checklist.

Key rules:

- B200/CUDA is the first production acceptance target, but B200 and NVIDIA
  Blackwell are validation targets for the CUDA backend, not core runtime
  assumptions.
- Runtime/backend boundaries must be capability-driven. Express device
  capability, supported dtypes, attention backend, tensor parallel support, KV
  memory registration, and Mooncake/RDMA/NVLink support through backend
  capability records.
- CPU tiny/reference models are reference/test backends only. They must not
  silently satisfy production accelerator requests.
- Fake/local snapshot KV transfer and real Mooncake-linked KV transfer must stay
  separate transfer backends.
- Unsupported backend, device, dtype, tensor parallel, attention, KV memory
  registration, or transfer combinations must fail fast with clear errors.
- Each feature must document what can be verified locally on a MacBook, what
  must be verified on B200/CUDA, and what other GPU backends support or how they
  fail fast.

## CLI Compatibility

Use community SGLang CLI names where possible. For device selection, use
`--device`, not a custom public runtime-backend flag. Internal runtime
capability types may be more detailed than the public CLI.
