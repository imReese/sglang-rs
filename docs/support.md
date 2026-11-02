# Models and Backends

This guide distinguishes implemented execution paths from full-checkpoint
and hardware acceptance. Return to the [project overview](../README.md).

## Model Coverage

`ModelRegistry` selects adapters from Hugging Face `architectures`. Adapters
parse configuration, map weights, compose shared Transformer components, and
declare capability requirements. Recognition of an architecture alone does
not establish that an executable backend exists.

| Hugging Face architecture | Shared components | CPU reference | CUDA implementation |
| --- | --- | --- | --- |
| `Qwen2ForCausalLM`, `Qwen3ForCausalLM` | Dense decoder, MHA/GQA | FP32 | BF16; dense tensor-parallel path |
| `Qwen3_5ForConditionalGeneration` | Gated DeltaNet + full attention | FP32 | BF16; single rank |
| `DeepseekV3ForCausalLM`, `KimiK25ForConditionalGeneration` | MLA + MoE | FP32 | BF16; single rank |
| `KimiLinearForCausalLM` | KDA + MLA/MoE | FP32 | BF16; single rank |
| `GlmMoeDsaForCausalLM`, `DeepseekV4ForCausalLM` | Shared MLA/MoE definitions | No executable path | No executable path |

The full Qwen3.5-0.8B checkpoint has been exercised on CPU. Other paths have
focused Transformer fixture tests and, where applicable, opt-in CUDA hardware
tests. Checkpoint variants and quantization formats are validated per adapter.
Vision/audio inputs are not supported by these text paths.

GLM DSA remains a thin adapter over shared DeepSeek/MLA-MoE definitions. Until
its required execution components exist, startup fails with the missing
capabilities rather than substituting another model.

## Backend Selection

| Device selection | Behavior |
| --- | --- |
| `--device cpu` | Explicit FP32 reference backend; single-rank execution |
| `--device cuda` | CUDA execution with capability checks and BF16 kernels |
| `--device auto` | Probes registered production backends, currently CUDA; never falls back to CPU |
| `--device metal`, `rocm`, or `musa` | No registered execution provider; startup fails explicitly |

CUDA execution requires the NVIDIA driver, NVRTC, and cuBLAS; dense tensor
parallelism additionally requires NCCL. Device placement uses ordinals and
rank configuration, not GPU product names. Dense tensor parallelism is
currently single-node; unavailable topology or component requirements fail
at startup. The current BF16 execution path requires compute capability 8.0
or newer. B200 is the first production acceptance target, not a hard-coded
runtime or CI requirement.

## Runtime Architecture

```text
HTTP / gRPC serving
        |
Engine / Scheduler / WorkerGroup
        |
ModelRunner + backend-owned execution resources
        |
Shared dense / hybrid / MLA / MoE components
        |
Backend kernels and device memory

ModelRegistry -> model adapter -> ModelDefinition
Active KV storage -> NexusKV descriptor -> transfer backend
```

- **Serving and scheduling:** request lifecycle, batch formation, admission,
  token budgets, completion, and cancellation are independent of model family.
- **Model execution:** adapters describe models; shared executors run their
  components. Models do not own separate schedulers or transfer lifecycles.
- **Backend resources:** providers create compatible executors and execution
  resources. `ModelRunner` owns the active KV pool; backends allocate its
  physical memory.
- **KV and transfer:** scheduling manages logical pages and token slots.
  Runtime KV storage directly implements NexusKV `KvCacheMemoryProvider`;
  transfer consumes its `TransferableKvCacheMemory` descriptor.
- **Kernels:** `sglang-kernel` contains CPU reference primitives and CUDA
  Driver, NVRTC, cuBLAS, and NCCL bindings. Hardware differences are expressed
  through capabilities rather than model-specific execution infrastructure.

## KV Capacity

`--page-size` is the number of token slots per physical KV page.
`--num-reserved-decode-tokens` is the total slot capacity and must be nonzero
and divisible by the page size. Invalid geometry fails before serving begins.
Prefix reuse operates on complete physical pages and is disabled for recurrent
models until prefix-state restoration is available.

For network transfer restrictions and shutdown behavior, see
[P/D disaggregation](disaggregation.md). For test commands and acceptance
boundaries, see [validation](validation.md).
