# Runtime Roadmap

## Objective

The next stage of sglang-rs is not to maximize the amount of upstream SGLang code rewritten in Rust. The objective is to make the runtime execution model explicit, measurable, and extensible.

Priority:

1. establish CPU/GPU worker boundaries;
2. remove GPU execution from the CPU scheduler's blocking path;
3. make KV movement a first-class asynchronous operation;
4. establish a robust L2 memory model;
5. make TP/DP/PP execution topology explicit;
6. only then broaden model/backend coverage.

## Phase 1 — Runtime Boundary

- Introduce a Rust-owned CPU worker loop.
- Introduce a GPU execution worker/runtime boundary.
- Define typed BatchPlan, KVPlan, TransferPlan, and ExecutionPlan.
- Ensure the scheduler can enqueue work without waiting for full GPU execution.
- Keep CPU and GPU workers in the same process.

**Acceptance:** the scheduler event loop continues CPU-side work while the GPU executes an already-submitted batch.

## Phase 2 — KV Pipeline

- Move KV allocation, matching, residency, and publication into an explicit cache manager.
- Separate transfer planning from transfer execution.
- Model H2D/D2H completion as explicit events.
- Make cache publication conditional on successful transfer completion.
- Add planned/submitted/completed/failed transfer counters.

**Acceptance:**

```text
H2D complete -> GPU page usable
D2H complete -> L2 page publishable
transfer failure -> page never published
```

## Phase 3 — L2 Memory

Benchmark rather than assume the best host-memory strategy:

| Strategy | Main advantage | Main risk |
| --- | --- | --- |
| Pre-pinned/registered pool | Direct DMA, no per-transfer registration | pinned-memory pressure |
| CUDA-created host pool | CUDA-ready allocation | large-pool/system-memory trade-offs |
| Normal DRAM + staging | smaller pinned footprint | extra host memcpy |

Required measurements:

- `cudaHostRegister()` latency;
- initialization cost of large pools;
- H2D/D2H bandwidth;
- CPU memcpy bandwidth;
- NUMA sensitivity;
- concurrent transfer behavior across GPUs;
- scheduler CPU utilization;
- end-to-end throughput;
- overlap with forward execution.

The key metric is total pipeline impact, not registration latency alone.

## Phase 4 — Execution Topology

Introduce:

```text
ExecutionTopology
├── DP
├── PP
└── TP
```

### TP

Keep model execution and collective communication behind the GPU execution boundary.

### PP

Represent each stage as:

```text
receive
   ↓
process
   ↓
release
   ↓
next iteration
```

Inter-stage communication should not force the CPU scheduler to become a pipeline state machine.

### DP

Keep request routing/admission on the CPU side while rank-local execution remains on the GPU side.

## Phase 5 — Measurement and Observability

Add:

- NVTX ranges for CPU scheduling, transfer submission, forward, and completion;
- transfer queue depth;
- H2D/D2H bytes and latency;
- CUDA event timestamps;
- cache hit/miss/reuse counters;
- L2 residency and eviction counters;
- scheduler idle/busy time;
- GPU compute/transfer overlap metrics.

A performance change should be accepted only when the timeline or counters demonstrate the intended effect.

## Phase 6 — Model Coverage

Only after the runtime boundary stabilizes:

- complete production dense-model execution;
- add MLA/KDA-specific state handling;
- add MoE execution;
- expand model compatibility;
- validate numerical equivalence against upstream SGLang.

Model-specific state should use the generic runtime state lifecycle rather than creating independent cache-management paths.

## Phase 7 — Backend Expansion

CUDA is the first production backend.

After the core runtime is stable, extend the same contracts to ROCm/HIP, MUSA, Metal/MLX where practical, and additional transfer backends.

Unsupported capabilities must remain explicit.

## Process Boundary: Later, Not First

A multi-process architecture is intentionally deferred.

Only consider:

```text
CPU process
     │
     │ IPC
     ▼
GPU process
```

after measuring that same-process workers cannot meet concrete requirements such as fault isolation, independent process lifecycle, CPU-side scalability, or deployment constraints.

If introduced, IPC should carry the same logical ExecutionPlan/transfer descriptors rather than leaking scheduler objects across the boundary.

## Success Criteria

The runtime architecture is successful when:

1. `run_batch` no longer serializes CPU scheduling with GPU execution;
2. CPU-side KV planning progresses while GPU work is in flight;
3. H2D/D2H is asynchronous and event-driven;
4. L2 has one authoritative logical owner;
5. cache publication is correctness-safe;
6. TP/DP/PP are represented explicitly;
7. performance claims are supported by timeline measurements;
8. adding a transfer backend does not require rewriting scheduler policy.

## Current Position

The project should be judged as an architecture/runtime effort rather than as a feature-count comparison with upstream SGLang.

The most valuable near-term result is a small, correct, measurable CPU-control/GPU-execution runtime. More model and backend coverage should follow that foundation rather than determine it.
