# Runtime Architecture

## Purpose

sglang-rs is an architectural reimplementation of SGLang's runtime in Rust. It is not intended to be a line-by-line translation of the upstream Python runtime.

The primary goal is to establish a native runtime boundary in which:

- request and batch scheduling are CPU-side control-plane work;
- KV/cache state is a first-class runtime resource;
- transfer decisions are separated from transfer execution;
- GPU execution remains asynchronous and owns CUDA streams, events, and model execution;
- Python remains an integration surface rather than the owner of the hot runtime event loop;
- backend-specific execution stays behind explicit capability boundaries.

The long-term architecture is a CPU control plane plus GPU execution plane inside one runtime process first. A process boundary remains an optional later optimization, not a prerequisite.

## Architectural Principle

> The CPU side decides what should happen; the GPU side executes how and when the GPU work happens.

This separates scheduling semantics from device execution without splitting cache ownership into two competing authorities.

```text
                         SGLang Runtime Process
┌─────────────────────────────────────────────────────────────────┐
│  Python / API integration                                       │
│            │                                                    │
│            ▼                                                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    CPU Control Plane                      │  │
│  │  Scheduler  Batch Planner  KV/Cache Manager               │  │
│  │                  │                                         │  │
│  │             Transfer Planner                               │  │
│  └───────────────────────┬───────────────────────────────────┘  │
│                          │ ExecutionPlan / commands              │
│  ┌───────────────────────▼───────────────────────────────────┐  │
│  │                    GPU Execution Plane                    │  │
│  │  ModelRunner  GPU KV Pool  CUDA Runtime  Transfer Exec   │  │
│  │                    │             │              │          │  │
│  │                    └─────────────┴──────────────┘          │  │
│  │                         streams / events                   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Why Workers Are Threads First

The first implementation target is to place the CPU worker and GPU execution worker/runtime in the same process, using Rust threads and asynchronous CUDA execution.

This avoids introducing CUDA IPC, shared-memory lifecycle management, and cross-process synchronization before they are proven necessary. It also allows one CUDA runtime domain, direct sharing of allocation handles, shared pinned-memory pools, direct CUDA stream/event ownership, low-overhead typed command queues, and simpler failure/shutdown semantics.

A separate process may become useful later for fault isolation or independent scaling. It should be introduced only after measurements show that threads are insufficient.

Python remains the public integration layer while Rust owns the long-lived worker loops and synchronization primitives.

## CPU Control Plane

The CPU control plane owns the meaning and lifecycle of cache state.

### Scheduler

The scheduler is responsible for request lifecycle, waiting/running queues, prefill/decode admission, batch formation, token budgets, request completion/abort, and parallel-execution coordination.

The scheduler should not block on the entire GPU execution of a batch.

### KV/Cache Manager

The KV/cache manager is the authoritative owner of logical cache state:

- page allocation and reuse;
- prefix matching;
- residency;
- reference/lifecycle state;
- eviction policy;
- cache publication;
- transfer state associated with cache operations.

There must be one authoritative logical state. The GPU execution plane must not maintain a second independent interpretation of L2 residency.

### Transfer Planner

The CPU side decides which pages must move, their source/destination logical locations, byte ranges, ordering/dependencies, and whether movement is required at all.

The result is a typed transfer description rather than an imperative request to a backend.

## GPU Execution Plane

The GPU side owns execution resources and device-side synchronization.

### ModelRunner

The model runner owns model execution and the data required to execute a prepared batch:

- GPU-resident model state;
- GPU KV page addressing;
- compute submission;
- model-specific execution;
- device-side synchronization.

### CUDA Runtime

The CUDA runtime owns context/runtime initialization, streams, events, device allocation, host/device copy submission, capability checks, and GPU-side synchronization.

### Transfer Executor

The transfer executor consumes transfer descriptions from the CPU control plane and turns them into concrete operations:

```text
TransferPlan
     │
     ├── local H2D/D2H
     ├── Mooncake
     ├── NIXL (future)
     └── other transport backends
```

The executor must not decide cache policy or prefix reuse.

## Cache Ownership and Physical Memory

L2 cache should remain a single logical component even when its physical storage is used by the GPU.

```text
                 L2 Cache
             logical ownership
                    │
        ┌───────────┴───────────┐
        │                       │
   metadata / allocator    physical memory
                                │
                         CUDA-visible pool
                                │
                                ▼
                           GPU transfer
```

The architecture distinguishes ownership from execution.

The CPU control plane owns page identity, allocation state, residency, eviction, and logical lifetime.

The GPU execution plane executes H2D/D2H, CUDA synchronization, and model consumption of transferred pages. It does not own an independent L2 cache state.

## Host Memory Registration

Dynamic `cudaHostRegister()` must not be part of the per-transfer fast path.

The following pattern is explicitly rejected:

```text
allocate host page
    -> cudaHostRegister
    -> H2D
```

Registration latency can otherwise become a cache-movement cost visible to the scheduler.

The preferred direction is to establish CUDA-ready host memory during runtime initialization and reuse it:

```text
runtime startup
    -> establish host/pinned pool
    -> register once if required
    -> reuse slots for many transfers
```

The exact strategy remains a benchmark question:

1. pre-established pinned/registered L2 pool;
2. CUDA-created host pool;
3. normal DRAM plus a bounded pinned staging pool.

The choice must be based on end-to-end measurements including NUMA locality, CPU memory bandwidth, H2D/D2H bandwidth, registration cost, and overlap with model execution.

## ExecutionPlan Boundary

The CPU and GPU planes should communicate through stable typed plans rather than sharing high-level scheduler objects.

```text
ExecutionPlan
├── BatchPlan
├── KVPlan
├── TransferPlan
└── SynchronizationPlan
```

A transfer command answers what data must move and where it belongs. It should not encode model-specific cache policy.

A GPU worker should be able to execute a plan without understanding why a particular prefix was selected.

## Overlap Model

The runtime should be designed around pipeline overlap rather than sequential batch execution.

```text
time ──────────────────────────────────────────────>

CPU worker:    prepare N+1 ─────────────── prepare N+2
                    │
GPU transfer:       H2D(N+1) ────
GPU compute:              Forward(N) ───────────────
GPU release:                         D2H/release(N-1)
```

The exact ordering is governed by CUDA events and resource dependencies.

Correctness must not depend on a visual assumption of overlap. Nsight Systems, NVTX ranges, CUDA events, and runtime counters should verify CPU preparation overlap, safe transfer/compute overlap, transfer-before-consumption, successful cache publication, and failure handling.

## Parallelism

Parallel execution is a separate dimension from scheduling.

The runtime should represent topology explicitly:

```text
ExecutionTopology
├── DP rank
├── PP stage
└── TP rank
```

PP should be modeled as an explicit pipeline of receive/process/release operations rather than hidden inside a monolithic scheduler loop.

```text
receive
   ↓
process
   ↓
release
   ↓
next iteration
```

Inter-stage communication and CUDA synchronization are execution concerns; request admission and cache policy remain control-plane concerns.

## Failure and State Rules

KV movement follows explicit state transitions:

```text
planned
  ↓
submitted
  ↓
in-flight
  ↓
completed ──→ publish
```

Failure follows:

```text
submitted / in-flight
  ↓
failed
  ↓
cleanup
  ↓
do not publish
```

GPU consumption requires:

```text
H2D submitted
    ↓
H2D complete
    ↓
page usable
    ↓
forward may consume
```

D2H requires:

```text
forward complete
    ↓
D2H complete
    ↓
L2 page resident
    ↓
page may be published
```

> A cache page becomes visible to future reuse only after its physical contents are known to be complete and valid.

## Relationship to NexusKV

sglang-rs and NexusKV should not duplicate the same abstraction.

```text
SGLang Rust
    │
    ├── request lifecycle
    ├── model-specific scheduling
    ├── execution topology
    └── cache lifecycle integration
             │
             ▼
         NexusKV
    ├── cache identity
    ├── descriptors
    ├── generic matching
    ├── reuse planning
    └── engine-neutral cache primitives
```

SGLang owns inference semantics. NexusKV owns reusable cache primitives and engine-neutral planning.

## Non-Goals

The architecture does not currently aim to:

- reproduce every upstream Python module in Rust;
- make Python and Rust schedulers co-own the same state;
- introduce multi-process CUDA IPC before measurement justifies it;
- support every accelerator backend before a stable backend interface exists;
- hide unsupported capabilities behind silent CPU or fake fallbacks;
- make the GPU worker responsible for cache policy.

## Architecture Decision Summary

1. Rust owns the CPU control plane.
2. GPU execution remains asynchronous and backend-specific.
3. CPU and GPU workers are same-process first.
4. L2 cache remains one logical component with one authoritative state.
5. Transfer planning and transfer execution are separate boundaries.

This is the foundation for evolving sglang-rs from a Rust rewrite into a native inference runtime.
