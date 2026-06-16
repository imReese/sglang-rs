# Mooncake-Linked PD Runtime Design

## Context

The project goal is to build a Rust SGLang runtime that removes Python-heavy
request scheduling and serving overhead while remaining compatible with the
community SGLang router and `sgl-model-gateway` PD launch surface. Upstream
SGLang Q2 2026 roadmap work calls out P/D disaggregation, native Rust API/gRPC
serving, scheduler refactors, and Rust migration as related efforts.

This repository already has most of the protocol scaffolding:

- `sglang-srt` exposes HTTP and gRPC worker servers.
- `sglang-router` mirrors the `sgl-model-gateway` PD worker-pair shape.
- `transfer.rs` defines `mooncake-link`, Mooncake engine FFI traits, bootstrap
  session tracking, transfer request construction, and unlinked fallback errors.
- `pd_bootstrap.rs` implements prefill bootstrap HTTP and ZMQ ingestion for
  route registration, decode KVArgs registration, and transfer metadata.
- GLM runtime work has started exposing KV cache page snapshots and real
  forward/cache behavior.

The first production-aligned milestone is not a local snapshot path. It is a
feature-gated Mooncake-linked network KV transfer path that can run real Rust
prefill and decode workers behind the standard router/gateway interface.

## Goal

Implement a Mooncake-linked Rust PD runtime path that can:

1. Start a Rust prefill worker with `--disaggregation-mode prefill` and
   `--disaggregation-transfer-backend mooncake`.
2. Start a Rust decode worker with `--disaggregation-mode decode` and
   `--disaggregation-transfer-backend mooncake`.
3. Register decode KV memory with Mooncake and publish decode KVArgs plus
   transfer metadata to the prefill bootstrap service.
4. Transfer prefill KV pages over Mooncake into decode KV pages.
5. Let decode generation wait for KV transfer success, then continue the real
   model forward path.
6. Route the pair through `sgl-router` and remain compatible with the community
   `sgl-model-gateway` PD flags:
   `--pd-disaggregation --prefill <url> <bootstrap_port> --decode <url>`.

## Non-Goals

This milestone does not implement the full Python Mooncake matrix. The
following remain follow-up work:

- Heterogeneous prefill/decode tensor-parallel slicing.
- Staging buffers for M-to-N TP transfer.
- Mamba, SWA, DSA auxiliary state transfer beyond the GLM KV path needed for
  the first service smoke.
- Multi-node Kubernetes discovery changes.
- A replacement for the community gateway. The gateway remains the router
  standard component.

## Architecture

The linked runtime keeps the current separation between routing, scheduling,
bootstrap metadata, transfer execution, and model execution.

`sgl-router` or `sgl-model-gateway` selects a PD worker pair. It sends a
prefill request containing `bootstrap_host`, `bootstrap_port`, and
`bootstrap_room`, then sends the matching decode request to the decode worker.

The decode worker owns destination KV memory. During service initialization it
creates a linked `SharedLinkedMooncakeTransferEngine`, registers its KV buffer
regions, and builds a `MooncakeDecodeBootstrapPublisher` containing the
Mooncake session id and destination KV layout. Before prefill transfer, the
publisher sends KVArgs registration and per-request transfer metadata through
the prefill bootstrap route.

The prefill worker owns source KV memory. Its bootstrap HTTP service advertises
rank routes and its ZMQ endpoints ingest decode metadata. After prefill forward
fills source KV pages, `MooncakeBootstrapKvCacheTransferExecutor` resolves the
decode remote layouts from bootstrap state and submits Mooncake write requests
for the exact page spans prepared by `KvCacheTransferPlan`.

The scheduler already treats PD decode requests as pending until
`DecodeBootstrapRegistry` reports `KvPoll::Success`. Linked Mooncake transfer
keeps that contract: submitted batches move through `Transferring`, are polled
via `transfer_status`, and mark the bootstrap room `Success` only when all
Mooncake tasks complete.

## Components

### Mooncake Engine Boundary

`SharedLinkedMooncakeTransferEngine` remains behind `mooncake-link`. It must
support:

- Initializing with `hostname`, `gpu_id`, metadata server `P2PHANDSHAKE`,
  protocol `tcp` or `rdma`, and optional IB device.
- Returning a Mooncake session id equivalent to Python's `host:rpc_port`.
- Registering and unregistering one or more KV memory regions.
- Opening remote segments for decode sessions.
- Submitting batch write requests and polling task status.

The unlinked build must keep returning the current clear runtime error so the
default workspace remains buildable without Mooncake libraries.

### KV Memory Layout Provider

Model runners need a typed way to expose Mooncake-transferable KV memory
without hard-coding GLM internals in `server.rs`.

The design adds a small runtime layout interface around existing model runner
boundaries. For the first implementation it should expose:

- Source base pointers for prefill KV memory.
- Destination base pointers for decode KV memory.
- Page size in bytes.
- Page count or registered byte length.
- Optional per-layer/per-tensor split metadata when the model stores K/V in
  multiple buffers.

GLM can initially use a packed page layout if its in-memory cache is already
stored as page-contiguous transfer units. If GLM exposes K/V split buffers, the
remote layout must use one pointer per split with `dst_kv_item_len` matching
the bytes per page for that split, exactly as the current
`build_mooncake_remote_kv_transfer_requests` expects.

### Decode Bootstrap Publisher

`MooncakeDecodeBootstrapPublisher` should be constructed from the live decode
KV layout rather than placeholder zero addresses. It sends:

- A one-time KVArgs registration per bootstrap endpoint, TP rank, PP rank, and
  Mooncake session id.
- Transfer metadata for each request span, including `bootstrap_room`,
  destination KV page indices, and decode prefix length.

The publisher remains synchronous at the worker boundary for now because
`FallibleModelWorker::try_generate_batch` is synchronous. Errors must include
the bootstrap address, route query, and ZMQ endpoint where possible.

### Prefill Transfer Executor

`MooncakeBootstrapKvCacheTransferExecutor` should continue to resolve
`MooncakeRemoteKvLayout` from the prefill bootstrap state. Transfer request
construction must validate:

- Non-zero page size and destination item length.
- Source page count equals span token count for this milestone.
- Destination KV index count equals span token count.
- Remote split layout byte size equals local page size.
- All address and offset arithmetic is checked.

The linked executor submits Mooncake batches, records submitted batch ids, and
polls them through the existing registry state machine.

### Server Builders

The launch path should keep the current mode/backend matrix:

- `Null`: normal HTTP/gRPC worker.
- `Prefill + Mooncake`: Rust prefill worker plus bootstrap HTTP/ZMQ service.
- `Decode + Mooncake`: Rust decode worker plus linked Mooncake transfer engine.
- Unsupported combinations return explicit errors.

For linked builds, prefill and decode builders should derive `MooncakeKvCacheLayout`
from actual model/runtime KV memory when available. If the selected model cannot
expose transferable KV memory, startup must fail with a model-specific error
instead of silently using address zero.

## Data Flow

1. Router receives a client generation request and chooses a prefill/decode
   worker pair.
2. Router injects `bootstrap_host`, `bootstrap_port`, and `bootstrap_room` into
   the prefill request and sends the decode request to the selected decode URL.
3. Decode worker enqueues the request, allocates destination KV pages, and
   publishes decode KVArgs plus per-room transfer metadata to prefill bootstrap.
4. Prefill worker runs prefill forward and fills source KV pages.
5. Prefill worker builds `KvCacheTransferPlan` from the prepared prefill batch.
6. Prefill worker resolves decode destination layouts from bootstrap state and
   submits Mooncake write requests.
7. Decode scheduler polls transfer status and keeps the request pending while
   the bootstrap room is `Bootstrapping`, `WaitingForInput`, or `Transferring`.
8. When all Mooncake tasks complete, decode marks the room `Success` and runs
   decode forward using the transferred KV pages.
9. Router streams or returns the final OpenAI-compatible response.

## Error Handling

Startup errors should fail fast for:

- `mooncake-link` requested but Mooncake libraries are not linkable.
- Missing KV cache model layout when decode uses Mooncake.
- Selected model/runtime does not expose transferable KV memory.
- Mooncake engine initialization, transport install, or memory registration
  failure.

Request-time errors should:

- Mark the bootstrap room `Failed` on transfer submit or status failure.
- Surface a worker runtime error through HTTP/gRPC with the original Mooncake
  status code or bootstrap/ZMQ failure context.
- Remove registry state when a request completes or fails.
- Never let decode run with an unknown or failed bootstrap room.

## Testing

The implementation should use TDD and keep tests split by environment:

- Default, no-Mooncake tests:
  - `cargo test --workspace` continues to pass.
  - Unlinked Mooncake paths still return the explicit
    `mooncake-link` feature error.
  - Pure request-building tests cover remote split layouts and overflow checks.

- Feature-gated linked tests:
  - Mooncake engine initializes with TCP transport using
    `MOONCAKE_HOME` or `MOONCAKE_BUILD_DIR`.
  - A small registered host buffer can be written to another registered host
    buffer through Mooncake using the Rust FFI wrapper.
  - Decode publisher sends KVArgs and transfer metadata that the prefill
    bootstrap service ingests into remote layouts.
  - A real Rust SRT prefill/decode pair behind `sgl-router` returns a successful
    `/generate` or `/v1/chat/completions` response in linked mode.

- Manual GPU smoke:
  - `scripts/run_glm5_pd_gpu.sh` should build with `mooncake-link` when
    requested and run prefill worker, decode worker, and router.
  - `SMOKE_CHAT=1` should exercise a real request once GLM checkpoint and
    Mooncake runtime are available.

## Rollout

1. Preserve the default unlinked build and existing tests.
2. Land linked Mooncake FFI and host-buffer smoke tests.
3. Wire decode KV memory registration and publisher layout from real runtime
   memory.
4. Wire prefill source KV memory into linked transfer executor.
5. Convert router real-SRT PD e2e from expected unlinked failure to linked
   success under `mooncake-link`.
6. Push the completed feature branch after the linked service smoke passes or
   after the implementation reaches the strongest locally verifiable checkpoint
   if external Mooncake hardware/runtime is unavailable.

## Acceptance Criteria

- The default workspace test suite passes without Mooncake installed.
- With `--features mooncake-link` and a valid Mooncake build, Rust can initialize
  Mooncake, register KV memory, submit a network write, and observe completion.
- A decode worker publishes non-zero destination KV addresses derived from live
  runtime memory.
- A prefill worker submits transfer requests using non-zero source KV addresses
  derived from live runtime memory.
- Decode generation remains pending until Mooncake reports success, then runs
  model forward against transferred KV.
- `sgl-router` can route to Rust prefill/decode workers using the same PD
  worker arguments as `sgl-model-gateway`.
- Any unsupported model/runtime path fails explicitly and does not fall back to
  mock, zero-address, or no-op transfer behavior.
