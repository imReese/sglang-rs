# Mooncake-Linked PD Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a feature-gated Rust SRT Mooncake-linked PD transfer path that registers live KV memory, transfers prefill KV pages over Mooncake, and can be routed by `sgl-router`/`sgl-model-gateway`.

**Architecture:** Keep the existing router/scheduler/bootstrap separation. Add a typed live-KV memory layout boundary, fix the linked Mooncake session/registration path, then wire prefill/decode server builders to use real non-zero KV memory addresses. Default builds keep the current unlinked Mooncake error path.

**Tech Stack:** Rust 2024, Tonic, Axum, Tokio, ZeroMQ bootstrap frames, SGLang Mooncake C ABI behind `mooncake-link`, existing GLM cached forward runtime.

---

## File Structure

- Modify `crates/sglang-srt/src/transfer.rs`: add runtime KV memory layout types/traits, linked engine session id and memory registration helpers, and tests for transfer request construction using live layouts.
- Modify `crates/sglang-srt/src/glm_runtime.rs`: expose GLM cached model KV memory as transferable Mooncake page regions without changing GLM forward semantics.
- Modify `crates/sglang-srt/src/server.rs`: derive Mooncake prefill/decode layouts from the loaded model runtime and build decode publishers with the linked engine's real session id.
- Modify `crates/sglang-srt/src/cli.rs`: add an optional internal Mooncake RPC port flag so the Rust worker can publish a deterministic Mooncake session id.
- Modify `crates/sglang-srt/tests/pd_config.rs`: cover linked engine config/session behavior.
- Modify `crates/sglang-srt/tests/pd_transfer_plan.rs`: cover live-memory layout request construction.
- Modify `crates/sglang-srt/tests/glm_runtime.rs`: cover GLM runtime KV memory layout export.
- Modify `crates/sglang-srt/tests/server_bootstrap.rs`: cover server builder failure on unsupported runtime and non-zero layouts for supported runtime.
- Modify `crates/sglang-router/tests/proxy/real_srt_pd.rs`: keep unlinked failure test and add feature-gated linked success test.
- Modify `scripts/run_glm5_pd_gpu.sh`: add a `MOONCAKE_LINK=1` build path and required env documentation.

## Task 1: Linked Mooncake Session Identity And Memory Registration API

**Files:**
- Modify: `crates/sglang-srt/src/transfer.rs`
- Modify: `crates/sglang-srt/src/cli.rs`
- Test: `crates/sglang-srt/tests/pd_config.rs`

- [ ] **Step 1: Inspect the local Mooncake C ABI before editing**

Run:

```bash
rg -n "get.*rpc|get.*port|rpc_port|TransferEngine" "${MOONCAKE_HOME:-/home/reese/workspace/code/kvcache-ai/Mooncake}" -g '*.h' -g '*.hpp' -g '*.cc' -g '*.cpp'
```

Expected: record whether a callable C/C++ ABI exists for the runtime RPC port. This plan still uses the explicit `--disaggregation-mooncake-rpc-port` flag because the current Rust binding already passes `rpc_port` into `createTransferEngine`, and deterministic service startup is easier to verify.

- [ ] **Step 2: Write the failing test for explicit session id construction**

Add this test to `crates/sglang-srt/tests/pd_config.rs`:

```rust
#[test]
fn mooncake_engine_config_builds_session_id_from_hostname_and_rpc_port() {
    let args = sglang_srt::cli::ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--host",
        "127.0.0.1",
        "--port",
        "30002",
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-mooncake-rpc-port",
        "41002",
        "--kv-cache-dtype",
        "bfloat16",
        "--kv-cache-num-layers",
        "1",
        "--kv-cache-kv-heads",
        "1",
        "--kv-cache-head-dim",
        "8",
    ])
    .expect("args should parse");
    let pd_config = sglang_srt::transfer::PdConfig::from_server_args(&args)
        .expect("PD config should parse");
    let engine_config = sglang_srt::transfer::MooncakeTransferEngineConfig::from_pd_config_for_rank(
        "127.0.0.1",
        0,
        &pd_config,
    );

    assert_eq!(engine_config.rpc_port, Some(41002));
    assert_eq!(engine_config.session_id(), "127.0.0.1:41002");
}
```

- [ ] **Step 3: Run the red test**

Run:

```bash
cargo test -p sglang-srt --test pd_config mooncake_engine_config_builds_session_id_from_hostname_and_rpc_port
```

Expected: FAIL because `--disaggregation-mooncake-rpc-port`, `rpc_port`, and `session_id()` do not exist.

- [ ] **Step 4: Implement the minimal config/API**

Add a field to `ServerArgs` and `PartialServerArgs` in `crates/sglang-srt/src/cli.rs`:

```rust
pub disaggregation_mooncake_rpc_port: Option<u16>,
```

Parse the flag in `ArgParser::parse`:

```rust
"--disaggregation-mooncake-rpc-port" => {
    let value = self.take_value("--disaggregation-mooncake-rpc-port")?;
    self.parsed.disaggregation_mooncake_rpc_port = Some(
        value
            .parse::<u16>()
            .map_err(|_| CliParseError::InvalidPort(value))?,
    );
}
```

Carry it into `PdConfig` in `crates/sglang-srt/src/transfer.rs`:

```rust
pub mooncake_rpc_port: Option<u16>,
```

Set it in `PdConfig::from_server_args_with_model_layout`:

```rust
mooncake_rpc_port: args.disaggregation_mooncake_rpc_port,
```

Extend `MooncakeTransferEngineConfig`:

```rust
pub struct MooncakeTransferEngineConfig {
    pub hostname: String,
    pub gpu_id: usize,
    pub metadata_server: String,
    pub protocol: String,
    pub device_name: String,
    pub rpc_port: Option<u16>,
}

impl MooncakeTransferEngineConfig {
    pub fn session_id(&self) -> String {
        match self.rpc_port {
            Some(port) => format!("{}:{port}", self.hostname),
            None => self.hostname.clone(),
        }
    }
}
```

Update `from_pd_config` to copy `config.mooncake_rpc_port`.

In `LinkedMooncakeTransferEngine::new`, pass `config.rpc_port.unwrap_or(0) as u64` into `createTransferEngine`.

- [ ] **Step 5: Run the green test and CLI tests**

Run:

```bash
cargo test -p sglang-srt --test pd_config mooncake_engine_config_builds_session_id_from_hostname_and_rpc_port
cargo test -p sglang-srt --test cli_args
```

Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/sglang-srt/src/cli.rs crates/sglang-srt/src/transfer.rs crates/sglang-srt/tests/pd_config.rs
git commit -m "feat: configure mooncake linked session identity"
```

## Task 2: Transferable KV Memory Layout Boundary

**Files:**
- Modify: `crates/sglang-srt/src/transfer.rs`
- Test: `crates/sglang-srt/tests/pd_transfer_plan.rs`

- [ ] **Step 1: Write the failing layout trait test**

Add this to `crates/sglang-srt/tests/pd_transfer_plan.rs`:

```rust
#[test]
fn transferable_kv_memory_builds_prefill_and_decode_mooncake_layouts() {
    use sglang_srt::transfer::{
        MooncakeKvCacheLayout, MooncakeRemoteKvLayout, TransferableKvCacheMemory,
        TransferableKvCacheRegion,
    };

    let memory = TransferableKvCacheMemory::new(
        vec![TransferableKvCacheRegion {
            base_addr: 0x1000,
            byte_len: 4096,
            page_size_bytes: 128,
        }],
        128,
    )
    .expect("memory should be valid");

    assert_eq!(
        memory.prefill_layout(0x200),
        MooncakeKvCacheLayout {
            source_base_addr: 0x1000,
            page_size_bytes: 128,
            target_base_offset: 0x200,
        }
    );
    assert_eq!(
        memory.decode_remote_layout(&[4, 5]),
        MooncakeRemoteKvLayout {
            dst_kv_ptrs: vec![0x1000],
            dst_kv_indices: vec![4, 5],
            dst_kv_item_len: 128,
        }
    );
}
```

- [ ] **Step 2: Run the red test**

Run:

```bash
cargo test -p sglang-srt --test pd_transfer_plan transferable_kv_memory_builds_prefill_and_decode_mooncake_layouts
```

Expected: FAIL because `TransferableKvCacheMemory` and `TransferableKvCacheRegion` do not exist.

- [ ] **Step 3: Implement the layout boundary**

Add to `crates/sglang-srt/src/transfer.rs` near `MooncakeKvCacheLayout`:

```rust
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransferableKvCacheRegion {
    pub base_addr: usize,
    pub byte_len: usize,
    pub page_size_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransferableKvCacheMemory {
    regions: Vec<TransferableKvCacheRegion>,
    page_size_bytes: usize,
}

impl TransferableKvCacheMemory {
    pub fn new(
        regions: Vec<TransferableKvCacheRegion>,
        page_size_bytes: usize,
    ) -> Result<Self, KvCacheTransferError> {
        if page_size_bytes == 0 {
            return Err(KvCacheTransferError::Runtime(
                "transferable KV memory page size must be non-zero".to_string(),
            ));
        }
        if regions.is_empty() {
            return Err(KvCacheTransferError::Runtime(
                "transferable KV memory must expose at least one region".to_string(),
            ));
        }
        for region in &regions {
            if region.base_addr == 0 {
                return Err(KvCacheTransferError::Runtime(
                    "transferable KV memory base address must be non-zero".to_string(),
                ));
            }
            if region.byte_len == 0 {
                return Err(KvCacheTransferError::Runtime(
                    "transferable KV memory region length must be non-zero".to_string(),
                ));
            }
            if region.page_size_bytes != page_size_bytes {
                return Err(KvCacheTransferError::Runtime(format!(
                    "transferable KV region page size {} does not match {page_size_bytes}",
                    region.page_size_bytes
                )));
            }
        }
        Ok(Self {
            regions,
            page_size_bytes,
        })
    }

    pub fn regions(&self) -> &[TransferableKvCacheRegion] {
        &self.regions
    }

    pub fn page_size_bytes(&self) -> usize {
        self.page_size_bytes
    }

    pub fn prefill_layout(&self, target_base_offset: u64) -> MooncakeKvCacheLayout {
        MooncakeKvCacheLayout {
            source_base_addr: self.regions[0].base_addr,
            page_size_bytes: self.page_size_bytes,
            target_base_offset,
        }
    }

    pub fn decode_remote_layout(&self, dst_kv_indices: &[i32]) -> MooncakeRemoteKvLayout {
        MooncakeRemoteKvLayout {
            dst_kv_ptrs: self.regions.iter().map(|region| region.base_addr as u64).collect(),
            dst_kv_indices: dst_kv_indices.to_vec(),
            dst_kv_item_len: self.page_size_bytes / self.regions.len(),
        }
    }
}

pub trait MooncakeKvCacheMemoryProvider {
    fn mooncake_kv_cache_memory(&self) -> Result<TransferableKvCacheMemory, KvCacheTransferError>;
}
```

- [ ] **Step 4: Add split-layout failure test**

Add this to `crates/sglang-srt/tests/pd_transfer_plan.rs`:

```rust
#[test]
fn transferable_kv_memory_rejects_zero_base_address() {
    use sglang_srt::transfer::{TransferableKvCacheMemory, TransferableKvCacheRegion};

    let error = TransferableKvCacheMemory::new(
        vec![TransferableKvCacheRegion {
            base_addr: 0,
            byte_len: 128,
            page_size_bytes: 128,
        }],
        128,
    )
    .expect_err("zero address should be rejected");

    assert!(
        error
            .to_string()
            .contains("base address must be non-zero"),
        "{error}"
    );
}
```

- [ ] **Step 5: Run the green tests**

Run:

```bash
cargo test -p sglang-srt --test pd_transfer_plan transferable_kv_memory
```

Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/sglang-srt/src/transfer.rs crates/sglang-srt/tests/pd_transfer_plan.rs
git commit -m "feat: add transferable mooncake kv memory layout"
```

## Task 3: Linked Mooncake Host-Buffer Smoke

**Files:**
- Modify: `crates/sglang-srt/src/transfer.rs`
- Test: `crates/sglang-srt/tests/pd_config.rs`

- [ ] **Step 1: Write feature-gated host-buffer transfer smoke**

Add this to `crates/sglang-srt/tests/pd_config.rs`:

```rust
#[cfg(feature = "mooncake-link")]
#[test]
#[ignore = "requires local Mooncake libraries and TCP-capable runtime"]
fn linked_mooncake_engine_transfers_registered_host_buffers() {
    use std::ffi::c_void;
    use std::time::{Duration, Instant};

    use sglang_srt::transfer::{
        MooncakeBufferEntry, MooncakeOpcode, MooncakeTransferEngineConfig,
        MooncakeTransferRequest, MooncakeTransferStatusCode, LinkedMooncakeTransferEngine,
    };

    let config = MooncakeTransferEngineConfig {
        hostname: "127.0.0.1".to_string(),
        gpu_id: 0,
        metadata_server: "P2PHANDSHAKE".to_string(),
        protocol: "tcp".to_string(),
        device_name: String::new(),
        rpc_port: Some(41011),
    };
    let engine = LinkedMooncakeTransferEngine::new(&config).expect("engine should initialize");
    let mut source = vec![1_u8, 2, 3, 4, 5, 6, 7, 8];
    let mut target = vec![0_u8; source.len()];
    let mut buffers = vec![
        MooncakeBufferEntry {
            addr: source.as_mut_ptr().cast::<c_void>(),
            length: source.len(),
        },
        MooncakeBufferEntry {
            addr: target.as_mut_ptr().cast::<c_void>(),
            length: target.len(),
        },
    ];
    engine
        .register_memory_batch(&mut buffers, "cpu:0")
        .expect("host buffers should register");

    let target_id = engine
        .open_segment(&config.session_id())
        .expect("local segment should open");
    let mut requests = vec![MooncakeTransferRequest {
        opcode: MooncakeOpcode::Write as i32,
        source: source.as_mut_ptr().cast::<c_void>(),
        target_id,
        target_offset: target.as_mut_ptr() as u64,
        length: source.len() as u64,
    }];
    let batch_id = engine
        .submit_transfer(&mut requests)
        .expect("transfer should submit");
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let status = engine
            .transfer_status(batch_id, 0)
            .expect("status should query");
        if status.status == MooncakeTransferStatusCode::Completed as i32 {
            break;
        }
        assert!(Instant::now() < deadline, "Mooncake transfer timed out");
        std::thread::sleep(Duration::from_millis(10));
    }
    engine.free_batch(batch_id).expect("batch should free");
    let mut addrs = vec![source.as_mut_ptr().cast::<c_void>(), target.as_mut_ptr().cast::<c_void>()];
    engine
        .unregister_memory_batch(&mut addrs)
        .expect("buffers should unregister");
    assert_eq!(target, source);
}
```

- [ ] **Step 2: Run the ignored linked test explicitly**

Run:

```bash
cargo test -p sglang-srt --features mooncake-link --test pd_config linked_mooncake_engine_transfers_registered_host_buffers -- --ignored
```

Expected before implementation: compile FAIL if the ABI or helper methods are missing. Expected after implementation: PASS on a host with Mooncake libraries available. If Mooncake libraries are missing, record the linker error in the implementation notes and continue with default tests.

- [ ] **Step 3: Implement missing linked helpers**

In `crates/sglang-srt/src/transfer.rs`, make these methods public if not already public:

```rust
impl LinkedMooncakeTransferEngine {
    pub fn register_memory_batch(
        &self,
        buffers: &mut [MooncakeBufferEntry],
        location: &str,
    ) -> Result<(), MooncakeError> {
        let location_c = CString::new(location)?;
        let code = unsafe {
            registerLocalMemoryBatch(
                self.handle,
                buffers.as_mut_ptr(),
                buffers.len(),
                location_c.as_ptr(),
            )
        };
        if code != 0 {
            return Err(MooncakeError::RegisterMemoryFailed(code));
        }
        Ok(())
    }
}
```

Also add matching `SharedLinkedMooncakeTransferEngine` delegators:

```rust
impl SharedLinkedMooncakeTransferEngine {
    pub fn register_memory_batch(
        &self,
        buffers: &mut [MooncakeBufferEntry],
        location: &str,
    ) -> Result<(), MooncakeError> {
        self.inner
            .lock()
            .expect("linked Mooncake engine lock should be held")
            .register_memory_batch(buffers, location)
    }

    pub fn session_id(&self, config: &MooncakeTransferEngineConfig) -> String {
        config.session_id()
    }
}
```

- [ ] **Step 4: Run default and linked checks**

Run:

```bash
cargo test -p sglang-srt --test pd_config linked_mooncake_engine_constructor_is_available_under_feature
cargo test -p sglang-srt --features mooncake-link --test pd_config linked_mooncake_engine_constructor_is_available_under_feature
```

Expected: default command PASS or filters out feature-gated test cleanly; linked command PASS when Mooncake libraries link.

- [ ] **Step 5: Commit**

```bash
git add crates/sglang-srt/src/transfer.rs crates/sglang-srt/tests/pd_config.rs
git commit -m "feat: smoke test linked mooncake host transfer"
```

## Task 4: GLM Runtime Transferable KV Memory

**Files:**
- Modify: `crates/sglang-srt/src/glm_runtime.rs`
- Test: `crates/sglang-srt/tests/glm_runtime.rs`

- [ ] **Step 1: Write the failing GLM transferable memory test**

Add this to `crates/sglang-srt/tests/glm_runtime.rs`:

```rust
#[test]
fn glm_cached_forward_model_exposes_nonzero_mooncake_kv_memory_after_prefill() {
    use sglang_srt::model_executor::{ForwardModel, ModelRunner};
    use sglang_srt::transfer::MooncakeKvCacheMemoryProvider;

    let model_dir = temp_model_dir("glm-runtime-mooncake-kv-memory");
    std::fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    write_glm_moe_dsa_attention_output_fixture(&model_dir);
    let artifacts = LocalModelArtifacts::from_model_path(&model_dir).expect("artifacts load");
    let runtime = GlmMoeDsaRuntime::from_local_model_artifacts(&artifacts)
        .expect("runtime should build")
        .load_tensor_parallel_shards(2)
        .expect("TP shards should load")
        .decode_f32_tensor_parallel_shards()
        .expect("F32 shards should decode");
    let mut runner = ModelRunner::new(GlmMoeDsaF32CachedForwardModel::new(runtime));
    let batch = single_request_prefill_worker_batch(vec![0], vec![CachePageId::from(0)]);

    runner
        .model_mut()
        .forward(&batch)
        .expect("prefill should populate KV");
    let memory = runner
        .model()
        .mooncake_kv_cache_memory()
        .expect("GLM model should expose KV memory");

    assert!(!memory.regions().is_empty());
    assert!(memory.regions()[0].base_addr > 0);
    assert!(memory.regions()[0].byte_len >= memory.page_size_bytes());
    assert!(memory.page_size_bytes() > 0);

    std::fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}
```

- [ ] **Step 2: Add the test helper**

Add this helper near the other GLM runtime test builders in `crates/sglang-srt/tests/glm_runtime.rs`:

```rust
fn single_request_prefill_worker_batch(
    input_ids: Vec<u32>,
    out_cache_pages: Vec<CachePageId>,
) -> ModelWorkerBatch {
    ModelWorkerBatch::new(
        sglang_srt::scheduler::ForwardMode::Prefill,
        vec![RequestId::from("glm-mooncake-memory")],
        input_ids,
        vec![0],
        vec![0],
        vec![out_cache_pages.len()],
        vec![0],
        vec![out_cache_pages.len()],
        out_cache_pages,
        vec![None],
        vec![0],
    )
    .expect("worker batch should build")
}
```

- [ ] **Step 3: Run the red test**

Run:

```bash
cargo test -p sglang-srt --test glm_runtime glm_cached_forward_model_exposes_nonzero_mooncake_kv_memory_after_prefill
```

Expected: FAIL because GLM does not implement `MooncakeKvCacheMemoryProvider` and may not expose a contiguous backing buffer yet.

- [ ] **Step 4: Implement GLM transferable memory**

In `crates/sglang-srt/src/glm_runtime.rs`, add a contiguous transfer backing store to `GlmMoeDsaF32CachedForwardModel`:

```rust
#[derive(Clone, Debug, PartialEq)]
struct GlmMoeDsaF32TransferPageStore {
    page_size_bytes: usize,
    pages: Vec<u8>,
}
```

Initialize it in `GlmMoeDsaF32CachedForwardModel::new` with a page size derived from `runtime.kv_cache_layout().token_size_bytes(KvCacheDtype::Bfloat16).unwrap_or(1)` and an empty `Vec<u8>`. When `export_kv_cache_pages` sees populated pages, resize the vector to `(max_page_index + 1) * page_size_bytes`, then serialize each page's layer projections into the page's byte range in stable layer/page order. Expose that backing store through:

```rust
impl MooncakeKvCacheMemoryProvider for GlmMoeDsaF32CachedForwardModel {
    fn mooncake_kv_cache_memory(&self) -> Result<TransferableKvCacheMemory, KvCacheTransferError> {
        TransferableKvCacheMemory::new(
            vec![TransferableKvCacheRegion {
                base_addr: self.transfer_pages.pages.as_ptr() as usize,
                byte_len: self.transfer_pages.pages.len(),
                page_size_bytes: self.transfer_pages.page_size_bytes,
            }],
            self.transfer_pages.page_size_bytes,
        )
    }
}
```

Use `as_mut_ptr()` instead of `as_ptr()` for decode-side layouts when decode memory is registered for writes.

- [ ] **Step 5: Run GLM targeted tests**

Run:

```bash
cargo test -p sglang-srt --test glm_runtime glm_cached_forward_model_exposes_nonzero_mooncake_kv_memory_after_prefill
cargo test -p sglang-srt --test glm_runtime glm_moe_dsa_transfer_worker_forwards_kv_page_snapshots_to_inner_model_runner
```

Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/sglang-srt/src/glm_runtime.rs crates/sglang-srt/tests/glm_runtime.rs
git commit -m "feat: expose glm kv memory for mooncake transfer"
```

## Task 5: Server Builders Use Live KV Memory

**Files:**
- Modify: `crates/sglang-srt/src/server.rs`
- Modify: `crates/sglang-srt/src/transfer.rs`
- Test: `crates/sglang-srt/tests/server_bootstrap.rs`

- [ ] **Step 1: Write unsupported-runtime startup failure test**

Add this to `crates/sglang-srt/tests/server_bootstrap.rs`:

```rust
#[test]
fn mooncake_decode_builder_rejects_runtime_without_transferable_kv_memory() {
    let args = ServerArgs::parse_from([
        "serve",
        "--model-path",
        "dummy",
        "--host",
        "127.0.0.1",
        "--port",
        "30002",
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--kv-cache-dtype",
        "bfloat16",
        "--kv-cache-num-layers",
        "1",
        "--kv-cache-kv-heads",
        "1",
        "--kv-cache-head-dim",
        "8",
    ])
    .expect("args should parse");
    let pd_config = PdConfig::from_server_args(&args).expect("pd config should parse");

    let error = sglang_srt::server::try_build_launch_mooncake_decode_http_router_service_for_test(
        &args,
        &pd_config,
    )
    .expect_err("dummy Space model should not expose Mooncake KV memory");

    assert!(
        error
            .to_string()
            .contains("does not expose transferable Mooncake KV memory"),
        "{error}"
    );
}
```

- [ ] **Step 2: Run the red test**

Run:

```bash
cargo test -p sglang-srt --test server_bootstrap mooncake_decode_builder_rejects_runtime_without_transferable_kv_memory
```

Expected: FAIL because the test-only builder and explicit error do not exist.

- [ ] **Step 3: Add the server error variant**

In `crates/sglang-srt/src/server.rs`, add:

```rust
UnsupportedMooncakeKvMemory {
    model_path: String,
    model_type: Option<String>,
},
```

to `ServerLaunchError`, `PartialEq`, and `Display` with this message:

```rust
"model {model_path} type {} does not expose transferable Mooncake KV memory"
```

- [ ] **Step 4: Build Mooncake layouts from the loaded model**

Replace zero-address layout calls in linked prefill/decode builders with helper functions:

```rust
fn mooncake_kv_memory_from_bootstrap_model(
    model: &BootstrapForwardModel,
) -> Result<TransferableKvCacheMemory, ServerLaunchError> {
    match model {
        BootstrapForwardModel::GlmMoeDsa(model) => model
            .mooncake_kv_cache_memory()
            .map_err(|error| ServerLaunchError::KvCacheTransfer(error.to_string())),
        BootstrapForwardModel::UnsupportedLocalModelRuntime { model_path, model_type } => {
            Err(ServerLaunchError::UnsupportedMooncakeKvMemory {
                model_path: model_path.display().to_string(),
                model_type: model_type.clone(),
            })
        }
        _ => Err(ServerLaunchError::UnsupportedMooncakeKvMemory {
            model_path: "<bootstrap>".to_string(),
            model_type: None,
        }),
    }
}
```

Add `ServerLaunchError::KvCacheTransfer(String)` to `Display` and `PartialEq` and use it for layout-provider errors.

- [ ] **Step 5: Add test-only builder wrapper**

Expose under `#[cfg(test)]` or `#[cfg(any(test, feature = "test-utils"))]`:

```rust
pub fn try_build_launch_mooncake_decode_http_router_service_for_test(
    args: &ServerArgs,
    pd_config: &PdConfig,
) -> Result<
    BootstrapPdHttpRouterService<
        MooncakeKvCacheTransferExecutor<UnlinkedMooncakeTransferEngine>,
        MooncakeDecodeBootstrapPublisher,
    >,
    ServerLaunchError,
> {
    try_build_launch_mooncake_decode_http_router_service(args, pd_config)
}
```

For non-`mooncake-link` builds, keep the unlinked service type and explicit unlinked error behavior.

- [ ] **Step 6: Run targeted server tests**

Run:

```bash
cargo test -p sglang-srt --test server_bootstrap mooncake_decode_builder_rejects_runtime_without_transferable_kv_memory
cargo test -p sglang-srt --test server_bootstrap mooncake_launch_decode_requires_kv_cache_layout
```

Expected: both PASS.

- [ ] **Step 7: Commit**

```bash
git add crates/sglang-srt/src/server.rs crates/sglang-srt/src/transfer.rs crates/sglang-srt/tests/server_bootstrap.rs
git commit -m "feat: require live kv memory for mooncake workers"
```

## Task 6: Decode Publisher Uses Live Session And Destination Layout

**Files:**
- Modify: `crates/sglang-srt/src/server.rs`
- Modify: `crates/sglang-srt/src/pd_bootstrap.rs`
- Test: `crates/sglang-srt/tests/pd_bootstrap_server.rs`

- [ ] **Step 1: Write failing publisher test**

Add this to `crates/sglang-srt/tests/pd_bootstrap_server.rs`:

```rust
#[test]
fn decode_bootstrap_publisher_uses_live_session_id_and_nonzero_kv_layout() {
    let publisher = MooncakeDecodeBootstrapPublisher::new(
        "127.0.0.1",
        41009,
        "127.0.0.1:41011",
    )
    .with_kv_cache_layout(MooncakeKvCacheLayout {
        source_base_addr: 0x7000,
        page_size_bytes: 256,
        target_base_offset: 0,
    });
    let registration = publisher
        .kv_args_registration_for_test()
        .expect("KVArgs registration should exist");

    assert_eq!(registration.endpoint, "127.0.0.1");
    assert_eq!(registration.dst_port, 41009);
    assert_eq!(registration.mooncake_session_id, "127.0.0.1:41011");
    assert_eq!(registration.dst_kv_ptrs, vec![0x7000]);
    assert_eq!(registration.dst_kv_item_len, 256);
}
```

- [ ] **Step 2: Run the red test**

Run:

```bash
cargo test -p sglang-srt --test pd_bootstrap_server decode_bootstrap_publisher_uses_live_session_id_and_nonzero_kv_layout
```

Expected: FAIL because `kv_args_registration_for_test` does not exist.

- [ ] **Step 3: Add the test accessor and zero-address guard**

In `crates/sglang-srt/src/pd_bootstrap.rs`:

```rust
impl MooncakeDecodeBootstrapPublisher {
    #[cfg(test)]
    pub fn kv_args_registration_for_test(&self) -> Option<MooncakeDecodeKvArgsRegistration> {
        self.kv_args_registration()
    }
}
```

In `kv_args_registration`, reject zero `source_base_addr` by returning `None` only when no layout exists, and make `publish_decode_bootstrap_metadata` return an error when layout exists with address zero:

```rust
if layout.source_base_addr == 0 {
    return None;
}
```

Then make the server builder fail before constructing the publisher if live memory returns a zero address.

- [ ] **Step 4: Run bootstrap tests**

Run:

```bash
cargo test -p sglang-srt --test pd_bootstrap_server decode_bootstrap_publisher_uses_live_session_id_and_nonzero_kv_layout
cargo test -p sglang-srt --test pd_bootstrap_server
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/sglang-srt/src/server.rs crates/sglang-srt/src/pd_bootstrap.rs crates/sglang-srt/tests/pd_bootstrap_server.rs
git commit -m "feat: publish live mooncake decode kv layout"
```

## Task 7: Router E2E Linked Success Path

**Files:**
- Modify: `crates/sglang-router/tests/proxy/real_srt_pd.rs`
- Modify: `scripts/run_glm5_pd_gpu.sh`

- [ ] **Step 1: Keep current unlinked failure test**

Rename the existing `router_pd_chat_reaches_real_rust_srt_mooncake_workers` test to:

```rust
#[cfg(not(feature = "mooncake-link"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn router_pd_chat_reaches_real_rust_srt_mooncake_workers_and_reports_unlinked_runtime() {
    let prefill_addr = unused_local_addr();
    let bootstrap_addr = unused_local_addr();
    let prefill_zmq_addr = unused_local_addr();
    let decode_addr = unused_local_addr();
    let (prefill_server, prefill_shutdown_tx) =
        spawn_prefill_worker(prefill_addr, bootstrap_addr, prefill_zmq_addr, false).await;
    let (decode_server, decode_shutdown_tx) =
        spawn_decode_worker(decode_addr, false).await;
    wait_for_health(prefill_addr).await;
    wait_for_health(decode_addr).await;

    let app = build_router(build_ctx(prefill_addr, decode_addr).await);
    let response = app
        .oneshot(chat_request("tiny", "hi", 1))
        .await
        .expect("router should respond");
    assert_eq!(
        response.status(),
        StatusCode::INTERNAL_SERVER_ERROR,
        "the default build should reach the unlinked Mooncake runtime"
    );
    let body = response
        .into_body()
        .collect()
        .await
        .expect("router response body should collect")
        .to_bytes();
    let body = std::str::from_utf8(&body).expect("router body should be UTF-8");
    assert!(
        body.contains(
            "mooncake transfer engine requires building sglang-srt with the mooncake-link feature"
        ),
        "router must have reached the real Rust SRT decode transfer runtime; body={body}"
    );

    prefill_shutdown_tx.send(()).expect("prefill should still run");
    decode_shutdown_tx.send(()).expect("decode should still run");
    prefill_server.await.expect("prefill joins").expect("prefill stops");
    decode_server.await.expect("decode joins").expect("decode stops");
}
```

- [ ] **Step 2: Add linked success e2e test**

Add a feature-gated test in `crates/sglang-router/tests/proxy/real_srt_pd.rs`:

```rust
#[cfg(feature = "mooncake-link")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires linked Mooncake runtime and GLM fixture-compatible KV memory"]
async fn router_pd_chat_completes_with_real_rust_srt_mooncake_workers() {
    let (prefill_addr, bootstrap_addr, prefill_zmq_addr, decode_addr) = test_addrs();
    let (prefill_server, prefill_shutdown_tx) =
        spawn_prefill_worker(prefill_addr, bootstrap_addr, prefill_zmq_addr, true).await;
    let (decode_server, decode_shutdown_tx) =
        spawn_decode_worker(decode_addr, true).await;
    wait_for_health(prefill_addr).await;
    wait_for_health(decode_addr).await;

    let app = build_router(build_ctx(prefill_addr, decode_addr).await);
    let response = app
        .oneshot(chat_request("tiny", "hi", 1))
        .await
        .expect("router should respond");
    assert_eq!(response.status(), StatusCode::OK);
    let body = response
        .into_body()
        .collect()
        .await
        .expect("body should collect")
        .to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&body).expect("body should be JSON");
    assert_eq!(body["model"], "tiny");
    assert!(body["choices"][0]["message"]["content"].is_string());

    prefill_shutdown_tx.send(()).expect("prefill should still run");
    decode_shutdown_tx.send(()).expect("decode should still run");
    prefill_server.await.expect("prefill joins").expect("prefill stops");
    decode_server.await.expect("decode joins").expect("decode stops");
}
```

Extract helpers from the existing test body:

```rust
fn test_addrs() -> (SocketAddr, SocketAddr, SocketAddr, SocketAddr) {
    (
        unused_local_addr(),
        unused_local_addr(),
        unused_local_addr(),
        unused_local_addr(),
    )
}
```

- [ ] **Step 3: Run unlinked router e2e**

Run:

```bash
cargo test -p sglang-router --test proxy real_srt_pd -- --nocapture
```

Expected: PASS with the unlinked runtime error assertion.

- [ ] **Step 4: Run linked router e2e when Mooncake is available**

Run:

```bash
cargo test -p sglang-router --features sglang-srt/mooncake-link --test proxy router_pd_chat_completes_with_real_rust_srt_mooncake_workers -- --ignored --nocapture
```

Expected with Mooncake available: PASS. Expected without Mooncake available: linker or runtime environment failure recorded in final notes, while default e2e remains PASS.

- [ ] **Step 5: Update the GPU smoke script**

In `scripts/run_glm5_pd_gpu.sh`, update build section:

```bash
MOONCAKE_LINK="${MOONCAKE_LINK:-0}"
if [[ "$BUILD" == "1" ]]; then
    if [[ "$MOONCAKE_LINK" == "1" ]]; then
        cargo build --release --features sglang-srt/mooncake-link --bin sglang-rs --bin sgl-router
    else
        cargo build --release --bin sglang-rs --bin sgl-router
    fi
fi
```

Add decode worker args when `MOONCAKE_RPC_PORT` is set:

```bash
[[ -n "${MOONCAKE_RPC_PORT:-}" ]] && decode_args+=(--disaggregation-mooncake-rpc-port "$MOONCAKE_RPC_PORT")
```

- [ ] **Step 6: Commit**

```bash
git add crates/sglang-router/tests/proxy/real_srt_pd.rs scripts/run_glm5_pd_gpu.sh
git commit -m "test: add linked mooncake pd router smoke"
```

## Task 8: Full Verification And Push

**Files:**
- No code files unless previous verification reveals a defect.

- [ ] **Step 1: Run formatting**

Run:

```bash
cargo fmt --all --check
```

Expected: PASS. If it fails, run `cargo fmt --all`, then rerun the check.

- [ ] **Step 2: Run default test suite**

Run:

```bash
cargo test --workspace
```

Expected: PASS without Mooncake installed.

- [ ] **Step 3: Run linked build check when Mooncake libraries exist**

Run:

```bash
cargo test -p sglang-srt --features mooncake-link --test pd_config linked_mooncake_engine_constructor_is_available_under_feature
```

Expected: PASS when `MOONCAKE_HOME` or `MOONCAKE_BUILD_DIR` points to a valid Mooncake build. If it fails due to missing external libraries, record exact linker output and keep the goal active.

- [ ] **Step 4: Run linked host-buffer smoke when Mooncake runtime exists**

Run:

```bash
cargo test -p sglang-srt --features mooncake-link --test pd_config linked_mooncake_engine_transfers_registered_host_buffers -- --ignored --nocapture
```

Expected: PASS on a Mooncake-capable host. If it fails due to absent runtime/hardware, record exact failure and do not mark the full goal complete.

- [ ] **Step 5: Run manual GLM PD smoke when checkpoint is present**

Run:

```bash
MOONCAKE_LINK=1 SMOKE_CHAT=1 MODEL_PATH=/GLM-5-0212-FP8 ./scripts/run_glm5_pd_gpu.sh
```

Expected: prefill worker, decode worker, and router start; `/healthz`, `/readyz`, `/v1/models`, and one chat completion succeed.

- [ ] **Step 6: Commit any verification fixes**

```bash
git status --short
git add crates/sglang-srt/src/cli.rs crates/sglang-srt/src/transfer.rs crates/sglang-srt/src/glm_runtime.rs crates/sglang-srt/src/server.rs crates/sglang-srt/src/pd_bootstrap.rs crates/sglang-srt/tests/pd_config.rs crates/sglang-srt/tests/pd_transfer_plan.rs crates/sglang-srt/tests/glm_runtime.rs crates/sglang-srt/tests/server_bootstrap.rs crates/sglang-srt/tests/pd_bootstrap_server.rs crates/sglang-router/tests/proxy/real_srt_pd.rs scripts/run_glm5_pd_gpu.sh
git commit -m "fix: stabilize mooncake linked pd runtime"
```

Only run this step if verification required code changes.

- [ ] **Step 7: Push completed feature**

Run:

```bash
git push origin main
```

Expected: push succeeds. If pushing `main` is rejected, create a branch with prefix `codex/` and push that branch.

## Self-Review Notes

- Spec coverage: Tasks 1-3 cover linked Mooncake ABI, session identity, memory registration, and transfer status. Tasks 4-6 cover live model KV memory and decode bootstrap publication. Task 7 covers gateway-compatible router e2e. Task 8 covers default and linked verification plus push.
- Placeholder scan: The plan uses no unresolved placeholder tokens. Commands that depend on local Mooncake availability have explicit expected outcomes for both available and unavailable external runtime states.
- Type consistency: The plan consistently uses `TransferableKvCacheMemory`, `TransferableKvCacheRegion`, `MooncakeKvCacheMemoryProvider`, `MooncakeTransferEngineConfig::session_id`, `MooncakeDecodeBootstrapPublisher`, and existing `MooncakeKvCacheTransferExecutor` types.
