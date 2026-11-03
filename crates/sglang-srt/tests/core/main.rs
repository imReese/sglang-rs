// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Core domain, scheduler, cache, and model architecture tests.
//!
//! These tests validate internal abstractions, scheduling decisions, cache eviction,
//! and model representation without spinning up external servers or sockets.

mod architecture_boundaries;
mod backend_capabilities;
mod cache_allocator;
mod chunked_prefill;
mod cli_args;
mod cuda_attention;
mod cuda_backend;
mod cuda_inference;
mod cuda_worker_group;
mod decode_loop;
mod kv_block_hash;
mod kv_cache;
mod model_artifacts;
mod model_worker_batch;
mod pd_config;
mod pd_separation;
mod pd_transfer_plan;
mod radix_cache;
mod request_lifecycle;
mod scheduler_batch;
mod scheduler_cache_allocation;
mod sglang_command;
mod tokenizer_loader;
mod transfer_lifecycle;

#[path = "../support/compressed_kimi_k25.rs"]
mod compressed_kimi_k25;

pub(crate) mod support {
    pub(crate) use super::compressed_kimi_k25;
}
