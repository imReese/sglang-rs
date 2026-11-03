// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Server, gRPC/HTTP protocol, and full service bootstrap integration tests.
//!
//! These tests validate the boundary communication layers, network protocol encoding,
//! and end-to-end service launch.

mod dp_coordinator;
mod grpc_proto;
mod grpc_server;
mod http_server;
mod pd_bootstrap;
mod pd_bootstrap_server;
mod router_protocol;
mod server_bootstrap;

#[path = "../support/qwen3_nonzero.rs"]
mod qwen3_nonzero;

pub(crate) mod support {
    pub(crate) use super::qwen3_nonzero;
}
