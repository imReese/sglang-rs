#[test]
fn common_runtime_does_not_depend_on_cuda_storage() {
    for source in [
        include_str!("../src/model_runtime.rs"),
        include_str!("../src/runtime_kv_cache.rs"),
    ] {
        assert!(!source.contains("CudaKvStorage"));
        assert!(!source.contains("allocate_cuda_kv_cache"));
        assert!(!source.contains("cuda_kv_cache("));
    }
}

#[test]
fn server_reuses_transfer_factory_without_constructing_mooncake_infrastructure() {
    let source = include_str!("../src/server.rs");

    for forbidden in [
        "SharedLinkedMooncakeTransferEngine::new",
        "RegisteredMooncakeKvCacheMemory::register",
        "MooncakeSessionTargetResolver::new",
        "UnlinkedMooncakeTransferEngine",
        "build_mooncake_transfer_backend(",
        "ProductionMooncakeTransferBackend",
    ] {
        assert!(!source.contains(forbidden), "server contains {forbidden}");
    }
    assert!(source.contains("TransferBackendFactory::build("));
    assert!(source.contains("ProductionPdRuntimeBundle"));
}

#[test]
fn transfer_and_artifact_primitives_do_not_rederive_model_specific_layouts() {
    let transfer = include_str!("../src/transfer.rs");
    let artifacts = include_str!("../src/model_artifacts.rs");

    assert!(!transfer.contains("HfModelConfig"));
    assert!(!transfer.contains("kv_cache_num_layers"));
    assert!(!transfer.contains("kv_cache_kv_heads"));
    assert!(!transfer.contains("kv_cache_head_dim"));
    for model_name in ["DeepSeek", "GlmMoe", "GLM-DSA", "Qwen"] {
        assert!(
            !artifacts.contains(model_name),
            "generic model artifacts contain {model_name}"
        );
    }
    for adapter_owned_field in [
        "first_k_dense_replace",
        "moe_layer_freq",
        "n_routed_experts",
        "qk_nope_head_dim",
        "linear_conv_kernel_dim",
        "layer_types",
    ] {
        assert!(
            !artifacts.contains(adapter_owned_field),
            "generic model artifacts contain adapter config field {adapter_owned_field}"
        );
    }
}

#[test]
fn production_launch_has_no_fake_transfer_backend() {
    let transfer = include_str!("../src/transfer.rs");
    let server = include_str!("../src/server.rs");

    assert!(!transfer.contains("\"fake\" =>"));
    assert!(!server.contains("TransferBackend::Fake"));
}

#[test]
fn backend_execution_bundle_prevents_runtime_kv_type_erasure() {
    let runtime = include_str!("../src/model_runtime.rs");
    let runtime_kv = include_str!("../src/runtime_kv_cache.rs");
    let server = include_str!("../src/server.rs");
    let runner = include_str!("../src/model_executor.rs");

    for source in [runtime, runtime_kv] {
        assert!(!source.contains("std::any::Any"));
        assert!(!source.contains("downcast"));
        assert!(!source.contains("as_any"));
    }
    assert!(runtime.contains("BackendExecutionBundle<E, K>"));
    assert!(!server.contains("take_runtime_kv_cache"));
    assert!(!runner.contains("install_runtime_kv_cache"));
}

#[test]
fn model_adapters_do_not_select_runtime_backend_providers() {
    for adapter in [
        include_str!("../src/models/qwen.rs"),
        include_str!("../src/models/qwen3_5.rs"),
        include_str!("../src/models/kimi_linear.rs"),
        include_str!("../src/models/deepseek.rs"),
        include_str!("../src/models/glm.rs"),
    ] {
        assert!(!adapter.contains("BackendProviderRegistry"));
        assert!(!adapter.contains("CudaBackend"));
        assert!(!adapter.contains("RuntimeBackend::"));
    }
}
