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
    ] {
        assert!(!source.contains(forbidden), "server contains {forbidden}");
    }
    assert_eq!(
        source.matches("build_mooncake_transfer_backend(").count(),
        4
    );
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
}

#[test]
fn production_launch_has_no_fake_transfer_backend() {
    let transfer = include_str!("../src/transfer.rs");
    let server = include_str!("../src/server.rs");

    assert!(!transfer.contains("\"fake\" =>"));
    assert!(!server.contains("TransferBackend::Fake"));
}
