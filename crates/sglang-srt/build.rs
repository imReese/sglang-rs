use std::env;
use std::path::PathBuf;

#[path = "build_support/mooncake_link.rs"]
mod mooncake_link;

fn main() {
    compile_proto();
    configure_mooncake_link();
}

fn compile_proto() {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|path| path.parent())
        .expect("crate lives under workspace/crates/sglang-srt");
    let proto_root = workspace_root.join("proto");
    let sglang_proto = proto_root.join("sglang/runtime/v1/sglang.proto");
    let descriptor_path = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"))
        .join("sglang_runtime_descriptor.bin");

    println!("cargo:rerun-if-changed={}", sglang_proto.display());
    tonic_prost_build::configure()
        .file_descriptor_set_path(descriptor_path)
        .compile_protos(&[sglang_proto], &[proto_root])
        .expect("sglang runtime proto should compile");
}

fn configure_mooncake_link() {
    println!("cargo:rerun-if-env-changed=MOONCAKE_HOME");
    println!("cargo:rerun-if-env-changed=MOONCAKE_BUILD_DIR");

    if env::var_os("CARGO_FEATURE_MOONCAKE_LINK").is_none() {
        return;
    }

    let plan = mooncake_link::resolve_link_plan_from_env().unwrap_or_else(|error| {
        panic!("mooncake-link feature requires built Mooncake native libraries:\n{error}")
    });

    for artifact in &plan.required_artifacts {
        println!("cargo:rerun-if-changed={}", artifact.display());
    }
    for dir in &plan.link_search_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    for lib in &plan.static_libs {
        println!("cargo:rustc-link-lib=static={lib}");
    }
    for lib in &plan.dynamic_libs {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    for lib in &plan.system_dynamic_libs {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}
