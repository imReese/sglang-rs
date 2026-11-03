use std::env;
use std::path::PathBuf;

fn main() {
    compile_proto();
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
    let protoc = protoc_bin_vendored::protoc_bin_path()
        .expect("vendored protoc binary should support the build host");
    let mut prost_config = tonic_prost_build::Config::new();
    prost_config.protoc_executable(protoc);

    println!("cargo:rerun-if-changed={}", sglang_proto.display());
    tonic_prost_build::configure()
        .file_descriptor_set_path(descriptor_path)
        .compile_with_config(prost_config, &[sglang_proto], &[proto_root])
        .expect("sglang runtime proto should compile");
}
