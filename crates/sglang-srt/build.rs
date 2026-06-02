use std::env;
use std::path::PathBuf;

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

    let build_dir = mooncake_build_dir();
    let transfer_engine_dir = build_dir.join("mooncake-transfer-engine").join("src");
    let base_dir = transfer_engine_dir.join("common").join("base");
    let common_src_dir = build_dir.join("mooncake-common").join("src");
    let common_dir = build_dir.join("mooncake-common");

    println!(
        "cargo:rustc-link-search=native={}",
        transfer_engine_dir.display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        common_src_dir.display()
    );
    println!("cargo:rustc-link-search=native={}", base_dir.display());
    println!("cargo:rustc-link-search=native={}", common_dir.display());
    println!("cargo:rustc-link-lib=static=transfer_engine");
    println!("cargo:rustc-link-lib=static=mooncake_common");
    println!("cargo:rustc-link-lib=static=base");
    println!("cargo:rustc-link-lib=dylib=asio");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=jsoncpp");
    println!("cargo:rustc-link-lib=dylib=curl");
    println!("cargo:rustc-link-lib=dylib=glog");
    println!("cargo:rustc-link-lib=dylib=gflags");
    println!("cargo:rustc-link-lib=dylib=ibverbs");
    println!("cargo:rustc-link-lib=dylib=numa");
    println!("cargo:rustc-link-lib=dylib=pthread");
}

fn mooncake_build_dir() -> PathBuf {
    if let Some(path) = env::var_os("MOONCAKE_BUILD_DIR") {
        return PathBuf::from(path);
    }

    let mooncake_home = env::var_os("MOONCAKE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/reese/workspace/code/kvcache-ai/Mooncake"));
    mooncake_home.join("build")
}
