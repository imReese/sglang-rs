use std::env;
use std::path::PathBuf;

fn main() {
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
