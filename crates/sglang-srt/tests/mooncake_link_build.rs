use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[path = "../build_support/mooncake_link.rs"]
mod mooncake_link;

#[test]
fn mooncake_link_exposes_env_resolver_for_build_script() {
    let resolver: fn()
        -> Result<mooncake_link::MooncakeLinkPlan, mooncake_link::MooncakeLinkError> =
        mooncake_link::resolve_link_plan_from_env;
    let _ = resolver;
}

#[test]
fn mooncake_link_resolves_cmake_build_artifacts() {
    let temp = test_temp_dir("resolve");
    let build_dir = temp.join("build");
    touch(
        &build_dir
            .join("mooncake-transfer-engine")
            .join("src")
            .join("libtransfer_engine.a"),
    );
    touch(
        &build_dir
            .join("mooncake-common")
            .join("src")
            .join("libmooncake_common.a"),
    );
    touch(
        &build_dir
            .join("mooncake-transfer-engine")
            .join("src")
            .join("common")
            .join("base")
            .join("libbase.a"),
    );
    touch(&build_dir.join("mooncake-common").join(dynamic_lib("asio")));

    let plan = mooncake_link::resolve_link_plan(Some(build_dir.clone()), None)
        .expect("Mooncake link plan should resolve from build artifacts");

    assert_eq!(plan.build_dir, build_dir);
    assert_eq!(
        plan.static_libs,
        ["transfer_engine", "mooncake_common", "base"]
    );
    assert_eq!(plan.dynamic_libs, ["asio"]);
    assert!(
        plan.link_search_dirs
            .contains(&build_dir.join("mooncake-transfer-engine").join("src"))
    );
    assert!(
        plan.link_search_dirs
            .contains(&build_dir.join("mooncake-common").join("src"))
    );
    assert!(
        plan.link_search_dirs.contains(
            &build_dir
                .join("mooncake-transfer-engine")
                .join("src")
                .join("common")
                .join("base")
        )
    );
    assert!(
        plan.link_search_dirs
            .contains(&build_dir.join("mooncake-common"))
    );
}

#[test]
fn mooncake_link_reports_missing_artifacts_with_build_hint() {
    let temp = test_temp_dir("missing");
    let build_dir = temp.join("build");
    fs::create_dir_all(&build_dir).expect("build dir should be created");

    let error = mooncake_link::resolve_link_plan(Some(build_dir.clone()), None)
        .expect_err("missing native artifacts should be reported before rustc link");

    let message = error.to_string();
    assert!(message.contains("libtransfer_engine"), "{message}");
    assert!(message.contains("MOONCAKE_BUILD_DIR"), "{message}");
    assert!(message.contains("cmake"), "{message}");
}

#[test]
fn mooncake_link_uses_explicit_home_without_scanning_default_roots() {
    let temp = test_temp_dir("home");
    let mooncake_home = temp.join("Mooncake");
    fs::create_dir_all(mooncake_home.join("build")).expect("build dir should be created");

    let error = mooncake_link::resolve_link_plan(None, Some(mooncake_home.clone()))
        .expect_err("missing explicit home build artifacts should be reported");

    let message = error.to_string();
    assert!(message.contains(&mooncake_home.join("build").display().to_string()));
    assert!(
        !message.contains("workspace/code/kvcache-ai/Mooncake"),
        "{message}"
    );
}

#[test]
fn mooncake_link_requires_explicit_discovery_configuration() {
    let error = mooncake_link::resolve_link_plan(None, None)
        .expect_err("Mooncake build discovery should not scan developer checkouts");

    let message = error.to_string();
    assert!(message.contains("MOONCAKE_BUILD_DIR"), "{message}");
    assert!(message.contains("MOONCAKE_HOME"), "{message}");
    assert!(!message.contains("reese"), "{message}");
}

fn touch(path: &Path) {
    fs::create_dir_all(path.parent().expect("test file has parent"))
        .expect("artifact parent should be created");
    fs::write(path, []).expect("artifact should be written");
}

fn test_temp_dir(name: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after Unix epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "sglang-srt-mooncake-link-{name}-{}-{nanos}",
        std::process::id()
    ));
    fs::create_dir_all(&path).expect("temp test dir should be created");
    path
}

fn dynamic_lib(name: &str) -> String {
    if cfg!(target_os = "macos") {
        format!("lib{name}.dylib")
    } else {
        format!("lib{name}.so")
    }
}
