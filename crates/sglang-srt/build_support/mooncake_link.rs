use std::env;
use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MooncakeLinkPlan {
    pub build_dir: PathBuf,
    pub link_search_dirs: Vec<PathBuf>,
    pub static_libs: Vec<&'static str>,
    pub dynamic_libs: Vec<&'static str>,
    pub system_dynamic_libs: Vec<&'static str>,
    pub required_artifacts: Vec<PathBuf>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MooncakeLinkError {
    message: String,
}

impl fmt::Display for MooncakeLinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for MooncakeLinkError {}

pub fn resolve_link_plan_from_env() -> Result<MooncakeLinkPlan, MooncakeLinkError> {
    resolve_link_plan(
        env::var_os("MOONCAKE_BUILD_DIR").map(PathBuf::from),
        env::var_os("MOONCAKE_HOME").map(PathBuf::from),
    )
}

pub fn resolve_link_plan(
    explicit_build_dir: Option<PathBuf>,
    explicit_home: Option<PathBuf>,
) -> Result<MooncakeLinkPlan, MooncakeLinkError> {
    let candidates = candidate_build_dirs(explicit_build_dir, explicit_home);
    if candidates.is_empty() {
        return Err(MooncakeLinkError {
            message: "Mooncake native link artifacts require an explicit MOONCAKE_BUILD_DIR or MOONCAKE_HOME; personal checkout paths are not searched"
                .to_string(),
        });
    }
    let mut missing_reports = Vec::new();

    for build_dir in candidates.iter() {
        match plan_for_build_dir(build_dir) {
            Ok(plan) => return Ok(plan),
            Err(error) => missing_reports.push(error.message),
        }
    }

    Err(MooncakeLinkError {
        message: format!(
            "Mooncake native link artifacts were not found.\n{}\nSet MOONCAKE_BUILD_DIR to a Mooncake CMake build directory containing libtransfer_engine, or build Mooncake first:\n  cd $MOONCAKE_HOME && mkdir -p build && cd build && cmake .. && cmake --build . --target transfer_engine -j\nChecked build dirs:\n  {}",
            missing_reports.join("\n"),
            candidates
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join("\n  ")
        ),
    })
}

fn candidate_build_dirs(
    explicit_build_dir: Option<PathBuf>,
    explicit_home: Option<PathBuf>,
) -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    if let Some(path) = explicit_build_dir {
        push_unique(&mut dirs, path);
        return dirs;
    }

    if let Some(home) = explicit_home {
        push_unique(&mut dirs, home.join("build"));
        return dirs;
    }

    dirs
}

fn plan_for_build_dir(build_dir: &Path) -> Result<MooncakeLinkPlan, MooncakeLinkError> {
    let transfer_engine_dir = build_dir.join("mooncake-transfer-engine").join("src");
    let common_src_dir = build_dir.join("mooncake-common").join("src");
    let base_dir = transfer_engine_dir.join("common").join("base");
    let common_dir = build_dir.join("mooncake-common");

    let required = [
        required_static_lib(&transfer_engine_dir, "transfer_engine"),
        required_static_lib(&common_src_dir, "mooncake_common"),
        required_static_lib(&base_dir, "base"),
        required_dynamic_lib(&common_dir, "asio"),
    ];
    let missing = required
        .iter()
        .filter(|path| !path.exists())
        .cloned()
        .collect::<Vec<_>>();

    if !missing.is_empty() {
        return Err(MooncakeLinkError {
            message: format!(
                "- {} is missing required Mooncake artifacts: {}",
                build_dir.display(),
                missing
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        });
    }

    Ok(MooncakeLinkPlan {
        build_dir: build_dir.to_path_buf(),
        link_search_dirs: vec![transfer_engine_dir, common_src_dir, base_dir, common_dir],
        static_libs: vec!["transfer_engine", "mooncake_common", "base"],
        dynamic_libs: vec!["asio"],
        system_dynamic_libs: system_dynamic_libs(),
        required_artifacts: required.into_iter().collect(),
    })
}

fn required_static_lib(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("lib{name}.a"))
}

fn required_dynamic_lib(dir: &Path, name: &str) -> PathBuf {
    if cfg!(target_os = "macos") {
        dir.join(format!("lib{name}.dylib"))
    } else {
        dir.join(format!("lib{name}.so"))
    }
}

fn system_dynamic_libs() -> Vec<&'static str> {
    let mut libs = vec![cxx_stdlib(), "jsoncpp", "curl", "glog", "gflags", "pthread"];
    if cfg!(target_os = "linux") {
        libs.extend(["ibverbs", "numa"]);
    }
    libs
}

fn cxx_stdlib() -> &'static str {
    if cfg!(target_os = "macos") {
        "c++"
    } else {
        "stdc++"
    }
}

fn push_unique(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.iter().any(|existing| existing == &path) {
        paths.push(path);
    }
}
