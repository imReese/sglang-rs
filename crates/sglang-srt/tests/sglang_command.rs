use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;

#[test]
fn sglang_binary_accepts_upstream_serve_command_shape() {
    let sglang = std::env::var("CARGO_BIN_EXE_sglang").expect("sglang binary should be built");
    let mut child = Command::new(sglang)
        .args([
            "serve",
            "--model-path",
            "dummy",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--tp-size",
            "1",
            "--dp-size",
            "8",
        ])
        .stdout(Stdio::piped())
        .spawn()
        .expect("sglang binary should start");

    let stdout = child.stdout.take().expect("child stdout should be piped");
    let mut reader = BufReader::new(stdout);
    let mut first_line = String::new();
    reader
        .read_line(&mut first_line)
        .expect("server should print startup line");

    assert!(first_line.contains("model_path=dummy"));
    assert!(first_line.contains("host=127.0.0.1"));
    assert!(first_line.contains("port=0"));
    assert!(first_line.contains("tp_size=1"));
    assert!(first_line.contains("dp_size=8"));
    assert!(first_line.contains("grpc_mode=false"));
    assert!(first_line.contains("serve http"));

    child.kill().expect("server process should stop");
    child.wait().expect("server process should be reaped");
}

#[test]
fn cpu_reference_smoke_script_runs_real_centralized_generation() {
    let script = workspace_root().join("scripts/run_cpu_reference_smoke.sh");

    let content =
        std::fs::read_to_string(&script).expect("CPU reference smoke script should exist");
    assert!(content.contains("SglangEmbeddingLmForCausalLM"));
    assert!(content.contains("sglang_embedding_lm"));
    assert!(content.contains("/v1/completions"));
    assert!(content.contains("text != \"world\""));
    assert!(!content.contains("/v1/chat/completions"));
    assert!(content.contains("--device cpu"));
    assert!(!content.contains("--disaggregation-mode"));
    assert!(!content.contains("--disaggregation-transfer-backend"));

    let status = Command::new("bash")
        .arg("-n")
        .arg(&script)
        .status()
        .expect("bash should syntax-check CPU reference smoke script");
    assert!(status.success(), "smoke script should pass bash -n");
}

#[test]
fn glm5_gpu_script_requires_a_model_and_has_valid_shell_syntax() {
    let script = workspace_root().join("scripts/run_glm5_pd_gpu.sh");

    let status = Command::new("bash")
        .arg("-n")
        .arg(&script)
        .status()
        .expect("bash should syntax-check GPU script");
    assert!(status.success(), "GPU script should pass bash -n");

    let output = Command::new("bash")
        .arg(&script)
        .env_remove("MODEL_PATH")
        .output()
        .expect("GPU script should report missing configuration");
    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("MODEL_PATH is required"),
        "GPU script should explain how to configure the model"
    );
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("sglang-srt crate should live under workspace/crates")
        .to_path_buf()
}
