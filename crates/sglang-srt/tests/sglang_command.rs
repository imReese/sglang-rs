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
fn cpu_pd_smoke_script_is_syntax_checked_and_asserts_real_generation() {
    let script = workspace_root().join("scripts/run_cpu_pd_smoke.sh");

    let content = std::fs::read_to_string(&script).expect("CPU PD smoke script should exist");
    assert!(content.contains("sglang_embedding_lm"));
    assert!(content.contains("content != \"world\""));
    assert!(content.contains("--pd-disaggregation"));
    assert!(content.contains("--disaggregation-transfer-backend"));
    assert!(content.contains("fake"));

    let status = Command::new("bash")
        .arg("-n")
        .arg(&script)
        .status()
        .expect("bash should syntax-check smoke script");
    assert!(status.success(), "smoke script should pass bash -n");
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("sglang-srt crate should live under workspace/crates")
        .to_path_buf()
}
