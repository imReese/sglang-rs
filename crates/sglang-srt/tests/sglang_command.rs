use std::process::Command;

#[test]
fn sglang_binary_accepts_upstream_serve_command_shape() {
    let sglang = std::env::var("CARGO_BIN_EXE_sglang").expect("sglang binary should be built");
    let output = Command::new(sglang)
        .args([
            "serve",
            "--model-path",
            "meta-llama/Llama-3.1-8B-Instruct",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "--tp-size",
            "1",
            "--dp-size",
            "8",
        ])
        .output()
        .expect("sglang binary should run");

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf8");
    assert!(stdout.contains("model_path=meta-llama/Llama-3.1-8B-Instruct"));
    assert!(stdout.contains("host=0.0.0.0"));
    assert!(stdout.contains("port=8080"));
    assert!(stdout.contains("tp_size=1"));
    assert!(stdout.contains("dp_size=8"));
    assert!(stdout.contains("grpc_mode=false"));
}
