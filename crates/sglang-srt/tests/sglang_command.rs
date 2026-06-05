use std::io::{BufRead, BufReader};
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
