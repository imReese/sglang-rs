use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Mutex;

use serde_json::json;
use sglang_srt::tokenizer::{ChatTemplateInput, HfTokenizer, RuntimeTokenizer, Tokenizer};

static HF_ENV_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn hf_tokenizer_loads_tokenizer_json_from_model_directory() {
    let model_dir = temp_model_dir("hf-tokenizer-dir");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("tokenizer.json"),
        word_level_tokenizer_json(),
    )
    .expect("tokenizer.json should be written");

    let tokenizer = HfTokenizer::from_tokenizer_path(&model_dir)
        .expect("tokenizer.json should load from model directory");

    assert_eq!(tokenizer.encode("hello world"), vec![1, 2]);
    assert_eq!(
        tokenizer
            .decode(&[1, 2])
            .expect("token ids should decode through HF tokenizer"),
        "hello world"
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn hf_tokenizer_loads_direct_tokenizer_json_path() {
    let model_dir = temp_model_dir("hf-tokenizer-file");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    let tokenizer_path = model_dir.join("tokenizer.json");
    fs::write(&tokenizer_path, word_level_tokenizer_json())
        .expect("tokenizer.json should be written");

    let tokenizer = HfTokenizer::from_tokenizer_path(&tokenizer_path)
        .expect("direct tokenizer.json path should load");

    assert_eq!(tokenizer.encode("hello world"), vec![1, 2]);

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn hf_tokenizer_applies_qwen_style_chat_template() {
    let model_dir = temp_model_dir("qwen-chat-template");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");
    fs::write(
        model_dir.join("tokenizer.json"),
        word_level_tokenizer_json(),
    )
    .expect("tokenizer.json should be written");
    fs::write(
        model_dir.join("tokenizer_config.json"),
        json!({
            "chat_template": "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        })
        .to_string(),
    )
    .expect("tokenizer_config.json should be written");
    let tokenizer = HfTokenizer::from_tokenizer_path(&model_dir)
        .expect("tokenizer and chat template should load");

    let prompt = tokenizer
        .apply_chat_template(&ChatTemplateInput {
            messages: vec![
                json!({"role": "system", "content": "You are concise."}),
                json!({"role": "user", "content": "Hello"}),
            ],
            ..Default::default()
        })
        .expect("Qwen-style chat template should render");

    assert_eq!(
        prompt,
        "<|im_start|>system\nYou are concise.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    );

    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn runtime_tokenizer_loads_repo_id_from_huggingface_cache_snapshot() {
    let hub_dir = temp_model_dir("hf-tokenizer-cache-hub");
    let snapshot_dir = hub_dir
        .join("models--zai-org--GLM-5-FP8")
        .join("snapshots")
        .join("abc123");
    fs::create_dir_all(&snapshot_dir).expect("snapshot dir should be created");
    fs::create_dir_all(hub_dir.join("models--zai-org--GLM-5-FP8").join("refs"))
        .expect("refs dir should be created");
    fs::write(
        hub_dir
            .join("models--zai-org--GLM-5-FP8")
            .join("refs")
            .join("main"),
        "abc123\n",
    )
    .expect("main ref should be written");
    fs::write(
        snapshot_dir.join("tokenizer.json"),
        word_level_tokenizer_json(),
    )
    .expect("tokenizer.json should be written");

    let tokenizer = RuntimeTokenizer::from_model_or_tokenizer_path_with_hf_cache(
        "zai-org/GLM-5-FP8",
        None,
        &hub_dir,
    )
    .expect("runtime tokenizer should resolve repo id through HF cache");

    assert_eq!(tokenizer.encode("hello world"), vec![1, 2]);

    fs::remove_dir_all(hub_dir).expect("temp hub dir should be removed");
}

#[test]
fn runtime_tokenizer_rejects_local_model_without_tokenizer() {
    let model_dir = temp_model_dir("missing-tokenizer");
    fs::create_dir_all(&model_dir).expect("temp model dir should be created");

    let error = RuntimeTokenizer::from_model_or_tokenizer_path(
        model_dir.to_str().expect("temp path should be utf8"),
        None,
    )
    .expect_err("a real local model must not fall back to the byte tokenizer");

    assert!(error.to_string().contains("tokenizer.json was not found"));
    fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
}

#[test]
fn runtime_tokenizer_keeps_explicit_missing_paths_local() {
    let missing = format!("./target/missing-tokenizer-path-{}", std::process::id());
    let model_error = RuntimeTokenizer::from_model_or_tokenizer_path(&missing, None)
        .expect_err("an explicit model path must not be treated as a Hub repo id");
    assert!(
        model_error
            .to_string()
            .contains("tokenizer.json was not found")
    );

    let tokenizer_error = RuntimeTokenizer::from_model_or_tokenizer_path("dummy", Some(&missing))
        .expect_err("an explicit tokenizer path must not be treated as a Hub repo id");
    assert!(
        tokenizer_error
            .to_string()
            .contains("tokenizer.json was not found")
    );
}

#[test]
fn runtime_tokenizer_downloads_repo_id_tokenizer_when_cache_is_missing() {
    let _env_guard = HF_ENV_LOCK
        .lock()
        .expect("HF env lock should not be poisoned");
    let hf_home = temp_model_dir("hf-tokenizer-download-home");
    let endpoint = start_fake_hf_file_endpoint("tokenizer.json", word_level_tokenizer_json());
    let _hf_home = EnvVarRestore::set("HF_HOME", &hf_home);
    let _hf_hub_cache = EnvVarRestore::set("HUGGINGFACE_HUB_CACHE", hf_home.join("hub"));
    let _hf_endpoint = EnvVarRestore::set("HF_ENDPOINT", endpoint);

    let tokenizer = RuntimeTokenizer::from_model_or_tokenizer_path("zai-org/GLM-5-FP8", None)
        .expect("runtime tokenizer should download repo tokenizer metadata");

    assert_eq!(tokenizer.encode("hello world"), vec![1, 2]);

    fs::remove_dir_all(hf_home).expect("temp HF home should be removed");
}

fn word_level_tokenizer_json() -> &'static str {
    r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "Whitespace"
  },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "[UNK]": 0,
      "hello": 1,
      "world": 2
    },
    "unk_token": "[UNK]"
  }
}"#
}

struct EnvVarRestore {
    name: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvVarRestore {
    fn set(name: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
        let previous = std::env::var_os(name);
        unsafe {
            std::env::set_var(name, value);
        }
        Self { name, previous }
    }
}

impl Drop for EnvVarRestore {
    fn drop(&mut self) {
        if let Some(value) = &self.previous {
            unsafe {
                std::env::set_var(self.name, value);
            }
        } else {
            unsafe {
                std::env::remove_var(self.name);
            }
        }
    }
}

fn start_fake_hf_file_endpoint(filename: &'static str, contents: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("fake HF endpoint should bind");
    let addr = listener
        .local_addr()
        .expect("fake HF endpoint should have address");

    std::thread::spawn(move || {
        for request_id in 0..2 {
            let (mut stream, _) = listener.accept().expect("fake HF request should connect");
            let request = read_http_request(&mut stream);
            assert!(
                request.starts_with(&format!("GET /zai-org/GLM-5-FP8/resolve/main/{filename} ")),
                "unexpected fake HF request: {request:?}"
            );

            let body = if request_id == 0 {
                &contents.as_bytes()[..1]
            } else {
                contents.as_bytes()
            };
            let status = if request_id == 0 {
                "206 Partial Content"
            } else {
                "200 OK"
            };
            let content_range = if request_id == 0 {
                format!("bytes 0-0/{}", contents.len())
            } else {
                format!("bytes 0-{}/{}", contents.len() - 1, contents.len())
            };
            let response = format!(
                "HTTP/1.1 {status}\r\n\
                 x-repo-commit: abc123\r\n\
                 etag: \"{filename}\"\r\n\
                 content-range: {content_range}\r\n\
                 content-length: {}\r\n\
                 connection: close\r\n\
                 \r\n",
                body.len()
            );
            stream
                .write_all(response.as_bytes())
                .expect("fake HF response headers should write");
            stream
                .write_all(body)
                .expect("fake HF response body should write");
        }
    });

    format!("http://{addr}")
}

fn read_http_request(stream: &mut TcpStream) -> String {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 1];
    while stream
        .read(&mut buffer)
        .expect("fake HF request should read")
        == 1
    {
        request.push(buffer[0]);
        if request.ends_with(b"\r\n\r\n") {
            break;
        }
    }
    String::from_utf8(request).expect("fake HF request should be utf8")
}

fn temp_model_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("sglang-rs-{name}-{}", std::process::id()))
}
