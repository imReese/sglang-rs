use std::fs;
use std::path::PathBuf;

use sglang_srt::tokenizer::{HfTokenizer, RuntimeTokenizer, Tokenizer};

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

fn temp_model_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("sglang-rs-{name}-{}", std::process::id()))
}
