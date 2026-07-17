#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BUILD="${BUILD:-1}"
PROFILE="${PROFILE:-debug}"
KEEP_RUNNING="${KEEP_RUNNING:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-$(python3 - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/target/cpu-reference-smoke-logs}"
MODEL_DIR="${MODEL_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/sglang-rs-cpu-reference-model.XXXXXX")}"

case "$PROFILE" in
    release) TARGET_DIR="$ROOT_DIR/target/release" ;;
    debug) TARGET_DIR="$ROOT_DIR/target/debug" ;;
    *)
        echo "PROFILE must be 'debug' or 'release', got '$PROFILE'" >&2
        exit 2
        ;;
esac

SGLANG_RS_BIN="${SGLANG_RS_BIN:-$TARGET_DIR/sglang-rs}"
mkdir -p "$LOG_DIR"

cleanup() {
    local exit_code=$?
    if [[ -n "${server_pid:-}" ]]; then
        kill "$server_pid" >/dev/null 2>&1 || true
        wait "$server_pid" >/dev/null 2>&1 || true
    fi
    if [[ -z "${MODEL_DIR_KEEP:-}" && "$MODEL_DIR" == "${TMPDIR:-/tmp}"/sglang-rs-cpu-reference-model.* ]]; then
        rm -rf "$MODEL_DIR"
    fi
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

if [[ "$BUILD" == "1" ]]; then
    if [[ "$PROFILE" == "release" ]]; then
        cargo build --release --bin sglang-rs
    else
        cargo build --bin sglang-rs
    fi
fi

python3 - "$MODEL_DIR" <<'PY'
import json
import struct
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
model_dir.mkdir(parents=True, exist_ok=True)
(model_dir / "config.json").write_text(json.dumps({
    "architectures": ["SglangEmbeddingLmForCausalLM"],
    "model_type": "sglang_embedding_lm",
    "vocab_size": 3,
    "hidden_size": 2,
    "eos_token_id": 2,
}), encoding="utf-8")
(model_dir / "tokenizer.json").write_text(json.dumps({
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [],
    "normalizer": None,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "WordLevel",
        "vocab": {"[UNK]": 0, "hi": 1, "world": 2},
        "unk_token": "[UNK]",
    },
}), encoding="utf-8")

tensors = [
    ("model.embed_tokens.weight", [3, 2], [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
    ("lm_head.weight", [3, 2], [0.0, 0.0, 0.25, 0.0, 1.0, 0.0]),
]
payload = bytearray()
metadata = {}
for name, shape, values in tensors:
    start = len(payload)
    for value in values:
        payload.extend(struct.pack("<f", value))
    metadata[name] = {
        "dtype": "F32",
        "shape": shape,
        "data_offsets": [start, len(payload)],
    }
header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
(model_dir / "model.safetensors").write_bytes(
    struct.pack("<Q", len(header)) + header + bytes(payload)
)
PY

"$SGLANG_RS_BIN" serve \
    --model-path "$MODEL_DIR" \
    --served-model-name tiny \
    --device cpu \
    --host "$HOST" \
    --port "$PORT" \
    >"$LOG_DIR/server.log" 2>&1 &
server_pid=$!

deadline=$((SECONDS + ${WAIT_TIMEOUT_SECS:-30}))
until curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; do
    if ((SECONDS >= deadline)); then
        echo "timed out waiting for CPU reference server; see $LOG_DIR/server.log" >&2
        exit 1
    fi
    sleep 0.2
done

response="$(curl -fsS "http://${HOST}:${PORT}/v1/completions" \
    -H 'content-type: application/json' \
    -d '{"model":"tiny","prompt":"hi","max_tokens":1}')"
echo "$response"

python3 - "$response" <<'PY'
import json
import sys

body = json.loads(sys.argv[1])
text = body["choices"][0]["text"]
if text != "world":
    raise SystemExit(f"expected completion text 'world', got {text!r}")
PY

echo "CPU reference smoke passed: centralized inference returned model token 'world'."
if [[ "$KEEP_RUNNING" == "1" ]]; then
    echo "Server is running on http://${HOST}:${PORT}. Press Ctrl-C to stop."
    wait
fi
