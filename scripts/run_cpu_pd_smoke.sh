#!/usr/bin/env bash
set -euo pipefail

# Local CPU PD-disaggregation smoke launcher for sglang-rs.
#
# This script starts a real process stack:
#   1. Rust SRT HTTP prefill worker
#   2. Rust SRT HTTP decode worker
#   3. Rust sgl-router in PD mode
#
# It generates a tiny safetensors-backed CPU embedding LM at runtime, sends an
# OpenAI chat request through the router, verifies the model-produced token, and
# exits. Set KEEP_RUNNING=1 to leave the stack up after the smoke request.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-tiny}"
BUILD="${BUILD:-1}"
PROFILE="${PROFILE:-debug}"
KEEP_RUNNING="${KEEP_RUNNING:-0}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/target/cpu-pd-smoke-logs}"
MODEL_DIR="${MODEL_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/sglang-rs-cpu-pd-model.XXXXXX")}"

PREFILL_HOST="${PREFILL_HOST:-127.0.0.1}"
DECODE_HOST="${DECODE_HOST:-127.0.0.1}"
ROUTER_HOST="${ROUTER_HOST:-127.0.0.1}"
ROUTER_CURL_HOST="${ROUTER_CURL_HOST:-127.0.0.1}"

pick_port() {
    python3 - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

PREFILL_PORT="${PREFILL_PORT:-$(pick_port)}"
DECODE_PORT="${DECODE_PORT:-$(pick_port)}"
ROUTER_PORT="${ROUTER_PORT:-$(pick_port)}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-$(pick_port)}"
PREFILL_ENGINE_INFO_PORT="${PREFILL_ENGINE_INFO_PORT:-$(pick_port)}"
DECODE_ENGINE_INFO_PORT="${DECODE_ENGINE_INFO_PORT:-$(pick_port)}"

case "$PROFILE" in
    release)
        TARGET_DIR="$ROOT_DIR/target/release"
        ;;
    debug)
        TARGET_DIR="$ROOT_DIR/target/debug"
        ;;
    *)
        echo "PROFILE must be 'debug' or 'release', got '$PROFILE'" >&2
        exit 2
        ;;
esac

SGLANG_RS_BIN="${SGLANG_RS_BIN:-$TARGET_DIR/sglang-rs}"
SGL_ROUTER_BIN="${SGL_ROUTER_BIN:-$TARGET_DIR/sgl-router}"

mkdir -p "$LOG_DIR"

write_model_fixture() {
    python3 - "$MODEL_DIR" <<'PY'
import json
import struct
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
model_dir.mkdir(parents=True, exist_ok=True)

(model_dir / "config.json").write_text(json.dumps({
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
    ("model.embed_tokens.weight", [3, 2], [
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
    ]),
    ("lm_head.weight", [3, 2], [
        0.0, 0.0,
        0.25, 0.0,
        1.0, 0.0,
    ]),
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
}

wait_for_http() {
    local url=$1
    local name=$2
    local deadline=$((SECONDS + ${WAIT_TIMEOUT_SECS:-30}))
    until curl -fsS "$url" >/dev/null 2>&1; do
        if ((SECONDS >= deadline)); then
            echo "timed out waiting for ${name}: ${url}" >&2
            return 1
        fi
        sleep 0.2
    done
}

pids=()
cleanup() {
    local exit_code=$?
    if ((${#pids[@]} > 0)); then
        kill "${pids[@]}" >/dev/null 2>&1 || true
        wait "${pids[@]}" >/dev/null 2>&1 || true
    fi
    if [[ -z "${MODEL_DIR_KEEP:-}" && "$MODEL_DIR" == "${TMPDIR:-/tmp}"/sglang-rs-cpu-pd-model.* ]]; then
        rm -rf "$MODEL_DIR"
    fi
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

if [[ "$BUILD" == "1" ]]; then
    if [[ "$PROFILE" == "release" ]]; then
        cargo build --release --bin sglang-rs --bin sgl-router
    else
        cargo build --bin sglang-rs --bin sgl-router
    fi
fi

write_model_fixture

prefill_args=(
    serve
    --model-path "$MODEL_DIR"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$PREFILL_HOST"
    --port "$PREFILL_PORT"
    --disaggregation-mode prefill
    --disaggregation-transfer-backend fake
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT"
    --engine-info-bootstrap-port "$PREFILL_ENGINE_INFO_PORT"
    --num-reserved-decode-tokens 8
)

decode_args=(
    serve
    --model-path "$MODEL_DIR"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$DECODE_HOST"
    --port "$DECODE_PORT"
    --disaggregation-mode decode
    --disaggregation-transfer-backend fake
    --engine-info-bootstrap-port "$DECODE_ENGINE_INFO_PORT"
    --num-reserved-decode-tokens 8
)

router_args=(
    launch
    --host "$ROUTER_HOST"
    --port "$ROUTER_PORT"
    --pd-disaggregation
    --prefill "http://${PREFILL_HOST}:${PREFILL_PORT}" "$BOOTSTRAP_PORT"
    --decode "http://${DECODE_HOST}:${DECODE_PORT}"
    --served-model-name "$SERVED_MODEL_NAME"
    --tokenizer-path "$MODEL_DIR/tokenizer.json"
    --policy round_robin
    --log-level "${ROUTER_LOG_LEVEL:-info}"
)

echo "logs: $LOG_DIR"
echo "model fixture: $MODEL_DIR"

"$SGLANG_RS_BIN" "${prefill_args[@]}" >"$LOG_DIR/prefill.log" 2>&1 &
pids+=("$!")
wait_for_http "http://${PREFILL_HOST}:${PREFILL_PORT}/health" prefill

"$SGLANG_RS_BIN" "${decode_args[@]}" >"$LOG_DIR/decode.log" 2>&1 &
pids+=("$!")
wait_for_http "http://${DECODE_HOST}:${DECODE_PORT}/health" decode

"$SGL_ROUTER_BIN" "${router_args[@]}" >"$LOG_DIR/router.log" 2>&1 &
pids+=("$!")
wait_for_http "http://${ROUTER_CURL_HOST}:${ROUTER_PORT}/healthz" router
wait_for_http "http://${ROUTER_CURL_HOST}:${ROUTER_PORT}/readyz" router-ready

response="$(curl -fsS "http://${ROUTER_CURL_HOST}:${ROUTER_PORT}/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}")"

echo "$response"

python3 - "$response" <<'PY'
import json
import sys

body = json.loads(sys.argv[1])
content = body["choices"][0]["message"]["content"]
if content != "world":
    raise SystemExit(f"expected assistant content 'world', got {content!r}")
if body["choices"][0]["finish_reason"] != "stop":
    raise SystemExit(f"expected finish_reason 'stop', got {body['choices'][0]['finish_reason']!r}")
PY

echo "CPU PD smoke passed: router returned \"content\":\"world\"."

if [[ "$KEEP_RUNNING" == "1" ]]; then
    echo "PD stack is running on http://${ROUTER_CURL_HOST}:${ROUTER_PORT}. Press Ctrl-C to stop."
    wait
fi
