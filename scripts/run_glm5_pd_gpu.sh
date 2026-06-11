#!/usr/bin/env bash
set -euo pipefail

# Single-host GLM-5 PD-disaggregation smoke launcher for sglang-rs.
#
# Usage:
#   MODEL_PATH=/GLM-5-0212-FP8 ./scripts/run_glm5_pd_gpu.sh
#
# Useful overrides:
#   TP_SIZE=8 DP_SIZE=1 ROUTER_PORT=8000 PREFILL_PORT=30001 DECODE_PORT=30002 \
#   BOOTSTRAP_PORT=8200 ZMQ_PORTS=7000-7007 ./scripts/run_glm5_pd_gpu.sh
#
# Notes:
# - Workers are launched in gRPC mode and routed through sgl-router's
#   sgl-model-gateway-compatible PD launch surface.
# - The current Rust GLM path can boot and expose PD/router metadata, but full
#   GLM transformer kernels are still being filled in. Keep SMOKE_CHAT=0 until
#   the forward path is complete enough for generation on your checkpoint.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-/GLM-5-0212-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-aiak_bzz2_glm_5_community_rd1}"

TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-1}"
PAGE_SIZE="${PAGE_SIZE:-64}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
TRANSFER_BACKEND="${TRANSFER_BACKEND:-mooncake}"
NUM_RESERVED_DECODE_TOKENS="${NUM_RESERVED_DECODE_TOKENS:-512}"

PREFILL_HOST="${PREFILL_HOST:-127.0.0.1}"
PREFILL_ADVERTISE_HOST="${PREFILL_ADVERTISE_HOST:-$PREFILL_HOST}"
PREFILL_PORT="${PREFILL_PORT:-30001}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8200}"
ZMQ_PORTS="${ZMQ_PORTS:-7000-7007}"

DECODE_HOST="${DECODE_HOST:-127.0.0.1}"
DECODE_ADVERTISE_HOST="${DECODE_ADVERTISE_HOST:-$DECODE_HOST}"
DECODE_PORT="${DECODE_PORT:-30002}"

ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
ROUTER_CURL_HOST="${ROUTER_CURL_HOST:-127.0.0.1}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
ROUTER_POLICY="${ROUTER_POLICY:-cache_aware}"

LOG_DIR="${LOG_DIR:-$ROOT_DIR/target/pd-logs}"
BUILD="${BUILD:-1}"
SMOKE="${SMOKE:-1}"
SMOKE_CHAT="${SMOKE_CHAT:-0}"

SGLANG_RS_BIN="${SGLANG_RS_BIN:-$ROOT_DIR/target/release/sglang-rs}"
SGL_ROUTER_BIN="${SGL_ROUTER_BIN:-$ROOT_DIR/target/release/sgl-router}"

mkdir -p "$LOG_DIR"

if [[ "$BUILD" == "1" ]]; then
    cargo build --release --bin sglang-rs --bin sgl-router
fi

pids=()
cleanup() {
    local exit_code=$?
    if ((${#pids[@]} > 0)); then
        kill "${pids[@]}" >/dev/null 2>&1 || true
        wait "${pids[@]}" >/dev/null 2>&1 || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

append_optional_srt_args() {
    local -n args_ref=$1

    [[ -n "${DIST_INIT_ADDR:-}" ]] && args_ref+=(--dist-init-addr "$DIST_INIT_ADDR")
    [[ -n "${IB_DEVICE:-}" ]] && args_ref+=(--disaggregation-ib-device "$IB_DEVICE")
    [[ -n "${TOKENIZER_PATH:-}" ]] && args_ref+=(--tokenizer-path "$TOKENIZER_PATH")
    [[ -n "${MAX_RUNNING_REQUESTS:-}" ]] && args_ref+=(--max-running-requests "$MAX_RUNNING_REQUESTS")
    [[ -n "${MAX_PREFILL_TOKENS:-}" ]] && args_ref+=(--max-prefill-tokens "$MAX_PREFILL_TOKENS")
    [[ -n "${MAX_TOTAL_TOKENS:-}" ]] && args_ref+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
    [[ -n "${MEM_FRACTION_STATIC:-}" ]] && args_ref+=(--mem-fraction-static "$MEM_FRACTION_STATIC")
    [[ -n "${CHUNKED_PREFILL_SIZE:-}" ]] && args_ref+=(--chunked-prefill-size "$CHUNKED_PREFILL_SIZE")
    [[ -n "${KV_CACHE_NUM_LAYERS:-}" ]] && args_ref+=(--kv-cache-num-layers "$KV_CACHE_NUM_LAYERS")
    [[ -n "${KV_CACHE_KV_HEADS:-}" ]] && args_ref+=(--kv-cache-kv-heads "$KV_CACHE_KV_HEADS")
    [[ -n "${KV_CACHE_HEAD_DIM:-}" ]] && args_ref+=(--kv-cache-head-dim "$KV_CACHE_HEAD_DIM")

    [[ "${ENABLE_DP_ATTENTION:-0}" == "1" ]] && args_ref+=(--enable-dp-attention)
    [[ "${ENABLE_DP_LM_HEAD:-0}" == "1" ]] && args_ref+=(--enable-dp-lm-head)
    [[ "${DISABLE_CUDA_GRAPH:-1}" == "1" ]] && args_ref+=(--disable-cuda-graph)
    [[ "${ALLOW_AUTO_TRUNCATE:-1}" == "1" ]] && args_ref+=(--allow-auto-truncate)
    [[ "${ENABLE_METRICS:-1}" == "1" ]] && args_ref+=(--enable-metrics)
    [[ "${ENABLE_CACHE_REPORT:-1}" == "1" ]] && args_ref+=(--enable-cache-report)
}

wait_for_tcp() {
    local host=$1
    local port=$2
    local name=$3
    local deadline=$((SECONDS + ${WAIT_TIMEOUT_SECS:-120}))
    until (echo >"/dev/tcp/${host}/${port}") >/dev/null 2>&1; do
        if ((SECONDS >= deadline)); then
            echo "timed out waiting for ${name} on ${host}:${port}" >&2
            return 1
        fi
        sleep 1
    done
}

common_worker_args=(
    serve
    --model-path "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --trust-remote-code
    --tp-size "$TP_SIZE"
    --dp-size "$DP_SIZE"
    --page-size "$PAGE_SIZE"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --num-reserved-decode-tokens "$NUM_RESERVED_DECODE_TOKENS"
    --grpc-mode
)
append_optional_srt_args common_worker_args

prefill_args=(
    "${common_worker_args[@]}"
    --host "$PREFILL_HOST"
    --port "$PREFILL_PORT"
    --disaggregation-mode prefill
    --disaggregation-transfer-backend "$TRANSFER_BACKEND"
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT"
    --disaggregation-zmq-ports "$ZMQ_PORTS"
)

decode_args=(
    "${common_worker_args[@]}"
    --host "$DECODE_HOST"
    --port "$DECODE_PORT"
    --disaggregation-mode decode
    --disaggregation-transfer-backend "$TRANSFER_BACKEND"
)

router_args=(
    launch
    --host "$ROUTER_HOST"
    --port "$ROUTER_PORT"
    --pd-disaggregation
    --prefill "grpc://${PREFILL_ADVERTISE_HOST}:${PREFILL_PORT}" "$BOOTSTRAP_PORT"
    --decode "grpc://${DECODE_ADVERTISE_HOST}:${DECODE_PORT}"
    --policy "$ROUTER_POLICY"
    --log-level "${ROUTER_LOG_LEVEL:-info}"
)

echo "logs: $LOG_DIR"
echo "starting prefill worker on ${PREFILL_HOST}:${PREFILL_PORT}, bootstrap ${BOOTSTRAP_PORT}, zmq ${ZMQ_PORTS}"
"$SGLANG_RS_BIN" "${prefill_args[@]}" >"$LOG_DIR/prefill.log" 2>&1 &
pids+=("$!")
wait_for_tcp "$PREFILL_HOST" "$PREFILL_PORT" prefill

echo "starting decode worker on ${DECODE_HOST}:${DECODE_PORT}"
"$SGLANG_RS_BIN" "${decode_args[@]}" >"$LOG_DIR/decode.log" 2>&1 &
pids+=("$!")
wait_for_tcp "$DECODE_HOST" "$DECODE_PORT" decode

echo "starting router on ${ROUTER_HOST}:${ROUTER_PORT}"
"$SGL_ROUTER_BIN" "${router_args[@]}" >"$LOG_DIR/router.log" 2>&1 &
pids+=("$!")
wait_for_tcp "$ROUTER_CURL_HOST" "$ROUTER_PORT" router

if [[ "$SMOKE" == "1" ]]; then
    echo "router healthz:"
    curl -fsS "http://${ROUTER_CURL_HOST}:${ROUTER_PORT}/healthz"
    echo

    echo "router readyz:"
    curl -fsS "http://${ROUTER_CURL_HOST}:${ROUTER_PORT}/readyz"
    echo

    echo "router models:"
    curl -fsS "http://${ROUTER_CURL_HOST}:${ROUTER_PORT}/v1/models"
    echo
fi

if [[ "$SMOKE_CHAT" == "1" ]]; then
    echo "router chat completion:"
    curl -fsS "http://${ROUTER_CURL_HOST}:${ROUTER_PORT}/v1/chat/completions" \
        -H 'content-type: application/json' \
        -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}"
    echo
fi

echo "PD stack is running. Press Ctrl-C to stop."
wait
