#!/usr/bin/env bash
# Pinned lifecycle adapter for MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_REPO="${QWEN38_27B_UPSTREAM_REPO:-https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark.git}"
UPSTREAM_REVISION="${QWEN38_27B_UPSTREAM_REVISION:-9fb18edf8cfb3364e8aa89258e6d5ab1fe1fd11a}"
RECIPE_DIR="${QWEN38_27B_RECIPE_DIR:-${PROJECT_ROOT}/data/dspark/miaai-qwen38-27b-sglang}"
PROFILE_FILE="${QWEN38_27B_CONFIG_FILE:-${PROJECT_ROOT}/config/dspark-qwen38-27b-sglang.env}"
PROJECT_ENV_FILE="${QWEN38_27B_PROJECT_ENV_FILE:-${PROJECT_ROOT}/.env.local}"
log() { printf '[qwen38-27b] %s\n' "$*"; }
die() { printf '[qwen38-27b] ERROR: %s\n' "$*" >&2; exit 1; }
value() { local key="$1" default="${2:-}" found; found="$(sed -n "s/^${key}=//p" "${PROFILE_FILE}" 2>/dev/null | tail -1)"; printf '%s' "${!key:-${found:-$default}}"; }
load_profile() {
  [[ -f "$PROFILE_FILE" ]] || die "Missing profile: $PROFILE_FILE"
  QUANT="$(value QUANT nvfp4)"; YARN="$(value YARN 0)"; CONTEXT_LENGTH="$(value CONTEXT_LENGTH 262144)"
  MAX_CONCURRENT_REQUESTS="$(value MAX_CONCURRENT_REQUESTS 10)"; CHUNKED_PREFILL="$(value CHUNKED_PREFILL 8192)"; CPUSET="$(value CPUSET 5-9,15-19)"
  SERVED_MODEL_NAME="$(value SERVED_MODEL_NAME qwen3.8-27b-sglang)"; SERVING_HOST="$(value SERVING_HOST 127.0.0.1)"; SERVING_PORT="$(value SERVING_PORT 8888)"
  DSPARK_PROXY_HOST="$(value DSPARK_PROXY_HOST 0.0.0.0)"; DSPARK_PROXY_PORT="$(value DSPARK_PROXY_PORT 8000)"; SPECULATIVE_MODE="$(value SPECULATIVE_MODE dspark)"
}
project_python() { [[ -x "${PROJECT_ROOT}/venv/bin/python" ]] && printf '%s' "${PROJECT_ROOT}/venv/bin/python" || printf '%s' python3; }
api_key() { printf '%s' "${API_KEY:-$(sed -n 's/^API_KEY=//p' "$PROJECT_ENV_FILE" 2>/dev/null | tail -1)}"; }
bootstrap() {
  if [[ -d "$RECIPE_DIR/.git" ]]; then return; fi
  [[ ! -e "$RECIPE_DIR" ]] || die "Recipe path exists but is not a git checkout: $RECIPE_DIR"
  mkdir -p "$(dirname "$RECIPE_DIR")"; git clone --no-tags "$UPSTREAM_REPO" "$RECIPE_DIR"
  git -C "$RECIPE_DIR" fetch --no-tags origin "$UPSTREAM_REVISION"; git -C "$RECIPE_DIR" checkout --detach "$UPSTREAM_REVISION"
}
check_pin() { [[ "$(git -C "$RECIPE_DIR" rev-parse HEAD 2>/dev/null)" == "$UPSTREAM_REVISION" ]] || die "Recipe pin missing; run qwen38-27b-sglang setup"; }
configure() {
  load_profile; check_pin; [[ "$SERVING_HOST" == 127.0.0.1 ]] || die "SERVING_HOST must be 127.0.0.1"
  [[ "$SPECULATIVE_MODE" == mtp || "$SPECULATIVE_MODE" == dspark ]] || die "SPECULATIVE_MODE must be mtp or dspark"
  # Upstream hard-codes 0.0.0.0. This ignored checkout is deliberately patched
  # to retain a private raw endpoint; client access belongs to the proxy.
  sed -i.bak 's/^HOST="0\.0\.0\.0"$/HOST="127.0.0.1"/' "$RECIPE_DIR/start.sh"; rm -f "$RECIPE_DIR/start.sh.bak"
  cat > "$RECIPE_DIR/.env" <<EOF
QUANT=$QUANT
YARN=$YARN
CONTEXT_LENGTH=$CONTEXT_LENGTH
MAX_CONCURRENT_REQUESTS=$MAX_CONCURRENT_REQUESTS
CHUNKED_PREFILL=$CHUNKED_PREFILL
CPUSET=$CPUSET
EOF
  log "Configured ${SPECULATIVE_MODE}, ${CONTEXT_LENGTH} context, ${MAX_CONCURRENT_REQUESTS} requests"
}
proxy() { local action="$1" py; py="$(project_python)"; (cd "$PROJECT_ROOT" && API_KEY="$(api_key)" DSPARK_PROXY_HOST="$DSPARK_PROXY_HOST" DSPARK_PROXY_PORT="$DSPARK_PROXY_PORT" DSPARK_PROXY_UPSTREAM_URL="http://127.0.0.1:$SERVING_PORT" "$py" -m ml.cli dspark-proxy "$action"); }
start() {
  configure; [[ -n "$(api_key)" ]] || die "API_KEY is missing from $PROJECT_ENV_FILE"
  if [[ "$SPECULATIVE_MODE" == dspark ]]; then (cd "$RECIPE_DIR" && exec ./start-dspark.sh)
  else (cd "$RECIPE_DIR" && exec ./start.sh); fi
}
start_service() { start; proxy serve; proxy smoke; log "Authenticated API: http://127.0.0.1:${DSPARK_PROXY_PORT}/v1"; }
usage() { cat <<'EOF'
Qwen3.8-27B SGLang on one DGX Spark
Actions: bootstrap configure check setup start status smoke logs stop update path help
Default uses the upstream DSpark block-7 draft at native 262K context. Set
SPECULATIVE_MODE=mtp for the in-checkpoint MTP path; use YaRN only with MTP.
EOF
}
case "${1:-help}" in
 bootstrap) bootstrap ;; configure) configure ;; check) load_profile; check_pin; [[ "$SERVING_HOST" == 127.0.0.1 ]] || die "SERVING_HOST must be loopback"; command -v docker >/dev/null || die "docker is required" ;;
 setup) bootstrap; configure; "$0" check ;; start) start_service ;;
 status) load_profile; curl -fsS --max-time 3 "http://127.0.0.1:$SERVING_PORT/v1/models" || true; proxy status ;;
 smoke) proxy smoke ;; logs) check_pin; (cd "$RECIPE_DIR" && tail -n 200 -f .sglang.log) ;;
 stop) load_profile; proxy stop || true; check_pin && (cd "$RECIPE_DIR" && ./stop.sh) ;;
 update) check_pin; git -C "$RECIPE_DIR" fetch --no-tags origin "$UPSTREAM_REVISION"; git -C "$RECIPE_DIR" checkout --detach "$UPSTREAM_REVISION" ;;
 path) printf 'recipe=%s\nprofile=%s\nupstream_revision=%s\n' "$RECIPE_DIR" "$PROFILE_FILE" "$UPSTREAM_REVISION" ;;
 help|-h|--help) usage ;; *) die "Unknown action: $1" ;;
esac
