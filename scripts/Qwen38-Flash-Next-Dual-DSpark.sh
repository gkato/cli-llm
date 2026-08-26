#!/usr/bin/env bash
# Qwen3.8 Flash Next NVFP4 on two linked DGX Sparks (GB10 / SM121).
#
# Lifecycle adapter for MiaAI-Lab's SGLang TP2 recipe:
# https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks
# The upstream checkout, base runtime image, and model revision are pinned.
# Generated kernel patches, caches, and the ~135 GB checkpoint stay below the
# ignored data/ tree; ml-compute keeps ownership of configuration and exposure.

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_REPO_DEFAULT="https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks.git"
UPSTREAM_REVISION_DEFAULT="dccb035c559f342fe8c0f65eb427671c6cf60730"
MODEL_REVISION_DEFAULT="7b719225242aacd3dbd3f9407468c2ee9a9d2594"
BASE_IMAGE_DEFAULT="lmsysorg/sglang:qwen38flashnext@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1"
RECIPE_DIR_DEFAULT="${PROJECT_ROOT}/data/dspark/miaai-qwen38-flash-next-dual-spark"
PROFILE_FILE_DEFAULT="${PROJECT_ROOT}/config/dspark-qwen38-flash-next-nvfp4.env"
PROJECT_ENV_FILE_DEFAULT="${PROJECT_ROOT}/.env.local"

UPSTREAM_REPO="${QWEN38_DSPARK_UPSTREAM_REPO:-${UPSTREAM_REPO_DEFAULT}}"
UPSTREAM_REVISION="${QWEN38_DSPARK_UPSTREAM_REVISION:-${UPSTREAM_REVISION_DEFAULT}}"
RECIPE_DIR="${QWEN38_DSPARK_RECIPE_DIR:-${RECIPE_DIR_DEFAULT}}"
PROFILE_FILE="${QWEN38_DSPARK_CONFIG_FILE:-${PROFILE_FILE_DEFAULT}}"
PROJECT_ENV_FILE="${QWEN38_DSPARK_PROJECT_ENV_FILE:-${PROJECT_ENV_FILE_DEFAULT}}"

log() { printf '[qwen38-flash-next] %s\n' "$*"; }
warn() { printf '[qwen38-flash-next] WARNING: %s\n' "$*" >&2; }
die() { printf '[qwen38-flash-next] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Qwen3.8 Flash Next NVFP4 — dual-DGX-Spark SGLang lifecycle

Run on the head/rank-0 Spark:

  python3 -m ml.cli qwen38-flash-next setup
  python3 -m ml.cli qwen38-flash-next download
  scripts/start-Qwen38-Flash-Next-Dual-DSpark.sh
  python3 -m ml.cli qwen38-flash-next smoke

Actions:
  bootstrap    Clone and detach the reviewed MiaAI-Lab upstream revision
  configure    Generate upstream .env from the checked-in ml-compute profile
  check        Validate the pin/profile and run upstream's two-node doctor
  setup        Run bootstrap, configure, and check
  download     Build/patch both images, download, verify, and rsync weights
  start        Launch worker-first TP=2, then the authenticated safety proxy
  status       Show both containers, the raw API, and proxy state
  memory       Show shared-memory and GPU use on both nodes
  smoke        Verify proxy safety and perform one Qwen completion
  logs         Follow the head SGLang container log
  logs-worker  Follow the worker SGLang container log
  stop         Stop the proxy and both SGLang containers; preserve caches
  update       Fetch and re-checkout the revision pinned by this script
  all          Run setup, download, and start
  path         Print checkout, profile, and project environment paths
  help         Show this help

Reviewed profile:
  - two GB10 nodes, SGLang TP=2, ModelOpt NVFP4, in-checkpoint NEXTN 3/1/4
  - 900K YaRN request ceiling, 1K prefill chunks, 0.82 UMA fraction
  - PLE n-gram embedding auto-offload and patched SM121 QSA attention
  - raw unauthenticated API on 127.0.0.1:8888 only
  - authenticated allow-list proxy on 0.0.0.0:8000
  - never use --load-format dummy: it can hard-freeze GB10 unified memory

Environment overrides:
  QWEN38_DSPARK_CONFIG_FILE       checked-in profile path
  QWEN38_DSPARK_RECIPE_DIR        ignored upstream checkout/data path
  QWEN38_DSPARK_PROJECT_ENV_FILE  API_KEY source (default: .env.local)
  HF_TOKEN                        optional Hugging Face token for download
  Any profile key may be exported for one invocation. Public raw binds,
  dummy weights, disabled kernel patching, and unsafe PLE placement are denied.
EOF
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

profile_file_value() {
  local key="$1" fallback="${2:-}" value=""
  if [[ -f "${PROFILE_FILE}" ]]; then
    value="$(sed -n "s/^${key}=//p" "${PROFILE_FILE}" | tail -n 1)"
  fi
  printf '%s' "${value:-${fallback}}"
}

profile_value() {
  local key="$1" fallback="${2:-}"
  if [[ -n "${!key+x}" ]]; then
    printf '%s' "${!key}"
  else
    profile_file_value "${key}" "${fallback}"
  fi
}

load_profile() {
  [[ -f "${PROFILE_FILE}" ]] || die "Missing profile: ${PROFILE_FILE}"
  local key fallback value
  while IFS='|' read -r key fallback; do
    [[ -n "${key}" ]] || continue
    value="$(profile_value "${key}" "${fallback}")"
    export "${key}=${value}"
  done <<EOF
HEAD_CX7_IP|10.0.22.1
WORKER_CX7_IP|10.0.22.2
HEAD_CX7_IF|enp1s0f1np1
WORKER_CX7_IF|enp1s0f0np0
HEAD_CX7_IB|rocep1s0f1
WORKER_CX7_IB|rocep1s0f0
WORKER_USER|
WORKER_HOST|spark2
WORKER_SSH|
HOST_BIND|127.0.0.1
PORT|8888
DIST_PORT|26400
SERVED_MODEL_NAME|Qwen3.8-Flash-Next-NVFP4
MEM_FRACTION_STATIC|0.82
CONTEXT_LENGTH|262144
QWEN38_EFFECTIVE_CONTEXT_LENGTH|900000
CHUNKED_PREFILL_SIZE|1024
MAX_RUNNING_REQUESTS|16
SPEC_STEPS|3
SPEC_TOPK|1
SPEC_DRAFT|4
ENABLE_DECODE_GRAPHS|1
CUDA_GRAPH_BS|1 2 3 4 5 6 7 8 10 12 14 16
ATTENTION_BACKEND|flashinfer
KV_CACHE_DTYPE|
PLE_OFFLOAD|
FP4_GEMM_BACKEND|
LINEAR_ATTN_PREFILL_BACKEND|
LINEAR_ATTN_DECODE_BACKEND|
MAMBA_RADIX_CACHE_STRATEGY|extra_buffer
MAMBA_TRACK_INTERVAL|64
MAMBA_FULL_MEMORY_RATIO|0.3
REASONING_PARSER|auto
TOOL_CALL_PARSER|auto
CPUSET|5-9,15-19
USE_HOST_NCCL|1
DOWNLOAD_MODE|rsync
HF_REVISION|${MODEL_REVISION_DEFAULT}
WAIT_TIMEOUT_MIN|90
NCCL_DEBUG|WARN
NCCL_CROSS_NIC|0
EXTRA_ARGS|
KERNEL_PATCH|1
BASE_IMAGE|${BASE_IMAGE_DEFAULT}
PATCHED_IMAGE|qwen38-flashnext-dspark:ml-compute
HF_HOME|${RECIPE_DIR}/cache/huggingface
WORKER_HF_HOME|
DSPARK_PROXY_HOST|0.0.0.0
DSPARK_PROXY_PORT|8000
QWEN38_MIN_AVAILABLE_GIB|112
EOF
}

upstream_keys() {
  cat <<'EOF'
HEAD_CX7_IP
WORKER_CX7_IP
HEAD_CX7_IF
WORKER_CX7_IF
HEAD_CX7_IB
WORKER_CX7_IB
WORKER_USER
WORKER_HOST
WORKER_SSH
HOST_BIND
PORT
DIST_PORT
SERVED_MODEL_NAME
MEM_FRACTION_STATIC
CONTEXT_LENGTH
CHUNKED_PREFILL_SIZE
MAX_RUNNING_REQUESTS
SPEC_STEPS
SPEC_TOPK
SPEC_DRAFT
ENABLE_DECODE_GRAPHS
CUDA_GRAPH_BS
ATTENTION_BACKEND
KV_CACHE_DTYPE
PLE_OFFLOAD
FP4_GEMM_BACKEND
LINEAR_ATTN_PREFILL_BACKEND
LINEAR_ATTN_DECODE_BACKEND
MAMBA_RADIX_CACHE_STRATEGY
MAMBA_TRACK_INTERVAL
MAMBA_FULL_MEMORY_RATIO
REASONING_PARSER
TOOL_CALL_PARSER
CPUSET
USE_HOST_NCCL
DOWNLOAD_MODE
HF_REVISION
WAIT_TIMEOUT_MIN
NCCL_DEBUG
NCCL_CROSS_NIC
EXTRA_ARGS
KERNEL_PATCH
BASE_IMAGE
PATCHED_IMAGE
HF_HOME
WORKER_HF_HOME
EOF
}

project_env_value() {
  local key="$1"
  [[ -f "${PROJECT_ENV_FILE}" ]] || return 0
  sed -n "s/^${key}=//p" "${PROJECT_ENV_FILE}" | tail -n 1
}

proxy_api_key() {
  printf '%s' "${API_KEY:-$(project_env_value API_KEY)}"
}

check_proxy_key() {
  local api_key
  api_key="$(proxy_api_key)"
  [[ -n "${api_key}" ]] || die "API_KEY is missing from ${PROJECT_ENV_FILE}"
  [[ "${api_key}" =~ ^[A-Za-z0-9._~-]+$ ]] \
    || die "API_KEY may contain only letters, digits, dot, underscore, tilde, and hyphen"
}

project_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s' "${PYTHON_BIN}"
  elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    printf '%s' "${VIRTUAL_ENV}/bin/python"
  elif [[ -x "${PROJECT_ROOT}/venv/bin/python" ]]; then
    printf '%s' "${PROJECT_ROOT}/venv/bin/python"
  elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    printf '%s' "${PROJECT_ROOT}/.venv/bin/python"
  else
    printf '%s' python3
  fi
}

recipe_head() {
  git -C "${RECIPE_DIR}" rev-parse HEAD 2>/dev/null || true
}

check_recipe_pin() {
  [[ -d "${RECIPE_DIR}/.git" ]] \
    || die "Recipe is not bootstrapped. Run: python3 -m ml.cli qwen38-flash-next bootstrap"
  local head
  head="$(recipe_head)"
  [[ "${head}" == "${UPSTREAM_REVISION}" ]] \
    || die "Upstream checkout is ${head:-unknown}, expected ${UPSTREAM_REVISION}; run qwen38-flash-next update"
}

sync_recipe_pin() {
  [[ -d "${RECIPE_DIR}/.git" ]] || die "Not a git checkout: ${RECIPE_DIR}"
  if ! git -C "${RECIPE_DIR}" diff --quiet \
      || ! git -C "${RECIPE_DIR}" diff --cached --quiet; then
    die "Upstream checkout has tracked modifications; preserve or revert them before update"
  fi
  if ! git -C "${RECIPE_DIR}" cat-file -e "${UPSTREAM_REVISION}^{commit}" 2>/dev/null; then
    git -C "${RECIPE_DIR}" fetch --no-tags origin "${UPSTREAM_REVISION}"
  fi
  git -C "${RECIPE_DIR}" checkout --detach "${UPSTREAM_REVISION}"
  log "Pinned upstream revision: ${UPSTREAM_REVISION}"
}

bootstrap() {
  need_cmd git
  if [[ -d "${RECIPE_DIR}/.git" ]]; then
    sync_recipe_pin
    return
  fi
  [[ ! -e "${RECIPE_DIR}" ]] || die "Recipe path exists but is not a git checkout: ${RECIPE_DIR}"
  mkdir -p "$(dirname "${RECIPE_DIR}")"
  git clone --no-tags "${UPSTREAM_REPO}" "${RECIPE_DIR}"
  sync_recipe_pin
}

validate_profile() {
  load_profile
  [[ "${HOST_BIND}" == "127.0.0.1" ]] \
    || die "HOST_BIND must be 127.0.0.1; expose only the authenticated safety proxy"
  [[ "${PORT}" == "8888" ]] \
    || warn "PORT differs from the reviewed private raw port 8888"
  [[ "${KERNEL_PATCH}" == "1" ]] \
    || die "KERNEL_PATCH must remain enabled: stock QSA attention cannot boot on SM121"
  [[ -z "${PLE_OFFLOAD}" || "${PLE_OFFLOAD}" == "1" ]] \
    || die "PLE_OFFLOAD=0 is unsafe on GB10 unified memory; use auto or explicit offload"
  [[ "${SPEC_TOPK}" == "1" ]] \
    || die "SPEC_TOPK must remain 1 for Qwen4Exp PLE speculative decoding"
  (( SPEC_DRAFT == SPEC_STEPS + 1 )) \
    || die "SPEC_DRAFT must equal SPEC_STEPS + 1"
  (( CHUNKED_PREFILL_SIZE <= 1024 )) \
    || die "CHUNKED_PREFILL_SIZE above 1024 is unsafe for the reviewed 900K QSA profile"
  awk -v value="${MEM_FRACTION_STATIC}" 'BEGIN {exit !(value <= 0.82)}' \
    || die "MEM_FRACTION_STATIC above 0.82 can exhaust GB10 unified memory"
  [[ " ${EXTRA_ARGS} " != *" --host "* && " ${EXTRA_ARGS} " != *" --host="* ]] \
    || die "EXTRA_ARGS may not override the private raw host"
  [[ " ${EXTRA_ARGS} " != *" --port "* && " ${EXTRA_ARGS} " != *" --port="* ]] \
    || die "EXTRA_ARGS may not override the private raw port"
  [[ " ${EXTRA_ARGS} " != *" --load-format dummy"* ]] \
    || die "--load-format dummy can hard-freeze a GB10 node and is forbidden"
  [[ " ${EXTRA_ARGS} " != *" --no-ple-offload-embedding"* ]] \
    || die "Disabling PLE offload is unsafe on this GB10 profile"
  [[ "${EXTRA_ARGS}" == *"--context-length ${QWEN38_EFFECTIVE_CONTEXT_LENGTH}"* ]] \
    || warn "EXTRA_ARGS no longer installs the registered ${QWEN38_EFFECTIVE_CONTEXT_LENGTH}-token ceiling"
  [[ "${EXTRA_ARGS}" == *"--json-model-override-args"* ]] \
    || warn "EXTRA_ARGS no longer installs the reviewed YaRN model override"
  [[ "${HF_REVISION}" == "${MODEL_REVISION_DEFAULT}" ]] \
    || warn "HF_REVISION differs from the reviewed model checkpoint"
  [[ "${BASE_IMAGE}" == "${BASE_IMAGE_DEFAULT}" ]] \
    || warn "BASE_IMAGE differs from the reviewed SGLang manifest"
}

configure() {
  check_recipe_pin
  validate_profile
  local env_tmp key
  env_tmp="$(mktemp "${RECIPE_DIR}/.env.ml-compute.XXXXXX")"
  {
    printf '# Generated by ml-compute; edit %s instead.\n' "${PROFILE_FILE}"
    while IFS= read -r key; do
      [[ -n "${key}" ]] || continue
      printf '%s=%s\n' "${key}" "${!key}"
    done < <(upstream_keys)
  } >"${env_tmp}"
  mv "${env_tmp}" "${RECIPE_DIR}/.env"
  log "Configured pinned upstream profile at ${RECIPE_DIR}/.env"
}

run_upstream() {
  check_recipe_pin
  validate_profile
  (cd "${RECIPE_DIR}" && ./start.sh "$@")
}

worker_ssh_target() {
  if [[ -n "${WORKER_SSH}" ]]; then
    printf '%s' "${WORKER_SSH}"
  elif [[ -n "${WORKER_USER}" ]]; then
    printf '%s@%s' "${WORKER_USER}" "${WORKER_HOST}"
  else
    printf '%s' "${WORKER_HOST}"
  fi
}

available_gib() {
  awk '/^MemAvailable:/ {printf "%.3f", $2 / 1048576}' /proc/meminfo
}

check_launch_memory() {
  local head_available worker_available required target
  head_available="$(available_gib)"
  required="${QWEN38_MIN_AVAILABLE_GIB}"
  target="$(worker_ssh_target)"
  worker_available="$(ssh -o ConnectTimeout=10 -o BatchMode=yes "${target}" \
    "awk '/^MemAvailable:/ {printf \"%.3f\", \$2 / 1048576}' /proc/meminfo")"
  awk -v have="${head_available}" -v need="${required}" 'BEGIN {exit !(have >= need)}' \
    || die "Head has ${head_available} GiB MemAvailable; cold launch requires ${required} GiB"
  awk -v have="${worker_available}" -v need="${required}" 'BEGIN {exit !(have >= need)}' \
    || die "Worker has ${worker_available} GiB MemAvailable; cold launch requires ${required} GiB"
  log "Cold-start memory: head ${head_available} GiB, worker ${worker_available} GiB (minimum ${required})"
}

check_host() {
  validate_profile
  need_cmd docker
  need_cmd curl
  need_cmd ssh
  check_proxy_key
  run_upstream doctor
  log "Dual-Spark Qwen preflight passed"
}

check_public_exposure() {
  local funnel_status raw_port
  command -v tailscale >/dev/null 2>&1 || return 0
  raw_port="${PORT:-8888}"
  funnel_status="$(tailscale funnel status 2>/dev/null || true)"
  if [[ -z "${funnel_status}" ]] && command -v sudo >/dev/null 2>&1; then
    funnel_status="$(sudo -n tailscale funnel status 2>/dev/null || true)"
  fi
  if grep -Eq "proxy http://(127\\.0\\.0\\.1|localhost):${raw_port}([ /]|$)" <<<"${funnel_status}"; then
    die "Tailscale Funnel targets unauthenticated raw port ${raw_port}. Point it to ${DSPARK_PROXY_PORT} instead"
  fi
}

run_proxy_cli() {
  local action="$1" python api_key
  python="$(project_python)"
  api_key="$(proxy_api_key)"
  (cd "${PROJECT_ROOT}" && \
    API_KEY="${api_key}" \
    DSPARK_PROXY_HOST="${DSPARK_PROXY_HOST}" \
    DSPARK_PROXY_PORT="${DSPARK_PROXY_PORT}" \
    DSPARK_PROXY_UPSTREAM_URL="http://127.0.0.1:${PORT}" \
    "${python}" -m ml.cli dspark-proxy "${action}")
}

raw_model_ready() {
  local python
  python="$(project_python)"
  curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null \
    | EXPECTED_MODEL="${SERVED_MODEL_NAME}" "${python}" -c \
      'import json, os, sys; data=json.load(sys.stdin); ids={m.get("id") for m in data.get("data", [])}; raise SystemExit(0 if os.environ["EXPECTED_MODEL"] in ids else 1)' \
    >/dev/null 2>&1
}

start_service() {
  check_recipe_pin
  validate_profile
  check_proxy_key
  check_public_exposure
  if raw_model_ready; then
    log "Pinned Qwen model is already ready on the private raw endpoint"
  else
    check_launch_memory
    run_upstream serve
  fi
  run_proxy_cli serve
  if ! run_proxy_cli smoke; then
    run_proxy_cli stop || true
    die "Safety proxy checks failed; raw SGLang remains private on 127.0.0.1:${PORT}"
  fi
  log "Public authenticated API: http://127.0.0.1:${DSPARK_PROXY_PORT}/v1"
  log "Raw unauthenticated API:  http://127.0.0.1:${PORT}/v1 (loopback only)"
}

show_status() {
  run_upstream status
  run_proxy_cli status
}

show_memory() {
  load_profile
  local target
  target="$(worker_ssh_target)"
  printf '%s\n' 'Head memory:'
  free -h
  nvidia-smi || true
  printf '\nWorker memory (%s):\n' "${target}"
  ssh -o ConnectTimeout=10 -o BatchMode=yes "${target}" 'free -h; nvidia-smi' || true
  warn "This 900K TP2 profile is dedicated to the Qwen cluster; stop other model workloads before launch"
}

inference_smoke() (
  run_proxy_cli smoke
  local python api_key response_file
  python="$(project_python)"
  api_key="$(proxy_api_key)"
  response_file="$(mktemp)"
  trap 'rm -f "${response_file}"' EXIT
  curl -fsS --max-time 600 \
    "http://127.0.0.1:${DSPARK_PROXY_PORT}/v1/chat/completions" \
    -H "Authorization: Bearer ${api_key}" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Return only the number 391.\"}],\"temperature\":0,\"max_tokens\":64,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    -o "${response_file}"
  "${python}" - "${response_file}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
choices = payload.get("choices") or []
message = choices[0].get("message", {}) if choices else {}
if not (message.get("content") or message.get("reasoning_content")):
    raise SystemExit(f"invalid chat response: {payload}")
print("Qwen3.8 Flash Next model completion passed")
PY
)

stop_service() {
  load_profile
  run_proxy_cli stop || true
  if [[ -d "${RECIPE_DIR}/.git" ]]; then
    run_upstream stop || true
  fi
  log "Stopped dual-Spark Qwen service; weights, images, and JIT caches were preserved"
}

action="${1:-help}"
case "${action}" in
  bootstrap) bootstrap ;;
  configure) configure ;;
  check) check_recipe_pin; check_host ;;
  setup) bootstrap; configure; check_host ;;
  download) run_upstream download ;;
  start) start_service ;;
  status) load_profile; show_status ;;
  memory) show_memory ;;
  smoke) load_profile; inference_smoke ;;
  logs) run_upstream logs head ;;
  logs-worker) run_upstream logs worker ;;
  stop) stop_service ;;
  update) sync_recipe_pin ;;
  all) bootstrap; configure; check_host; run_upstream download; start_service ;;
  path)
    printf 'recipe=%s\nprofile=%s\nproject_env=%s\nupstream_revision=%s\nmodel_revision=%s\n' \
      "${RECIPE_DIR}" "${PROFILE_FILE}" "${PROJECT_ENV_FILE}" \
      "${UPSTREAM_REVISION}" "${MODEL_REVISION_DEFAULT}"
    ;;
  help|-h|--help) usage ;;
  *) die "Unknown action: ${action}. Run qwen38-flash-next help" ;;
esac
