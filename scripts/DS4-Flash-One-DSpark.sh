#!/usr/bin/env bash
# DeepSeek V4 Flash 0731 on one dedicated DGX Spark.
#
# Lifecycle wrapper for MiaAI-Lab's EXL3/ExLlamaV3 TP1 recipe:
# https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark
# The reviewed upstream revision, model revision, and runtime image are pinned
# below. The upstream checkout and its ~107 GB weights remain under ignored
# data/dspark/. Run this script only on the single Spark that will serve it.

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_REPO_DEFAULT="https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark.git"
UPSTREAM_REVISION_DEFAULT="fdcd538fbf95fb15b2d6850db9613d22b2c889b8"
RECIPE_DIR_DEFAULT="${PROJECT_ROOT}/data/dspark/miaai-deepseek-v4-flash-one-spark"
PROFILE_FILE_DEFAULT="${PROJECT_ROOT}/config/dspark-one-deepseek-v4-flash-0731.env"
PROJECT_ENV_FILE_DEFAULT="${PROJECT_ROOT}/.env.local"

MODEL_REPO_DEFAULT="0xSero/deepseek-v4-flash-0731-spark"
MODEL_REVISION_DEFAULT="22f28d32b9b29b4352eaa380ff8c2c170b2847ab"
IMAGE_DEFAULT="ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4"

UPSTREAM_REPO="${DSPARK_ONE_UPSTREAM_REPO:-${UPSTREAM_REPO_DEFAULT}}"
UPSTREAM_REVISION="${DSPARK_ONE_UPSTREAM_REVISION:-${UPSTREAM_REVISION_DEFAULT}}"
RECIPE_DIR="${DSPARK_ONE_RECIPE_DIR:-${RECIPE_DIR_DEFAULT}}"
PROFILE_FILE="${DSPARK_ONE_CONFIG_FILE:-${PROFILE_FILE_DEFAULT}}"
PROJECT_ENV_FILE="${DSPARK_ONE_PROJECT_ENV_FILE:-${PROJECT_ENV_FILE_DEFAULT}}"

log() { printf '[dspark-one] %s\n' "$*"; }
warn() { printf '[dspark-one] WARNING: %s\n' "$*" >&2; }
die() { printf '[dspark-one] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
DeepSeek V4 Flash 0731 — one-Spark EXL3 setup and lifecycle

Run on the dedicated third DGX Spark:

  python3 -m ml.cli dspark-one setup
  python3 -m ml.cli dspark-one build
  python3 -m ml.cli dspark-one download
  scripts/start-DS4-Flash-One-DSpark.sh
  python3 -m ml.cli dspark-one smoke

Actions:
  bootstrap  Clone and detach the reviewed MiaAI-Lab upstream revision
  configure  Generate upstream compose.yml from the checked-in safe profile
  check      Validate pin, GB10/Docker, memory, disk, API key, and private bind
  setup      Run bootstrap, configure, and check
  build      Pull the digest-pinned SparkInfer runtime image
  download   Download, coalesce, and checksum the ~107 GB EXL3 checkpoint
  gpu-check  Initialize CUDA inside the pinned runtime image
  start      Start raw vLLM on 127.0.0.1:8888, then the auth proxy on :8000
  status     Show the upstream container, raw API, and proxy state
  memory     Show host memory/GPU use; this is a dedicated-host profile
  smoke      Verify proxy auth/deny rules and perform one model completion
  logs       Follow upstream container logs
  stop       Stop proxy and remove the container; weights/caches are preserved
  update     Fetch and re-checkout the revision pinned by this script
  all        Run setup, build, download, gpu-check, and start
  path       Print checkout, profile, and project environment paths
  help       Show this help

Important defaults:
  - one node, TP=1, EXL3 3.0 bpw, DSpark K5 speculative decoding
  - MAX_MODEL_LEN=384000, MAX_NUM_SEQS=1, GPU_MEMORY_UTILIZATION=0.94
  - native 432-byte NVFP4 KV records (KV_RECORD=stock432)
  - upstream reports about 44-47 structured decode tok/s
  - at least 114.3 GiB MemAvailable is required before a cold launch
  - this profile leaves no room for the Harness on the same Spark
  - API_KEY is read from .env.local only by the safety proxy

Environment overrides:
  DSPARK_ONE_CONFIG_FILE       checked-in profile path
  DSPARK_ONE_RECIPE_DIR        ignored upstream checkout/data path
  DSPARK_ONE_PROJECT_ENV_FILE  API_KEY source (default: project .env.local)
  DSPARK_ONE_NO_WAIT=1         start the model without waiting; proxy stays off
  HF_TOKEN                     optional Hugging Face token for download
  Any key in config/dspark-one-deepseek-v4-flash-0731.env may be exported
  for a one-command override. Unsafe public raw binds are always rejected.
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
  while read -r key fallback; do
    value="$(profile_value "${key}" "${fallback}")"
    export "${key}=${value}"
  done <<EOF
MODEL_REPO ${MODEL_REPO_DEFAULT}
MODEL_REVISION ${MODEL_REVISION_DEFAULT}
SERVED_MODEL_NAME deepseek-v4-flash-0731
MAX_MODEL_LEN 384000
MAX_NUM_SEQS 1
MAX_NUM_BATCHED_TOKENS 8224
GPU_MEMORY_UTILIZATION 0.94
KV_RECORD stock432
MODE dspark
LONG_PREFILL_TOKEN_THRESHOLD 1024
MAX_NUM_PARTIAL_PREFILLS 0
MAX_CUDAGRAPH_CAPTURE_SIZE 24
CUDAGRAPH_CAPTURE_SIZES 6,12,24
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS 0
KV_OFFLOAD_GB 0
VERIFY_MODEL_CHECKSUMS 1
ABLATE 0
DEFAULT_CHAT_TEMPLATE_KWARGS_THINKING true
DEFAULT_CHAT_TEMPLATE_KWARGS_EFFORT max
SERVING_HOST 127.0.0.1
SERVING_PORT 8888
DSPARK_PROXY_HOST 0.0.0.0
DSPARK_PROXY_PORT 8000
DSPARK_ONE_MIN_AVAILABLE_GIB 114.3
DSPARK_ONE_MIN_DISK_GIB 130
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
    || die "Recipe is not bootstrapped. Run: python3 -m ml.cli dspark-one bootstrap"
  local head
  head="$(recipe_head)"
  [[ "${head}" == "${UPSTREAM_REVISION}" ]] \
    || die "Upstream checkout is ${head:-unknown}, expected ${UPSTREAM_REVISION}; run dspark-one update"
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
  [[ "${MODEL_REPO}" == "${MODEL_REPO_DEFAULT}" ]] \
    || warn "MODEL_REPO differs from the reviewed EXL3 checkpoint"
  [[ "${MODEL_REVISION}" == "${MODEL_REVISION_DEFAULT}" ]] \
    || warn "MODEL_REVISION differs from the reviewed checkpoint pin"
  [[ "${SERVING_HOST}" == "127.0.0.1" ]] \
    || die "SERVING_HOST must be 127.0.0.1; expose only the authenticated safety proxy"
  [[ "${SERVING_PORT}" == "8888" ]] \
    || warn "SERVING_PORT differs from the reviewed raw port 8888"
  [[ "${MAX_MODEL_LEN}" == "384000" ]] \
    || warn "MAX_MODEL_LEN differs from MiaAI-Lab's validated 384K profile"
  [[ "${MAX_NUM_SEQS}" == "1" ]] \
    || warn "MAX_NUM_SEQS differs from the validated deep-context profile"
  [[ "${MAX_NUM_BATCHED_TOKENS}" == "8224" ]] \
    || die "MAX_NUM_BATCHED_TOKENS must remain 8224; lowering it can crash the locked MLA workspace"
  [[ "${GPU_MEMORY_UTILIZATION}" == "0.94" ]] \
    || warn "GPU_MEMORY_UTILIZATION differs from the validated 0.94 profile"
  [[ "${KV_RECORD}" == "stock432" ]] \
    || warn "KV_RECORD differs from the validated native NVFP4 stock432 layout"
  [[ "${ABLATE}" == "0" ]] \
    || warn "ABLATE is enabled; this changes the model's refusal behavior and safety profile"
}

run_upstream() {
  check_recipe_pin
  validate_profile
  (cd "${RECIPE_DIR}" && ./start.sh "$@")
}

configure() {
  check_recipe_pin
  validate_profile
  run_upstream compose-gen
  log "Configured one-Spark profile at ${RECIPE_DIR}/compose.yml"
}

available_gib() {
  awk '/^MemAvailable:/ {printf "%.3f", $2 / 1048576}' /proc/meminfo
}

check_launch_memory() {
  local available required
  available="$(available_gib)"
  required="${DSPARK_ONE_MIN_AVAILABLE_GIB}"
  awk -v have="${available}" -v need="${required}" 'BEGIN {exit !(have >= need)}' \
    || die "Only ${available} GiB MemAvailable; this 0.94 UMA profile requires at least ${required} GiB. Stop other models and the Harness first"
  log "MemAvailable ${available} GiB (minimum ${required} GiB)"
}

check_disk() {
  local target available required
  target="$(dirname "${RECIPE_DIR}")"
  mkdir -p "${target}"
  available="$(( $(df -Pk "${target}" | awk 'NR==2 {print $4}') / 1024 / 1024 ))"
  required="${DSPARK_ONE_MIN_DISK_GIB}"
  (( available >= required )) \
    || warn "Only ${available} GiB free at ${target}; first setup wants at least ${required} GiB"
}

raw_health_url() {
  printf 'http://127.0.0.1:%s/health' "${SERVING_PORT:-8888}"
}

check_host() {
  validate_profile
  need_cmd docker
  need_cmd curl
  need_cmd nvidia-smi
  [[ "$(uname -m)" == "aarch64" ]] \
    || die "This runtime is aarch64-only; host reports $(uname -m)"
  docker compose version >/dev/null 2>&1 || die "Docker Compose v2 is unavailable"
  docker info >/dev/null 2>&1 \
    || die "Docker daemon is not accessible by user $(id -un); add the user to the docker group and re-login"
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q GB10 \
    || die "NVIDIA GB10 was not detected"
  if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet earlyoom; then
    die "earlyoom is active; upstream requires: sudo systemctl disable --now earlyoom"
  fi
  check_disk
  if ! curl -fsS --max-time 3 "$(raw_health_url)" >/dev/null 2>&1; then
    check_launch_memory
    local compute_apps
    compute_apps="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)"
    [[ -z "${compute_apps}" ]] \
      || die "Another CUDA compute workload is active: ${compute_apps//$'\n'/; }. Stop it before allocating 94% of UMA"
  fi
  check_proxy_key
  log "One-Spark preflight passed"
}

check_public_exposure() {
  local funnel_status raw_port
  command -v tailscale >/dev/null 2>&1 || return 0
  raw_port="${SERVING_PORT:-8888}"
  funnel_status="$(tailscale funnel status 2>/dev/null || true)"
  if [[ -z "${funnel_status}" ]] && command -v sudo >/dev/null 2>&1; then
    funnel_status="$(sudo -n tailscale funnel status 2>/dev/null || true)"
  fi
  if grep -Eq "proxy http://(127\\.0\\.0\\.1|localhost):${raw_port}([ /]|$)" <<<"${funnel_status}"; then
    die "Tailscale Funnel targets unauthenticated raw port ${raw_port}. Run: sudo tailscale funnel --https=443 off; sudo tailscale funnel --bg=true ${DSPARK_PROXY_PORT}"
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
    DSPARK_PROXY_UPSTREAM_URL="http://127.0.0.1:${SERVING_PORT}" \
    "${python}" -m ml.cli dspark-proxy "${action}")
}

start_service() {
  check_host
  check_public_exposure
  if [[ "${DSPARK_ONE_NO_WAIT:-0}" == "1" ]]; then
    run_upstream --no-wait
    warn "Started without waiting; the public proxy remains off. Run dspark-one start again after /health is ready"
    return
  fi
  run_upstream start
  run_proxy_cli serve
  if ! run_proxy_cli smoke; then
    run_proxy_cli stop || true
    die "Safety proxy checks failed; raw model remains private on 127.0.0.1:${SERVING_PORT}"
  fi
  log "Public authenticated API: http://127.0.0.1:${DSPARK_PROXY_PORT}/v1"
  log "Raw unauthenticated API:  http://127.0.0.1:${SERVING_PORT}/v1 (loopback only)"
}

gpu_check() {
  validate_profile
  need_cmd docker
  docker run --rm --gpus all \
    --entrypoint /opt/runtime-venv/bin/python \
    "${IMAGE_DEFAULT}" -c \
    'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)'
}

show_status() {
  run_upstream status
  if curl -fsS --max-time 5 "$(raw_health_url)" >/dev/null 2>&1; then
    log "Raw model health: ready"
  else
    warn "Raw model health: unavailable"
  fi
  run_proxy_cli status
}

show_memory() {
  free -h
  nvidia-smi || true
  local available
  available="$(available_gib)"
  log "MemAvailable: ${available} GiB"
  warn "This 0.94 profile is dedicated to the model; do not start the Harness on this Spark"
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
    -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Return only the number 391.\"}],\"temperature\":0,\"max_completion_tokens\":64,\"chat_template_kwargs\":{\"thinking\":false}}" \
    -o "${response_file}"
  "${python}" - "${response_file}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
choices = payload.get("choices") or []
if not choices or not (choices[0].get("message") or {}).get("content"):
    raise SystemExit(f"invalid chat response: {payload}")
print("One-Spark model completion passed")
PY
)

stop_service() {
  load_profile
  run_proxy_cli stop || true
  if [[ -d "${RECIPE_DIR}/.git" ]]; then
    run_upstream down || true
  fi
  log "Stopped one-Spark service; weights and caches were preserved"
}

action="${1:-help}"
case "${action}" in
  bootstrap) bootstrap ;;
  configure) configure ;;
  check) check_recipe_pin; check_host ;;
  setup) bootstrap; configure; check_host ;;
  build) run_upstream pull ;;
  download) check_recipe_pin; validate_profile; (cd "${RECIPE_DIR}" && ./download.sh) ;;
  gpu-check) gpu_check ;;
  start) start_service ;;
  status) load_profile; show_status ;;
  memory) load_profile; show_memory ;;
  smoke) load_profile; inference_smoke ;;
  logs) run_upstream logs ;;
  stop) stop_service ;;
  update) sync_recipe_pin ;;
  all) bootstrap; configure; check_host; run_upstream pull; (cd "${RECIPE_DIR}" && ./download.sh); gpu_check; start_service ;;
  path)
    printf 'recipe=%s\nprofile=%s\nproject_env=%s\nupstream_revision=%s\n' \
      "${RECIPE_DIR}" "${PROFILE_FILE}" "${PROJECT_ENV_FILE}" "${UPSTREAM_REVISION}"
    ;;
  help|-h|--help) usage ;;
  *) die "Unknown action: ${action}. Run dspark-one help" ;;
esac
