#!/usr/bin/env bash
# Qwen3.8 Flash Next NVFP4 on two linked DGX Sparks using MiaAI's vLLM recipe.
#
# This is intentionally separate from Qwen38-Flash-Next-Dual-DSpark.sh, which
# retains MiaAI's older SGLang/NVFP4-KV implementation.

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_REPO_DEFAULT="https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks.git"
UPSTREAM_REVISION_DEFAULT="169fbad266f2791335a3102f0d3d625e7c295563"
MODEL_REVISION_DEFAULT="7b719225242aacd3dbd3f9407468c2ee9a9d2594"
VLLM_IMAGE_DEFAULT="vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8"
RUNTIME_DIR_DEFAULT="${PROJECT_ROOT}/data/dspark/qwen38-flash-next-vllm"
RECIPE_DIR_DEFAULT="${RUNTIME_DIR_DEFAULT}/miaai-vllm-dual-spark"
PROFILE_FILE_DEFAULT="${PROJECT_ROOT}/config/dspark-qwen38-flash-next-vllm.env"
PROJECT_ENV_FILE_DEFAULT="${PROJECT_ROOT}/.env.local"
PATCHER="${PROJECT_ROOT}/scripts/patch_miaai_qwen38_vllm_launcher.py"

UPSTREAM_REPO="${QWEN38_VLLM_UPSTREAM_REPO:-${UPSTREAM_REPO_DEFAULT}}"
UPSTREAM_REVISION="${QWEN38_VLLM_UPSTREAM_REVISION:-${UPSTREAM_REVISION_DEFAULT}}"
RUNTIME_DIR="${QWEN38_VLLM_RUNTIME_DIR:-${RUNTIME_DIR_DEFAULT}}"
RECIPE_DIR="${QWEN38_VLLM_RECIPE_DIR:-${RECIPE_DIR_DEFAULT}}"
PROFILE_FILE="${QWEN38_VLLM_CONFIG_FILE:-${PROFILE_FILE_DEFAULT}}"
PROJECT_ENV_FILE="${QWEN38_VLLM_PROJECT_ENV_FILE:-${PROJECT_ENV_FILE_DEFAULT}}"
GENERATED_LAUNCHER="${RECIPE_DIR}/start.ml-compute.sh"

log() { printf '[qwen38-flash-next-vllm] %s\n' "$*"; }
warn() { printf '[qwen38-flash-next-vllm] WARNING: %s\n' "$*" >&2; }
die() { printf '[qwen38-flash-next-vllm] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Qwen3.8 Flash Next NVFP4 — MiaAI vLLM TP2+EP+MTP3 lifecycle

Run on the head/rank-0 Spark:

  python3 -m ml.cli qwen38-flash-next-vllm setup
  python3 -m ml.cli qwen38-flash-next-vllm pull
  python3 -m ml.cli qwen38-flash-next-vllm download
  python3 -m ml.cli qwen38-flash-next-vllm start
  python3 -m ml.cli qwen38-flash-next-vllm smoke

Actions:
  bootstrap    Clone and detach MiaAI's reviewed vLLM revision
  configure    Resolve RoCE devices and materialize the pinned launcher/profile
  check        Validate pins, performance settings, nodes, memory, and API key
  setup        Run bootstrap, configure, and check
  pull         Pull the digest-pinned vLLM image on both nodes
  download     Download the pinned model snapshot and rsync it to the worker
  start        Launch worker-first vLLM TP2+EP+MTP3, then the safety proxy
  status       Show both containers, the private raw API, and proxy state
  memory       Show shared-memory and GPU use on both nodes
  smoke        Verify proxy safety and perform one Qwen completion
  logs         Follow the head vLLM container log
  logs-worker  Follow the worker vLLM container log
  stop         Stop the proxy and both vLLM containers; preserve caches
  update       Re-checkout the revision pinned by this script and reconfigure
  all          Run setup, pull, download, and start
  path         Print checkout, profile, image, and revision paths
  help         Show this help

MiaAI performance profile (kept unchanged):
  - vLLM TP=2 + expert parallel + MTP3 on two GB10 nodes
  - 1,000,000-token YaRN context, BF16 KV, 0.835 GPU memory utilization
  - 8 sequences, 8192 batched tokens, FULL_DECODE_ONLY CUDA graphs
  - GPU-resident FP8 PLE shim; PLE CPU offload disabled
  - qwen3 reasoning and qwen3_coder tool parsing

ml-compute adds immutable upstream/model/image pins, automatic RoCE discovery,
memory/readiness rollback, and a private raw bind on 127.0.0.1:8888 behind the
authenticated allow-list proxy on 0.0.0.0:8000. These changes do not alter the
model execution parameters used for MiaAI's reported performance.
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

absolute_project_path() {
  local path="$1"
  if [[ "${path}" == /* ]]; then
    printf '%s' "${path}"
  else
    printf '%s/%s' "${PROJECT_ROOT}" "${path#./}"
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
HEAD_CX7_IP|192.168.177.10
WORKER_CX7_IP|192.168.177.11
HEAD_CX7_IF|enp1s0f1np1
WORKER_CX7_IF|enp1s0f1np1
HEAD_CX7_IB|rocep1s0f1
WORKER_CX7_IB|rocep1s0f1
WORKER_USER|totalpass
MASTER_PORT|50000
NCCL_IB_GID_INDEX|3
HOST_BIND|127.0.0.1
PORT|8888
DSPARK_PROXY_HOST|0.0.0.0
DSPARK_PROXY_PORT|8000
MODEL_ID|RadixArk/Qwen3.8-Flash-Next-NVFP4
MODEL_REVISION|${MODEL_REVISION_DEFAULT}
SERVED_MODEL_NAME|qwen3.8-flash-next
VLLM_IMAGE|${VLLM_IMAGE_DEFAULT}
TENSOR_PARALLEL_SIZE|2
ENABLE_EXPERT_PARALLEL|true
MTP_NUM_SPECULATIVE_TOKENS|3
MAX_MODEL_LEN|1000000
YARN_ENABLE|true
YARN_FACTOR|4.0
VLLM_ALLOW_LONG_MAX_MODEL_LEN|1
GPU_MEMORY_UTILIZATION|0.835
MAX_NUM_SEQS|8
MAX_NUM_BATCHED_TOKENS|8192
KV_CACHE_DTYPE|auto
PLE_OFFLOAD|false
HF_HOME|${RUNTIME_DIR}/cache/huggingface
WORKER_HF_HOME|
WAIT_TIMEOUT_MIN|90
NCCL_DEBUG|WARN
EXTRA_VLLM_ARGS|
EXTRA_DOCKER_ARGS|
QWEN38_VLLM_MIN_AVAILABLE_GIB|108
QWEN38_VLLM_MIN_RUNTIME_AVAILABLE_GIB|6
EOF
  HF_HOME="$(absolute_project_path "${HF_HOME}")"
  export HF_HOME
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

worker_target() {
  if [[ -n "${WORKER_USER}" ]]; then
    printf '%s@%s' "${WORKER_USER}" "${WORKER_CX7_IP}"
  else
    printf '%s' "${WORKER_CX7_IP}"
  fi
}

wrun() {
  local target
  target="$(worker_target)"
  ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -o BatchMode=yes \
    "${target}" "$1"
}

recipe_head() {
  git -C "${RECIPE_DIR}" rev-parse HEAD 2>/dev/null || true
}

check_recipe_pin() {
  local head
  [[ -d "${RECIPE_DIR}/.git" ]] \
    || die "Recipe is not bootstrapped. Run qwen38-flash-next-vllm bootstrap"
  head="$(recipe_head)"
  [[ "${head}" == "${UPSTREAM_REVISION}" ]] \
    || die "Upstream checkout is ${head:-unknown}, expected ${UPSTREAM_REVISION}"
}

sync_recipe_pin() {
  if ! git -C "${RECIPE_DIR}" cat-file -e "${UPSTREAM_REVISION}^{commit}" 2>/dev/null; then
    git -C "${RECIPE_DIR}" fetch --no-tags origin "${UPSTREAM_REVISION}"
  fi
  git -C "${RECIPE_DIR}" checkout --detach "${UPSTREAM_REVISION}"
  log "Pinned MiaAI vLLM revision: ${UPSTREAM_REVISION}"
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
    || die "HOST_BIND must remain 127.0.0.1; expose only the authenticated proxy"
  [[ "${PORT}" == "8888" ]] || die "PORT must remain the reviewed raw port 8888"
  [[ "${MODEL_REVISION}" == "${MODEL_REVISION_DEFAULT}" ]] \
    || die "MODEL_REVISION differs from the reviewed MiaAI checkpoint"
  [[ "${VLLM_IMAGE}" == "${VLLM_IMAGE_DEFAULT}" ]] \
    || die "VLLM_IMAGE differs from the reviewed digest pin"
  [[ "${TENSOR_PARALLEL_SIZE}" == "2" ]] || die "TENSOR_PARALLEL_SIZE must remain 2"
  [[ "${ENABLE_EXPERT_PARALLEL}" == "true" ]] || die "Expert parallel must remain enabled"
  [[ "${MTP_NUM_SPECULATIVE_TOKENS}" == "3" ]] || die "MTP must remain at MiaAI's measured 3 tokens"
  [[ "${MAX_MODEL_LEN}" == "1000000" ]] || die "MAX_MODEL_LEN must remain 1000000"
  [[ "${YARN_ENABLE}" == "true" && "${YARN_FACTOR}" == "4.0" ]] \
    || die "MiaAI's 1M profile requires YaRN factor 4.0"
  [[ "${VLLM_ALLOW_LONG_MAX_MODEL_LEN}" == "1" ]] \
    || die "VLLM_ALLOW_LONG_MAX_MODEL_LEN must remain enabled"
  [[ "${GPU_MEMORY_UTILIZATION}" == "0.835" ]] || die "GPU_MEMORY_UTILIZATION must remain 0.835"
  [[ "${MAX_NUM_SEQS}" == "8" ]] || die "MAX_NUM_SEQS must remain 8"
  [[ "${MAX_NUM_BATCHED_TOKENS}" == "8192" ]] || die "MAX_NUM_BATCHED_TOKENS must remain 8192"
  [[ "${KV_CACHE_DTYPE}" == "auto" ]] || die "KV_CACHE_DTYPE must remain auto (BF16)"
  [[ "${PLE_OFFLOAD}" == "false" ]] \
    || die "PLE_OFFLOAD must remain false; MiaAI measured insufficient GB10 host headroom"
  [[ -z "${EXTRA_VLLM_ARGS}" && -z "${EXTRA_DOCKER_ARGS}" ]] \
    || die "Extra arguments are disabled so the measured MiaAI command line remains exact"
}

local_netdev_for_ip() {
  local wanted="$1"
  ip -o -4 addr show | awk -v wanted="${wanted}" '
    { split($4, address, "/") }
    address[1] == wanted { print $2; exit }
  '
}

worker_netdev_for_ip() {
  local wanted="$1"
  wrun "ip -o -4 addr show" | awk -v wanted="${wanted}" '
    { split($4, address, "/") }
    address[1] == wanted { print $2; exit }
  '
}

local_hca_for_netdev() {
  local netdev="$1" path
  for path in /sys/class/infiniband/*/device/net/"${netdev}"; do
    [[ -e "${path}" ]] || continue
    basename "$(dirname "$(dirname "$(dirname "${path}")")")"
    return
  done
}

worker_hca_for_netdev() {
  local netdev="$1"
  wrun "for path in /sys/class/infiniband/*/device/net/'${netdev}'; do [ -e \"\$path\" ] || continue; basename \"\$(dirname \"\$(dirname \"\$(dirname \"\$path\")\")\")\"; break; done"
}

resolve_cluster_interfaces() {
  local head_netdev worker_netdev head_hca worker_hca
  need_cmd ip
  head_netdev="$(local_netdev_for_ip "${HEAD_CX7_IP}")"
  worker_netdev="$(worker_netdev_for_ip "${WORKER_CX7_IP}")"
  [[ -n "${head_netdev}" ]] || die "No head interface owns ${HEAD_CX7_IP}"
  [[ -n "${worker_netdev}" ]] || die "No worker interface owns ${WORKER_CX7_IP}"
  head_hca="$(local_hca_for_netdev "${head_netdev}")"
  worker_hca="$(worker_hca_for_netdev "${worker_netdev}")"
  [[ -n "${head_hca}" ]] || die "No head RDMA HCA maps to ${head_netdev}"
  [[ -n "${worker_hca}" ]] || die "No worker RDMA HCA maps to ${worker_netdev}"
  HEAD_CX7_IF="${head_netdev}"
  WORKER_CX7_IF="${worker_netdev}"
  HEAD_CX7_IB="${head_hca}"
  WORKER_CX7_IB="${worker_hca}"
  export HEAD_CX7_IF WORKER_CX7_IF HEAD_CX7_IB WORKER_CX7_IB
  log "RoCE mapping: head ${HEAD_CX7_IF}/${HEAD_CX7_IB}, worker ${WORKER_CX7_IF}/${WORKER_CX7_IB}"
}

materialize_launcher() {
  local python
  python="$(project_python)"
  [[ -x "${PATCHER}" || -f "${PATCHER}" ]] || die "Missing launcher patcher: ${PATCHER}"
  "${python}" "${PATCHER}" "${RECIPE_DIR}/start.sh" "${GENERATED_LAUNCHER}"
  bash -n "${GENERATED_LAUNCHER}" || die "Generated vLLM launcher is invalid"
}

configure() {
  check_recipe_pin
  validate_profile
  resolve_cluster_interfaces
  mkdir -p "${HF_HOME}"
  local env_tmp
  env_tmp="$(mktemp "${RECIPE_DIR}/.env.ml-compute.XXXXXX")"
  {
    printf '# Generated by ml-compute; edit %s instead.\n' "${PROFILE_FILE}"
    printf 'HEAD_IP=%q\n' "${HEAD_CX7_IP}"
    printf 'WORKER_IP=%q\n' "${WORKER_CX7_IP}"
    printf 'WORKER_USER=%q\n' "${WORKER_USER}"
    printf 'IFACE=%q\n' "${HEAD_CX7_IF}"
    printf 'WORKER_IFACE=%q\n' "${WORKER_CX7_IF}"
    printf 'IB_HCA=%q\n' "=${HEAD_CX7_IB}"
    printf 'WORKER_IB_HCA=%q\n' "=${WORKER_CX7_IB}"
    printf 'IB_GID_INDEX=%q\n' "${NCCL_IB_GID_INDEX}"
    printf 'MODEL_ID=%q\n' "${MODEL_ID}"
    printf 'HF_REVISION=%q\n' "${MODEL_REVISION}"
    printf 'SERVED_MODEL_NAME=%q\n' "${SERVED_MODEL_NAME}"
    printf 'MAX_MODEL_LEN=%q\n' "${MAX_MODEL_LEN}"
    printf 'YARN_ENABLE=%q\n' "${YARN_ENABLE}"
    printf 'YARN_FACTOR=%q\n' "${YARN_FACTOR}"
    printf 'GPU_MEMORY_UTILIZATION=%q\n' "${GPU_MEMORY_UTILIZATION}"
    printf 'MAX_NUM_SEQS=%q\n' "${MAX_NUM_SEQS}"
    printf 'MAX_NUM_BATCHED_TOKENS=%q\n' "${MAX_NUM_BATCHED_TOKENS}"
    printf 'PORT=%q\n' "${PORT}"
    printf 'HOST_BIND=%q\n' "${HOST_BIND}"
    printf 'KV_CACHE_DTYPE=%q\n' "${KV_CACHE_DTYPE}"
    printf 'TENSOR_PARALLEL_SIZE=%q\n' "${TENSOR_PARALLEL_SIZE}"
    printf 'ENABLE_EXPERT_PARALLEL=%q\n' "${ENABLE_EXPERT_PARALLEL}"
    printf 'MTP_NUM_SPECULATIVE_TOKENS=%q\n' "${MTP_NUM_SPECULATIVE_TOKENS}"
    printf 'IMAGE=%q\n' "${VLLM_IMAGE}"
    printf 'PLE_OFFLOAD=%q\n' "${PLE_OFFLOAD}"
    printf 'HF_HOME=%q\n' "${HF_HOME}"
    printf 'WORKER_HF_HOME=%q\n' "${WORKER_HF_HOME}"
    printf 'EXTRA_VLLM_ARGS=%q\n' "${EXTRA_VLLM_ARGS}"
    printf 'EXTRA_DOCKER_ARGS=%q\n' "${EXTRA_DOCKER_ARGS}"
    printf 'VLLM_ALLOW_LONG_MAX_MODEL_LEN=%q\n' "${VLLM_ALLOW_LONG_MAX_MODEL_LEN}"
    printf 'MASTER_PORT=%q\n' "${MASTER_PORT}"
  } >"${env_tmp}"
  mv "${env_tmp}" "${RECIPE_DIR}/.env"
  materialize_launcher
  log "Configured pinned MiaAI vLLM profile at ${RECIPE_DIR}/.env"
}

available_gib() {
  awk '/^MemAvailable:/ {printf "%.3f", $2 / 1048576}' /proc/meminfo
}

worker_available_gib() {
  wrun "awk '/^MemAvailable:/ {printf \"%.3f\", \$2 / 1048576}' /proc/meminfo"
}

check_launch_memory() {
  local head_available worker_available required
  head_available="$(available_gib)"
  worker_available="$(worker_available_gib)"
  required="${QWEN38_VLLM_MIN_AVAILABLE_GIB}"
  awk -v have="${head_available}" -v need="${required}" 'BEGIN {exit !(have >= need)}' \
    || die "Head has ${head_available} GiB MemAvailable; vLLM cold launch requires ${required} GiB"
  awk -v have="${worker_available}" -v need="${required}" 'BEGIN {exit !(have >= need)}' \
    || die "Worker has ${worker_available} GiB MemAvailable; vLLM cold launch requires ${required} GiB"
  log "Cold-start memory: head ${head_available} GiB, worker ${worker_available} GiB"
}

drop_page_caches() {
  sync
  if sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
    log "Dropped head page cache before unified-memory launch"
  else
    warn "Could not drop the head page cache without prompting"
  fi
  if wrun "sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'" 2>/dev/null; then
    log "Dropped worker page cache before unified-memory launch"
  else
    warn "Could not drop the worker page cache without prompting"
  fi
}

stop_raw_containers() {
  docker rm -f vllm-fn >/dev/null 2>&1 || true
  wrun "docker rm -f vllm-fn >/dev/null 2>&1 || true" || true
}

check_runtime_memory_headroom() {
  local head_available worker_available required
  head_available="$(available_gib)"
  worker_available="$(worker_available_gib)"
  required="${QWEN38_VLLM_MIN_RUNTIME_AVAILABLE_GIB}"
  if ! awk -v have="${head_available}" -v need="${required}" 'BEGIN {exit !(have >= need)}' \
      || ! awk -v have="${worker_available}" -v need="${required}" 'BEGIN {exit !(have >= need)}'; then
    warn "Post-launch memory is head=${head_available} GiB worker=${worker_available} GiB; minimum is ${required} GiB"
    run_proxy_cli stop || true
    stop_raw_containers
    die "Qwen vLLM launch rolled back to preserve unified-memory headroom"
  fi
  log "Runtime memory reserve passed: head ${head_available} GiB, worker ${worker_available} GiB"
}

check_host() {
  check_recipe_pin
  validate_profile
  [[ -x "${GENERATED_LAUNCHER}" ]] || die "Run configure before check"
  need_cmd docker
  need_cmd curl
  need_cmd ssh
  need_cmd rsync
  check_proxy_key
  wrun "docker version >/dev/null"
  log "Dual-Spark MiaAI vLLM preflight passed"
}

pull_images() {
  check_recipe_pin
  validate_profile
  docker pull "${VLLM_IMAGE}"
  wrun "docker pull '${VLLM_IMAGE}'"
  log "Digest-pinned vLLM image is present on both nodes"
}

run_launcher() {
  check_recipe_pin
  validate_profile
  [[ -x "${GENERATED_LAUNCHER}" ]] || die "Run configure before launching"
  (cd "${RECIPE_DIR}" && "${GENERATED_LAUNCHER}" "$@")
}

worker_hf_home() {
  local remote_home
  if [[ -n "${WORKER_HF_HOME}" ]]; then
    printf '%s' "${WORKER_HF_HOME}"
    return
  fi
  if [[ "${HF_HOME}" == "${HOME}" || "${HF_HOME}" == "${HOME}/"* ]]; then
    remote_home="$(wrun 'printf %s "$HOME"')"
    printf '%s%s' "${remote_home}" "${HF_HOME#${HOME}}"
  else
    printf '%s' "${HF_HOME}"
  fi
}

verify_model_snapshot() {
  local repo_name head_snapshot worker_snapshot
  repo_name="models--${MODEL_ID%%/*}--${MODEL_ID##*/}"
  head_snapshot="${HF_HOME}/hub/${repo_name}/snapshots/${MODEL_REVISION}"
  worker_snapshot="$(worker_hf_home)/hub/${repo_name}/snapshots/${MODEL_REVISION}"
  [[ -f "${head_snapshot}/config.json" ]] \
    || die "Pinned model snapshot is incomplete on head: ${head_snapshot}"
  wrun "test -f '${worker_snapshot}/config.json'" \
    || die "Pinned model snapshot is incomplete on worker: ${worker_snapshot}"
  log "Verified pinned model snapshot on both nodes: ${MODEL_REVISION}"
}

download_model() {
  run_launcher --no-launch
  verify_model_snapshot
}

check_public_exposure() {
  local funnel_status
  command -v tailscale >/dev/null 2>&1 || return 0
  funnel_status="$(tailscale funnel status 2>/dev/null || true)"
  if [[ -z "${funnel_status}" ]] && command -v sudo >/dev/null 2>&1; then
    funnel_status="$(sudo -n tailscale funnel status 2>/dev/null || true)"
  fi
  if grep -Eq "proxy http://(127\\.0\\.0\\.1|localhost):${PORT}([ /]|$)" <<<"${funnel_status}"; then
    die "Tailscale Funnel targets raw vLLM port ${PORT}; point it to ${DSPARK_PROXY_PORT}"
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
  check_host
  check_public_exposure
  if raw_model_ready; then
    log "Pinned Qwen vLLM model is already ready on the private raw endpoint"
  else
    run_proxy_cli stop || true
    stop_raw_containers
    drop_page_caches
    check_launch_memory
    if command -v timeout >/dev/null 2>&1; then
      (cd "${RECIPE_DIR}" && timeout "${WAIT_TIMEOUT_MIN}m" "${GENERATED_LAUNCHER}" --launch) \
        || { stop_raw_containers; die "MiaAI vLLM launcher failed or timed out"; }
    else
      run_launcher --launch
    fi
    if ! raw_model_ready; then
      stop_raw_containers
      die "MiaAI launcher returned without the pinned Qwen model ready"
    fi
  fi
  check_runtime_memory_headroom
  run_proxy_cli serve
  if ! run_proxy_cli smoke; then
    run_proxy_cli stop || true
    die "Safety proxy checks failed; raw vLLM remains private on 127.0.0.1:${PORT}"
  fi
  log "Public authenticated API: http://127.0.0.1:${DSPARK_PROXY_PORT}/v1"
  log "Raw unauthenticated API:  http://127.0.0.1:${PORT}/v1 (loopback only)"
}

show_status() {
  printf '%s\n' 'Head container:'
  docker ps -a --filter name=vllm-fn --format '  {{.Names}}  {{.Status}}' || true
  printf '%s\n' 'Worker container:'
  wrun "docker ps -a --filter name=vllm-fn --format '  {{.Names}}  {{.Status}}'" || true
  if raw_model_ready; then log "Raw API ready: http://127.0.0.1:${PORT}/v1"; else warn "Raw API not ready"; fi
  run_proxy_cli status
}

show_memory() {
  printf '%s\n' 'Head memory:'
  free -h
  nvidia-smi || true
  printf '\nWorker memory (%s):\n' "$(worker_target)"
  wrun 'free -h; nvidia-smi' || true
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
    -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Return only the number 391.\"}],\"temperature\":0,\"max_tokens\":256}" \
    -o "${response_file}"
  "${python}" - "${response_file}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
choices = payload.get("choices") or []
message = choices[0].get("message", {}) if choices else {}
if not (message.get("content") or message.get("reasoning_content")):
    raise SystemExit(f"invalid chat response: {payload}")
print("Qwen3.8 Flash Next vLLM completion passed")
PY
)

stop_service() {
  load_profile
  run_proxy_cli stop || true
  stop_raw_containers
  log "Stopped Qwen vLLM service; weights, image, patch, and JIT caches were preserved"
}

action="${1:-help}"
case "${action}" in
  bootstrap) bootstrap ;;
  configure) configure ;;
  check) check_host ;;
  setup) bootstrap; configure; check_host ;;
  pull) pull_images ;;
  download) download_model ;;
  start) start_service ;;
  status) load_profile; show_status ;;
  memory) load_profile; show_memory ;;
  smoke) load_profile; inference_smoke ;;
  logs) exec docker logs -f vllm-fn ;;
  logs-worker) load_profile; exec ssh -t "$(worker_target)" docker logs -f vllm-fn ;;
  stop) stop_service ;;
  update) sync_recipe_pin; configure ;;
  all) bootstrap; configure; check_host; pull_images; download_model; start_service ;;
  path)
    load_profile
    printf 'recipe=%s\nprofile=%s\nhf_home=%s\nupstream_revision=%s\nmodel_revision=%s\nimage=%s\n' \
      "${RECIPE_DIR}" "${PROFILE_FILE}" "${HF_HOME}" "${UPSTREAM_REVISION}" \
      "${MODEL_REVISION}" "${VLLM_IMAGE}"
    ;;
  help|-h|--help) usage ;;
  *) die "Unknown action: ${action}. Run qwen38-flash-next-vllm help" ;;
esac
