#!/usr/bin/env bash
# GLM-5.3 Flash NVFP4 on two linked DGX Sparks (GB10 / SM121).
#
# The model currently requires its dedicated arm64 vLLM image. This lifecycle
# follows vLLM/NVIDIA's two-node Ray pattern, pins the image and model revision,
# mirrors the checkpoint, and keeps the raw API behind ml-compute's proxy.

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE_FILE_DEFAULT="${PROJECT_ROOT}/config/dspark-glm53-flash-nvfp4.env"
RUNTIME_DIR_DEFAULT="${PROJECT_ROOT}/data/dspark/glm53-flash"
PROJECT_ENV_FILE_DEFAULT="${PROJECT_ROOT}/.env.local"

MODEL_ID_DEFAULT="LibertAIDAI/GLM-5.3-Flash-NVFP4"
MODEL_REVISION_DEFAULT="11d73216cd636238e82e1d77fe1042ffab36e7fa"
VLLM_IMAGE_DEFAULT="vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce"
VLLM_CLUSTER_REFERENCE="51c1ee9b7c8acbba4899a8ebffd390685d171946"

PROFILE_FILE="${GLM53_DSPARK_CONFIG_FILE:-${PROFILE_FILE_DEFAULT}}"
RUNTIME_DIR="${GLM53_DSPARK_RUNTIME_DIR:-${RUNTIME_DIR_DEFAULT}}"
PROJECT_ENV_FILE="${GLM53_DSPARK_PROJECT_ENV_FILE:-${PROJECT_ENV_FILE_DEFAULT}}"
LOG_DIR="${RUNTIME_DIR}/logs"
VLLM_LOG="${LOG_DIR}/vllm.log"
RESOLVED_ENV="${RUNTIME_DIR}/.env.glm53"

log() { printf '[glm53-flash] %s\n' "$*"; }
warn() { printf '[glm53-flash] WARNING: %s\n' "$*" >&2; }
die() { printf '[glm53-flash] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
GLM-5.3 Flash NVFP4 — dual-DGX-Spark vLLM/Ray lifecycle

Run on the head/rank-0 Spark:

  python3 -m ml.cli glm53-flash setup
  python3 -m ml.cli glm53-flash pull
  python3 -m ml.cli glm53-flash download
  scripts/start-GLM53-Flash-Dual-DSpark.sh
  python3 -m ml.cli glm53-flash smoke

Actions:
  configure    Validate and materialize the resolved ml-compute profile
  check        Validate both GB10 nodes, RoCE/SSH, memory, disk, and API key
  setup        Run configure and check
  pull         Pull the digest-pinned dedicated vLLM image on both nodes
  download     Download the pinned ~181 GiB snapshot and rsync it to the worker
  gpu-check    Initialize CUDA and import vLLM in the image on both nodes
  start        Start Ray, launch vLLM TP=2, then expose the safety proxy
  status       Show Ray containers, cluster resources, raw API, and proxy state
  memory       Show shared-memory and GPU use on both nodes
  smoke        Verify proxy safety and perform one GLM completion
  logs         Follow the head vLLM server log
  logs-worker  Follow the worker Ray container log
  stop         Stop the proxy and both Ray containers; preserve caches
  all          Run setup, pull, download, gpu-check, and start
  path         Print profile, runtime, cache, log, and pinned revisions
  help         Show this help

Reviewed bring-up profile:
  - two GB10 nodes, Ray + vLLM tensor parallelism TP=2
  - 32K context, one sequence, 0.84 UMA fraction per node
  - Marlin MoE fallback + eager execution for SM121 compatibility
  - model-native glm47 tools, glm45 reasoning, in-checkpoint MTP=5
  - raw unauthenticated API on 127.0.0.1:8888 only
  - authenticated allow-list proxy on 0.0.0.0:8000

This is intentionally marked experimental: the checkpoint fits in two Sparks,
but the publisher does not list GB10/SM121 among verified runtime targets.
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
RAY_HEAD_PORT|6379
GLM53_HEAD_CONTAINER|glm53-flash-ray-head
GLM53_WORKER_CONTAINER|glm53-flash-ray-worker
HOST_BIND|127.0.0.1
PORT|8888
DSPARK_PROXY_HOST|0.0.0.0
DSPARK_PROXY_PORT|8000
MODEL_ID|${MODEL_ID_DEFAULT}
MODEL_REVISION|${MODEL_REVISION_DEFAULT}
SERVED_MODEL_NAME|GLM-5.3-Flash-NVFP4
VLLM_IMAGE|${VLLM_IMAGE_DEFAULT}
TENSOR_PARALLEL_SIZE|2
MAX_MODEL_LEN|32768
MAX_NUM_SEQS|1
MAX_NUM_BATCHED_TOKENS|4096
GPU_MEMORY_UTILIZATION|0.84
KV_CACHE_DTYPE|auto
MOE_BACKEND|marlin
ENFORCE_EAGER|1
TOOL_CALL_PARSER|glm47
REASONING_PARSER|glm45
MTP_SPECULATIVE_TOKENS|5
HF_HOME|${RUNTIME_DIR}/cache/huggingface
WORKER_HF_HOME|
DOWNLOAD_MODE|rsync
VLLM_ENGINE_READY_TIMEOUT_S|3600
CPUSET|5-9,15-19
NCCL_DEBUG|WARN
NCCL_CROSS_NIC|0
GLM53_MIN_AVAILABLE_GIB|112
GLM53_MIN_DISK_GIB|240
EOF
}

resolved_keys() {
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
RAY_HEAD_PORT
GLM53_HEAD_CONTAINER
GLM53_WORKER_CONTAINER
HOST_BIND
PORT
DSPARK_PROXY_HOST
DSPARK_PROXY_PORT
MODEL_ID
MODEL_REVISION
SERVED_MODEL_NAME
VLLM_IMAGE
TENSOR_PARALLEL_SIZE
MAX_MODEL_LEN
MAX_NUM_SEQS
MAX_NUM_BATCHED_TOKENS
GPU_MEMORY_UTILIZATION
KV_CACHE_DTYPE
MOE_BACKEND
ENFORCE_EAGER
TOOL_CALL_PARSER
REASONING_PARSER
MTP_SPECULATIVE_TOKENS
HF_HOME
WORKER_HF_HOME
DOWNLOAD_MODE
VLLM_ENGINE_READY_TIMEOUT_S
CPUSET
NCCL_DEBUG
NCCL_CROSS_NIC
GLM53_MIN_AVAILABLE_GIB
GLM53_MIN_DISK_GIB
EOF
}

validate_profile() {
  load_profile
  [[ "${HOST_BIND}" == "127.0.0.1" ]] \
    || die "HOST_BIND must be 127.0.0.1; expose only the authenticated safety proxy"
  [[ "${TENSOR_PARALLEL_SIZE}" == "2" ]] \
    || die "This lifecycle is specifically reviewed for two Sparks and TP=2"
  [[ "${MAX_NUM_SEQS}" == "1" ]] \
    || die "MAX_NUM_SEQS must remain 1 for the two-Spark bring-up profile"
  (( MAX_MODEL_LEN <= 32768 )) \
    || die "MAX_MODEL_LEN above 32768 is not validated with this 181 GiB checkpoint on two Sparks"
  awk -v value="${GPU_MEMORY_UTILIZATION}" 'BEGIN {exit !(value <= 0.84)}' \
    || die "GPU_MEMORY_UTILIZATION above 0.84 is not safe for the reviewed GB10 profile"
  [[ "${MOE_BACKEND}" == "marlin" ]] \
    || die "MOE_BACKEND must remain marlin until native GLM FP4 MoE kernels are verified on SM121"
  [[ "${ENFORCE_EAGER}" == "1" ]] \
    || die "ENFORCE_EAGER must remain enabled for the conservative SM121 profile"
  (( MTP_SPECULATIVE_TOKENS >= 0 && MTP_SPECULATIVE_TOKENS <= 5 )) \
    || die "MTP_SPECULATIVE_TOKENS must be between 0 and the published value 5"
  [[ "${MODEL_ID}" == "${MODEL_ID_DEFAULT}" ]] \
    || warn "MODEL_ID differs from the reviewed GLM checkpoint"
  [[ "${MODEL_REVISION}" == "${MODEL_REVISION_DEFAULT}" ]] \
    || warn "MODEL_REVISION differs from the reviewed checkpoint pin"
  [[ "${VLLM_IMAGE}" == "${VLLM_IMAGE_DEFAULT}" ]] \
    || warn "VLLM_IMAGE differs from the reviewed dedicated arm64 manifest"
}

configure() {
  validate_profile
  mkdir -p "${RUNTIME_DIR}" "${LOG_DIR}" "${HF_HOME}"
  local env_tmp key
  env_tmp="$(mktemp "${RUNTIME_DIR}/.env.glm53.XXXXXX")"
  {
    printf '# Generated by ml-compute; edit %s instead.\n' "${PROFILE_FILE}"
    while IFS= read -r key; do
      [[ -n "${key}" ]] || continue
      printf '%s=%s\n' "${key}" "${!key}"
    done < <(resolved_keys)
  } >"${env_tmp}"
  mv "${env_tmp}" "${RESOLVED_ENV}"
  log "Validated and materialized profile at ${RESOLVED_ENV}"
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

worker_ssh_target() {
  if [[ -n "${WORKER_SSH}" ]]; then
    printf '%s' "${WORKER_SSH}"
  elif [[ -n "${WORKER_USER}" ]]; then
    printf '%s@%s' "${WORKER_USER}" "${WORKER_HOST}"
  else
    printf '%s' "${WORKER_HOST}"
  fi
}

wrun() {
  local target
  target="$(worker_ssh_target)"
  ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o BatchMode=yes \
    "${target}" "$1"
}

worker_docker() {
  local quoted=() arg
  for arg in "$@"; do
    quoted+=("$(printf '%q' "${arg}")")
  done
  wrun "docker ${quoted[*]}"
}

remote_home() {
  wrun 'printf %s "$HOME"'
}

worker_hf_home() {
  if [[ -n "${WORKER_HF_HOME}" ]]; then
    printf '%s' "${WORKER_HF_HOME}"
  else
    printf '%s/.cache/huggingface' "$(remote_home)"
  fi
}

model_cache_name() {
  printf 'models--%s' "${MODEL_ID//\//--}"
}

snapshot_path() {
  printf '%s/hub/%s/snapshots/%s' "${HF_HOME}" "$(model_cache_name)" "${MODEL_REVISION}"
}

worker_snapshot_path() {
  printf '%s/hub/%s/snapshots/%s' "$(worker_hf_home)" "$(model_cache_name)" "${MODEL_REVISION}"
}

container_snapshot_path() {
  printf '/root/.cache/huggingface/hub/%s/snapshots/%s' \
    "$(model_cache_name)" "${MODEL_REVISION}"
}

verify_snapshot() {
  local snapshot="$1" label="$2" count
  [[ -f "${snapshot}/config.json" ]] || die "${label}: missing config.json in pinned snapshot"
  [[ -f "${snapshot}/model.safetensors.index.json" ]] \
    || die "${label}: missing model.safetensors.index.json"
  count="$(find -L "${snapshot}" -maxdepth 1 -type f -name '*.safetensors' | wc -l | tr -d ' ')"
  (( count >= 120 )) || die "${label}: only ${count}/120 safetensor shards are present"
  log "${label}: pinned snapshot verified (${count} shards)"
}

verify_worker_snapshot() {
  local snapshot count
  snapshot="$(worker_snapshot_path)"
  wrun "test -f '${snapshot}/config.json' && test -f '${snapshot}/model.safetensors.index.json'" \
    || return 1
  count="$(wrun "find -L '${snapshot}' -maxdepth 1 -type f -name '*.safetensors' | wc -l" | tr -d ' ')"
  (( count >= 120 ))
}

check_disk() {
  local head_free worker_free worker_cache required
  mkdir -p "${HF_HOME}"
  head_free="$(( $(df -Pk "${HF_HOME}" | awk 'NR==2 {print $4}') / 1024 / 1024 ))"
  worker_cache="$(worker_hf_home)"
  wrun "mkdir -p '${worker_cache}'"
  worker_free="$(wrun "df -Pk '${worker_cache}'" | awk 'NR==2 {print $4}')"
  worker_free="$(( worker_free / 1024 / 1024 ))"
  required="${GLM53_MIN_DISK_GIB}"
  (( head_free >= required )) \
    || die "Head has ${head_free} GiB free under ${HF_HOME}; first download requires ${required} GiB"
  (( worker_free >= required )) \
    || die "Worker has ${worker_free} GiB free under ${worker_cache}; mirror requires ${required} GiB"
  log "Disk free: head ${head_free} GiB, worker ${worker_free} GiB"
}

available_gib() {
  awk '/^MemAvailable:/ {printf "%.3f", $2 / 1048576}' /proc/meminfo
}

check_launch_memory() {
  local head_available worker_available required
  head_available="$(available_gib)"
  worker_available="$(wrun "awk '/^MemAvailable:/ {printf \"%.3f\", \$2 / 1048576}' /proc/meminfo")"
  required="${GLM53_MIN_AVAILABLE_GIB}"
  awk -v have="${head_available}" -v need="${required}" 'BEGIN {exit !(have >= need)}' \
    || die "Head has ${head_available} GiB MemAvailable; cold launch requires ${required} GiB"
  awk -v have="${worker_available}" -v need="${required}" 'BEGIN {exit !(have >= need)}' \
    || die "Worker has ${worker_available} GiB MemAvailable; cold launch requires ${required} GiB"
  log "Cold-start memory: head ${head_available} GiB, worker ${worker_available} GiB"
}

check_host() {
  validate_profile
  need_cmd docker
  need_cmd curl
  need_cmd rsync
  need_cmd ssh
  need_cmd nvidia-smi
  [[ "$(uname -m)" == "aarch64" ]] \
    || die "The pinned runtime is arm64-only; head reports $(uname -m)"
  docker info >/dev/null 2>&1 || die "Docker daemon is unavailable on the head"
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q GB10 \
    || die "NVIDIA GB10 was not detected on the head"
  wrun "test \"\$(uname -m)\" = aarch64" \
    || die "Worker is not aarch64"
  worker_docker info >/dev/null 2>&1 || die "Docker daemon is unavailable on the worker"
  wrun "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q GB10" \
    || die "NVIDIA GB10 was not detected on the worker"
  ping -c 1 -W 2 "${WORKER_CX7_IP}" >/dev/null 2>&1 \
    || die "Direct RoCE address ${WORKER_CX7_IP} is unreachable from the head"
  check_disk
  check_proxy_key
  log "Dual-Spark GLM preflight passed"
}

ensure_image() {
  validate_profile
  if docker image inspect "${VLLM_IMAGE}" >/dev/null 2>&1; then
    log "Dedicated GLM image present on head"
  else
    docker pull "${VLLM_IMAGE}"
  fi
  if worker_docker image inspect "${VLLM_IMAGE}" >/dev/null 2>&1; then
    log "Dedicated GLM image present on worker"
  else
    worker_docker pull "${VLLM_IMAGE}"
  fi
}

gpu_check() {
  ensure_image
  docker run --rm --gpus all --entrypoint python3 "${VLLM_IMAGE}" -c \
    'import ray, torch, vllm; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), vllm.__version__, ray.__version__)'
  worker_docker run --rm --gpus all --entrypoint python3 "${VLLM_IMAGE}" -c \
    'import ray, torch, vllm; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), vllm.__version__, ray.__version__)'
}

download_on_head() {
  local snapshot
  snapshot="$(snapshot_path)"
  if [[ -d "${snapshot}" ]]; then
    verify_snapshot "${snapshot}" "head cache"
    return
  fi
  mkdir -p "${HF_HOME}"
  log "Downloading ${MODEL_ID}@${MODEL_REVISION} (~181 GiB)"
  if command -v hf >/dev/null 2>&1; then
    HF_HOME="${HF_HOME}" hf download "${MODEL_ID}" --revision "${MODEL_REVISION}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_HOME="${HF_HOME}" huggingface-cli download "${MODEL_ID}" --revision "${MODEL_REVISION}"
  else
    docker run --rm --network host --entrypoint python3 \
      -e HF_HOME=/root/.cache/huggingface -e HF_TOKEN="${HF_TOKEN:-}" \
      -v "${HF_HOME}:/root/.cache/huggingface" \
      "${VLLM_IMAGE}" -c \
      "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}', revision='${MODEL_REVISION}', token=None)"
  fi
  verify_snapshot "${snapshot}" "head cache"
}

sync_weights_to_worker() {
  local head_repo worker_repo worker_cache
  if verify_worker_snapshot; then
    log "Worker cache: pinned snapshot already verified"
    return
  fi
  [[ "${DOWNLOAD_MODE}" == "rsync" ]] \
    || die "Only DOWNLOAD_MODE=rsync is reviewed for this two-Spark recipe"
  head_repo="${HF_HOME}/hub/$(model_cache_name)"
  worker_cache="$(worker_hf_home)"
  worker_repo="${worker_cache}/hub/$(model_cache_name)"
  wrun "mkdir -p '${worker_repo}'"
  log "Mirroring the pinned GLM cache to the worker over RoCE (resumable)"
  rsync -a --partial --info=progress2,stats1 --exclude '.locks' \
    -e 'ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o BatchMode=yes' \
    "${head_repo}/" "$(worker_ssh_target):${worker_repo}/"
  verify_worker_snapshot || die "Worker snapshot verification failed after rsync"
  log "Worker cache: pinned snapshot verified"
}

download_model() {
  validate_profile
  check_disk
  ensure_image
  download_on_head
  sync_weights_to_worker
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

check_public_exposure() {
  local funnel_status
  command -v tailscale >/dev/null 2>&1 || return 0
  funnel_status="$(tailscale funnel status 2>/dev/null || true)"
  if [[ -z "${funnel_status}" ]] && command -v sudo >/dev/null 2>&1; then
    funnel_status="$(sudo -n tailscale funnel status 2>/dev/null || true)"
  fi
  if grep -Eq "proxy http://(127\\.0\\.0\\.1|localhost):${PORT}([ /]|$)" <<<"${funnel_status}"; then
    die "Tailscale Funnel targets unauthenticated raw port ${PORT}. Point it to ${DSPARK_PROXY_PORT} instead"
  fi
}

raw_model_ready() {
  local python
  python="$(project_python)"
  curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null \
    | EXPECTED_MODEL="${SERVED_MODEL_NAME}" "${python}" -c \
      'import json, os, sys; data=json.load(sys.stdin); ids={m.get("id") for m in data.get("data", [])}; raise SystemExit(0 if os.environ["EXPECTED_MODEL"] in ids else 1)' \
    >/dev/null 2>&1
}

stop_ray_cluster() {
  docker rm -f "${GLM53_HEAD_CONTAINER}" >/dev/null 2>&1 || true
  worker_docker rm -f "${GLM53_WORKER_CONTAINER}" >/dev/null 2>&1 || true
}

check_port_available() {
  local host="$1" port="$2" label="$3"
  if (exec 3<>"/dev/tcp/${host}/${port}") >/dev/null 2>&1; then
    die "${label} port ${host}:${port} is already in use; stop the active model backend first"
  fi
}

start_ray_cluster() {
  local worker_cache pin=()
  worker_cache="$(worker_hf_home)"
  [[ -n "${CPUSET}" ]] && pin=(--cpuset-cpus "${CPUSET}")
  mkdir -p "${LOG_DIR}" "${HF_HOME}"
  wrun "mkdir -p '${worker_cache}'"

  log "Starting Ray head on ${HEAD_CX7_IP}:${RAY_HEAD_PORT}"
  docker run -d \
    --name "${GLM53_HEAD_CONTAINER}" \
    --entrypoint /bin/bash \
    --network host --ipc host --gpus all \
    --device /dev/infiniband --cap-add IPC_LOCK \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    "${pin[@]}" \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    -v "${LOG_DIR}:/ml-compute-logs" \
    -e VLLM_HOST_IP="${HEAD_CX7_IP}" \
    -e UCX_NET_DEVICES="${HEAD_CX7_IF}" \
    -e NCCL_SOCKET_IFNAME="${HEAD_CX7_IF}" \
    -e OMPI_MCA_btl_tcp_if_include="${HEAD_CX7_IF}" \
    -e GLOO_SOCKET_IFNAME="${HEAD_CX7_IF}" \
    -e TP_SOCKET_IFNAME="${HEAD_CX7_IF}" \
    -e NCCL_IB_HCA="${HEAD_CX7_IB}" \
    -e NCCL_IB_DISABLE=0 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_NVLS_ENABLE=0 \
    -e NCCL_CUMEM_ENABLE=0 -e NCCL_CROSS_NIC="${NCCL_CROSS_NIC}" \
    -e NCCL_DEBUG="${NCCL_DEBUG}" -e RAY_memory_monitor_refresh_ms=0 \
    -e MASTER_ADDR="${HEAD_CX7_IP}" \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S}" \
    "${VLLM_IMAGE}" -lc \
    "exec ray start --block --head --node-ip-address=${HEAD_CX7_IP} --port=${RAY_HEAD_PORT} --include-dashboard=false" \
    >/dev/null

  sleep 3
  docker ps --format '{{.Names}}' | grep -qx "${GLM53_HEAD_CONTAINER}" \
    || die "Ray head container exited; inspect: docker logs ${GLM53_HEAD_CONTAINER}"

  log "Starting Ray worker on ${WORKER_CX7_IP}"
  worker_docker run -d \
    --name "${GLM53_WORKER_CONTAINER}" \
    --entrypoint /bin/bash \
    --network host --ipc host --gpus all \
    --device /dev/infiniband --cap-add IPC_LOCK \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    "${pin[@]}" \
    -v "${worker_cache}:/root/.cache/huggingface" \
    -e VLLM_HOST_IP="${WORKER_CX7_IP}" \
    -e UCX_NET_DEVICES="${WORKER_CX7_IF}" \
    -e NCCL_SOCKET_IFNAME="${WORKER_CX7_IF}" \
    -e OMPI_MCA_btl_tcp_if_include="${WORKER_CX7_IF}" \
    -e GLOO_SOCKET_IFNAME="${WORKER_CX7_IF}" \
    -e TP_SOCKET_IFNAME="${WORKER_CX7_IF}" \
    -e NCCL_IB_HCA="${WORKER_CX7_IB}" \
    -e NCCL_IB_DISABLE=0 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_NVLS_ENABLE=0 \
    -e NCCL_CUMEM_ENABLE=0 -e NCCL_CROSS_NIC="${NCCL_CROSS_NIC}" \
    -e NCCL_DEBUG="${NCCL_DEBUG}" -e RAY_memory_monitor_refresh_ms=0 \
    -e MASTER_ADDR="${HEAD_CX7_IP}" \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S}" \
    "${VLLM_IMAGE}" -lc \
    "exec ray start --block --address=${HEAD_CX7_IP}:${RAY_HEAD_PORT} --node-ip-address=${WORKER_CX7_IP}" \
    >/dev/null

  local attempt
  for attempt in $(seq 1 60); do
    if docker exec "${GLM53_HEAD_CONTAINER}" python3 -c \
      'import ray; ray.init(address="auto", logging_level="ERROR"); raise SystemExit(0 if ray.cluster_resources().get("GPU", 0) >= 2 else 1)' \
      >/dev/null 2>&1; then
      log "Ray cluster reports two GPUs"
      return
    fi
    sleep 2
  done
  die "Ray worker did not join within 120 seconds"
}

launch_vllm() {
  local command
  local args=(
    vllm serve "$(container_snapshot_path)"
    --served-model-name "${SERVED_MODEL_NAME}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --distributed-executor-backend ray
    --host "${HOST_BIND}"
    --port "${PORT}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --tool-call-parser "${TOOL_CALL_PARSER}"
    --enable-auto-tool-choice
    --reasoning-parser "${REASONING_PARSER}"
    --moe-backend "${MOE_BACKEND}"
    --trust-remote-code
  )
  [[ "${KV_CACHE_DTYPE}" != "auto" ]] && args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
  [[ "${ENFORCE_EAGER}" == "1" ]] && args+=(--enforce-eager)
  if (( MTP_SPECULATIVE_TOKENS > 0 )); then
    args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_SPECULATIVE_TOKENS}}")
  fi
  printf -v command '%q ' "${args[@]}"
  : >"${VLLM_LOG}"
  log "Launching GLM vLLM TP=2 through Ray"
  docker exec -d "${GLM53_HEAD_CONTAINER}" /bin/bash -lc \
    "exec ${command} >>/ml-compute-logs/vllm.log 2>&1"
}

wait_ready() {
  local deadline now
  deadline="$(( $(date +%s) + VLLM_ENGINE_READY_TIMEOUT_S ))"
  while :; do
    if raw_model_ready; then
      log "Raw GLM API is ready on ${HOST_BIND}:${PORT}"
      return
    fi
    if ! docker exec "${GLM53_HEAD_CONTAINER}" pgrep -f '[v]llm serve' >/dev/null 2>&1; then
      tail -n 160 "${VLLM_LOG}" || true
      die "vLLM exited before readiness"
    fi
    now="$(date +%s)"
    if (( now >= deadline )); then
      tail -n 160 "${VLLM_LOG}" || true
      die "vLLM did not become ready within ${VLLM_ENGINE_READY_TIMEOUT_S} seconds"
    fi
    sleep 15
  done
}

start_service() {
  validate_profile
  check_proxy_key
  check_public_exposure
  if raw_model_ready; then
    log "Pinned GLM model is already ready on the private raw endpoint"
  else
    verify_snapshot "$(snapshot_path)" "head cache"
    verify_worker_snapshot || die "Pinned worker snapshot is missing; run glm53-flash download"
    ensure_image
    stop_ray_cluster
    check_port_available "127.0.0.1" "${PORT}" "raw API"
    check_port_available "${HEAD_CX7_IP}" "${RAY_HEAD_PORT}" "Ray head"
    check_launch_memory
    start_ray_cluster
    launch_vllm
    wait_ready
  fi
  run_proxy_cli serve
  if ! run_proxy_cli smoke; then
    run_proxy_cli stop || true
    die "Safety proxy checks failed; raw vLLM remains private on 127.0.0.1:${PORT}"
  fi
  log "Public authenticated API: http://127.0.0.1:${DSPARK_PROXY_PORT}/v1"
  log "Raw unauthenticated API:  http://127.0.0.1:${PORT}/v1 (loopback only)"
}

show_status() {
  printf 'Head Ray container:\n'
  docker ps -a --filter "name=${GLM53_HEAD_CONTAINER}" --format '  {{.Names}}  {{.Status}}' || true
  printf 'Worker Ray container:\n'
  worker_docker ps -a --filter "name=${GLM53_WORKER_CONTAINER}" --format '  {{.Names}}  {{.Status}}' || true
  if docker ps --format '{{.Names}}' | grep -qx "${GLM53_HEAD_CONTAINER}"; then
    docker exec "${GLM53_HEAD_CONTAINER}" ray status 2>/dev/null || true
  fi
  if raw_model_ready; then
    log "Raw GLM API: ready"
  else
    warn "Raw GLM API: unavailable"
  fi
  run_proxy_cli status
}

show_memory() {
  local target
  target="$(worker_ssh_target)"
  printf '%s\n' 'Head memory:'
  free -h
  nvidia-smi || true
  printf '\nWorker memory (%s):\n' "${target}"
  wrun 'free -h; nvidia-smi' || true
  warn "The 181 GiB checkpoint is a dedicated two-Spark workload"
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
    -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Return only the number 391.\"}],\"temperature\":0,\"max_tokens\":64}" \
    -o "${response_file}"
  "${python}" - "${response_file}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
choices = payload.get("choices") or []
message = choices[0].get("message", {}) if choices else {}
if not (message.get("content") or message.get("reasoning_content")):
    raise SystemExit(f"invalid chat response: {payload}")
print("GLM-5.3 Flash model completion passed")
PY
)

stop_service() {
  load_profile
  run_proxy_cli stop || true
  stop_ray_cluster
  log "Stopped dual-Spark GLM service; weights, image, and logs were preserved"
}

action="${1:-help}"
case "${action}" in
  configure) configure ;;
  check) check_host ;;
  setup) configure; check_host ;;
  pull) ensure_image ;;
  download) download_model ;;
  gpu-check) gpu_check ;;
  start) start_service ;;
  status) load_profile; show_status ;;
  memory) load_profile; show_memory ;;
  smoke) load_profile; inference_smoke ;;
  logs) load_profile; touch "${VLLM_LOG}"; exec tail -n 100 -f "${VLLM_LOG}" ;;
  logs-worker) load_profile; exec ssh -t "$(worker_ssh_target)" docker logs -f "${GLM53_WORKER_CONTAINER}" ;;
  stop) stop_service ;;
  all) configure; check_host; ensure_image; download_model; gpu_check; start_service ;;
  path)
    load_profile
    printf 'profile=%s\nruntime=%s\nhf_home=%s\nlog=%s\nmodel_revision=%s\nimage=%s\nvllm_cluster_reference=%s\n' \
      "${PROFILE_FILE}" "${RUNTIME_DIR}" "${HF_HOME}" "${VLLM_LOG}" \
      "${MODEL_REVISION}" "${VLLM_IMAGE}" "${VLLM_CLUSTER_REFERENCE}"
    ;;
  help|-h|--help) usage ;;
  *) die "Unknown action: ${action}. Run glm53-flash help" ;;
esac
