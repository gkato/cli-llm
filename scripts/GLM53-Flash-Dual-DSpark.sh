#!/usr/bin/env bash
# GLM-5.3 Flash EXL3 + DFlash2 on two linked DGX Sparks (GB10 / SM121).
#
# Lifecycle adapter for MiaAI-Lab's validated vLLM/MP TP=2 recipe:
# https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
#
# The public ml-compute contract intentionally remains stable: the
# `glm53-flash` CLI, this script path, the legacy-named profile, the
# data/dspark/glm53-flash runtime tree, and the authenticated proxy are kept.
# Only the serving backend changes from NVFP4/Ray to EXL3/MP/DFlash2.

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE_FILE_DEFAULT="${PROJECT_ROOT}/config/dspark-glm53-flash-nvfp4.env"
RUNTIME_DIR_DEFAULT="${PROJECT_ROOT}/data/dspark/glm53-flash"
RECIPE_DIR_DEFAULT="${RUNTIME_DIR_DEFAULT}/miaai-exl3-dual-spark"
PROJECT_ENV_FILE_DEFAULT="${PROJECT_ROOT}/.env.local"

UPSTREAM_REPO_DEFAULT="https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks.git"
UPSTREAM_REVISION_DEFAULT="66e2643d612adb2dced7da230ce52b96fe7f82cc"
MODEL_ID_DEFAULT="brandonmusic/GLM-5.3-Flash-tr3-4bpw"
MODEL_REVISION_DEFAULT="5ab363a8dcf6405955fd5f99671e01a1c9fb124b"
DFLASH_MODEL_ID_DEFAULT="incoai/GLM-5.3-Flash-DFlash2"
DFLASH_MODEL_REVISION_DEFAULT="7d74cdd881ed7e32c31175984a67823127b66cfe"
VLLM_BASE_IMAGE_DEFAULT="vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce"
VLLM_SOURCE_IMAGE_DEFAULT="ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58"
VLLM_IMAGE_DEFAULT="ml-compute/glm53-flash-exl3:mp-dflash2-v1-66e2643"

PROFILE_FILE="${GLM53_DSPARK_CONFIG_FILE:-${PROFILE_FILE_DEFAULT}}"
RUNTIME_DIR="${GLM53_DSPARK_RUNTIME_DIR:-${RUNTIME_DIR_DEFAULT}}"
RECIPE_DIR="${GLM53_DSPARK_RECIPE_DIR:-${RECIPE_DIR_DEFAULT}}"
PROJECT_ENV_FILE="${GLM53_DSPARK_PROJECT_ENV_FILE:-${PROJECT_ENV_FILE_DEFAULT}}"
UPSTREAM_REPO="${GLM53_DSPARK_UPSTREAM_REPO:-${UPSTREAM_REPO_DEFAULT}}"
UPSTREAM_REVISION="${GLM53_DSPARK_UPSTREAM_REVISION:-${UPSTREAM_REVISION_DEFAULT}}"
RESOLVED_ENV="${RUNTIME_DIR}/.env.glm53"
UPSTREAM_ENV="${RECIPE_DIR}/.env"

log() { printf '[glm53-flash] %s\n' "$*"; }
warn() { printf '[glm53-flash] WARNING: %s\n' "$*" >&2; }
die() { printf '[glm53-flash] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
GLM-5.3 Flash EXL3 + DFlash2 — MiaAI dual-DGX-Spark vLLM/MP lifecycle

Run on the head/rank-0 Spark:

  python3 -m ml.cli glm53-flash setup
  python3 -m ml.cli glm53-flash pull
  python3 -m ml.cli glm53-flash download
  scripts/start-GLM53-Flash-Dual-DSpark.sh
  python3 -m ml.cli glm53-flash smoke

Actions:
  bootstrap    Clone and detach the reviewed MiaAI-Lab upstream revision
  configure    Validate and materialize the resolved ml-compute profile
  check        Validate pins, both GB10 nodes, memory, disk, and API key
  setup        Run bootstrap, configure, and check
  pull         Pull the digest-pinned EXL3 image and ship its local tag
  download     Download pinned EXL3 + DFlash2 snapshots and mirror them
  gpu-check    Run MiaAI's EXL3/SM121/DFlash2 GPU self-check on both nodes
  start        Run worker-first MP TP=2, then the authenticated safety proxy
  status       Show both containers, raw API, and proxy state
  diagnose     Print head and worker container diagnostics
  memory       Show shared-memory and GPU use on both nodes
  smoke        Verify proxy safety and perform one GLM completion
  logs         Follow the head container log
  logs-worker  Follow the worker container log
  stop         Stop the proxy and both containers; preserve caches
  update       Fetch and re-checkout the revision pinned by this script
  all          Run setup, download, pull, gpu-check, and start
  path         Print checkout, profile, cache, images, and revisions
  help         Show this help

Reviewed MiaAI profile:
  - fused EXL3/TR3 K4 routed experts with native SM121 cubins
  - direct vLLM multiprocessing executor, two nodes, TP=2
  - DFlash2 k=7 on draft TP=1; MTP k=2 remains the rollback mode
  - 900K context, 4 sequences, 1024-token prefill chunks, FP8 MLA KV
  - CUDA graphs, prefix caching, image/video, glm47 tools, glm45 reasoning
  - raw unauthenticated API on 127.0.0.1:8888 only
  - authenticated allow-list proxy on 0.0.0.0:8000

The DFlash2 checkpoint is CC BY-NC-ND 4.0. For a commercial deployment,
review its license and use SPEC_METHOD=mtp unless separate permission exists.
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
MASTER_PORT|29521
GLM53_HEAD_CONTAINER|glm53-flash-head
GLM53_WORKER_CONTAINER|glm53-flash-worker
HOST_BIND|127.0.0.1
PORT|8888
DSPARK_PROXY_HOST|0.0.0.0
DSPARK_PROXY_PORT|8000
MODEL_ID|${MODEL_ID_DEFAULT}
MODEL_REVISION|${MODEL_REVISION_DEFAULT}
DFLASH_MODEL_ID|${DFLASH_MODEL_ID_DEFAULT}
DFLASH_MODEL_REVISION|${DFLASH_MODEL_REVISION_DEFAULT}
SERVED_MODEL_NAME|GLM-5.3-Flash-EXL3
VLLM_BASE_IMAGE|${VLLM_BASE_IMAGE_DEFAULT}
VLLM_SOURCE_IMAGE|${VLLM_SOURCE_IMAGE_DEFAULT}
VLLM_IMAGE|${VLLM_IMAGE_DEFAULT}
TENSOR_PARALLEL_SIZE|2
NUM_NODES|2
DISTRIBUTED_EXECUTOR_BACKEND|mp
QUANTIZATION|exl3
MAX_MODEL_LEN|900000
MAX_NUM_SEQS|4
MAX_NUM_BATCHED_TOKENS|1024
GPU_MEMORY_UTILIZATION|0.87
KV_CACHE_DTYPE|fp8
ENFORCE_EAGER|0
EXL3_FUSED_MOE|1
ENABLE_PREFIX_CACHING|1
SPEC_METHOD|dflash
DFLASH_SPECULATIVE_TOKENS|7
DFLASH_DRAFT_TP|1
MTP_SPECULATIVE_TOKENS|2
TOOL_CALL_PARSER|glm47
REASONING_PARSER|glm45
LANGUAGE_MODEL_ONLY|0
LIMIT_MM|{"image":4,"video":1}
SKIP_MM_PROFILING|1
TORCH_CUDA_ARCH_LIST|12.1a
FLASHINFER_CUDA_ARCH_LIST|12.1a
HF_HOME|${RUNTIME_DIR}/cache/huggingface
WORKER_HF_HOME|
DOWNLOAD_MODE|rsync
VLLM_ENGINE_READY_TIMEOUT_S|3600
CACHE_ROOT|${RUNTIME_DIR}/cache/vllm
WORKER_VLLM_CACHE|
USE_HOST_NCCL|0
NCCL_HOST_DIR|
WORKER_NCCL_HOST_DIR|
NCCL_SO_NAME|libnccl.so.2.30.7
NCCL_DEBUG|WARN
NCCL_IB_GID_INDEX|3
NCCL_CROSS_NIC|0
EXTRA_ARGS|
GLM53_MIN_AVAILABLE_GIB|112
GLM53_MIN_DISK_GIB|220
EOF

  if [[ "${HF_HOME}" != /* ]]; then
    HF_HOME="${PROJECT_ROOT}/${HF_HOME#./}"
    export HF_HOME
  fi
  if [[ "${CACHE_ROOT}" != /* ]]; then
    CACHE_ROOT="${PROJECT_ROOT}/${CACHE_ROOT#./}"
    export CACHE_ROOT
  fi
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
MASTER_PORT
GLM53_HEAD_CONTAINER
GLM53_WORKER_CONTAINER
HOST_BIND
PORT
DSPARK_PROXY_HOST
DSPARK_PROXY_PORT
MODEL_ID
MODEL_REVISION
DFLASH_MODEL_ID
DFLASH_MODEL_REVISION
SERVED_MODEL_NAME
VLLM_BASE_IMAGE
VLLM_SOURCE_IMAGE
VLLM_IMAGE
TENSOR_PARALLEL_SIZE
NUM_NODES
DISTRIBUTED_EXECUTOR_BACKEND
QUANTIZATION
MAX_MODEL_LEN
MAX_NUM_SEQS
MAX_NUM_BATCHED_TOKENS
GPU_MEMORY_UTILIZATION
KV_CACHE_DTYPE
ENFORCE_EAGER
EXL3_FUSED_MOE
ENABLE_PREFIX_CACHING
SPEC_METHOD
DFLASH_SPECULATIVE_TOKENS
DFLASH_DRAFT_TP
MTP_SPECULATIVE_TOKENS
TOOL_CALL_PARSER
REASONING_PARSER
LANGUAGE_MODEL_ONLY
LIMIT_MM
SKIP_MM_PROFILING
TORCH_CUDA_ARCH_LIST
FLASHINFER_CUDA_ARCH_LIST
HF_HOME
WORKER_HF_HOME
DOWNLOAD_MODE
VLLM_ENGINE_READY_TIMEOUT_S
CACHE_ROOT
WORKER_VLLM_CACHE
USE_HOST_NCCL
NCCL_HOST_DIR
WORKER_NCCL_HOST_DIR
NCCL_SO_NAME
NCCL_DEBUG
NCCL_IB_GID_INDEX
NCCL_CROSS_NIC
EXTRA_ARGS
GLM53_MIN_AVAILABLE_GIB
GLM53_MIN_DISK_GIB
EOF
}

validate_profile() {
  load_profile
  [[ "${HOST_BIND}" == "127.0.0.1" ]] \
    || die "HOST_BIND must be 127.0.0.1; expose only the authenticated safety proxy"
  [[ "${TENSOR_PARALLEL_SIZE}" == "2" && "${NUM_NODES}" == "2" ]] \
    || die "The reviewed MiaAI lifecycle is exactly two Sparks with TP=2"
  [[ "${DISTRIBUTED_EXECUTOR_BACKEND}" == "mp" ]] \
    || die "DISTRIBUTED_EXECUTOR_BACKEND must remain mp"
  [[ "${QUANTIZATION}" == "exl3" ]] \
    || die "QUANTIZATION must remain exl3 for the packed TR3 checkpoint"
  (( MAX_MODEL_LEN > 0 && MAX_MODEL_LEN <= 900000 )) \
    || die "MAX_MODEL_LEN must be between 1 and MiaAI's reviewed 900000"
  (( MAX_NUM_SEQS > 0 && MAX_NUM_SEQS <= 4 )) \
    || die "MAX_NUM_SEQS must be between 1 and MiaAI's reviewed value 4"
  (( MAX_NUM_BATCHED_TOKENS > 0 && MAX_NUM_BATCHED_TOKENS <= 1024 )) \
    || die "MAX_NUM_BATCHED_TOKENS above 1024 can exhaust the GB10 indexer on long prefill"
  awk -v value="${GPU_MEMORY_UTILIZATION}" 'BEGIN {exit !(value > 0 && value <= 0.87)}' \
    || die "GPU_MEMORY_UTILIZATION must not exceed MiaAI's reviewed 0.87"
  [[ "${KV_CACHE_DTYPE}" == "fp8" ]] \
    || die "KV_CACHE_DTYPE must remain fp8 for packed fp8_ds_mla"
  [[ "${ENFORCE_EAGER}" == "0" ]] \
    || die "ENFORCE_EAGER must remain 0 so the reviewed CUDA graph path is used"
  [[ "${EXL3_FUSED_MOE}" == "1" ]] \
    || die "EXL3_FUSED_MOE must remain enabled for the reviewed decode path"
  [[ "${ENABLE_PREFIX_CACHING}" == "1" ]] \
    || die "ENABLE_PREFIX_CACHING must remain enabled"
  [[ "${TOOL_CALL_PARSER}" == "glm47" && "${REASONING_PARSER}" == "glm45" ]] \
    || die "The pinned launcher requires glm47 tools and glm45 reasoning"
  [[ "${SPEC_METHOD}" == "dflash" || "${SPEC_METHOD}" == "mtp" || "${SPEC_METHOD}" == "none" ]] \
    || die "SPEC_METHOD must be dflash, mtp, or none"
  if [[ "${SPEC_METHOD}" == "dflash" ]]; then
    [[ "${DFLASH_SPECULATIVE_TOKENS}" == "7" ]] \
      || die "DFlash2 is trained for block size 8 and requires exactly 7 draft tokens"
    [[ "${DFLASH_DRAFT_TP}" == "1" ]] \
      || die "DFLASH_DRAFT_TP must remain 1 to avoid CX7 traffic on each draft step"
  fi
  (( MTP_SPECULATIVE_TOKENS >= 0 && MTP_SPECULATIVE_TOKENS <= 2 )) \
    || die "MTP_SPECULATIVE_TOKENS must be between 0 and MiaAI's rollback value 2"
  [[ "${SKIP_MM_PROFILING}" == "1" ]] \
    || die "SKIP_MM_PROFILING must remain enabled; the maximum MM dummy profile OOMs UMA"
  [[ "${LANGUAGE_MODEL_ONLY}" == "0" || "${LANGUAGE_MODEL_ONLY}" == "1" ]] \
    || die "LANGUAGE_MODEL_ONLY must be 0 or 1"
  [[ "${LIMIT_MM}" == '{"image":4,"video":1}' ]] \
    || die 'LIMIT_MM must remain valid JSON: {"image":4,"video":1}'
  [[ "${MODEL_ID}" == "${MODEL_ID_DEFAULT}" && "${MODEL_REVISION}" == "${MODEL_REVISION_DEFAULT}" ]] \
    || die "MODEL_ID and MODEL_REVISION must remain on the measured EXL3 snapshot"
  [[ "${DFLASH_MODEL_ID}" == "${DFLASH_MODEL_ID_DEFAULT}" \
    && "${DFLASH_MODEL_REVISION}" == "${DFLASH_MODEL_REVISION_DEFAULT}" ]] \
    || die "DFlash2 model and revision must remain pinned"
  [[ "${SERVED_MODEL_NAME}" == "GLM-5.3-Flash-EXL3" ]] \
    || die "SERVED_MODEL_NAME must remain GLM-5.3-Flash-EXL3"
  [[ "${VLLM_BASE_IMAGE}" == "${VLLM_BASE_IMAGE_DEFAULT}" ]] \
    || die "VLLM_BASE_IMAGE must remain the reviewed publisher manifest"
  [[ "${VLLM_SOURCE_IMAGE}" == "${VLLM_SOURCE_IMAGE_DEFAULT}" ]] \
    || die "VLLM_SOURCE_IMAGE must remain digest-pinned"
  [[ "${VLLM_IMAGE}" == "${VLLM_IMAGE_DEFAULT}" ]] \
    || warn "VLLM_IMAGE differs from the revision-keyed local tag"
  [[ "${GLM53_HEAD_CONTAINER}" == "glm53-flash-head" \
    && "${GLM53_WORKER_CONTAINER}" == "glm53-flash-worker" ]] \
    || die "Container names must preserve the ml-compute dual-Spark lifecycle"
  [[ "${USE_HOST_NCCL}" == "0" ]] \
    || die "USE_HOST_NCCL must remain 0; a second NCCL preload conflicts with this image"
  [[ " ${EXTRA_ARGS} " != *" --host "* && " ${EXTRA_ARGS} " != *" --host="* ]] \
    || die "EXTRA_ARGS may not override the private raw host"
  [[ " ${EXTRA_ARGS} " != *" --port "* && " ${EXTRA_ARGS} " != *" --port="* ]] \
    || die "EXTRA_ARGS may not override the private raw port"
  [[ "${EXTRA_ARGS}" != *"--quantization"* ]] \
    || die "EXTRA_ARGS may not override EXL3 quantization"
}

warn_dflash_license() {
  if [[ "${SPEC_METHOD}" == "dflash" ]]; then
    warn "DFlash2 is CC BY-NC-ND 4.0; use SPEC_METHOD=mtp for commercial service without separate permission"
  fi
}

recipe_head() {
  git -C "${RECIPE_DIR}" rev-parse HEAD 2>/dev/null || true
}

check_recipe_pin() {
  [[ -d "${RECIPE_DIR}/.git" ]] \
    || die "Recipe is not bootstrapped. Run: python3 -m ml.cli glm53-flash bootstrap"
  local head
  head="$(recipe_head)"
  [[ "${head}" == "${UPSTREAM_REVISION}" ]] \
    || die "Upstream checkout is ${head:-unknown}, expected ${UPSTREAM_REVISION}; run glm53-flash update"
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
  log "Pinned MiaAI EXL3 upstream revision: ${UPSTREAM_REVISION}"
}

bootstrap() {
  need_cmd git
  if [[ -d "${RECIPE_DIR}/.git" ]]; then
    sync_recipe_pin
    return
  fi
  [[ ! -e "${RECIPE_DIR}" ]] \
    || die "Recipe path exists but is not a git checkout: ${RECIPE_DIR}"
  mkdir -p "$(dirname "${RECIPE_DIR}")"
  git clone --no-tags "${UPSTREAM_REPO}" "${RECIPE_DIR}"
  sync_recipe_pin
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
  ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -o BatchMode=yes \
    "${target}" "$1"
}

worker_docker() {
  local quoted=() arg
  for arg in "$@"; do
    quoted+=("$(printf '%q' "${arg}")")
  done
  wrun "docker ${quoted[*]}"
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
  head_netdev="$(local_netdev_for_ip "${HEAD_CX7_IP}")"
  worker_netdev="$(worker_netdev_for_ip "${WORKER_CX7_IP}")"
  [[ -n "${head_netdev}" ]] || die "No head interface owns ${HEAD_CX7_IP}"
  [[ -n "${worker_netdev}" ]] || die "No worker interface owns ${WORKER_CX7_IP}"

  head_hca="$(local_hca_for_netdev "${head_netdev}")"
  worker_hca="$(worker_hca_for_netdev "${worker_netdev}")"
  [[ -n "${head_hca}" ]] || die "No head RDMA HCA maps to ${head_netdev}"
  [[ -n "${worker_hca}" ]] || die "No worker RDMA HCA maps to ${worker_netdev}"

  if [[ "${HEAD_CX7_IF}" != "${head_netdev}" || "${HEAD_CX7_IB}" != "${head_hca}" ]]; then
    warn "Head RoCE mapping detected as ${head_netdev}/${head_hca}; profile had ${HEAD_CX7_IF}/${HEAD_CX7_IB}"
  fi
  if [[ "${WORKER_CX7_IF}" != "${worker_netdev}" || "${WORKER_CX7_IB}" != "${worker_hca}" ]]; then
    warn "Worker RoCE mapping detected as ${worker_netdev}/${worker_hca}; profile had ${WORKER_CX7_IF}/${WORKER_CX7_IB}"
  fi
  HEAD_CX7_IF="${head_netdev}"
  WORKER_CX7_IF="${worker_netdev}"
  HEAD_CX7_IB="${head_hca}"
  WORKER_CX7_IB="${worker_hca}"
  export HEAD_CX7_IF WORKER_CX7_IF HEAD_CX7_IB WORKER_CX7_IB
  log "RoCE mapping: head ${HEAD_CX7_IP}=${HEAD_CX7_IF}/${HEAD_CX7_IB}, worker ${WORKER_CX7_IP}=${WORKER_CX7_IF}/${WORKER_CX7_IB}"
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

worker_home() {
  local cache
  cache="$(worker_hf_home)"
  if [[ "${cache}" == */.cache/huggingface ]]; then
    printf '%s' "${cache%/.cache/huggingface}"
  else
    die "WORKER_HF_HOME must end in /.cache/huggingface for MiaAI's mounted-cache layout"
  fi
}

worker_vllm_cache() {
  if [[ -n "${WORKER_VLLM_CACHE}" ]]; then
    printf '%s' "${WORKER_VLLM_CACHE}"
  else
    printf '%s/.cache/vllm-glm53-flash' "$(worker_home)"
  fi
}

model_cache_name() {
  local repo_id="$1"
  printf 'models--%s' "${repo_id//\//--}"
}

snapshot_path() {
  local repo_id="$1" revision="$2"
  printf '%s/hub/%s/snapshots/%s' "${HF_HOME}" "$(model_cache_name "${repo_id}")" "${revision}"
}

worker_snapshot_path() {
  local repo_id="$1" revision="$2"
  printf '%s/hub/%s/snapshots/%s' "$(worker_hf_home)" "$(model_cache_name "${repo_id}")" "${revision}"
}

verify_target_snapshot() {
  local snapshot="$1" label="$2" count
  [[ -f "${snapshot}/config.json" ]] || die "${label}: missing config.json"
  [[ -f "${snapshot}/model.safetensors.index.json" ]] || die "${label}: missing model index"
  count="$(find -L "${snapshot}" -maxdepth 1 -type f -name '*.safetensors' | wc -l | tr -d ' ')"
  (( count >= 120 )) || die "${label}: only ${count}/120 target shards are present"
  log "${label}: EXL3 snapshot verified (${count} shards)"
}

verify_dflash_snapshot() {
  local snapshot="$1" label="$2"
  [[ -f "${snapshot}/config.json" ]] || die "${label}: missing DFlash2 config.json"
  [[ -f "${snapshot}/model.safetensors" ]] || die "${label}: missing DFlash2 model.safetensors"
  log "${label}: DFlash2 snapshot verified"
}

verify_worker_snapshots() {
  local target draft command
  target="$(worker_snapshot_path "${MODEL_ID}" "${MODEL_REVISION}")"
  command="test -f '${target}/config.json' && test -f '${target}/model.safetensors.index.json' && test \"\$(find -L '${target}' -maxdepth 1 -type f -name '*.safetensors' | wc -l)\" -ge 120"
  if [[ "${SPEC_METHOD}" == "dflash" ]]; then
    draft="$(worker_snapshot_path "${DFLASH_MODEL_ID}" "${DFLASH_MODEL_REVISION}")"
    command+=" && test -f '${draft}/config.json' && test -f '${draft}/model.safetensors'"
  fi
  wrun "${command}"
}

pin_head_ref() {
  local repo_id="$1" revision="$2" repo tmp
  repo="${HF_HOME}/hub/$(model_cache_name "${repo_id}")"
  mkdir -p "${repo}/refs"
  tmp="$(mktemp "${repo}/refs/.main.ml-compute.XXXXXX")"
  printf '%s\n' "${revision}" >"${tmp}"
  mv "${tmp}" "${repo}/refs/main"
}

pin_worker_ref() {
  local repo_id="$1" revision="$2" repo
  repo="$(worker_hf_home)/hub/$(model_cache_name "${repo_id}")"
  wrun "mkdir -p '${repo}/refs'; printf '%s\\n' '${revision}' > '${repo}/refs/.main.ml-compute'; mv '${repo}/refs/.main.ml-compute' '${repo}/refs/main'"
}

pin_model_refs() {
  pin_head_ref "${MODEL_ID}" "${MODEL_REVISION}"
  pin_worker_ref "${MODEL_ID}" "${MODEL_REVISION}"
  if [[ "${SPEC_METHOD}" == "dflash" ]]; then
    pin_head_ref "${DFLASH_MODEL_ID}" "${DFLASH_MODEL_REVISION}"
    pin_worker_ref "${DFLASH_MODEL_ID}" "${DFLASH_MODEL_REVISION}"
  fi
  log "Pinned active target/draft cache refs/main on both nodes"
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
    || die "Head has ${head_free} GiB free; initial staging requires ${required} GiB"
  (( worker_free >= required )) \
    || die "Worker has ${worker_free} GiB free; mirror requires ${required} GiB"
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
  check_recipe_pin
  validate_profile
  warn_dflash_license
  need_cmd docker
  need_cmd curl
  need_cmd rsync
  need_cmd ssh
  need_cmd ip
  need_cmd nvidia-smi
  [[ "$(uname -m)" == "aarch64" ]] \
    || die "The pinned runtime is arm64-only; head reports $(uname -m)"
  docker info >/dev/null 2>&1 || die "Docker daemon is unavailable on the head"
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q GB10 \
    || die "NVIDIA GB10 was not detected on the head"
  wrun "test \"\$(uname -m)\" = aarch64" || die "Worker is not aarch64"
  worker_docker info >/dev/null 2>&1 || die "Docker daemon is unavailable on the worker"
  wrun "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q GB10" \
    || die "NVIDIA GB10 was not detected on the worker"
  resolve_cluster_interfaces
  ping -c 1 -W 2 "${WORKER_CX7_IP}" >/dev/null 2>&1 \
    || die "Direct RoCE address ${WORKER_CX7_IP} is unreachable from the head"
  check_disk
  check_proxy_key
  log "Pinned MiaAI EXL3 dual-Spark preflight passed"
}

write_upstream_env() {
  check_recipe_pin
  local remote_root tmp
  remote_root="$(worker_home)"
  tmp="$(mktemp "${RECIPE_DIR}/.env.ml-compute.XXXXXX")"
  {
    printf '# Generated by ml-compute; edit %s instead.\n' "${PROFILE_FILE}"
    printf 'HEAD_IP=%s\n' "${HEAD_CX7_IP}"
    printf 'WORKER_IP=%s\n' "${WORKER_CX7_IP}"
    printf 'WORKER_USER=%s\n' "${WORKER_USER}"
    printf 'WORKER_HOME=%s\n' "${remote_root}"
    printf 'WORKER_SSH=%s\n' "$(worker_ssh_target)"
    printf 'HEAD_CX7_IF=%s\n' "${HEAD_CX7_IF}"
    printf 'WORKER_CX7_IF=%s\n' "${WORKER_CX7_IF}"
    printf 'HEAD_CX7_IB=%s\n' "${HEAD_CX7_IB}"
    printf 'WORKER_CX7_IB=%s\n' "${WORKER_CX7_IB}"
    printf 'MODEL=%s\n' "${MODEL_ID}"
    printf 'HF_HOME=%s\n' "${HF_HOME}"
    printf 'IMAGE=%s\n' "${VLLM_IMAGE}"
    printf 'PORT=%s\n' "${PORT}"
    printf 'HOST_BIND=%s\n' "${HOST_BIND}"
    printf 'TP=%s\n' "${TENSOR_PARALLEL_SIZE}"
    printf 'NNODES=%s\n' "${NUM_NODES}"
    printf 'MASTER_PORT=%s\n' "${MASTER_PORT}"
    printf 'QUANTIZATION=%s\n' "${QUANTIZATION}"
    printf 'ENFORCE_EAGER=%s\n' "${ENFORCE_EAGER}"
    printf 'EXL3_FUSED_MOE=%s\n' "${EXL3_FUSED_MOE}"
    printf 'SERVED_MODEL_NAME=%s\n' "${SERVED_MODEL_NAME}"
    printf 'SPEC_METHOD=%s\n' "${SPEC_METHOD}"
    printf 'DFLASH_MODEL=%s\n' "${DFLASH_MODEL_ID}"
    printf 'DFLASH_TOKENS=%s\n' "${DFLASH_SPECULATIVE_TOKENS}"
    printf 'DFLASH_DRAFT_TP=%s\n' "${DFLASH_DRAFT_TP}"
    printf 'MTP_TOKENS=%s\n' "${MTP_SPECULATIVE_TOKENS}"
    printf 'MAX_MODEL_LEN=%s\n' "${MAX_MODEL_LEN}"
    printf 'MAX_NUM_SEQS=%s\n' "${MAX_NUM_SEQS}"
    printf 'MAX_NUM_BATCHED_TOKENS=%s\n' "${MAX_NUM_BATCHED_TOKENS}"
    printf 'GPU_MEM_UTIL=%s\n' "${GPU_MEMORY_UTILIZATION}"
    printf 'KV_CACHE_DTYPE=%s\n' "${KV_CACHE_DTYPE}"
    printf 'LANGUAGE_MODEL_ONLY=%s\n' "${LANGUAGE_MODEL_ONLY}"
    printf 'SKIP_MM_PROFILING=%s\n' "${SKIP_MM_PROFILING}"
    printf 'LIMIT_MM=%q\n' "${LIMIT_MM}"
    printf 'TORCH_CUDA_ARCH_LIST=%s\n' "${TORCH_CUDA_ARCH_LIST}"
    printf 'FLASHINFER_CUDA_ARCH_LIST=%s\n' "${FLASHINFER_CUDA_ARCH_LIST}"
    printf 'USE_HOST_NCCL=%s\n' "${USE_HOST_NCCL}"
    printf 'NCCL_HOST_DIR=%s\n' "${NCCL_HOST_DIR:-${HOME}/nccl-2.30.7}"
    printf 'WORKER_NCCL_HOST_DIR=%s\n' "${WORKER_NCCL_HOST_DIR:-${remote_root}/nccl-2.30.7}"
    printf 'NCCL_SO_NAME=%s\n' "${NCCL_SO_NAME}"
    printf 'NCCL_IB_GID_INDEX=%s\n' "${NCCL_IB_GID_INDEX}"
    printf 'NCCL_CROSS_NIC=%s\n' "${NCCL_CROSS_NIC}"
    printf 'NCCL_DEBUG=%s\n' "${NCCL_DEBUG}"
    printf 'READY_TIMEOUT=%s\n' "${VLLM_ENGINE_READY_TIMEOUT_S}"
    printf 'CACHE_ROOT=%s\n' "${CACHE_ROOT}"
    printf 'WORKER_VLLM_CACHE=%s\n' "$(worker_vllm_cache)"
    printf 'CONTAINER_HEAD=%s\n' "${GLM53_HEAD_CONTAINER}"
    printf 'CONTAINER_WORKER=%s\n' "${GLM53_WORKER_CONTAINER}"
    printf 'EXTRA_ARGS=%q\n' "${EXTRA_ARGS}"
  } >"${tmp}"
  mv "${tmp}" "${UPSTREAM_ENV}"
}

configure() {
  check_recipe_pin
  validate_profile
  mkdir -p "${RUNTIME_DIR}" "${RUNTIME_DIR}/logs" "${HF_HOME}" "${CACHE_ROOT}"
  local env_tmp key
  env_tmp="$(mktemp "${RUNTIME_DIR}/.env.glm53.XXXXXX")"
  {
    printf '# Generated by ml-compute; edit %s instead.\n' "${PROFILE_FILE}"
    printf 'UPSTREAM_REVISION=%s\n' "${UPSTREAM_REVISION}"
    while IFS= read -r key; do
      [[ -n "${key}" ]] || continue
      printf '%s=%s\n' "${key}" "${!key}"
    done < <(resolved_keys)
  } >"${env_tmp}"
  mv "${env_tmp}" "${RESOLVED_ENV}"
  write_upstream_env
  log "Validated profile at ${RESOLVED_ENV} and materialized MiaAI environment"
}

image_id() {
  docker image inspect --format '{{.Id}}' "$1" 2>/dev/null || true
}

worker_image_id() {
  worker_docker image inspect --format '{{.Id}}' "$1" 2>/dev/null || true
}

ship_image_to_worker() {
  log "Shipping ${VLLM_IMAGE} to worker via docker save | docker load"
  docker save "${VLLM_IMAGE}" | wrun 'docker load >/dev/null'
}

ensure_image() {
  check_recipe_pin
  validate_profile
  local source_id local_id worker_id
  if ! docker image inspect "${VLLM_SOURCE_IMAGE}" >/dev/null 2>&1; then
    log "Pulling digest-pinned MiaAI EXL3 image"
    docker pull "${VLLM_SOURCE_IMAGE}"
  fi
  source_id="$(image_id "${VLLM_SOURCE_IMAGE}")"
  [[ -n "${source_id}" ]] || die "Unable to inspect pinned source image"
  local_id="$(image_id "${VLLM_IMAGE}")"
  if [[ "${local_id}" != "${source_id}" ]]; then
    docker tag "${VLLM_SOURCE_IMAGE}" "${VLLM_IMAGE}"
    local_id="$(image_id "${VLLM_IMAGE}")"
  fi
  [[ "${local_id}" == "${source_id}" ]] \
    || die "Local runtime tag does not resolve to the pinned OCI digest"
  worker_id="$(worker_image_id "${VLLM_IMAGE}")"
  if [[ "${worker_id}" != "${local_id}" ]]; then
    ship_image_to_worker
    worker_id="$(worker_image_id "${VLLM_IMAGE}")"
  fi
  [[ "${worker_id}" == "${local_id}" ]] \
    || die "Worker image ID does not match the head"
  log "Digest-pinned EXL3 image is ready on both nodes"
}

gpu_check() {
  ensure_image
  local -a probe=(--rm --gpus all -e EXL3_SELFCHECK_GPU=1 --entrypoint python3 "${VLLM_IMAGE}" /opt/glm53/test_exl3_overlay.py)
  log "Running EXL3/SM121/DFlash2 GPU self-check on the head"
  docker run "${probe[@]}"
  log "Running EXL3/SM121/DFlash2 GPU self-check on the worker"
  worker_docker run "${probe[@]}"
}

download_snapshot() {
  local repo_id="$1" revision="$2" description="$3" python
  log "Downloading ${repo_id}@${revision} (${description})"
  if command -v hf >/dev/null 2>&1; then
    HF_HOME="${HF_HOME}" hf download "${repo_id}" --revision "${revision}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_HOME="${HF_HOME}" huggingface-cli download "${repo_id}" --revision "${revision}"
  else
    python="$(project_python)"
    HF_HOME="${HF_HOME}" MODEL_ID="${repo_id}" MODEL_REVISION="${revision}" \
      "${python}" -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ["MODEL_ID"], revision=os.environ["MODEL_REVISION"], token=os.environ.get("HF_TOKEN"))'
  fi
}

download_on_head() {
  local target draft
  target="$(snapshot_path "${MODEL_ID}" "${MODEL_REVISION}")"
  if [[ ! -d "${target}" ]]; then
    download_snapshot "${MODEL_ID}" "${MODEL_REVISION}" "~164 GiB EXL3 target"
  fi
  verify_target_snapshot "${target}" "head cache"
  pin_head_ref "${MODEL_ID}" "${MODEL_REVISION}"
  if [[ "${SPEC_METHOD}" == "dflash" ]]; then
    draft="$(snapshot_path "${DFLASH_MODEL_ID}" "${DFLASH_MODEL_REVISION}")"
    if [[ ! -d "${draft}" ]]; then
      download_snapshot "${DFLASH_MODEL_ID}" "${DFLASH_MODEL_REVISION}" "~2.3 GiB DFlash2 draft"
    fi
    verify_dflash_snapshot "${draft}" "head cache"
    pin_head_ref "${DFLASH_MODEL_ID}" "${DFLASH_MODEL_REVISION}"
  fi
}

sync_repo_to_worker() {
  local repo_id="$1" head_repo worker_repo
  head_repo="${HF_HOME}/hub/$(model_cache_name "${repo_id}")"
  worker_repo="$(worker_hf_home)/hub/$(model_cache_name "${repo_id}")"
  wrun "mkdir -p '${worker_repo}'"
  rsync -a --partial --info=progress2,stats1 --exclude '.locks' \
    -e 'ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -o BatchMode=yes' \
    "${head_repo}/" "$(worker_ssh_target):${worker_repo}/"
}

sync_weights_to_worker() {
  if verify_worker_snapshots; then
    log "Worker cache: active target/draft snapshots already verified"
    return
  fi
  [[ "${DOWNLOAD_MODE}" == "rsync" ]] \
    || die "Only DOWNLOAD_MODE=rsync is reviewed for this two-Spark recipe"
  log "Mirroring the pinned EXL3 target to the worker over RoCE"
  sync_repo_to_worker "${MODEL_ID}"
  if [[ "${SPEC_METHOD}" == "dflash" ]]; then
    log "Mirroring the pinned DFlash2 draft to the worker over RoCE"
    sync_repo_to_worker "${DFLASH_MODEL_ID}"
  fi
  verify_worker_snapshots || die "Worker snapshot verification failed after rsync"
}

download_model() {
  validate_profile
  warn_dflash_license
  check_disk
  download_on_head
  sync_weights_to_worker
  pin_model_refs
}

materialize_upstream_launcher() {
  local source="${RECIPE_DIR}/start.sh"
  local target="${RECIPE_DIR}/.ml-compute-start.sh"
  local tmp
  [[ -f "${source}" ]] || die "Pinned upstream launcher is missing: ${source}"
  [[ -f "${RECIPE_DIR}/files/chat_template.jinja" ]] \
    || die "Pinned upstream chat template is missing"
  tmp="$(mktemp "${RECIPE_DIR}/.ml-compute-start.XXXXXX")"
  if ! awk '
    BEGIN { host = 0; endpoint = 0 }
    /^[[:space:]]+--host 0\.0\.0\.0$/ {
      print "    --host \"${HOST_BIND:-127.0.0.1}\""
      host++
      next
    }
    /log "  endpoints  : http:\/\/127\.0\.0\.1:/ && /LAN:/ {
      print "    log \"  endpoint   : http://127.0.0.1:${PORT}/v1 (private raw API)\""
      endpoint++
      next
    }
    { print }
    END { if (host != 2 || endpoint != 1) exit 42 }
  ' "${source}" >"${tmp}"; then
    rm -f "${tmp}"
    die "Pinned MiaAI launcher no longer matches the reviewed loopback patch"
  fi
  chmod +x "${tmp}"
  mv "${tmp}" "${target}"
}

run_upstream() (
  check_recipe_pin
  validate_profile
  resolve_cluster_interfaces
  write_upstream_env
  materialize_upstream_launcher
  cd "${RECIPE_DIR}"
  SKIP_PULL=1 \
  SKIP_DOWNLOAD=1 \
  SKIP_SYNC=1 \
  HF_HUB_OFFLINE=1 \
  TAIL=0 \
  ./.ml-compute-start.sh "$@"
)

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

stop_legacy_cluster() {
  docker rm -f glm53-flash-ray-head >/dev/null 2>&1 || true
  worker_docker rm -f glm53-flash-ray-worker >/dev/null 2>&1 || true
}

show_failure_diagnostics() {
  printf '\nHead container state:\n'
  docker inspect --format '{{json .State}}' "${GLM53_HEAD_CONTAINER}" 2>/dev/null || true
  printf '\nHead container log:\n'
  docker logs --tail 500 "${GLM53_HEAD_CONTAINER}" 2>&1 || true
  printf '\nWorker container state:\n'
  worker_docker inspect --format '{{json .State}}' "${GLM53_WORKER_CONTAINER}" 2>/dev/null || true
  printf '\nWorker container log:\n'
  worker_docker logs --tail 500 "${GLM53_WORKER_CONTAINER}" 2>&1 || true
}

start_service() {
  check_recipe_pin
  validate_profile
  warn_dflash_license
  check_proxy_key
  check_public_exposure
  if raw_model_ready; then
    log "Pinned EXL3 model is already ready on the private raw endpoint"
  else
    verify_target_snapshot "$(snapshot_path "${MODEL_ID}" "${MODEL_REVISION}")" "head cache"
    if [[ "${SPEC_METHOD}" == "dflash" ]]; then
      verify_dflash_snapshot "$(snapshot_path "${DFLASH_MODEL_ID}" "${DFLASH_MODEL_REVISION}")" "head cache"
    fi
    verify_worker_snapshots || die "Pinned worker snapshots are missing; run glm53-flash download"
    pin_model_refs
    ensure_image
    stop_legacy_cluster
    run_upstream stop >/dev/null 2>&1 || true
    check_launch_memory
    run_upstream start
  fi
  run_proxy_cli serve
  if ! run_proxy_cli smoke; then
    run_proxy_cli stop || true
    die "Safety proxy checks failed; raw vLLM remains private on 127.0.0.1:${PORT}"
  fi
  if ! inference_smoke; then
    run_proxy_cli stop || true
    show_failure_diagnostics
    die "Post-start GLM completion failed"
  fi
  log "Watching EngineCore for 20 seconds after the first completion"
  local attempt
  for attempt in 1 2 3 4; do
    sleep 5
    if ! raw_model_ready; then
      run_proxy_cli stop || true
      show_failure_diagnostics
      die "EngineCore exited after the first completion"
    fi
  done
  log "Public authenticated API: http://127.0.0.1:${DSPARK_PROXY_PORT}/v1"
  log "Raw unauthenticated API:  http://127.0.0.1:${PORT}/v1 (loopback only)"
}

show_status() {
  run_upstream status
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
  warn "The ~164 GiB EXL3 target plus DFlash2 is a dedicated two-Spark workload"
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
print("GLM-5.3 Flash EXL3 completion passed")
PY
)

stop_service() {
  load_profile
  run_proxy_cli stop || true
  if [[ -d "${RECIPE_DIR}/.git" ]]; then
    run_upstream stop || true
  else
    docker rm -f "${GLM53_HEAD_CONTAINER}" >/dev/null 2>&1 || true
    worker_docker rm -f "${GLM53_WORKER_CONTAINER}" >/dev/null 2>&1 || true
  fi
  stop_legacy_cluster
  log "Stopped dual-Spark GLM service; weights, images, and compile caches were preserved"
}

action="${1:-help}"
case "${action}" in
  bootstrap) bootstrap ;;
  configure) configure ;;
  check) check_host ;;
  setup) bootstrap; configure; check_host ;;
  pull) ensure_image ;;
  download) download_model ;;
  gpu-check) gpu_check ;;
  start) start_service ;;
  status) load_profile; show_status ;;
  diagnose) load_profile; show_failure_diagnostics ;;
  memory) load_profile; show_memory ;;
  smoke) load_profile; inference_smoke ;;
  logs) load_profile; run_upstream logs ;;
  logs-worker) load_profile; run_upstream logs worker ;;
  stop) stop_service ;;
  update) sync_recipe_pin ;;
  all) bootstrap; configure; check_host; download_model; ensure_image; gpu_check; start_service ;;
  path)
    load_profile
    printf 'recipe=%s\nprofile=%s\nruntime=%s\nhf_home=%s\nupstream_revision=%s\nmodel_revision=%s\ndflash_revision=%s\nbase_image=%s\nsource_image=%s\nimage=%s\nexecutor=%s\nspec_method=%s\n' \
      "${RECIPE_DIR}" "${PROFILE_FILE}" "${RUNTIME_DIR}" "${HF_HOME}" \
      "${UPSTREAM_REVISION}" "${MODEL_REVISION}" "${DFLASH_MODEL_REVISION}" \
      "${VLLM_BASE_IMAGE}" "${VLLM_SOURCE_IMAGE}" "${VLLM_IMAGE}" \
      "${DISTRIBUTED_EXECUTOR_BACKEND}" "${SPEC_METHOD}"
    ;;
  help|-h|--help) usage ;;
  *) die "Unknown action: ${action}. Run glm53-flash help" ;;
esac
