#!/usr/bin/env bash
# GLM-5.3 Flash NVFP4 on two linked DGX Sparks (GB10 / SM121).
#
# Lifecycle adapter for MiaAI-Lab's validated vLLM/Ray TP=2 recipe:
# https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark
# The upstream checkout, model revision, and publisher base image are pinned.
# Generated images, compile caches, and the ~181 GiB checkpoint stay below the
# ignored data tree; ml-compute keeps ownership of configuration and exposure.

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE_FILE_DEFAULT="${PROJECT_ROOT}/config/dspark-glm53-flash-nvfp4.env"
RUNTIME_DIR_DEFAULT="${PROJECT_ROOT}/data/dspark/glm53-flash"
RECIPE_DIR_DEFAULT="${RUNTIME_DIR_DEFAULT}/miaai-dual-spark"
PROJECT_ENV_FILE_DEFAULT="${PROJECT_ROOT}/.env.local"

UPSTREAM_REPO_DEFAULT="https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark.git"
UPSTREAM_REVISION_DEFAULT="aed98a13ca75140d2691cc5c651ea5817d9a3e44"
MODEL_ID_DEFAULT="LibertAIDAI/GLM-5.3-Flash-NVFP4"
MODEL_REVISION_DEFAULT="11d73216cd636238e82e1d77fe1042ffab36e7fa"
VLLM_BASE_IMAGE_DEFAULT="vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce"
KERNEL_IMAGE_DEFAULT="ml-compute/glm53-flash-sm121-kernel:v8-aed98a1"
VLLM_IMAGE_DEFAULT="ml-compute/glm53-flash-sm121:mm-ray-v1-aed98a1"
RAY_VERSION_DEFAULT="2.58.0"

PROFILE_FILE="${GLM53_DSPARK_CONFIG_FILE:-${PROFILE_FILE_DEFAULT}}"
RUNTIME_DIR="${GLM53_DSPARK_RUNTIME_DIR:-${RUNTIME_DIR_DEFAULT}}"
RECIPE_DIR="${GLM53_DSPARK_RECIPE_DIR:-${RECIPE_DIR_DEFAULT}}"
PROJECT_ENV_FILE="${GLM53_DSPARK_PROJECT_ENV_FILE:-${PROJECT_ENV_FILE_DEFAULT}}"
UPSTREAM_REPO="${GLM53_DSPARK_UPSTREAM_REPO:-${UPSTREAM_REPO_DEFAULT}}"
UPSTREAM_REVISION="${GLM53_DSPARK_UPSTREAM_REVISION:-${UPSTREAM_REVISION_DEFAULT}}"
RESOLVED_ENV="${RUNTIME_DIR}/.env.glm53"

log() { printf '[glm53-flash] %s\n' "$*"; }
warn() { printf '[glm53-flash] WARNING: %s\n' "$*" >&2; }
die() { printf '[glm53-flash] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
GLM-5.3 Flash NVFP4 — MiaAI dual-DGX-Spark vLLM/Ray lifecycle

Run on the head/rank-0 Spark:

  python3 -m ml.cli glm53-flash setup
  python3 -m ml.cli glm53-flash pull
  python3 -m ml.cli glm53-flash download
  scripts/start-GLM53-Flash-Dual-DSpark.sh
  python3 -m ml.cli glm53-flash smoke

Actions:
  bootstrap    Clone and detach the reviewed MiaAI-Lab upstream revision
  configure    Validate and materialize the resolved ml-compute profile
  check        Validate the pin, both GB10 nodes, memory, disk, and API key
  setup        Run bootstrap, configure, and check
  pull         Build MiaAI's pinned SM121 kernel/serving image and ship it
  download     Download the pinned ~181 GiB snapshot and rsync it to the worker
  gpu-check    Verify CUDA, Ray 2.58, and the SM121/NoPE patch on both nodes
  start        Run MiaAI's worker-first TP=2 launch, then the safety proxy
  status       Show both containers, Ray status, raw API, and proxy state
  diagnose     Print the head log and relevant head/worker Ray logs
  memory       Show shared-memory and GPU use on both nodes
  smoke        Verify proxy safety and perform one GLM completion
  logs         Follow MiaAI's head container log
  logs-worker  Follow MiaAI's worker container log
  stop         Stop the proxy and both containers; preserve caches
  update       Fetch and re-checkout the revision pinned by this script
  all          Run setup, download, pull, gpu-check, and start
  path         Print checkout, profile, cache, image, and revision paths
  help         Show this help

Reviewed MiaAI profile:
  - patched SM121 image: SM90 NoPE sparse MLA + FlashInfer FA2
  - Ray 2.58.0 default executor with a 4 GiB object store per UMA node
  - 256K context, 8 sequences, FP8 KV, block size 2304
  - Marlin MoE + eager execution and MM profiling disabled during boot
  - glm47 tools, glm45 reasoning, in-checkpoint MTP=4
  - raw unauthenticated API on 127.0.0.1:8888 only
  - authenticated allow-list proxy on 0.0.0.0:8000

The previous RayExecutorV2 layer is intentionally not used: it reused actor
handles across Ray jobs during EngineCore initialization on this vLLM build.
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
GLM53_HEAD_CONTAINER|glm53-flash-head
GLM53_WORKER_CONTAINER|glm53-flash-worker
HOST_BIND|127.0.0.1
PORT|8888
DSPARK_PROXY_HOST|0.0.0.0
DSPARK_PROXY_PORT|8000
MODEL_ID|${MODEL_ID_DEFAULT}
MODEL_REVISION|${MODEL_REVISION_DEFAULT}
SERVED_MODEL_NAME|${MODEL_ID_DEFAULT}
VLLM_BASE_IMAGE|${VLLM_BASE_IMAGE_DEFAULT}
KERNEL_IMAGE|${KERNEL_IMAGE_DEFAULT}
VLLM_IMAGE|${VLLM_IMAGE_DEFAULT}
RAY_VERSION|${RAY_VERSION_DEFAULT}
RAY_OBJECT_STORE_MEMORY|4294967296
TENSOR_PARALLEL_SIZE|2
MAX_MODEL_LEN|262144
MAX_NUM_SEQS|8
GPU_MEMORY_UTILIZATION|0.84
BLOCK_SIZE|2304
KV_CACHE_DTYPE|fp8_e4m3
KV_CACHE_MEMORY|
MOE_BACKEND|marlin
ENFORCE_EAGER|1
TOOL_CALL_PARSER|glm47
REASONING_PARSER|glm45
MTP_SPECULATIVE_TOKENS|4
LIMIT_MM|{"image":4,"video":1}
SKIP_MM_PROFILING|1
TORCH_CUDA_ARCH_LIST|12.1a
FLASHINFER_CUDA_ARCH_LIST|12.1a
HF_HOME|${RUNTIME_DIR}/cache/huggingface
WORKER_HF_HOME|
DOWNLOAD_MODE|rsync
VLLM_ENGINE_READY_TIMEOUT_S|3600
CLUSTER_WAIT_ITERS|120
CACHE_VOLUME|glm53-flash-cache-sm121
USE_HOST_NCCL|1
NCCL_HOST_DIR|
WORKER_NCCL_HOST_DIR|
NCCL_SO_NAME|libnccl.so.2.30.7
NCCL_DEBUG|WARN
NCCL_IB_GID_INDEX|3
NCCL_CROSS_NIC|0
EXTRA_ARGS|
GLM53_MIN_AVAILABLE_GIB|112
GLM53_MIN_DISK_GIB|240
EOF

  # MiaAI's launcher runs from RECIPE_DIR. Resolve a user-supplied relative
  # HF_HOME against the ml-compute root before that directory change so the
  # cache verified here is the same cache mounted by the upstream launcher.
  if [[ "${HF_HOME}" != /* ]]; then
    HF_HOME="${PROJECT_ROOT}/${HF_HOME#./}"
    export HF_HOME
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
VLLM_BASE_IMAGE
KERNEL_IMAGE
VLLM_IMAGE
RAY_VERSION
RAY_OBJECT_STORE_MEMORY
TENSOR_PARALLEL_SIZE
MAX_MODEL_LEN
MAX_NUM_SEQS
GPU_MEMORY_UTILIZATION
BLOCK_SIZE
KV_CACHE_DTYPE
KV_CACHE_MEMORY
MOE_BACKEND
ENFORCE_EAGER
TOOL_CALL_PARSER
REASONING_PARSER
MTP_SPECULATIVE_TOKENS
LIMIT_MM
SKIP_MM_PROFILING
TORCH_CUDA_ARCH_LIST
FLASHINFER_CUDA_ARCH_LIST
HF_HOME
WORKER_HF_HOME
DOWNLOAD_MODE
VLLM_ENGINE_READY_TIMEOUT_S
CLUSTER_WAIT_ITERS
CACHE_VOLUME
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
  [[ "${TENSOR_PARALLEL_SIZE}" == "2" ]] \
    || die "MiaAI's reviewed lifecycle is specifically two Sparks and TP=2"
  (( MAX_MODEL_LEN <= 262144 )) \
    || die "MAX_MODEL_LEN above MiaAI's reviewed 262144-token profile is not enabled"
  (( MAX_NUM_SEQS <= 8 )) \
    || die "MAX_NUM_SEQS above MiaAI's reviewed value 8 is not enabled"
  awk -v value="${GPU_MEMORY_UTILIZATION}" 'BEGIN {exit !(value <= 0.84)}' \
    || die "GPU_MEMORY_UTILIZATION above 0.84 can exhaust GB10 UMA during MM warmup"
  [[ "${BLOCK_SIZE}" == "2304" ]] \
    || die "BLOCK_SIZE must remain 2304 for MiaAI's DeepGEMM paged-MQA profile"
  [[ "${KV_CACHE_DTYPE}" == "fp8_e4m3" ]] \
    || die "KV_CACHE_DTYPE must remain fp8_e4m3 for the reviewed memory profile"
  [[ "${MOE_BACKEND}" == "marlin" ]] \
    || die "MOE_BACKEND must remain marlin on GB10/SM121"
  [[ "${ENFORCE_EAGER}" == "1" ]] \
    || die "ENFORCE_EAGER must remain enabled on GB10 UMA"
  [[ "${SKIP_MM_PROFILING}" == "1" ]] \
    || die "SKIP_MM_PROFILING must remain enabled; the max-size MM dummy forward OOMs UMA"
  (( MTP_SPECULATIVE_TOKENS >= 0 && MTP_SPECULATIVE_TOKENS <= 4 )) \
    || die "MTP_SPECULATIVE_TOKENS must be between 0 and MiaAI's reviewed value 4"
  [[ "${RAY_VERSION}" == "${RAY_VERSION_DEFAULT}" ]] \
    || die "RAY_VERSION must remain ${RAY_VERSION_DEFAULT} for the pinned MiaAI recipe"
  (( RAY_OBJECT_STORE_MEMORY <= 4294967296 )) \
    || die "Ray object store above 4 GiB steals memory from the GB10 GPU budget"
  [[ "${MODEL_ID}" == "${MODEL_ID_DEFAULT}" ]] \
    || die "MODEL_ID must remain ${MODEL_ID_DEFAULT}"
  [[ "${MODEL_REVISION}" == "${MODEL_REVISION_DEFAULT}" ]] \
    || die "MODEL_REVISION must remain the reviewed checkpoint pin"
  [[ "${SERVED_MODEL_NAME}" == "${MODEL_ID_DEFAULT}" ]] \
    || die "SERVED_MODEL_NAME must match MiaAI's public model name ${MODEL_ID_DEFAULT}"
  [[ "${VLLM_BASE_IMAGE}" == "${VLLM_BASE_IMAGE_DEFAULT}" ]] \
    || die "VLLM_BASE_IMAGE must remain the reviewed publisher manifest"
  [[ "${KERNEL_IMAGE}" == "${KERNEL_IMAGE_DEFAULT}" ]] \
    || warn "KERNEL_IMAGE differs from the revision-keyed local tag"
  [[ "${VLLM_IMAGE}" == "${VLLM_IMAGE_DEFAULT}" ]] \
    || warn "VLLM_IMAGE differs from the revision-keyed local tag"
  [[ "${GLM53_HEAD_CONTAINER}" == "glm53-flash-head" \
    && "${GLM53_WORKER_CONTAINER}" == "glm53-flash-worker" ]] \
    || die "Container names must match the pinned upstream lifecycle"
  [[ "${USE_HOST_NCCL}" == "0" || "${USE_HOST_NCCL}" == "1" ]] \
    || die "USE_HOST_NCCL must be 0 or 1"
  [[ " ${EXTRA_ARGS} " != *" --host "* && " ${EXTRA_ARGS} " != *" --host="* ]] \
    || die "EXTRA_ARGS may not override the private raw host"
  [[ " ${EXTRA_ARGS} " != *" --port "* && " ${EXTRA_ARGS} " != *" --port="* ]] \
    || die "EXTRA_ARGS may not override the private raw port"
  [[ "${EXTRA_ARGS}" != *"VLLM_USE_RAY_V2_EXECUTOR_BACKEND"* ]] \
    || die "RayExecutorV2 is not used by the validated MiaAI strategy"
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
  log "Pinned MiaAI upstream revision: ${UPSTREAM_REVISION}"
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

configure() {
  check_recipe_pin
  validate_profile
  mkdir -p "${RUNTIME_DIR}" "${RUNTIME_DIR}/logs" "${HF_HOME}"
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

model_cache_name() {
  printf 'models--%s' "${MODEL_ID//\//--}"
}

snapshot_path() {
  printf '%s/hub/%s/snapshots/%s' "${HF_HOME}" "$(model_cache_name)" "${MODEL_REVISION}"
}

worker_snapshot_path() {
  printf '%s/hub/%s/snapshots/%s' "$(worker_hf_home)" "$(model_cache_name)" "${MODEL_REVISION}"
}

verify_snapshot() {
  local snapshot="$1" label="$2" count
  [[ -f "${snapshot}/config.json" ]] || die "${label}: missing config.json in pinned snapshot"
  [[ -f "${snapshot}/processor_config.json" ]] \
    || die "${label}: missing processor_config.json in pinned snapshot"
  [[ -f "${snapshot}/model.safetensors.index.json" ]] \
    || die "${label}: missing model.safetensors.index.json"
  [[ -f "${snapshot}/chat_template.jinja" ]] \
    || die "${label}: missing chat_template.jinja"
  count="$(find -L "${snapshot}" -maxdepth 1 -type f -name '*.safetensors' | wc -l | tr -d ' ')"
  (( count >= 120 )) || die "${label}: only ${count}/120 safetensor shards are present"
  log "${label}: pinned snapshot verified (${count} shards)"
}

verify_worker_snapshot() {
  local snapshot count
  snapshot="$(worker_snapshot_path)"
  wrun "test -f '${snapshot}/config.json' && test -f '${snapshot}/processor_config.json' && test -f '${snapshot}/model.safetensors.index.json' && test -f '${snapshot}/chat_template.jinja'" \
    || return 1
  count="$(wrun "find -L '${snapshot}' -maxdepth 1 -type f -name '*.safetensors' | wc -l" | tr -d ' ')"
  (( count >= 120 ))
}

pin_model_refs() {
  local head_repo head_tmp worker_repo
  head_repo="${HF_HOME}/hub/$(model_cache_name)"
  worker_repo="$(worker_hf_home)/hub/$(model_cache_name)"
  mkdir -p "${head_repo}/refs"
  head_tmp="$(mktemp "${head_repo}/refs/.main.ml-compute.XXXXXX")"
  printf '%s\n' "${MODEL_REVISION}" >"${head_tmp}"
  mv "${head_tmp}" "${head_repo}/refs/main"
  wrun "mkdir -p '${worker_repo}/refs'; printf '%s\\n' '${MODEL_REVISION}' > '${worker_repo}/refs/.main.ml-compute'; mv '${worker_repo}/refs/.main.ml-compute' '${worker_repo}/refs/main'"
  log "Pinned MiaAI cache refs/main to ${MODEL_REVISION} on both nodes"
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
  check_recipe_pin
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
  wrun "test \"\$(uname -m)\" = aarch64" || die "Worker is not aarch64"
  worker_docker info >/dev/null 2>&1 || die "Docker daemon is unavailable on the worker"
  wrun "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q GB10" \
    || die "NVIDIA GB10 was not detected on the worker"
  ping -c 1 -W 2 "${WORKER_CX7_IP}" >/dev/null 2>&1 \
    || die "Direct RoCE address ${WORKER_CX7_IP} is unreachable from the head"
  check_disk
  check_proxy_key
  log "Pinned MiaAI dual-Spark GLM preflight passed"
}

image_revision() {
  docker image inspect --format '{{ index .Config.Labels "org.ml-compute.upstream.revision" }}' \
    "$1" 2>/dev/null || true
}

worker_image_revision() {
  worker_docker image inspect --format '{{ index .Config.Labels "org.ml-compute.upstream.revision" }}' \
    "$1" 2>/dev/null || true
}

build_image_on_head() {
  local kernel_dockerfile serving_dockerfile
  kernel_dockerfile="${RECIPE_DIR}/files/Dockerfile"
  serving_dockerfile="${RECIPE_DIR}/files/Dockerfile.mm-ray"
  [[ -f "${kernel_dockerfile}" && -f "${serving_dockerfile}" ]] \
    || die "Pinned MiaAI Dockerfiles are missing under ${RECIPE_DIR}/files"

  if ! docker image inspect "${VLLM_BASE_IMAGE}" >/dev/null 2>&1; then
    docker pull "${VLLM_BASE_IMAGE}"
  fi

  log "Building MiaAI SM121/NoPE kernel image ${KERNEL_IMAGE}"
  {
    awk -v from="FROM ${VLLM_BASE_IMAGE}" 'NR == 1 {$0 = from} {print}' \
      "${kernel_dockerfile}"
    printf '\nLABEL org.ml-compute.upstream.revision="%s"\n' "${UPSTREAM_REVISION}"
  } | docker build --network host -f - --tag "${KERNEL_IMAGE}" "${RECIPE_DIR}/files"

  log "Building MiaAI Ray/MM serving image ${VLLM_IMAGE}"
  {
    awk -v from="FROM ${KERNEL_IMAGE}" 'NR == 1 {$0 = from} {print}' \
      "${serving_dockerfile}"
    printf '\nLABEL org.ml-compute.upstream.revision="%s" org.ml-compute.ray.version="%s"\n' \
      "${UPSTREAM_REVISION}" "${RAY_VERSION}"
  } | docker build --network host -f - --tag "${VLLM_IMAGE}" "${RECIPE_DIR}/files"
}

ship_image_to_worker() {
  log "Shipping ${VLLM_IMAGE} to worker via docker save | docker load"
  docker save "${VLLM_IMAGE}" | wrun 'docker load >/dev/null'
}

ensure_image() {
  check_recipe_pin
  validate_profile
  local head_revision worker_revision
  head_revision="$(image_revision "${VLLM_IMAGE}")"
  worker_revision="$(worker_image_revision "${VLLM_IMAGE}")"
  if [[ "${head_revision}" != "${UPSTREAM_REVISION}" ]]; then
    build_image_on_head
    head_revision="$(image_revision "${VLLM_IMAGE}")"
  fi
  [[ "${head_revision}" == "${UPSTREAM_REVISION}" ]] \
    || die "Head image is not labeled for the pinned MiaAI revision"
  if [[ "${worker_revision}" != "${UPSTREAM_REVISION}" ]]; then
    ship_image_to_worker
    worker_revision="$(worker_image_revision "${VLLM_IMAGE}")"
  fi
  [[ "${worker_revision}" == "${UPSTREAM_REVISION}" ]] \
    || die "Worker image is not labeled for the pinned MiaAI revision"
  log "MiaAI SM121 serving image is ready on both nodes (Ray ${RAY_VERSION})"
}

gpu_check() {
  ensure_image
  local probe
  probe="python3 -c 'import importlib.metadata as m, pathlib, ray, torch, vllm; names={d.metadata.get(\"Name\", \"\").lower().replace(\"_\", \"-\") for d in m.distributions()}; cuda=pathlib.Path(\"/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py\").read_text(); sm90=pathlib.Path(\"/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py\").read_text(); assert torch.cuda.is_available(); assert ray.__version__ == \"${RAY_VERSION}\"; assert \"cupy-cuda12x\" not in names; assert \"FLASHINFER_MLA_SPARSE_SM90\" in cuda; assert \"capability.major in (9, 12)\" in sm90; print(torch.cuda.get_device_name(0), \"vllm\", vllm.__version__, \"ray\", ray.__version__, \"sm121_nope_patch=1\")'"
  docker run --rm --gpus all --entrypoint /bin/bash "${VLLM_IMAGE}" -lc "${probe}"
  worker_docker run --rm --gpus all --entrypoint /bin/bash "${VLLM_IMAGE}" -lc "${probe}"
}

download_on_head() {
  local snapshot python
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
    python="$(project_python)"
    HF_HOME="${HF_HOME}" MODEL_ID="${MODEL_ID}" MODEL_REVISION="${MODEL_REVISION}" \
      "${python}" -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ["MODEL_ID"], revision=os.environ["MODEL_REVISION"], token=os.environ.get("HF_TOKEN"))'
  fi
  verify_snapshot "${snapshot}" "head cache"
}

sync_weights_to_worker() {
  local head_repo worker_repo
  if verify_worker_snapshot; then
    log "Worker cache: pinned snapshot already verified"
    return
  fi
  [[ "${DOWNLOAD_MODE}" == "rsync" ]] \
    || die "Only DOWNLOAD_MODE=rsync is reviewed for this two-Spark recipe"
  head_repo="${HF_HOME}/hub/$(model_cache_name)"
  worker_repo="$(worker_hf_home)/hub/$(model_cache_name)"
  wrun "mkdir -p '${worker_repo}'"
  log "Mirroring the pinned GLM cache to the worker over RoCE (resumable)"
  rsync -a --partial --info=progress2,stats1 --exclude '.locks' \
    -e 'ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -o BatchMode=yes' \
    "${head_repo}/" "$(worker_ssh_target):${worker_repo}/"
  verify_worker_snapshot || die "Worker snapshot verification failed after rsync"
  log "Worker cache: pinned snapshot verified"
}

download_model() {
  validate_profile
  check_disk
  download_on_head
  sync_weights_to_worker
  pin_model_refs
}

run_upstream() (
  check_recipe_pin
  validate_profile
  local target remote_root private_args
  target="$(worker_ssh_target)"
  remote_root="$(worker_home)"
  private_args="${EXTRA_ARGS:+${EXTRA_ARGS} }--host ${HOST_BIND}"
  cd "${RECIPE_DIR}"
  IMAGE="${VLLM_IMAGE}" \
  RAY_VERSION="${RAY_VERSION}" \
  HEAD_IP="${HEAD_CX7_IP}" \
  WORKER_IP="${WORKER_CX7_IP}" \
  WORKER_SSH="${target}" \
  WORKER_HOME="${remote_root}" \
  HEAD_CX7_IF="${HEAD_CX7_IF}" \
  WORKER_CX7_IF="${WORKER_CX7_IF}" \
  HEAD_CX7_IB="${HEAD_CX7_IB}" \
  WORKER_CX7_IB="${WORKER_CX7_IB}" \
  RAY_PORT="${RAY_HEAD_PORT}" \
  RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY}" \
  TP="${TENSOR_PARALLEL_SIZE}" \
  PORT="${PORT}" \
  MTP_TOKENS="${MTP_SPECULATIVE_TOKENS}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION}" \
  BLOCK_SIZE="${BLOCK_SIZE}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  KV_CACHE_DTYPE="${KV_CACHE_DTYPE}" \
  KV_CACHE_MEMORY="${KV_CACHE_MEMORY}" \
  LIMIT_MM="${LIMIT_MM}" \
  SKIP_MM_PROFILING="${SKIP_MM_PROFILING}" \
  MOE_BACKEND="${MOE_BACKEND}" \
  ENFORCE_EAGER="${ENFORCE_EAGER}" \
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
  FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST}" \
  HF_HOME="${HF_HOME}" \
  READY_TIMEOUT="${VLLM_ENGINE_READY_TIMEOUT_S}" \
  CLUSTER_WAIT_ITERS="${CLUSTER_WAIT_ITERS}" \
  CACHE_VOLUME="${CACHE_VOLUME}" \
  USE_HOST_NCCL="${USE_HOST_NCCL}" \
  NCCL_HOST_DIR="${NCCL_HOST_DIR:-${HOME}/nccl-2.30.7}" \
  WORKER_NCCL_HOST_DIR="${WORKER_NCCL_HOST_DIR:-${remote_root}/nccl-2.30.7}" \
  NCCL_SO_NAME="${NCCL_SO_NAME}" \
  NCCL_DEBUG="${NCCL_DEBUG}" \
  NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX}" \
  NCCL_CROSS_NIC="${NCCL_CROSS_NIC}" \
  CHAT_TEMPLATE_URL="https://huggingface.co/${MODEL_ID}/resolve/${MODEL_REVISION}/chat_template.jinja" \
  HF_HUB_OFFLINE=1 \
  SKIP_DOWNLOAD=1 \
  SKIP_SYNC=1 \
  EXTRA_ARGS="${private_args}" \
  TAIL=0 \
  ./start.sh "$@"
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
  local scan
  scan='root=/tmp/ray/session_latest/logs; [ -d "$root" ] || exit 0; find "$root" -maxdepth 1 -type f \( -name "worker-*.err" -o -name "worker-*.out" -o -name "python-core-worker-*.log" \) -size +0c -printf "%T@ %p\\n" | sort -n | tail -n 16 | cut -d" " -f2- | while IFS= read -r file; do printf "\\n===== %s =====\\n" "$file"; tail -n 180 "$file"; done'
  printf '\nHead container log:\n'
  docker logs --tail 400 "${GLM53_HEAD_CONTAINER}" 2>&1 || true
  printf '\nRay cluster status:\n'
  docker exec "${GLM53_HEAD_CONTAINER}" ray status 2>&1 || true
  printf '\nRecent Ray worker logs (head):\n'
  docker exec "${GLM53_HEAD_CONTAINER}" /bin/bash -lc "${scan}" 2>&1 || true
  printf '\nRecent Ray worker logs (worker):\n'
  worker_docker exec "${GLM53_WORKER_CONTAINER}" /bin/bash -lc "${scan}" 2>&1 || true
}

start_service() {
  check_recipe_pin
  validate_profile
  check_proxy_key
  check_public_exposure
  if raw_model_ready; then
    log "Pinned GLM model is already ready on the private raw endpoint"
  else
    verify_snapshot "$(snapshot_path)" "head cache"
    verify_worker_snapshot || die "Pinned worker snapshot is missing; run glm53-flash download"
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
print("GLM-5.3 Flash model completion passed")
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
  logs) load_profile; run_upstream logs head ;;
  logs-worker) load_profile; run_upstream logs worker ;;
  stop) stop_service ;;
  update) sync_recipe_pin ;;
  all) bootstrap; configure; check_host; download_model; ensure_image; gpu_check; start_service ;;
  path)
    load_profile
    printf 'recipe=%s\nprofile=%s\nruntime=%s\nhf_home=%s\nupstream_revision=%s\nmodel_revision=%s\nbase_image=%s\nkernel_image=%s\nimage=%s\nray_version=%s\nobject_store_bytes=%s\n' \
      "${RECIPE_DIR}" "${PROFILE_FILE}" "${RUNTIME_DIR}" "${HF_HOME}" \
      "${UPSTREAM_REVISION}" "${MODEL_REVISION}" "${VLLM_BASE_IMAGE}" \
      "${KERNEL_IMAGE}" "${VLLM_IMAGE}" "${RAY_VERSION}" "${RAY_OBJECT_STORE_MEMORY}"
    ;;
  help|-h|--help) usage ;;
  *) die "Unknown action: ${action}. Run glm53-flash help" ;;
esac
