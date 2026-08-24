#!/usr/bin/env bash
# DeepSeek-V4-Flash-0731 on two GB10/DGX Spark-class systems.
#
# This is an opinionated wrapper around MiaAI-Lab's maintained two-node recipe:
# https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
# It keeps the existing ml.cli lifecycle while using MiaAI's immutable Anemll
# runtime and current launch-time hotfix set. Run every command on the head node.

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_REPO="${DSPARK_UPSTREAM_REPO:-https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark.git}"
RECIPE_DIR="${DSPARK_RECIPE_DIR:-${PROJECT_ROOT}/data/dspark/miaai-deepseek-v4-flash-0731}"
ENV_FILE="${DSPARK_ENV_FILE:-${RECIPE_DIR}/.env.dspark}"
CONFIG_FILE="${DSPARK_CONFIG_FILE:-${PROJECT_ROOT}/config/dspark-spark4e89-thinkstationpgx.env}"
PROJECT_ENV_FILE="${DSPARK_PROJECT_ENV_FILE:-${PROJECT_ROOT}/.env.local}"
LEGACY_RECIPE_DIR="${DSPARK_LEGACY_RECIPE_DIR:-${PROJECT_ROOT}/data/dspark/deepseek-v4-flash-0731}"

# Defaults for this installation. Override any of these in the command environment.
WORKER_HOST_DEFAULT="totalpass@192.168.177.11"
MODEL_DEFAULT="deepseek-ai/DeepSeek-V4-Flash-0731"
SERVED_MODEL_DEFAULT="deepseek-v4-flash-0731"
REVISION_DEFAULT="9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
IMAGE_DEFAULT="ghcr.io/anemll/dspark-vllm-gx10:0.1.1@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8"

log() { printf '[dspark] %s\n' "$*"; }
warn() { printf '[dspark] WARNING: %s\n' "$*" >&2; }
die() { printf '[dspark] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
DeepSeek V4 Flash 0731 — two-Spark setup and lifecycle

Run on the NVIDIA/head Spark:

  # 1. Configure the QSFP/ConnectX-7 link with NVIDIA Sync, then inspect it.
  python3 -m ml.cli dspark network

  # 2. Clone/configure from the committed machine profile and check both nodes.
  python3 -m ml.cli dspark setup

  # 3. Pull the pinned image on both nodes, download/mirror weights, and start.
  python3 -m ml.cli dspark build
  python3 -m ml.cli dspark download
  scripts/start-DS4-Flash-DSpark.sh

  # 4. Operate and validate the endpoint.
  python3 -m ml.cli dspark status
  python3 -m ml.cli dspark gpu-check
  python3 -m ml.cli dspark smoke
  python3 -m ml.cli dspark logs
  python3 -m ml.cli dspark stop

Actions:
  network    Show ConnectX/RoCE state and configuration guidance (read-only)
  bootstrap  Clone the upstream recipe and create .env.dspark if absent
  configure  Apply environment overrides and the validated 0731 profile
  check      Validate local/remote prerequisites and the generated config
  setup      Run bootstrap, configure, and check
  build      Pull/verify the immutable Anemll image on both nodes
  download   Download and verify weights, then mirror them to the worker
  start      Launch vLLM worker-first, then the authenticated proxy
  status     Show head and worker container state
  gpu-check  Prove PyTorch can initialize GB10 inside the runtime on both nodes
  memory     Show head/worker MemAvailable and verify the Harness budget
  smoke      Test model inference plus proxy auth/deny rules
  logs       Follow the distributed server logs
  stop       Stop the head and worker services
  legacy-stop Stop the former Stage-C deployment during first cutover
  update     Fast-forward the upstream recipe (never changes this project)
  all        Run setup, build, download, and start
  path       Print the upstream checkout and environment file locations

Configuration environment variables:
  Common overrides:
    DSPARK_CONFIG_FILE       Machine profile (default: config/dspark-spark4e89-thinkstationpgx.env)
    DSPARK_PROJECT_ENV_FILE  API_KEY source (default: project .env.local)
    WORKER_HOST              SSH target (profile: totalpass@192.168.177.11)
    WORKER_SCRIPT_DIR        Dedicated deployment path on the worker
    HF_CACHE                 Head Hugging Face cache
    WORKER_HF_CACHE          Worker Hugging Face cache
    DSPARK_RECIPE_DIR        Upstream checkout on the head
    DSPARK_VLLM_HOST         Raw API bind override (profile: 127.0.0.1)
    DSPARK_VLLM_PORT         Raw vLLM port override (profile: 8888)
    MAX_MODEL_LEN            Profile default 524288 (512K)
    MAX_NUM_SEQS             Profile default 4
    MAX_NUM_BATCHED_TOKENS   Profile default 4096
    GPU_MEMORY_UTILIZATION_TEXT  Profile default 0.70 on both TP ranks
    MTP_NUM_TOKENS           Default and recommended value: 5
    WORKER_AVAILABLE_TARGET_GIB  Required MemAvailable after model start (24)
    DSPARK_VLLM_IMAGE        Immutable Anemll image reference
    NCCL_IB_GID_INDEX        RoCE v2/IPv4 GID index (profile: 3)
    ALLOW_ACTIVE_VLLM=1      Bypass the competing-workload launch guard

Notes:
  - Cluster addresses, interfaces, paths, and memory limits are committed in
    config/dspark-spark4e89-thinkstationpgx.env.
  - Process-environment values override the committed profile for one command.
  - API authentication is required and sourced from API_KEY in .env.local.
  - Raw vLLM is loopback-only on 8888. The authenticated allow-list proxy is
    the network-facing service on 8000.
  - `setup` never replaces an existing .env.dspark; it updates known keys in place.
  - Build and download are large/slow operations and run only when explicitly requested.
  - NVIDIA's generic NIM path here is single-node. This 0731 TP=2 path uses
    MiaAI-Lab's hotfixes over a digest-pinned Anemll runtime.
EOF
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

profile_value() {
  local key="$1" fallback="${2:-}" value=""

  # A value explicitly exported by the caller wins over the committed profile.
  if declare -p "${key}" >/dev/null 2>&1; then
    printf '%s' "${!key}"
    return
  fi

  value="$(profile_file_value "${key}" "${fallback}")"
  printf '%s' "${value:-${fallback}}"
}

profile_file_value() {
  local key="$1" fallback="${2:-}" value=""
  if [[ -f "${CONFIG_FILE}" ]]; then
    value="$(sed -n "s/^${key}=//p" "${CONFIG_FILE}" | tail -n 1)"
  fi
  printf '%s' "${value:-${fallback}}"
}

require_checkout() {
  [[ -d "${RECIPE_DIR}/.git" ]] || die "Recipe is not bootstrapped. Run: python3 -m ml.cli dspark bootstrap"
}

require_env_file() {
  [[ -f "${ENV_FILE}" ]] || die "Missing ${ENV_FILE}. Run: python3 -m ml.cli dspark bootstrap"
}

env_value() {
  local key="$1"
  require_env_file
  sed -n "s/^${key}=//p" "${ENV_FILE}" | tail -n 1
}

set_env_value() {
  local key="$1" value="$2" tmp
  tmp="$(mktemp "${ENV_FILE}.XXXXXX")"
  awk -v key="${key}" -v value="${value}" '
    BEGIN { found = 0 }
    $0 ~ ("^" key "=") { print key "=" value; found = 1; next }
    { print }
    END { if (!found) print key "=" value }
  ' "${ENV_FILE}" >"${tmp}"
  mv "${tmp}" "${ENV_FILE}"
}

project_env_value() {
  local key="$1"
  [[ -f "${PROJECT_ENV_FILE}" ]] || return 0
  sed -n "s/^${key}=//p" "${PROJECT_ENV_FILE}" | tail -n 1
}

configure_api_auth() {
  local api_key
  api_key="${API_KEY:-$(project_env_value API_KEY)}"
  [[ -n "${api_key}" ]] || die "API_KEY is missing from ${PROJECT_ENV_FILE}"
  [[ "${api_key}" =~ ^[A-Za-z0-9._~-]+$ ]] \
    || die "API_KEY may contain only letters, digits, dot, underscore, tilde, and hyphen"

  set_env_value VLLM_API_KEY "${api_key}"
  set_env_value DSPARK_API_KEYS ""
  chmod 600 "${ENV_FILE}"
  log "vLLM and proxy authentication configured from ${PROJECT_ENV_FILE} (key not displayed)"
}

worker_host() {
  if [[ -n "${WORKER_HOST:-}" ]]; then
    printf '%s' "${WORKER_HOST}"
  elif [[ -f "${ENV_FILE}" ]] && [[ -n "$(env_value WORKER_HOST)" ]]; then
    env_value WORKER_HOST
  else
    profile_value WORKER_HOST "${WORKER_HOST_DEFAULT}"
  fi
}

remote_home() {
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$(worker_host)" 'printf "%s" "$HOME"'
}

bootstrap() {
  need_cmd git
  if [[ -d "${RECIPE_DIR}/.git" ]]; then
    log "Upstream recipe already exists: ${RECIPE_DIR}"
  elif [[ -e "${RECIPE_DIR}" ]]; then
    die "${RECIPE_DIR} exists but is not a Git checkout; set DSPARK_RECIPE_DIR to a dedicated path"
  else
    mkdir -p "$(dirname "${RECIPE_DIR}")"
    log "Cloning maintained two-node runtime into ${RECIPE_DIR}"
    git clone "${UPSTREAM_REPO}" "${RECIPE_DIR}"
  fi

  if [[ ! -f "${ENV_FILE}" ]]; then
    cp "${RECIPE_DIR}/.env.dspark.example" "${ENV_FILE}"
    chmod 600 "${ENV_FILE}"
    log "Created ${ENV_FILE}"
  else
    log "Keeping existing ${ENV_FILE}"
  fi
}

configure() {
  require_checkout
  require_env_file
  [[ -f "${CONFIG_FILE}" ]] || die "Missing cluster profile: ${CONFIG_FILE}"

  # Use this machine pair by default, while leaving fabric addressing explicit.
  if [[ "$(env_value WORKER_HOST)" == "worker-host-or-roce-ip" || -z "$(env_value WORKER_HOST)" ]]; then
    set_env_value WORKER_HOST "$(profile_value WORKER_HOST "${WORKER_HOST_DEFAULT}")"
  fi

  set_env_value ABLITERATED "$(profile_value ABLITERATED 0)"
  set_env_value DSPARK_MODEL_OFFICIAL "$(profile_value DSPARK_MODEL_OFFICIAL "${MODEL_DEFAULT}")"
  set_env_value DSPARK_REVISION "$(profile_value DSPARK_REVISION "${REVISION_DEFAULT}")"
  set_env_value SERVED_MODEL_NAME "$(profile_value SERVED_MODEL_NAME "${SERVED_MODEL_DEFAULT}")"
  set_env_value PROJECT_NAME "$(profile_value PROJECT_NAME deepseek-v4-flash-0731)"
  # ml.config loads generic .env.local values into the ml.cli process. Never
  # let its single-node VLLM_HOST=0.0.0.0 override the private DSpark bind;
  # deliberate cluster overrides use the DSPARK_ prefix, as the port does.
  set_env_value VLLM_HOST "$(profile_value DSPARK_VLLM_HOST "$(profile_file_value VLLM_HOST 127.0.0.1)")"
  # ml.config loads the generic .env.local, which normally contains
  # VLLM_PORT=8000 for single-node backends. Do not let that implicit value
  # override this cluster's dedicated port. A deliberate DSpark override uses
  # DSPARK_VLLM_PORT instead.
  set_env_value VLLM_PORT "$(profile_value DSPARK_VLLM_PORT "$(profile_file_value VLLM_PORT 8888)")"
  set_env_value MAX_MODEL_LEN "$(profile_value MAX_MODEL_LEN 524288)"
  set_env_value MAX_NUM_SEQS "$(profile_value MAX_NUM_SEQS 4)"
  set_env_value MAX_NUM_BATCHED_TOKENS "$(profile_value MAX_NUM_BATCHED_TOKENS 4096)"
  set_env_value LONG_PREFILL_TOKEN_THRESHOLD "$(profile_value LONG_PREFILL_TOKEN_THRESHOLD 1024)"
  set_env_value GPU_MEMORY_UTILIZATION_TEXT "$(profile_value GPU_MEMORY_UTILIZATION_TEXT 0.70)"
  set_env_value MTP_NUM_TOKENS "$(profile_value MTP_NUM_TOKENS 5)"
  set_env_value DEFAULT_THINKING "$(profile_value DEFAULT_THINKING low)"
  set_env_value DSPARK_VLLM_IMAGE "$(profile_value DSPARK_VLLM_IMAGE "${IMAGE_DEFAULT}")"
  set_env_value IMAGE_PYTHON "$(profile_value IMAGE_PYTHON /usr/bin/python3)"
  set_env_value ENABLE_VL_SIDECAR "$(profile_value ENABLE_VL_SIDECAR 0)"
  set_env_value PREPARE_VL_SIDECAR_MODEL "$(profile_value PREPARE_VL_SIDECAR_MODEL 0)"
  set_env_value HF_HUB_OFFLINE "$(profile_value HF_HUB_OFFLINE 1)"
  set_env_value TRANSFORMERS_OFFLINE "$(profile_value TRANSFORMERS_OFFLINE 1)"
  set_env_value HF_HUB_DISABLE_XET "$(profile_value HF_HUB_DISABLE_XET 1)"
  set_env_value DSPARK_MAX_INFLIGHT_PREFILLS "$(profile_value DSPARK_MAX_INFLIGHT_PREFILLS 2)"
  set_env_value DSPARK_ISSUE43_SCHED_DIAG "$(profile_value DSPARK_ISSUE43_SCHED_DIAG 0)"
  set_env_value VLLM_PREFIX_CACHE_RETENTION_INTERVAL "$(profile_value VLLM_PREFIX_CACHE_RETENTION_INTERVAL 4096)"
  set_env_value VLLM_USE_BREAKABLE_CUDAGRAPH "$(profile_value VLLM_USE_BREAKABLE_CUDAGRAPH 0)"
  set_env_value VLLM_ALLOW_LONG_MAX_MODEL_LEN 1
  set_env_value VLLM_USE_B12X_MOE 1
  set_env_value VLLM_USE_B12X_WO_PROJECTION 1
  set_env_value VLLM_USE_FLASHINFER_SAMPLER 1
  set_env_value DSPARK_ENABLE_ISSUE31_GPU_HOTFIX "$(profile_value DSPARK_ENABLE_ISSUE31_GPU_HOTFIX 0)"
  set_env_value DSPARK_SKIP_HOTFIX "$(profile_value DSPARK_SKIP_HOTFIX 0)"
  set_env_value DSPARK_SKIP_ISSUE22_HOTFIX "$(profile_value DSPARK_SKIP_ISSUE22_HOTFIX 0)"
  set_env_value DSPARK_SKIP_SPIN_WAIT_HOTFIX "$(profile_value DSPARK_SKIP_SPIN_WAIT_HOTFIX 0)"
  set_env_value DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX "$(profile_value DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX 0)"

  configure_api_auth

  for key in WORKER_HOST WORKER_SCRIPT_DIR HF_CACHE WORKER_HF_CACHE \
             MASTER_ADDR VLLM_HOST_IP WORKER_VLLM_HOST_IP NCCL_IB_HCA \
             NCCL_SOCKET_IFNAME TP_SOCKET_IFNAME GLOO_SOCKET_IFNAME \
             NCCL_IB_MERGE_NICS NCCL_CROSS_NIC \
             NCCL_IB_GID_AUTO NCCL_IB_GID_INDEX WORKER_NCCL_IB_GID_INDEX \
             MASTER_PORT; do
    set_env_value "${key}" "$(profile_value "${key}" "$(env_value "${key}")")"
  done

  # MASTER_ADDR and the head's vLLM transport address are normally identical.
  if [[ -n "${MASTER_ADDR:-}" && -z "${VLLM_HOST_IP:-}" ]]; then
    set_env_value VLLM_HOST_IP "${MASTER_ADDR}"
  fi

  # The head and Lenovo worker use different account names, so mirroring the
  # head's absolute checkout/cache paths is unsafe. Discover the remote home
  # when passwordless SSH is available, unless explicit paths were supplied.
  local wh rh
  wh="$(worker_host)"
  if ssh -o BatchMode=yes -o ConnectTimeout=8 "${wh}" true >/dev/null 2>&1; then
    rh="$(remote_home)"
    if [[ -z "$(env_value WORKER_SCRIPT_DIR)" ]]; then
      set_env_value WORKER_SCRIPT_DIR "${rh}/deepseek-v4-flash-miaai-runtime"
    fi
    if [[ -z "$(env_value WORKER_HF_CACHE)" ]]; then
      set_env_value WORKER_HF_CACHE "${rh}/.cache/huggingface"
    fi
  else
    warn "Passwordless SSH to ${wh} is not ready; worker-local paths were not auto-discovered"
  fi

  log "Configured ${ENV_FILE}"
  log "Profile: official 0731@${REVISION_DEFAULT:0:12}, 512K, NVFP4 KV, 0.70 memory, low thinking, TP=2"
  log "Ports: raw vLLM $(env_value VLLM_HOST):$(env_value VLLM_PORT); authenticated allow-list proxy $(profile_value DSPARK_PROXY_HOST 0.0.0.0):$(profile_value DSPARK_PROXY_PORT 8000)"
}

show_network() {
  log "Local ConnectX/RoCE mapping"
  if command -v ibdev2netdev >/dev/null 2>&1; then
    ibdev2netdev || true
  else
    warn "ibdev2netdev is not installed"
  fi

  printf '\n'
  log "Recommended network setup"
  cat <<'EOF'
  1. Connect the same physical QSFP port on both machines with one approved cable.
  2. Use NVIDIA Sync > Cluster Assistant to configure and test the two-node link.
     It is safer than guessing which of the four logical interfaces maps to the cable.
  3. On both nodes, verify at least one matching row reports (Up):
       ibdev2netdev
       ip -br -4 address
  4. Verify the dedicated addresses can reach each other with ping.
  5. Set NCCL_IB_HCA to the RoCE name on the left of the selected row and
     NCCL_SOCKET_IFNAME to the Ethernet name on the right.

This installation uses both active f1 lanes. The recipe verifies both links,
their IPv4 addresses, and MTU 9000 before launch. It intentionally does not make
persistent netplan or sudo network changes.
EOF

  if [[ -f "${ENV_FILE}" ]]; then
    local wh
    wh="$(worker_host)"
    printf '\n'
    log "Worker ConnectX/RoCE mapping (${wh})"
    ssh -o BatchMode=yes -o ConnectTimeout=8 "${wh}" 'ibdev2netdev; ip -br -4 address' \
      || warn "Could not inspect worker over passwordless SSH"
  fi
}

validate_value() {
  local key="$1" value
  value="$(env_value "${key}")"
  [[ -n "${value}" ]] || die "${key} is empty in ${ENV_FILE}"
  case "${value}" in
    *-roce-ip|rocepXsYfZ|enpXsYfZnpN|worker-host-or-roce-ip)
      die "${key} still contains the example placeholder: ${value}"
      ;;
  esac
}

check_local() {
  local failed=0 hca nic gid_index dev gid head_ips head_ip index
  local -a hcas nics head_ip_values
  for cmd in docker git ssh rsync nvidia-smi ibdev2netdev ip; do
    if command -v "${cmd}" >/dev/null 2>&1; then
      log "local command OK: ${cmd}"
    else
      warn "local command missing: ${cmd}"
      failed=1
    fi
  done

  if docker compose version >/dev/null 2>&1; then
    log "local Docker Compose OK"
  else
    warn "local Docker Compose is unavailable"
    failed=1
  fi
  if docker info >/dev/null 2>&1; then
    log "local Docker daemon access OK"
  else
    warn "local Docker daemon is not accessible by user $(id -un)"
    warn "run: sudo usermod -aG docker $(id -un), then log out/in or run: newgrp docker"
    failed=1
  fi

  if nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | grep -q 'GB10'; then
    log "local GB10 GPU detected"
  else
    warn "local NVIDIA GB10 was not detected"
    failed=1
  fi

  hca="$(env_value NCCL_IB_HCA)"
  nic="$(profile_value FABRIC_IFNAMES "$(env_value NCCL_SOCKET_IFNAME)")"
  gid_index="$(env_value NCCL_IB_GID_INDEX)"
  head_ips="$(profile_value HEAD_FABRIC_IPS "$(env_value VLLM_HOST_IP)")"
  IFS=',' read -r -a hcas <<<"${hca}"
  IFS=',' read -r -a nics <<<"${nic}"
  IFS=',' read -r -a head_ip_values <<<"${head_ips}"
  [[ "${#nics[@]}" == "${#head_ip_values[@]}" ]] \
    || die "NCCL_SOCKET_IFNAME and HEAD_FABRIC_IPS must contain the same number of entries"
  for dev in "${hcas[@]}"; do
    if ibdev2netdev 2>/dev/null | grep -F "${dev}" | grep -q '(Up)'; then
      log "local RoCE device is Up: ${dev}"
    else
      warn "local RoCE device is not Up: ${dev}"
      failed=1
    fi
    gid="$(cat "/sys/class/infiniband/${dev}/ports/1/gids/${gid_index}" 2>/dev/null || true)"
    if [[ -n "${gid}" && "${gid}" != "0000:0000:0000:0000:0000:0000:0000:0000" ]]; then
      log "local RoCE GID ${gid_index} exists on ${dev}: ${gid}"
    else
      warn "local RoCE GID ${gid_index} is missing/empty on ${dev}"
      failed=1
    fi
  done
  for index in "${!nics[@]}"; do
    dev="${nics[${index}]}"
    head_ip="${head_ip_values[${index}]}"
    if ip -4 -o addr show dev "${dev}" 2>/dev/null | grep -Fq " ${head_ip}/"; then
      log "local fabric address found: ${dev}=${head_ip}"
    else
      warn "local fabric address missing: ${dev}=${head_ip}"
      failed=1
    fi
    if [[ "$(cat "/sys/class/net/${dev}/mtu" 2>/dev/null || true)" != 9000 ]]; then
      warn "local fabric MTU is not 9000: ${dev}"
      failed=1
    fi
  done

  return "${failed}"
}

check_remote() {
  local wh hca nic worker_ip failed=0 dev gid_index worker_ips index
  local -a hcas nics worker_ip_values
  wh="$(worker_host)"
  hca="$(env_value NCCL_IB_HCA)"
  nic="$(profile_value FABRIC_IFNAMES "$(env_value NCCL_SOCKET_IFNAME)")"
  worker_ip="$(env_value WORKER_VLLM_HOST_IP)"
  gid_index="$(env_value NCCL_IB_GID_INDEX)"
  worker_ips="$(profile_value WORKER_FABRIC_IPS "${worker_ip}")"

  if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "${wh}" true; then
    warn "passwordless SSH failed: ${wh}"
    return 1
  fi
  log "passwordless SSH OK: ${wh}"

  if ssh "${wh}" 'docker compose version >/dev/null && docker info >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader | grep -q GB10'; then
    log "worker Docker daemon, Compose, and GB10 GPU detected"
  else
    warn "worker Docker daemon/Compose/GB10 check failed"
    warn "verify on the worker: id -nG; docker info"
    failed=1
  fi
  IFS=',' read -r -a hcas <<<"${hca}"
  IFS=',' read -r -a nics <<<"${nic}"
  IFS=',' read -r -a worker_ip_values <<<"${worker_ips}"
  [[ "${#nics[@]}" == "${#worker_ip_values[@]}" ]] \
    || die "NCCL_SOCKET_IFNAME and WORKER_FABRIC_IPS must contain the same number of entries"
  for dev in "${hcas[@]}"; do
    if ssh "${wh}" "ibdev2netdev | grep -F '${dev}' | grep -q '(Up)'"; then
      log "worker RoCE device is Up: ${dev}"
    else
      warn "worker RoCE device is not Up: ${dev}"
      failed=1
    fi
    if ssh "${wh}" "gid=\$(cat '/sys/class/infiniband/${dev}/ports/1/gids/${gid_index}' 2>/dev/null); test -n \"\$gid\" && test \"\$gid\" != 0000:0000:0000:0000:0000:0000:0000:0000"; then
      log "worker RoCE GID ${gid_index} exists on ${dev}"
    else
      warn "worker RoCE GID ${gid_index} is missing/empty on ${dev}"
      failed=1
    fi
  done
  for index in "${!nics[@]}"; do
    dev="${nics[${index}]}"
    worker_ip="${worker_ip_values[${index}]}"
    if ssh "${wh}" "ip -4 -o addr show dev '${dev}' | grep -Fq ' ${worker_ip}/'"; then
      log "worker fabric address found: ${dev}=${worker_ip}"
    else
      warn "worker fabric address missing: ${dev}=${worker_ip}"
      failed=1
    fi
    if ! ssh "${wh}" "test \"\$(cat '/sys/class/net/${dev}/mtu')\" = 9000"; then
      warn "worker fabric MTU is not 9000: ${dev}"
      failed=1
    fi
    if ping -c 2 -W 2 -M do -s 8972 -I "${dev}" "${worker_ip}" >/dev/null 2>&1; then
      log "jumbo-frame fabric path OK: ${dev} -> ${worker_ip}"
    else
      warn "jumbo-frame fabric path failed: ${dev} -> ${worker_ip}"
      failed=1
    fi
  done
  return "${failed}"
}

check_config() {
  require_checkout
  require_env_file
  for key in WORKER_HOST WORKER_SCRIPT_DIR MASTER_ADDR VLLM_HOST_IP \
             WORKER_VLLM_HOST_IP NCCL_IB_HCA NCCL_SOCKET_IFNAME HF_CACHE \
             WORKER_HF_CACHE DSPARK_MODEL_OFFICIAL DSPARK_REVISION \
             DSPARK_VLLM_IMAGE VLLM_API_KEY; do
    validate_value "${key}"
  done

  [[ "$(env_value DSPARK_MODEL_OFFICIAL)" == "${MODEL_DEFAULT}" ]] \
    || warn "Using non-default checkpoint: $(env_value DSPARK_MODEL_OFFICIAL)"
  [[ "$(env_value DSPARK_REVISION)" == "${REVISION_DEFAULT}" ]] \
    || warn "DSPARK_REVISION differs from MiaAI-Lab's tested official pin"
  [[ "$(env_value DSPARK_VLLM_IMAGE)" == "${IMAGE_DEFAULT}" ]] \
    || warn "DSPARK_VLLM_IMAGE differs from the reviewed immutable Anemll image"
  [[ "$(env_value VLLM_HOST)" == "127.0.0.1" ]] \
    || die "VLLM_HOST must be 127.0.0.1; expose only the safety proxy"
  [[ "$(env_value VLLM_PORT)" == "8888" ]] \
    || warn "Raw vLLM port differs from the reviewed port 8888"
  [[ "$(env_value VLLM_USE_B12X_MOE)" == "1" ]] \
    || die "VLLM_USE_B12X_MOE must be 1; disabling it drops decode toward ~29 tok/s"
  [[ "$(env_value MTP_NUM_TOKENS)" == "5" ]] \
    || warn "MTP_NUM_TOKENS differs from the current verified value 5"
  [[ "$(env_value MAX_MODEL_LEN)" == "524288" ]] \
    || warn "MAX_MODEL_LEN differs from the 512K coexistence profile"
  [[ "$(env_value MAX_NUM_SEQS)" == "4" ]] \
    || warn "MAX_NUM_SEQS differs from the A/B benchmark profile (4)"
  [[ "$(env_value GPU_MEMORY_UTILIZATION_TEXT)" == "0.70" ]] \
    || warn "GPU_MEMORY_UTILIZATION_TEXT differs from the conservative Harness profile (0.70)"
  [[ "$(env_value DEFAULT_THINKING)" =~ ^(off|low)$ ]] \
    || warn "DEFAULT_THINKING is not low/off; coding requests will spend more tokens by default"

  local failed=0
  check_local || failed=1
  check_remote || failed=1

  if [[ -x "${RECIPE_DIR}/validate-dspark-config.sh" ]]; then
    log "Running upstream configuration validator"
    (cd "${RECIPE_DIR}" && ./validate-dspark-config.sh) || failed=1
  fi

  [[ "${failed}" == 0 ]] || die "Preflight failed; fix the warnings above before build/start"
  log "Two-node DSpark preflight passed"
}

memory_available_kib() {
  awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo
}

show_memory() {
  local wh head_kib worker_kib target_gib target_kib harness_gib
  wh="$(worker_host)"
  target_gib="$(profile_value WORKER_AVAILABLE_TARGET_GIB 24)"
  harness_gib="$(profile_value HARNESS_MEMORY_LIMIT_GIB 16)"
  target_kib="$(awk -v gib="${target_gib}" 'BEGIN { printf "%.0f", gib * 1048576 }')"
  head_kib="$(memory_available_kib)"
  worker_kib="$(ssh -o BatchMode=yes -o ConnectTimeout=8 "${wh}" \
    "awk '/^MemAvailable:/ { print \$2; exit }' /proc/meminfo")" \
    || die "Could not read worker memory from ${wh}"

  awk -v head="${head_kib}" -v worker="${worker_kib}" \
      -v target="${target_gib}" -v harness="${harness_gib}" '
    BEGIN {
      printf "[dspark] Head MemAvailable:   %.1f GiB\n", head / 1048576
      printf "[dspark] Worker MemAvailable: %.1f GiB\n", worker / 1048576
      printf "[dspark] Worker target:       %.1f GiB (%g GiB Harness + operating headroom)\n", target, harness
    }
  '

  if (( worker_kib < target_kib )); then
    warn "Worker is below the ${target_gib} GiB coexistence target; do not start the Harness yet"
    return 1
  fi
  log "Worker has enough available memory for the capped ${harness_gib} GiB Harness"
}

start_cluster() {
  check_public_exposure
  configure_api_auth
  check_config
  # Keep the old endpoint alive through every read-only validation. Cut over
  # only once the new checkout, profile, fabric, cache, and image are ready.
  if [[ "${DSPARK_CUTOVER_LEGACY:-0}" == 1 ]]; then
    check_cutover_artifacts
    legacy_stop
  fi
  ensure_gpus_free
  check_gpu_containers
  run_upstream_with_api start-deepseek-v4-flash-dspark.sh
  if ! check_kv_capacity; then
    warn "Stopping the new ranks because the configured 512K capacity was not verified"
    run_upstream_with_api stop-deepseek-v4-flash-dspark.sh || true
    die "512K KV-capacity gate failed; the public proxy was not started"
  fi
  if ! show_memory; then
    warn "The model is running, but the Harness memory target was not met"
  fi
  start_proxy
}

check_cutover_artifacts() {
  local image wh model revision head_cache worker_cache hub_dir failed=0
  image="$(env_value DSPARK_VLLM_IMAGE)"
  wh="$(worker_host)"
  model="$(env_value DSPARK_MODEL_OFFICIAL)"
  revision="$(env_value DSPARK_REVISION)"
  head_cache="$(env_value HF_CACHE)"
  worker_cache="$(env_value WORKER_HF_CACHE)"
  hub_dir="models--${model//\//--}"

  if docker image inspect "${image}" >/dev/null 2>&1; then
    log "cutover image present on head"
  else
    warn "cutover image absent on head: ${image}"
    failed=1
  fi
  if ssh "${wh}" "docker image inspect $(printf '%q' "${image}") >/dev/null 2>&1"; then
    log "cutover image present on worker"
  else
    warn "cutover image absent on worker ${wh}: ${image}"
    failed=1
  fi

  if [[ -d "${head_cache}/hub/${hub_dir}/snapshots/${revision}" ]]; then
    log "pinned model snapshot present on head"
  else
    warn "pinned model snapshot absent on head: ${model}@${revision}"
    failed=1
  fi
  if ssh "${wh}" "test -d $(printf '%q' "${worker_cache}/hub/${hub_dir}/snapshots/${revision}")"; then
    log "pinned model snapshot present on worker"
  else
    warn "pinned model snapshot absent on worker ${wh}: ${model}@${revision}"
    failed=1
  fi

  if [[ "${failed}" != 0 ]]; then
    die "Cutover artifacts are incomplete; run dspark build and dspark download before stopping the legacy service"
  fi
}

check_kv_capacity() {
  local python port max_model_len api_key
  python="$(project_python)"
  port="$(env_value VLLM_PORT)"
  max_model_len="$(env_value MAX_MODEL_LEN)"
  api_key="$(env_value VLLM_API_KEY)"
  DSPARK_METRICS_URL="http://127.0.0.1:${port:-8888}/metrics" \
  DSPARK_REQUIRED_TOKENS="${max_model_len:-524288}" \
  DSPARK_METRICS_API_KEY="${api_key}" \
    "${python}" - <<'PY'
import os
import re
import urllib.request

url = os.environ["DSPARK_METRICS_URL"]
required = int(os.environ["DSPARK_REQUIRED_TOKENS"])
request = urllib.request.Request(
    url,
    headers={"Authorization": f"Bearer {os.environ['DSPARK_METRICS_API_KEY']}"},
)
with urllib.request.urlopen(request, timeout=10) as response:
    metrics = response.read().decode("utf-8", errors="replace")
match = re.search(r"vllm:cache_config_info\{([^}]*)\}", metrics)
if not match:
    raise SystemExit("[dspark] ERROR: vLLM cache_config_info metric is missing")
labels = dict(re.findall(r'(\w+)="([^"]*)"', match.group(1)))
try:
    block_size = int(float(labels["block_size"]))
    gpu_blocks = int(float(labels["num_gpu_blocks"]))
except (KeyError, ValueError) as exc:
    raise SystemExit(f"[dspark] ERROR: invalid cache_config_info labels: {labels}") from exc
capacity = block_size * gpu_blocks
print(
    f"[dspark] Measured KV capacity: {capacity:,} tokens "
    f"({gpu_blocks} blocks x {block_size})"
)
if capacity < required:
    raise SystemExit(
        f"[dspark] ERROR: KV capacity {capacity:,} is below MAX_MODEL_LEN "
        f"{required:,}; raise GPU_MEMORY_UTILIZATION_TEXT toward 0.72"
    )
PY
}

check_public_exposure() {
  local funnel_status
  command -v tailscale >/dev/null 2>&1 || return 0
  funnel_status="$(tailscale funnel status 2>/dev/null || true)"
  if [[ -z "${funnel_status}" ]] && command -v sudo >/dev/null 2>&1; then
    funnel_status="$(sudo -n tailscale funnel status 2>/dev/null || true)"
  fi
  if grep -Eq 'proxy http://127\.0\.0\.1:8888|proxy http://localhost:8888' <<<"${funnel_status}"; then
    die "Tailscale Funnel still targets raw vLLM port 8888. Run: sudo tailscale funnel --https=443 off; sudo tailscale funnel --bg=true 8000"
  fi
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

run_proxy_cli() {
  local action="$1" python proxy_host proxy_port raw_port
  python="$(project_python)"
  proxy_host="$(profile_value DSPARK_PROXY_HOST 0.0.0.0)"
  proxy_port="$(profile_value DSPARK_PROXY_PORT 8000)"
  if [[ -f "${ENV_FILE}" ]]; then
    raw_port="$(env_value VLLM_PORT)"
  else
    raw_port=8888
  fi
  (cd "${PROJECT_ROOT}" && \
    DSPARK_PROXY_HOST="${proxy_host}" \
    DSPARK_PROXY_PORT="${proxy_port}" \
    DSPARK_PROXY_UPSTREAM_URL="http://127.0.0.1:${raw_port:-8888}" \
    "${python}" -m ml.cli dspark-proxy "${action}")
}

start_proxy() {
  run_proxy_cli serve
}

proxy_status() {
  run_proxy_cli status
}

proxy_smoke() {
  run_proxy_cli smoke
}

stop_proxy() {
  run_proxy_cli stop
}

gpu_probe_python() {
  cat <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() is false")
if torch.cuda.device_count() != 1:
    raise RuntimeError(f"expected one GB10, found {torch.cuda.device_count()}")
print(f"CUDA OK: {torch.cuda.get_device_name(0)}; torch={torch.__version__}; cuda={torch.version.cuda}")
PY
}

gpu_probe_argv() {
  local image="$1" image_python python_code
  python_code="$(gpu_probe_python)"
  image_python="$(env_value IMAGE_PYTHON)"
  image_python="${image_python:-/usr/bin/python3}"
  GPU_PROBE_ARGV=(
    docker run --rm --gpus all
    --entrypoint "${image_python}" "${image}" -c "${python_code}"
  )
}

print_cdi_repair() {
  local node="$1"
  warn "CUDA could not initialize in the runtime container on ${node}."
  warn "On ${node}, refresh NVIDIA's device specification, then rerun gpu-check:"
  warn "  sudo systemctl restart nvidia-cdi-refresh.service"
  warn "  nvidia-ctk --debug cdi list"
  warn "If that service does not exist, generate the spec manually:"
  warn "  sudo mkdir -p /var/run/cdi"
  warn "  sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml"
}

check_gpu_containers() {
  require_env_file
  local image wh failed=0 remote_command
  local -a probe
  image="$(env_value DSPARK_VLLM_IMAGE)"
  image="${image:-${IMAGE_DEFAULT}}"
  wh="$(worker_host)"
  gpu_probe_argv "${image}"
  probe=("${GPU_PROBE_ARGV[@]}")

  if ! docker image inspect "${image}" >/dev/null 2>&1; then
    warn "Runtime image is absent on head: ${image}"
    failed=1
  elif "${probe[@]}"; then
    log "head GPU-container initialization passed"
  else
    print_cdi_repair "head ($(hostname))"
    failed=1
  fi

  printf -v remote_command '%q ' "${probe[@]}"
  if ! ssh "${wh}" "docker image inspect $(printf '%q' "${image}") >/dev/null 2>&1"; then
    warn "Runtime image is absent on worker ${wh}: ${image}"
    failed=1
  elif ssh "${wh}" "${remote_command}"; then
    log "worker GPU-container initialization passed: ${wh}"
  else
    print_cdi_repair "worker (${wh})"
    failed=1
  fi

  [[ "${failed}" == 0 ]] || die "GPU-container preflight failed; do not debug NCCL or model loading until both probes pass"
}

competing_workload() {
  local where="$1" wh="${2:-}" output=""
  if [[ "${where}" == local ]]; then
    output="$(nvidia-smi 2>/dev/null | awk '/VLLM::EngineCore|vllm/ { print }' || true)"
  else
    output="$(ssh "${wh}" "nvidia-smi 2>/dev/null | awk '/VLLM::EngineCore|vllm/ { print }'" 2>/dev/null || true)"
  fi
  if [[ -n "${output}" ]]; then
    warn "Competing vLLM process on ${where}:"
    printf '%s\n' "${output}" >&2
    return 0
  fi
  return 1
}

ensure_gpus_free() {
  [[ "${ALLOW_ACTIVE_VLLM:-0}" == 1 ]] && return 0
  local wh busy=0
  wh="$(worker_host)"
  competing_workload local && busy=1
  competing_workload worker "${wh}" && busy=1
  [[ "${busy}" == 0 ]] || die "Stop the existing vLLM services on both nodes, then retry. Override only with ALLOW_ACTIVE_VLLM=1"
}

run_upstream() {
  local script="$1"
  shift
  require_checkout
  require_env_file
  [[ -x "${RECIPE_DIR}/${script}" ]] || die "Upstream script is missing or not executable: ${script}"
  (cd "${RECIPE_DIR}" && "./${script}" "$@")
}

run_upstream_with_api() (
  local api_key
  require_env_file
  api_key="$(env_value VLLM_API_KEY)"
  [[ -n "${api_key}" ]] || die "VLLM_API_KEY is missing from ${ENV_FILE}; run dspark configure"
  # MiaAI's lifecycle scripts source VLLM_API_KEY from .env.dspark, redact it
  # from startup logs, and add it to their own readiness/smoke requests.
  run_upstream "$@"
)

pull_runtime_image() {
  require_env_file
  local image wh head_id worker_id
  image="$(env_value DSPARK_VLLM_IMAGE)"
  [[ "${image}" == *@sha256:* ]] \
    || die "DSPARK_VLLM_IMAGE must include an immutable @sha256 digest"
  wh="$(worker_host)"
  log "Pulling immutable Anemll runtime on head"
  docker pull "${image}"
  log "Pulling the identical runtime on worker ${wh}"
  ssh "${wh}" "docker pull $(printf '%q' "${image}")"
  head_id="$(docker image inspect "${image}" --format '{{.Id}}')"
  worker_id="$(ssh "${wh}" "docker image inspect $(printf '%q' "${image}") --format '{{.Id}}'")"
  [[ -n "${head_id}" && "${head_id}" == "${worker_id}" ]] \
    || die "Pinned image IDs differ between ranks: head=${head_id:-missing}, worker=${worker_id:-missing}"
  log "Pinned runtime verified on both ranks: ${head_id}"
}

legacy_stop() {
  if [[ "${LEGACY_RECIPE_DIR}" == "${RECIPE_DIR}" || ! -d "${LEGACY_RECIPE_DIR}" ]]; then
    log "No separate legacy deployment found"
    return 0
  fi
  if [[ ! -x "${LEGACY_RECIPE_DIR}/stop-deepseek-v4-flash-dspark.sh" ]]; then
    warn "Legacy directory exists without a stop script: ${LEGACY_RECIPE_DIR}"
    return 0
  fi
  log "Stopping the previous Stage-C deployment before cutover"
  (cd "${LEGACY_RECIPE_DIR}" && ./stop-deepseek-v4-flash-dspark.sh)
}

status_cluster() {
  local failed=0
  run_upstream_with_api status-deepseek-v4-flash-dspark.sh || failed=1
  printf '\n'
  proxy_status || failed=1
  return "${failed}"
}

smoke_cluster() {
  run_upstream_with_api smoke-deepseek-v4-flash-dspark.sh
  proxy_smoke
}

stop_cluster() {
  local failed=0
  stop_proxy || failed=1
  run_upstream_with_api stop-deepseek-v4-flash-dspark.sh || failed=1
  return "${failed}"
}

action="${1:-help}"
case "${action}" in
  help|-h|--help) usage ;;
  network) show_network ;;
  bootstrap) bootstrap ;;
  configure) configure ;;
  check) check_config ;;
  setup) bootstrap; configure; check_config ;;
  build) check_config; pull_runtime_image ;;
  download) check_config; run_upstream prepare-dspark-model-cache.sh --official --yes ;;
  start) start_cluster ;;
  status) status_cluster ;;
  gpu-check) check_gpu_containers ;;
  memory) show_memory ;;
  smoke) smoke_cluster ;;
  logs) run_upstream_with_api logs-deepseek-v4-flash-dspark.sh ;;
  stop) stop_cluster ;;
  legacy-stop) legacy_stop ;;
  update)
    require_checkout
    log "Fast-forwarding upstream recipe"
    git -C "${RECIPE_DIR}" pull --ff-only
    ;;
  all)
    bootstrap
    configure
    check_config
    pull_runtime_image
    run_upstream prepare-dspark-model-cache.sh --official --yes
    DSPARK_CUTOVER_LEGACY=1 start_cluster
    ;;
  path)
    printf 'CONFIG_FILE=%s\nRECIPE_DIR=%s\nENV_FILE=%s\n' "${CONFIG_FILE}" "${RECIPE_DIR}" "${ENV_FILE}"
    ;;
  *) die "Unknown action: ${action}. Run '$0 help'." ;;
esac
