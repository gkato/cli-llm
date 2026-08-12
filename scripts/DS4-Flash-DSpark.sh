#!/usr/bin/env bash
# DeepSeek-V4-Flash-0731 on two GB10/DGX Spark-class systems.
#
# This is a thin, opinionated wrapper around the maintained two-node recipe:
# https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark
# It deliberately does not reimplement the patched vLLM image or its worker-first
# launcher. Run every command on the head node.

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_REPO="${DSPARK_UPSTREAM_REPO:-https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark.git}"
RECIPE_DIR="${DSPARK_RECIPE_DIR:-${PROJECT_ROOT}/data/dspark/deepseek-v4-flash-0731}"
ENV_FILE="${DSPARK_ENV_FILE:-${RECIPE_DIR}/.env.dspark}"

# Defaults for this installation. Override any of these in the command environment.
WORKER_HOST_DEFAULT="totalpass@thinkstationpgx-fd9c"
MODEL_DEFAULT="deepseek-ai/DeepSeek-V4-Flash-0731"
SERVED_MODEL_DEFAULT="deepseek-v4-flash-0731"

log() { printf '[dspark] %s\n' "$*"; }
warn() { printf '[dspark] WARNING: %s\n' "$*" >&2; }
die() { printf '[dspark] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
DeepSeek V4 Flash 0731 — two-Spark setup and lifecycle

Run on the NVIDIA/head Spark:

  # 1. Configure the QSFP/ConnectX-7 link with NVIDIA Sync, then inspect it.
  python3 -m ml.cli dspark network

  # 2. Clone the maintained runtime and generate .env.dspark.
  WORKER_HOST=totalpass@thinkstationpgx-fd9c \
  MASTER_ADDR=192.168.100.10 \
  WORKER_VLLM_HOST_IP=192.168.100.11 \
  NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 \
  NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1 \
    python3 -m ml.cli dspark setup

  # 3. Build on both nodes, download/mirror weights, and start worker-first.
  python3 -m ml.cli dspark build
  python3 -m ml.cli dspark download
  python3 -m ml.cli dspark start

  # 4. Operate and validate the endpoint.
  python3 -m ml.cli dspark status
  python3 -m ml.cli dspark smoke
  python3 -m ml.cli dspark logs
  python3 -m ml.cli dspark stop

Actions:
  network    Show ConnectX/RoCE state and configuration guidance (read-only)
  bootstrap  Clone the upstream recipe and create .env.dspark if absent
  configure  Apply environment overrides and the validated 0731 profile
  check      Validate local/remote prerequisites and the generated config
  setup      Run bootstrap, configure, and check
  build      Build/sync the patched Stage-C vLLM image on both nodes
  download   Download and verify weights, then mirror them to the worker
  start      Refuse competing GPU workloads, then launch worker-first
  status     Show head and worker container state
  smoke      Run the upstream OpenAI API smoke test
  logs       Follow the distributed server logs
  stop       Stop the head and worker services
  update     Fast-forward the upstream recipe (never changes this project)
  all        Run setup, build, download, and start
  path       Print the upstream checkout and environment file locations

Configuration environment variables:
  Required for a new setup:
    MASTER_ADDR              Head node's dedicated ConnectX-7 IPv4 address
    WORKER_VLLM_HOST_IP      Worker node's dedicated ConnectX-7 IPv4 address
    NCCL_IB_HCA              RoCE device selected by `ibdev2netdev`
    NCCL_SOCKET_IFNAME       Matching Linux Ethernet interface

  Common overrides:
    WORKER_HOST              SSH target (default: totalpass@thinkstationpgx-fd9c)
    WORKER_SCRIPT_DIR        Dedicated deployment path on the worker
    HF_CACHE                 Head Hugging Face cache
    WORKER_HF_CACHE          Worker Hugging Face cache
    DSPARK_RECIPE_DIR        Upstream checkout on the head
    VLLM_HOST                API bind address (default: 0.0.0.0)
    VLLM_PORT                API port (default: 8888)
    KV_CACHE_DTYPE           nvfp4_ds_mla (default) or fp8_ds_mla
    MAX_MODEL_LEN            Default 1048576; use 200000 for higher concurrency
    MAX_NUM_SEQS             Default 12; upstream short-context profile uses 16
    GPU_MEMORY_UTILIZATION   Default 0.85
    MTP_NUM_TOKENS           Default and recommended value: 5
    NCCL_IB_GID_INDEX        Override only if required (upstream template defaults to 0)
    ALLOW_ACTIVE_VLLM=1      Bypass the competing-workload launch guard

Notes:
  - The interface names above are examples. Use the names printed on your nodes.
  - `setup` never replaces an existing .env.dspark; it updates known keys in place.
  - Build and download are large/slow operations and run only when explicitly requested.
  - NVIDIA has a generic DeepSeek V4 Flash NIM, but the current ml.cli NIM
    backend is single-node. This 0731 TP=2 path uses a custom patched vLLM image.
EOF
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
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

set_if_supplied() {
  local key="$1"
  if [[ -n "${!key:-}" ]]; then
    set_env_value "${key}" "${!key}"
  fi
}

worker_host() {
  if [[ -n "${WORKER_HOST:-}" ]]; then
    printf '%s' "${WORKER_HOST}"
  elif [[ -f "${ENV_FILE}" ]] && [[ -n "$(env_value WORKER_HOST)" ]]; then
    env_value WORKER_HOST
  else
    printf '%s' "${WORKER_HOST_DEFAULT}"
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

  # Use this machine pair by default, while leaving fabric addressing explicit.
  if [[ "$(env_value WORKER_HOST)" == "worker-host-or-roce-ip" || -z "$(env_value WORKER_HOST)" ]]; then
    set_env_value WORKER_HOST "${WORKER_HOST:-${WORKER_HOST_DEFAULT}}"
  fi

  set_env_value DSPARK_MODEL "${DSPARK_MODEL:-${MODEL_DEFAULT}}"
  set_env_value SERVED_MODEL_NAME "${SERVED_MODEL_NAME:-${SERVED_MODEL_DEFAULT}}"
  set_env_value VLLM_HOST "${VLLM_HOST:-0.0.0.0}"
  set_env_value VLLM_PORT "${VLLM_PORT:-8888}"
  set_env_value MAX_MODEL_LEN "${MAX_MODEL_LEN:-1048576}"
  set_env_value MAX_NUM_SEQS "${MAX_NUM_SEQS:-12}"
  set_env_value MAX_NUM_BATCHED_TOKENS "${MAX_NUM_BATCHED_TOKENS:-8192}"
  set_env_value GPU_MEMORY_UTILIZATION "${GPU_MEMORY_UTILIZATION:-0.85}"
  set_env_value MTP_NUM_TOKENS "${MTP_NUM_TOKENS:-5}"
  set_env_value KV_CACHE_DTYPE "${KV_CACHE_DTYPE:-nvfp4_ds_mla}"
  set_env_value VLLM_USE_B12X_MOE 1
  set_env_value VLLM_USE_B12X_WO_PROJECTION 1
  set_env_value VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK 1
  set_env_value VLLM_USE_FLASHINFER_SAMPLER 1

  for key in WORKER_HOST WORKER_SCRIPT_DIR HF_CACHE WORKER_HF_CACHE \
             MASTER_ADDR VLLM_HOST_IP WORKER_VLLM_HOST_IP NCCL_IB_HCA \
             NCCL_SOCKET_IFNAME NCCL_IB_GID_INDEX MASTER_PORT; do
    set_if_supplied "${key}"
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
      set_env_value WORKER_SCRIPT_DIR "${rh}/deepseek-v4-flash-dspark-runtime"
    fi
    if [[ -z "$(env_value WORKER_HF_CACHE)" ]]; then
      set_env_value WORKER_HF_CACHE "${rh}/.cache/huggingface"
    fi
  else
    warn "Passwordless SSH to ${wh} is not ready; worker-local paths were not auto-discovered"
  fi

  log "Configured ${ENV_FILE}"
  log "Profile: official 0731, 1M context, NVFP4 KV, MTP=5, TP=2"
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

Your Lenovo worker currently reported every ConnectX interface as (Down). Do not
build or launch until the cable/port and IP configuration make a matching row Up
on both machines. The recipe intentionally does not make persistent netplan or
sudo network changes.
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
  local failed=0 hca nic dev
  local -a hcas nics
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

  if nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | grep -q 'GB10'; then
    log "local GB10 GPU detected"
  else
    warn "local NVIDIA GB10 was not detected"
    failed=1
  fi

  hca="$(env_value NCCL_IB_HCA)"
  nic="$(env_value NCCL_SOCKET_IFNAME)"
  IFS=',' read -r -a hcas <<<"${hca}"
  IFS=',' read -r -a nics <<<"${nic}"
  for dev in "${hcas[@]}"; do
    if ibdev2netdev 2>/dev/null | grep -F "${dev}" | grep -q '(Up)'; then
      log "local RoCE device is Up: ${dev}"
    else
      warn "local RoCE device is not Up: ${dev}"
      failed=1
    fi
  done
  for dev in "${nics[@]}"; do
    if ip -4 -o addr show dev "${dev}" 2>/dev/null | grep -q 'inet '; then
      log "local fabric IPv4 found on ${dev}"
    else
      warn "local fabric interface has no IPv4 address: ${dev}"
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
  local wh hca nic worker_ip failed=0 dev
  local -a hcas nics
  wh="$(worker_host)"
  hca="$(env_value NCCL_IB_HCA)"
  nic="$(env_value NCCL_SOCKET_IFNAME)"
  worker_ip="$(env_value WORKER_VLLM_HOST_IP)"

  if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "${wh}" true; then
    warn "passwordless SSH failed: ${wh}"
    return 1
  fi
  log "passwordless SSH OK: ${wh}"

  if ssh "${wh}" 'docker compose version >/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader | grep -q GB10'; then
    log "worker Docker Compose and GB10 GPU detected"
  else
    warn "worker Docker Compose/GB10 check failed"
    failed=1
  fi
  IFS=',' read -r -a hcas <<<"${hca}"
  IFS=',' read -r -a nics <<<"${nic}"
  for dev in "${hcas[@]}"; do
    if ssh "${wh}" "ibdev2netdev | grep -F '${dev}' | grep -q '(Up)'"; then
      log "worker RoCE device is Up: ${dev}"
    else
      warn "worker RoCE device is not Up: ${dev}"
      failed=1
    fi
  done
  for dev in "${nics[@]}"; do
    if ssh "${wh}" "ip -4 -o addr show dev '${dev}' | grep -q 'inet '"; then
      log "worker fabric IPv4 found on ${dev}"
    else
      warn "worker fabric interface has no IPv4 address: ${dev}"
      failed=1
    fi
    if ! ssh "${wh}" "test \"\$(cat '/sys/class/net/${dev}/mtu')\" = 9000"; then
      warn "worker fabric MTU is not 9000: ${dev}"
      failed=1
    fi
  done
  if ping -c 2 -W 2 "${worker_ip}" >/dev/null 2>&1; then
    log "head can reach worker fabric IP: ${worker_ip}"
  else
    warn "head cannot ping worker fabric IP: ${worker_ip}"
    failed=1
  fi
  return "${failed}"
}

check_config() {
  require_checkout
  require_env_file
  for key in WORKER_HOST WORKER_SCRIPT_DIR MASTER_ADDR VLLM_HOST_IP \
             WORKER_VLLM_HOST_IP NCCL_IB_HCA NCCL_SOCKET_IFNAME HF_CACHE \
             WORKER_HF_CACHE DSPARK_MODEL KV_CACHE_DTYPE; do
    validate_value "${key}"
  done

  [[ "$(env_value DSPARK_MODEL)" == "${MODEL_DEFAULT}" ]] \
    || warn "Using non-default checkpoint: $(env_value DSPARK_MODEL)"
  [[ "$(env_value VLLM_USE_B12X_MOE)" == "1" ]] \
    || die "VLLM_USE_B12X_MOE must be 1; disabling it drops decode toward ~29 tok/s"
  [[ "$(env_value MTP_NUM_TOKENS)" == "5" ]] \
    || warn "MTP_NUM_TOKENS differs from the current verified value 5"

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

action="${1:-help}"
case "${action}" in
  help|-h|--help) usage ;;
  network) show_network ;;
  bootstrap) bootstrap ;;
  configure) configure ;;
  check) check_config ;;
  setup) bootstrap; configure; check_config ;;
  build) check_config; run_upstream build-dspark-vllm-runtime.sh ;;
  download) check_config; run_upstream prepare-dspark-model-cache.sh ;;
  start) check_config; ensure_gpus_free; run_upstream start-deepseek-v4-flash-dspark.sh ;;
  status) run_upstream status-deepseek-v4-flash-dspark.sh ;;
  smoke) run_upstream smoke-deepseek-v4-flash-dspark.sh ;;
  logs) run_upstream logs-deepseek-v4-flash-dspark.sh ;;
  stop) run_upstream stop-deepseek-v4-flash-dspark.sh ;;
  update)
    require_checkout
    log "Fast-forwarding upstream recipe"
    git -C "${RECIPE_DIR}" pull --ff-only
    ;;
  all)
    bootstrap
    configure
    check_config
    run_upstream build-dspark-vllm-runtime.sh
    run_upstream prepare-dspark-model-cache.sh
    ensure_gpus_free
    run_upstream start-deepseek-v4-flash-dspark.sh
    ;;
  path)
    printf 'RECIPE_DIR=%s\nENV_FILE=%s\n' "${RECIPE_DIR}" "${ENV_FILE}"
    ;;
  *) die "Unknown action: ${action}. Run '$0 help'." ;;
esac
