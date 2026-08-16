#!/usr/bin/env bash
# DeepSeek-V4-Flash-0731 on two GB10/DGX Spark-class systems.
#
# This is a thin, opinionated wrapper around the maintained two-node recipe:
# https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark
# It deliberately does not reimplement the patched vLLM image or its worker-first
# launcher. Run every command on the head node.

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_REPO="${DSPARK_UPSTREAM_REPO:-https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark.git}"
RECIPE_DIR="${DSPARK_RECIPE_DIR:-${PROJECT_ROOT}/data/dspark/deepseek-v4-flash-0731}"
ENV_FILE="${DSPARK_ENV_FILE:-${RECIPE_DIR}/.env.dspark}"
CONFIG_FILE="${DSPARK_CONFIG_FILE:-${PROJECT_ROOT}/config/dspark-spark4e89-thinkstationpgx.env}"
PROJECT_ENV_FILE="${DSPARK_PROJECT_ENV_FILE:-${PROJECT_ROOT}/.env.local}"
AUTH_COMPOSE_FILE="${DSPARK_AUTH_COMPOSE_FILE:-${RECIPE_DIR}/docker-compose.dspark.auth.yml}"

# Defaults for this installation. Override any of these in the command environment.
WORKER_HOST_DEFAULT="totalpass@192.168.177.11"
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

  # 2. Clone/configure from the committed machine profile and check both nodes.
  python3 -m ml.cli dspark setup

  # 3. Build on both nodes, download/mirror weights, and start worker-first.
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
  build      Build/sync the patched Stage-C vLLM image on both nodes
  download   Download and verify weights, then mirror them to the worker
  start      Refuse competing GPU workloads, then launch worker-first
  status     Show head and worker container state
  gpu-check  Prove PyTorch can initialize GB10 inside the runtime on both nodes
  memory     Show head/worker MemAvailable and verify the Harness budget
  smoke      Run the upstream OpenAI API smoke test
  logs       Follow the distributed server logs
  stop       Stop the head and worker services
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
    VLLM_HOST                API bind address (default: 0.0.0.0)
    DSPARK_VLLM_PORT         DSpark API port override (profile default: 8888)
    KV_CACHE_DTYPE           nvfp4_ds_mla (default) or fp8_ds_mla
    MAX_MODEL_LEN            Profile default 262144 (256K)
    MAX_NUM_SEQS             Profile default 4
    MAX_NUM_BATCHED_TOKENS   Profile default 4096
    GPU_MEMORY_UTILIZATION   Profile default 0.72 on both TP ranks
    MTP_NUM_TOKENS           Default and recommended value: 5
    WORKER_AVAILABLE_TARGET_GIB  Required MemAvailable after model start (24)
    DSPARK_USE_BUILTIN_DOCKERFILE_FRONTEND  Avoid docker/dockerfile:1 pull (default: 1)
    NCCL_IB_GID_INDEX        RoCE v2/IPv4 GID index (profile: 3)
    ALLOW_ACTIVE_VLLM=1      Bypass the competing-workload launch guard

Notes:
  - Cluster addresses, interfaces, paths, and memory limits are committed in
    config/dspark-spark4e89-thinkstationpgx.env.
  - Process-environment values override the committed profile for one command.
  - API authentication is required and sourced from API_KEY in .env.local.
  - `setup` never replaces an existing .env.dspark; it updates known keys in place.
  - Build and download are large/slow operations and run only when explicitly requested.
  - NVIDIA has a generic DeepSeek V4 Flash NIM, but the current ml.cli NIM
    backend is single-node. This 0731 TP=2 path uses a custom patched vLLM image.
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
  local api_key base_compose tmp
  api_key="${API_KEY:-$(project_env_value API_KEY)}"
  [[ -n "${api_key}" ]] || die "API_KEY is missing from ${PROJECT_ENV_FILE}"
  [[ "${api_key}" =~ ^[A-Za-z0-9._~-]+$ ]] \
    || die "API_KEY may contain only letters, digits, dot, underscore, tilde, and hyphen"

  set_env_value VLLM_API_KEY "${api_key}"
  chmod 600 "${ENV_FILE}"

  base_compose="${RECIPE_DIR}/docker-compose.dspark.yml"
  [[ -f "${base_compose}" ]] || die "Missing upstream Compose file: ${base_compose}"
  tmp="$(mktemp "${AUTH_COMPOSE_FILE}.XXXXXX")"
  if ! awk '
    !added && /^    environment:[[:space:]]*$/ {
      print
      print "      VLLM_API_KEY: \"${VLLM_API_KEY:?VLLM_API_KEY must be set}\""
      added = 1
      next
    }
    { print }
    END { if (!added) exit 42 }
  ' "${base_compose}" >"${tmp}"; then
    rm -f "${tmp}"
    die "Could not add VLLM_API_KEY to the generated Compose file"
  fi
  chmod 644 "${tmp}"
  mv "${tmp}" "${AUTH_COMPOSE_FILE}"
  log "vLLM API authentication enabled from ${PROJECT_ENV_FILE} (key not displayed)"
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

  set_env_value DSPARK_MODEL "$(profile_value DSPARK_MODEL "${MODEL_DEFAULT}")"
  set_env_value SERVED_MODEL_NAME "$(profile_value SERVED_MODEL_NAME "${SERVED_MODEL_DEFAULT}")"
  set_env_value VLLM_HOST "$(profile_value VLLM_HOST 0.0.0.0)"
  # ml.config loads the generic .env.local, which normally contains
  # VLLM_PORT=8000 for single-node backends. Do not let that implicit value
  # override this cluster's dedicated port. A deliberate DSpark override uses
  # DSPARK_VLLM_PORT instead.
  set_env_value VLLM_PORT "$(profile_value DSPARK_VLLM_PORT "$(profile_file_value VLLM_PORT 8888)")"
  set_env_value MAX_MODEL_LEN "$(profile_value MAX_MODEL_LEN 262144)"
  set_env_value MAX_NUM_SEQS "$(profile_value MAX_NUM_SEQS 4)"
  set_env_value MAX_NUM_BATCHED_TOKENS "$(profile_value MAX_NUM_BATCHED_TOKENS 4096)"
  set_env_value GPU_MEMORY_UTILIZATION "$(profile_value GPU_MEMORY_UTILIZATION 0.72)"
  set_env_value MTP_NUM_TOKENS "$(profile_value MTP_NUM_TOKENS 5)"
  set_env_value KV_CACHE_DTYPE "$(profile_value KV_CACHE_DTYPE nvfp4_ds_mla)"
  set_env_value VLLM_USE_B12X_MOE 1
  set_env_value VLLM_USE_B12X_WO_PROJECTION 1
  set_env_value VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK 1
  set_env_value VLLM_USE_FLASHINFER_SAMPLER 1

  configure_api_auth

  for key in WORKER_HOST WORKER_SCRIPT_DIR HF_CACHE WORKER_HF_CACHE \
             MASTER_ADDR VLLM_HOST_IP WORKER_VLLM_HOST_IP NCCL_IB_HCA \
             NCCL_SOCKET_IFNAME NCCL_IB_GID_INDEX MASTER_PORT; do
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
      set_env_value WORKER_SCRIPT_DIR "${rh}/deepseek-v4-flash-dspark-runtime"
    fi
    if [[ -z "$(env_value WORKER_HF_CACHE)" ]]; then
      set_env_value WORKER_HF_CACHE "${rh}/.cache/huggingface"
    fi
  else
    warn "Passwordless SSH to ${wh} is not ready; worker-local paths were not auto-discovered"
  fi

  log "Configured ${ENV_FILE}"
  log "Profile: official 0731, 256K context, NVFP4 KV, 0.72 memory, MTP=5, TP=2"
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
  nic="$(env_value NCCL_SOCKET_IFNAME)"
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
  nic="$(env_value NCCL_SOCKET_IFNAME)"
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
             WORKER_HF_CACHE DSPARK_MODEL KV_CACHE_DTYPE VLLM_API_KEY; do
    validate_value "${key}"
  done

  [[ "$(env_value DSPARK_MODEL)" == "${MODEL_DEFAULT}" ]] \
    || warn "Using non-default checkpoint: $(env_value DSPARK_MODEL)"
  [[ "$(env_value VLLM_USE_B12X_MOE)" == "1" ]] \
    || die "VLLM_USE_B12X_MOE must be 1; disabling it drops decode toward ~29 tok/s"
  [[ "$(env_value MTP_NUM_TOKENS)" == "5" ]] \
    || warn "MTP_NUM_TOKENS differs from the current verified value 5"
  [[ "$(env_value MAX_MODEL_LEN)" == "262144" ]] \
    || warn "MAX_MODEL_LEN differs from the 256K coexistence profile"
  [[ "$(env_value GPU_MEMORY_UTILIZATION)" == "0.72" ]] \
    || warn "GPU_MEMORY_UTILIZATION differs from the Harness coexistence profile (0.72)"

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
  configure_api_auth
  check_config
  ensure_gpus_free
  check_gpu_containers
  run_upstream_with_api start-deepseek-v4-flash-dspark.sh
  if ! show_memory; then
    warn "The model is running, but the Harness memory target was not met"
  fi
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
  local image="$1" python_code
  python_code="$(gpu_probe_python)"
  GPU_PROBE_ARGV=(
    docker run --rm --gpus all
    --entrypoint bash "${image}" -lc
    "export PATH=\"/opt/env/bin:/opt/env/nvvm/bin:/opt/env/targets/sbsa-linux/nvvm/bin:\${PATH:-}\"; export LD_LIBRARY_PATH=\"/opt/env/lib:/opt/env/targets/sbsa-linux/lib:\${LD_LIBRARY_PATH:-}\"; exec /opt/env/bin/python -c $(printf '%q' "${python_code}")"
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
  image="${image:-vllm-dspark-runtime:dspark-nvfp4-stage-c}"
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
  local port api_key curl_home
  require_env_file
  port="$(env_value VLLM_PORT)"
  port="${port:-8888}"
  api_key="$(env_value VLLM_API_KEY)"
  [[ -n "${api_key}" ]] || die "VLLM_API_KEY is missing from ${ENV_FILE}; run dspark configure"
  [[ -f "${AUTH_COMPOSE_FILE}" ]] || die "Missing authenticated Compose file; run dspark configure"

  # Upstream readiness and smoke scripts call curl directly. Give those calls
  # an isolated curl config so the Bearer token is never placed in their
  # command lines or printed in rendered configuration.
  curl_home="$(mktemp -d /tmp/ml-compute-dspark-curl.XXXXXX)"
  chmod 700 "${curl_home}"
  umask 077
  printf 'header = "Authorization: Bearer %s"\n' "${api_key}" >"${curl_home}/.curlrc"
  trap 'rm -rf "${curl_home}"' EXIT
  export CURL_HOME="${curl_home}"
  export COMPOSE_FILE="${AUTH_COMPOSE_FILE}"
  export API_URL="${API_URL:-http://127.0.0.1:${port}/v1/models}"
  export CHAT_URL="${CHAT_URL:-http://127.0.0.1:${port}/v1/chat/completions}"
  run_upstream "$@"
)

run_upstream_build() (
  local dockerfile backup_file patched_file frontend_mode
  dockerfile="${RECIPE_DIR}/recipe/Dockerfile.dspark-runtime-overlay"
  frontend_mode="$(profile_value DSPARK_USE_BUILTIN_DOCKERFILE_FRONTEND 1)"

  if [[ "${frontend_mode}" != 1 ]] || ! head -n 1 "${dockerfile}" | grep -q '^# syntax=docker/dockerfile:1'; then
    run_upstream build-dspark-vllm-runtime.sh
    return
  fi

  # The upstream overlay Dockerfile declares docker/dockerfile:1, causing an
  # extra Docker Hub pull. Omitting the directive makes BuildKit use its bundled
  # Dockerfile frontend. Patch only for this invocation; the EXIT trap restores
  # the pristine upstream file even when a build is interrupted. The temporary
  # file is also what the upstream rsync sends to the worker, so both nodes use
  # the same frontend without permanently dirtying the head checkout.
  backup_file="$(mktemp /tmp/ml-compute-dspark-dockerfile.XXXXXX)"
  patched_file="$(mktemp /tmp/ml-compute-dspark-dockerfile-patched.XXXXXX)"
  cp -p "${dockerfile}" "${backup_file}"
  trap 'mv "${backup_file}" "${dockerfile}"; if [[ -e "${patched_file}" ]]; then rm -f "${patched_file}"; fi' EXIT

  tail -n +2 "${dockerfile}" >"${patched_file}"
  chmod 644 "${patched_file}"
  mv "${patched_file}" "${dockerfile}"

  log "Building with Docker's bundled Dockerfile frontend (external syntax directive disabled temporarily)"
  run_upstream build-dspark-vllm-runtime.sh
)

action="${1:-help}"
case "${action}" in
  help|-h|--help) usage ;;
  network) show_network ;;
  bootstrap) bootstrap ;;
  configure) configure ;;
  check) check_config ;;
  setup) bootstrap; configure; check_config ;;
  build) check_config; run_upstream_build ;;
  download) check_config; run_upstream prepare-dspark-model-cache.sh ;;
  start) start_cluster ;;
  status) run_upstream_with_api status-deepseek-v4-flash-dspark.sh ;;
  gpu-check) check_gpu_containers ;;
  memory) show_memory ;;
  smoke) run_upstream_with_api smoke-deepseek-v4-flash-dspark.sh ;;
  logs) run_upstream_with_api logs-deepseek-v4-flash-dspark.sh ;;
  stop) run_upstream_with_api stop-deepseek-v4-flash-dspark.sh ;;
  update)
    require_checkout
    log "Fast-forwarding upstream recipe"
    git -C "${RECIPE_DIR}" pull --ff-only
    ;;
  all)
    bootstrap
    configure
    check_config
    run_upstream_build
    run_upstream prepare-dspark-model-cache.sh
    start_cluster
    ;;
  path)
    printf 'CONFIG_FILE=%s\nRECIPE_DIR=%s\nENV_FILE=%s\n' "${CONFIG_FILE}" "${RECIPE_DIR}" "${ENV_FILE}"
    ;;
  *) die "Unknown action: ${action}. Run '$0 help'." ;;
esac
