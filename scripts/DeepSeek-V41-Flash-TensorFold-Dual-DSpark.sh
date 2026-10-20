#!/usr/bin/env bash
# Pinned lifecycle adapter for MiaAI-Lab's DeepSeek V4.1 Flash TensorFold recipe.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${DSV41_TF_CONFIG_FILE:-${ROOT}/config/dspark-deepseek-v41-flash-tensorfold.env}"
RUNTIME="${DSV41_TF_RUNTIME_DIR:-${ROOT}/data/dspark/deepseek-v41-flash-tensorfold}"
RECIPE="${DSV41_TF_RECIPE_DIR:-${RUNTIME}/recipe}"
ENV_FILE="${RECIPE}/.env"
PROJECT_ENV_FILE="${DSV41_TF_PROJECT_ENV_FILE:-${ROOT}/.env.local}"

PINNED_RECIPE_REF=31eea4608f8bb292a92e2f7a1db400e928620c78
PINNED_MODEL_REF=6d54be4fca338d79b17c3b03cc9a423bb3ac6718
PINNED_ENGRAM_REF=1bae0e1088993b9b82de08f59de5be5a34e42170
PINNED_ENGINE_REF=629c524a0ea57512a5342abb498d1b7c25353343
PINNED_INPUTS_REF=59402086381a42c74d53053443224eec2790b4c5
PINNED_KIT_REF=abe07c7d509a0199cc7fec15263be2aa52ac652f
PINNED_BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
PINNED_LOCAL_IMAGE=tensorfold-dsv41:exl3-tp2-3fd5edeb40cf

log() { printf '[deepseek-v41-tensorfold] %s\n' "$*"; }
warn() { printf '[deepseek-v41-tensorfold] WARNING: %s\n' "$*" >&2; }
die() { printf '[deepseek-v41-tensorfold] ERROR: %s\n' "$*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"; }

[[ -f "$PROFILE" ]] || die "Missing profile: $PROFILE"
set -a
# shellcheck disable=SC1090
source "$PROFILE"
set +a

usage() {
  cat <<'EOF'
DeepSeek V4.1 Flash 1017 EXL3/TensorFold on two DGX Sparks

Run on the head/rank-0 Spark:
  python3 -m ml.cli deepseek-v41-flash-tensorfold setup
  python3 -m ml.cli deepseek-v41-flash-tensorfold prepare
  python3 -m ml.cli deepseek-v41-flash-tensorfold start
  python3 -m ml.cli deepseek-v41-flash-tensorfold smoke

Actions: bootstrap, update, configure, check, setup, prepare, download, build,
         dry-run, start, restart, stop, status, memory, logs, logs-worker,
         smoke, all, path, help

The first preparation needs about 505 GB on the head, 310 GB on the worker,
and a read-only NFS export of the head Hugging Face cache to the worker. The
TensorFold image is built locally from digest-pinned and commit-pinned inputs.
EOF
}

project_python() {
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    printf '%s' "${VIRTUAL_ENV}/bin/python"
  elif [[ -x "${ROOT}/venv/bin/python" ]]; then
    printf '%s' "${ROOT}/venv/bin/python"
  else
    printf '%s' python3
  fi
}

project_api_key() {
  printf '%s' "${API_KEY:-$(sed -n 's/^API_KEY=//p' "$PROJECT_ENV_FILE" 2>/dev/null | tail -1)}"
}

check_proxy_key() {
  [[ -n "$(project_api_key)" ]] || die "API_KEY is missing from $PROJECT_ENV_FILE"
}

proxy() {
  local action="$1" python
  python="$(project_python)"
  (cd "$ROOT" && \
    API_KEY="$(project_api_key)" \
    DSPARK_PROXY_HOST="$DSPARK_PROXY_HOST" \
    DSPARK_PROXY_PORT="$DSPARK_PROXY_PORT" \
    DSPARK_PROXY_UPSTREAM_URL="http://127.0.0.1:${DSV41_TF_PORT}" \
    "$python" -m ml.cli dspark-proxy "$action")
}

worker() {
  ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
    "$DSV41_TF_WORKER" "$@"
}

validate_profile() {
  [[ "$DSV41_TF_REF" == "$PINNED_RECIPE_REF" ]] || die "Recipe revision is not the reviewed pin"
  [[ "$DSV41_TF_MODEL_REVISION" == "$PINNED_MODEL_REF" ]] || die "Model revision is not the reviewed pin"
  [[ "$DSV41_TF_ENGRAM_REVISION" == "$PINNED_ENGRAM_REF" ]] || die "Engram revision is not the reviewed pin"
  [[ "$DSV41_TF_ENGINE_REF" == "$PINNED_ENGINE_REF" ]] || die "TensorFold engine revision is not the reviewed pin"
  [[ "$DSV41_TF_INPUTS_REF" == "$PINNED_INPUTS_REF" ]] || die "TensorFold input-tool revision is not the reviewed pin"
  [[ "$DSV41_TF_KIT_REF" == "$PINNED_KIT_REF" ]] || die "Kernel-kit revision is not the reviewed pin"
  [[ "$DSV41_TF_BASE_IMAGE" == "$PINNED_BASE_IMAGE" ]] || die "NVIDIA base image is not digest-pinned"
  [[ "$DSV41_TF_IMAGE" == "$PINNED_LOCAL_IMAGE" ]] || die "Local image tag does not match the pinned build inputs"
  [[ "$DSV41_TF_PULL" == 0 ]] || die "PULL must remain 0; build from pinned inputs instead of a mutable image tag"
  [[ "$DSV41_TF_HOST" == 127.0.0.1 ]] || die "Raw TensorFold must remain on 127.0.0.1"
  [[ "$DSV41_TF_PORT" == 8888 ]] || die "Raw TensorFold must remain on reviewed port 8888"
  [[ "$DSV41_TF_ABLIT" == 0 ]] || die "The gated abliterated overlay is intentionally disabled"
  [[ "$DSV41_TF_CONTEXT" =~ ^[0-9]+$ && "$DSV41_TF_CONTEXT" -le 1048576 ]] \
    || die "Context must be an integer no greater than 1048576"
  [[ "$DSV41_TF_PARALLEL" =~ ^[1-4]$ ]] || die "Parallel streams must be between 1 and 4"
}

recipe_is_clean() {
  git -C "$RECIPE" diff --quiet -- && git -C "$RECIPE" diff --cached --quiet --
}

sync_recipe() {
  validate_profile
  need_cmd git
  mkdir -p "$RUNTIME"
  if [[ ! -d "$RECIPE/.git" ]]; then
    [[ ! -e "$RECIPE" ]] || die "Recipe path exists but is not a git checkout: $RECIPE"
    git clone --filter=blob:none --no-checkout "$DSV41_TF_REPO" "$RECIPE"
  elif ! recipe_is_clean; then
    die "Recipe has local tracked changes; preserve or stash them before updating"
  fi
  git -C "$RECIPE" fetch --depth 1 origin "$DSV41_TF_REF"
  git -C "$RECIPE" checkout --detach FETCH_HEAD
  [[ "$(git -C "$RECIPE" rev-parse HEAD)" == "$DSV41_TF_REF" ]] \
    || die "Recipe did not resolve to pinned revision $DSV41_TF_REF"
  log "Pinned upstream recipe at ${DSV41_TF_REF:0:12}"
}

check_recipe_pin() {
  [[ -d "$RECIPE/.git" ]] || die "Recipe missing; run deepseek-v41-flash-tensorfold setup"
  [[ "$(git -C "$RECIPE" rev-parse HEAD 2>/dev/null)" == "$DSV41_TF_REF" ]] \
    || die "Recipe is not at pinned revision $DSV41_TF_REF; run update"
  recipe_is_clean || die "Recipe has local tracked changes; preserving them"
}

configure_recipe() {
  local tmp
  validate_profile
  check_recipe_pin
  tmp="$(mktemp "${ENV_FILE}.XXXXXX")"
  cat >"$tmp" <<EOF
WORKER=${DSV41_TF_WORKER}
FABRIC_PEER=${DSV41_TF_FABRIC_PEER}
MASTER_ADDR=${DSV41_TF_MASTER_ADDR}
MASTER_PORT=${DSV41_TF_MASTER_PORT}
MODEL_ID=${DSV41_TF_MODEL_ID}
MODEL_REVISION=${DSV41_TF_MODEL_REVISION}
ENGRAM_ID=${DSV41_TF_ENGRAM_ID}
ENGRAM_REVISION=${DSV41_TF_ENGRAM_REVISION}
TF_REPO=${DSV41_TF_ENGINE_REPO}
TF_REF=${DSV41_TF_ENGINE_REF}
INPUTS_REF=${DSV41_TF_INPUTS_REF}
KIT_REPO=${DSV41_TF_KIT_REPO}
KIT_REF=${DSV41_TF_KIT_REF}
ZIG_VERSION=${DSV41_TF_ZIG_VERSION}
ZIG_SHA256=${DSV41_TF_ZIG_SHA256}
BASE_IMAGE=${DSV41_TF_BASE_IMAGE}
IMAGE=${DSV41_TF_IMAGE}
GHCR_IMAGE=${DSV41_TF_GHCR_IMAGE}
IMAGE_TAG=${DSV41_TF_IMAGE_TAG}
IMAGE_DIGEST=${DSV41_TF_IMAGE_DIGEST}
CONTAINER_NAME=${DSV41_TF_CONTAINER}
SERVED_NAME=${DSV41_TF_SERVED_NAME}
HOST=${DSV41_TF_HOST}
PORT=${DSV41_TF_PORT}
CONTEXT=${DSV41_TF_CONTEXT}
PARALLEL=${DSV41_TF_PARALLEL}
ROUND_ROWS=${DSV41_TF_ROUND_ROWS}
MAX_TOKENS=${DSV41_TF_MAX_TOKENS}
THINKING=${DSV41_TF_THINKING}
REASONING_EFFORT=${DSV41_TF_REASONING_EFFORT}
THINKING_BUDGET=${DSV41_TF_THINKING_BUDGET}
TEMPERATURE=${DSV41_TF_TEMPERATURE}
TOP_P=${DSV41_TF_TOP_P}
TOP_K=${DSV41_TF_TOP_K}
DRAFTS=${DSV41_TF_DRAFTS}
RDMA=${DSV41_TF_RDMA}
GRAPHS=${DSV41_TF_GRAPHS}
SIDE=${DSV41_TF_SIDE}
PREFETCH=${DSV41_TF_PREFETCH}
ENGRAM_AIO=${DSV41_TF_ENGRAM_AIO}
REPLAY_FLOOR=${DSV41_TF_REPLAY_FLOOR}
PREFILL_INTERLEAVE_MS=${DSV41_TF_PREFILL_INTERLEAVE_MS}
HF_CACHE=${DSV41_TF_HF_CACHE}
NFS_PATH=${DSV41_TF_NFS_PATH}
NFS_SERVER=${DSV41_TF_NFS_SERVER}
NFS_VOLUME=${DSV41_TF_NFS_VOLUME}
HEAD_ENGRAM_DIR=${DSV41_TF_HEAD_ENGRAM_DIR}
HEAD_NFS_SERVER=${DSV41_TF_HEAD_NFS_SERVER}
HEAD_NFS_VOLUME=${DSV41_TF_HEAD_NFS_VOLUME}
MIN_AVAIL_GIB=${DSV41_TF_MIN_AVAIL_GIB}
MIN_AVAIL_WAIT_S=${DSV41_TF_MIN_AVAIL_WAIT_S}
MEMGUARD_GIB=${DSV41_TF_MEMGUARD_GIB}
MEMGUARD_HOLD_S=${DSV41_TF_MEMGUARD_HOLD_S}
MEMGUARD_HARD_GIB=${DSV41_TF_MEMGUARD_HARD_GIB}
ABLIT=${DSV41_TF_ABLIT}
PULL=${DSV41_TF_PULL}
PREPARE=${DSV41_TF_PREPARE}
HF_HUB_OFFLINE=${DSV41_TF_HF_HUB_OFFLINE}
API_KEY=0
EOF
  chmod 0600 "$tmp"
  mv "$tmp" "$ENV_FILE"
  log "Configured pinned recipe at $RECIPE"
}

run_recipe() {
  (cd "$RECIPE" && env -u API_KEY "$@")
}

check_cluster() {
  validate_profile
  check_recipe_pin
  check_proxy_key
  need_cmd docker
  need_cmd ssh
  need_cmd hf
  docker info >/dev/null 2>&1 || die "Docker is not ready on the head Spark"
  worker "docker info >/dev/null" || die "Docker or SSH is not ready on $DSV41_TF_WORKER"
  log "Recipe, Docker, SSH, Hugging Face CLI, and proxy key are ready"
  warn "Preparation still requires the NFS export documented in README.md"
}

prepare_recipe() {
  configure_recipe
  run_recipe ./scripts/prepare.sh "$@"
}

start_service() {
  local mode="${1:-start}"
  configure_recipe
  check_proxy_key
  proxy stop || true
  if [[ "$mode" == restart ]]; then
    run_recipe ./start.sh restart
  else
    run_recipe ./start.sh
  fi
  proxy serve
  proxy smoke
  log "Authenticated API: http://127.0.0.1:${DSPARK_PROXY_PORT}/v1"
}

memory_status() {
  printf '%s\n' 'Head Spark:'
  free -h
  docker stats --no-stream "$DSV41_TF_CONTAINER" 2>/dev/null || true
  printf '\nWorker Spark (%s):\n' "$DSV41_TF_WORKER"
  worker "free -h; docker stats --no-stream '$DSV41_TF_CONTAINER' 2>/dev/null || true"
}

action="${1:-help}"
case "$action" in
  help|-h|--help) usage ;;
  bootstrap) sync_recipe ;;
  update) sync_recipe; configure_recipe ;;
  configure) configure_recipe ;;
  check) check_cluster ;;
  setup) sync_recipe; configure_recipe; check_cluster ;;
  prepare|download) prepare_recipe ;;
  build) prepare_recipe --rebuild ;;
  dry-run) configure_recipe; run_recipe env DRY_RUN=1 ./start.sh ;;
  start) start_service start ;;
  restart) start_service restart ;;
  stop) proxy stop || true; check_recipe_pin; run_recipe ./stop.sh ;;
  status)
    curl -fsS "http://127.0.0.1:${DSV41_TF_PORT}/health" || true
    proxy status
    ;;
  memory) memory_status ;;
  logs) docker logs -f "$DSV41_TF_CONTAINER" ;;
  logs-worker) worker "docker logs -f '$DSV41_TF_CONTAINER'" ;;
  smoke) check_proxy_key; proxy smoke ;;
  all) sync_recipe; configure_recipe; check_cluster; prepare_recipe; start_service start ;;
  path)
    printf 'RECIPE=%s\nPROFILE=%s\nUPSTREAM_REVISION=%s\nMODEL_REVISION=%s\nIMAGE=%s\n' \
      "$RECIPE" "$PROFILE" "$DSV41_TF_REF" "$DSV41_TF_MODEL_REVISION" "$DSV41_TF_IMAGE"
    ;;
  *) die "Unknown action: $action" ;;
esac
