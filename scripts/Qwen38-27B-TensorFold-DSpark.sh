#!/usr/bin/env bash
# Pinned lifecycle adapter for MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${QWEN38_TF_CONFIG_FILE:-${ROOT}/config/dspark-qwen38-27b-tensorfold.env}"
RUNTIME="${QWEN38_TF_RUNTIME_DIR:-${ROOT}/data/dspark/qwen38-27b-tensorfold}"
RECIPE="${QWEN38_TF_RECIPE_DIR:-${RUNTIME}/recipe}"
ENV_FILE="${RECIPE}/.env"
PROJECT_ENV_FILE="${QWEN38_TF_PROJECT_ENV_FILE:-${ROOT}/.env.local}"

log() { printf '[qwen38-27b-tensorfold] %s\n' "$*"; }
die() { printf '[qwen38-27b-tensorfold] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -f "$PROFILE" ]] || die "Missing profile: $PROFILE"
set -a
# shellcheck disable=SC1090
source "$PROFILE"
set +a

HF_CACHE="${QWEN38_TF_HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"

project_python() {
  [[ -x "${ROOT}/venv/bin/python" ]] && printf '%s' "${ROOT}/venv/bin/python" || printf '%s' python3
}

api_key() {
  printf '%s' "${API_KEY:-$(sed -n 's/^API_KEY=//p' "$PROJECT_ENV_FILE" 2>/dev/null | tail -1)}"
}

proxy() {
  local action="$1" python
  python="$(project_python)"
  (cd "$ROOT" && \
    API_KEY="$(api_key)" \
    DSPARK_PROXY_HOST="${DSPARK_PROXY_HOST}" \
    DSPARK_PROXY_PORT="${DSPARK_PROXY_PORT}" \
    DSPARK_PROXY_UPSTREAM_URL="http://127.0.0.1:${QWEN38_TF_PORT}" \
    "$python" -m ml.cli dspark-proxy "$action")
}

check_proxy_key() {
  [[ -n "$(api_key)" ]] || die "API_KEY is missing from $PROJECT_ENV_FILE"
}

sync_recipe() {
  mkdir -p "$RUNTIME"
  if [[ ! -d "$RECIPE/.git" ]]; then
    [[ ! -e "$RECIPE" ]] || die "Recipe path exists but is not a git checkout: $RECIPE"
    git clone --filter=blob:none --no-checkout "$QWEN38_TF_REPO" "$RECIPE"
  elif ! git -C "$RECIPE" diff --quiet --; then
    die "Recipe has local tracked changes; preserving them instead of updating"
  fi
  git -C "$RECIPE" fetch --depth 1 origin "$QWEN38_TF_REF"
  git -C "$RECIPE" checkout --detach FETCH_HEAD
  check_pin
}

check_pin() {
  [[ -d "$RECIPE/.git" ]] || die "Recipe missing; run qwen38-27b-tensorfold setup"
  [[ "$(git -C "$RECIPE" rev-parse HEAD 2>/dev/null)" == "$QWEN38_TF_REF" ]] \
    || die "Recipe is not at pinned revision $QWEN38_TF_REF"
}

configure_recipe() {
  check_pin
  [[ "$QWEN38_TF_HOST" == 127.0.0.1 ]] || die "QWEN38_TF_HOST must be 127.0.0.1"
  cat > "$ENV_FILE" <<EOF
MODEL_ID=${QWEN38_TF_MODEL_ID}
DRAFT_ID=${QWEN38_TF_DRAFT_ID}
MODEL_REVISION=${QWEN38_TF_MODEL_REVISION}
DRAFT_REVISION=${QWEN38_TF_DRAFT_REVISION}
IMAGE=${QWEN38_TF_IMAGE}
GHCR_IMAGE=${QWEN38_TF_IMAGE_REPO}
CONTAINER_NAME=${QWEN38_TF_CONTAINER}
SERVED_NAME=${QWEN38_TF_SERVED_NAME}
HOST=${QWEN38_TF_HOST}
PORT=${QWEN38_TF_PORT}
PARALLEL=${QWEN38_TF_PARALLEL}
CONTEXT=${QWEN38_TF_CONTEXT}
YARN_FACTOR=${QWEN38_TF_YARN_FACTOR}
PREFILL_FP8=${QWEN38_TF_PREFILL_FP8}
KV_DTYPE=${QWEN38_TF_KV_DTYPE}
KV_POOL_GB=${QWEN38_TF_KV_POOL_GB}
TENSORFOLD_MEMORY_RESERVE_GIB=${QWEN38_TF_MEMORY_RESERVE_GIB}
VISION=${QWEN38_TF_VISION}
VISION_URLS=${QWEN38_TF_VISION_URLS}
TENSORFOLD_MAX_IMAGES=${QWEN38_TF_MAX_IMAGES}
TENSORFOLD_IMAGE_TOKENS=${QWEN38_TF_IMAGE_TOKENS}
TENSORFOLD_VIDEO_TOKENS=${QWEN38_TF_VIDEO_TOKENS}
TEMPERATURE=${QWEN38_TF_TEMPERATURE}
TOP_P=${QWEN38_TF_TOP_P}
TOP_K=${QWEN38_TF_TOP_K}
THINKING=${QWEN38_TF_THINKING}
HF_CACHE=${HF_CACHE}
EOF
}

ensure_pinned_image() {
  local reference label expected_hash
  command -v docker >/dev/null || die "docker is required"
  reference="${QWEN38_TF_IMAGE_REPO}@${QWEN38_TF_IMAGE_DIGEST}"
  if ! docker image inspect "$reference" >/dev/null 2>&1; then
    log "Pulling pinned TensorFold image $reference"
    docker pull "$reference"
  fi
  label="$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$reference" 2>/dev/null)"
  expected_hash="${QWEN38_TF_IMAGE_TAG##*-}"
  [[ "$label" == "$expected_hash" ]] \
    || die "Pinned image patch label is '$label', expected '$expected_hash'"
  docker tag "$reference" "$QWEN38_TF_IMAGE"
}

snapshot_dir() {
  local repo="$1" revision="$2"
  printf '%s/hub/models--%s/snapshots/%s' "$HF_CACHE" "${repo//\//--}" "$revision"
}

set_snapshot_ref() {
  local repo="$1" revision="$2" model_dir snapshot
  model_dir="${HF_CACHE}/hub/models--${repo//\//--}"
  snapshot="$(snapshot_dir "$repo" "$revision")"
  [[ -f "$snapshot/config.json" ]] || return 1
  mkdir -p "$model_dir/refs"
  printf '%s\n' "$revision" > "$model_dir/refs/main"
}

download_snapshot() {
  local repo="$1" revision="$2" snapshot
  local -a token=()
  snapshot="$(snapshot_dir "$repo" "$revision")"
  if [[ ! -f "$snapshot/config.json" ]]; then
    log "Downloading $repo at pinned revision ${revision:0:12}"
    if command -v hf >/dev/null; then
      hf download "$repo" --revision "$revision" --cache-dir "$HF_CACHE/hub" >/dev/null
    else
      [[ -n "${HF_TOKEN:-}" ]] && token=(-e HF_TOKEN)
      docker run --rm --network host --entrypoint python "${token[@]}" \
        -v "$HF_CACHE":/root/.cache/huggingface \
        "$QWEN38_TF_IMAGE" \
        -c 'from huggingface_hub import snapshot_download; import sys; snapshot_download(sys.argv[1], revision=sys.argv[2], cache_dir="/root/.cache/huggingface/hub")' \
        "$repo" "$revision"
    fi
  fi
  set_snapshot_ref "$repo" "$revision" \
    || die "Pinned snapshot is incomplete: $repo@$revision"
}

pin_snapshot_refs() {
  set_snapshot_ref "$QWEN38_TF_MODEL_ID" "$QWEN38_TF_MODEL_REVISION" \
    || die "Pinned target snapshot is missing; run setup"
  set_snapshot_ref "$QWEN38_TF_DRAFT_ID" "$QWEN38_TF_DRAFT_REVISION" \
    || die "Pinned draft snapshot is missing; run setup"
}

prepare_recipe() {
  configure_recipe
  ensure_pinned_image
  (cd "$RECIPE" && ./scripts/prepare.sh)
  download_snapshot "$QWEN38_TF_MODEL_ID" "$QWEN38_TF_MODEL_REVISION"
  download_snapshot "$QWEN38_TF_DRAFT_ID" "$QWEN38_TF_DRAFT_REVISION"
  log "Pinned image, target, and drafter are ready"
}

start_service() {
  local mode="$1"
  configure_recipe
  pin_snapshot_refs
  check_proxy_key
  if [[ "$mode" == restart ]]; then
    proxy stop || true
    (cd "$RECIPE" && ./start.sh restart)
  else
    (cd "$RECIPE" && ./start.sh)
  fi
  proxy serve
  proxy smoke
  log "Authenticated API: http://127.0.0.1:${DSPARK_PROXY_PORT}/v1"
}

usage() {
  cat <<'EOF'
Qwen3.8-27B TensorFold on one DGX Spark

Actions: bootstrap, update, configure, check, setup, prepare, download, start,
         restart, stop, status, logs, smoke, path, help

Typical:
  python3 -m ml.cli qwen38-27b-tensorfold setup
  python3 -m ml.cli qwen38-27b-tensorfold start
  python3 -m ml.cli qwen38-27b-tensorfold smoke
EOF
}

case "${1:-help}" in
  bootstrap) sync_recipe ;;
  update) sync_recipe; configure_recipe ;;
  configure) configure_recipe ;;
  check) check_pin; configure_recipe; pin_snapshot_refs; command -v docker >/dev/null || die "docker is required" ;;
  setup) sync_recipe; prepare_recipe ;;
  prepare|download) prepare_recipe ;;
  start) start_service start ;;
  restart) start_service restart ;;
  stop) proxy stop || true; check_pin; (cd "$RECIPE" && ./stop.sh) ;;
  status)
    curl -fsS "http://127.0.0.1:${QWEN38_TF_PORT}/health" || true
    proxy status
    ;;
  logs) docker logs -f "$QWEN38_TF_CONTAINER" ;;
  smoke) check_proxy_key; proxy smoke ;;
  path)
    printf 'RECIPE=%s\nPROFILE=%s\nUPSTREAM_REVISION=%s\nIMAGE_DIGEST=%s\n' \
      "$RECIPE" "$PROFILE" "$QWEN38_TF_REF" "$QWEN38_TF_IMAGE_DIGEST"
    ;;
  help|-h|--help) usage ;;
  *) die "Unknown action: ${1:-}" ;;
esac
