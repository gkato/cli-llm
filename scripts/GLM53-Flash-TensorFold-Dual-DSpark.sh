#!/usr/bin/env bash
# Lifecycle adapter for MiaAI-Lab's TensorFold GLM-5.3 Flash recipe.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${TENSORFOLD_CONFIG_FILE:-${ROOT}/config/dspark-glm53-flash-tensorfold.env}"
RUNTIME="${TENSORFOLD_RUNTIME_DIR:-${ROOT}/data/dspark/glm53-flash-tensorfold}"
RECIPE="${TENSORFOLD_RECIPE_DIR:-${RUNTIME}/recipe}"
ENV_FILE="${RECIPE}/.env"
PROJECT_ENV_FILE="${TENSORFOLD_PROJECT_ENV_FILE:-${ROOT}/.env.local}"

die() { printf '[glm53-tensorfold] ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '[glm53-tensorfold] %s\n' "$*"; }

patch_nvidia_runtime() {
  local start="${RECIPE}/start.sh" tmp
  [[ -f "$start" ]] || die "Missing upstream launcher: $start"
  grep -q 'ml-compute: GB10 CUDA compatibility runtime' "$start" && return
  grep -q 'RUN_ARGS=(--gpus all ' "$start" \
    || die "Upstream Docker launcher changed; review the NVIDIA runtime patch"
  tmp="$(mktemp)"
  awk '
    /ENV_ARGS=\(-e HF_HUB_OFFLINE=/ {
      print
      print "  # ml-compute: GB10 CUDA compatibility runtime (driver 580 + CUDA 13.x image)."
      print "  ENV_ARGS+=(-e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)"
      next
    }
    { print }
  ' "$start" >"$tmp"
  sed -i.bak 's/RUN_ARGS=(--gpus all /RUN_ARGS=(--runtime=nvidia /' "$tmp"
  rm -f "${tmp}.bak"
  chmod 0755 "$tmp"
  mv "$tmp" "$start"
}

generated_patch_only() {
  local current="$1" base patched legacy rc=1
  base="$(mktemp)"; patched="$(mktemp)"; legacy="$(mktemp)"
  git -C "$RECIPE" show HEAD:start.sh >"$base" || return 1
  cp "$base" "$patched"
  awk '
    /ENV_ARGS=\(-e HF_HUB_OFFLINE=/ {
      print
      print "  # ml-compute: GB10 CUDA compatibility runtime (driver 580 + CUDA 13.x image)."
      print "  ENV_ARGS+=(-e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)"
      next
    }
    { print }
  ' "$base" >"${patched}.new"
  mv "${patched}.new" "$patched"
  sed -i.bak 's/RUN_ARGS=(--gpus all /RUN_ARGS=(--runtime=nvidia /' "$patched"
  rm -f "${patched}.bak"
  if cmp -s "$current" "$patched"; then
    rc=0
  else
    awk '
      /ENV_ARGS=\(-e HF_HUB_OFFLINE=/ {
        print
        print "  ENV_ARGS+=(-e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)"
        next
      }
      { print }
    ' "$base" >"$legacy"
    sed -i.bak 's/RUN_ARGS=(--gpus all /RUN_ARGS=(--runtime=nvidia /' "$legacy"
    rm -f "${legacy}.bak"
    cmp -s "$current" "$legacy" && rc=0
  fi
  rm -f "$base" "$patched" "$legacy"
  return "$rc"
}

sync_recipe() {
  mkdir -p "$RUNTIME"
  if [[ ! -d "$RECIPE/.git" ]]; then
    git clone --filter=blob:none --no-checkout "$TENSORFOLD_REPO" "$RECIPE"
  elif ! git -C "$RECIPE" diff --quiet --; then
    if [[ "$(git -C "$RECIPE" diff --name-only)" == "start.sh" ]] \
      && generated_patch_only "${RECIPE}/start.sh"; then
      git -C "$RECIPE" checkout -- start.sh
    else
      die "Recipe has local tracked changes; preserving them instead of updating"
    fi
  fi
  git -C "$RECIPE" fetch --depth 1 origin "$TENSORFOLD_REF"
  git -C "$RECIPE" checkout --detach FETCH_HEAD
  [[ "$(git -C "$RECIPE" rev-parse HEAD)" == "$TENSORFOLD_REF" ]] \
    || die "TensorFold recipe did not resolve to the pinned commit"
  patch_nvidia_runtime
}

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
    DSPARK_PROXY_UPSTREAM_URL="http://127.0.0.1:${TENSORFOLD_PORT}" \
    "$python" -m ml.cli dspark-proxy "$action")
}

check_proxy_key() {
  [[ -n "$(api_key)" ]] || die "API_KEY is missing from $PROJECT_ENV_FILE"
}

[[ -f "$PROFILE" ]] || die "Missing profile: $PROFILE"
set -a
# shellcheck disable=SC1090
source "$PROFILE"
set +a

action="${1:-help}"
case "$action" in
  help)
    cat <<'EOF'
GLM-5.3 Flash TensorFold on two DGX Sparks

Actions: bootstrap, update, configure, setup, start, restart, stop, status, logs,
         download, prepare, smoke, path, help

Typical:
  python3 -m ml.cli glm53-flash-tensorfold setup
  python3 -m ml.cli glm53-flash-tensorfold start
EOF
    exit 0
    ;;
  bootstrap|update|setup)
    sync_recipe
    ;;
  path)
    printf 'RECIPE=%s\nPROFILE=%s\nIMAGE_DIGEST=%s\n' "$RECIPE" "$PROFILE" "$TENSORFOLD_IMAGE_DIGEST"
    exit 0
    ;;
esac

[[ -d "$RECIPE" ]] || die "Recipe missing; run bootstrap first"
patch_nvidia_runtime

if [[ "$action" == configure || "$action" == setup ]]; then
  mkdir -p "$RECIPE"
  cat > "$ENV_FILE" <<EOF
MODEL_ID=${TENSORFOLD_MODEL_ID}
MODEL_REVISION=${TENSORFOLD_MODEL_REVISION}
DFLASH2_REVISION=${TENSORFOLD_DFLASH_REVISION}
WORKER=${TENSORFOLD_WORKER}
FABRIC_PEER=${TENSORFOLD_FABRIC_PEER}
SERVED_NAME=${TENSORFOLD_SERVED_NAME}
HOST=${TENSORFOLD_HOST}
PORT=${TENSORFOLD_PORT}
PARALLEL=${TENSORFOLD_PARALLEL}
CONTEXT=${TENSORFOLD_CONTEXT}
KV=${TENSORFOLD_KV}
DENSE=${TENSORFOLD_DENSE}
DRAFTER=${TENSORFOLD_DRAFTER}
VISION=${TENSORFOLD_VISION}
COMM=${TENSORFOLD_COMM}
COPY=${TENSORFOLD_COPY}
COPY_MAX=${TENSORFOLD_COPY_MAX}
MAX_TOKENS=${TENSORFOLD_MAX_TOKENS}
MULTI_PREFILL=${TENSORFOLD_MULTI_PREFILL}
SHARED_PREFIX=${TENSORFOLD_SHARED_PREFIX}
FILL_ROWS=${TENSORFOLD_FILL_ROWS}
TF_GLM_MTP=auto
IMAGE_TAG=${TENSORFOLD_IMAGE_TAG}
IMAGE_DIGEST=${TENSORFOLD_IMAGE_DIGEST}
EOF
fi

case "$action" in
  setup|prepare) (cd "$RECIPE" && ./scripts/prepare.sh) ;;
  configure|bootstrap|update) log "Configured TensorFold recipe at $RECIPE" ;;
  start)
    check_proxy_key
    (cd "$RECIPE" && ./start.sh)
    proxy serve
    proxy smoke
    log "Authenticated API: http://127.0.0.1:${DSPARK_PROXY_PORT}/v1"
    ;;
  restart)
    check_proxy_key
    proxy stop || true
    (cd "$RECIPE" && ./start.sh restart)
    proxy serve
    proxy smoke
    log "Authenticated API: http://127.0.0.1:${DSPARK_PROXY_PORT}/v1"
    ;;
  stop) proxy stop || true; (cd "$RECIPE" && ./stop.sh) ;;
  status)
    curl -fsS "http://127.0.0.1:${TENSORFOLD_PORT}/health" || true
    proxy status
    ;;
  logs) docker logs -f glm53-flash-tf ;;
  download) (cd "$RECIPE" && ./scripts/prepare.sh) ;;
  smoke) check_proxy_key; proxy smoke ;;
  *) die "Unknown action: $action" ;;
esac
