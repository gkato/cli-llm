#!/usr/bin/env bash
# Lifecycle adapter for MiaAI-Lab's TensorFold GLM-5.3 Flash recipe.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${TENSORFOLD_CONFIG_FILE:-${ROOT}/config/dspark-glm53-flash-tensorfold.env}"
RUNTIME="${TENSORFOLD_RUNTIME_DIR:-${ROOT}/data/dspark/glm53-flash-tensorfold}"
RECIPE="${TENSORFOLD_RECIPE_DIR:-${RUNTIME}/recipe}"
ENV_FILE="${RECIPE}/.env"

die() { printf '[glm53-tensorfold] ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '[glm53-tensorfold] %s\n' "$*"; }

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

Actions: bootstrap, configure, setup, start, restart, stop, status, logs,
         download, prepare, smoke, path, help

Typical:
  python3 -m ml.cli glm53-flash-tensorfold setup
  python3 -m ml.cli glm53-flash-tensorfold start
EOF
    exit 0
    ;;
  bootstrap)
    mkdir -p "$RUNTIME"
    if [[ ! -d "$RECIPE/.git" ]]; then
      git clone --branch "${TENSORFOLD_REF}" --depth 1 "$TENSORFOLD_REPO" "$RECIPE"
    else
      git -C "$RECIPE" fetch --depth 1 origin "$TENSORFOLD_REF"
      git -C "$RECIPE" checkout --detach FETCH_HEAD
    fi
    ;;
  path)
    printf 'RECIPE=%s\nPROFILE=%s\nIMAGE_DIGEST=%s\n' "$RECIPE" "$PROFILE" "$TENSORFOLD_IMAGE_DIGEST"
    exit 0
    ;;
esac

[[ -d "$RECIPE" ]] || die "Recipe missing; run bootstrap first"

if [[ "$action" == configure || "$action" == setup ]]; then
  mkdir -p "$RECIPE"
  cat > "$ENV_FILE" <<EOF
MODEL_ID=Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw
MODEL_REVISION=${TENSORFOLD_MODEL_REVISION}
DFLASH2_REVISION=${TENSORFOLD_DFLASH_REVISION}
WORKER=${TENSORFOLD_WORKER}
FABRIC_PEER=${TENSORFOLD_FABRIC_PEER}
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
TF_GLM_MTP=auto
IMAGE_DIGEST=${TENSORFOLD_IMAGE_DIGEST}
EOF
fi

case "$action" in
  setup|prepare) (cd "$RECIPE" && ./scripts/prepare.sh) ;;
  configure|bootstrap) log "Configured TensorFold recipe at $RECIPE" ;;
  start) (cd "$RECIPE" && ./start.sh) ;;
  restart) (cd "$RECIPE" && ./start.sh restart) ;;
  stop) (cd "$RECIPE" && ./stop.sh) ;;
  status) (cd "$RECIPE" && curl -fsS "http://127.0.0.1:${TENSORFOLD_PORT}/health") ;;
  logs) docker logs -f glm53-flash-tf ;;
  download) (cd "$RECIPE" && ./scripts/prepare.sh) ;;
  smoke) curl -fsS "http://127.0.0.1:${TENSORFOLD_PORT}/v1/models" ;;
  *) die "Unknown action: $action" ;;
esac
