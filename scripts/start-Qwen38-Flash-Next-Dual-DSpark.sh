#!/usr/bin/env bash
# Start Qwen3.8 Flash Next NVFP4 on the configured two-DGX-Spark cluster.
# Pass --first-run to bootstrap, patch/build, download, sync, and then start.

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="${PYTHON_BIN}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${PROJECT_ROOT}/venv/bin/python" ]]; then
  PYTHON="${PROJECT_ROOT}/venv/bin/python"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${PROJECT_ROOT}/.venv/bin/python"
else
  PYTHON=python3
fi

case "${1:-}" in
  --first-run)
    "${PYTHON}" -m ml.cli qwen38-flash-next setup
    "${PYTHON}" -m ml.cli qwen38-flash-next download
    ;;
  -h|--help)
    cat <<'EOF'
Usage: scripts/start-Qwen38-Flash-Next-Dual-DSpark.sh [--first-run]

With no argument, reapply the checked-in cluster profile, launch SGLang TP=2
worker-first on private port 8888, and start the authenticated proxy on 8000.
--first-run also clones the pinned MiaAI-Lab recipe, builds the digest-pinned
SM121 kernel-patch image on both nodes, and downloads/verifies/rsyncs ~135 GB
of model weights before launch.
EOF
    exit 0
    ;;
  "") ;;
  *)
    printf 'Unknown option: %s\n' "$1" >&2
    exit 2
    ;;
esac

"${PYTHON}" -m ml.cli qwen38-flash-next configure
"${PYTHON}" -m ml.cli qwen38-flash-next start
