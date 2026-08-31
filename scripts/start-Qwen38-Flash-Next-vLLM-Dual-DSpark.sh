#!/usr/bin/env bash
# Start MiaAI's Qwen3.8 Flash Next vLLM profile on two linked DGX Sparks.
# Pass --first-run to clone, configure, pull, download, and then start.

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
    "${PYTHON}" -m ml.cli qwen38-flash-next-vllm setup
    "${PYTHON}" -m ml.cli qwen38-flash-next-vllm pull
    "${PYTHON}" -m ml.cli qwen38-flash-next-vllm download
    ;;
  -h|--help)
    cat <<'EOF'
Usage: scripts/start-Qwen38-Flash-Next-vLLM-Dual-DSpark.sh [--first-run]

With no argument, reapply the checked-in MiaAI performance profile, launch
vLLM TP2+EP+MTP3 worker-first on private port 8888, and start the authenticated
proxy on port 8000. --first-run also clones the pinned recipe, pulls the
digest-pinned image, and downloads/rsyncs the pinned ~126 GiB checkpoint.
EOF
    exit 0
    ;;
  "") ;;
  *)
    printf 'Unknown option: %s\n' "$1" >&2
    exit 2
    ;;
esac

"${PYTHON}" -m ml.cli qwen38-flash-next-vllm configure
"${PYTHON}" -m ml.cli qwen38-flash-next-vllm start
