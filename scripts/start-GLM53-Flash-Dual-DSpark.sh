#!/usr/bin/env bash
# Start GLM-5.3 Flash NVFP4 on the configured two-DGX-Spark cluster.
# Pass --first-run to preflight, pull, download, mirror, GPU-check, and start.

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
    "${PYTHON}" -m ml.cli glm53-flash setup
    "${PYTHON}" -m ml.cli glm53-flash pull
    "${PYTHON}" -m ml.cli glm53-flash download
    "${PYTHON}" -m ml.cli glm53-flash gpu-check
    ;;
  -h|--help)
    cat <<'EOF'
Usage: scripts/start-GLM53-Flash-Dual-DSpark.sh [--first-run]

With no argument, validate the checked-in profile, start a two-node Ray
cluster, launch private vLLM TP=2, and start the authenticated proxy on 8000.
--first-run also preflights both Sparks, pulls the digest-pinned dedicated
arm64 image, and downloads/verifies/rsyncs the ~181 GiB model snapshot.

This is an experimental GB10 profile because the model publisher has not yet
listed SM121 among its verified targets. The conservative default uses Marlin,
eager execution, 32K context, and one request.
EOF
    exit 0
    ;;
  "") ;;
  *)
    printf 'Unknown option: %s\n' "$1" >&2
    exit 2
    ;;
esac

"${PYTHON}" -m ml.cli glm53-flash configure
"${PYTHON}" -m ml.cli glm53-flash start
