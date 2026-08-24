#!/usr/bin/env bash
# Start DeepSeek V4 Flash 0731 on one dedicated DGX Spark. This entry point
# mirrors start-DS4-Flash-DSpark.sh without touching the two-node deployment.

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
    "${PYTHON}" -m ml.cli dspark-one setup
    "${PYTHON}" -m ml.cli dspark-one build
    "${PYTHON}" -m ml.cli dspark-one download
    "${PYTHON}" -m ml.cli dspark-one gpu-check
    ;;
  --no-wait)
    "${PYTHON}" -m ml.cli dspark-one configure
    DSPARK_ONE_NO_WAIT=1 "${PYTHON}" -m ml.cli dspark-one start
    exit $?
    ;;
  -h|--help)
    cat <<'EOF'
Usage: scripts/start-DS4-Flash-One-DSpark.sh [--first-run|--no-wait]

With no argument, reapply the checked-in one-Spark profile, start the raw
EXL3 model on loopback port 8888, and start the authenticated proxy on 8000.
--first-run also clones the pinned MiaAI-Lab recipe, pulls its digest-pinned
image, downloads/coalesces ~107 GB of weights, and checks CUDA.
--no-wait starts only the private raw server; run the command again without
the flag after /health is ready to validate it and expose the proxy.
EOF
    exit 0
    ;;
  "") ;;
  *)
    printf 'Unknown option: %s\n' "$1" >&2
    exit 2
    ;;
esac

"${PYTHON}" -m ml.cli dspark-one configure
"${PYTHON}" -m ml.cli dspark-one start
