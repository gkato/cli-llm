#!/usr/bin/env bash
# Start GLM-5.3 Flash NVFP4 on the configured two-DGX-Spark cluster.
# Pass --first-run to pin MiaAI, build, download, mirror, GPU-check, and start.

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

With no argument, validate the checked-in profile, run MiaAI-Lab's pinned
two-node Ray/vLLM lifecycle, and start the authenticated proxy on 8000.
--first-run also clones the pinned upstream recipe, builds and ships its
SM121/NoPE kernel image, and downloads/verifies/rsyncs the ~181 GiB snapshot.

The reviewed profile uses Ray 2.58 with 4 GiB object stores, Marlin + eager,
FP8 KV, 256K context, and disabled MM dummy profiling during initialization.
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
