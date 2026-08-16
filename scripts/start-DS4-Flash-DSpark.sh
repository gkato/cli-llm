#!/usr/bin/env bash
# Start the configured two-node DeepSeek V4 Flash service from the NVIDIA/head
# Spark. Pass --first-run to bootstrap, build, download, and then start.

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="${PYTHON_BIN}"
elif [[ -x "${PROJECT_ROOT}/venv/bin/python" ]]; then
  PYTHON="${PROJECT_ROOT}/venv/bin/python"
else
  PYTHON="python3"
fi

case "${1:-}" in
  --first-run)
    "${PYTHON}" -m ml.cli dspark setup
    "${PYTHON}" -m ml.cli dspark build
    "${PYTHON}" -m ml.cli dspark download
    ;;
  -h|--help)
    cat <<'EOF'
Usage: scripts/start-DS4-Flash-DSpark.sh [--first-run]

With no argument, reapply the committed cluster profile and start the already
prepared model worker-first. --first-run also clones/builds the runtime and
downloads/mirrors the model before starting it.
EOF
    exit 0
    ;;
  "") ;;
  *)
    printf 'Unknown option: %s\n' "$1" >&2
    exit 2
    ;;
esac

# Reapply the checked-in profile so an old generated .env.dspark cannot retain
# stale 1M-context or 0.85-memory values.
"${PYTHON}" -m ml.cli dspark configure
"${PYTHON}" -m ml.cli dspark start
