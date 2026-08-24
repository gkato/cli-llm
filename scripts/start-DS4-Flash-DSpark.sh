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
    FIRST_RUN=1
    ;;
  --cutover)
    # Image/cache were staged separately; perform only the checked legacy
    # cutover and start sequence.
    FIRST_RUN=1
    ;;
  -h|--help)
    cat <<'EOF'
Usage: scripts/start-DS4-Flash-DSpark.sh [--first-run|--cutover]

With no argument, reapply the committed cluster profile, start the model
worker-first on loopback port 8888, then start the authenticated proxy on 8000.
--first-run also clones MiaAI-Lab's recipe, pulls the pinned Anemll image,
downloads/mirrors the model, and performs the legacy Stage-C cutover.
--cutover skips staging and performs only the validated legacy cutover/start.
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
# stale context, memory, runtime-image, or public-bind values.
"${PYTHON}" -m ml.cli dspark configure
if [[ "${FIRST_RUN:-0}" == 1 ]]; then
  # The dspark start action first rejects a Funnel still pointing at unsafe
  # port 8888, then stops the legacy deployment immediately before cutover.
  DSPARK_CUTOVER_LEGACY=1 "${PYTHON}" -m ml.cli dspark start
else
  "${PYTHON}" -m ml.cli dspark start
fi
