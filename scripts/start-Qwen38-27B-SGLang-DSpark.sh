#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON_BIN:-python3}"
if [[ "${1:-}" == --first-run ]]; then
  "$PYTHON" -m ml.cli qwen38-27b-sglang setup
elif [[ -n "${1:-}" ]]; then
  echo 'Usage: scripts/start-Qwen38-27B-SGLang-DSpark.sh [--first-run]' >&2; exit 2
fi
"$PYTHON" -m ml.cli qwen38-27b-sglang start
