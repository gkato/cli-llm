#!/usr/bin/env bash
# Reproducible before/after benchmark for the two-Spark DeepSeek deployment.

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

case "${1:-full}" in
  full) RUN_LONG=1 ;;
  quick) RUN_LONG=0 ;;
  -h|--help)
    printf '%s\n' \
      'Usage: scripts/bench_dspark_ab.sh [full|quick]' \
      '' \
      'full:  C1/C2/C4 throughput, forced tool call, 32K/128K TTFT, memory' \
      'quick: C1/C2/C4 throughput, forced tool call, memory' \
      '' \
      'Set BENCH_LABEL=current before migration and BENCH_LABEL=miaai-512k after.'
    exit 0
    ;;
  *) printf 'Unknown profile: %s\n' "$1" >&2; exit 2 ;;
esac

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="${PYTHON_BIN}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${PROJECT_ROOT}/venv/bin/python" ]]; then
  PYTHON="${PROJECT_ROOT}/venv/bin/python"
else
  PYTHON=python3
fi

BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000}"
MODEL="${VLLM_MODEL:-deepseek-v4-flash-0731}"
LABEL="${BENCH_LABEL:-miaai-512k}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ID="${LABEL}-${STAMP}"
RESULT_DIR="${RESULT_DIR:-${PROJECT_ROOT}/data/benchmarks/dspark/${RUN_ID}}"
mkdir -p "${RESULT_DIR}"

capture_memory_snapshot() {
  local destination="$1"
  local -a statuses

  # `ml.cli dspark memory` intentionally returns 1 when the Harness coexistence
  # target is missed. That is benchmark metadata, not a reason to skip the
  # throughput run. Preserve a real tee/write failure while recording and
  # continuing past the expected memory warning.
  set +e
  "${PYTHON}" -m ml.cli dspark memory 2>&1 | tee "${destination}"
  statuses=("${PIPESTATUS[@]}")
  set -e
  if (( statuses[1] != 0 )); then
    printf 'Could not write memory snapshot: %s\n' "${destination}" >&2
    return "${statuses[1]}"
  fi
  if (( statuses[0] != 0 )); then
    printf '[dspark] NOTE: memory target missed; throughput benchmark will continue\n' \
      | tee -a "${destination}"
  fi
}

printf 'DSpark A/B label: %s\nResults: %s\n' "${LABEL}" "${RESULT_DIR}"
capture_memory_snapshot "${RESULT_DIR}/memory-before.txt"

VLLM_BASE_URL="${BASE_URL}" \
VLLM_MODEL="${MODEL}" \
VLLM_TOKENIZER="deepseek-ai/DeepSeek-V4-Flash-0731" \
CONCURRENCIES="1 2 4" \
RESULT_DIR="${RESULT_DIR}" \
RUN_ID="${RUN_ID}" \
scripts/bench_vllm.sh standard

RESULT_DIR="${RESULT_DIR}" BASE_URL="${BASE_URL}" MODEL="${MODEL}" \
  "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

import requests
from dotenv import dotenv_values

root = Path.cwd()
api_key = os.environ.get("OPEN_API_KEY") or dotenv_values(root / ".env.local").get("API_KEY")
if not api_key:
    raise SystemExit("API_KEY is missing from .env.local")
payload = {
    "model": os.environ["MODEL"],
    "messages": [
        {
            "role": "user",
            "content": "Use get_weather for Sao Paulo. Do not answer without the tool.",
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Return weather for one city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ],
    "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    "temperature": 0,
    "max_tokens": 128,
}
response = requests.post(
    os.environ["BASE_URL"].rstrip("/") + "/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json=payload,
    timeout=600,
)
response.raise_for_status()
data = response.json()
output = Path(os.environ["RESULT_DIR"]) / "tool-call.json"
output.write_text(json.dumps(data, indent=2), encoding="utf-8")
calls = data["choices"][0]["message"].get("tool_calls") or []
if not calls or calls[0].get("function", {}).get("name") != "get_weather":
    raise SystemExit(f"forced tool-call check failed; inspect {output}")
print(f"Tool-call check passed: {output}")
PY

if [[ "${RUN_LONG}" == 1 ]]; then
  VLLM_BASE_URL="${BASE_URL}" \
  VLLM_MODEL="${MODEL}" \
  VLLM_TOKENIZER="deepseek-ai/DeepSeek-V4-Flash-0731" \
  NUM_PROMPTS=4 NUM_WARMUPS=1 INPUT_LEN=32768 OUTPUT_LEN=64 \
  CONCURRENCIES=1 RESULT_DIR="${RESULT_DIR}" RUN_ID="${RUN_ID}" \
  scripts/bench_vllm.sh standard

  VLLM_BASE_URL="${BASE_URL}" \
  VLLM_MODEL="${MODEL}" \
  VLLM_TOKENIZER="deepseek-ai/DeepSeek-V4-Flash-0731" \
  NUM_PROMPTS=2 NUM_WARMUPS=0 INPUT_LEN=131072 OUTPUT_LEN=64 \
  CONCURRENCIES=1 RESULT_DIR="${RESULT_DIR}" RUN_ID="${RUN_ID}" \
  scripts/bench_vllm.sh standard
fi

capture_memory_snapshot "${RESULT_DIR}/memory-after.txt"
printf 'DSpark A/B benchmark complete: %s\n' "${RESULT_DIR}"
