#!/usr/bin/env bash
set -euo pipefail

# Benchmark an OpenAI-compatible vLLM server with deterministic random prompts.
# Defaults exercise the Gemma 4 NVFP4 profile at its configured concurrency
# ceiling. Override any setting with an environment variable; for example:
#
#   OPEN_API_KEY=... \
#   VLLM_BASE_URL=http://thinkstationpgx-fd9c.tail1c73a3.ts.net/v1 \
#   scripts/bench_vllm.sh
#
# For a longer-context run:
#
#   OPEN_API_KEY=... INPUT_LEN=8192 OUTPUT_LEN=512 NUM_PROMPTS=32 \
#   scripts/bench_vllm.sh

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

BASE_URL=${VLLM_BASE_URL:-http://127.0.0.1:8000}
BASE_URL=${BASE_URL%/}
BASE_URL=${BASE_URL%/v1}

MODEL=${VLLM_MODEL:-nvidia/Gemma-4-31B-IT-NVFP4}
NUM_PROMPTS=${NUM_PROMPTS:-16}
NUM_WARMUPS=${NUM_WARMUPS:-2}
INPUT_LEN=${INPUT_LEN:-2048}
OUTPUT_LEN=${OUTPUT_LEN:-256}
CONCURRENCIES=${CONCURRENCIES:-"1 2"}
SEED=${SEED:-42}
RESULT_DIR=${RESULT_DIR:-$PROJECT_ROOT/data/benchmarks/vllm}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}

BENCH_API_KEY=${OPEN_API_KEY:-${OPENAI_API_KEY:-}}
if [[ -z "$BENCH_API_KEY" ]]; then
  echo "Set OPEN_API_KEY (or OPENAI_API_KEY) before running the benchmark." >&2
  exit 2
fi
export OPENAI_API_KEY=$BENCH_API_KEY
unset BENCH_API_KEY

for value_name in NUM_PROMPTS NUM_WARMUPS INPUT_LEN OUTPUT_LEN SEED; do
  value=${!value_name}
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$value_name must be a non-negative integer, got: $value" >&2
    exit 2
  fi
done
if (( NUM_PROMPTS < 1 || INPUT_LEN < 1 || OUTPUT_LEN < 1 )); then
  echo "NUM_PROMPTS, INPUT_LEN, and OUTPUT_LEN must be greater than zero." >&2
  exit 2
fi

if command -v vllm >/dev/null 2>&1; then
  VLLM_BIN=$(command -v vllm)
elif [[ -x "$PROJECT_ROOT/venv/bin/vllm" ]]; then
  VLLM_BIN=$PROJECT_ROOT/venv/bin/vllm
else
  echo "vllm is not installed or available on PATH." >&2
  echo "Activate the project venv, then retry." >&2
  exit 127
fi

if ! curl --fail --silent --show-error \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  "$BASE_URL/v1/models" >/dev/null; then
  echo "Could not authenticate to the vLLM server at $BASE_URL." >&2
  exit 1
fi

mkdir -p "$RESULT_DIR"
MODEL_SLUG=${MODEL//\//-}
read -r -a CONCURRENCY_VALUES <<< "$CONCURRENCIES"
if (( ${#CONCURRENCY_VALUES[@]} == 0 )); then
  echo "CONCURRENCIES must contain at least one positive integer." >&2
  exit 2
fi

RESULT_FILES=()
for concurrency in "${CONCURRENCY_VALUES[@]}"; do
  if [[ ! "$concurrency" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid concurrency value: $concurrency" >&2
    exit 2
  fi

  result_filename="${MODEL_SLUG}-c${concurrency}-${INPUT_LEN}in-${OUTPUT_LEN}out-${RUN_ID}.json"
  RESULT_FILES+=("$RESULT_DIR/$result_filename")

  echo
  echo "Benchmarking $MODEL"
  echo "  Server:       $BASE_URL"
  echo "  Concurrency:  $concurrency"
  echo "  Requests:     $NUM_PROMPTS (+ $NUM_WARMUPS warmups)"
  echo "  Tokens:       $INPUT_LEN input / $OUTPUT_LEN output"
  echo "  Result:       $RESULT_DIR/$result_filename"
  echo

  "$VLLM_BIN" bench serve \
    --backend openai-chat \
    --base-url "$BASE_URL" \
    --endpoint /v1/chat/completions \
    --model "$MODEL" \
    --tokenizer "$MODEL" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio 0 \
    --num-prompts "$NUM_PROMPTS" \
    --num-warmups "$NUM_WARMUPS" \
    --max-concurrency "$concurrency" \
    --temperature 0 \
    --seed "$SEED" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --ready-check-timeout-sec 60 \
    --save-result \
    --result-dir "$RESULT_DIR" \
    --result-filename "$result_filename"
done

python3 - "${RESULT_FILES[@]}" <<'PY'
import json
import sys
from pathlib import Path


def metric(data, key):
    value = data.get(key)
    return "-" if value is None else f"{value:.2f}"


print("\nComparison")
print(
    f"{'Conc.':>5}  {'Req/s':>8}  {'Out tok/s':>10}  "
    f"{'P50 TTFT':>10}  {'P99 TTFT':>10}  {'Mean TPOT':>10}  {'P99 E2E':>10}  {'Failed':>6}"
)
for filename in sys.argv[1:]:
    path = Path(filename)
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    print(
        f"{data.get('max_concurrency', '-'):>5}  "
        f"{metric(data, 'request_throughput'):>8}  "
        f"{metric(data, 'output_throughput'):>10}  "
        f"{metric(data, 'p50_ttft_ms'):>10}  "
        f"{metric(data, 'p99_ttft_ms'):>10}  "
        f"{metric(data, 'mean_tpot_ms'):>10}  "
        f"{metric(data, 'p99_e2el_ms'):>10}  "
        f"{data.get('failed', 0):>6}"
    )
print(f"\nJSON results: {Path(sys.argv[1]).parent}")
PY
