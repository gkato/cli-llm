#!/usr/bin/env bash
# Lightweight streamed-decode benchmark for the one-Spark Qwen3.8-27B recipe.
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="${QWEN38_BENCH_BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${QWEN38_BENCH_MODEL:-qwen3.8-27b-sglang}"
RUNS="${QWEN38_BENCH_RUNS:-3}"
MAX_TOKENS="${QWEN38_BENCH_MAX_TOKENS:-400}"
PROMPT="${QWEN38_BENCH_PROMPT:-Explain why FP8 KV cache can increase context capacity in exactly 220 words.}"
API_KEY="${API_KEY:-$(sed -n 's/^API_KEY=//p' "${PROJECT_ROOT}/.env.local" 2>/dev/null | tail -1)}"

[[ -n "$API_KEY" ]] || { echo "API_KEY is missing from ${PROJECT_ROOT}/.env.local" >&2; exit 1; }
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "QWEN38_BENCH_RUNS must be a positive integer" >&2; exit 2; }

run() {
  local label="$1" output timing
  output="$(mktemp)"
  trap 'rm -f "$output"' RETURN
  timing="$(curl -fsSN --max-time 1800 -w '%{time_total} %{time_starttransfer}' "$BASE_URL/chat/completions" \
    -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json, os; print(json.dumps({"model": os.environ["MODEL"], "messages": [{"role": "user", "content": os.environ["PROMPT"]}], "stream": True, "stream_options": {"include_usage": True}, "temperature": 0, "max_completion_tokens": int(os.environ["MAX_TOKENS"]), "chat_template_kwargs": {"enable_thinking": False}}))')" \
    -o "$output")"
  python3 - "$label" "$output" $timing <<'PY'
import json, sys

label, path, total, first_byte = sys.argv[1:]
total, first_byte = float(total), float(first_byte)
tokens = None
for raw in open(path, encoding="utf-8"):
    if not raw.startswith("data: ") or raw.rstrip() == "data: [DONE]":
        continue
    try:
        event = json.loads(raw[6:])
    except json.JSONDecodeError:
        continue
    if event.get("usage"):
        tokens = event["usage"].get("completion_tokens")
if not tokens:
    raise SystemExit("No completion usage received; ensure SGLang emits stream_options usage.")
decode = max(total - first_byte, 0.001)
print(f"{label}: completion_tokens={tokens} total={total:.3f}s TTFT={first_byte:.3f}s decode={tokens / decode:.2f} tok/s")
PY
}

echo "Qwen3.8-27B streamed decode benchmark: model=$MODEL runs=$RUNS max_tokens=$MAX_TOKENS"
echo "Warm-up (not counted)"
MODEL="$MODEL" PROMPT="$PROMPT" MAX_TOKENS="$MAX_TOKENS" run warmup >/dev/null
for i in $(seq 1 "$RUNS"); do
  MODEL="$MODEL" PROMPT="$PROMPT" MAX_TOKENS="$MAX_TOKENS" run "run-$i"
done
