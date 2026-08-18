#!/usr/bin/env bash

# PDF OCR through baidu/Unlimited-OCR's OpenAI-compatible vLLM endpoint.
#
# The filename is kept for compatibility with the earlier Gemma OCR helper,
# but the implementation now uses Unlimited-OCR's required prompt and decoding
# recipe. PDF pages are rendered to PNG and sent together in one request so the
# model can perform its native long-horizon, multi-page parsing.

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

readonly DEFAULT_BASE_URL="https://promaxgb10-6d9f.tail1c73a3.ts.net/v1"
readonly DEFAULT_MODEL="baidu/Unlimited-OCR"
readonly DEFAULT_MAX_TOKENS="8192"
readonly DEFAULT_PAGE_CAP="20"
readonly DEFAULT_CURL_TIMEOUT="3600"

usage() {
  cat <<EOF
Usage:
  $0 INPUT.pdf [OUTPUT_DIRECTORY]

The API key is read from UNLIMITED_API_KEY, API_KEY, or the project's
.env.local file (in that order).

Optional environment variables:
  UNLIMITED_BASE_URL    Default: $DEFAULT_BASE_URL
  UNLIMITED_MODEL       Default: $DEFAULT_MODEL
  UNLIMITED_MAX_TOKENS  Default: $DEFAULT_MAX_TOKENS
  PAGE_CAP              Maximum PDF pages to send; default: $DEFAULT_PAGE_CAP
  CURL_TIMEOUT          Request timeout in seconds; default: $DEFAULT_CURL_TIMEOUT
  PDFIUM_BIN            pypdfium2 executable (auto-detected by default)

Output:
  response.json         Full OpenAI-compatible response
  transcription.raw.txt  Model output including grounding tokens
  transcription.md      Best-effort cleaned Markdown
  pages/                Rendered page images
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

[[ $# -ge 1 && $# -le 2 ]] || {
  usage >&2
  exit 2
}

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
readonly PDF_PATH="$1"

[[ -f "$PDF_PATH" ]] || die "PDF not found: $PDF_PATH"
case "$PDF_PATH" in
  *.[pP][dD][fF]) ;;
  *) die "Input must have a .pdf extension: $PDF_PATH" ;;
esac

for dependency in curl jq base64 python3; do
  command -v "$dependency" >/dev/null 2>&1 || die "$dependency is required"
done

PDFIUM_BIN="${PDFIUM_BIN:-}"
if [[ -z "$PDFIUM_BIN" ]]; then
  PDFIUM_BIN="$(command -v pypdfium2 || true)"
fi
[[ -n "$PDFIUM_BIN" && -x "$PDFIUM_BIN" ]] || die \
  "pypdfium2 was not found. Install it with: python3 -m pip install pypdfium2 Pillow"

api_key="${UNLIMITED_API_KEY:-${API_KEY:-}}"
if [[ -z "$api_key" && -f "$PROJECT_ROOT/.env.local" ]]; then
  while IFS='=' read -r env_name env_value; do
    if [[ "$env_name" == "API_KEY" ]]; then
      api_key="$env_value"
      break
    fi
  done < "$PROJECT_ROOT/.env.local"
fi
[[ -n "$api_key" ]] || die \
  "Set UNLIMITED_API_KEY or API_KEY, or add API_KEY to $PROJECT_ROOT/.env.local"

readonly BASE_URL="${UNLIMITED_BASE_URL:-$DEFAULT_BASE_URL}"
readonly MODEL="${UNLIMITED_MODEL:-$DEFAULT_MODEL}"
readonly MAX_TOKENS="${UNLIMITED_MAX_TOKENS:-$DEFAULT_MAX_TOKENS}"
readonly PAGE_CAP="${PAGE_CAP:-$DEFAULT_PAGE_CAP}"
readonly CURL_TIMEOUT="${CURL_TIMEOUT:-$DEFAULT_CURL_TIMEOUT}"

for value_name in MAX_TOKENS PAGE_CAP CURL_TIMEOUT; do
  value="${!value_name}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$value_name must be a positive integer"
done

pdf_filename="$(basename -- "$PDF_PATH")"
pdf_stem="${pdf_filename%.*}"
timestamp="$(date '+%Y%m%d-%H%M%S')"
readonly OUTPUT_DIR="${2:-./${pdf_stem}-unlimited-ocr-${timestamp}}"

[[ ! -e "$OUTPUT_DIR" ]] || die "Output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/pages"

page_count="$($PDFIUM_BIN pdfinfo "$PDF_PATH" | awk -F ': *' '/^Page Count:/ {print $2; exit}')"
[[ "$page_count" =~ ^[1-9][0-9]*$ ]] || die "Could not determine the PDF page count"

last_page="$page_count"
if (( last_page > PAGE_CAP )); then
  last_page="$PAGE_CAP"
  printf 'PDF has %d pages; processing only the first %d.\n' \
    "$page_count" "$PAGE_CAP" >&2
fi

printf 'Rendering %d page(s) to lossless PNG at PDFium scale 2...\n' "$last_page"
"$PDFIUM_BIN" render "$PDF_PATH" \
  --pages "1-$last_page" \
  --output "$OUTPUT_DIR/pages" \
  --prefix page \
  --format png \
  --engine pil \
  --scale 2 \
  --linear

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/unlimited-ocr.XXXXXX")"
auth_config="$tmp_dir/curl-auth"
images_ndjson="$tmp_dir/images.ndjson"
images_json="$tmp_dir/images.json"
request_json="$tmp_dir/request.json"
cleanup() {
  rm -rf -- "$tmp_dir"
}
trap cleanup EXIT

: > "$auth_config"
chmod 600 "$auth_config"
printf 'header = "Authorization: Bearer %s"\n' "$api_key" > "$auth_config"
unset api_key

: > "$images_ndjson"
page_number=1
while (( page_number <= last_page )); do
  page_png="$OUTPUT_DIR/pages/page${page_number}.png"
  [[ -f "$page_png" ]] || die "Renderer did not create expected file: $page_png"
  printf 'Encoding page %d/%d...\n' "$page_number" "$last_page"

  # Stream base64 into jq so large page payloads never appear in argv.
  base64 < "$page_png" | tr -d '\n' | jq -Rs \
    '{type: "image_url", image_url: {url: ("data:image/png;base64," + .)}}' \
    >> "$images_ndjson"
  ((page_number += 1))
done
jq -s '.' "$images_ndjson" > "$images_json"

if (( last_page == 1 )); then
  prompt='<image>document parsing.'
  ngram_window=128
else
  prompt='<image>Multi page parsing.'
  ngram_window=1024
fi

jq -n \
  --arg model "$MODEL" \
  --arg prompt "$prompt" \
  --argjson max_tokens "$MAX_TOKENS" \
  --argjson ngram_window "$ngram_window" \
  --slurpfile images "$images_json" \
  '{
    model: $model,
    messages: [{
      role: "user",
      content: ([{type: "text", text: $prompt}] + $images[0])
    }],
    max_tokens: $max_tokens,
    temperature: 0.0,
    skip_special_tokens: false,
    vllm_xargs: {
      ngram_size: 35,
      window_size: $ngram_window
    }
  }' > "$request_json"

readonly RESPONSE_JSON="$OUTPUT_DIR/response.json"
readonly RAW_TEXT="$OUTPUT_DIR/transcription.raw.txt"
readonly CLEAN_TEXT="$OUTPUT_DIR/transcription.md"

printf 'Sending %d page(s) to %s at %s...\n' \
  "$last_page" "$MODEL" "${BASE_URL%/}/chat/completions"
if ! curl --silent --show-error --fail-with-body \
    --max-time "$CURL_TIMEOUT" \
    --config "$auth_config" \
    --request POST \
    --url "${BASE_URL%/}/chat/completions" \
    --header 'Content-Type: application/json' \
    --data-binary "@$request_json" \
    --output "$RESPONSE_JSON"; then
  error_message="$(
    jq -r '.error.message // .detail // "curl request failed"' \
      "$RESPONSE_JSON" 2>/dev/null || printf 'curl request failed'
  )"
  die "OCR request failed: $error_message"
fi

if ! jq -er \
    '.choices[0].message.content // "" | select(length > 0)' \
    "$RESPONSE_JSON" > "$RAW_TEXT"; then
  die "Unlimited-OCR returned no content; inspect $RESPONSE_JSON"
fi

# Preserve the raw output and also produce convenient Markdown with the model's
# grounding/reference tokens removed. The raw file remains authoritative.
python3 - "$RAW_TEXT" "$CLEAN_TEXT" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
clean = re.sub(r"<\|ref\|>(.*?)<\|/ref\|>", r"\1", source, flags=re.DOTALL)
clean = re.sub(r"<\|det\|>.*?<\|/det\|>", "", clean, flags=re.DOTALL)
clean = re.sub(r"[ \t]+\n", "\n", clean)
clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
Path(sys.argv[2]).write_text(clean + "\n")
PY

printf '\nComplete.\n'
printf '  Clean Markdown: %s\n' "$CLEAN_TEXT"
printf '  Raw model text: %s\n' "$RAW_TEXT"
printf '  API response:   %s\n' "$RESPONSE_JSON"
