#!/usr/bin/env bash

# Standalone PDF OCR through a Gemma vision model.
#
# This script does not call LLM Playground. It reproduces its wire-level flow:
#   PDF -> first 20 pages as PNG (PDFium/PIL, scale 2) -> one sequential
#   OpenAI-compatible /chat/completions request per page -> combined text.

set -euo pipefail

readonly DEFAULT_BASE_URL="https://localhost:8000/v1"
readonly DEFAULT_MODEL="nvidia/Gemma-4-31B-IT-NVFP4"
readonly DEFAULT_MAX_TOKENS="8192"
readonly PAGE_CAP=20

readonly OCR_PROMPT='Transcribe this document page exactly as it appears. Preserve the original layout, headers, lists, tables, dates, IDs, and any structured content. Do not summarize, interpret, or paraphrase.

Output ONLY the transcribed text — no commentary, no markdown fences, no preamble, no <think> tags.'

usage() {
  printf '%s\n' \
    "Usage:" \
    "  GEMMA_API_KEY=... $0 INPUT.pdf [OUTPUT_DIRECTORY]" \
    "" \
    "Optional environment variables:" \
    "  GEMMA_BASE_URL    Default: $DEFAULT_BASE_URL" \
    "  GEMMA_MODEL       Default: $DEFAULT_MODEL" \
    "  GEMMA_MAX_TOKENS  Default: $DEFAULT_MAX_TOKENS" \
    "  PDFIUM_BIN        pypdfium2 executable (auto-detected by default)"
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

readonly PDF_PATH="$1"
[[ -f "$PDF_PATH" ]] || die "PDF not found: $PDF_PATH"
case "$PDF_PATH" in
  *.[pP][dD][fF]) ;;
  *) die "Input must have a .pdf extension: $PDF_PATH" ;;
esac

command -v curl >/dev/null 2>&1 || die "curl is required"
command -v jq >/dev/null 2>&1 || die "jq is required"
command -v base64 >/dev/null 2>&1 || die "base64 is required"

PDFIUM_BIN="${PDFIUM_BIN:-}"
if [[ -z "$PDFIUM_BIN" ]]; then
  PDFIUM_BIN="$(command -v pypdfium2 || true)"
fi
[[ -n "$PDFIUM_BIN" && -x "$PDFIUM_BIN" ]] || die \
  "pypdfium2 was not found. Install it with: python3 -m pip install pypdfium2 Pillow"

: "${GEMMA_API_KEY:?Set GEMMA_API_KEY to the model server API key}"
readonly GEMMA_BASE_URL="${GEMMA_BASE_URL:-$DEFAULT_BASE_URL}"
readonly GEMMA_MODEL="${GEMMA_MODEL:-$DEFAULT_MODEL}"
readonly GEMMA_MAX_TOKENS="${GEMMA_MAX_TOKENS:-$DEFAULT_MAX_TOKENS}"
[[ "$GEMMA_MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || die \
  "GEMMA_MAX_TOKENS must be a positive integer"

pdf_filename="$(basename "$PDF_PATH")"
pdf_stem="${pdf_filename%.*}"
timestamp="$(date '+%Y%m%d-%H%M%S')"
readonly OUTPUT_DIR="${2:-./${pdf_stem}-gemma-ocr-${timestamp}}"

[[ ! -e "$OUTPUT_DIR" ]] || die "Output path already exists: $OUTPUT_DIR"
mkdir "$OUTPUT_DIR"
mkdir "$OUTPUT_DIR/pages" "$OUTPUT_DIR/responses" "$OUTPUT_DIR/text"

page_count="$("$PDFIUM_BIN" pdfinfo "$PDF_PATH" | awk -F ': *' '/^Page Count:/ {print $2; exit}')"
[[ "$page_count" =~ ^[1-9][0-9]*$ ]] || die "Could not determine the PDF page count"

last_page="$page_count"
if (( last_page > PAGE_CAP )); then
  last_page="$PAGE_CAP"
  printf 'PDF has %d pages; processing only the first %d.\n' "$page_count" "$PAGE_CAP" >&2
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

readonly COMBINED_TEXT="$OUTPUT_DIR/transcription.txt"
: > "$COMBINED_TEXT"

page_number=1
success_count=0
while (( page_number <= last_page )); do
  page_png="$OUTPUT_DIR/pages/page${page_number}.png"
  response_json="$OUTPUT_DIR/responses/page${page_number}.json"
  page_text="$OUTPUT_DIR/text/page${page_number}.txt"

  [[ -f "$page_png" ]] || die "Renderer did not create expected file: $page_png"
  printf 'Sending page %d/%d to %s...\n' "$page_number" "$last_page" "$GEMMA_MODEL"

  # Keep the large base64 payload off the process argument list: jq reads it
  # from stdin and constructs the OpenAI-compatible multimodal JSON request.
  if base64 < "$page_png" | tr -d '\n' | jq -Rs \
      --arg model "$GEMMA_MODEL" \
      --arg prompt "$OCR_PROMPT" \
      --argjson max_tokens "$GEMMA_MAX_TOKENS" \
      '{
        model: $model,
        messages: [{
          role: "user",
          content: [
            {type: "text", text: $prompt},
            {
              type: "image_url",
              image_url: {url: ("data:image/png;base64," + .)}
            }
          ]
        }],
        max_tokens: $max_tokens,
        temperature: 0.0
      }' | curl --silent --show-error --fail-with-body \
        --request POST \
        --url "${GEMMA_BASE_URL%/}/chat/completions" \
        --header "Authorization: Bearer $GEMMA_API_KEY" \
        --header 'Content-Type: application/json' \
        --data-binary @- \
        --output "$response_json"; then
    if jq -e '
      (.choices[0].message.content? // "" | sub("^\\s+"; "") | sub("\\s+$"; "") | length) > 0
    ' "$response_json" >/dev/null; then
      jq -jr '
        .choices[0].message.content
        | sub("^\\s+"; "")
        | sub("\\s+$"; "")
      ' "$response_json" > "$page_text"
      ((success_count += 1))
    else
      jq -jr --argjson page "$page_number" '
        "[Page \($page) OCR returned empty content]"
      ' > "$page_text"
    fi
  else
    error_message="$(jq -r '.error.message // .detail // "curl request failed"' "$response_json" 2>/dev/null || printf 'curl request failed')"
    printf '[Page %d OCR failed: %s]' "$page_number" "$error_message" > "$page_text"
    printf 'Page %d failed: %s\n' "$page_number" "$error_message" >&2
  fi

  if (( page_number > 1 )); then
    printf '\n\n--- PAGE BREAK ---\n\n' >> "$COMBINED_TEXT"
  fi
  cat "$page_text" >> "$COMBINED_TEXT"

  ((page_number += 1))
done

if (( success_count == 0 )); then
  die "Gemma produced no usable content across all pages; see $OUTPUT_DIR/responses"
fi

printf '\nComplete: %s\n' "$COMBINED_TEXT"
printf 'Successful pages: %d/%d\n' "$success_count" "$last_page"
