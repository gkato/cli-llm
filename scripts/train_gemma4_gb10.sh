#!/bin/bash
# Fine-tune Gemma 4 31B on NVIDIA GB10 / DGX Spark (128 GB unified memory).
#
# Wraps Makefile.gb10 with Gemma 4-specific defaults:
#   - bf16 LoRA by default (cleaner gradients than QLoRA, and you have
#     the memory for it on 128 GB unified). Pass QLORA=1 to fall back
#     to 4-bit base if you OOM at high seq_len.
#   - MAX_SEQ_LEN=30720 matches the upstream 30k token filter.
#   - SAVE_EVERY=30 — frequent enough to recover, sparse enough not to
#     waste NVMe on a 200+ step run.
#
# Usage:
#   scripts/train_gemma4_gb10.sh                   # uses defaults below
#   DATASET=... ADAPTER_NAME=... scripts/train_gemma4_gb10.sh
#
# Override any param via env:
#   MAX_SEQ_LEN=20480 BATCH_SIZE=1 scripts/train_gemma4_gb10.sh

set -e
cd "$(dirname "$0")/.."

DATASET=${DATASET:-datasets/Gemma_4_31B_-FP8_GB10_-_125_Cases____Synthetic3_-_Merged_-_Cleaned_-_PII_Fix-messages.jsonl}
ADAPTER_NAME=${ADAPTER_NAME:-gemma4-31b-gb10-001}
BASE_MODEL=${BASE_MODEL:-google/gemma-4-31B-it}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-25600}    # 25k — tight fit to dataset's actual max (~24,819 tokens)
EPOCHS=${EPOCHS:-3}
RANK=${RANK:-16}
LR=${LR:-2e-4}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
SAVE_EVERY=${SAVE_EVERY:-30}
QLORA=${QLORA:-}   # default OFF (bf16 LoRA); set QLORA=1 to fall back if OOM

echo "=========================================="
echo "Gemma 4 31B fine-tune on GB10"
echo "=========================================="
echo "  Dataset:        $DATASET"
echo "  Adapter name:   $ADAPTER_NAME"
echo "  Base model:     $BASE_MODEL"
echo "  Max seq len:    $MAX_SEQ_LEN"
echo "  Epochs:         $EPOCHS"
echo "  Rank:           $RANK (alpha=$((RANK*2)))"
echo "  Learning rate:  $LR"
echo "  Batch × accum:  $BATCH_SIZE × $GRAD_ACCUM (effective $((BATCH_SIZE*GRAD_ACCUM)))"
echo "  Save every:     $SAVE_EVERY steps"
echo "  Precision:      $([ -n "$QLORA" ] && echo 'QLoRA (4-bit base)' || echo 'bf16 LoRA')"
echo "=========================================="
echo ""
echo "VRAM expectation on GB10 (128 GB unified):"
if [ -n "$QLORA" ]; then
  echo "  - 4-bit base:        ~16 GB"
  echo "  - LoRA + optimizer:  ~1 GB"
  echo "  - Activations @  16k: ~25 GB → peak ~42 GB"
  echo "  - Activations @  24k: ~35 GB → peak ~52 GB"
  echo "  - Activations @  30k: ~45 GB → peak ~62 GB  ✓ trivial fit"
else
  echo "  - bf16 base:         ~62 GB"
  echo "  - LoRA + optimizer:  ~1 GB"
  echo "  - Activations @  16k: ~22 GB → peak ~85 GB    ✓ comfortable"
  echo "  - Activations @  24k: ~30 GB → peak ~93 GB    ✓ fits"
  echo "  - Activations @  30k: ~40 GB → peak ~103 GB   ⚠ tight, smoke first!"
fi
echo ""
echo "Expected wall-time at ${MAX_SEQ_LEN} seq_len on Gemma 4 31B:"
echo "  - 16k:  ~10-15 h"
echo "  - 24k:  ~18-25 h"
echo "  - 30k:  ~25-40 h"
echo "Attention compute is O(n²) — long context is expensive."
echo ""
echo "If bf16 LoRA OOMs mid-run: pass QLORA=1 and retrain."
echo ""

# Skip the interactive prompt under nohup / non-tty.
if [ -t 0 ]; then
  read -p "Press Enter to start, or Ctrl+C to abort..." _
fi

make -f Makefile.gb10 train \
  DATASET="$DATASET" \
  ADAPTER_NAME="$ADAPTER_NAME" \
  BASE_MODEL="$BASE_MODEL" \
  MAX_SEQ_LEN="$MAX_SEQ_LEN" \
  EPOCHS="$EPOCHS" \
  RANK="$RANK" \
  LR="$LR" \
  BATCH_SIZE="$BATCH_SIZE" \
  GRAD_ACCUM="$GRAD_ACCUM" \
  SAVE_EVERY="$SAVE_EVERY" \
  QLORA="$QLORA"
