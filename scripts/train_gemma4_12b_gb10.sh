#!/usr/bin/env bash
# Fine-tune Gemma 4 12B-IT on NVIDIA GB10 / DGX Spark (128 GB unified memory),
# fully detached from the SSH session so it survives drops / laptop sleep.
#
# Sibling of scripts/train_gemma4_gb10.sh (the 31B script) and built on the
# same detach pattern as scripts/train_distill.sh:
#   1. Token-filters the dataset to MAX_SEQ_LEN (foreground, fast).
#   2. Launches ml.distill_train_hf with setsid + nohup so it's reparented to
#      init and ignores SIGHUP — closing your SSH session can't kill it.
#   3. Streams stdout+stderr to data/logs/gemma4_12b_<adapter>_<ts>.log.
#   4. Writes data/distill_train.pid + data/distill_train.meta so the existing
#      scripts/train_status.sh reports on it unchanged.
#
# 12B-specific defaults:
#   - bf16 LoRA (the 12B base is only ~24 GB; no reason to quantize on 128 GB
#     unified). Pass QLORA=1 only if you push seq_len/batch high enough to OOM.
#   - BASE_MODEL=google/gemma-4-12B-it (matches the gemma4-12b-it bf16 registry
#     entry used to serve the merged result).
#   - MAX_SEQ_LEN=25600 is a starting default. RUN THE TOKEN REPORT FIRST to
#     fit it to this dataset's actual max, then override:
#       make -f Makefile.gb10 report \
#         DATASET=datasets/Merged_Prompt_Fine_Tuning_V2_-_Gemma4_12B-messages.jsonl \
#         BASE_MODEL=google/gemma-4-12B-it MAX_SEQ_LEN=25600
#
# Usage:
#   scripts/train_gemma4_12b_gb10.sh                 # uses defaults below
#   DATASET=... ADAPTER_NAME=... scripts/train_gemma4_12b_gb10.sh
#   MAX_SEQ_LEN=20480 EPOCHS=4 scripts/train_gemma4_12b_gb10.sh
#
# After it starts:
#   bash scripts/train_status.sh            # check progress
#   bash scripts/train_status.sh --tail     # follow live
#   kill $(cat data/distill_train.pid)      # stop training

set -euo pipefail

cd "$(dirname "$0")/.."   # project root

# ── Config (override via env) ────────────────────────────────────────────────
DATASET=${DATASET:-datasets/Merged_Prompt_Fine_Tuning_V2_-_Gemma4_12B-messages.jsonl}
ADAPTER_NAME=${ADAPTER_NAME:-gemma4-12b-gb10-001}
BASE_MODEL=${BASE_MODEL:-google/gemma-4-12B-it}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-25600}    # starting default — run `report` to fit to dataset
EPOCHS=${EPOCHS:-3}
RANK=${RANK:-16}
LR=${LR:-2e-4}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
SAVE_EVERY=${SAVE_EVERY:-30}
QLORA=${QLORA:-}   # default OFF (bf16 LoRA); set QLORA=1 to fall back if OOM

# Reduce CUDA fragmentation — lets the allocator return memory to the device.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_DIR=data/logs
PID_FILE=data/distill_train.pid
META_FILE=data/distill_train.meta
TRAIN_SET="${DATASET%.jsonl}_train.jsonl"
QLORA_FLAG=$([ -n "$QLORA" ] && echo "--qlora" || echo "")
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="$LOG_DIR/gemma4_12b_${ADAPTER_NAME}_${TIMESTAMP}.log"

echo "=========================================="
echo "Gemma 4 12B-IT fine-tune on GB10 (detached)"
echo "=========================================="
echo "  Dataset:        $DATASET"
echo "  Train set:      $TRAIN_SET"
echo "  Adapter name:   $ADAPTER_NAME"
echo "  Base model:     $BASE_MODEL"
echo "  Max seq len:    $MAX_SEQ_LEN"
echo "  Epochs:         $EPOCHS"
echo "  Rank:           $RANK (alpha=$((RANK*2)))"
echo "  Learning rate:  $LR"
echo "  Batch × accum:  $BATCH_SIZE × $GRAD_ACCUM (effective $((BATCH_SIZE*GRAD_ACCUM)))"
echo "  Save every:     $SAVE_EVERY steps"
echo "  Precision:      $([ -n "$QLORA" ] && echo 'QLoRA (4-bit base)' || echo 'bf16 LoRA')"
echo "  Log:            $LOG_FILE"
echo "=========================================="
echo ""
echo "VRAM expectation on GB10 (128 GB unified):"
if [ -n "$QLORA" ]; then
  echo "  - 4-bit base:        ~7 GB"
  echo "  - LoRA + optimizer:  ~1 GB"
  echo "  - Activations @  25k: ~18 GB → peak ~26 GB  ✓ trivial fit"
else
  echo "  - bf16 base:         ~24 GB"
  echo "  - LoRA + optimizer:  ~1 GB"
  echo "  - Activations @  16k: ~12 GB → peak ~37 GB   ✓ comfortable"
  echo "  - Activations @  25k: ~18 GB → peak ~43 GB   ✓ comfortable"
  echo "  - Activations @  30k: ~24 GB → peak ~49 GB   ✓ fits easily"
fi
echo ""
echo "Expected wall-time on Gemma 4 12B (177 records × $EPOCHS epochs):"
echo "  - 16k:  ~1.5-3 h   25k:  ~3-5 h   30k:  ~4-6 h"
echo "Attention compute is O(n²) — long context is expensive."
echo ""

# ── Pre-flight ───────────────────────────────────────────────────────────────
echo "== Pre-flight =="

# 1. Already running?
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "✗ A training run is already active (PID $(cat "$PID_FILE"))."
  echo "  Stop it first:  kill \$(cat $PID_FILE)"
  exit 1
fi
[ -f "$PID_FILE" ] && rm -f "$PID_FILE"

# 2. Dataset present?
if [ ! -f "$DATASET" ]; then
  echo "✗ Dataset not found at $DATASET"
  exit 1
fi
echo "✓ dataset:     $DATASET ($(wc -l < "$DATASET") records)"

# 3. vLLM not holding the GPU?
if python3 -m ml.cli status 2>/dev/null | grep -q "ready ✓"; then
  echo "✗ vLLM is running and would compete for memory."
  echo "  Stop it first:  python3 -m ml.cli stop"
  exit 1
fi

# 4. GPU present?
if command -v nvidia-smi >/dev/null; then
  echo "✓ GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
  echo "⚠  nvidia-smi not found — continuing, but check you're on the GB10."
fi

# ── Step 1: token-filter (foreground) ────────────────────────────────────────
echo ""
echo "== Token-cap $DATASET to $MAX_SEQ_LEN tokens =="
python3 -m ml.distill_token_filter \
  --input      "$DATASET" \
  --output     "$TRAIN_SET" \
  --base       "$BASE_MODEL" \
  --max-tokens "$MAX_SEQ_LEN"
echo "✓ filtered → $TRAIN_SET ($(wc -l < "$TRAIN_SET") records kept)"

# ── Step 2: launch training detached ─────────────────────────────────────────
echo ""
echo "== Launching detached training =="
# setsid + nohup: new session (no controlling tty) + ignore SIGHUP, so closing
# your SSH session can't kill it. stderr → stdout into one log. stdin from
# /dev/null so distill_train_hf never blocks on a prompt.
setsid nohup python3 -u -m ml.distill_train_hf \
  --dataset      "$TRAIN_SET" \
  --adapter-name "$ADAPTER_NAME" \
  --base         "$BASE_MODEL" \
  --epochs       "$EPOCHS" \
  --rank         "$RANK" \
  --lr           "$LR" \
  --max-seq-len  "$MAX_SEQ_LEN" \
  --batch-size   "$BATCH_SIZE" \
  --grad-accum   "$GRAD_ACCUM" \
  --save-every   "$SAVE_EVERY" \
  $QLORA_FLAG \
  > "$LOG_FILE" 2>&1 < /dev/null &

PID=$!
echo "$PID" > "$PID_FILE"

# Confirm it didn't immediately crash.
sleep 3
if kill -0 "$PID" 2>/dev/null; then
  echo "✓ Training started — PID $PID"
else
  echo "✗ Training failed to start. Last lines of the log:"
  tail -50 "$LOG_FILE" || true
  rm -f "$PID_FILE"
  exit 1
fi

# Metadata for scripts/train_status.sh
cat > "$META_FILE" <<EOF
adapter_name=$ADAPTER_NAME
log_file=$LOG_FILE
pid=$PID
started_at=$(date -Iseconds)
dataset=$TRAIN_SET
base_model=$BASE_MODEL
epochs=$EPOCHS
rank=$RANK
lr=$LR
max_seq_len=$MAX_SEQ_LEN
EOF

echo ""
echo "Monitor:"
echo "  bash scripts/train_status.sh           # progress + latest loss"
echo "  bash scripts/train_status.sh --tail    # follow live"
echo "  tail -f $LOG_FILE"
echo ""
echo "Stop:"
echo "  kill \$(cat $PID_FILE)"
echo ""
echo "Adapter will be saved to: data/adapters/$ADAPTER_NAME"
