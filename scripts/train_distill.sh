#!/usr/bin/env bash
# Run distillation training in the background, fully detached from the SSH
# session. Survives SSH drops, moving between networks, laptop sleep, etc.
#
# What it does:
#   1. Sanity-checks that vLLM isn't running (training needs full GPU).
#   2. Verifies the base model and dataset are in place.
#   3. Launches the training script with nohup + setsid so the process is
#      reparented to init and ignores SIGHUP. The child has no controlling
#      terminal, so closing your SSH session can't kill it.
#   4. Writes stdout/stderr to data/logs/distill_train_<timestamp>.log.
#   5. Saves the PID to data/distill_train.pid so you can stop or check on it.
#
# Usage:
#   bash scripts/train_distill.sh                # uses defaults
#   ADAPTER_NAME=foo bash scripts/train_distill.sh   # override adapter name
#
# After it starts, you can:
#   - bash scripts/train_status.sh               # check progress
#   - tail -f data/logs/distill_train_*.log      # follow live
#   - kill $(cat data/distill_train.pid)         # stop training
#
set -euo pipefail

# ── Config (override via env) ────────────────────────────────────────────────
ADAPTER_NAME=${ADAPTER_NAME:-v12-gemma4-distill-001}
DATASET=${DATASET:-data/datasets/distill_train.jsonl}
BASE_MODEL=${BASE_MODEL:-google/gemma-4-31B-it}
EPOCHS=${EPOCHS:-3}
RANK=${RANK:-16}
LR=${LR:-2e-4}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}

# Reduce CUDA fragmentation. PyTorch's default allocator strands free memory
# in unmappable chunks; expandable_segments lets it return memory to the
# device. Saves 1-3 GB of effective VRAM during QLoRA training.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/.."   # project root

LOG_DIR=data/logs
PID_FILE=data/distill_train.pid
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="$LOG_DIR/distill_train_${ADAPTER_NAME}_${TIMESTAMP}.log"

# ── Pre-flight checks ────────────────────────────────────────────────────────

echo "== Pre-flight =="

# 1. Already running?
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "✗ Training already running (PID $(cat $PID_FILE))."
  echo "  Use 'kill \$(cat $PID_FILE)' to stop it first, or wait for it to finish."
  exit 1
fi
[ -f "$PID_FILE" ] && rm -f "$PID_FILE"

# 2. vLLM running?
if python3 -m ml.cli status 2>/dev/null | grep -q "ready ✓"; then
  echo "✗ vLLM is currently running and would compete for GPU."
  echo "  Stop it first:  python3 -m ml.cli stop"
  exit 1
fi

# 3. Dataset exists?
if [ ! -f "$DATASET" ]; then
  echo "✗ Dataset not found at $DATASET"
  echo "  Generate it with:  make -f Makefile.distill clean-data && python3 -m ml.distill_augment ..."
  exit 1
fi
N_LINES=$(wc -l < "$DATASET")
echo "✓ dataset:     $DATASET ($N_LINES records)"

# 4. Base model cached?
BASE_DIR="/workspace/.cache/huggingface/hub/models--${BASE_MODEL//\//--}"
if [ -d "$BASE_DIR" ]; then
  BASE_SIZE=$(du -sh "$BASE_DIR" 2>/dev/null | cut -f1)
  echo "✓ base model:  $BASE_MODEL ($BASE_SIZE cached)"
else
  echo "⚠  base model not found at $BASE_DIR — script may have to download it."
fi

# 5. GPU available?
if ! command -v nvidia-smi >/dev/null; then
  echo "✗ nvidia-smi not available — no GPU?"
  exit 1
fi
GPU_USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$GPU_USED_MIB" -gt 1000 ]; then
  echo "⚠  GPU has $GPU_USED_MIB MiB already used. May be tight."
fi
echo "✓ GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# 6. Disk free
DISK_FREE=$(df -h /workspace | awk 'NR==2 {print $4}')
echo "✓ disk free:   $DISK_FREE"

# ── Launch ──────────────────────────────────────────────────────────────────

echo ""
echo "== Launching training =="
echo "  adapter:     $ADAPTER_NAME"
echo "  epochs:      $EPOCHS"
echo "  rank:        $RANK"
echo "  lr:          $LR"
echo "  max_seq_len: $MAX_SEQ_LEN"
echo "  log:         $LOG_FILE"
echo "  pid file:    $PID_FILE"
echo ""

# setsid + nohup makes the process fully detached:
#   - new session (no controlling tty) so SSH disconnect can't send SIGHUP
#   - nohup also ignores SIGHUP as a belt-and-suspenders
#   - reparented to init via the double-fork that setsid creates
# Note: stderr -> stdout so we capture everything in one log.
setsid nohup python3 -u -m ml.distill_train \
  --dataset "$DATASET" \
  --adapter-name "$ADAPTER_NAME" \
  --base "$BASE_MODEL" \
  --epochs "$EPOCHS" \
  --rank "$RANK" \
  --lr "$LR" \
  --max-seq-len "$MAX_SEQ_LEN" \
  > "$LOG_FILE" 2>&1 < /dev/null &

PID=$!
echo $PID > "$PID_FILE"

# Wait a moment to confirm it actually started (didn't immediately crash)
sleep 3
if kill -0 "$PID" 2>/dev/null; then
  echo "✓ Training started — PID $PID"
  echo ""
  echo "Monitor:"
  echo "  bash scripts/train_status.sh"
  echo "  tail -f $LOG_FILE"
  echo ""
  echo "Stop:"
  echo "  kill \$(cat $PID_FILE)"
else
  echo "✗ Training failed to start. Check the log:"
  echo "  tail -50 $LOG_FILE"
  rm -f "$PID_FILE"
  exit 1
fi

# Save metadata about this run so train_status.sh can find it
cat > data/distill_train.meta <<EOF
adapter_name=$ADAPTER_NAME
log_file=$LOG_FILE
pid=$PID
started_at=$(date -Iseconds)
dataset=$DATASET
base_model=$BASE_MODEL
epochs=$EPOCHS
rank=$RANK
lr=$LR
max_seq_len=$MAX_SEQ_LEN
EOF
