#!/usr/bin/env bash
# Show the status of a background distillation training run started by
# scripts/train_distill.sh. Survives SSH reconnects — pulls everything from
# disk (PID file, log file, metadata).
#
# Usage:
#   bash scripts/train_status.sh
#   bash scripts/train_status.sh --tail            # also tail the log live
#   bash scripts/train_status.sh --gpu             # also show nvidia-smi
#
set -euo pipefail

cd "$(dirname "$0")/.."

PID_FILE=data/distill_train.pid
META_FILE=data/distill_train.meta

TAIL=0
GPU=0
for arg in "$@"; do
  case "$arg" in
    --tail) TAIL=1 ;;
    --gpu)  GPU=1 ;;
    *) echo "unknown arg: $arg"; exit 2 ;;
  esac
done

if [ ! -f "$PID_FILE" ]; then
  echo "No training run on record (no $PID_FILE)."
  echo "Start one with: bash scripts/train_distill.sh"
  exit 0
fi

PID=$(cat "$PID_FILE")
LOG_FILE=""
ADAPTER_NAME=""
STARTED_AT=""
if [ -f "$META_FILE" ]; then
  while IFS='=' read -r k v; do
    case "$k" in
      log_file)     LOG_FILE="$v" ;;
      adapter_name) ADAPTER_NAME="$v" ;;
      started_at)   STARTED_AT="$v" ;;
    esac
  done < "$META_FILE"
fi

echo "== distillation training =="
echo "  pid:        $PID"
echo "  adapter:    $ADAPTER_NAME"
echo "  started:    $STARTED_AT"
echo "  log:        $LOG_FILE"

# Is the process alive?
if kill -0 "$PID" 2>/dev/null; then
  STATE="running"
  # How long has it been running?
  if [ -n "$STARTED_AT" ]; then
    START_SEC=$(date -d "$STARTED_AT" +%s 2>/dev/null || echo 0)
    NOW_SEC=$(date +%s)
    if [ "$START_SEC" -gt 0 ]; then
      ELAPSED=$((NOW_SEC - START_SEC))
      H=$((ELAPSED / 3600))
      M=$(((ELAPSED % 3600) / 60))
      S=$((ELAPSED % 60))
      echo "  elapsed:    ${H}h ${M}m ${S}s"
    fi
  fi
else
  STATE="not running (process gone)"
fi
echo "  state:      $STATE"

# Latest log progress
if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
  LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
  echo "  log size:   $LOG_SIZE"

  # Most recent training step (loss + epoch)
  LAST_STEP=$(grep -E "'loss':" "$LOG_FILE" 2>/dev/null | tail -1 || true)
  if [ -n "$LAST_STEP" ]; then
    # Pull loss / epoch / lr if present
    LOSS=$(echo "$LAST_STEP" | grep -oE "'loss': [0-9.]+" | head -1 | awk '{print $2}')
    EPOCH=$(echo "$LAST_STEP" | grep -oE "'epoch': [0-9.]+" | head -1 | awk '{print $2}')
    LR_OUT=$(echo "$LAST_STEP" | grep -oE "'learning_rate': [0-9.eE+-]+" | head -1 | awk '{print $2}')
    echo ""
    echo "  Latest step:"
    echo "    loss:  $LOSS"
    echo "    epoch: $EPOCH"
    echo "    lr:    $LR_OUT"
  fi

  # Has training completed?
  if grep -q "Adapter saved to" "$LOG_FILE" 2>/dev/null; then
    echo ""
    echo "✓ Training completed successfully."
    grep "final_loss" "$LOG_FILE" 2>/dev/null | tail -1 || true
  fi

  # Any errors recently?
  ERRORS=$(grep -iE "error|traceback|exception" "$LOG_FILE" 2>/dev/null | tail -3 || true)
  if [ -n "$ERRORS" ]; then
    echo ""
    echo "⚠  Recent errors:"
    echo "$ERRORS" | sed 's/^/    /'
  fi
fi

# GPU
if [ "$GPU" -eq 1 ]; then
  echo ""
  echo "== GPU =="
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
             --format=csv,noheader
fi

# Tail
if [ "$TAIL" -eq 1 ] && [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
  echo ""
  echo "== Live tail (Ctrl+C to exit) =="
  tail -n 40 -f "$LOG_FILE"
fi
