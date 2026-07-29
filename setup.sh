#!/bin/bash
set -e

echo "=== ml-compute setup (vLLM) ==="

cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

# Remote-mode = disposable container/VM (RunPod, Vast.ai, k8s pod). Skip
# the venv, install globally. On a personal workstation we always venv.
REMOTE_MODE=${REMOTE_MODE:-0}
if [ -n "$VAST_CONTAINERD_ID" ] || [ -n "$KUBERNETES_SERVICE_HOST" ] || [ -n "$RUNPOD_POD_ID" ]; then
  REMOTE_MODE=1
fi

# Architecture detection — NVIDIA GB10 / DGX Spark is aarch64 with Grace CPU.
# Many ML wheels (vLLM, xformers, flash-attn, unsloth) are x86-only.
ARCH=$(uname -m)
IS_ARM64=0
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
  IS_ARM64=1
fi

# 1. Persistent directories
echo "Creating data directories..."
mkdir -p data/logs data/hf_cache

# 2. Python detection — pick the newest available 3.x
PY=""
for cand in python3.12 python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    PY="$cand"
    break
  fi
done
echo "Using Python: $PY ($($PY --version))"
echo "Architecture: $ARCH"

# 3. Python environment
if [ "$REMOTE_MODE" -eq 0 ]; then
  if [ ! -d venv ]; then
    echo "Creating Python virtual environment..."
    "$PY" -m venv venv
  fi
  source venv/bin/activate
  PY="python"
fi

# ---------------------------------------------------------------------------
# Pip flags for PEP 668 (Ubuntu/Debian externally-managed environments)
# ---------------------------------------------------------------------------
PIP_FLAGS=""
if [ "$REMOTE_MODE" -eq 1 ]; then
  if "$PY" -m pip install --dry-run --quiet pip 2>&1 | grep -q "externally-managed-environment"; then
    PIP_FLAGS="--break-system-packages"
    echo "  (PEP 668 environment — using --break-system-packages)"
  fi
fi

echo "Installing Python dependencies (this may take a while — vLLM is large)..."
"$PY" -m pip install $PIP_FLAGS --upgrade pip

# ---------------------------------------------------------------------------
# Architecture-aware install: requirements.txt has the x86 path; GB10/arm64
# needs a different set (HF stack only; vLLM has experimental ARM support
# but Unsloth and friends do not).
# ---------------------------------------------------------------------------
if [ "$IS_ARM64" -eq 1 ]; then
  echo ""
  echo "→ ARM64 detected — installing GB10-optimized stack (no Unsloth)."
  echo "  For serving on GB10 we recommend NVIDIA NIM / TensorRT-LLM containers."
  echo "  This pip install gives you training (HF + PEFT + accelerate)."
  echo ""
  "$PY" -m pip install $PIP_FLAGS \
    huggingface-hub hf-transfer hf-xet \
    click pyyaml python-dotenv requests \
    torch torchvision torchaudio \
    transformers peft accelerate trl datasets safetensors \
    sentencepiece protobuf tokenizers
  # vLLM on ARM64 is experimental; try, fall back gracefully if it fails
  if "$PY" -m pip install $PIP_FLAGS vllm 2>/tmp/vllm_install.log; then
    echo "  ✓ vLLM installed (ARM64 build)"
  else
    echo "  ⚠ vLLM install failed on ARM64 — see /tmp/vllm_install.log"
    echo "    For serving use NVIDIA NIM: docker run nvcr.io/nim/..."
  fi
else
  "$PY" -m pip install $PIP_FLAGS -r requirements.txt
fi

# 4. .env.local: bootstrap and ensure API_KEY is present
if [ ! -f .env.local ]; then
  cp .env.example .env.local
  echo "→ Created .env.local from template"
fi

if ! grep -q "^API_KEY=.\+" .env.local; then
  KEY="sk-mlc-$("$PY" -c 'import secrets; print(secrets.token_urlsafe(24))')"
  if grep -q "^API_KEY=" .env.local; then
    sed -i.bak "s|^API_KEY=.*|API_KEY=$KEY|" .env.local && rm -f .env.local.bak
  else
    echo "API_KEY=$KEY" >> .env.local
  fi
  echo "→ Generated API_KEY in .env.local"
fi

# 5. GPU check + arch-specific env hints
echo ""
echo "Checking GPU..."
"$PY" - <<'PY' || echo "⚠ Could not check GPU (torch not loaded yet — vLLM will install it)"
try:
    import torch, platform
    if torch.cuda.is_available():
        d = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        cap = torch.cuda.get_device_capability(0)
        sm = f"sm_{cap[0]}{cap[1]}"
        cpu_arch = platform.machine()
        print(f"✓ GPU:  {d}")
        print(f"  VRAM: {vram:.1f} GB ({sm})")
        print(f"  CPU:  {cpu_arch}")
        # Detect GB10 / Grace+Blackwell unified-memory architecture
        is_gb10 = (cpu_arch in ("aarch64", "arm64")) and cap[0] >= 12
        if is_gb10:
            print("  → GB10 / DGX Spark detected (Grace + Blackwell, unified memory)")
            print("    Recommended:")
            print("      - Training: bf16 LoRA via HF+accelerate (no QLoRA needed)")
            print("      - Serving:  NVIDIA NIM container, or vLLM if it installed")
            print("    Env vars:")
            print(f"      export TORCH_CUDA_ARCH_LIST=\"{cap[0]}.{cap[1]}+PTX\"")
            print(f"      export VLLM_FLASHINFER_FORCE_TARGET=sm_{cap[0]}{cap[1]}")
        elif cap[0] >= 12:
            print("  → Blackwell detected. Recommended env vars:")
            print(f"      export TORCH_CUDA_ARCH_LIST=\"{cap[0]}.{cap[1]}+PTX\"")
            print(f"      export VLLM_FLASHINFER_FORCE_TARGET=sm_{cap[0]}{cap[1]}")
    else:
        print("⚠ No CUDA GPU detected (vLLM requires a GPU)")
except Exception as e:
    print(f"⚠ {e}")
PY

# 6. Persist alias so future `python3 -m ml.cli ...` works after pod restart
if [ "$REMOTE_MODE" -eq 1 ] && [ "$PY" != "python3" ]; then
  if ! grep -q "alias python3=$PY" ~/.bashrc 2>/dev/null; then
    echo "alias python3=$PY" >> ~/.bashrc
    echo "→ Added 'alias python3=$PY' to ~/.bashrc"
  fi
fi

# 7. Friendly next steps
API_KEY_VALUE=$(grep "^API_KEY=" .env.local | cut -d= -f2-)
echo ""
echo "✓ Setup complete"
echo ""
echo "Quick start:"
if [ "$REMOTE_MODE" -eq 0 ]; then
  echo "  source venv/bin/activate"
fi
if [ "$IS_ARM64" -eq 1 ]; then
  echo "  $PY -m ml.cli info                                # check stack health"
  echo "  make -f Makefile.gb10 train DATASET=...           # bf16 LoRA training"
  echo "  # For serving, prefer NVIDIA NIM containers on GB10"
else
  echo "  $PY -m ml.cli models pull qwen3.5-4b-fp8"
  echo "  $PY -m ml.cli serve qwen3.5-4b-fp8"
  echo "  $PY -m ml.cli status"
fi
echo ""
echo "Your API key (use this from llm-playground):"
echo "  $API_KEY_VALUE"
echo ""
echo "To call the server later:"
echo "  curl http://localhost:8000/v1/models -H 'Authorization: Bearer \$API_KEY'"
