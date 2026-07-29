"""Compatibility patches to make PEFT / LoRA fine-tuning work on Gemma 4.

Two known problems:

1. **Gemma4ClippableLinear wraps q/k/v/o_proj.** LoRA must target the inner
   `.linear` (already handled in distill_train.py's TARGET_MODULES). But the
   wrapper's forward applies `torch.where(out > clip, ...)` style ops that
   can silently zero gradients when activations saturate. The fix: during
   training, replace the wrapper's forward with a pure pass-through.

2. **AWQ-quantized Gemma 4 + gradient checkpointing.** The AWQ Linear's
   weight isn't a leaf tensor, so `enable_input_require_grads()` alone isn't
   enough — gradient checkpointing must use `use_reentrant=False` (already
   handled) AND the wrapper bypass from (1) must be active.

Usage in distill_train.py — call BEFORE `get_peft_model`:

    from ml.gemma4_peft_patch import patch_gemma4_for_training
    if model_type == "gemma4":
        patch_gemma4_for_training(model, verbose=True)
"""
from __future__ import annotations

import torch.nn as nn


def patch_gemma4_for_training(model: nn.Module, verbose: bool = False) -> int:
    """Patch all Gemma4ClippableLinear modules so gradients flow during training.

    Replaces `forward(x) -> clamp(linear(x))` with `forward(x) -> linear(x)`.
    Clipping is restored at eval time when `model.eval()` is called, since the
    patched forward also checks `self.training` and re-applies the clip.

    Returns the number of modules patched.
    """
    patched = 0
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        # Match the wrapper by class name to avoid importing private classes.
        if cls_name not in ("Gemma4ClippableLinear", "Gemma3ClippableLinear"):
            continue
        if not hasattr(module, "linear"):
            continue
        # Preserve original forward for eval-time use.
        if not hasattr(module, "_orig_forward"):
            module._orig_forward = module.forward

        def _patched_forward(self, x, _orig=module._orig_forward):
            if self.training:
                # Skip the clip in training so gradients flow through the
                # inner LoRA-wrapped Linear cleanly.
                return self.linear(x)
            return _orig(x)

        # Bind as a method on this specific instance.
        import types
        module.forward = types.MethodType(_patched_forward, module)
        patched += 1

    if verbose:
        print(f"  patched {patched} Gemma4ClippableLinear module(s) for training")
    return patched
