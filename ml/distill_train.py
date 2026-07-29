"""QLoRA fine-tune a base model on the distillation dataset.

Supports two flavors of base model:

  1. **bf16 base** (e.g. google/gemma-4-31B-it):
       loaded with bitsandbytes 4-bit (NF4) at runtime.
       Uses prepare_model_for_kbit_training.

  2. **Pre-quantized AWQ base** (e.g. QuantTrio/Qwen3.5-27B-AWQ):
       loaded directly; weights are already 4-bit.
       LoRA adapter is attached on top of frozen AWQ layers.

Detection is automatic: the model's `config.json` is inspected for a
`quantization_config.quant_method == "awq"` field. If present → AWQ path.

Uses transformers + peft + trl SFTTrainer (no Unsloth — transformers 5.8
is too new for current Unsloth, plain peft is more reliable).

Inputs:
- JSONL with `messages` records (system / user / assistant)
- A base model (HF id or local path)

Output:
- LoRA adapter saved to data/adapters/<adapter-name>/

Usage:
    # Smoke test (5 min, 20 examples)
    python -m ml.distill_train \
        --dataset      data/datasets/distill_train.jsonl \
        --adapter-name qwen35-smoke \
        --base         QuantTrio/Qwen3.5-27B-AWQ \
        --epochs       1 \
        --limit        20

    # Full run
    python -m ml.distill_train \
        --dataset      data/datasets/distill_train.jsonl \
        --adapter-name v12-qwen35-distill-001 \
        --base         QuantTrio/Qwen3.5-27B-AWQ \
        --epochs       3 \
        --rank         16 \
        --lr           2e-4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# Per-architecture LoRA target_modules.
#
# Gemma 4 is multimodal: the model has both a `vision_tower` (which uses
# Gemma4ClippableLinear with an inner `.linear`) and a `language_model`
# (Gemma4TextAttention with plain nn.Linear projections). Text-only inputs
# only flow through the language_model. Targeting `q_proj.linear` previously
# matched only the vision tower → LoRA attached to dead layers → grad_norm=0.
#
# Use a regex string (PEFT treats `target_modules: str` as a regex via
# re.fullmatch) to scope LoRA to the language model's attention only.
#
# Qwen / Llama / Mistral / etc. use plain nn.Linear — standard names work.
TARGET_MODULES_BY_MODEL_TYPE: dict[str, list[str] | str] = {
    "gemma4": r".*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
    "qwen2":  ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qwen3":  ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qwen3_5":["q_proj", "k_proj", "v_proj", "o_proj"],
    "llama":  ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mistral":["q_proj", "k_proj", "v_proj", "o_proj"],
}
TARGET_MODULES_DEFAULT = ["q_proj", "k_proj", "v_proj", "o_proj"]


def detect_quantization(base: str) -> tuple[bool, str | None]:
    """Return (is_pre_quantized, quant_method). Looks at the model's config.json.

    Pre-quantized formats we recognize:
        - AWQ ("quant_method": "awq")
        - GPTQ ("quant_method": "gptq")
        - bitsandbytes 4-bit / 8-bit (rare — usually quantized at runtime)
    """
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(base, trust_remote_code=False)
        qc = getattr(cfg, "quantization_config", None) or {}
        if isinstance(qc, dict):
            qm = qc.get("quant_method")
        else:
            qm = getattr(qc, "quant_method", None)
        return (bool(qm), qm)
    except Exception:
        return (False, None)


def detect_model_type(base: str) -> str:
    """Read model_type from config.json (e.g. 'gemma4', 'qwen3', 'qwen3_5', 'llama')."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(base, trust_remote_code=False)
        return getattr(cfg, "model_type", "unknown")
    except Exception:
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--adapter-name", required=True)
    p.add_argument("--base", default="google/gemma-4-31B-it")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=4096,
                   help="Truncate inputs longer than this (token count).")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--output-dir", type=Path, default=Path("data/adapters"))
    p.add_argument("--limit", type=int, default=None,
                   help="Train on only the first N examples (smoke test). "
                        "Verifies grad_norm > 0 in a few minutes before "
                        "committing to a full run.")
    p.add_argument("--target-modules", nargs="*", default=None,
                   help="Override LoRA target modules (default: auto by model_type).")
    p.add_argument("--attn-impl", default="auto",
                   choices=["auto", "flash_attention_2", "sdpa", "eager",
                            "flex_attention", "chunked"],
                   help="Attention backend. For Gemma 4 (head_dim=1024) on a "
                        "single A100, neither flash-attn (≤256), flex_attention "
                        "(SM budget), nor sdpa math (OOM at seq 24K) work. The "
                        "'chunked' option uses our pure-PyTorch query-chunked "
                        "attention — O(chunk × seq) memory, unbounded head_dim.")
    p.add_argument("--debug-modules", action="store_true",
                   help="Print trainable param patterns + sample module names "
                        "after LoRA attachment, then exit. Use to diagnose "
                        "target_modules matching issues.")
    p.add_argument("--no-grad-ckpt", action="store_true",
                   help="Disable gradient checkpointing. Uses more VRAM but "
                        "isolates GC as a possible gradient-killer.")
    p.add_argument("--liger", action="store_true",
                   help="Enable Liger Kernel (fused linear cross-entropy). "
                        "Eliminates the [seq x vocab] logits OOM, allowing "
                        "much longer seq lengths. Requires `pip install liger-kernel`.")
    p.add_argument("--ce-chunk", type=int, default=256,
                   help="Chunk size (in tokens) for the chunked cross-entropy "
                        "patch. Smaller = less peak memory per chunk. Default 256 "
                        "(~256 MB upcast for 256K-vocab models). Reduce if you "
                        "still OOM in CE; raise if you have spare memory.")
    p.add_argument("--strip-vision", action="store_true",
                   help="For Gemma 4 (multimodal): delete the vision_tower and "
                        "multi_modal_projector from VRAM after model load. They "
                        "are unused for text-only SFT and free ~2-3 GB.")
    args = p.parse_args()

    # Compatibility shim: peft 0.19.x expects an older gptqmodel class name
    # for AWQ adapter dispatch. gptqmodel 7.0+ renamed it. Alias it before
    # peft tries to import.
    try:
        import gptqmodel.nn_modules.qlinear.gemm_awq as _gemm_awq
        if not hasattr(_gemm_awq, "AwqGEMMQuantLinear") and hasattr(_gemm_awq, "AwqGEMMLinear"):
            _gemm_awq.AwqGEMMQuantLinear = _gemm_awq.AwqGEMMLinear
    except ImportError:
        pass

    import torch
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    adapter_dir = args.output_dir / args.adapter_name
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # ── Detect base model flavor ────────────────────────────────────────────
    is_quantized, quant_method = detect_quantization(args.base)
    model_type = detect_model_type(args.base)

    if args.target_modules:
        target_modules = list(args.target_modules)
        target_source = "explicit --target-modules"
    else:
        target_modules = TARGET_MODULES_BY_MODEL_TYPE.get(model_type, TARGET_MODULES_DEFAULT)
        target_source = f"auto for model_type={model_type}"

    print(f"== Configuration ==")
    print(f"  base model:    {args.base}")
    print(f"  model_type:    {model_type}")
    if is_quantized:
        print(f"  quant_method:  {quant_method} (pre-quantized — skipping bnb 4-bit)")
    else:
        print(f"  quant_method:  none (will load with bitsandbytes 4-bit NF4)")
    print(f"  target_mods:   {target_modules}  [{target_source}]")
    print(f"  dataset:       {args.dataset}")
    print(f"  adapter dir:   {adapter_dir}")
    print(f"  epochs:        {args.epochs}")
    print(f"  rank:          {args.rank}")
    print(f"  lr:            {args.lr}")
    print(f"  max seq len:   {args.max_seq_len}")
    print(f"  batch x accum: {args.batch_size} x {args.grad_accum}")
    if args.limit:
        print(f"  ⚠  SMOKE TEST: --limit {args.limit}")

    # ── 1. Tokenizer ────────────────────────────────────────────────────────
    print("\n== Loading tokenizer ==")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. Model ────────────────────────────────────────────────────────────
    # Resolve attention implementation. Flash-attn 2 is constant-memory in
    # seq length — required to fit longer sequences on a single A6000.
    def _resolve_attn(choice: str) -> str:
        if choice != "auto":
            return choice
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except ImportError:
            return "sdpa"
    attn_impl = _resolve_attn(args.attn_impl)

    # Register our custom chunked attention BEFORE loading the model so
    # transformers' AttentionInterface knows about it. Must happen here.
    if attn_impl == "chunked":
        from ml.gemma4_chunked_attn import register_chunked_attention
        register_chunked_attention()

    print(f"  attn impl:     {attn_impl}")

    common_load_kwargs = dict(
        device_map="auto",
        trust_remote_code=False,
        attn_implementation=attn_impl,
    )

    if is_quantized:
        # Pre-quantized (AWQ/GPTQ). Load directly; weights are already 4-bit.
        # LoRA adapter sits on top of frozen quantized layers.
        print("\n== Loading pre-quantized base ==")
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            torch_dtype=torch.bfloat16,  # for LoRA + activations
            **common_load_kwargs,
        )
        # Freeze all base weights — they can't be trained anyway, but be explicit.
        for param in model.parameters():
            param.requires_grad = False
        model.config.use_cache = False  # required for gradient checkpointing
        # Enable input grad so checkpointed activations can backprop through to LoRA.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        # bf16 base — wrap in bnb 4-bit at load time (classic QLoRA).
        print("\n== Loading base model in 4-bit (bnb NF4) ==")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            **common_load_kwargs,
        )
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False

    # ── 2b. Gemma 4 wrapper patch (gradient flow fix) ───────────────────────
    if model_type in ("gemma4", "gemma3"):
        from ml.gemma4_peft_patch import patch_gemma4_for_training
        print(f"\n== Patching {model_type} clippable-linear wrappers ==")
        patch_gemma4_for_training(model, verbose=True)

    # ── 2c. Optionally strip the vision tower (Gemma 4 multimodal) ─────────
    if args.strip_vision and model_type == "gemma4":
        print("\n== Stripping vision tower + multi-modal projector ==")
        # The multimodal wrapper is model.model (Gemma4Model). Vision lives there.
        inner = getattr(model, "model", None)
        freed_any = False
        for attr in ("vision_tower", "multi_modal_projector"):
            container = inner if inner is not None and hasattr(inner, attr) else (
                model if hasattr(model, attr) else None
            )
            if container is None:
                continue
            sub = getattr(container, attr)
            n_params = sum(p.numel() for p in sub.parameters())
            print(f"  removing {attr}: {n_params:,} params")
            # Replace with Identity so any forward references don't crash.
            setattr(container, attr, torch.nn.Identity())
            del sub
            freed_any = True
        if freed_any:
            import gc as _gc
            _gc.collect()
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            print(f"  GPU free after strip: {free / 1e9:.2f} GB / {total / 1e9:.2f} GB")

    # ── 3. LoRA ─────────────────────────────────────────────────────────────
    print("\n== Attaching LoRA adapters ==")
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if args.debug_modules:
        print("\n== Trainable parameter detail (by pattern) ==")
        import re as _re
        by_pattern: dict[str, list[tuple[str, int, tuple]]] = {}
        for n, prm in model.named_parameters():
            if not prm.requires_grad:
                continue
            pat = _re.sub(r"\.\d+\.", ".<i>.", n)
            by_pattern.setdefault(pat, []).append((n, prm.numel(), tuple(prm.shape)))
        for pat in sorted(by_pattern):
            items = by_pattern[pat]
            total = sum(x[1] for x in items)
            _, _, shape = items[0]
            print(f"  {pat:80s} count={len(items):>4} shape={shape!s:30s} sum={total:>14,}")
        print("\n== Sample of attention modules in the model (first 6) ==")
        seen = 0
        for n, mod in model.named_modules():
            if "self_attn" in n and n.count(".") <= 6 and seen < 6:
                print(f"  {n}: {type(mod).__name__}")
                seen += 1
        print("\n(exiting because --debug-modules)")
        sys.exit(0)

    # ── 4. Dataset ──────────────────────────────────────────────────────────
    print("\n== Loading dataset ==")
    ds = load_dataset("json", data_files=str(args.dataset), split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
        print(f"  SMOKE TEST: limited to {len(ds)} records")
    else:
        print(f"  records: {len(ds)}")

    # ── 5. Training config ──────────────────────────────────────────────────
    tokenizer.model_max_length = args.max_seq_len

    # Liger Kernel: fused linear cross-entropy. Eliminates the [seq x vocab]
    # logits tensor in the LM head, which is the dominant memory hog for
    # long seq + large vocab (Gemma ~256K vocab). Patches transformers in-place.
    if args.liger:
        try:
            import liger_kernel.transformers as _lk
        except ImportError:
            print("  Liger Kernel: requested but not installed — `pip install liger-kernel`")
            raise
        # Try model-specific patch first (gemma4), fall back to gemma3.
        # Gemma 4 shares architectural primitives with Gemma 3 (RMSNorm,
        # RoPE, GeGLU MLP) so the gemma3 patches usually apply cleanly.
        applied_via = None
        for fn_name in (f"apply_liger_kernel_to_{model_type}",
                        "apply_liger_kernel_to_gemma3"):
            if hasattr(_lk, fn_name):
                kw = dict(rope=True, cross_entropy=False,
                          fused_linear_cross_entropy=True, rms_norm=True)
                # geglu/swiglu name varies by family — try geglu, ignore TypeError.
                try:
                    getattr(_lk, fn_name)(geglu=True, **kw)
                except TypeError:
                    getattr(_lk, fn_name)(**kw)
                applied_via = fn_name
                break
        if applied_via:
            print(f"  Liger Kernel: enabled via {applied_via}")
        else:
            print(f"  Liger Kernel: no patch function found for model_type={model_type}")

        # Belt-and-suspenders: Liger's per-model patch may not actually
        # affect Gemma 4 (the gemma3 patch targets a different module path).
        # Monkey-patch transformers' generic CE utility to process logits in
        # chunks, eliminating the [seq x vocab] fp32 upcast (~6 GB on Gemma).
        # This affects every model that uses transformers.loss.loss_utils.
        from transformers.loss import loss_utils as _loss_utils
        _orig_fce = _loss_utils.fixed_cross_entropy
        _ce_chunk = args.ce_chunk

        def _chunked_fixed_cross_entropy(source, target, num_items_in_batch=None,
                                         ignore_index=-100, **kwargs):
            import torch.nn.functional as F
            N = source.shape[0]
            if N <= _ce_chunk:
                return _orig_fce(source, target, num_items_in_batch,
                                 ignore_index, **kwargs)
            # Sum-reduction in chunks → avoid full [N x V] fp32 upcast.
            total = torch.zeros((), device=source.device, dtype=torch.float32)
            for i in range(0, N, _ce_chunk):
                total = total + F.cross_entropy(
                    source[i:i + _ce_chunk], target[i:i + _ce_chunk],
                    ignore_index=ignore_index, reduction="sum",
                )
            if num_items_in_batch is None:
                num_items_in_batch = (target != ignore_index).sum()
            if isinstance(num_items_in_batch, torch.Tensor):
                num_items_in_batch = num_items_in_batch.to(total.device)
            return total / num_items_in_batch

        _loss_utils.fixed_cross_entropy = _chunked_fixed_cross_entropy
        print(f"  patched transformers.loss.loss_utils.fixed_cross_entropy → "
              f"chunked (chunk_size={_ce_chunk})")

        # ALSO patch ForCausalLMLoss itself — it calls `logits.float()` on
        # the FULL [B, S, V] tensor BEFORE delegating to fixed_cross_entropy,
        # which alone allocates ~20 GB at seq 28K on Gemma's 256K vocab and
        # OOMs even though our chunked CE below would handle it. Replace
        # the whole function with a chunked variant that upcasts per-chunk.
        import torch.nn as _nn
        def _chunked_for_causal_lm_loss(logits, labels, vocab_size,
                                        num_items_in_batch=None, ignore_index=-100,
                                        shift_labels=None, **kwargs):
            if shift_labels is None:
                labels = _nn.functional.pad(labels, (0, 1), value=ignore_index)
                shift_labels = labels[..., 1:].contiguous()
            logits_flat = logits.view(-1, vocab_size)       # bf16, no upcast yet
            labels_flat = shift_labels.view(-1).to(logits_flat.device)
            return _chunked_fixed_cross_entropy(
                logits_flat, labels_flat, num_items_in_batch, ignore_index, **kwargs
            )

        _loss_utils.ForCausalLMLoss = _chunked_for_causal_lm_loss
        # Also patch the LOSS_MAPPING registry so models pick up the new fn
        # via self.loss_function lookups.
        try:
            from transformers.loss.loss_utils import LOSS_MAPPING
            for k in list(LOSS_MAPPING.keys()):
                if "ForCausalLMLoss" in k or k.endswith("ForCausalLM"):
                    LOSS_MAPPING[k] = _chunked_for_causal_lm_loss
        except Exception:
            pass
        print(f"  patched transformers.loss.loss_utils.ForCausalLMLoss → chunked")

        # Most important: `loss_function` is bound on the model instance at
        # __init__ time, so patching LOSS_MAPPING after the fact is too late.
        # Walk the wrappers (PEFT → LoraModel → real model → inner language_model)
        # and override the attribute directly on every candidate.
        def _install_loss_fn(target_model, fn):
            patched_targets = []
            visited = set()
            def _walk(obj, depth=0):
                if id(obj) in visited or depth > 4: return
                visited.add(id(obj))
                # Try to install on this object
                try:
                    object.__setattr__(obj, "loss_function", fn)
                    patched_targets.append(type(obj).__name__)
                except Exception:
                    pass
                # Try common wrapper attribute names
                for attr in ("base_model", "model", "language_model"):
                    sub = getattr(obj, attr, None)
                    if sub is not None and hasattr(sub, "forward"):
                        _walk(sub, depth + 1)
            _walk(target_model)
            return patched_targets

        _patched_on = _install_loss_fn(model, _chunked_for_causal_lm_loss)
        print(f"  installed chunked loss_function on: {_patched_on}")

    sft_config = SFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=2,           # log more frequently — smoke tests benefit
        save_steps=args.save_every,
        save_total_limit=2,
        report_to="none",
        max_length=args.max_seq_len,
        packing=False,
    )

    try:
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=ds,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=ds,
            tokenizer=tokenizer,
        )

    print("\n== Training ==")
    result = trainer.train()

    print("\n== Saving adapter ==")
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metadata = {
        "adapter_name": args.adapter_name,
        "base_model": args.base,
        "model_type": model_type,
        "quant_method": quant_method,
        "target_modules": target_modules,
        "dataset": str(args.dataset),
        "epochs": args.epochs,
        "rank": args.rank,
        "lr": args.lr,
        "max_seq_len": args.max_seq_len,
        "limit": args.limit,
        "final_loss": float(result.training_loss) if hasattr(result, "training_loss") else None,
    }
    (adapter_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\n✓ Adapter saved to {adapter_dir}")
    print(f"  final_loss: {metadata.get('final_loss')}")


if __name__ == "__main__":
    main()
