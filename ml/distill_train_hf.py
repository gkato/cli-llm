"""SFT LoRA fine-tuning via HF Transformers + PEFT + TRL (no Unsloth).

Built for NVIDIA GB10 / DGX Spark (Grace+Blackwell, unified memory). Unsloth
relies on x86-only Triton kernels and does not install cleanly on aarch64.
This script uses only the mainline HF stack, which has full ARM64 support.

Differences vs distill_train_unsloth.py:
  - Loads base in bf16 (not 4-bit). With GB10's ~96 GB unified memory the
    quality cost of QLoRA isn't worth it; plain LoRA gives cleaner gradients.
  - Uses HF Trainer + TRL's SFTTrainer (same TRL version as the Unsloth path).
  - Gradient checkpointing via PyTorch native (works on ARM64).
  - Optional 4-bit base if you really need the VRAM headroom: --qlora flag.

Usage:
    python -m ml.distill_train_hf \
        --dataset       datasets/my_train.jsonl \
        --adapter-name  my-ft-001 \
        --base          Qwen/Qwen3.5-9B \
        --epochs        3 \
        --rank          16 \
        --lr            2e-4 \
        --max-seq-len   16384 \
        --batch-size    1 \
        --grad-accum    8
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--adapter-name", required=True)
    p.add_argument("--base", required=True,
                   help="HF model id, e.g. Qwen/Qwen3.5-9B")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=None,
                   help="LoRA alpha. Default = rank*2.")
    p.add_argument("--lora-dropout", type=float, default=0.05,
                   help="LoRA dropout (HF supports non-zero, unlike Unsloth).")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=16384)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--save-every", type=int, default=40)
    p.add_argument("--output-dir", type=Path, default=Path("data/adapters"))
    p.add_argument("--limit", type=int, default=None,
                   help="Train on only the first N records (smoke test).")
    p.add_argument("--biggest-first", action="store_true",
                   help="Sort the dataset by total content length DESC before "
                        "applying --limit. Used to stress-test VRAM headroom "
                        "on the worst-case (longest) samples.")
    p.add_argument("--target-modules", nargs="*",
                   default=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])
    p.add_argument("--qlora", action="store_true",
                   help="Load base in 4-bit (QLoRA). Default is full bf16. "
                        "Only use this if you're VRAM-constrained — on GB10 "
                        "the 96 GB unified memory makes bf16 the better choice.")
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    adapter_dir = args.output_dir / args.adapter_name
    adapter_dir.mkdir(parents=True, exist_ok=True)

    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.rank * 2

    print("== Configuration ==")
    print(f"  base model:    {args.base}")
    print(f"  dataset:       {args.dataset}")
    print(f"  adapter dir:   {adapter_dir}")
    print(f"  precision:     {'4-bit (QLoRA)' if args.qlora else 'bf16 (LoRA)'}")
    print(f"  epochs:        {args.epochs}")
    print(f"  rank:          {args.rank} (alpha={lora_alpha}, dropout={args.lora_dropout})")
    print(f"  lr:            {args.lr}")
    print(f"  max seq len:   {args.max_seq_len}")
    print(f"  batch x accum: {args.batch_size} x {args.grad_accum}")
    print(f"  target_mods:   {args.target_modules}")
    if args.limit:
        print(f"  ⚠  SMOKE TEST: --limit {args.limit}")

    # ── 1. Tokenizer + model ────────────────────────────────────────────────
    print("\n== Loading tokenizer + base ==")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detect multimodal models (e.g. Gemma 4) — they have a vision tower
    # we need to freeze for text-only LoRA fine-tuning, AND accelerate's
    # device_map="auto" trips on the vision tower's buffers. Force
    # device_map={"":0} so everything sits on the single GPU — correct
    # for single-GPU boxes (RunPod, GB10) anyway.
    is_multimodal = False
    try:
        _cfg = AutoConfig.from_pretrained(args.base, trust_remote_code=False)
        is_multimodal = (hasattr(_cfg, "vision_config")
                         or "vision_tower" in str(_cfg).lower())
    except Exception:
        pass
    if is_multimodal:
        print(f"  → Multimodal model detected — will freeze vision tower for text-only LoRA")

    if args.qlora:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            quantization_config=bnb,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # Plain bf16 — preferred on GB10's unified memory.
        # device_map={"":0} keeps everything on GPU 0 (incl. multimodal
        # buffers that "auto" can't place); fine on single-GPU boxes.
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",  # SDPA works on ARM64; flash-attn is x86-only
        )

    # Freeze vision tower so LoRA only sees text-side gradients
    if is_multimodal and hasattr(model, "vision_tower"):
        frozen = 0
        for p in model.vision_tower.parameters():
            p.requires_grad = False
            frozen += p.numel()
        print(f"  ✓ Froze vision tower ({frozen/1e6:.1f}M params)")

    model.config.use_cache = False  # required when grad-checkpointing

    # ── 2. LoRA ────────────────────────────────────────────────────────────
    # On multimodal models, the same projection names (q_proj, k_proj, ...)
    # exist in BOTH the vision tower and the language model. Restrict LoRA
    # to the language-model side via PEFT's target_modules regex so we
    # don't waste capacity on the (frozen) vision tower.
    if is_multimodal:
        target_modules_arg = "^(?:.*language_model.*|.*model\\.layers.*)" \
                             "\\.(?:" + "|".join(args.target_modules) + ")$"
        print(f"\n== Attaching LoRA adapters (text-only via regex) ==")
    else:
        target_modules_arg = args.target_modules
        print(f"\n== Attaching LoRA adapters ==")

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules_arg,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── 3. Dataset ─────────────────────────────────────────────────────────
    print("\n== Loading dataset ==")
    ds = load_dataset("json", data_files=str(args.dataset), split="train")

    if args.biggest_first:
        def _len_score(rec):
            return sum(len(m.get("content", "")) for m in rec.get("messages", []))
        # datasets supports add_column → sort → remove_column
        ds = ds.map(lambda r: {"_len": _len_score(r)})
        ds = ds.sort("_len", reverse=True)
        ds = ds.remove_columns("_len")
        print(f"  --biggest-first: sorted by total content length, descending")

    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
        kind = "BIGGEST" if args.biggest_first else "FIRST"
        print(f"  SMOKE TEST: kept {kind} {len(ds)} records")
    else:
        print(f"  records: {len(ds)}")

    # ── 4. SFT config + trainer ────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",        # plain AdamW; on GB10 the 8-bit optim isn't needed
        logging_steps=2,
        save_steps=args.save_every,
        save_total_limit=2,
        report_to="none",
        max_length=args.max_seq_len,
        packing=False,
        seed=args.seed,
        dataloader_num_workers=2,
    )

    def _format_chat(examples):
        msgs = examples["messages"]
        if isinstance(msgs, list) and len(msgs) > 0 and isinstance(msgs[0], list):
            return [tokenizer.apply_chat_template(m, tokenize=False,
                                                  add_generation_prompt=False) for m in msgs]
        return [tokenizer.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=False)]

    print("\n== Building SFTTrainer ==")
    try:
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=ds,
            processing_class=tokenizer,
            formatting_func=_format_chat,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=ds,
            tokenizer=tokenizer,
            formatting_func=_format_chat,
        )

    print("\n== Training ==")
    result = trainer.train()

    print("\n== Saving adapter ==")
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metadata = {
        "adapter_name": args.adapter_name,
        "base_model": args.base,
        "engine": "hf+peft+trl",
        "precision": "4bit_qlora" if args.qlora else "bf16_lora",
        "target_modules": args.target_modules,
        "dataset": str(args.dataset),
        "epochs": args.epochs,
        "rank": args.rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": args.lora_dropout,
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
