"""Unsloth-based SFT for Gemma 4 31B — drop-in replacement for distill_train.py.

Why this exists
---------------
Stock transformers + peft has irreconcilable memory issues with Gemma 4 on
single A100:
  - head_dim=1024 rejects flash-attn / flex_attention / mem-efficient SDPA
  - 256K vocab + 28K seq → ~14 GB bf16 logits + ~20 GB fp32 upcast = OOM
  - Even with our chunked attention + chunked CE patches, peak hits 79 GB

Unsloth ships custom Triton kernels that handle exactly this combination
("31B QLoRA works with 22GB" per their docs). Same dataset, same LoRA
hyperparams, same output adapter — different training engine.

Usage:
    python -m ml.distill_train_unsloth \
        --dataset       datasets/v14_full_combined.jsonl \
        --adapter-name  gemma4-v14-full-001 \
        --epochs        3 \
        --rank          16 \
        --lr            2e-4 \
        --max-seq-len   28672
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, type=Path,
                   help="JSONL with `messages` records (system/user/assistant)")
    p.add_argument("--adapter-name", required=True)
    p.add_argument("--base", default="unsloth/gemma-4-31B-it",
                   help="Unsloth pre-quantized base (their HF repo). For bf16 "
                        "you can also use google/gemma-4-31B-it but Unsloth's "
                        "4-bit variant is faster to load.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=None,
                   help="LoRA alpha. Default = rank*2 (matches our previous runs).")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=24576,
                   help="Max sequence length. Unsloth handles long context "
                        "with custom kernels; 28K fits on A100-80G.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--output-dir", type=Path, default=Path("data/adapters"))
    p.add_argument("--limit", type=int, default=None,
                   help="Train on only the first N records (smoke test).")
    p.add_argument("--target-modules", nargs="*",
                   default=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
                   help="LoRA target modules (Unsloth uses standard names).")
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()

    # IMPORTANT: unsloth must be imported BEFORE transformers/peft/trl so its
    # monkey-patches kick in. The order matters.
    from unsloth import FastModel
    from unsloth.chat_templates import standardize_data_formats

    import torch
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    adapter_dir = args.output_dir / args.adapter_name
    adapter_dir.mkdir(parents=True, exist_ok=True)

    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.rank * 2

    print("== Configuration ==")
    print(f"  base model:    {args.base}")
    print(f"  dataset:       {args.dataset}")
    print(f"  adapter dir:   {adapter_dir}")
    print(f"  epochs:        {args.epochs}")
    print(f"  rank:          {args.rank} (alpha={lora_alpha})")
    print(f"  lr:            {args.lr}")
    print(f"  max seq len:   {args.max_seq_len}")
    print(f"  batch x accum: {args.batch_size} x {args.grad_accum}")
    print(f"  target_mods:   {args.target_modules}")
    if args.limit:
        print(f"  ⚠  SMOKE TEST: --limit {args.limit}")

    # ── 1. Load model + tokenizer via Unsloth ──────────────────────────────
    print("\n== Loading base via Unsloth (4-bit) ==")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_seq_len,
        dtype=None,                  # auto: bf16 on A100
        load_in_4bit=True,
        full_finetuning=False,
    )

    # ── 2. Attach LoRA via Unsloth ─────────────────────────────────────────
    print("\n== Attaching LoRA adapters (Unsloth) ==")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,   # text-only SFT
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.rank,
        target_modules=args.target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=0,    # Unsloth fast-path requires 0 (warning otherwise)
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's memory-efficient GC
        random_state=args.seed,
    )

    # ── 3. Dataset ─────────────────────────────────────────────────────────
    print("\n== Loading dataset ==")
    ds = load_dataset("json", data_files=str(args.dataset), split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
        print(f"  SMOKE TEST: limited to {len(ds)} records")
    else:
        print(f"  records: {len(ds)}")
    # Normalize message field names if needed (Unsloth's helper)
    try:
        ds = standardize_data_formats(ds)
    except Exception as e:
        print(f"  standardize_data_formats skipped: {e}")

    # ── 4. SFT config ──────────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        learning_rate=args.lr,
        bf16=True,
        # NOTE: do NOT set gradient_checkpointing here — Unsloth's
        # use_gradient_checkpointing="unsloth" already wired it in.
        optim="adamw_8bit",
        logging_steps=2,
        save_steps=args.save_every,
        save_total_limit=2,
        report_to="none",
        max_length=args.max_seq_len,
        packing=False,
        seed=args.seed,
    )

    # Unsloth's wrapped SFTTrainer requires formatting_func to return a LIST
    # of strings (batched form). The function may be called with a single
    # example OR a batched dict-of-lists; handle both safely.
    def _format_chat(examples):
        msgs_field = examples["messages"]
        # Batched: msgs_field is a list of message-lists
        if isinstance(msgs_field, list) and len(msgs_field) > 0 and isinstance(msgs_field[0], list):
            return [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                for m in msgs_field
            ]
        # Single example: msgs_field is a list of {role, content} dicts
        return [
            tokenizer.apply_chat_template(msgs_field, tokenize=False, add_generation_prompt=False)
        ]

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
        "engine": "unsloth",
        "target_modules": args.target_modules,
        "dataset": str(args.dataset),
        "epochs": args.epochs,
        "rank": args.rank,
        "lora_alpha": lora_alpha,
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
