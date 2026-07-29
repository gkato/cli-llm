"""Drop dataset records whose chat-template token count exceeds a cap.

Uses the actual target-model tokenizer (not chars/4 estimates), so the
output is a guaranteed-fits-in-max_seq_length training set.

Usage:
    python -m ml.distill_token_filter \
        --input    datasets/Qwen_3.5_4B_FP8_-_CMS_Optimized_V3_-_Unified-messages_clean.jsonl \
        --output   datasets/Qwen_3.5_4B_FP8_-_CMS_Optimized_V3_-_train.jsonl \
        --base     Qwen/Qwen3.5-4B \
        --max-tokens 8192
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path,
                   help="JSONL with `messages` records")
    p.add_argument("--output", required=True, type=Path,
                   help="Filtered JSONL output")
    p.add_argument("--base", required=True,
                   help="HF model id whose tokenizer to use (e.g. Qwen/Qwen3.5-4B)")
    p.add_argument("--max-tokens", type=int, required=True,
                   help="Drop records whose chat-template tokenization exceeds this")
    p.add_argument("--report-only", action="store_true",
                   help="Print stats and what would be dropped; do not write output")
    args = p.parse_args()

    from transformers import AutoTokenizer

    print(f"== Loading tokenizer: {args.base} ==")
    tok = AutoTokenizer.from_pretrained(args.base)

    def count(messages: list[dict]) -> int:
        out = tok.apply_chat_template(messages, tokenize=True,
                                      add_generation_prompt=False)
        # transformers ≥5 returns BatchEncoding; ≤4 returns a plain list of ids.
        if hasattr(out, "input_ids"):
            return len(out["input_ids"])
        return len(out)

    kept, dropped = [], []
    with args.input.open() as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            n = count(rec["messages"])
            if n > args.max_tokens:
                dropped.append((line_no, n))
            else:
                kept.append((rec, n))

    if dropped:
        print(f"\n== Dropped {len(dropped)} record(s) > {args.max_tokens} tokens ==")
        for line_no, n in sorted(dropped, key=lambda x: -x[1])[:20]:
            print(f"  line {line_no}: {n:,} tokens")

    # Stats on what we kept
    counts = sorted(n for _, n in kept)
    if counts:
        n = len(counts)
        print(f"\n== Kept {n} record(s) ==")
        print(f"  p50={counts[n//2]:>6,}  "
              f"p90={counts[int(n*0.90)]:>6,}  "
              f"p95={counts[int(n*0.95)]:>6,}  "
              f"p99={counts[int(n*0.99)]:>6,}  "
              f"max={counts[-1]:>6,}")

    if args.report_only:
        print("\n(report-only — no file written)")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fout:
        for rec, _ in kept:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n✓ Wrote {args.output} ({args.output.stat().st_size/1e6:.2f} MB, "
          f"{len(kept)} records)")


if __name__ == "__main__":
    main()
