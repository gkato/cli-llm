"""Combine a real (seed) and synthetic dataset for SFT, with optional upsampling.

Produces a single JSONL with `messages` records. Optionally upsamples the real
dataset so it dominates loss when a much-larger synthetic set is in the mix.

Example (3× real + 1× synthetic):
    python -m ml.distill_combine \
        --real    datasets/SFT_-_Merged_Prompt_V14_-_Gemma4_31B-AWQ-messages.jsonl \
        --synth   datasets/SFT_-_Merged_Prompt_V14_-_Gemma4_31B-AWQ___Synthetic___5_17_2026-messages.jsonl \
        --output  datasets/distill_train_v14_combined.jsonl \
        --real-multiplier 3
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠ skipped malformed line {i} in {path.name}: {e}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", required=True, type=Path,
                   help="Real (seed) dataset JSONL")
    p.add_argument("--synth", required=True, type=Path,
                   help="Synthetic dataset JSONL")
    p.add_argument("--output", required=True, type=Path,
                   help="Combined output JSONL")
    p.add_argument("--real-multiplier", type=int, default=3,
                   help="Upsample real this many times (default: 3)")
    p.add_argument("--synth-multiplier", type=int, default=1,
                   help="Upsample synthetic this many times (default: 1)")
    p.add_argument("--max-user-chars", type=int, default=None,
                   help="Drop records whose user-message content exceeds this "
                        "many chars (avoids extreme outliers). Default: no cap.")
    p.add_argument("--shuffle", action="store_true", default=True,
                   help="Shuffle the combined output (default on)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    real_rows = load_jsonl(args.real)
    synth_rows = load_jsonl(args.synth)
    print(f"== Loaded ==")
    print(f"  real:  {len(real_rows):>4} records")
    print(f"  synth: {len(synth_rows):>4} records")

    def filter_long(rows: list[dict], label: str) -> list[dict]:
        if not args.max_user_chars:
            return rows
        kept, dropped = [], 0
        for r in rows:
            longest_user = max(
                (len(m["content"]) for m in r.get("messages", []) if m["role"] == "user"),
                default=0,
            )
            if longest_user > args.max_user_chars:
                dropped += 1
            else:
                kept.append(r)
        if dropped:
            print(f"  {label}: dropped {dropped} record(s) over {args.max_user_chars} chars")
        return kept

    real_rows = filter_long(real_rows, "real")
    synth_rows = filter_long(synth_rows, "synth")

    combined = real_rows * args.real_multiplier + synth_rows * args.synth_multiplier
    print(f"== Combined ==")
    print(f"  real  × {args.real_multiplier} = {len(real_rows) * args.real_multiplier}")
    print(f"  synth × {args.synth_multiplier} = {len(synth_rows) * args.synth_multiplier}")
    print(f"  total            = {len(combined)}")

    if args.shuffle:
        random.Random(args.seed).shuffle(combined)
        print(f"  shuffled (seed={args.seed})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✓ Wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
