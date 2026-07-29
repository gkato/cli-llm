"""Synthetic expansion of the SFT seed dataset.

For each seed (input, output) pair we ask Gemini to produce N paraphrased
variants of the *input grievance text* while preserving the underlying
classification facts. We then validate each variant against the original
output by checking that running it through the teacher (Gemini) produces
the same `document_classification`. Variants that drift in classification
are dropped.

Failure cases get more variants than passing cases — concentrated gradient
pressure on the patterns we want to fix.

Inputs:
- Cleaned seed JSONL (output of distill_clean.py)
- A list of seed indexes that are "Gemma 30B failures we want to amplify"
  (passed via --failure-cases-jsonl, the cases we identified earlier)

Output:
- An expanded JSONL ready for training.

Usage:
    python -m ml.distill_synth \
        --input  data/datasets/distill_seed_clean.jsonl \
        --output data/datasets/distill_train.jsonl \
        --pass-variants 4 \
        --fail-variants 12 \
        --failure-cases data/datasets/gemma_failures.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

# Lazy import google genai so we can run --help without it
def _client():
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY_GOOGLE") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in env / .env.local")
    return genai.Client(api_key=api_key)


PARAPHRASE_INSTR = """You will receive a Medicare grievance/appeal/dispute document. Your task: produce a paraphrased VARIANT that:

1. Keeps the same case classification (same intent, same denial/approval/dispute, same urgency, same parties).
2. Reworks surface details that don't change classification:
   - Swap names, member IDs, NPIs, dates (use plausible but different values)
   - Reword sentences (different phrasing, similar meaning)
   - Re-order paragraphs slightly
   - Vary letter formatting (fax header style, OCR artifacts, line breaks)
3. Preserves clinical facts and the action being requested.
4. Keeps the same approximate length (within 30%) as the input.

OUTPUT: just the variant text. No JSON, no commentary, no preamble. Begin immediately with the variant document text.

ORIGINAL DOCUMENT:
"""


def paraphrase_one(client, original_text: str, model: str = "gemini-2.5-flash") -> str | None:
    """Ask Gemini to produce one paraphrase. Returns variant text or None on failure."""
    try:
        from google.genai import types
        cfg = types.GenerateContentConfig(
            max_output_tokens=16_384,
            temperature=0.85,
            thinking_config=types.ThinkingConfig(thinkingBudget=2_048),
        )
        resp = client.models.generate_content(
            model=model,
            contents=PARAPHRASE_INSTR + "\n\n" + original_text,
            config=cfg,
        )
        return (resp.text or "").strip() or None
    except Exception as e:
        print(f"    paraphrase failed: {e}", file=sys.stderr)
        return None


def label_one(client, system_prompt: str, user_text: str, model: str = "gemini-2.5-flash") -> dict | None:
    """Ask Gemini (acting as teacher) to label a variant. Returns parsed JSON or None."""
    try:
        from google.genai import types
        cfg = types.GenerateContentConfig(
            max_output_tokens=16_384,
            temperature=0.0,
            system_instruction=system_prompt,
            thinking_config=types.ThinkingConfig(thinkingBudget=4_096),
        )
        resp = client.models.generate_content(
            model=model,
            contents=user_text,
            config=cfg,
        )
        text = (resp.text or "").strip()
        # Strip markdown fences if any
        import re
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
        return json.loads(text)
    except Exception as e:
        print(f"    label failed: {e}", file=sys.stderr)
        return None


def classification_of(output_obj: dict) -> str | None:
    try:
        return output_obj["extracted_data"]["document_classification"]
    except (KeyError, TypeError):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--pass-variants", type=int, default=4,
                   help="Variants per passing case (default 4 → 5x including original)")
    p.add_argument("--fail-variants", type=int, default=12,
                   help="Variants per failure case (default 12 → 13x including original)")
    p.add_argument("--failure-cases", type=Path, default=None,
                   help="Optional file with list of case identifiers (one per line) to upweight")
    p.add_argument("--max-cases", type=int, default=None,
                   help="Process only first N seeds (for testing)")
    p.add_argument("--gemini-model", default="gemini-2.5-flash")
    p.add_argument("--sleep", type=float, default=0.5,
                   help="Seconds to sleep between API calls (rate limiting)")
    args = p.parse_args()

    # Load seeds
    seeds: list[dict] = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    if args.max_cases:
        seeds = seeds[:args.max_cases]
    print(f"Loaded {len(seeds)} seeds")

    # Load failure case identifiers (file paths or other markers)
    failure_markers: set[str] = set()
    if args.failure_cases and args.failure_cases.exists():
        for line in args.failure_cases.read_text().splitlines():
            line = line.strip()
            if line:
                failure_markers.add(line)
        print(f"Will amplify {len(failure_markers)} failure-case markers")

    # Reuse the seed's system prompt as the labeling prompt for variants
    system_prompt = seeds[0]["messages"][0]["content"] if seeds else ""

    client = _client()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_kept = 0
    n_drift = 0
    n_failed_paraphrase = 0
    n_failed_label = 0

    with open(args.output, "w") as fout:
        for idx, seed in enumerate(seeds):
            user_msg = seed["messages"][1]["content"]
            asst_msg = seed["messages"][2]["content"]
            try:
                gold = json.loads(asst_msg)
            except json.JSONDecodeError:
                continue
            gold_cls = classification_of(gold)

            # Always include the original
            fout.write(json.dumps(seed, ensure_ascii=False) + "\n")
            n_kept += 1

            # Decide variant count
            is_failure = any(m in user_msg or m in str(seed) for m in failure_markers)
            n_variants = args.fail_variants if is_failure else args.pass_variants
            tag = "FAIL" if is_failure else "pass"
            print(f"\nSeed {idx + 1}/{len(seeds)} [{tag}] ({len(user_msg):,} chars) → generating {n_variants} variants")

            for v in range(n_variants):
                variant_text = paraphrase_one(client, user_msg, args.gemini_model)
                if not variant_text:
                    n_failed_paraphrase += 1
                    continue
                time.sleep(args.sleep)

                variant_label = label_one(client, system_prompt, variant_text, args.gemini_model)
                if variant_label is None:
                    n_failed_label += 1
                    continue
                time.sleep(args.sleep)

                variant_cls = classification_of(variant_label)
                if variant_cls != gold_cls:
                    print(f"    variant {v + 1}: classification drift {gold_cls!r} → {variant_cls!r} (dropped)")
                    n_drift += 1
                    continue

                new_rec = {
                    "messages": [
                        seed["messages"][0],
                        {"role": "user", "content": variant_text},
                        {"role": "assistant", "content": json.dumps(variant_label, ensure_ascii=False, indent=2)},
                    ]
                }
                fout.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                n_kept += 1
                print(f"    variant {v + 1}: kept ({len(variant_text):,} chars)")

    print(f"\n=== distill_synth summary ===")
    print(f"  total kept:                {n_kept}")
    print(f"  dropped (classification drift): {n_drift}")
    print(f"  dropped (paraphrase failed):    {n_failed_paraphrase}")
    print(f"  dropped (labeling failed):      {n_failed_label}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
