"""Clean the SFT seed JSONL exported from llm-playground.

Operations:
- Strip ```json ... ``` markdown fences from assistant outputs.
- Validate every assistant output parses as JSON.
- Drop or truncate records whose user input is excessively long
  (default: >40k tokens approx → 160k chars).
- Re-write the system prompt to the short version we built (in case the
  exporter used a different one, or for consistency).

Usage:
    python -m ml.distill_clean \
        --input  data/datasets/Merged_Prompt_V12_-__SFT_-_Gemma4-31b-awq-messages.jsonl \
        --output data/datasets/distill_seed_clean.jsonl \
        --system data/prompts/V12_distill_short.txt \
        --max-user-chars 160000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def strip_fence(text: str) -> str:
    text = text.strip()
    m = FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    return text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--system", required=True, type=Path,
                   help="Path to the canonical short system prompt (.txt)")
    p.add_argument("--strip-prefix-file", type=Path, default=None,
                   help="Path to a .txt file whose contents should be stripped "
                        "from the START of every user message. Used when the "
                        "export embedded a long system prompt inside the user "
                        "message — strip it so only case content remains, then "
                        "the canonical short prompt is prepended.")
    p.add_argument("--max-user-chars", type=int, default=160_000,
                   help="Drop records whose user content exceeds this size (chars). "
                        "Applied AFTER prefix stripping.")
    args = p.parse_args()

    sys_prompt = args.system.read_text().strip()
    strip_prefix = args.strip_prefix_file.read_text().strip() if args.strip_prefix_file else None
    if strip_prefix:
        print(f"  stripping prefix from --strip-prefix-file: {len(strip_prefix):,} chars")

    kept = 0
    dropped_oversize = 0
    dropped_invalid = 0
    dropped_no_prefix = 0
    fences_stripped = 0
    prefixes_stripped = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line_no, raw in enumerate(fin, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"  line {line_no}: invalid JSONL ({e}) — dropped", file=sys.stderr)
                dropped_invalid += 1
                continue

            msgs = rec.get("messages") or []
            user_msg = next((m for m in msgs if m.get("role") == "user"), None)
            asst_msg = next((m for m in msgs if m.get("role") == "assistant"), None)
            if not user_msg or not asst_msg:
                print(f"  line {line_no}: missing user/assistant — dropped", file=sys.stderr)
                dropped_invalid += 1
                continue

            user_content = user_msg["content"] or ""

            if strip_prefix:
                if user_content.startswith(strip_prefix):
                    user_content = user_content[len(strip_prefix):].lstrip("\n").lstrip()
                    prefixes_stripped += 1
                else:
                    print(f"  line {line_no}: user content does NOT start with strip-prefix — dropped", file=sys.stderr)
                    dropped_no_prefix += 1
                    continue

            if len(user_content) > args.max_user_chars:
                print(f"  line {line_no}: user content {len(user_content):,} chars > {args.max_user_chars:,} — dropped", file=sys.stderr)
                dropped_oversize += 1
                continue

            asst_clean = strip_fence(asst_msg["content"] or "")
            if asst_clean != asst_msg["content"]:
                fences_stripped += 1

            try:
                json.loads(asst_clean)
            except json.JSONDecodeError as e:
                print(f"  line {line_no}: assistant content is not valid JSON ({e}) — dropped", file=sys.stderr)
                dropped_invalid += 1
                continue

            new_rec = {
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": asst_clean},
                ]
            }
            fout.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"\n=== distill_clean summary ===")
    print(f"  kept:                {kept}")
    print(f"  dropped (oversize):  {dropped_oversize}")
    print(f"  dropped (invalid):   {dropped_invalid}")
    if strip_prefix:
        print(f"  dropped (no prefix): {dropped_no_prefix}")
        print(f"  prefixes stripped:   {prefixes_stripped}")
    print(f"  markdown fences stripped: {fences_stripped}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
