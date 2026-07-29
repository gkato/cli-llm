"""Build a *focused* fine-tuning dataset from backtest failures.

Strategy: instead of re-training on the broad 704-example set, build a small
high-signal dataset that concentrates on the specific decision boundary the
base model fails at — plus enough correctly-classified "anchor" cases of all
classes to prevent catastrophic forgetting.

Pipeline:

  1. Load the backtest Excel (Results sheet: Case name / PII data / Asserts / Output).
  2. For each row:
       - Parse `Asserts` into (json_path, expected_value) pairs.
       - Parse `Output` into a JSON object (the model's prediction).
       - Determine gold class (from asserts) and predicted class (from output).
       - Mark as FAILURE if any assertion fails AND gold == --failure-class.
       - Mark as ANCHOR if all assertions pass.
  3. For FAILURE cases, build a *corrected* assistant output by:
       a) Starting from the model's own (wrong) JSON  (so structure is intact)
       b) Applying CLASS_DEFAULTS for the gold class (sets service_type,
          clears wrong-class fields)
       c) Applying asserts on top (the human-verified gold)
  4. For ANCHOR cases, use the model's output as-is (we trust it — asserts pass).
  5. Augment heavily via PII substitution (failure cases get more variants).
  6. Write JSONL with {"messages": [system, user, assistant]} records.

Output is ready to feed directly to `python -m ml.distill_train`.

Usage:
    python -m ml.distill_focus \
        --backtest         data/backtest-Merged_Prompt_V12_Gemma_30B-2026-05-11.xlsx \
        --short-prompt     data/prompts/V12_distill_short.txt \
        --output           data/datasets/distill_focus_psa_v1.jsonl \
        --failure-class    "Pre-Service Appeal" \
        --variants-per-failure 40 \
        --variants-per-anchor  15 \
        --anchors-per-class    6
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import openpyxl  # type: ignore[import-not-found]

from ml.distill_augment import (
    apply_substitutions,
    build_substitution_map,
    inject_ocr_noise,
)


# Per-class field defaults — used to patch a misclassified output into a
# correctly-shaped output for the target class. Sets the fields that DO belong
# to the class (e.g. PSA → service_type=Pre-Service) and clears the fields
# that DON'T (e.g. PSA → case_type_post_service="").
#
# Values for the class's "own" case_specific_field (e.g. case_type_pre_service
# for a PSA) come from the asserts; we don't second-guess them.
CLASS_DEFAULTS: dict[str, dict[str, str]] = {
    "Pre-Service Appeal": {
        "extracted_data.service_type": "Pre-Service",
        "extracted_data.case_specific_fields.case_type_post_service": "",
        "extracted_data.case_specific_fields.case_type_grievance": "",
        "extracted_data.case_specific_fields.type_of_issue": "",
    },
    "Post-Service Appeal": {
        "extracted_data.service_type": "Post-Service",
        "extracted_data.case_specific_fields.case_type_pre_service": "",
        "extracted_data.case_specific_fields.case_type_grievance": "",
        "extracted_data.case_specific_fields.type_of_issue": "",
    },
    "Grievance": {
        "extracted_data.service_type": "",
        "extracted_data.case_specific_fields.case_type_pre_service": "",
        "extracted_data.case_specific_fields.case_type_post_service": "",
    },
    "Provider Dispute": {
        "extracted_data.service_type": "",
        "extracted_data.case_specific_fields.case_type_pre_service": "",
        "extracted_data.case_specific_fields.case_type_post_service": "",
        "extracted_data.case_specific_fields.case_type_grievance": "",
        "extracted_data.case_specific_fields.type_of_issue": "",
    },
    "Inquiry": {
        "extracted_data.service_type": "",
        "extracted_data.case_specific_fields.case_type_pre_service": "",
        "extracted_data.case_specific_fields.case_type_post_service": "",
        "extracted_data.case_specific_fields.case_type_grievance": "",
        "extracted_data.case_specific_fields.type_of_issue": "",
    },
}


# ─── JSON helpers ─────────────────────────────────────────────────────────────

def find_last_top_level_json(text: str) -> str | None:
    """Return the last balanced ``{...}`` in text that parses as JSON, or None."""
    last = None
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth, in_str, esc, end = 0, False, False, -1
        for j in range(i, n):
            c = text[j]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end == -1:
            break
        try:
            json.loads(text[i:end + 1])
            last = text[i:end + 1]
        except (json.JSONDecodeError, ValueError):
            pass
        i = end + 1
    return last


def _strip_jpath(path: str) -> str:
    if path.startswith("$."):
        return path[2:]
    if path.startswith("$"):
        return path[1:]
    return path


def jget(obj: dict, path: str):
    cur = obj
    for part in _strip_jpath(path).split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    return cur


def jset(obj: dict, path: str, value) -> None:
    """Set a dotted path in a nested dict, creating intermediate dicts as needed."""
    parts = _strip_jpath(path).split(".")
    cur = obj
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


# ─── Asserts parser ───────────────────────────────────────────────────────────

# Assert format (semicolon-separated, pipe-delimited within each):
#   name = "val" | json_path_exact @$.path | expected=val ; name2 = ...
ASSERT_SEP = " ; "


def parse_asserts(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not text:
        return out
    for chunk in text.split(ASSERT_SEP):
        parts = [p.strip() for p in chunk.split("|")]
        if len(parts) < 3:
            continue
        m_path = re.search(r"@\s*(\S+)", parts[1])
        m_exp = re.search(r"expected\s*=\s*(.*)$", parts[2])
        if m_path and m_exp:
            path = m_path.group(1).strip()
            expected = m_exp.group(1).strip().strip('"').strip("'")
            out.append((path, expected))
    return out


# ─── Backtest loader ──────────────────────────────────────────────────────────

def load_backtest(xlsx_path: Path) -> list[dict]:
    """Each row → {name, input, asserts, pred_obj}."""
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb["Results"]
    cases: list[dict] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        name, pii, asserts, output = row
        if name is None:
            continue
        pred_obj = None
        if output:
            block = find_last_top_level_json(str(output))
            if block:
                try:
                    pred_obj = json.loads(block)
                except json.JSONDecodeError:
                    pred_obj = None
        cases.append({
            "name": str(name),
            "input": str(pii or ""),
            "asserts": parse_asserts(str(asserts or "")),
            "pred_obj": pred_obj,
        })
    return cases


# ─── Case categorization ──────────────────────────────────────────────────────

def gold_class_of(case: dict) -> str | None:
    for path, expected in case["asserts"]:
        if "document_classification" in path:
            return expected
    return None


def all_asserts_pass(case: dict) -> bool:
    if case["pred_obj"] is None:
        return False
    for path, expected in case["asserts"]:
        actual = jget(case["pred_obj"], path)
        actual_s = "" if actual is None else str(actual)
        if actual_s.strip() != expected.strip():
            return False
    return True


# ─── Build a corrected assistant output for a failure case ────────────────────

def build_corrected_output(case: dict, gold_class: str,
                            donor_pred_per_class: dict[str, dict]) -> str:
    """Return a JSON string suitable as the assistant message.

    Order of operations:
      1. Deep-copy the case's own pred_obj as the base (so member info, dates,
         etc. survive). Fall back to a donor template of the same class if the
         case has no parseable pred_obj.
      2. Apply CLASS_DEFAULTS[gold_class] — sets/clears class-shape fields.
      3. Apply asserts — overrides anything the human verified.
    """
    if case["pred_obj"] is not None:
        result = json.loads(json.dumps(case["pred_obj"]))  # deep copy
    elif gold_class in donor_pred_per_class:
        result = json.loads(json.dumps(donor_pred_per_class[gold_class]))
    else:
        result = {"extracted_data": {}}

    for path, value in CLASS_DEFAULTS.get(gold_class, {}).items():
        jset(result, path, value)

    for path, expected in case["asserts"]:
        jset(result, path, expected)

    return json.dumps(result, ensure_ascii=False)


# ─── Variant emission ─────────────────────────────────────────────────────────

def emit_variants(fout, system: str, user: str, asst: str,
                  n_variants: int, ocr_rate: float,
                  rng_master: random.Random) -> int:
    """Write original + n_variants. Returns count actually written (variants
    whose substituted JSON fails to parse are dropped)."""
    base = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": asst},
        ]
    }
    fout.write(json.dumps(base, ensure_ascii=False) + "\n")
    n = 1
    for _ in range(n_variants):
        rng = random.Random(rng_master.randint(0, 2**31))
        subs = build_substitution_map(rng, user + "\n" + asst)
        new_user = apply_substitutions(user, subs)
        new_asst = apply_substitutions(asst, subs)
        if ocr_rate > 0:
            new_user = inject_ocr_noise(rng, new_user, rate=ocr_rate)
        try:
            json.loads(new_asst)
        except json.JSONDecodeError:
            continue
        rec = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": new_user},
                {"role": "assistant", "content": new_asst},
            ]
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
    return n


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--backtest", required=True, type=Path,
                   help="Backtest Excel (Results sheet)")
    p.add_argument("--short-prompt", required=True, type=Path,
                   help="System prompt text file")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--failure-class", default="Pre-Service Appeal",
                   help="Gold class we want to rescue (default: 'Pre-Service Appeal')")
    p.add_argument("--variants-per-failure", type=int, default=40,
                   help="Heavy augmentation for the boundary we're fixing")
    p.add_argument("--variants-per-anchor", type=int, default=15,
                   help="Lighter augmentation for correctly-classified anchors")
    p.add_argument("--anchors-per-class", type=int, default=6,
                   help="How many correctly-classified examples to keep PER class")
    p.add_argument("--ocr-noise-rate", type=float, default=0.003)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng_master = random.Random(args.seed)
    system_prompt = args.short_prompt.read_text().strip()

    cases = load_backtest(args.backtest)
    print(f"Loaded {len(cases)} backtest cases\n")

    failures: list[dict] = []
    anchors: dict[str, list[dict]] = defaultdict(list)
    donor_pred_per_class: dict[str, dict] = {}

    for c in cases:
        gold = gold_class_of(c)
        if gold is None:
            continue
        if all_asserts_pass(c):
            anchors[gold].append(c)
            donor_pred_per_class.setdefault(gold, c["pred_obj"])
        elif gold == args.failure_class:
            failures.append(c)

    print(f"Failure cases (gold={args.failure_class!r}, asserts failed): {len(failures)}")
    for c in failures:
        pred_class = jget(c["pred_obj"] or {}, "$.extracted_data.document_classification") \
                     if c["pred_obj"] else "<no_output>"
        print(f"  - {c['name'][:80]}  predicted={pred_class!r}")

    print("\nAnchor pool (correctly classified, full asserts pass):")
    for cls, lst in sorted(anchors.items()):
        marker = " ← failure class" if cls == args.failure_class else ""
        print(f"  {cls:<35}  {len(lst)} available{marker}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_by_class: dict[str, int] = defaultdict(int)

    with open(args.output, "w") as fout:
        # FAILURE CASES — heavy variants of corrected outputs
        for case in failures:
            corrected = build_corrected_output(
                case, args.failure_class, donor_pred_per_class,
            )
            n = emit_variants(
                fout, system_prompt, case["input"], corrected,
                args.variants_per_failure, args.ocr_noise_rate, rng_master,
            )
            n_written += n
            n_by_class[args.failure_class] += n

        # ANCHOR CASES — lighter variants of as-is outputs
        for cls, anchor_cases in sorted(anchors.items()):
            chosen = anchor_cases[: args.anchors_per_class]
            for case in chosen:
                asst = json.dumps(case["pred_obj"], ensure_ascii=False)
                n = emit_variants(
                    fout, system_prompt, case["input"], asst,
                    args.variants_per_anchor, args.ocr_noise_rate, rng_master,
                )
                n_written += n
                n_by_class[cls] += n

    # Summary
    print(f"\n=== Focused dataset built ===")
    print(f"  Output:                  {args.output}")
    print(f"  Total records written:   {n_written}")
    print(f"  Failure class:           {args.failure_class!r}")
    print(f"\n  Records per class:")
    for cls, n in sorted(n_by_class.items(), key=lambda kv: -kv[1]):
        marker = " ← failure (heavy)" if cls == args.failure_class else ""
        print(f"    {cls:<35}  {n:>5}{marker}")

    # Quick sanity: how many distinct seed cases contributed?
    n_failure_seeds = len(failures)
    n_anchor_seeds = sum(min(len(v), args.anchors_per_class) for v in anchors.values())
    print(f"\n  Distinct seeds used: {n_failure_seeds + n_anchor_seeds} "
          f"({n_failure_seeds} failure + {n_anchor_seeds} anchor)")


if __name__ == "__main__":
    main()
