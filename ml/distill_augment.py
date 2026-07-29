"""Deterministic algorithmic augmentation for the SFT seed dataset.

Each seed (input, output JSON) becomes N variants by:

1. Replacing PII placeholders (names, dates, IDs, addresses, phones) with new
   plausible-but-different values, applied IDENTICALLY to both the user input
   AND the assistant's JSON output. This preserves classification while
   creating fresh surface forms.

2. Optionally injecting mild OCR-noise patterns (random char drops, common
   homoglyph swaps, extra whitespace) that mimic what real fax/PDF inputs
   look like.

The failure-prone classes (Pre-Service Appeal, Provider Dispute) get more
variants → concentrated gradient pressure on the patterns Gemma 30B
struggles with.

Why algorithmic instead of LLM paraphrasing:
- Free, fast (~1 sec per variant)
- Deterministic, reproducible
- Zero risk of classification drift
- Surface diversity is what helps the model — semantic rewording doesn't
  add much when the structured label space is small (5 classes).

Usage:
    python -m ml.distill_augment \
        --input  data/datasets/distill_seed_clean.jsonl \
        --output data/datasets/distill_train.jsonl \
        --pass-variants 4 \
        --fail-variants 12

The script also writes a duplicate of each original (so train.jsonl
contains: 1 original + N variants per seed).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import string
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────
# PII inventories — pools to draw from when randomizing
# ────────────────────────────────────────────────────────────────────────────

FIRST_NAMES = [
    "Aaron", "Abigail", "Ahmed", "Alan", "Alicia", "Amelia", "Andre", "Angela",
    "Anthony", "Aria", "Asha", "Aubrey", "Barbara", "Beatrice", "Brandon",
    "Brenda", "Bryan", "Camila", "Carlos", "Caroline", "Celia", "Charles",
    "Chen", "Chloe", "Christine", "Damian", "Daniel", "Deborah", "Diana",
    "Diego", "Donald", "Dorothy", "Edward", "Elena", "Eli", "Elizabeth", "Eric",
    "Esther", "Evelyn", "Fatima", "Felix", "Frances", "Gabriel", "George",
    "Grace", "Hannah", "Harold", "Helen", "Henry", "Hiroshi", "Ingrid", "Isaac",
    "Isabella", "Jacob", "James", "Jasmine", "Jennifer", "John", "Joseph",
    "Joyce", "Julia", "Kareem", "Karen", "Katherine", "Kenneth", "Kevin",
    "Kira", "Lakshmi", "Linda", "Liu", "Lucas", "Maria", "Mark", "Mary",
    "Matthew", "Maya", "Michael", "Michelle", "Miguel", "Naomi", "Nathaniel",
    "Nicholas", "Norma", "Oliver", "Olivia", "Omar", "Patrick", "Paul",
    "Priya", "Quinn", "Rachel", "Ramon", "Raymond", "Rebecca", "Richard",
    "Robert", "Roberto", "Roxanne", "Ruby", "Samuel", "Sarah", "Sebastian",
    "Sofia", "Stephen", "Susan", "Tariq", "Thelma", "Thomas", "Tyrone",
    "Valeria", "Victor", "Walter", "Wendy", "William", "Yolanda", "Zara",
]

LAST_NAMES = [
    "Anderson", "Bailey", "Becker", "Bennett", "Brooks", "Bryant", "Cabrera",
    "Cardenas", "Carter", "Castillo", "Chen", "Cole", "Collins", "Cooper",
    "Cox", "Cruz", "Davis", "Diaz", "Edwards", "Espinoza", "Evans", "Faulkner",
    "Fischer", "Foster", "Garcia", "Gomez", "Gonzalez", "Graham", "Greene",
    "Hamlin", "Hanson", "Harper", "Hayes", "Hernandez", "Hoffman", "Holt",
    "Hsu", "Ibarra", "Jackson", "Jenkins", "Jimenez", "Johnson", "Kim",
    "Kowalski", "Lara", "Lee", "Lewis", "Lin", "Lopez", "Madsen", "Marshall",
    "Martin", "Martinez", "Matthews", "Meyer", "Mitchell", "Morales", "Moreno",
    "Morgan", "Murphy", "Nelson", "Nguyen", "Norris", "Okafor", "O'Brien",
    "Oliveira", "Ortega", "Owens", "Patel", "Perez", "Peterson", "Phelps",
    "Phillips", "Powell", "Quintana", "Ramirez", "Reyes", "Rivera", "Rogers",
    "Rojas", "Romano", "Sanchez", "Sato", "Schaefer", "Singh", "Smith",
    "Stewart", "Suarez", "Sullivan", "Tanaka", "Theriot", "Thompson", "Torres",
    "Vargas", "Vega", "Wagner", "Walker", "Wang", "Washington", "Williams",
    "Wong", "Yamamoto", "Young", "Zhang",
]

STREETS = [
    "Maple St", "Oak Ave", "Cedar Ln", "Pine Rd", "Birch Way", "Elm Dr",
    "Walnut Ct", "Cypress Blvd", "Magnolia Pl", "Hickory Hill", "Linden Pkwy",
    "Sycamore Trl", "Aspen Cir", "Willow Ridge", "Sequoia Pass",
]

CITIES = [
    "Springfield, IL 62701", "Madison, WI 53703", "Boulder, CO 80301",
    "Asheville, NC 28801", "Burlington, VT 05401", "Tacoma, WA 98402",
    "Fort Wayne, IN 46802", "Mesa, AZ 85201", "Tulsa, OK 74103",
    "Albany, NY 12207", "Tallahassee, FL 32301", "Cheyenne, WY 82001",
    "Bismarck, ND 58501", "Augusta, ME 04330", "Lansing, MI 48901",
]


# ────────────────────────────────────────────────────────────────────────────
# Token generators
# ────────────────────────────────────────────────────────────────────────────

def gen_first_name(rng: random.Random) -> str:
    return rng.choice(FIRST_NAMES)


def gen_last_name(rng: random.Random) -> str:
    return rng.choice(LAST_NAMES)


def gen_full_name(rng: random.Random) -> str:
    return f"{gen_first_name(rng)} {gen_last_name(rng)}"


def gen_phone(rng: random.Random) -> str:
    return f"({rng.randint(200, 999)}) {rng.randint(200, 999)}-{rng.randint(1000, 9999)}"


def gen_member_id(rng: random.Random) -> str:
    prefix = rng.choice(["INV", "MID", "MBR", "PLN", "SBI"])
    return f"{prefix}{rng.randint(1000000, 9999999):07d}"


def gen_npi(rng: random.Random) -> str:
    # 10-digit NPI
    return str(rng.randint(1000000000, 9999999999))


def gen_account_number(rng: random.Random) -> str:
    return "".join(rng.choices(string.digits, k=rng.randint(8, 12)))


def gen_address(rng: random.Random) -> str:
    return f"{rng.randint(100, 9999)} {rng.choice(STREETS)}, {rng.choice(CITIES)}"


def gen_email(rng: random.Random) -> str:
    domain = rng.choice(["example.com", "mail.com", "inbox.org", "fastmail.net"])
    return f"{gen_first_name(rng).lower()}.{gen_last_name(rng).lower()}@{domain}"


def shift_date(date_str: str, days: int) -> str:
    """Shift any MM-DD-YYYY or MM/DD/YYYY date by N days. Return original on parse fail."""
    for sep in ("-", "/"):
        m = re.match(rf"(\d{{1,2}}){sep}(\d{{1,2}}){sep}(\d{{4}})", date_str)
        if m:
            try:
                d = datetime(int(m.group(3)), int(m.group(1)), int(m.group(2)))
                shifted = d + timedelta(days=days)
                return shifted.strftime(f"%m{sep}%d{sep}%Y")
            except ValueError:
                return date_str
    return date_str


# ────────────────────────────────────────────────────────────────────────────
# Augmentation: build a substitution map and apply it to BOTH input and output
# ────────────────────────────────────────────────────────────────────────────

PLACEHOLDER_PATTERNS = [
    # Common Inovaare OCR placeholders we observed in the export
    (re.compile(r"\[PRIVATE_PERSON\]"), "person"),
    (re.compile(r"\[PRIVATE_DATE\]"), "date"),
    (re.compile(r"\[PRIVATE_PHONE\]"), "phone"),
    (re.compile(r"\[PRIVATE_EMAIL\]"), "email"),
    (re.compile(r"\[PRIVATE_ADDRESS\]"), "address"),
    (re.compile(r"\[ACCOUNT_NUMBER\]"), "account"),
]


def build_substitution_map(rng: random.Random, text: str) -> dict[str, str]:
    """Return a dict mapping each unique placeholder occurrence to a stable
    replacement value, so the SAME placeholder gets the SAME substitute
    everywhere it appears in this variant.
    """
    subs: dict[str, str] = {}
    seen: dict[str, int] = {}
    for pattern, kind in PLACEHOLDER_PATTERNS:
        for match in pattern.finditer(text):
            tok = match.group(0)
            # Use occurrence-stable key (same text → same value within variant)
            if tok in subs:
                continue
            if kind == "person":
                subs[tok] = gen_full_name(rng)
            elif kind == "date":
                # Random date in 2024-2025
                year = rng.choice([2024, 2025])
                month = rng.randint(1, 12)
                day = rng.randint(1, 28)
                subs[tok] = f"{month:02d}-{day:02d}-{year}"
            elif kind == "phone":
                subs[tok] = gen_phone(rng)
            elif kind == "email":
                subs[tok] = gen_email(rng)
            elif kind == "address":
                subs[tok] = gen_address(rng)
            elif kind == "account":
                subs[tok] = gen_account_number(rng)
    return subs


def apply_substitutions(text: str, subs: dict[str, str]) -> str:
    out = text
    for src, dst in subs.items():
        out = out.replace(src, dst)
    return out


# ────────────────────────────────────────────────────────────────────────────
# Optional mild OCR noise (controlled, low rate)
# ────────────────────────────────────────────────────────────────────────────

def inject_ocr_noise(rng: random.Random, text: str, rate: float = 0.005) -> str:
    """Randomly drop or swap a small fraction of characters to mimic OCR errors.

    Only touches alphanumeric chars; leaves whitespace and punctuation intact
    to preserve formatting. Default rate 0.5% is barely noticeable but adds
    real surface diversity.
    """
    if rate <= 0:
        return text
    out = []
    for c in text:
        r = rng.random()
        if c.isalnum() and r < rate:
            choice = rng.choice(["drop", "swap", "double"])
            if choice == "drop":
                continue
            elif choice == "swap":
                # Replace with a similar-looking char (homoglyph)
                homoglyphs = {
                    "0": "O", "O": "0", "1": "l", "l": "1", "I": "1",
                    "5": "S", "S": "5", "8": "B", "B": "8", "G": "6",
                    "rn": "m", "m": "rn",
                }
                out.append(homoglyphs.get(c, c))
            elif choice == "double":
                out.append(c)
                out.append(c)
        else:
            out.append(c)
    return "".join(out)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

# Classes Gemma 30B failed on most often — amplify their training signal
HIGH_PRIORITY_CLASSES = {"Pre-Service Appeal", "Provider Dispute"}


def get_classification(record: dict) -> str | None:
    try:
        out = json.loads(record["messages"][2]["content"])
        return out["extracted_data"]["document_classification"]
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--pass-variants", type=int, default=4,
                   help="Variants per non-priority case (default 4)")
    p.add_argument("--fail-variants", type=int, default=12,
                   help="Variants per Pre-Service Appeal / Provider Dispute case (default 12)")
    p.add_argument("--ocr-noise-rate", type=float, default=0.003,
                   help="Per-char OCR noise rate (default 0.003 = 0.3%%)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng_master = random.Random(args.seed)

    # Load seeds
    seeds: list[dict] = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    print(f"Loaded {len(seeds)} cleaned seeds")

    # Stats
    n_kept = 0
    by_class = {}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fout:
        for idx, seed in enumerate(seeds):
            cls = get_classification(seed) or "(unknown)"
            n_variants = args.fail_variants if cls in HIGH_PRIORITY_CLASSES else args.pass_variants

            # Always include the original
            fout.write(json.dumps(seed, ensure_ascii=False) + "\n")
            n_kept += 1
            by_class.setdefault(cls, [0, 0])
            by_class[cls][0] += 1

            # Generate variants
            user_text = seed["messages"][1]["content"]
            asst_text = seed["messages"][2]["content"]

            for v in range(n_variants):
                rng = random.Random(rng_master.randint(0, 2**31))

                # Build substitution map from input (covers all placeholders we see)
                subs = build_substitution_map(rng, user_text + "\n" + asst_text)

                # Apply substitutions to BOTH user and assistant
                new_user = apply_substitutions(user_text, subs)
                new_asst = apply_substitutions(asst_text, subs)

                # Optional OCR noise on the user side only
                if args.ocr_noise_rate > 0:
                    new_user = inject_ocr_noise(rng, new_user, rate=args.ocr_noise_rate)

                # Validate the assistant output is still parseable JSON
                try:
                    json.loads(new_asst)
                except json.JSONDecodeError:
                    # Substitution might have broken JSON syntax; skip this variant
                    continue

                new_rec = {
                    "messages": [
                        seed["messages"][0],  # system unchanged
                        {"role": "user", "content": new_user},
                        {"role": "assistant", "content": new_asst},
                    ]
                }
                fout.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                n_kept += 1
                by_class[cls][1] += 1

    # Summary
    print(f"\n=== Augmentation summary ===")
    print(f"  total records written: {n_kept}")
    print(f"  output: {args.output}\n")
    print(f"  By class:")
    for cls, (n_orig, n_var) in sorted(by_class.items(), key=lambda x: -x[1][0] - x[1][1]):
        priority = " [HIGH-PRIORITY]" if cls in HIGH_PRIORITY_CLASSES else ""
        print(f"    {cls:<35} originals={n_orig}  variants={n_var}  total={n_orig + n_var}{priority}")


if __name__ == "__main__":
    main()
