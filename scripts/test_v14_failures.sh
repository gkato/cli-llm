#!/usr/bin/env bash
# Re-run the 8 cases that stock Gemma 4 31B failed on the V14 backtest,
# against the LoRA fine-tune using the SHORT prompt (the SFT's training
# distribution). If most flip to correct → SFT learned the rules; the
# earlier full-prompt failures are a context-mismatch issue.
#
# Usage:  bash scripts/test_v14_failures.sh
#
# Requires: vLLM serving gemma4-31b-awq-v14ft on localhost:8000
#           Run from the repo root.

set -euo pipefail

API_KEY="$(grep ^API_KEY= .env.local | cut -d= -f2-)"
ENDPOINT="http://localhost:8000/v1/chat/completions"
MODEL="gemma4-v14-001"
SHORT_PROMPT_FILE="data/prompts/V12_distill_short.txt"
CASES_FILE="data/gemma4_v14_failures_for_retest.jsonl"

if [[ ! -f "$SHORT_PROMPT_FILE" ]]; then
  echo "Missing $SHORT_PROMPT_FILE" >&2; exit 1
fi
if [[ ! -f "$CASES_FILE" ]]; then
  echo "Missing $CASES_FILE" >&2; exit 1
fi

python3 - "$API_KEY" "$ENDPOINT" "$MODEL" "$SHORT_PROMPT_FILE" "$CASES_FILE" <<'PY'
import sys, json, urllib.request, urllib.error
api_key, endpoint, model, sys_prompt_path, cases_path = sys.argv[1:]

system = open(sys_prompt_path).read().strip()
print(f"System prompt: {len(system)} chars ({sys_prompt_path})")
print(f"Endpoint:      {endpoint}")
print(f"Model:         {model}")
print("=" * 78)

def classify_case(user_content):
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": 1200,
    }).encode()
    req = urllib.request.Request(
        endpoint, data=body,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=180).read())
    except urllib.error.HTTPError as e:
        return None, f"HTTPError {e.code}: {e.read()[:200]}"
    text = resp["choices"][0]["message"]["content"].strip()
    clean = text
    if clean.startswith("```"):
        clean = clean.split("```", 2)[1]
        if clean.startswith("json"): clean = clean[4:]
        clean = clean.rsplit("```", 1)[0].strip()
    try:
        obj = json.loads(clean)
    except Exception as e:
        return None, f"JSON parse fail: {e}; raw={text[:300]!r}"
    return obj, None

pass_count = 0
fail_count = 0
err_count = 0
for i, line in enumerate(open(cases_path), 1):
    case = json.loads(line)
    name = case["case_name"]
    print(f"\n[{i}/8] {name}")
    expected = dict(case["expected_failures"])  # field -> expected value
    obj, err = classify_case(case["input_text"])
    if err:
        print(f"   ✗ ERROR: {err}")
        err_count += 1
        continue
    extracted = obj.get("extracted_data", {})
    cs = extracted.get("case_specific_fields", {}) or {}
    all_ok = True
    for field, exp_val in expected.items():
        got = extracted.get(field) or cs.get(field) or ""
        ok = str(got).strip().lower() == str(exp_val).strip().lower()
        marker = "✓" if ok else "✗"
        all_ok &= ok
        print(f"   {marker} {field}: expected={exp_val!r}  got={got!r}")
    if all_ok: pass_count += 1
    else:      fail_count += 1

print()
print("=" * 78)
print(f"PASS: {pass_count}/8   FAIL: {fail_count}/8   ERROR: {err_count}/8")
PY
