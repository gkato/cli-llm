"""Evaluate a trained LoRA adapter against the seed dataset.

For each record in the eval set:
- Send the user message to the base model with the adapter loaded
- Extract the document_classification (and optionally other key fields) from
  the response
- Compare against the gold (assistant) output
- Report aggregate accuracy per assertion type and overall

Two ways to run inference:

A) "transformers" mode (default): loads base + adapter into memory, slow
   but self-contained. Good for one-off eval immediately after training.

B) "vllm" mode: assumes a vLLM server is already running with --enable-lora
   and the adapter loaded; just hits the API. Fast for repeated evals.

Usage (transformers mode):
    python -m ml.distill_eval \
        --eval-set data/datasets/distill_seed_clean.jsonl \
        --adapter data/adapters/v12-gemma4-distill-001 \
        --base    google/gemma-4-31B-it \
        --max-new-tokens 4096

Usage (vllm mode):
    python -m ml.distill_eval \
        --eval-set data/datasets/distill_seed_clean.jsonl \
        --vllm-url http://localhost:8000/v1 \
        --vllm-model google/gemma-4-31B-it \
        --vllm-adapter v12-gemma4-distill-001
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


def find_last_top_level_json(text: str) -> str | None:
    """Bracket-counting scanner; returns the last balanced {…} that parses."""
    last_valid = None
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
            last_valid = text[i:end + 1]
        except (json.JSONDecodeError, ValueError):
            pass
        i = end + 1
    return last_valid


def extract_clean_json(text: str) -> dict | None:
    """Strip <think>, fences, then parse the last balanced JSON object.

    Important: if a ``<think>`` tag is opened but never closed (model ran out
    of ``max_new_tokens`` mid-reasoning), we drop everything from that opening
    onward. Otherwise ``find_last_top_level_json`` would happily extract
    tentative JSON from inside the reasoning sketchpad and score it as the
    final answer — which is how earlier eval runs got fake "Provider Dispute"
    misclassifications.
    """
    if not text:
        return None
    # Strip COMPLETE <think>...</think> blocks first
    text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text)
    # If <think> still appears, it's unclosed — discard the tail entirely
    if "<think>" in text:
        text = text[:text.index("<think>")].rstrip()
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*", "", text).strip()

    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    block = find_last_top_level_json(text)
    if block:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            return None
    return None


def jpath(obj, path):
    """Tiny dotted-path resolver. '$.a.b' → obj['a']['b']."""
    if path.startswith("$."):
        path = path[2:]
    elif path.startswith("$"):
        path = path[1:]
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
        if cur is None:
            return None
    return cur


# Fields we always score
CORE_FIELDS = [
    "$.extracted_data.document_classification",
    "$.extracted_data.service_type",
    "$.extracted_data.case_specific_fields.case_type_pre_service",
    "$.extracted_data.case_specific_fields.case_type_post_service",
    "$.extracted_data.case_specific_fields.case_type_grievance",
    "$.extracted_data.case_specific_fields.type_of_issue",
    "$.extracted_data.manual_review",
]


def score(gold: dict, predicted: dict) -> dict:
    """Compare predicted vs gold on the core fields."""
    results = {}
    for path in CORE_FIELDS:
        g = jpath(gold, path)
        p = jpath(predicted, path)
        # Coerce booleans/None for fair compare
        gs = "" if g is None else str(g)
        ps = "" if p is None else str(p)
        results[path] = {
            "gold": gs,
            "pred": ps,
            "match": gs == ps,
        }
    return results


def _apply_chat_template(tokenizer, messages, add_generation_prompt=True, enable_thinking=None):
    """Wrapper around apply_chat_template that always returns a BatchEncoding
    with ``input_ids`` and ``attention_mask`` — transformers 5.x changed the
    return shape under ``return_tensors="pt"`` to a dict-like BatchEncoding,
    so we explicitly ask for the dict and rely on the keys.

    ``enable_thinking`` — Qwen3-specific knob. When False, the chat template
    pre-fills an empty ``<think></think>`` so the model skips reasoning and
    jumps straight to the JSON. Saves 3-5k tokens of output and ~5× wall time.
    Passed via kwargs only when set, since other tokenizers don't accept it.
    """
    kwargs = dict(
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=add_generation_prompt,
    )
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **kwargs)


_PREFILL_EMPTY_THINK = "<think>\n\n</think>\n\n"


def infer_transformers(model, tokenizer, system: str, user: str, max_new_tokens: int = 4096, max_input_tokens: int = 28000, enable_thinking: bool | None = None, prefill_empty_think: bool = False) -> str:
    """Greedy generate. Truncates the *user* message (in tokens) if the combined
    chat-template input would exceed ``max_input_tokens`` — keeps system + tail
    of the user message, since the tail is usually where the case data lives.
    """
    import torch
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    enc = _apply_chat_template(
        tokenizer, messages, add_generation_prompt=True, enable_thinking=enable_thinking,
    )
    seq_len = enc["input_ids"].shape[-1]
    if seq_len > max_input_tokens:
        # Truncate the *user* message from the left (keep the tail — case data
        # usually lives there). Build by: tokenize user alone, slice, decode,
        # re-template.
        sys_only = _apply_chat_template(
            tokenizer, [{"role": "system", "content": system}],
            add_generation_prompt=False,
        )
        sys_len = sys_only["input_ids"].shape[-1]
        user_budget = max(512, max_input_tokens - sys_len - 64)
        user_ids = tokenizer(user, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if user_ids.shape[0] > user_budget:
            user_ids = user_ids[-user_budget:]
            user = tokenizer.decode(user_ids, skip_special_tokens=True)
        enc = _apply_chat_template(
            tokenizer,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        seq_len = enc["input_ids"].shape[-1]

    input_ids = enc["input_ids"]
    if prefill_empty_think:
        # Append a pre-filled empty <think></think> block directly to the prompt
        # tokens. Qwen3 sees the reasoning block as already done and emits the
        # final answer immediately. Works even if the chat-template flag
        # `enable_thinking=False` is ignored by the loaded tokenizer.
        prefill_ids = tokenizer(_PREFILL_EMPTY_THINK, return_tensors="pt",
                                add_special_tokens=False).input_ids
        input_ids = torch.cat([input_ids, prefill_ids], dim=-1)
        seq_len = input_ids.shape[-1]

    input_ids = input_ids.to(model.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        if prefill_empty_think:
            prefill_mask = torch.ones(
                (attention_mask.shape[0], prefill_ids.shape[-1]),
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([attention_mask, prefill_mask], dim=-1)
        attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][seq_len:], skip_special_tokens=True)


def infer_vllm(client, model_name: str, system: str, user: str, max_new_tokens: int = 4096, adapter: str | None = None) -> str:
    extra_body = {}
    if adapter:
        extra_body["lora_request"] = adapter
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_new_tokens,
        temperature=0.0,
        extra_body=extra_body,
    )
    return resp.choices[0].message.content or ""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-set", required=True, type=Path)
    p.add_argument("--adapter", type=Path, default=None,
                   help="Adapter dir (transformers mode only)")
    p.add_argument("--base", default="google/gemma-4-31B-it")
    p.add_argument("--vllm-url", default=None,
                   help="If set, use vLLM API instead of loading model locally")
    p.add_argument("--vllm-model", default=None,
                   help="Model name vLLM is serving")
    p.add_argument("--vllm-adapter", default=None,
                   help="LoRA adapter name registered in vLLM")
    p.add_argument("--vllm-api-key", default=None)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--output", type=Path, default=Path("data/eval_results.jsonl"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--yarn-factor", type=float, default=None,
                   help="Apply YaRN rope scaling. Factor 3.0 extends Qwen3's "
                        "32k native context to ~98k; 4.0 to ~131k. Transformers "
                        "mode only.")
    p.add_argument("--yarn-orig-pos", type=int, default=32768,
                   help="Original max_position_embeddings before YaRN. Default "
                        "32768 (Qwen3 native).")
    p.add_argument("--max-input-tokens", type=int, default=28000,
                   help="Hard cap on chat-template input length (tokens). User "
                        "message tail is kept if it would exceed this. Bump "
                        "alongside --yarn-factor (e.g. 92000 for 96k context).")
    p.add_argument("--no-thinking", action="store_true",
                   help="Qwen3-only: disable <think> reasoning in the chat "
                        "template. Recommended for distilled adapters whose "
                        "training data was raw JSON (no <think> wrapping). "
                        "Cuts inference time ~5× and prevents truncation of "
                        "the final answer.")
    p.add_argument("--prefill-empty-think", action="store_true",
                   help="Append <think>\\n\\n</think>\\n\\n directly to the "
                        "prompt tokens after templating. Belt-and-suspenders "
                        "for when --no-thinking doesn't take effect — Qwen3 "
                        "sees reasoning as already done and emits JSON "
                        "immediately. Use TOGETHER with --no-thinking.")
    args = p.parse_args()

    # Load eval set
    records = []
    with open(args.eval_set) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.limit:
        records = records[:args.limit]
    print(f"Loaded {len(records)} eval records")

    # Set up inference
    if args.vllm_url:
        from openai import OpenAI
        client = OpenAI(base_url=args.vllm_url, api_key=args.vllm_api_key or "EMPTY")
        infer_fn = lambda sys_, user_: infer_vllm(
            client, args.vllm_model, sys_, user_,
            max_new_tokens=args.max_new_tokens, adapter=args.vllm_adapter,
        )
        print(f"Inference: vLLM @ {args.vllm_url}, model={args.vllm_model}, adapter={args.vllm_adapter}")
    else:
        import torch
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        print(f"Loading {args.base} in 4-bit + adapter from {args.adapter}")

        config = AutoConfig.from_pretrained(args.base, trust_remote_code=False)
        if args.yarn_factor:
            new_max_pos = int(args.yarn_orig_pos * args.yarn_factor)
            # Some transformers releases read base from config.rope_theta inside
            # _compute_yarn_parameters; if it's None at that point we crash with
            # `NoneType ** Tensor`. Snapshot rope_theta before we touch
            # rope_scaling so we can pin it back if the overwrite drops it.
            orig_rope_theta = getattr(config, "rope_theta", None)
            print(f"  pre-YaRN  rope_theta={orig_rope_theta}  "
                  f"rope_scaling={getattr(config, 'rope_scaling', None)}  "
                  f"head_dim={getattr(config, 'head_dim', None)}")
            config.rope_scaling = {
                "rope_type": "yarn",
                "factor": args.yarn_factor,
                "original_max_position_embeddings": args.yarn_orig_pos,
            }
            # Belt-and-suspenders: ensure rope_theta is still set.
            if orig_rope_theta is None:
                # Qwen3 default; safe fallback if config didn't expose it.
                orig_rope_theta = 1_000_000.0
            config.rope_theta = orig_rope_theta
            config.max_position_embeddings = new_max_pos
            print(f"YaRN: factor={args.yarn_factor} "
                  f"orig_max_pos={args.yarn_orig_pos} → "
                  f"extended max_pos={new_max_pos}  "
                  f"rope_theta={config.rope_theta}")

        tokenizer = AutoTokenizer.from_pretrained(args.base)
        # Match tokenizer's model_max_length to the (possibly-extended) context.
        # Gemma 4 is multimodal — position embeddings live on the nested text config.
        def _resolve_max_pos(cfg):
            for attr in ("max_position_embeddings",):
                if hasattr(cfg, attr):
                    return getattr(cfg, attr)
            for nested in ("text_config", "language_config"):
                if hasattr(cfg, nested):
                    sub = getattr(cfg, nested)
                    if hasattr(sub, "max_position_embeddings"):
                        return sub.max_position_embeddings
            # Fall through: use a sensible default rather than crashing.
            return 8192
        tokenizer.model_max_length = _resolve_max_pos(config)
        model = AutoModelForCausalLM.from_pretrained(
            args.base, config=config, quantization_config=bnb, device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        # Strip vision tower for Gemma 4 multimodal — unused for text-only eval,
        # frees ~2-3 GB VRAM (same trick used in training).
        if hasattr(model, "model") and hasattr(model.model, "vision_tower"):
            print("  stripping vision_tower + multi_modal_projector")
            model.model.vision_tower = torch.nn.Identity()
            if hasattr(model.model, "multi_modal_projector"):
                model.model.multi_modal_projector = torch.nn.Identity()
            import gc as _gc; _gc.collect(); torch.cuda.empty_cache()
            _free, _total = torch.cuda.mem_get_info()
            print(f"  GPU free after strip: {_free/1e9:.2f} GB / {_total/1e9:.2f} GB")
        if args.adapter:
            model = PeftModel.from_pretrained(model, str(args.adapter))
        model.eval()
        thinking = False if args.no_thinking else None  # None = template default
        infer_fn = lambda sys_, user_: infer_transformers(
            model, tokenizer, sys_, user_,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            enable_thinking=thinking,
            prefill_empty_think=args.prefill_empty_think,
        )

    # Run eval
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pass_count = 0
    field_correct = Counter()
    field_total = Counter()
    failures = []

    with open(args.output, "w") as fout:
        for i, rec in enumerate(records):
            sys_msg = rec["messages"][0]["content"]
            user_msg = rec["messages"][1]["content"]
            try:
                gold = json.loads(rec["messages"][2]["content"])
            except json.JSONDecodeError:
                continue

            t0 = time.monotonic()
            try:
                raw = infer_fn(sys_msg, user_msg)
            except Exception as e:
                import traceback
                err_repr = repr(e) or type(e).__name__
                print(f"  case {i + 1}: inference failed: {err_repr}", file=sys.stderr)
                # Print full traceback for the FIRST failure so we can diagnose;
                # subsequent failures just get the repr to keep logs short.
                if i == 0:
                    traceback.print_exc()
                fout.write(json.dumps({
                    "case_index": i, "passed": False,
                    "error": err_repr,
                }) + "\n")
                continue
            elapsed = time.monotonic() - t0

            predicted = extract_clean_json(raw)
            if predicted is None:
                print(f"  case {i + 1}: predicted output not parseable as JSON "
                      f"(raw_len={len(raw)})")
                # Save both ends so we can see truncation point — the tail tells
                # us whether the model ran out of tokens (cuts mid-field) or
                # produced something fundamentally unparseable.
                raw_save = raw if len(raw) <= 4000 else (
                    raw[:2000] + f"\n...[truncated {len(raw)-4000} chars]...\n" + raw[-2000:]
                )
                fout.write(json.dumps({
                    "case_index": i, "passed": False, "error": "no_json",
                    "raw_len": len(raw),
                    "raw_output": raw_save,
                }) + "\n")
                continue

            results = score(gold, predicted)
            all_match = all(r["match"] for r in results.values())
            for path, r in results.items():
                field_total[path] += 1
                if r["match"]:
                    field_correct[path] += 1

            if all_match:
                pass_count += 1
                status = "✓"
            else:
                fail_paths = [p for p, r in results.items() if not r["match"]]
                failures.append((i, fail_paths, results))
                status = "✗"

            print(f"  case {i + 1:>3}/{len(records)} [{status}] {elapsed:.1f}s")

            fout.write(json.dumps({
                "case_index": i,
                "passed": all_match,
                "elapsed_s": elapsed,
                "results": results,
            }) + "\n")

    print("\n=== Eval summary ===")
    print(f"  Cases:  {len(records)}")
    print(f"  Passed: {pass_count} ({100 * pass_count / len(records):.1f}%)")
    print(f"  Failed: {len(records) - pass_count}")
    print(f"\n  Per-field accuracy:")
    for path in CORE_FIELDS:
        if field_total[path]:
            acc = 100 * field_correct[path] / field_total[path]
            print(f"    {path:<60} {field_correct[path]:>3}/{field_total[path]:<3} ({acc:.1f}%)")

    if failures:
        print(f"\n  Failure samples (first 5):")
        for idx, paths, results in failures[:5]:
            print(f"\n    case {idx + 1}: failed on {len(paths)} field(s)")
            for path in paths:
                r = results[path]
                print(f"      {path}: pred={r['pred']!r}  gold={r['gold']!r}")


if __name__ == "__main__":
    main()
