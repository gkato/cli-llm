#!/usr/bin/env python3
"""Benchmark the DSv4 server without initializing a host-side tokenizer.

DeepSeek V4 Flash uses the runtime's custom DSv4 encoder and does not expose a
normal Hugging Face ``tokenizer.model``.  This runner therefore sends unique
text prompts through the public, authenticated API and uses vLLM's streamed
``usage`` counters for the actual input and output token counts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_UNIT = "benchmark context datum "


def parse_positive_ints(value: str) -> list[int]:
    """Parse comma/space-separated positive integers, preserving order."""
    parts = value.replace(",", " ").split()
    if not parts:
        raise argparse.ArgumentTypeError("at least one concurrency is required")
    try:
        numbers = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "concurrencies must be comma/space-separated integers"
        ) from exc
    if any(number < 1 for number in numbers):
        raise argparse.ArgumentTypeError("concurrencies must be positive")
    return numbers


def percentile(values: list[float], percent: float) -> float | None:
    """Return a linearly interpolated percentile without NumPy."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def normalize_api_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    return normalized if normalized.endswith("/v1") else normalized + "/v1"


def build_prompt(target_tokens: int, nonce: str, output_tokens: int) -> str:
    """Build a cold-prefix prompt close to the requested DSv4 token count.

    MiaAI-Lab's DSpark benchmark starts with target/3 repetitions of this same
    prompt unit and then verifies it with the private ``/tokenize`` route.  The
    safety proxy deliberately does not expose that route, so this benchmark
    keeps the generator and records the authoritative prompt count returned by
    vLLM with each completion instead.
    """
    repetitions = max(1, target_tokens // 3)
    return (
        f"unique benchmark request {nonce} "
        + PROMPT_UNIT * repetitions
        + f"\nReturn exactly {output_tokens} numbered lowercase English words."
    )


def load_api_key() -> str:
    from dotenv import dotenv_values

    key = os.environ.get("OPEN_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        key = dotenv_values(PROJECT_ROOT / ".env.local").get("API_KEY")
    if not key:
        raise RuntimeError(
            "Set OPEN_API_KEY (or OPENAI_API_KEY), or add API_KEY to .env.local."
        )
    return str(key)


def stream_request(
    *,
    endpoint: str,
    headers: dict[str, str],
    model: str,
    prompt: str,
    output_tokens: int,
    timeout: float,
    request_number: int,
) -> dict[str, Any]:
    import requests

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "ignore_eos": True,
        "chat_template_kwargs": {"thinking": False},
    }
    started = time.perf_counter()
    first_token_at: float | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None

    with requests.post(
        endpoint,
        headers=headers,
        json=body,
        stream=True,
        timeout=(10, timeout),
    ) as response:
        if not response.ok:
            detail = response.text[:500].replace("\n", " ")
            raise RuntimeError(f"HTTP {response.status_code}: {detail}")
        # requests defaults to 512-byte chunks, which can materially distort
        # TTFT for SSE.  One-byte chunks let it yield complete lines promptly.
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                continue
            event = json.loads(payload)
            choices = event.get("choices") or []
            if choices:
                choice = choices[0]
                delta = choice.get("delta") or {}
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                emitted = (
                    delta.get("content")
                    or delta.get("reasoning")
                    or delta.get("reasoning_content")
                )
                if emitted and first_token_at is None:
                    first_token_at = time.perf_counter()
            if event.get("usage"):
                usage = event["usage"]

    finished = time.perf_counter()
    if not usage:
        raise RuntimeError("stream ended without a usage record")
    prompt_count = int(usage.get("prompt_tokens") or 0)
    output_count = int(usage.get("completion_tokens") or 0)
    if prompt_count < 1 or output_count < 1:
        raise RuntimeError(f"invalid streamed usage counters: {usage!r}")

    first = first_token_at or finished
    ttft_seconds = first - started
    decode_seconds = max(finished - first, 0.001)
    e2e_seconds = finished - started
    tpot_seconds = decode_seconds / max(output_count - 1, 1)
    return {
        "request_number": request_number,
        "prompt_tokens": prompt_count,
        "output_tokens": output_count,
        "ttft_ms": ttft_seconds * 1000,
        "tpot_ms": tpot_seconds * 1000,
        "e2el_ms": e2e_seconds * 1000,
        # Keep this compatible with the MiaAI-Lab cell definition.
        "per_stream_decode_tok_s": output_count / decode_seconds,
        "finish_reason": finish_reason,
    }


def run_one(
    *,
    endpoint: str,
    headers: dict[str, str],
    model: str,
    input_tokens: int,
    output_tokens: int,
    timeout: float,
    seed: int,
    concurrency: int,
    request_number: int,
    warmup: bool,
) -> dict[str, Any]:
    random_suffix = random.Random(seed + request_number).getrandbits(64)
    kind = "warmup" if warmup else "measure"
    nonce = (
        f"{kind}-c{concurrency}-r{request_number}-{random_suffix:x}-"
        f"{uuid.uuid4().hex}"
    )
    return stream_request(
        endpoint=endpoint,
        headers=headers,
        model=model,
        prompt=build_prompt(input_tokens, nonce, output_tokens),
        output_tokens=output_tokens,
        timeout=timeout,
        request_number=request_number,
    )


def add_percentiles(report: dict[str, Any], records: list[dict[str, Any]]) -> None:
    for source, suffix in (
        ("ttft_ms", "ttft_ms"),
        ("tpot_ms", "tpot_ms"),
        ("e2el_ms", "e2el_ms"),
    ):
        values = [float(record[source]) for record in records]
        for percent in (50, 90, 95, 99):
            report[f"p{percent}_{suffix}"] = percentile(values, percent)


def build_report(
    *,
    model: str,
    base_url: str,
    profile: str,
    concurrency: int,
    input_tokens: int,
    output_tokens: int,
    requested: int,
    elapsed: float,
    records: list[dict[str, Any]],
    failures: list[str],
) -> dict[str, Any]:
    actual_inputs = [record["prompt_tokens"] for record in records]
    actual_outputs = [record["output_tokens"] for record in records]
    decode_rates = [record["per_stream_decode_tok_s"] for record in records]
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model,
        "base_url": base_url,
        "profile": profile,
        "methodology": "dspark-streaming-server-usage-cold-prefix",
        "max_concurrency": concurrency,
        "target_input_tokens": input_tokens,
        "target_output_tokens": output_tokens,
        "num_prompts": requested,
        "completed": len(records),
        "failed": len(failures),
        "duration": elapsed,
        "request_throughput": len(records) / max(elapsed, 0.001),
        "input_throughput": sum(actual_inputs) / max(elapsed, 0.001),
        "output_throughput": sum(actual_outputs) / max(elapsed, 0.001),
        "mean_input_tokens": statistics.fmean(actual_inputs) if records else None,
        "mean_output_tokens": statistics.fmean(actual_outputs) if records else None,
        "median_per_stream_decode_tok_s": (
            statistics.median(decode_rates) if records else None
        ),
        "mean_tpot_ms": (
            statistics.fmean(record["tpot_ms"] for record in records)
            if records
            else None
        ),
        "requests": sorted(records, key=lambda item: item["request_number"]),
        "failures": failures,
    }
    add_percentiles(report, records)
    return report


def print_report(report: dict[str, Any], result_file: Path) -> None:
    print(
        f"  C{report['max_concurrency']} complete: "
        f"aggregate={report['output_throughput']:.2f} output tok/s, "
        f"median stream={report['median_per_stream_decode_tok_s']:.2f} tok/s, "
        f"P50 TTFT={report['p50_ttft_ms']:.0f} ms, "
        f"actual input={report['mean_input_tokens']:.0f} tokens",
        flush=True,
    )
    print(f"  Result: {result_file}", flush=True)


def display(value: Any, width: int, precision: int = 2) -> str:
    if value is None:
        return f"{'-':>{width}}"
    return f"{float(value):>{width}.{precision}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--profile", default="standard")
    parser.add_argument("--input-len", type=int, default=2048)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--num-warmups", type=int, default=2)
    parser.add_argument("--concurrencies", type=parse_positive_ints, default=[1, 2, 4])
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.input_len < 1 or args.output_len < 1 or args.num_prompts < 1:
        parser.error("input/output lengths and num-prompts must be positive")
    if args.num_warmups < 0 or args.timeout <= 0:
        parser.error("num-warmups must be non-negative and timeout must be positive")
    return args


def main() -> int:
    args = parse_args()
    import requests

    api_url = normalize_api_url(args.base_url)
    endpoint = api_url + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {load_api_key()}",
        "Content-Type": "application/json",
    }
    health = requests.get(api_url + "/models", headers=headers, timeout=30)
    if not health.ok:
        print(
            f"Could not authenticate to the DSpark API at {api_url}: "
            f"HTTP {health.status_code}",
            file=sys.stderr,
        )
        return 1

    args.result_dir.mkdir(parents=True, exist_ok=True)
    model_slug = args.model.replace("/", "-")
    reports: list[dict[str, Any]] = []
    any_failed = False

    print("DSpark tokenizer-independent streaming benchmark", flush=True)
    print(f"  Server: {api_url}", flush=True)
    print(f"  Model:  {args.model}", flush=True)
    print(
        f"  Shape:  ~{args.input_len} input / {args.output_len} output tokens; "
        f"C={args.concurrencies}",
        flush=True,
    )

    for concurrency in args.concurrencies:
        print(
            f"\nC{concurrency}: {args.num_prompts} measured requests "
            f"(+ {args.num_warmups} warmups)",
            flush=True,
        )
        for warmup_number in range(1, args.num_warmups + 1):
            started = time.perf_counter()
            try:
                record = run_one(
                    endpoint=endpoint,
                    headers=headers,
                    model=args.model,
                    input_tokens=args.input_len,
                    output_tokens=args.output_len,
                    timeout=args.timeout,
                    seed=args.seed,
                    concurrency=concurrency,
                    request_number=-warmup_number,
                    warmup=True,
                )
                print(
                    f"  warmup {warmup_number}/{args.num_warmups}: "
                    f"{record['prompt_tokens']} in, {record['output_tokens']} out, "
                    f"{time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - record remote failures
                print(f"  warmup failed: {exc}", file=sys.stderr, flush=True)
                return 1

        records: list[dict[str, Any]] = []
        failures: list[str] = []
        case_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    run_one,
                    endpoint=endpoint,
                    headers=headers,
                    model=args.model,
                    input_tokens=args.input_len,
                    output_tokens=args.output_len,
                    timeout=args.timeout,
                    seed=args.seed,
                    concurrency=concurrency,
                    request_number=request_number,
                    warmup=False,
                ): request_number
                for request_number in range(1, args.num_prompts + 1)
            }
            finished_count = 0
            for future in as_completed(futures):
                request_number = futures[future]
                finished_count += 1
                try:
                    record = future.result()
                    records.append(record)
                    print(
                        f"  request {finished_count}/{args.num_prompts} "
                        f"(r{request_number}): {record['prompt_tokens']} in, "
                        f"{record['output_tokens']} out, "
                        f"TTFT {record['ttft_ms']:.0f} ms, "
                        f"decode {record['per_stream_decode_tok_s']:.1f} tok/s",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 - preserve benchmark
                    message = f"request {request_number}: {exc}"
                    failures.append(message)
                    print(f"  FAILED {message}", file=sys.stderr, flush=True)
        elapsed = time.perf_counter() - case_started
        report = build_report(
            model=args.model,
            base_url=api_url,
            profile=args.profile,
            concurrency=concurrency,
            input_tokens=args.input_len,
            output_tokens=args.output_len,
            requested=args.num_prompts,
            elapsed=elapsed,
            records=records,
            failures=failures,
        )
        filename = (
            f"{model_slug}-{args.profile}-c{concurrency}-{args.input_len}in-"
            f"{args.output_len}out-{args.run_id}.json"
        )
        result_file = args.result_dir / filename
        result_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        reports.append(report)
        if records:
            print_report(report, result_file)
        if failures or len(records) != args.num_prompts:
            any_failed = True

    print("\nComparison", flush=True)
    print(
        f"{'Conc.':>5}  {'Req/s':>8}  {'Agg tok/s':>10}  {'Stream tok/s':>12}  "
        f"{'Actual in':>10}  {'P50 TTFT':>10}  {'P99 TTFT':>10}  "
        f"{'Mean TPOT':>10}  {'Failed':>6}",
        flush=True,
    )
    for report in reports:
        print(
            f"{report['max_concurrency']:>5}  "
            f"{display(report['request_throughput'], 8)}  "
            f"{display(report['output_throughput'], 10)}  "
            f"{display(report['median_per_stream_decode_tok_s'], 12)}  "
            f"{display(report['mean_input_tokens'], 10, 0)}  "
            f"{display(report['p50_ttft_ms'], 10, 0)}  "
            f"{display(report['p99_ttft_ms'], 10, 0)}  "
            f"{display(report['mean_tpot_ms'], 10)}  "
            f"{report['failed']:>6}",
            flush=True,
        )
    print(f"\nJSON results: {args.result_dir}", flush=True)
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
