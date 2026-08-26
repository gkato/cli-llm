"""CLI and orchestration for a fair two-endpoint coding-model comparison."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .client import Completion, Endpoint, complete
from .execution import DockerExecutor, ExecutionInfrastructureError
from .report import DIMENSIONS, render_html
from .scoring import (
    assemble_humaneval_program,
    parse_prompt_tool_calls,
    score_bfcl,
    score_cruxeval,
    score_nl2bash,
    score_rubric,
)
from .suites import (
    BFCL_REVISION,
    CRUXEVAL_REVISION,
    HUMANEVAL_REVISION,
    NL2BASH_REVISION,
    Case,
    build_suite,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "model-compare.yaml"
DEFAULT_CUSTOM_SUITE = PROJECT_ROOT / "registry" / "model_compare_custom.yaml"
DEFAULT_EVALUATOR_IMAGE = "ml-compute-model-compare-eval:py312-numpy2.3.2"
DEFAULT_EVALUATOR_DOCKERFILE = PROJECT_ROOT / "docker" / "model-compare-eval" / "Dockerfile"
PROTECTED_REQUEST_FIELDS = {
    "model",
    "messages",
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "stream",
    "stream_options",
    "tools",
    "tool_choice",
}


class ConfigurationError(RuntimeError):
    pass


def load_config(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except FileNotFoundError as exc:
        raise ConfigurationError(
            f"configuration not found: {path}. Copy config/model-compare.example.yaml first."
        ) from exc
    models = config.get("models") or []
    if len(models) != 2:
        raise ConfigurationError("configuration must contain exactly two models")
    for index, model in enumerate(models, 1):
        missing = [
            key for key in ("label", "base_url", "model_id", "api_key_env") if not model.get(key)
        ]
        if missing:
            raise ConfigurationError(f"model {index} is missing: {', '.join(missing)}")
    benchmark = config.setdefault("benchmark", {})
    benchmark.setdefault("samples_per_dimension", 5)
    benchmark.setdefault("seed", 42)
    benchmark.setdefault("tool_mode", "prompt")
    benchmark.setdefault("timeout_seconds", 900)
    benchmark.setdefault("retries", 2)
    benchmark.setdefault("warmup_requests", 1)
    benchmark.setdefault("max_tokens_per_request", 16384)
    benchmark.setdefault("docker_image", DEFAULT_EVALUATOR_IMAGE)
    benchmark.setdefault("coding_timeout_seconds", 30)
    if int(benchmark["samples_per_dimension"]) < 1:
        raise ConfigurationError("samples_per_dimension must be positive")
    if float(benchmark["timeout_seconds"]) <= 0 or float(benchmark["coding_timeout_seconds"]) <= 0:
        raise ConfigurationError("timeouts must be positive")
    if int(benchmark["retries"]) < 0 or int(benchmark["warmup_requests"]) < 0:
        raise ConfigurationError("retries and warmup_requests must be non-negative")
    if (
        benchmark["max_tokens_per_request"] is not None
        and int(benchmark["max_tokens_per_request"]) < 1
    ):
        raise ConfigurationError("max_tokens_per_request must be positive or null")
    config.setdefault("title", "Coding model comparison")
    config.setdefault("output_dir", "data/model-comparisons")
    return config


def endpoints_from_config(config: dict[str, Any]) -> list[Endpoint]:
    from dotenv import dotenv_values

    local_secrets = dotenv_values(PROJECT_ROOT / ".env.local")
    endpoints = []
    for model in config["models"]:
        environment_name = str(model["api_key_env"])
        api_key = os.environ.get(environment_name) or local_secrets.get(environment_name)
        if not api_key:
            raise ConfigurationError(
                f"{environment_name} is not set in the environment or .env.local"
            )
        extra_headers = model.get("extra_headers") or {}
        if not isinstance(extra_headers, dict):
            raise ConfigurationError(f"extra_headers for {model['label']} must be a mapping")
        request_body = model.get("request_body") or {}
        if not isinstance(request_body, dict):
            raise ConfigurationError(f"request_body for {model['label']} must be a mapping")
        conflicts = PROTECTED_REQUEST_FIELDS & set(request_body)
        if conflicts:
            raise ConfigurationError(
                f"request_body for {model['label']} cannot override: "
                + ", ".join(sorted(conflicts))
            )
        try:
            json.dumps(request_body)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"request_body for {model['label']} must contain JSON-compatible values"
            ) from exc
        endpoints.append(
            Endpoint(
                label=str(model["label"]),
                base_url=str(model["base_url"]),
                model_id=str(model["model_id"]),
                api_key=api_key,
                extra_headers={str(key): str(value) for key, value in extra_headers.items()},
                request_body=request_body,
            )
        )
    return endpoints


def _safe_config(config: dict[str, Any]) -> dict[str, Any]:
    benchmark_keys = {
        "samples_per_dimension",
        "seed",
        "tool_mode",
        "timeout_seconds",
        "retries",
        "warmup_requests",
        "max_tokens_per_request",
        "docker_image",
        "coding_timeout_seconds",
    }
    return {
        "title": config["title"],
        "models": [
            {
                "label": item["label"],
                "base_url": item["base_url"],
                "model_id": item["model_id"],
                "api_key_env": item["api_key_env"],
                "extra_header_names": sorted((item.get("extra_headers") or {}).keys()),
                "request_body": item.get("request_body") or {},
            }
            for item in config["models"]
        ],
        "benchmark": {
            key: value for key, value in config["benchmark"].items() if key in benchmark_keys
        },
    }


def _redact(text: str, endpoint: Endpoint) -> str:
    redacted = text.replace(endpoint.api_key, "[REDACTED]")
    for value in endpoint.extra_headers.values():
        if value:
            redacted = redacted.replace(value, "[REDACTED]")
    return redacted


def _score_case(
    case: Case, completion: Completion, executor: DockerExecutor
) -> tuple[float, str, float]:
    if case.scorer == "humaneval":
        program = assemble_humaneval_program(
            case.payload["prompt"],
            completion.text,
            case.payload["test"],
            case.payload["entry_point"],
        )
        execution = executor.run(program)
        return (1.0 if execution.passed else 0.0), execution.detail, execution.duration_seconds
    if case.scorer == "cruxeval":
        score, detail = score_cruxeval(completion.text, case.payload["expected"])
        return score, detail, 0.0
    if case.scorer == "nl2bash":
        score, detail = score_nl2bash(completion.text, case.payload["expected"])
        return score, detail, 0.0
    if case.scorer == "bfcl":
        calls = completion.tool_calls
        if case.payload["tool_mode"] == "prompt":
            calls = parse_prompt_tool_calls(completion.text)
        score, detail = score_bfcl(calls, case.payload["expected_calls"])
        return score, detail, 0.0
    if case.scorer == "rubric":
        score, detail = score_rubric(completion.text, case.payload["rubric"])
        return score, detail, 0.0
    raise RuntimeError(f"unknown scorer: {case.scorer}")


def _completion_metrics(completion: Completion) -> dict[str, Any]:
    return {
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "decode_tokens": completion.decode_token_count,
        "usage_source": completion.usage_source,
        "ttft_ms": completion.ttft_seconds * 1000,
        "decode_ms": completion.decode_seconds * 1000,
        "e2e_ms": completion.e2e_seconds * 1000,
        "decode_tokens_per_second": completion.decode_tokens_per_second,
        "finish_reason": completion.finish_reason,
        "attempts": completion.attempts,
        "response_mode": completion.response_mode,
    }


def run_case(
    endpoint: Endpoint,
    case: Case,
    *,
    executor: DockerExecutor,
    max_tokens: int | None,
    timeout_seconds: float,
    retries: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "case_id": case.case_id,
        "dimension": case.dimension,
        "source": case.source,
        "source_revision": case.source_revision,
        "scorer": case.scorer,
        "score": 0.0,
        "status": "error",
        "detail": "",
        "error": "",
    }
    try:
        completion = complete(
            endpoint,
            messages=case.messages,
            tools=case.tools,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            temperature=0.0,
            retries=retries,
        )
        score, detail, evaluation_seconds = _score_case(case, completion, executor)
        result.update(
            {
                "score": score,
                "status": "complete",
                "detail": detail,
                "response": completion.text,
                "tool_calls": completion.tool_calls,
                "metrics": _completion_metrics(completion),
                "evaluation_seconds": evaluation_seconds,
            }
        )
    except Exception as exc:  # noqa: BLE001 - preserve all per-case endpoint failures
        result["error"] = _redact(f"{type(exc).__name__}: {exc}", endpoint)
    result["case_wall_seconds"] = time.perf_counter() - started
    return result


def summarize_model(cases: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    dimension_scores: dict[str, float] = {}
    for key, _ in DIMENSIONS:
        values = [float(case["score"]) for case in cases if case["dimension"] == key]
        dimension_scores[key] = statistics.fmean(values) if values else 0.0
    completed = [case for case in cases if case["status"] == "complete"]
    metrics = [case["metrics"] for case in completed if case.get("metrics")]
    output_tokens = sum(int(item["completion_tokens"]) for item in metrics)
    decode_tokens = sum(int(item["decode_tokens"]) for item in metrics)
    prompt_tokens = sum(int(item["prompt_tokens"]) for item in metrics)
    decode_seconds = sum(float(item["decode_ms"]) / 1000 for item in metrics)
    ttfts = [float(item["ttft_ms"]) for item in metrics]
    e2es = [float(item["e2e_ms"]) for item in metrics]
    return {
        "overall_score": statistics.fmean(dimension_scores.values()),
        "dimension_scores": dimension_scores,
        "total_cases": len(cases),
        "completed_cases": len(completed),
        "passed_cases": sum(case["score"] >= 0.999 for case in cases),
        "error_cases": sum(case["status"] == "error" for case in cases),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "median_ttft_ms": statistics.median(ttfts) if ttfts else None,
        "median_e2e_ms": statistics.median(e2es) if e2es else None,
        "decode_tokens_per_second": decode_tokens / max(decode_seconds, 0.001) if metrics else None,
        "effective_tokens_per_second": output_tokens / max(wall_seconds, 0.001) if metrics else None,
        "benchmark_wall_seconds": wall_seconds,
        "estimated_usage_cases": sum(item["usage_source"] == "estimated" for item in metrics),
        "non_stream_cases": sum(item["response_mode"] == "non_stream" for item in metrics),
        "length_limited_cases": sum(item["finish_reason"] == "length" for item in metrics),
    }


def run_model(
    endpoint: Endpoint,
    cases: list[Case],
    *,
    executor: DockerExecutor,
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    warmup_errors = []
    for number in range(int(benchmark["warmup_requests"])):
        try:
            complete(
                endpoint,
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                max_tokens=8,
                timeout_seconds=float(benchmark["timeout_seconds"]),
                temperature=0.0,
                retries=int(benchmark["retries"]),
            )
        except Exception as exc:  # noqa: BLE001 - preserve and continue to case diagnostics
            warmup_errors.append(
                _redact(f"warmup {number + 1}: {type(exc).__name__}: {exc}", endpoint)
            )
    results = []
    print(f"\n{endpoint.label} ({endpoint.model_id})", flush=True)
    for index, case in enumerate(cases, 1):
        print(
            f"  [{index:02d}/{len(cases):02d}] {case.dimension}: {case.case_id} ... ",
            end="",
            flush=True,
        )
        result = run_case(
            endpoint,
            case,
            executor=executor,
            max_tokens=(
                int(benchmark["max_tokens_per_request"])
                if benchmark["max_tokens_per_request"] is not None
                else None
            ),
            timeout_seconds=float(benchmark["timeout_seconds"]),
            retries=int(benchmark["retries"]),
        )
        results.append(result)
        if result["status"] == "complete":
            metric = result["metrics"]
            print(
                f"{result['score'] * 100:5.1f}  "
                f"{metric['e2e_ms']:.0f} ms  {metric['decode_tokens_per_second']:.1f} tok/s",
                flush=True,
            )
        else:
            print(f"ERROR {result['error'][:160]}", flush=True)
    wall_seconds = time.perf_counter() - started
    return {
        "label": endpoint.label,
        "base_url": endpoint.base_url,
        "model_id": endpoint.model_id,
        "warmup_errors": warmup_errors,
        "cases": results,
        "summary": summarize_model(results, wall_seconds),
    }


def _warnings(models: list[dict[str, Any]], tool_mode: str) -> list[str]:
    warnings = [
        "Code-planning and documentation supplements use transparent custom concept rubrics; they are not external leaderboard scores.",
        "The NL2Bash score uses an adapted syntax metric and is not the original human-judged leaderboard metric.",
    ]
    if tool_mode == "prompt":
        warnings.append(
            "BFCL ran in portable prompt mode rather than native API tool-calling mode. Use tool_mode: native when both endpoints support identical tool semantics."
        )
    for model in models:
        summary = model["summary"]
        if summary["estimated_usage_cases"]:
            warnings.append(
                f"{model['label']}: {summary['estimated_usage_cases']} case(s) lacked endpoint usage; token counts were estimated."
            )
        if summary["non_stream_cases"]:
            warnings.append(
                f"{model['label']}: {summary['non_stream_cases']} response(s) were non-streaming; TTFT equals end-to-end latency for those cases."
            )
        if summary["length_limited_cases"]:
            warnings.append(
                f"{model['label']}: {summary['length_limited_cases']} response(s) ended at the output-token limit."
            )
        if summary["error_cases"]:
            warnings.append(
                f"{model['label']}: {summary['error_cases']} case(s) errored and counted as zero."
            )
        warnings.extend(f"{model['label']} {error}" for error in model.get("warmup_errors", []))
    return warnings


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, help="Override output_dir from YAML")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Download/checksum datasets and print the selected cases without calling models",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    comparison_started = time.perf_counter()
    try:
        config = load_config(args.config)
        benchmark = config["benchmark"]
        output_dir = args.output_dir or Path(config["output_dir"])
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        cache_dir = output_dir / "cache"
        print("Preparing pinned benchmark inputs...", flush=True)
        suite = build_suite(
            cache_dir=cache_dir,
            custom_suite_path=DEFAULT_CUSTOM_SUITE,
            samples_per_dimension=int(benchmark["samples_per_dimension"]),
            seed=int(benchmark["seed"]),
            tool_mode=str(benchmark["tool_mode"]),
        )
        counts = {
            key: sum(case.dimension == key for case in suite.cases) for key, _ in DIMENSIONS
        }
        print("Selected cases: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
        if args.prepare_only:
            print(f"Cached {len(suite.artifacts)} artifacts in {cache_dir}")
            for case in suite.cases:
                print(f"  {case.dimension:24} {case.case_id} ({case.source})")
            return 0

        endpoints = endpoints_from_config(config)
        evaluator_image = str(benchmark["docker_image"])
        executor = DockerExecutor(
            image=evaluator_image,
            timeout_seconds=float(benchmark["coding_timeout_seconds"]),
            dockerfile=(
                DEFAULT_EVALUATOR_DOCKERFILE
                if evaluator_image == DEFAULT_EVALUATOR_IMAGE
                else None
            ),
        )
        print("Checking isolated code-execution environment...", flush=True)
        executor.preflight()
        models = [
            run_model(
                endpoint,
                suite.cases,
                executor=executor,
                benchmark=benchmark,
            )
            for endpoint in endpoints
        ]
    except (ConfigurationError, ExecutionInfrastructureError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - concise top-level diagnostic
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    created_at = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "title": str(config["title"]),
        "created_at": created_at,
        "run_id": run_id,
        "seed": int(benchmark["seed"]),
        "comparison_wall_seconds": time.perf_counter() - comparison_started,
        "configuration": _safe_config(config),
        "methodology": {
            "quality_aggregate": "unweighted mean of five dimensions",
            "errors": "scheduled errors score zero",
            "temperature": 0.0,
            "max_tokens_per_request": benchmark["max_tokens_per_request"],
            "tool_mode": str(benchmark["tool_mode"]),
            "execution_backend": "isolated Docker",
            "execution_image": str(benchmark["docker_image"]),
            "execution_numpy_version": executor.numpy_version,
            "revisions": {
                "HumanEval+": HUMANEVAL_REVISION,
                "CRUXEval": CRUXEVAL_REVISION,
                "NL2Bash": NL2BASH_REVISION,
                "BFCL": BFCL_REVISION,
            },
        },
        "benchmark_artifacts": suite.artifacts,
        "models": models,
    }
    report["warnings"] = _warnings(models, str(benchmark["tool_mode"]))
    json_path = output_dir / f"model-comparison-{run_id}.json"
    html_path = output_dir / f"model-comparison-{run_id}.html"
    _write_json(json_path, report)
    render_html(report, html_path)

    print("\nComparison complete", flush=True)
    for model in models:
        summary = model["summary"]
        print(
            f"  {model['label']}: quality={summary['overall_score'] * 100:.1f}/100, "
            f"decode={summary['decode_tokens_per_second'] or 0:.1f} tok/s, "
            f"full-run={summary['effective_tokens_per_second'] or 0:.1f} tok/s",
            flush=True,
        )
    print(f"  HTML: {html_path}")
    print(f"  JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
