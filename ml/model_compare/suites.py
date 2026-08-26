"""Pinned benchmark dataset loaders and prompt construction."""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


HUMANEVAL_REVISION = "d32357cf319e50e9c8d8dab5ea876c72b0fd321b"
CRUXEVAL_REVISION = "b96af0450242eb4da433032b90998f25588a5d0f"
NL2BASH_REVISION = "d6b9f5bdff45621d190134e31ab63b7bf7002190"
BFCL_REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"

HF_RESOLVE = "https://huggingface.co/datasets/{dataset}/resolve/{revision}/{path}"
GITHUB_RAW = "https://raw.githubusercontent.com/{repo}/{revision}/{path}"


@dataclass
class Case:
    case_id: str
    dimension: str
    source: str
    source_revision: str
    messages: list[dict[str, Any]]
    scorer: str
    payload: dict[str, Any] = field(default_factory=dict)
    tools: list[dict[str, Any]] | None = None


@dataclass
class Suite:
    cases: list[Case]
    artifacts: list[dict[str, Any]]


class DatasetError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_cached(url: str, path: Path, timeout_seconds: float = 120) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ml-compute-model-compare/1"})
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                with temporary.open("wb") as target:
                    while chunk := response.read(1024 * 1024):
                        target.write(chunk)
            temporary.replace(path)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise DatasetError(f"could not download {url}: {exc}") from exc
    return {"url": url, "path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"invalid JSONL at {path}:{number}") from exc
    return rows


def _sample(rows: list[Any], count: int, seed: int, salt: str) -> list[Any]:
    if count >= len(rows):
        return list(rows)
    salted_seed = int(hashlib.sha256(f"{seed}:{salt}".encode()).hexdigest()[:16], 16)
    return random.Random(salted_seed).sample(rows, count)


def _humaneval_cases(cache: Path, count: int, seed: int) -> tuple[list[Case], list[dict[str, Any]]]:
    url = HF_RESOLVE.format(
        dataset="evalplus/humanevalplus", revision=HUMANEVAL_REVISION, path="test.jsonl"
    )
    path = cache / f"humanevalplus-{HUMANEVAL_REVISION[:12]}.jsonl"
    artifact = download_cached(url, path)
    cases = []
    for row in _sample(_read_jsonl(path), count, seed, "humanevalplus"):
        cases.append(
            Case(
                case_id=row["task_id"],
                dimension="coding",
                source="HumanEval+",
                source_revision=HUMANEVAL_REVISION,
                scorer="humaneval",
                messages=[
                    {
                        "role": "system",
                        "content": "Complete the Python function. Return only executable Python code, without commentary.",
                    },
                    {"role": "user", "content": row["prompt"]},
                ],
                payload={
                    "prompt": row["prompt"],
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                },
            )
        )
    return cases, [artifact]


def _cruxeval_cases(cache: Path, count: int, seed: int) -> tuple[list[Case], list[dict[str, Any]]]:
    url = HF_RESOLVE.format(
        dataset="cruxeval-org/cruxeval", revision=CRUXEVAL_REVISION, path="test.jsonl"
    )
    path = cache / f"cruxeval-{CRUXEVAL_REVISION[:12]}.jsonl"
    artifact = download_cached(url, path)
    cases = []
    for row in _sample(_read_jsonl(path), count, seed, "cruxeval"):
        prompt = (
            "Predict the exact Python value returned by the function call. "
            "Return only one Python literal.\n\n"
            f"{row['code']}\n\nCall: f({row['input']})"
        )
        cases.append(
            Case(
                case_id=row["id"],
                dimension="analysis_documentation",
                source="CRUXEval-O",
                source_revision=CRUXEVAL_REVISION,
                scorer="cruxeval",
                messages=[
                    {"role": "system", "content": "Analyze Python precisely and follow the requested output format."},
                    {"role": "user", "content": prompt},
                ],
                payload={"expected": row["output"]},
            )
        )
    return cases, [artifact]


def _nl2bash_cases(cache: Path, count: int, seed: int) -> tuple[list[Case], list[dict[str, Any]]]:
    artifacts = []
    paths: dict[str, Path] = {}
    for extension in ("nl", "cm"):
        url = GITHUB_RAW.format(
            repo="TellinaTool/nl2bash",
            revision=NL2BASH_REVISION,
            path=f"data/bash/all.{extension}",
        )
        path = cache / f"nl2bash-{NL2BASH_REVISION[:12]}.{extension}"
        artifacts.append(download_cached(url, path))
        paths[extension] = path
    descriptions = paths["nl"].read_text(encoding="utf-8").splitlines()
    commands = paths["cm"].read_text(encoding="utf-8").splitlines()
    if len(descriptions) != len(commands):
        raise DatasetError("NL2Bash descriptions and commands have different lengths")
    pairs = list(enumerate(zip(descriptions, commands)))
    cases = []
    for index, (description, command) in _sample(pairs, count, seed, "nl2bash"):
        cases.append(
            Case(
                case_id=f"nl2bash/{index}",
                dimension="terminal_bash",
                source="NL2Bash",
                source_revision=NL2BASH_REVISION,
                scorer="nl2bash",
                messages=[
                    {
                        "role": "system",
                        "content": "Translate the request to one Bash command. Return only the command, without a prompt marker or explanation.",
                    },
                    {"role": "user", "content": description},
                ],
                payload={"expected": command},
            )
        )
    return cases, artifacts


def _native_tools(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools = []
    for function in functions:
        copied = json.loads(json.dumps(function))
        parameters = copied.get("parameters") or {}
        if parameters.get("type") == "dict":
            parameters["type"] = "object"
        tools.append({"type": "function", "function": copied})
    return tools


def _bfcl_cases(
    cache: Path, count: int, seed: int, tool_mode: str
) -> tuple[list[Case], list[dict[str, Any]]]:
    categories = ("simple_python", "multiple", "parallel")
    loaded: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    artifacts = []
    base = "berkeley-function-call-leaderboard/bfcl_eval/data"
    for category in categories:
        rows_by_kind = {}
        for kind, relative in (
            ("questions", f"{base}/BFCL_v4_{category}.json"),
            ("answers", f"{base}/possible_answer/BFCL_v4_{category}.json"),
        ):
            url = GITHUB_RAW.format(
                repo="EnlightenedAI/BFCL", revision=BFCL_REVISION, path=relative
            )
            path = cache / f"bfcl-v4-{category}-{kind}-{BFCL_REVISION[:12]}.jsonl"
            artifacts.append(download_cached(url, path))
            rows_by_kind[kind] = {row["id"]: row for row in _read_jsonl(path)}
        ids = sorted(set(rows_by_kind["questions"]) & set(rows_by_kind["answers"]))
        loaded[category] = [
            (rows_by_kind["questions"][case_id], rows_by_kind["answers"][case_id])
            for case_id in ids
        ]

    selected: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    per_category = max(1, (count + len(categories) - 1) // len(categories))
    for category in categories:
        for question, answer in _sample(loaded[category], per_category, seed, f"bfcl:{category}"):
            selected.append((category, question, answer))
    selected = selected[:count]

    cases = []
    for category, question, answer in selected:
        messages = question["question"][0]
        functions = question["function"]
        tools = _native_tools(functions) if tool_mode == "native" else None
        if tool_mode == "prompt":
            instruction = (
                "Select and call the supplied tools. Return JSON only in this shape: "
                '{"tool_calls":[{"name":"tool.name","arguments":{"key":"value"}}]}. '
                "Include every required parallel call and no commentary.\n\n"
                f"Tools:\n{json.dumps(functions, indent=2, sort_keys=True)}"
            )
            messages = [{"role": "system", "content": instruction}, *messages]
        cases.append(
            Case(
                case_id=question["id"],
                dimension="orchestration",
                source=f"BFCL v4/{category}",
                source_revision=BFCL_REVISION,
                scorer="bfcl",
                messages=messages,
                tools=tools,
                payload={"expected_calls": answer["ground_truth"], "tool_mode": tool_mode},
            )
        )
    return cases, artifacts


def _custom_cases(path: Path, count: int, seed: int) -> list[Case]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    cases = []
    for dimension, rows in config.get("dimensions", {}).items():
        for row in _sample(rows, count, seed, f"custom:{dimension}"):
            cases.append(
                Case(
                    case_id=row["id"],
                    dimension=dimension,
                    source=row.get("source", "ml-compute custom evaluation v1"),
                    source_revision=str(config.get("version", 1)),
                    scorer="rubric",
                    messages=[
                        {"role": "system", "content": row["system"]},
                        {"role": "user", "content": row["prompt"]},
                    ],
                    payload={"rubric": row["rubric"]},
                )
            )
    return cases


def build_suite(
    *,
    cache_dir: Path,
    custom_suite_path: Path,
    samples_per_dimension: int,
    seed: int,
    tool_mode: str,
) -> Suite:
    """Download immutable inputs and build one identical case set for both models."""
    if samples_per_dimension < 1:
        raise ValueError("samples_per_dimension must be positive")
    if tool_mode not in {"prompt", "native"}:
        raise ValueError("tool_mode must be 'prompt' or 'native'")
    cases: list[Case] = []
    artifacts: list[dict[str, Any]] = []
    for loader in (_humaneval_cases, _cruxeval_cases, _nl2bash_cases):
        loaded_cases, loaded_artifacts = loader(cache_dir, samples_per_dimension, seed)
        cases.extend(loaded_cases)
        artifacts.extend(loaded_artifacts)
    bfcl_cases, bfcl_artifacts = _bfcl_cases(
        cache_dir, samples_per_dimension, seed, tool_mode
    )
    cases.extend(bfcl_cases)
    artifacts.extend(bfcl_artifacts)

    custom = _custom_cases(custom_suite_path, samples_per_dimension, seed)
    artifacts.append(
        {
            "url": "local:registry/model_compare_custom.yaml",
            "path": str(custom_suite_path),
            "sha256": _sha256(custom_suite_path),
            "bytes": custom_suite_path.stat().st_size,
        }
    )
    planning = [case for case in custom if case.dimension == "code_planning"]
    documentation = [case for case in custom if case.dimension == "analysis_documentation"]
    cases.extend(planning[:samples_per_dimension])
    # Split analysis/documentation evenly between CRUXEval and documentation cases.
    official_analysis = [case for case in cases if case.source == "CRUXEval-O"]
    keep_official = max(1, (samples_per_dimension + 1) // 2)
    cases = [case for case in cases if case.source != "CRUXEval-O"]
    cases.extend(official_analysis[:keep_official])
    cases.extend(documentation[: max(0, samples_per_dimension - keep_official)])

    dimension_order = {
        "code_planning": 0,
        "coding": 1,
        "analysis_documentation": 2,
        "terminal_bash": 3,
        "orchestration": 4,
    }
    cases.sort(key=lambda case: (dimension_order[case.dimension], case.case_id))
    return Suite(cases=cases, artifacts=artifacts)
