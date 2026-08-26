"""Deterministic scorers for the model-comparison benchmark cases."""

from __future__ import annotations

import ast
import difflib
import json
import re
import shlex
from collections import Counter
from typing import Any


def strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:python|bash|sh|json)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


def assemble_humaneval_program(
    prompt: str, generation: str, test: str, entry_point: str
) -> str:
    """Turn common chat-style answers into a HumanEval executable."""
    fenced = re.search(
        r"```(?:python)?[ \t]*(?:\r?\n)?(.*?)```", generation, re.DOTALL | re.IGNORECASE
    )
    # Leading indentation is meaningful for completion-style HumanEval output.
    code = fenced.group(1).rstrip() if fenced else generation.rstrip()
    if re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", code):
        function_start = re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", prompt)
        preamble = prompt[: function_start.start()] if function_start else ""
        if preamble.strip() and preamble.strip() not in code:
            solution = preamble.rstrip() + "\n\n" + code
        else:
            solution = code
    else:
        solution = prompt.rstrip() + "\n" + code
    return f"{solution.rstrip()}\n\n{test.rstrip()}\n\ncheck({entry_point})\n"


def score_cruxeval(generation: str, expected: str) -> tuple[float, str]:
    answer = strip_code_fence(generation).strip()
    answer = re.sub(r"^(?:output|answer|result)\s*:\s*", "", answer, flags=re.IGNORECASE)
    candidates = [answer]
    if "\n" in answer:
        candidates.extend(line.strip() for line in reversed(answer.splitlines()) if line.strip())
    try:
        expected_value = ast.literal_eval(expected)
    except (ValueError, SyntaxError):
        expected_value = expected.strip()
    for candidate in candidates:
        try:
            actual_value = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            actual_value = candidate.strip()
        if actual_value == expected_value:
            return 1.0, "exact literal match"
    return 0.0, f"expected {expected[:160]!r}; got {answer[:160]!r}"


def _shell_tokens(command: str) -> list[str]:
    command = strip_code_fence(command).strip()
    command = re.sub(r"^(?:command\s*:\s*|\$\s*)", "", command, flags=re.IGNORECASE)
    command = command.splitlines()[0].strip() if command else ""
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return command.split()


def score_nl2bash(generation: str, expected: str) -> tuple[float, str]:
    """Adapted NL2Bash syntax score: utility, token F1, and sequence similarity."""
    actual_tokens = _shell_tokens(generation)
    expected_tokens = _shell_tokens(expected)
    if not actual_tokens or not expected_tokens:
        return 0.0, "empty or unparsable command"
    if actual_tokens == expected_tokens:
        return 1.0, "exact normalized command match"

    utility = 1.0 if actual_tokens[0] == expected_tokens[0] else 0.0
    actual_counts, expected_counts = Counter(actual_tokens), Counter(expected_tokens)
    overlap = sum((actual_counts & expected_counts).values())
    precision = overlap / len(actual_tokens)
    recall = overlap / len(expected_tokens)
    token_f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    sequence = difflib.SequenceMatcher(a=actual_tokens, b=expected_tokens).ratio()
    score = 0.40 * utility + 0.35 * token_f1 + 0.25 * sequence
    return score, f"utility={utility:.2f}, token_f1={token_f1:.2f}, sequence={sequence:.2f}"


def parse_prompt_tool_calls(text: str) -> list[dict[str, Any]]:
    raw = strip_code_fence(text)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    calls = payload.get("tool_calls", payload if isinstance(payload, list) else [])
    if not isinstance(calls, list):
        return []
    normalized: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else call
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        normalized.append({"name": function.get("name", ""), "arguments": arguments})
    return normalized


def _equivalent(actual: Any, allowed: list[Any]) -> bool:
    for candidate in allowed:
        if actual == candidate:
            return True
        if str(actual).strip() == str(candidate).strip():
            return True
    return False


def _call_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    if len(expected) != 1:
        return False
    expected_name, expected_arguments = next(iter(expected.items()))
    if actual.get("name") != expected_name:
        return False
    actual_arguments = actual.get("arguments")
    if not isinstance(actual_arguments, dict):
        return False
    for key, allowed in expected_arguments.items():
        if "" in allowed and key not in actual_arguments:
            continue
        if key not in actual_arguments or not _equivalent(actual_arguments[key], allowed):
            return False
    unexpected = set(actual_arguments) - set(expected_arguments)
    return not unexpected


def score_bfcl(
    actual_calls: list[dict[str, Any]], expected_calls: list[dict[str, Any]]
) -> tuple[float, str]:
    """Order-independent BFCL-style all-or-nothing AST comparison."""
    if len(actual_calls) != len(expected_calls):
        return 0.0, f"expected {len(expected_calls)} call(s), got {len(actual_calls)}"
    unmatched = list(actual_calls)
    for expected in expected_calls:
        for index, actual in enumerate(unmatched):
            if _call_matches(actual, expected):
                unmatched.pop(index)
                break
        else:
            return 0.0, f"missing or incorrect call: {json.dumps(expected, sort_keys=True)}"
    return 1.0, "all tool calls and arguments matched"


def score_rubric(text: str, rubric: dict[str, Any]) -> tuple[float, str]:
    """Score transparent concept coverage and apply declared anti-pattern penalties."""
    normalized = re.sub(r"\s+", " ", text.lower())
    criteria = rubric.get("criteria") or []
    if not criteria:
        return 0.0, "rubric has no criteria"
    earned = 0
    matched: list[str] = []
    missed: list[str] = []
    for criterion in criteria:
        alternatives = [str(item).lower() for item in criterion.get("any", [])]
        identifier = str(criterion.get("id", "criterion"))
        if alternatives and any(term in normalized for term in alternatives):
            earned += 1
            matched.append(identifier)
        else:
            missed.append(identifier)
    penalty_hits = [
        term for term in rubric.get("penalty_terms", []) if str(term).lower() in normalized
    ]
    score = earned / len(criteria)
    if penalty_hits:
        score = max(0.0, score - min(0.25, 0.05 * len(penalty_hits)))
    detail = f"matched={matched}; missed={missed}"
    if penalty_hits:
        detail += f"; penalties={penalty_hits}"
    return score, detail
