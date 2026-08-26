import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from ml.model_compare.client import (
    Endpoint,
    _parse_json_response,
    _parse_stream_response,
    complete,
    estimate_tokens,
    normalize_chat_url,
)
from ml.model_compare.execution import DockerExecutor, ExecutionInfrastructureError
from ml.model_compare.report import render_html
from ml.model_compare.runner import (
    ConfigurationError,
    _redact,
    endpoints_from_config,
    load_config,
    summarize_model,
)
from ml.model_compare.scoring import (
    assemble_humaneval_program,
    parse_prompt_tool_calls,
    score_bfcl,
    score_cruxeval,
    score_nl2bash,
    score_rubric,
)


class ClientTests(unittest.TestCase):
    class FakeResponse:
        status_code = 200

        def __init__(self, *, payload=None, lines=None):
            self.payload = payload
            self.lines = lines or []
            self.ok = True
            self.headers = {"content-type": "application/json"}
            self.text = ""

        def json(self):
            return self.payload

        def iter_lines(self, **_kwargs):
            return iter(self.lines)

        def close(self):
            pass

    def test_normalize_chat_url_accepts_all_supported_shapes(self):
        expected = "http://localhost:8000/v1/chat/completions"
        self.assertEqual(normalize_chat_url("http://localhost:8000"), expected)
        self.assertEqual(normalize_chat_url("http://localhost:8000/v1"), expected)
        self.assertEqual(normalize_chat_url(expected), expected)

    def test_token_estimate_is_nonzero_and_deterministic(self):
        self.assertEqual(estimate_tokens("hello, world"), estimate_tokens("hello, world"))
        self.assertGreater(estimate_tokens("hello, world"), 0)

    def test_endpoint_does_not_put_key_in_url(self):
        endpoint = Endpoint("A", "https://example.test/v1", "model", "secret")
        self.assertNotIn("secret", endpoint.chat_url)

    def test_stream_parser_assembles_content_tools_and_usage(self):
        lines = [
            'data: {"choices":[{"delta":{"content":"hi"}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"look","arguments":"{\\"id\\":"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"up","arguments":"7}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":3}}',
            "data: [DONE]",
        ]
        completion = _parse_stream_response(self.FakeResponse(lines=lines), 0.0, 1)
        self.assertEqual(completion.text, "hi")
        self.assertEqual(completion.tool_calls[0]["name"], "lookup")
        self.assertEqual(completion.tool_calls[0]["arguments"], {"id": 7})
        self.assertEqual(completion.completion_tokens, 3)
        self.assertEqual(completion.decode_token_count, 2)
        self.assertEqual(completion.response_mode, "stream")

    def test_non_stream_parser_marks_latency_mode(self):
        response = self.FakeResponse(
            payload={
                "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }
        )
        completion = _parse_json_response(response, 0.0, 1)
        self.assertEqual(completion.text, "OK")
        self.assertEqual(completion.decode_token_count, 1)
        self.assertEqual(completion.response_mode, "non_stream")

    def test_optional_cap_and_reasoning_fields_are_sent(self):
        response = self.FakeResponse(
            payload={
                "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }
        )
        endpoint = Endpoint(
            "A",
            "https://example.test/v1",
            "model",
            "secret",
            request_body={"reasoning_effort": "low"},
        )
        with patch("requests.post", return_value=response) as post:
            complete(
                endpoint,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=None,
                timeout_seconds=10,
                retries=0,
            )
        sent = post.call_args.kwargs["json"]
        self.assertNotIn("max_tokens", sent)
        self.assertEqual(sent["reasoning_effort"], "low")


class ScoringTests(unittest.TestCase):
    def test_cruxeval_accepts_fenced_literal(self):
        score, _ = score_cruxeval("```python\n[1, 2, 3]\n```", "[1, 2, 3]")
        self.assertEqual(score, 1.0)

    def test_nl2bash_exact_and_partial_scores(self):
        exact, _ = score_nl2bash("```bash\nfind . -name '*.py'\n```", "find . -name '*.py'")
        partial, _ = score_nl2bash("find /tmp -name '*.py'", "find . -name '*.py'")
        wrong, _ = score_nl2bash("ls -la", "find . -name '*.py'")
        self.assertEqual(exact, 1.0)
        self.assertGreater(partial, wrong)

    def test_bfcl_matches_parallel_calls_and_optional_default(self):
        actual = [
            {"name": "spotify.play", "arguments": {"artist": "Maroon 5", "duration": 15}},
            {"name": "spotify.play", "arguments": {"artist": "Taylor Swift", "duration": 20}},
        ]
        expected = [
            {"spotify.play": {"artist": ["Taylor Swift"], "duration": [20]}},
            {"spotify.play": {"artist": ["Maroon 5"], "duration": [15]}},
        ]
        self.assertEqual(score_bfcl(actual, expected)[0], 1.0)
        optional_actual = [{"name": "shape", "arguments": {"side": 3}}]
        optional_expected = [{"shape": {"side": [3], "unit": ["", "cm"]}}]
        self.assertEqual(score_bfcl(optional_actual, optional_expected)[0], 1.0)

    def test_prompt_tool_call_parser(self):
        calls = parse_prompt_tool_calls(
            '```json\n{"tool_calls":[{"name":"lookup","arguments":{"id":7}}]}\n```'
        )
        self.assertEqual(calls, [{"name": "lookup", "arguments": {"id": 7}}])

    def test_rubric_is_transparent_and_penalized(self):
        rubric = {
            "criteria": [
                {"id": "tests", "any": ["regression test"]},
                {"id": "rollout", "any": ["canary"]},
            ],
            "penalty_terms": ["disable tests"],
        }
        full, _ = score_rubric("Add a regression test and canary rollout.", rubric)
        penalized, _ = score_rubric(
            "Add a regression test and canary rollout, then disable tests.", rubric
        )
        self.assertEqual(full, 1.0)
        self.assertLess(penalized, full)

    def test_humaneval_assembly_supports_completion_and_full_function(self):
        prompt = "def add(a, b):\n"
        test = "def check(candidate):\n    assert candidate(1, 2) == 3"
        completion = assemble_humaneval_program(prompt, "    return a + b", test, "add")
        full = assemble_humaneval_program(
            prompt, "```python\ndef add(a, b):\n    return a + b\n```", test, "add"
        )
        self.assertIn("def add(a, b):\n    return a + b", completion)
        self.assertIn("check(add)", full)

    def test_humaneval_assembly_preserves_fenced_completion_indentation(self):
        prompt = 'def add(a, b):\n    """Return the sum."""\n'
        test = "def check(candidate):\n    assert candidate(1, 2) == 3"
        program = assemble_humaneval_program(
            prompt,
            "```python\n    return a + b\n```",
            test,
            "add",
        )
        compile(program, "<humaneval>", "exec")
        self.assertIn('"""Return the sum."""\n    return a + b', program)


class ExecutionTests(unittest.TestCase):
    @patch("ml.model_compare.execution.subprocess.run")
    @patch("ml.model_compare.execution.shutil.which", return_value="/usr/bin/docker")
    def test_preflight_rejects_an_image_without_numpy(self, _which, run):
        run.side_effect = [
            unittest.mock.Mock(returncode=0, stdout="27.0", stderr=""),
            unittest.mock.Mock(returncode=0, stdout="[]", stderr=""),
            unittest.mock.Mock(
                returncode=1,
                stdout="",
                stderr="ModuleNotFoundError: No module named 'numpy'",
            ),
        ]
        executor = DockerExecutor(image="bad-evaluator", timeout_seconds=30)
        with self.assertRaisesRegex(ExecutionInfrastructureError, "cannot import.*numpy"):
            executor.preflight()

    @patch("ml.model_compare.execution.subprocess.run")
    @patch("ml.model_compare.execution.shutil.which", return_value="/usr/bin/docker")
    def test_preflight_builds_default_image_and_records_numpy(self, _which, run):
        run.side_effect = [
            unittest.mock.Mock(returncode=0, stdout="27.0", stderr=""),
            unittest.mock.Mock(returncode=1, stdout="", stderr="not found"),
            unittest.mock.Mock(returncode=0, stdout="built", stderr=""),
            unittest.mock.Mock(returncode=0, stdout="2.3.2\n", stderr=""),
        ]
        with tempfile.TemporaryDirectory() as directory:
            dockerfile = Path(directory) / "Dockerfile"
            dockerfile.write_text("FROM python:3.12-slim\n", encoding="utf-8")
            executor = DockerExecutor(
                image="local-evaluator",
                timeout_seconds=30,
                dockerfile=dockerfile,
            )
            executor.preflight()
        self.assertEqual(executor.numpy_version, "2.3.2")
        self.assertEqual(run.call_args_list[2].args[0][:3], ["docker", "build", "--tag"])


class RunnerTests(unittest.TestCase):
    def test_error_redaction_covers_key_and_custom_header_values(self):
        endpoint = Endpoint(
            "A", "https://a", "a", "key-secret", {"X-Provider-Key": "header-secret"}
        )
        redacted = _redact("key-secret and header-secret", endpoint)
        self.assertNotIn("secret", redacted)
        self.assertEqual(redacted, "[REDACTED] and [REDACTED]")

    def test_api_keys_are_read_from_environment(self):
        config = {
            "models": [
                {"label": "A", "base_url": "https://a", "model_id": "a", "api_key_env": "KEY_A"},
                {"label": "B", "base_url": "https://b", "model_id": "b", "api_key_env": "KEY_B"},
            ]
        }
        with patch.dict(os.environ, {"KEY_A": "alpha", "KEY_B": "beta"}, clear=False):
            endpoints = endpoints_from_config(config)
        self.assertEqual([endpoint.api_key for endpoint in endpoints], ["alpha", "beta"])

    def test_api_keys_can_come_from_ignored_dotenv_file(self):
        config = {
            "models": [
                {"label": "A", "base_url": "https://a", "model_id": "a", "api_key_env": "KEY_A"},
                {"label": "B", "base_url": "https://b", "model_id": "b", "api_key_env": "KEY_B"},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env.local").write_text("KEY_A=alpha\nKEY_B=beta\n", encoding="utf-8")
            with patch("ml.model_compare.runner.PROJECT_ROOT", root), patch.dict(
                os.environ, {}, clear=True
            ):
                endpoints = endpoints_from_config(config)
        self.assertEqual([endpoint.api_key for endpoint in endpoints], ["alpha", "beta"])

    def test_reasoning_request_body_is_loaded_and_core_fields_are_protected(self):
        config = {
            "models": [
                {
                    "label": "A",
                    "base_url": "https://a",
                    "model_id": "a",
                    "api_key_env": "KEY_A",
                    "request_body": {"reasoning_effort": "low"},
                },
                {"label": "B", "base_url": "https://b", "model_id": "b", "api_key_env": "KEY_B"},
            ]
        }
        with patch.dict(os.environ, {"KEY_A": "alpha", "KEY_B": "beta"}, clear=False):
            endpoints = endpoints_from_config(config)
        self.assertEqual(endpoints[0].request_body, {"reasoning_effort": "low"})

        config["models"][0]["request_body"] = {"max_tokens": 7}
        with patch.dict(os.environ, {"KEY_A": "alpha", "KEY_B": "beta"}, clear=False):
            with self.assertRaises(ConfigurationError):
                endpoints_from_config(config)

    def test_budget_defaults_large_and_accepts_null(self):
        base = {
            "models": [
                {"label": "A", "base_url": "https://a", "model_id": "a", "api_key_env": "KEY_A"},
                {"label": "B", "base_url": "https://b", "model_id": "b", "api_key_env": "KEY_B"},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(base), encoding="utf-8")
            self.assertEqual(load_config(path)["benchmark"]["max_tokens_per_request"], 16384)
            base["benchmark"] = {"max_tokens_per_request": None}
            path.write_text(yaml.safe_dump(base), encoding="utf-8")
            self.assertIsNone(load_config(path)["benchmark"]["max_tokens_per_request"])

    def test_errors_count_as_zero_in_dimension_summary(self):
        cases = []
        for dimension in (
            "code_planning",
            "coding",
            "analysis_documentation",
            "terminal_bash",
            "orchestration",
        ):
            cases.append(
                {
                    "dimension": dimension,
                    "score": 1.0 if dimension != "coding" else 0.0,
                    "status": "complete" if dimension != "coding" else "error",
                }
            )
        summary = summarize_model(cases, 10.0)
        self.assertAlmostEqual(summary["overall_score"], 0.8)
        self.assertEqual(summary["error_cases"], 1)


class ReportTests(unittest.TestCase):
    def test_html_has_each_dimension_and_no_secret(self):
        dimensions = {
            "code_planning": 0.8,
            "coding": 0.7,
            "analysis_documentation": 0.6,
            "terminal_bash": 0.5,
            "orchestration": 0.4,
        }
        model_template = {
            "base_url": "https://example.test/v1",
            "warmup_errors": [],
            "cases": [],
            "summary": {
                "dimension_scores": dimensions,
                "overall_score": 0.6,
                "passed_cases": 1,
                "total_cases": 5,
                "error_cases": 0,
                "median_ttft_ms": 10,
                "median_e2e_ms": 100,
                "decode_tokens_per_second": 40,
                "effective_tokens_per_second": 30,
            },
        }
        report = {
            "title": "Comparison",
            "created_at": "2026-08-25T12:00:00+00:00",
            "seed": 42,
            "comparison_wall_seconds": 10,
            "methodology": {"tool_mode": "prompt"},
            "warnings": [],
            "models": [
                {**model_template, "label": "A", "model_id": "a"},
                {**model_template, "label": "B", "model_id": "b"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.html"
            render_html(report, path)
            document = path.read_text(encoding="utf-8")
        for title in (
            "Code Planning",
            "Coding",
            "Code Analysis / Documentation",
            "Terminal / Bash",
            "Orchestration",
        ):
            self.assertIn(title, document)
        self.assertNotIn("secret", document)
        self.assertEqual(document.count("class='chart'"), 9)


if __name__ == "__main__":
    unittest.main()
