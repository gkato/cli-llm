import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from ml.cli import cli
from ml.config import get_models


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "Qwen38-27B-SGLang-DSpark.sh"
STARTER = ROOT / "scripts" / "start-Qwen38-27B-SGLang-DSpark.sh"
PROFILE = ROOT / "config" / "dspark-qwen38-27b-sglang.env"


def values():
    return dict(
        line.split("=", 1)
        for line in PROFILE.read_text().splitlines()
        if line and not line.startswith("#")
    )


class Qwen38SGLangRecipeTests(unittest.TestCase):
    def test_profile_keeps_raw_api_private(self):
        profile = values()
        self.assertEqual(profile["QUANT"], "nvfp4")
        self.assertEqual(profile["YARN"], "1")
        self.assertEqual(profile["CONTEXT_LENGTH"], "524288")
        self.assertEqual(profile["SPECULATIVE_MODE"], "mtp")
        self.assertEqual(profile["SERVING_HOST"], "127.0.0.1")
        self.assertEqual(profile["DSPARK_PROXY_PORT"], "8000")

    def test_registry_matches_pinned_recipe(self):
        model = get_models()["qwen3.8-27b-nvfp4-sglang-dspark"]
        self.assertEqual(model["serve_backend"], "qwen38-27b-sglang")
        self.assertEqual(model["nodes"], 1)
        self.assertEqual(model["runtime"], "sglang")
        self.assertEqual(model["upstream_revision"], "9fb18edf8cfb3364e8aa89258e6d5ab1fe1fd11a")
        self.assertEqual(model["raw_api_url"], "http://127.0.0.1:8888")

    def test_scripts_are_syntax_valid_and_patch_private_bind(self):
        for script in (SCRIPT, STARTER):
            result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        text = SCRIPT.read_text()
        self.assertIn('HOST="127.0.0.1"', text)
        self.assertIn("dspark-proxy", text)

    @patch("ml.cli.subprocess.run")
    def test_cli_dispatches_to_recipe(self, run):
        run.return_value = SimpleNamespace(returncode=0)
        result = CliRunner().invoke(cli, ["qwen38-27b-sglang", "status"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(run.call_args.args[0][0].endswith("Qwen38-27B-SGLang-DSpark.sh"))
