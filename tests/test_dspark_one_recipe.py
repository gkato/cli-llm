import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from ml.cli import cli
from ml.config import get_models


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "DS4-Flash-One-DSpark.sh"
STARTER = ROOT / "scripts" / "start-DS4-Flash-One-DSpark.sh"
PROFILE = ROOT / "config" / "dspark-one-deepseek-v4-flash-0731.env"


def profile_values() -> dict[str, str]:
    values = {}
    for line in PROFILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    return values


class DSparkOneRecipeTests(unittest.TestCase):
    def test_profile_preserves_reviewed_deep_context_and_private_bind(self):
        profile = profile_values()

        self.assertEqual(profile["MAX_MODEL_LEN"], "384000")
        self.assertEqual(profile["MAX_NUM_SEQS"], "1")
        self.assertEqual(profile["MAX_NUM_BATCHED_TOKENS"], "8224")
        self.assertEqual(profile["GPU_MEMORY_UTILIZATION"], "0.94")
        self.assertEqual(profile["KV_RECORD"], "stock432")
        self.assertEqual(profile["ABLATE"], "0")
        self.assertEqual(profile["SERVING_HOST"], "127.0.0.1")
        self.assertEqual(profile["SERVING_PORT"], "8888")
        self.assertEqual(profile["DSPARK_PROXY_PORT"], "8000")
        self.assertEqual(profile["DSPARK_ONE_MIN_AVAILABLE_GIB"], "114.3")

    def test_registry_matches_pinned_recipe(self):
        model = get_models()["deepseek-v4-flash-0731-dspark-one"]

        self.assertEqual(model["serve_backend"], "dspark-one")
        self.assertEqual(model["nodes"], 1)
        self.assertEqual(model["tensor_parallel_size"], 1)
        self.assertEqual(model["quantization"], "exl3")
        self.assertEqual(model["bits_per_weight"], 3.0)
        self.assertEqual(model["max_model_len"], 384000)
        self.assertEqual(model["max_num_batched_tokens"], 8224)
        self.assertEqual(model["raw_api_url"], "http://127.0.0.1:8888")
        self.assertEqual(model["proxy_bind"], "0.0.0.0:8000")
        self.assertTrue(model["dedicated_host"])
        self.assertFalse(model["harness_coexistence"])
        self.assertEqual(
            model["upstream_revision"],
            "fdcd538fbf95fb15b2d6850db9613d22b2c889b8",
        )
        self.assertIn("@sha256:", model["runtime_image"])

    def test_lifecycle_enforces_proxy_boundary_and_upstream_pins(self):
        script = SCRIPT.read_text(encoding="utf-8")

        self.assertIn(
            'UPSTREAM_REVISION_DEFAULT="fdcd538fbf95fb15b2d6850db9613d22b2c889b8"',
            script,
        )
        self.assertIn('[[ "${SERVING_HOST}" == "127.0.0.1" ]]', script)
        self.assertIn("run_proxy_cli serve", script)
        self.assertIn("run_proxy_cli smoke", script)
        self.assertIn("the public proxy remains off", script)
        self.assertIn("Tailscale Funnel targets unauthenticated raw port", script)

    def test_first_run_executes_the_full_local_bootstrap(self):
        starter = STARTER.read_text(encoding="utf-8")

        for action in ("setup", "build", "download", "gpu-check"):
            self.assertIn(f'dspark-one {action}', starter)
        self.assertIn('dspark-one configure', starter)
        self.assertIn('dspark-one start', starter)

    @patch("ml.cli.subprocess.run")
    def test_cli_dispatches_to_independent_one_spark_script(self, run):
        run.return_value = SimpleNamespace(returncode=0)

        result = CliRunner().invoke(cli, ["dspark-one", "status"])

        self.assertEqual(result.exit_code, 0, result.output)
        command = run.call_args.args[0]
        self.assertTrue(command[0].endswith("scripts/DS4-Flash-One-DSpark.sh"))
        self.assertEqual(command[1], "status")


if __name__ == "__main__":
    unittest.main()
