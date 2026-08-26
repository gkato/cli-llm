import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from ml.cli import cli
from ml.config import get_models


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "Qwen38-Flash-Next-Dual-DSpark.sh"
STARTER = ROOT / "scripts" / "start-Qwen38-Flash-Next-Dual-DSpark.sh"
PROFILE = ROOT / "config" / "dspark-qwen38-flash-next-nvfp4.env"


def profile_values() -> dict[str, str]:
    values = {}
    for line in PROFILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    return values


class Qwen38FlashNextRecipeTests(unittest.TestCase):
    def test_profile_preserves_measured_long_context_and_private_bind(self):
        profile = profile_values()

        self.assertEqual(profile["HOST_BIND"], "127.0.0.1")
        self.assertEqual(profile["PORT"], "8888")
        self.assertEqual(profile["DSPARK_PROXY_PORT"], "8000")
        self.assertEqual(profile["MEM_FRACTION_STATIC"], "0.82")
        self.assertEqual(profile["CONTEXT_LENGTH"], "262144")
        self.assertEqual(profile["QWEN38_EFFECTIVE_CONTEXT_LENGTH"], "900000")
        self.assertEqual(profile["CHUNKED_PREFILL_SIZE"], "1024")
        self.assertEqual(profile["MAMBA_FULL_MEMORY_RATIO"], "0.3")
        self.assertEqual(profile["SPEC_STEPS"], "3")
        self.assertEqual(profile["SPEC_TOPK"], "1")
        self.assertEqual(profile["SPEC_DRAFT"], "4")
        self.assertIn("--context-length 900000", profile["EXTRA_ARGS"])
        self.assertIn('"rope_type":"yarn"', profile["EXTRA_ARGS"])

    def test_registry_matches_pinned_sglang_recipe(self):
        model = get_models()["qwen3.8-flash-next-nvfp4-dspark"]

        self.assertEqual(model["serve_backend"], "qwen38-flash-next")
        self.assertEqual(model["nodes"], 2)
        self.assertEqual(model["tensor_parallel_size"], 2)
        self.assertEqual(model["runtime"], "sglang")
        self.assertEqual(model["quantization"], "modelopt_fp4")
        self.assertEqual(model["max_model_len"], 900000)
        self.assertEqual(model["chunked_prefill_size"], 1024)
        self.assertEqual(model["raw_api_url"], "http://127.0.0.1:8888")
        self.assertEqual(model["proxy_bind"], "0.0.0.0:8000")
        self.assertTrue(model["multimodal"])
        self.assertTrue(model["dedicated_cluster"])
        self.assertFalse(model["harness_coexistence"])
        self.assertEqual(
            model["upstream_revision"],
            "dccb035c559f342fe8c0f65eb427671c6cf60730",
        )
        self.assertEqual(
            model["model_revision"],
            "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
        )
        self.assertIn("@sha256:", model["runtime_base_image"])

    def test_lifecycle_enforces_kernel_memory_and_proxy_boundaries(self):
        script = SCRIPT.read_text(encoding="utf-8")

        self.assertIn(
            'UPSTREAM_REVISION_DEFAULT="dccb035c559f342fe8c0f65eb427671c6cf60730"',
            script,
        )
        self.assertIn('[[ "${HOST_BIND}" == "127.0.0.1" ]]', script)
        self.assertIn('[[ "${KERNEL_PATCH}" == "1" ]]', script)
        self.assertIn("--load-format dummy can hard-freeze", script)
        self.assertIn("PLE_OFFLOAD=0 is unsafe", script)
        self.assertIn("run_proxy_cli serve", script)
        self.assertIn("run_proxy_cli smoke", script)
        self.assertIn("Tailscale Funnel targets unauthenticated raw port", script)

    def test_first_run_uses_upstream_combined_image_and_download_action(self):
        starter = STARTER.read_text(encoding="utf-8")

        self.assertIn("qwen38-flash-next setup", starter)
        self.assertIn("qwen38-flash-next download", starter)
        self.assertIn("qwen38-flash-next configure", starter)
        self.assertIn("qwen38-flash-next start", starter)

    def test_shell_entrypoints_are_syntax_valid(self):
        for path in (SCRIPT, STARTER):
            result = subprocess.run(
                ["bash", "-n", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    @patch("ml.cli.subprocess.run")
    def test_cli_dispatches_to_independent_qwen_script(self, run):
        run.return_value = SimpleNamespace(returncode=0)

        result = CliRunner().invoke(cli, ["qwen38-flash-next", "status"])

        self.assertEqual(result.exit_code, 0, result.output)
        command = run.call_args.args[0]
        self.assertTrue(
            command[0].endswith("scripts/Qwen38-Flash-Next-Dual-DSpark.sh")
        )
        self.assertEqual(command[1], "status")


if __name__ == "__main__":
    unittest.main()
