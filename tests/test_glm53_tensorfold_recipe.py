import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from ml.cli import GLM53_FLASH_TENSORFOLD_ACTIONS, cli
from ml.config import get_models


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config" / "dspark-glm53-flash-tensorfold.env"
SCRIPT = ROOT / "scripts" / "GLM53-Flash-TensorFold-Dual-DSpark.sh"


def profile_values() -> dict[str, str]:
    values = {}
    for line in PROFILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    return values


class GLM53TensorFoldRecipeTests(unittest.TestCase):
    def test_profile_pins_reviewed_upstream_artifacts(self):
        profile = profile_values()

        self.assertEqual(
            profile["TENSORFOLD_REF"],
            "3fd60dfc836fab0f1ed0fd4f0afaf2a1d62b2aab",
        )
        self.assertEqual(
            profile["TENSORFOLD_MODEL_ID"],
            "Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold",
        )
        self.assertEqual(
            profile["TENSORFOLD_MODEL_REVISION"],
            "078455ffe6472f9a52fbc1139f58b9db2881b25c",
        )
        self.assertEqual(profile["TENSORFOLD_IMAGE_TAG"], "v0.6.0-e66c657a433d")
        self.assertEqual(profile["TENSORFOLD_DENSE"], "fp8")
        self.assertEqual(profile["TENSORFOLD_MULTI_PREFILL"], "1")
        self.assertEqual(profile["TENSORFOLD_SHARED_PREFIX"], "1")
        self.assertEqual(profile["TENSORFOLD_FILL_ROWS"], "2048")
        self.assertEqual(profile["TENSORFOLD_HOST"], "127.0.0.1")
        self.assertEqual(profile["DSPARK_PROXY_PORT"], "8000")

    def test_registry_matches_pinned_tensorfold_recipe(self):
        model = get_models()["glm-5.3-flash-exl3-tensorfold-dspark"]

        self.assertEqual(model["serve_backend"], "glm53-flash-tensorfold")
        self.assertEqual(model["upstream_revision"], "3fd60dfc836fab0f1ed0fd4f0afaf2a1d62b2aab")
        self.assertEqual(model["model_revision"], "078455ffe6472f9a52fbc1139f58b9db2881b25c")
        self.assertEqual(model["runtime"], "tensorfold")
        self.assertEqual(model["raw_api_url"], "http://127.0.0.1:8888")

    def test_adapter_patches_the_known_worker_cuda_compatibility_issue(self):
        script = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("patch_nvidia_runtime", script)
        self.assertIn("--runtime=nvidia", script)
        self.assertIn("NVIDIA_VISIBLE_DEVICES=all", script)
        self.assertIn("NVIDIA_DRIVER_CAPABILITIES=compute,utility", script)
        self.assertIn("generated_patch_only", script)
        self.assertIn("git -C \"$RECIPE\" diff --quiet", script)
        self.assertIn("MULTI_PREFILL=${TENSORFOLD_MULTI_PREFILL}", script)
        self.assertIn("FILL_ROWS=${TENSORFOLD_FILL_ROWS}", script)
        self.assertIn("chmod 0755 \"$tmp\"", script)
        self.assertIn("HOST=${TENSORFOLD_HOST}", script)
        self.assertIn("proxy serve", script)
        self.assertIn("update", GLM53_FLASH_TENSORFOLD_ACTIONS)

        result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @patch("ml.cli.subprocess.run")
    def test_cli_dispatches_to_recipe(self, run):
        run.return_value = SimpleNamespace(returncode=0)
        result = CliRunner().invoke(cli, ["glm53-flash-tensorfold", "status"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(run.call_args.args[0][0].endswith("GLM53-Flash-TensorFold-Dual-DSpark.sh"))


if __name__ == "__main__":
    unittest.main()
