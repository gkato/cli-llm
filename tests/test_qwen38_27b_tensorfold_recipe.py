import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from ml.cli import QWEN38_27B_TENSORFOLD_ACTIONS, cli
from ml.config import get_models


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config" / "dspark-qwen38-27b-tensorfold.env"
SCRIPT = ROOT / "scripts" / "Qwen38-27B-TensorFold-DSpark.sh"


def profile_values() -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in PROFILE.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )


class Qwen38TensorFoldRecipeTests(unittest.TestCase):
    def test_profile_pins_upstream_model_drafter_and_image(self):
        profile = profile_values()

        self.assertEqual(
            profile["QWEN38_TF_REF"],
            "c11ff138871cf43d12898d79dbfa4dc952d17ca4",
        )
        self.assertEqual(
            profile["QWEN38_TF_MODEL_REVISION"],
            "22d8d538154e0e5b6f8dcb1fcc74a60a0798104c",
        )
        self.assertEqual(
            profile["QWEN38_TF_DRAFT_REVISION"],
            "50307d4c4cde6860d4eee73e2547cd786fe8e8a4",
        )
        self.assertEqual(profile["QWEN38_TF_IMAGE_TAG"], "v0.6.0-fb91a0019fdc")
        self.assertTrue(profile["QWEN38_TF_IMAGE_DIGEST"].startswith("sha256:"))
        self.assertEqual(profile["QWEN38_TF_HOST"], "127.0.0.1")
        self.assertEqual(profile["DSPARK_PROXY_PORT"], "8000")
        self.assertEqual(profile["QWEN38_TF_PARALLEL"], "8")
        self.assertEqual(profile["QWEN38_TF_CONTEXT"], "262144")
        self.assertEqual(profile["QWEN38_TF_KV_DTYPE"], "fp8")
        self.assertEqual(profile["QWEN38_TF_KV_POOL_GB"], "auto")

    def test_registry_matches_pinned_recipe(self):
        model = get_models()["qwen3.8-27b-mlx-tensorfold-dspark"]

        self.assertEqual(model["serve_backend"], "qwen38-27b-tensorfold")
        self.assertEqual(model["runtime"], "tensorfold")
        self.assertEqual(model["nodes"], 1)
        self.assertEqual(model["max_model_len"], 262144)
        self.assertEqual(model["max_num_seqs"], 8)
        self.assertEqual(model["raw_api_url"], "http://127.0.0.1:8888")
        self.assertEqual(
            model["upstream_revision"],
            "c11ff138871cf43d12898d79dbfa4dc952d17ca4",
        )

    def test_adapter_is_syntax_valid_and_enforces_pins(self):
        result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("ensure_pinned_image", script)
        self.assertIn("download_snapshot", script)
        self.assertIn("set_snapshot_ref", script)
        self.assertIn("QWEN38_TF_IMAGE_DIGEST", script)
        self.assertIn("HOST=${QWEN38_TF_HOST}", script)
        self.assertIn("proxy serve", script)
        self.assertIn("update", QWEN38_27B_TENSORFOLD_ACTIONS)

    @patch("ml.cli.subprocess.run")
    def test_cli_dispatches_to_recipe(self, run):
        run.return_value = SimpleNamespace(returncode=0)
        result = CliRunner().invoke(cli, ["qwen38-27b-tensorfold", "status"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(run.call_args.args[0][0].endswith("Qwen38-27B-TensorFold-DSpark.sh"))


if __name__ == "__main__":
    unittest.main()
