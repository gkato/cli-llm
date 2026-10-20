import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from ml.cli import DEEPSEEK_V41_TENSORFOLD_ACTIONS, cli
from ml.config import get_models


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config" / "dspark-deepseek-v41-flash-tensorfold.env"
SCRIPT = ROOT / "scripts" / "DeepSeek-V41-Flash-TensorFold-Dual-DSpark.sh"


def profile_values() -> dict[str, str]:
    values = {}
    for line in PROFILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    return values


class DeepSeekV41TensorFoldRecipeTests(unittest.TestCase):
    def test_profile_pins_all_upstream_build_inputs(self):
        profile = profile_values()

        self.assertEqual(
            profile["DSV41_TF_REF"],
            "31eea4608f8bb292a92e2f7a1db400e928620c78",
        )
        self.assertEqual(
            profile["DSV41_TF_MODEL_REVISION"],
            "45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b",
        )
        self.assertEqual(
            profile["DSV41_TF_ENGRAM_REVISION"],
            "1bae0e1088993b9b82de08f59de5be5a34e42170",
        )
        self.assertEqual(
            profile["DSV41_TF_ENGINE_REF"],
            "629c524a0ea57512a5342abb498d1b7c25353343",
        )
        self.assertEqual(
            profile["DSV41_TF_INPUTS_REF"],
            "59402086381a42c74d53053443224eec2790b4c5",
        )
        self.assertEqual(
            profile["DSV41_TF_KIT_REF"],
            "abe07c7d509a0199cc7fec15263be2aa52ac652f",
        )
        self.assertIn("@sha256:", profile["DSV41_TF_BASE_IMAGE"])
        self.assertEqual(profile["DSV41_TF_CONTEXT"], "524288")
        self.assertEqual(profile["DSV41_TF_PARALLEL"], "4")
        self.assertEqual(profile["DSV41_TF_HOST"], "127.0.0.1")
        self.assertEqual(profile["DSPARK_PROXY_PORT"], "8000")
        self.assertEqual(profile["DSV41_TF_ABLIT"], "0")
        self.assertEqual(profile["DSV41_TF_PULL"], "0")
        self.assertEqual(profile["DSV41_TF_PREPARE_HF_HUB_OFFLINE"], "0")
        self.assertEqual(profile["DSV41_TF_HF_HUB_OFFLINE"], "1")

    def test_registry_describes_the_separate_tensorfold_backend(self):
        model = get_models()["deepseek-v4.1-flash-1017-exl3-tensorfold-dspark"]

        self.assertEqual(model["serve_backend"], "deepseek-v41-flash-tensorfold")
        self.assertEqual(model["runtime"], "tensorfold-native")
        self.assertEqual(model["upstream_revision"], profile_values()["DSV41_TF_REF"])
        self.assertEqual(model["model_revision"], profile_values()["DSV41_TF_MODEL_REVISION"])
        self.assertEqual(model["engram_revision"], profile_values()["DSV41_TF_ENGRAM_REVISION"])
        self.assertEqual(model["max_model_len"], 524288)
        self.assertEqual(model["native_context_len"], 1048576)
        self.assertEqual(model["max_num_seqs"], 4)
        self.assertEqual(model["raw_api_url"], "http://127.0.0.1:8888")
        self.assertTrue(model["dedicated_cluster"])

    def test_lifecycle_preserves_proxy_and_update_safety(self):
        script = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("git -C \"$RECIPE\" diff --quiet", script)
        self.assertIn("git -C \"$RECIPE\" diff --cached --quiet", script)
        self.assertIn("env -u API_KEY", script)
        self.assertIn("HOST=${DSV41_TF_HOST}", script)
        self.assertIn("API_KEY=0", script)
        self.assertIn("proxy serve", script)
        self.assertIn("proxy smoke", script)
        self.assertIn("NFS_PATH=${DSV41_TF_NFS_PATH}", script)
        self.assertIn(
            'run_recipe env HF_HUB_OFFLINE="$DSV41_TF_PREPARE_HF_HUB_OFFLINE"',
            script,
        )
        for action in ("setup", "prepare", "dry-run", "start", "update", "smoke"):
            self.assertIn(action, DEEPSEEK_V41_TENSORFOLD_ACTIONS)

        result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @patch("ml.cli.subprocess.run")
    def test_cli_dispatches_to_recipe(self, run):
        run.return_value = SimpleNamespace(returncode=0)
        result = CliRunner().invoke(cli, ["deepseek-v41-flash-tensorfold", "status"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(
            run.call_args.args[0][0].endswith(
                "DeepSeek-V41-Flash-TensorFold-Dual-DSpark.sh"
            )
        )


if __name__ == "__main__":
    unittest.main()
