import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from ml.cli import cli
from ml.config import get_models


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "GLM53-Flash-Dual-DSpark.sh"
STARTER = ROOT / "scripts" / "start-GLM53-Flash-Dual-DSpark.sh"
PROFILE = ROOT / "config" / "dspark-glm53-flash-nvfp4.env"


def profile_values() -> dict[str, str]:
    values = {}
    for line in PROFILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    return values


class GLM53FlashRecipeTests(unittest.TestCase):
    def test_profile_matches_miaai_sm121_recipe_and_is_private(self):
        profile = profile_values()

        self.assertEqual(profile["TENSOR_PARALLEL_SIZE"], "2")
        self.assertEqual(profile["HOST_BIND"], "127.0.0.1")
        self.assertEqual(profile["PORT"], "8888")
        self.assertEqual(profile["DSPARK_PROXY_PORT"], "8000")
        self.assertEqual(profile["MAX_MODEL_LEN"], "262144")
        self.assertEqual(profile["MAX_NUM_SEQS"], "8")
        self.assertEqual(profile["GPU_MEMORY_UTILIZATION"], "0.84")
        self.assertEqual(profile["BLOCK_SIZE"], "2304")
        self.assertEqual(profile["KV_CACHE_DTYPE"], "fp8_e4m3")
        self.assertEqual(profile["MOE_BACKEND"], "marlin")
        self.assertEqual(profile["ENFORCE_EAGER"], "1")
        self.assertEqual(profile["SKIP_MM_PROFILING"], "1")
        self.assertEqual(profile["MTP_SPECULATIVE_TOKENS"], "4")
        self.assertEqual(profile["RAY_VERSION"], "2.58.0")
        self.assertEqual(profile["RAY_OBJECT_STORE_MEMORY"], "4294967296")
        self.assertEqual(
            profile["VLLM_IMAGE"],
            "ml-compute/glm53-flash-sm121:mm-ray-v1-aed98a1",
        )
        self.assertNotIn("VLLM_USE_RAY_V2_EXECUTOR_BACKEND", profile)
        self.assertIn("@sha256:", profile["VLLM_BASE_IMAGE"])
        self.assertEqual(profile["GLM53_MIN_AVAILABLE_GIB"], "112")
        self.assertEqual(profile["GLM53_MIN_DISK_GIB"], "240")

    def test_registry_matches_pinned_miaai_recipe(self):
        model = get_models()["glm-5.3-flash-nvfp4-dspark"]

        self.assertEqual(model["serve_backend"], "glm53-flash")
        self.assertEqual(model["nodes"], 2)
        self.assertEqual(model["tensor_parallel_size"], 2)
        self.assertEqual(model["distributed_executor_backend"], "ray")
        self.assertEqual(model["runtime"], "vllm")
        self.assertEqual(model["max_model_len"], 262144)
        self.assertEqual(model["max_num_seqs"], 8)
        self.assertEqual(model["checkpoint_size_gib"], 181)
        self.assertEqual(model["moe_backend"], "marlin")
        self.assertTrue(model["enforce_eager"])
        self.assertFalse(model["experimental"])
        self.assertTrue(model["verified_on_gb10"])
        self.assertEqual(
            model["upstream_revision"],
            "aed98a13ca75140d2691cc5c651ea5817d9a3e44",
        )
        self.assertEqual(
            model["model_revision"],
            "11d73216cd636238e82e1d77fe1042ffab36e7fa",
        )
        self.assertEqual(model["ray_version"], "2.58.0")
        self.assertEqual(model["ray_object_store_bytes"], 4294967296)
        self.assertEqual(
            model["runtime_image"],
            "ml-compute/glm53-flash-sm121:mm-ray-v1-aed98a1",
        )
        self.assertEqual(model["runtime_kernel_patch"], "sm121-sm90-nope-fa2")
        self.assertFalse(model["ray_executor_v2"])
        self.assertIn("@sha256:", model["runtime_base_image"])

    def test_lifecycle_pins_upstream_kernel_and_proxy_boundary(self):
        script = SCRIPT.read_text(encoding="utf-8")

        self.assertIn(
            'UPSTREAM_REVISION_DEFAULT="aed98a13ca75140d2691cc5c651ea5817d9a3e44"',
            script,
        )
        self.assertIn(
            'MODEL_REVISION_DEFAULT="11d73216cd636238e82e1d77fe1042ffab36e7fa"',
            script,
        )
        self.assertIn("sync_recipe_pin", script)
        self.assertIn("files/Dockerfile.mm-ray", script)
        self.assertIn("org.ml-compute.upstream.revision", script)
        self.assertIn("RAY_OBJECT_STORE_MEMORY", script)
        self.assertIn("SKIP_MM_PROFILING", script)
        self.assertIn("sm121_nope_patch=1", script)
        self.assertIn('[[ "${HOST_BIND}" == "127.0.0.1" ]]', script)
        self.assertIn('[[ "${MOE_BACKEND}" == "marlin" ]]', script)
        self.assertIn("--host ${HOST_BIND}", script)
        self.assertIn("run_proxy_cli serve", script)
        self.assertIn("run_proxy_cli smoke", script)
        self.assertIn("Tailscale Funnel targets unauthenticated raw port", script)
        self.assertIn("show_failure_diagnostics", script)
        self.assertIn("/tmp/ray/session_latest/logs", script)
        self.assertNotIn("VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1", script)

    def test_first_run_stages_every_required_artifact(self):
        starter = STARTER.read_text(encoding="utf-8")

        for action in ("setup", "pull", "download", "gpu-check"):
            self.assertIn(f"glm53-flash {action}", starter)
        self.assertIn("glm53-flash configure", starter)
        self.assertIn("glm53-flash start", starter)

    def test_shell_entrypoints_are_syntax_valid_and_executable(self):
        for path in (SCRIPT, STARTER):
            result = subprocess.run(
                ["bash", "-n", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(path.stat().st_mode & 0o100)

    @patch("ml.cli.subprocess.run")
    def test_cli_dispatches_to_glm_recipe(self, run):
        run.return_value = SimpleNamespace(returncode=0)

        result = CliRunner().invoke(cli, ["glm53-flash", "status"])

        self.assertEqual(result.exit_code, 0, result.output)
        command = run.call_args.args[0]
        self.assertTrue(command[0].endswith("scripts/GLM53-Flash-Dual-DSpark.sh"))
        self.assertEqual(command[1], "status")


if __name__ == "__main__":
    unittest.main()
