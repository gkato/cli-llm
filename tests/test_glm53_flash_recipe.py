import os
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
    def test_profile_matches_miaai_exl3_recipe_and_is_private(self):
        profile = profile_values()

        self.assertEqual(profile["TENSOR_PARALLEL_SIZE"], "2")
        self.assertEqual(profile["NUM_NODES"], "2")
        self.assertEqual(profile["DISTRIBUTED_EXECUTOR_BACKEND"], "mp")
        self.assertEqual(profile["HOST_BIND"], "127.0.0.1")
        self.assertEqual(profile["PORT"], "8888")
        self.assertEqual(profile["DSPARK_PROXY_PORT"], "8000")
        self.assertEqual(profile["MODEL_ID"], "brandonmusic/GLM-5.3-Flash-tr3-4bpw")
        self.assertEqual(profile["QUANTIZATION"], "exl3")
        self.assertEqual(profile["MAX_MODEL_LEN"], "1000000")
        self.assertEqual(profile["MAX_NUM_SEQS"], "4")
        self.assertEqual(profile["MAX_NUM_BATCHED_TOKENS"], "1024")
        self.assertEqual(profile["GPU_MEMORY_UTILIZATION"], "0.87")
        self.assertEqual(profile["KV_CACHE_DTYPE"], "fp8")
        self.assertEqual(profile["ENFORCE_EAGER"], "0")
        self.assertEqual(profile["EXL3_FUSED_MOE"], "1")
        self.assertEqual(profile["ENABLE_PREFIX_CACHING"], "1")
        self.assertEqual(profile["GLM53_SUPPRESS_STOPS_IN_REASONING"], "1")
        self.assertEqual(profile["GLM53_MIXED_PREFILL_CHUNK"], "skip")
        self.assertEqual(profile["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"], "1800")
        self.assertEqual(profile["GLM53_BOOT_SHAPE_WARMUP"], "1")
        self.assertEqual(profile["GLM53_WARMUP_REQ_TIMEOUT"], "240")
        self.assertEqual(profile["CG_ESTIMATE"], "1")
        self.assertEqual(profile["SKIP_MM_PROFILING"], "1")
        self.assertEqual(profile["SPEC_METHOD"], "dflash")
        self.assertEqual(profile["DFLASH_SPECULATIVE_TOKENS"], "7")
        self.assertEqual(profile["DFLASH_DRAFT_TP"], "1")
        self.assertEqual(profile["MTP_SPECULATIVE_TOKENS"], "2")
        self.assertEqual(profile["USE_HOST_NCCL"], "0")
        self.assertEqual(profile["WORKER_CX7_IF"], "enp1s0f1np1")
        self.assertEqual(profile["WORKER_CX7_IB"], "rocep1s0f1")
        self.assertEqual(
            profile["VLLM_IMAGE"],
            "ml-compute/glm53-flash-exl3:mp-dflash2-v3-0e2e78f",
        )
        self.assertIn("@sha256:", profile["VLLM_SOURCE_IMAGE"])
        self.assertIn("@sha256:", profile["VLLM_BASE_IMAGE"])
        self.assertNotIn("RAY_VERSION", profile)
        self.assertNotIn("RAY_OBJECT_STORE_MEMORY", profile)
        self.assertEqual(profile["GLM53_MIN_AVAILABLE_GIB"], "112")
        self.assertEqual(profile["GLM53_MIN_DISK_GIB"], "220")

    def test_registry_matches_pinned_miaai_recipe(self):
        model = get_models()["glm-5.3-flash-nvfp4-dspark"]

        self.assertEqual(model["serve_backend"], "glm53-flash")
        self.assertEqual(model["nodes"], 2)
        self.assertEqual(model["tensor_parallel_size"], 2)
        self.assertEqual(model["distributed_executor_backend"], "mp")
        self.assertEqual(model["runtime"], "vllm")
        self.assertEqual(model["hf_id"], "brandonmusic/GLM-5.3-Flash-tr3-4bpw")
        self.assertEqual(model["max_model_len"], 1000000)
        self.assertEqual(model["max_num_seqs"], 4)
        self.assertEqual(model["max_num_batched_tokens"], 1024)
        self.assertEqual(model["checkpoint_size_gib"], 164)
        self.assertEqual(model["gpu_memory_utilization"], 0.87)
        self.assertEqual(model["quantization"], "exl3")
        self.assertEqual(model["moe_backend"], "exl3_fused")
        self.assertFalse(model["enforce_eager"])
        self.assertTrue(model["cuda_graphs"])
        self.assertFalse(model["experimental"])
        self.assertTrue(model["verified_on_gb10"])
        self.assertEqual(
            model["upstream_revision"],
            "0e2e78f3de83624e6733b918724da27fc9040156",
        )
        self.assertEqual(
            model["model_revision"],
            "5ab363a8dcf6405955fd5f99671e01a1c9fb124b",
        )
        self.assertEqual(
            model["runtime_image"],
            "ml-compute/glm53-flash-exl3:mp-dflash2-v3-0e2e78f",
        )
        self.assertIn("@sha256:", model["runtime_source_image"])
        self.assertEqual(
            model["runtime_kernel_patch"],
            "exl3-sm121-dflash2-slotshare-apc-xgrammar-overlay",
        )
        self.assertEqual(model["dflash_kv_slot_sharing"], "padded_mla")
        self.assertEqual(model["mixed_prefill_policy"], "skip")
        self.assertTrue(model["boot_shape_warmup"])
        self.assertTrue(model["suppress_stops_during_reasoning"])
        self.assertTrue(model["cuda_graph_estimate_deduction"])
        self.assertTrue(model["xgrammar_speculative_termination_patch"])
        self.assertTrue(model["worker_startup_fail_fast"])
        self.assertEqual(
            model["worker_cuda_device_passthrough"],
            "explicit_nonprivileged",
        )
        self.assertEqual(model["speculative_method"], "dflash")
        self.assertEqual(model["speculative_tokens"], 7)
        self.assertEqual(model["speculative_draft_tensor_parallel_size"], 1)
        self.assertEqual(model["speculative_draft_license"], "CC-BY-NC-ND-4.0")
        self.assertEqual(model["commercial_speculative_method"], "mtp")
        self.assertIn("@sha256:", model["runtime_base_image"])

    def test_lifecycle_pins_upstream_kernel_and_proxy_boundary(self):
        script = SCRIPT.read_text(encoding="utf-8")

        self.assertIn(
            'UPSTREAM_REVISION_DEFAULT="0e2e78f3de83624e6733b918724da27fc9040156"',
            script,
        )
        self.assertIn(
            'MODEL_REVISION_DEFAULT="5ab363a8dcf6405955fd5f99671e01a1c9fb124b"',
            script,
        )
        self.assertIn(
            'DFLASH_MODEL_REVISION_DEFAULT="7d74cdd881ed7e32c31175984a67823127b66cfe"',
            script,
        )
        self.assertIn("sync_recipe_pin", script)
        self.assertIn("VLLM_SOURCE_IMAGE_DEFAULT", script)
        self.assertIn("verify_target_snapshot", script)
        self.assertIn("verify_dflash_snapshot", script)
        self.assertIn("SKIP_MM_PROFILING", script)
        self.assertIn("GLM53_SUPPRESS_STOPS_IN_REASONING", script)
        self.assertIn("GLM53_MIXED_PREFILL_CHUNK", script)
        self.assertIn("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", script)
        self.assertIn("GLM53_BOOT_SHAPE_WARMUP", script)
        self.assertIn("GLM53_WARMUP_REQ_TIMEOUT", script)
        self.assertIn("CG_ESTIMATE", script)
        self.assertIn("printf 'VLLM_API_KEY=\\n'", script)
        self.assertIn("--runtime=nvidia --device=/dev/nvidia0", script)
        self.assertIn("worker_devices != 1", script)
        self.assertIn("--device=/dev/nvidia-uvm-tools", script)
        self.assertIn("Retagging the matching immutable image", script)
        self.assertIn("materialize_upstream_launcher", script)
        self.assertIn("resolve_cluster_interfaces", script)
        self.assertIn("local_netdev_for_ip", script)
        self.assertIn("worker_hca_for_netdev", script)
        self.assertIn("HOST_BIND:-127.0.0.1", script)
        self.assertIn("host != 2 || endpoint != 1", script)
        self.assertIn('[[ "${HOST_BIND}" == "127.0.0.1" ]]', script)
        self.assertIn('[[ "${DISTRIBUTED_EXECUTOR_BACKEND}" == "mp" ]]', script)
        self.assertIn('[[ "${QUANTIZATION}" == "exl3" ]]', script)
        self.assertIn('if [[ "${SPEC_METHOD}" == "dflash" ]]', script)
        self.assertIn("DFlash2 is CC BY-NC-ND 4.0", script)
        self.assertIn("run_proxy_cli serve", script)
        self.assertIn("run_proxy_cli smoke", script)
        self.assertIn("Tailscale Funnel targets unauthenticated raw port", script)
        self.assertIn("show_failure_diagnostics", script)
        self.assertIn("Watching EngineCore for 20 seconds", script)
        self.assertIn("Post-start GLM completion failed", script)
        self.assertNotIn("RAY_OBJECT_STORE_MEMORY", script)

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

    def test_relative_hf_home_is_rooted_before_upstream_directory_change(self):
        env = os.environ.copy()
        env["HF_HOME"] = "./data/hf_cache"

        result = subprocess.run(
            ["bash", str(SCRIPT), "path"],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"hf_home={ROOT}/data/hf_cache\n", result.stdout)

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
