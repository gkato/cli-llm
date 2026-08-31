import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from ml.cli import cli
from ml.config import get_models


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "Qwen38-Flash-Next-vLLM-Dual-DSpark.sh"
STARTER = ROOT / "scripts" / "start-Qwen38-Flash-Next-vLLM-Dual-DSpark.sh"
PATCHER = ROOT / "scripts" / "patch_miaai_qwen38_vllm_launcher.py"
PROFILE = ROOT / "config" / "dspark-qwen38-flash-next-vllm.env"


def profile_values() -> dict[str, str]:
    values = {}
    for line in PROFILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    return values


class Qwen38FlashNextVllmRecipeTests(unittest.TestCase):
    def test_profile_matches_miaai_measured_vllm_settings(self):
        profile = profile_values()

        self.assertEqual(profile["HOST_BIND"], "127.0.0.1")
        self.assertEqual(profile["PORT"], "8888")
        self.assertEqual(profile["DSPARK_PROXY_PORT"], "8000")
        self.assertEqual(profile["TENSOR_PARALLEL_SIZE"], "2")
        self.assertEqual(profile["ENABLE_EXPERT_PARALLEL"], "true")
        self.assertEqual(profile["MTP_NUM_SPECULATIVE_TOKENS"], "3")
        self.assertEqual(profile["MAX_MODEL_LEN"], "1000000")
        self.assertEqual(profile["YARN_ENABLE"], "true")
        self.assertEqual(profile["YARN_FACTOR"], "4.0")
        self.assertEqual(profile["GPU_MEMORY_UTILIZATION"], "0.835")
        self.assertEqual(profile["MAX_NUM_SEQS"], "8")
        self.assertEqual(profile["MAX_NUM_BATCHED_TOKENS"], "8192")
        self.assertEqual(profile["KV_CACHE_DTYPE"], "auto")
        self.assertEqual(profile["PLE_OFFLOAD"], "false")
        self.assertEqual(profile["QWEN38_VLLM_MIN_RUNTIME_AVAILABLE_GIB"], "6")
        self.assertEqual(
            profile["MODEL_REVISION"],
            "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
        )
        self.assertIn("@sha256:", profile["VLLM_IMAGE"])

    def test_registry_has_separate_current_miaai_backend(self):
        model = get_models()["qwen3.8-flash-next-nvfp4-vllm-dspark"]

        self.assertEqual(model["serve_backend"], "qwen38-flash-next-vllm")
        self.assertEqual(model["runtime"], "vllm")
        self.assertEqual(model["distributed_executor_backend"], "mp")
        self.assertTrue(model["expert_parallel"])
        self.assertEqual(model["max_model_len"], 1000000)
        self.assertEqual(model["max_num_seqs"], 8)
        self.assertEqual(model["max_num_batched_tokens"], 8192)
        self.assertEqual(model["gpu_memory_utilization"], 0.835)
        self.assertEqual(model["worker_memory_reserve_gib"], 6)
        self.assertEqual(model["kv_cache_dtype"], "bf16")
        self.assertEqual(model["measured_kv_cache_tokens"], 2481424)
        self.assertEqual(model["speculative_method"], "mtp")
        self.assertEqual(model["speculative_tokens"], 3)
        self.assertEqual(
            model["upstream_revision"],
            "169fbad266f2791335a3102f0d3d625e7c295563",
        )
        self.assertIn("@sha256:", model["runtime_image"])
        self.assertEqual(model["raw_api_url"], "http://127.0.0.1:8888")

    def test_lifecycle_enforces_performance_pins_and_proxy_boundary(self):
        script = SCRIPT.read_text(encoding="utf-8")

        self.assertIn(
            'UPSTREAM_REVISION_DEFAULT="169fbad266f2791335a3102f0d3d625e7c295563"',
            script,
        )
        self.assertIn(
            'MODEL_REVISION_DEFAULT="7b719225242aacd3dbd3f9407468c2ee9a9d2594"',
            script,
        )
        self.assertIn("VLLM_IMAGE_DEFAULT=", script)
        self.assertIn('[[ "${GPU_MEMORY_UTILIZATION}" == "0.835" ]]', script)
        self.assertIn('[[ "${MAX_NUM_BATCHED_TOKENS}" == "8192" ]]', script)
        self.assertIn('[[ "${KV_CACHE_DTYPE}" == "auto" ]]', script)
        self.assertIn('[[ "${PLE_OFFLOAD}" == "false" ]]', script)
        self.assertIn("materialize_launcher", script)
        self.assertIn("resolve_cluster_interfaces", script)
        self.assertIn("verify_model_snapshot", script)
        # Snapshot verification must assert every weight shard, not just
        # config.json, so a partial cache cannot pass and stall the launch.
        self.assertIn("snapshot_shard_files", script)
        self.assertIn("model.safetensors.index.json", script)
        self.assertIn("*.incomplete", script)
        self.assertIn("drop_page_caches", script)
        self.assertIn("check_runtime_memory_headroom", script)
        self.assertIn("run_proxy_cli serve", script)
        self.assertIn("run_proxy_cli smoke", script)
        self.assertIn("Tailscale Funnel targets raw vLLM port", script)

    def test_launcher_overlay_only_adds_pin_and_private_bind_controls(self):
        slash = "\\"
        fixture = """#!/usr/bin/env bash
HF_TOKEN="${HF_TOKEN:-}"
HEAD_HAS=$( [[ -d "$HUB_PATH/models--${ORG}--${NAME}" ]] && echo 1 || echo 0 )
WORKER_HAS=$(ssh_worker "test -d '$REMOTE_HUB/models--${ORG}--${NAME}' && echo 1 || echo 0" 2>/dev/null || echo 0)
HF_HOME="$HF_CACHE_DIR" uvx hf download "$MODEL_ID" --cache-dir "$HUB_PATH"
    elif command -v huggingface-cli &>/dev/null; then
HF_HOME="$HF_CACHE_DIR" huggingface-cli download "$MODEL_ID" --cache-dir "$HUB_PATH"
HF_HOME="$HF_CACHE_DIR" hf download "$MODEL_ID" --cache-dir "$HUB_PATH"
    $MODEL_ID __SLASH__
    --served-model-name $SERVED_MODEL_NAME __SLASH__
    $MODEL_ID __SLASH__
    --served-model-name $SERVED_MODEL_NAME __SLASH__
    --host 0.0.0.0 __SLASH__
""".replace("__SLASH__", slash)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "start.sh"
            destination = Path(tmp) / "start.ml-compute.sh"
            source.write_text(fixture, encoding="utf-8")
            result = subprocess.run(
                [str(PATCHER), str(source), str(destination)],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            patched = destination.read_text(encoding="utf-8")
            self.assertIn("HF_REVISION must pin the model snapshot", patched)
            self.assertEqual(patched.count("--revision $HF_REVISION"), 2)
            self.assertIn("--host $HOST_BIND", patched)
            self.assertNotIn("--host 0.0.0.0", patched)
            self.assertIn("snapshots/$HF_REVISION/config.json", patched)
            self.assertIn(
                "elif command -v huggingface-cli &>/dev/null "
                "&& ! command -v hf &>/dev/null; then",
                patched,
            )

    def test_first_run_stages_every_required_artifact(self):
        starter = STARTER.read_text(encoding="utf-8")

        for action in ("setup", "pull", "download", "configure", "start"):
            self.assertIn(f"qwen38-flash-next-vllm {action}", starter)

    def test_shell_entrypoints_are_valid_and_executable(self):
        for path in (SCRIPT, STARTER):
            result = subprocess.run(
                ["bash", "-n", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(path.stat().st_mode & 0o100)

    def test_path_action_does_not_require_cluster_access(self):
        env = os.environ.copy()
        env["HF_HOME"] = "./data/qwen-vllm-test"
        result = subprocess.run(
            ["bash", str(SCRIPT), "path"],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"hf_home={ROOT}/data/qwen-vllm-test\n", result.stdout)

    @patch("ml.cli.subprocess.run")
    def test_cli_dispatches_to_vllm_recipe(self, run):
        run.return_value = SimpleNamespace(returncode=0)

        result = CliRunner().invoke(cli, ["qwen38-flash-next-vllm", "status"])

        self.assertEqual(result.exit_code, 0, result.output)
        command = run.call_args.args[0]
        self.assertTrue(
            command[0].endswith("scripts/Qwen38-Flash-Next-vLLM-Dual-DSpark.sh")
        )
        self.assertEqual(command[1], "status")


if __name__ == "__main__":
    unittest.main()
