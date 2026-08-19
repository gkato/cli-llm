import os
import unittest
from unittest.mock import Mock, patch

from ml import docker_server


class DockerServerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "hf_id": "baidu/Unlimited-OCR",
            "served_model_name": "baidu/Unlimited-OCR",
            "serve_backend": "docker",
            "docker_image": "vllm/vllm-openai:unlimited-ocr-arm64-cu130",
            "docker_env": {"VLLM_FLASHINFER_FORCE_TARGET": "sm_121"},
            "max_model_len": 32768,
            "max_num_seqs": 1,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.2,
            "extra_args": [
                "--logits_processors",
                "vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor",
            ],
        }

    def test_build_run_cmd_places_docker_and_vllm_args_correctly(self):
        with patch.dict(os.environ, {}, clear=True):
            cmd = docker_server._build_run_cmd(
                "unlimited-ocr",
                self.cfg["docker_image"],
                self.cfg,
                port=8123,
                foreground=False,
                gpu_count=None,
                cache_dir="/tmp/ml-compute-test-cache",
                api_key="secret",
            )

        image_index = cmd.index(self.cfg["docker_image"])
        self.assertEqual(cmd[:3], ["docker", "run", "-d"])
        self.assertIn("--network", cmd[:image_index])
        self.assertIn("VLLM_FLASHINFER_FORCE_TARGET=sm_121", cmd[:image_index])
        self.assertEqual(cmd[image_index + 1], "baidu/Unlimited-OCR")
        self.assertIn("--max-model-len", cmd[image_index:])
        self.assertIn("--logits_processors", cmd[image_index:])
        self.assertEqual(cmd[-2:], ["--api-key", "secret"])

    def test_build_run_cmd_rejects_zero_gpus(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            docker_server._build_run_cmd(
                "unlimited-ocr",
                self.cfg["docker_image"],
                self.cfg,
                port=8000,
                foreground=False,
                gpu_count=0,
                cache_dir="/tmp/ml-compute-test-cache",
                api_key=None,
            )

    def test_co_residency_requires_opt_in_and_distinct_port(self):
        local = {"port": 8000, "backend": "vllm"}
        with self.assertRaisesRegex(RuntimeError, "allow-co-resident"):
            docker_server._validate_local_co_residency(local, 8001, False)
        with self.assertRaisesRegex(RuntimeError, "Port 8000"):
            docker_server._validate_local_co_residency(local, 8000, True)
        docker_server._validate_local_co_residency(local, 8001, True)

    @patch("ml.docker_server.requests.get")
    @patch("ml.docker_server.get_api_key", return_value="secret")
    @patch("ml.docker_server._container_exists", return_value=True)
    @patch("ml.docker_server._container_running", return_value=True)
    @patch("ml.docker_server._read_state")
    def test_status_checks_authenticated_readiness(
        self,
        read_state,
        _container_running,
        _container_exists,
        _get_api_key,
        get,
    ):
        read_state.return_value = {
            "container_id": "abcdef1234567890",
            "alias": "unlimited-ocr",
            "image": self.cfg["docker_image"],
            "hf_id": self.cfg["hf_id"],
            "served_model_name": self.cfg["served_model_name"],
            "port": 8000,
            "started_at": "2026-08-17T00:00:00+00:00",
        }
        get.return_value = Mock(ok=True)

        result = docker_server.status()

        self.assertTrue(result["running"])
        self.assertTrue(result["ready"])
        get.assert_called_once_with(
            "http://localhost:8000/v1/models",
            headers={"Authorization": "Bearer secret"},
            timeout=2,
        )


if __name__ == "__main__":
    unittest.main()
