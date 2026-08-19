import sys
import unittest
from unittest.mock import Mock, patch

from ml import vision_server


class VisionServerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "hf_id": "PaddlePaddle/PP-OCRv6_medium_det_safetensors",
            "served_model_name": "pp-ocrv6-medium-det",
            "serve_backend": "vision",
            "default_port": 8002,
            "vision_config": {
                "device": "cuda:0",
                "dtype": "float16",
                "max_concurrency": 1,
                "threshold": 0.2,
                "box_threshold": 0.45,
                "max_candidates": 3000,
                "unclip_ratio": 1.4,
            },
        }

    def test_build_cmd_uses_registry_inference_settings(self):
        cmd = vision_server._build_cmd("pp-ocrv6-medium-det", self.cfg, 8123)

        self.assertEqual(cmd[:3], [sys.executable, "-m", "ml.vision_api"])
        self.assertEqual(cmd[cmd.index("--model-id") + 1], self.cfg["hf_id"])
        self.assertEqual(cmd[cmd.index("--port") + 1], "8123")
        self.assertEqual(cmd[cmd.index("--dtype") + 1], "float16")
        self.assertEqual(cmd[cmd.index("--max-concurrency") + 1], "1")
        self.assertEqual(cmd[cmd.index("--box-threshold") + 1], "0.45")

    @patch("ml.vision_server.get_models")
    def test_resolve_model_rejects_non_vision_backend(self, get_models):
        get_models.return_value = {"text-model": {"serve_backend": "vllm"}}
        with self.assertRaisesRegex(ValueError, "not Transformers vision"):
            vision_server.resolve_model("text-model")

    @patch("ml.vision_server.requests.get")
    @patch("ml.vision_server.get_api_key", return_value="secret")
    @patch("ml.vision_server._running_state")
    def test_status_checks_authenticated_readiness(
        self, running_state, _get_api_key, get
    ):
        running_state.return_value = {
            "pid": 123,
            "alias": "pp-ocrv6-medium-det",
            "hf_id": self.cfg["hf_id"],
            "served_model_name": "pp-ocrv6-medium-det",
            "port": 8002,
            "log_path": "/tmp/vision.log",
            "started_at": "2026-08-18T00:00:00+00:00",
        }
        get.return_value = Mock(ok=True)

        result = vision_server.status()

        self.assertTrue(result["running"])
        self.assertTrue(result["ready"])
        get.assert_called_once_with(
            "http://localhost:8002/v1/models",
            headers={"Authorization": "Bearer secret"},
            timeout=2,
        )


if __name__ == "__main__":
    unittest.main()
