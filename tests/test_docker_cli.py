import unittest
from unittest.mock import patch

from click.testing import CliRunner

from ml.cli import cli


class DockerCliTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @patch("ml.docker_server.start")
    def test_serve(self, start):
        start.return_value = {
            "container_name": "ml-compute-vllm-docker",
            "container_id": "abcdef1234567890",
            "image": "vllm/image:tag",
            "served_model_name": "baidu/Unlimited-OCR",
            "port": 8000,
        }
        result = self.runner.invoke(cli, ["docker", "serve", "unlimited-ocr"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Started Docker vLLM container", result.output)
        start.assert_called_once_with(
            "unlimited-ocr", port=None, foreground=False, gpu_count=None
        )

    @patch("ml.docker_server.status")
    def test_status(self, status):
        status.return_value = {
            "running": True,
            "ready": True,
            "container_name": "ml-compute-vllm-docker",
            "container_id": "abcdef1234567890",
            "alias": "unlimited-ocr",
            "image": "vllm/image:tag",
            "served_model_name": "baidu/Unlimited-OCR",
            "port": 8000,
            "started_at": "now",
        }
        result = self.runner.invoke(cli, ["docker", "status"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("ready", result.output)

    @patch("ml.docker_server.stop", return_value=True)
    def test_stop(self, stop):
        result = self.runner.invoke(cli, ["docker", "stop"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Stopped and removed", result.output)
        stop.assert_called_once_with()

    @patch("ml.docker_server.tail_logs")
    def test_logs(self, tail_logs):
        result = self.runner.invoke(cli, ["docker", "logs", "-f", "-n", "25"])

        self.assertEqual(result.exit_code, 0, result.output)
        tail_logs.assert_called_once_with(lines=25, follow=True)


if __name__ == "__main__":
    unittest.main()
