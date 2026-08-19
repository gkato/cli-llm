import unittest
from unittest.mock import patch

from click.testing import CliRunner

from ml.cli import cli


class VisionCliTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @patch("ml.vision_server.start")
    def test_serve(self, start):
        start.return_value = {
            "pid": 123,
            "served_model_name": "pp-ocrv6-medium-det",
            "port": 8002,
        }
        result = self.runner.invoke(
            cli, ["vision", "serve", "pp-ocrv6-medium-det"]
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Started Transformers vision server", result.output)
        self.assertIn("/v1/text/detections", result.output)
        start.assert_called_once_with(
            "pp-ocrv6-medium-det", port=None, foreground=False
        )

    @patch("ml.vision_server.status")
    def test_status(self, status):
        status.return_value = {
            "running": True,
            "ready": True,
            "served_model_name": "pp-ocrv6-medium-det",
            "pid": 123,
            "port": 8002,
            "started_at": "now",
            "log_path": "/tmp/vision.log",
        }
        result = self.runner.invoke(cli, ["vision", "status"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("ready", result.output)

    @patch("ml.vision_server.stop", return_value=True)
    def test_stop(self, stop):
        result = self.runner.invoke(cli, ["vision", "stop"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Stopped Transformers vision server", result.output)
        stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
