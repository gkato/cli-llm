import unittest
from unittest.mock import patch

from click.testing import CliRunner

from ml.cli import cli


class RouterCliTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @patch("ml.router_server.start")
    def test_serve(self, start):
        start.return_value = {
            "pid": 123,
            "host": "0.0.0.0",
            "port": 8000,
        }
        result = self.runner.invoke(cli, ["router", "serve"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Started model router", result.output)
        self.assertIn("registry/router.yaml", result.output)
        start.assert_called_once_with(port=None, foreground=False)

    @patch("ml.router_server.status")
    def test_status(self, status):
        status.return_value = {
            "running": True,
            "ready": False,
            "pid": 123,
            "port": 8000,
            "log_path": "/tmp/router.log",
            "backends": {
                "qwen": {"ready": True},
                "unlimited_ocr": {"ready": False},
            },
        }
        result = self.runner.invoke(cli, ["router", "status"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("degraded", result.output)
        self.assertIn("qwen: ready", result.output)
        self.assertIn("unlimited_ocr: unavailable", result.output)

    @patch("ml.router_server.stop", return_value=True)
    def test_stop(self, stop):
        result = self.runner.invoke(cli, ["router", "stop"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Stopped model router", result.output)
        stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
