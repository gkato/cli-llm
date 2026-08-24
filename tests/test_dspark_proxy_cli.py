import unittest
from unittest.mock import patch

from click.testing import CliRunner

from ml.cli import cli


class DSparkProxyCliTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @patch("ml.dspark_proxy_server.start")
    def test_serve(self, start):
        start.return_value = {"pid": 123, "host": "0.0.0.0", "port": 8000}
        result = self.runner.invoke(cli, ["dspark-proxy", "serve"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("safety proxy running", result.output)
        self.assertIn("127.0.0.1:8888", result.output)

    @patch("ml.dspark_proxy_server.smoke")
    def test_smoke(self, smoke):
        smoke.return_value = {
            "authorized_models": 200,
            "unauthenticated_models": 401,
            "denied_invocations": 404,
        }
        result = self.runner.invoke(cli, ["dspark-proxy", "smoke"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("safety smoke passed", result.output)
        self.assertIn("/invocations", result.output)


if __name__ == "__main__":
    unittest.main()
