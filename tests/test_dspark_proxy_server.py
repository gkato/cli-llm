import sys
import unittest
from unittest.mock import Mock, patch

from ml import dspark_proxy_server


class DSparkProxyServerTests(unittest.TestCase):
    def test_build_cmd_uses_registry_host_and_port(self):
        cmd = dspark_proxy_server._build_cmd({"host": "0.0.0.0"}, 8123)
        self.assertEqual(cmd[:3], [sys.executable, "-m", "ml.dspark_proxy_api"])
        self.assertEqual(cmd[cmd.index("--host") + 1], "0.0.0.0")
        self.assertEqual(cmd[cmd.index("--port") + 1], "8123")

    @patch("ml.dspark_proxy_server.requests.get")
    @patch("ml.dspark_proxy_server._running_state")
    def test_status_reports_proxy_readiness(self, running_state, get):
        running_state.return_value = {
            "pid": 123,
            "host": "0.0.0.0",
            "port": 8000,
            "log_path": "/tmp/dspark_proxy.log",
        }
        get.return_value = Mock(ok=True, json=lambda: {"ready": True})

        result = dspark_proxy_server.status()

        self.assertTrue(result["running"])
        self.assertTrue(result["ready"])
        get.assert_called_once_with("http://127.0.0.1:8000/health", timeout=5)

    @patch("ml.dspark_proxy_server.get_api_key", return_value="secret")
    @patch("ml.dspark_proxy_server.requests.post")
    @patch("ml.dspark_proxy_server.requests.get")
    def test_smoke_checks_auth_and_denied_route(self, get, post, _api_key):
        get.side_effect = [Mock(ok=True, status_code=200), Mock(status_code=401)]
        post.return_value = Mock(status_code=404)

        result = dspark_proxy_server.smoke()

        self.assertEqual(result["authorized_models"], 200)
        self.assertEqual(result["unauthenticated_models"], 401)
        self.assertEqual(result["denied_invocations"], 404)
        self.assertIn("Authorization", get.call_args_list[0].kwargs["headers"])


if __name__ == "__main__":
    unittest.main()
