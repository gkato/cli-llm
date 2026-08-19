import sys
import unittest
from unittest.mock import Mock, patch

from ml import router_server


class RouterServerTests(unittest.TestCase):
    def test_build_cmd_uses_registry_host_and_port(self):
        cmd = router_server._build_cmd({"host": "0.0.0.0"}, 8123)
        self.assertEqual(cmd[:3], [sys.executable, "-m", "ml.router_api"])
        self.assertEqual(cmd[cmd.index("--host") + 1], "0.0.0.0")
        self.assertEqual(cmd[cmd.index("--port") + 1], "8123")

    @patch("ml.router_server.requests.get")
    @patch("ml.router_server._running_state")
    def test_status_reports_backend_readiness(self, running_state, get):
        running_state.return_value = {
            "pid": 123,
            "host": "0.0.0.0",
            "port": 8000,
            "log_path": "/tmp/router.log",
            "started_at": "2026-08-18T00:00:00+00:00",
        }
        get.return_value = Mock(
            ok=True,
            json=lambda: {
                "ready": True,
                "backends": {"qwen": {"ready": True, "status": 200}},
            },
        )

        result = router_server.status()

        self.assertTrue(result["running"])
        self.assertTrue(result["ready"])
        self.assertTrue(result["backends"]["qwen"]["ready"])
        get.assert_called_once_with("http://localhost:8000/health", timeout=5)


if __name__ == "__main__":
    unittest.main()
