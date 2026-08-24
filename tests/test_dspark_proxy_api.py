import unittest

import httpx

from ml.config import get_dspark_proxy_config, get_models
from ml.dspark_proxy_api import (
    build_upstream_headers,
    is_allowed_route,
    is_authorized,
    validate_config,
)


class DSparkProxyApiTests(unittest.TestCase):
    def test_allows_only_reviewed_openai_routes(self):
        allowed = [
            ("GET", "/v1/models"),
            ("POST", "/v1/chat/completions"),
            ("POST", "/v1/completions"),
            ("POST", "/v1/responses"),
            ("GET", "/v1/responses/resp_abc-123"),
            ("DELETE", "/v1/responses/resp_abc-123"),
            ("POST", "/v1/responses/resp_abc-123/cancel"),
            ("POST", "/v1/messages"),
            ("POST", "/v1/messages/count_tokens"),
        ]
        for method, path in allowed:
            with self.subTest(method=method, path=path):
                self.assertTrue(is_allowed_route(method, path))

    def test_denies_vllm_unguarded_and_unknown_routes(self):
        denied = [
            ("POST", "/invocations"),
            ("POST", "/generative_scoring"),
            ("POST", "/tokenize"),
            ("POST", "/detokenize"),
            ("GET", "/metrics"),
            ("POST", "/v1/models"),
            ("PUT", "/v1/responses/resp_1"),
            ("POST", "/v1/future-admin-route"),
        ]
        for method, path in denied:
            with self.subTest(method=method, path=path):
                self.assertFalse(is_allowed_route(method, path))

    def test_bearer_auth_is_case_insensitive_but_exact(self):
        self.assertTrue(is_authorized("Bearer secret", "secret"))
        self.assertTrue(is_authorized("bearer secret", "secret"))
        self.assertFalse(is_authorized("Bearer wrong", "secret"))
        self.assertFalse(is_authorized("Basic secret", "secret"))
        self.assertFalse(is_authorized(None, "secret"))

    def test_replaces_caller_auth_and_forwarding_headers_without_duplicates(self):
        headers = build_upstream_headers(
            (
                ("authorization", "Bearer caller"),
                ("x-forwarded-for", "spoofed"),
                ("x-forwarded-proto", "http"),
                ("accept", "application/json"),
            ),
            "internal",
            "127.0.0.1",
            "https",
        )
        request = httpx.Request("GET", "http://upstream/v1/models", headers=headers)

        self.assertEqual(
            request.headers.get_list("authorization"), ["Bearer internal"]
        )
        self.assertEqual(request.headers.get_list("x-forwarded-for"), ["127.0.0.1"])
        self.assertEqual(request.headers.get_list("x-forwarded-proto"), ["https"])
        self.assertEqual(request.headers["accept"], "application/json")

    def test_registry_upstream_is_loopback_and_ports_are_split(self):
        config = get_dspark_proxy_config()
        validate_config(config)
        self.assertEqual(config["upstream_url"], "http://127.0.0.1:8888")
        self.assertEqual(config["port"], 8000)

    def test_dspark_registry_uses_512k_coexistence_profile(self):
        profile = get_models()["deepseek-v4-flash-0731-dspark"]
        self.assertEqual(profile["max_model_len"], 524288)
        self.assertEqual(profile["max_num_seqs"], 4)
        self.assertEqual(profile["gpu_memory_utilization"], 0.75)
        self.assertEqual(profile["worker_gpu_memory_utilization"], 0.73)
        self.assertEqual(profile["raw_api_url"], "http://127.0.0.1:8888")
        self.assertEqual(profile["proxy_bind"], "0.0.0.0:8000")
        self.assertEqual(profile["proxy_url"], "http://127.0.0.1:8000")

    def test_rejects_non_loopback_upstream(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            validate_config({"upstream_url": "http://192.168.1.10:8888"})


if __name__ == "__main__":
    unittest.main()
