import json
import unittest

from ml.config import get_models, get_router_config
from ml.router_api import (
    RoutingError,
    advertised_models,
    route_request,
    validate_config,
)


class RouterApiTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "backends": {
                "qwen": {
                    "url": "http://127.0.0.1:8101",
                    "served_model": "unsloth/Qwen3.8-27B-NVFP4",
                    "models": [
                        "unsloth/Qwen3.8-27B-NVFP4",
                        "qwen3.8-27b-nvfp4-coserve",
                    ],
                },
                "detector": {
                    "url": "http://127.0.0.1:8103",
                    "served_model": "pp-ocrv6-medium-det",
                    "models": ["pp-ocrv6-medium-det"],
                },
            },
            "path_routes": {"/v1/text/detections": "detector"},
        }

    def test_routes_json_by_model_and_rewrites_alias(self):
        body = json.dumps(
            {
                "model": "qwen3.8-27b-nvfp4-coserve",
                "messages": [{"role": "user", "content": "hi"}],
            }
        ).encode()

        backend, forwarded = route_request(
            self.config,
            path="/v1/chat/completions",
            content_type="application/json",
            body=body,
            model_header=None,
        )

        self.assertEqual(backend, "qwen")
        self.assertEqual(
            json.loads(forwarded)["model"], "unsloth/Qwen3.8-27B-NVFP4"
        )

    def test_routes_raw_image_by_path(self):
        body = b"png-bytes"
        backend, forwarded = route_request(
            self.config,
            path="/v1/text/detections",
            content_type="image/png",
            body=body,
            model_header=None,
        )

        self.assertEqual(backend, "detector")
        self.assertEqual(forwarded, body)

    def test_routes_raw_image_by_model_header(self):
        backend, _ = route_request(
            self.config,
            path="/custom/detect",
            content_type="image/jpeg",
            body=b"jpeg-bytes",
            model_header="pp-ocrv6-medium-det",
        )
        self.assertEqual(backend, "detector")

    def test_rejects_conflicting_model_header(self):
        with self.assertRaisesRegex(RoutingError, "disagree"):
            route_request(
                self.config,
                path="/v1/chat/completions",
                content_type="application/json",
                body=b'{"model":"qwen3.8-27b-nvfp4-coserve"}',
                model_header="pp-ocrv6-medium-det",
            )

    def test_validates_and_advertises_all_accepted_ids(self):
        validate_config(self.config)
        ids = {entry["id"] for entry in advertised_models(self.config)}
        self.assertEqual(
            ids,
            {
                "unsloth/Qwen3.8-27B-NVFP4",
                "qwen3.8-27b-nvfp4-coserve",
                "pp-ocrv6-medium-det",
            },
        )

    def test_rejects_zero_global_concurrency(self):
        self.config["max_concurrency"] = 0
        with self.assertRaisesRegex(ValueError, "at least 1"):
            validate_config(self.config)

    def test_registry_qwen_route_matches_coserve_profile(self):
        router = get_router_config()
        models = get_models()
        backend = router["backends"]["qwen"]
        alias = "qwen3.8-27b-nvfp4-coserve"
        profile = models[alias]

        self.assertIn(alias, backend["models"])
        self.assertEqual(profile["served_model_name"], backend["served_model"])
        self.assertEqual(profile["gpu_memory_utilization"], 0.30)
        self.assertEqual(profile["max_model_len"], 32768)
        self.assertEqual(profile["max_num_seqs"], 1)
        self.assertNotIn("quantization", profile)
        self.assertNotIn("speculative_config", profile)
        self.assertNotIn("language_model_only", profile)


if __name__ == "__main__":
    unittest.main()
