import json
import unittest

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
                    "served_model": "nvidia/Qwen3.6-27B-NVFP4",
                    "models": [
                        "nvidia/Qwen3.6-27B-NVFP4",
                        "qwen3.6-27b-nvfp4-coserve",
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
                "model": "qwen3.6-27b-nvfp4-coserve",
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
            json.loads(forwarded)["model"], "nvidia/Qwen3.6-27B-NVFP4"
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
                body=b'{"model":"qwen3.6-27b-nvfp4-coserve"}',
                model_header="pp-ocrv6-medium-det",
            )

    def test_validates_and_advertises_all_accepted_ids(self):
        validate_config(self.config)
        ids = {entry["id"] for entry in advertised_models(self.config)}
        self.assertEqual(
            ids,
            {
                "nvidia/Qwen3.6-27B-NVFP4",
                "qwen3.6-27b-nvfp4-coserve",
                "pp-ocrv6-medium-det",
            },
        )

    def test_rejects_zero_global_concurrency(self):
        self.config["max_concurrency"] = 0
        with self.assertRaisesRegex(ValueError, "at least 1"):
            validate_config(self.config)


if __name__ == "__main__":
    unittest.main()
