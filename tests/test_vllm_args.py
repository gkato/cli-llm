import unittest
from unittest.mock import patch

from ml.vllm_args import build_vllm_serve_args


class BuildVllmServeArgsTests(unittest.TestCase):
    def test_translates_registry_fields_and_extra_args(self):
        cfg = {
            "served_model_name": "Unlimited-OCR",
            "max_model_len": 32768,
            "max_num_seqs": 1,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.2,
            "rope_scaling": {"rope_type": "yarn", "factor": 2.0},
            "extra_args": ["--no-enable-prefix-caching", "--mm-processor-cache-gb", "0"],
        }

        args = build_vllm_serve_args(
            "baidu/Unlimited-OCR",
            cfg,
            host="0.0.0.0",
            port=8000,
            download_dir="/cache/hub",
            api_key="secret",
        )

        self.assertEqual(args[0], "baidu/Unlimited-OCR")
        self.assertIn("Unlimited-OCR", args)
        self.assertIn("--dtype", args)
        self.assertIn("bfloat16", args)
        self.assertIn("--no-enable-prefix-caching", args)
        self.assertEqual(
            args[args.index("--rope-scaling") + 1],
            '{"rope_type":"yarn","factor":2.0}',
        )
        self.assertEqual(args[-2:], ["--api-key", "secret"])

    def test_rejects_non_mapping_json_option(self):
        with self.assertRaisesRegex(ValueError, "rope_scaling"):
            build_vllm_serve_args(
                "org/model",
                {"rope_scaling": [1, 2]},
                host="127.0.0.1",
                port=8000,
            )

    @patch("ml.vllm_server.get_api_key", return_value="secret")
    @patch("ml.vllm_server.get_models")
    def test_host_vllm_builder_uses_shared_translation(self, get_models, _get_api_key):
        from ml.vllm_server import _build_cmd

        get_models.return_value = {
            "example": {
                "hf_id": "org/model",
                "dtype": "bfloat16",
                "max_model_len": 4096,
            }
        }
        cmd = _build_cmd("example", "org/model", 9000)

        self.assertEqual(cmd[:3], ["vllm", "serve", "org/model"])
        self.assertIn("--max-model-len", cmd)
        self.assertEqual(cmd[-2:], ["--api-key", "secret"])


if __name__ == "__main__":
    unittest.main()
