import unittest

from scripts.check_dspark_kv_capacity import (
    parse_cache_capacity,
    parse_startup_capacity,
    startup_capacity_fits,
)


class DSparkKVCapacityTests(unittest.TestCase):
    def test_uses_model_aware_startup_capacity_for_requested_window(self):
        logs = "\n".join(
            (
                "GPU KV cache size: 871,864 tokens",
                "Maximum concurrency for 524,288 tokens per request: 1.66x",
            )
        )

        capacity, concurrency = parse_startup_capacity(logs, 524_288)

        self.assertEqual(capacity, 871_864)
        self.assertEqual(concurrency, 1.66)

    def test_ignores_concurrency_for_a_different_request_ceiling(self):
        logs = "Maximum concurrency for 1,048,576 tokens per request: 0.83x"

        capacity, concurrency = parse_startup_capacity(logs, 524_288)

        self.assertIsNone(capacity)
        self.assertIsNone(concurrency)

    def test_gates_on_startup_concurrency_not_the_broken_metric(self):
        self.assertTrue(startup_capacity_fits(302_945, 1.16, 262_144))
        self.assertFalse(startup_capacity_fits(123_433, 0.47, 262_144))

    def test_prefers_direct_hybrid_cache_token_capacity(self):
        metrics = (
            'vllm:cache_config_info{block_size="4",num_gpu_blocks="3991",'
            'kv_cache_size_tokens="1021696",kv_cache_max_concurrency="1.949"} 1\n'
        )

        capacity, details = parse_cache_capacity(metrics)

        self.assertEqual(capacity, 1_021_696)
        self.assertIn("kv_cache_size_tokens", details)
        self.assertNotEqual(capacity, 3_991 * 4)

    def test_falls_back_for_legacy_non_hybrid_metrics(self):
        metrics = (
            'vllm:cache_config_info{block_size="16",num_gpu_blocks="40000"} 1\n'
        )

        capacity, details = parse_cache_capacity(metrics)

        self.assertEqual(capacity, 640_000)
        self.assertIn("legacy", details)

    def test_rejects_missing_cache_metric(self):
        with self.assertRaisesRegex(ValueError, "cache_config_info metric is missing"):
            parse_cache_capacity("vllm:num_requests_running 0\n")


if __name__ == "__main__":
    unittest.main()
