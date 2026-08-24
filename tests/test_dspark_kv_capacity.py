import unittest

from scripts.check_dspark_kv_capacity import parse_cache_capacity


class DSparkKVCapacityTests(unittest.TestCase):
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
