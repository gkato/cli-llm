import importlib.util
import unittest
from pathlib import Path


class DSparkBenchScriptTests(unittest.TestCase):
    def test_memory_warning_does_not_abort_throughput_benchmark(self):
        script = (
            Path(__file__).resolve().parents[1] / "scripts" / "bench_dspark_ab.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("capture_memory_snapshot()", script)
        self.assertIn('statuses=("${PIPESTATUS[@]}")', script)
        self.assertIn(
            'capture_memory_snapshot "${RESULT_DIR}/memory-before.txt"', script
        )
        self.assertIn(
            'capture_memory_snapshot "${RESULT_DIR}/memory-after.txt"', script
        )

    def test_dspark_benchmark_does_not_use_hf_tokenizer(self):
        script = (
            Path(__file__).resolve().parents[1] / "scripts" / "bench_dspark_ab.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("scripts/bench_dspark_throughput.py", script)
        self.assertNotIn("VLLM_TOKENIZER", script)
        self.assertNotIn("scripts/bench_vllm.sh", script)


class DSparkThroughputRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "bench_dspark_throughput.py"
        )
        spec = importlib.util.spec_from_file_location("bench_dspark_throughput", path)
        assert spec and spec.loader
        cls.bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.bench)

    def test_prompt_has_unique_cold_prefix_and_miaai_prompt_unit(self):
        first = self.bench.build_prompt(2048, "nonce-one", 128)
        second = self.bench.build_prompt(2048, "nonce-two", 128)

        self.assertTrue(first.startswith("unique benchmark request nonce-one "))
        self.assertNotEqual(first, second)
        self.assertIn("benchmark context datum ", first)

    def test_base_url_and_concurrency_parsing(self):
        self.assertEqual(
            self.bench.normalize_api_url("http://127.0.0.1:8000"),
            "http://127.0.0.1:8000/v1",
        )
        self.assertEqual(self.bench.parse_positive_ints("1,2 4"), [1, 2, 4])

    def test_percentile_interpolates(self):
        self.assertEqual(self.bench.percentile([1.0, 3.0], 50), 2.0)


if __name__ == "__main__":
    unittest.main()
