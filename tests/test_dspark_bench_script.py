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


if __name__ == "__main__":
    unittest.main()
