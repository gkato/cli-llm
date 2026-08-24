import unittest
from pathlib import Path


class DSparkScriptTests(unittest.TestCase):
    def test_generic_vllm_host_cannot_override_private_dspark_bind(self):
        script = (
            Path(__file__).resolve().parents[1] / "scripts" / "DS4-Flash-DSpark.sh"
        ).read_text(encoding="utf-8")

        self.assertIn(
            'profile_value DSPARK_VLLM_HOST "$(profile_file_value VLLM_HOST 127.0.0.1)"',
            script,
        )
        self.assertNotIn('profile_value VLLM_HOST 127.0.0.1', script)

    def test_cutover_checks_artifacts_before_stopping_legacy_service(self):
        script = (
            Path(__file__).resolve().parents[1] / "scripts" / "DS4-Flash-DSpark.sh"
        ).read_text(encoding="utf-8")
        start_cluster = script.split("start_cluster() {", 1)[1].split("\n}", 1)[0]

        self.assertLess(
            start_cluster.index("check_cutover_artifacts"),
            start_cluster.index("legacy_stop"),
        )


if __name__ == "__main__":
    unittest.main()
