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


if __name__ == "__main__":
    unittest.main()
