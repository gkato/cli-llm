import stat
import tempfile
import unittest
from pathlib import Path

from scripts.patch_miaai_worker_util import build_overlay


class MiaAIWorkerUtilPatchTests(unittest.TestCase):
    def test_builds_asymmetric_launcher_without_changing_head_assignment(self):
        source_text = "\n".join(
            (
                "#!/usr/bin/env bash",
                "export GPU_MEMORY_UTILIZATION ENABLE_VL_SIDECAR DSPARK_SERVE_MODE",
                'compose_base GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION"',
                "remote_compose \"GPU_MEMORY_UTILIZATION='$GPU_MEMORY_UTILIZATION' config\"",
                "remote_compose \"GPU_MEMORY_UTILIZATION='$GPU_MEMORY_UTILIZATION' up\"",
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "upstream.sh"
            destination = Path(directory) / "overlay.sh"
            source.write_text(source_text, encoding="utf-8")
            source.chmod(0o755)

            build_overlay(source, destination)
            patched = destination.read_text(encoding="utf-8")

            self.assertIn('GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION"', patched)
            self.assertEqual(
                patched.count(
                    "GPU_MEMORY_UTILIZATION='$WORKER_GPU_MEMORY_UTILIZATION'"
                ),
                2,
            )
            self.assertIn(
                'WORKER_GPU_MEMORY_UTILIZATION="${WORKER_GPU_MEMORY_UTILIZATION:-$GPU_MEMORY_UTILIZATION}"',
                patched,
            )
            self.assertTrue(destination.stat().st_mode & stat.S_IXUSR)

    def test_fails_closed_when_upstream_worker_commands_change(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "upstream.sh"
            destination = Path(directory) / "overlay.sh"
            source.write_text(
                "export GPU_MEMORY_UTILIZATION ENABLE_VL_SIDECAR DSPARK_SERVE_MODE\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "worker Compose commands changed"):
                build_overlay(source, destination)


if __name__ == "__main__":
    unittest.main()
