#!/usr/bin/env python3
"""Build a version-checked MiaAI launcher with per-node memory utilization."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile


EXPORT_MARKER = "export GPU_MEMORY_UTILIZATION ENABLE_VL_SIDECAR DSPARK_SERVE_MODE"
WORKER_ASSIGNMENT = "GPU_MEMORY_UTILIZATION='$GPU_MEMORY_UTILIZATION'"
PATCHED_WORKER_ASSIGNMENT = (
    "GPU_MEMORY_UTILIZATION='$WORKER_GPU_MEMORY_UTILIZATION'"
)


def build_overlay(source: Path, destination: Path) -> None:
    text = source.read_text(encoding="utf-8")

    if text.count(EXPORT_MARKER) != 1:
        raise RuntimeError(
            "MiaAI launcher profile marker changed; review the upstream update before "
            "starting the cluster"
        )
    if text.count(WORKER_ASSIGNMENT) != 2:
        raise RuntimeError(
            "MiaAI worker Compose commands changed; review the upstream update before "
            "starting the cluster"
        )

    patched = text.replace(
        EXPORT_MARKER,
        "\n".join(
            (
                EXPORT_MARKER,
                "# ml-compute overlay: retain a lower worker allocation for the Harness.",
                'WORKER_GPU_MEMORY_UTILIZATION="${WORKER_GPU_MEMORY_UTILIZATION:-$GPU_MEMORY_UTILIZATION}"',
                "export WORKER_GPU_MEMORY_UTILIZATION",
            )
        ),
        1,
    )
    patched = patched.replace(WORKER_ASSIGNMENT, PATCHED_WORKER_ASSIGNMENT)

    destination.parent.mkdir(parents=True, exist_ok=True)
    source_mode = source.stat().st_mode & 0o777
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as output:
        output.write(patched)
        temporary = Path(output.name)
    try:
        os.chmod(temporary, source_mode | 0o100)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {Path(sys.argv[0]).name} SOURCE DESTINATION", file=sys.stderr)
        return 2
    try:
        build_overlay(Path(sys.argv[1]), Path(sys.argv[2]))
    except (OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
