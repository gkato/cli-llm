"""Isolated execution for generated HumanEval+ Python programs."""

from __future__ import annotations

import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecutionResult:
    passed: bool
    detail: str
    duration_seconds: float


class ExecutionInfrastructureError(RuntimeError):
    pass


class DockerExecutor:
    """Run untrusted generated code in a resource-capped, networkless container."""

    def __init__(
        self,
        *,
        image: str,
        timeout_seconds: float,
        memory: str = "256m",
        cpus: str = "1",
        dockerfile: Path | None = None,
        build_context: Path | None = None,
    ) -> None:
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.memory = memory
        self.cpus = cpus
        self.dockerfile = dockerfile
        self.build_context = build_context or (dockerfile.parent if dockerfile else None)
        self.numpy_version: str | None = None

    def _build_or_pull_image(self) -> None:
        if self.dockerfile is not None:
            if not self.dockerfile.is_file() or self.build_context is None:
                raise ExecutionInfrastructureError(
                    f"evaluator Dockerfile is unavailable: {self.dockerfile}"
                )
            prepare = subprocess.run(
                [
                    "docker",
                    "build",
                    "--tag",
                    self.image,
                    "--file",
                    str(self.dockerfile),
                    str(self.build_context),
                ],
                capture_output=True,
                text=True,
                timeout=900,
                check=False,
            )
            action = "build"
        else:
            prepare = subprocess.run(
                ["docker", "pull", self.image],
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
            action = "pull"
        if prepare.returncode:
            detail = (prepare.stderr or prepare.stdout).strip()[-2000:]
            raise ExecutionInfrastructureError(
                f"could not {action} evaluator image {self.image}: {detail}"
            )

    def _verify_dependencies(self) -> None:
        probe = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--pull=never",
                "--network=none",
                "--memory=256m",
                "--memory-swap=256m",
                "--cpus=1",
                "--pids-limit=64",
                "--user=65534:65534",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                self.image,
                "python",
                "-I",
                "-c",
                "import numpy; print(numpy.__version__)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if probe.returncode:
            detail = (probe.stderr or probe.stdout).strip()[-1000:]
            raise ExecutionInfrastructureError(
                f"evaluator image {self.image} cannot import the HumanEval+ "
                f"dependency numpy: {detail}"
            )
        self.numpy_version = probe.stdout.strip()

    def preflight(self) -> None:
        if not shutil.which("docker"):
            raise ExecutionInfrastructureError(
                "Docker is required to execute HumanEval+ safely, but it is not on PATH."
            )
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()[:500]
            raise ExecutionInfrastructureError(f"Docker is unavailable: {detail}")
        inspect = subprocess.run(
            ["docker", "image", "inspect", self.image],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if inspect.returncode:
            self._build_or_pull_image()
        self._verify_dependencies()

    def run(self, program: str) -> ExecutionResult:
        name = "mlc-eval-" + uuid.uuid4().hex[:16]
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--pull=missing",
            "--network=none",
            f"--memory={self.memory}",
            f"--memory-swap={self.memory}",
            f"--cpus={self.cpus}",
            "--pids-limit=64",
            "--user=65534:65534",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
            "-i",
            self.image,
            "python",
            "-I",
            "-",
        ]
        import time

        started = time.perf_counter()
        try:
            result = subprocess.run(
                command,
                input=program,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["docker", "rm", "-f", name],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            return ExecutionResult(
                passed=False,
                detail=f"execution exceeded {self.timeout_seconds:g}s",
                duration_seconds=time.perf_counter() - started,
            )
        duration = time.perf_counter() - started
        if result.returncode == 125:
            detail = (result.stderr or result.stdout).strip()[:1000]
            raise ExecutionInfrastructureError(f"Docker could not start the evaluator: {detail}")
        output = (result.stderr or result.stdout).strip()
        if result.returncode == 0:
            return ExecutionResult(True, "all EvalPlus tests passed", duration)
        return ExecutionResult(False, output[-2000:] or f"Python exited {result.returncode}", duration)
