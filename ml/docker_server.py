"""Registry-driven vLLM serving in Docker.

This backend is for models that require a dedicated vLLM image rather than
the host's installed wheel. It intentionally manages one fixed container,
mirroring the one-server-at-a-time lifecycle used by the other backends.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import requests

from ml.config import DATA, HF_CACHE, VLLM_HOST, VLLM_PORT, get_api_key, get_models
from ml.state import running_state
from ml.vllm_args import build_vllm_serve_args

CONTAINER_NAME = "ml-compute-vllm-docker"
DOCKER_STATE = DATA / "docker_state.json"
NIM_CONTAINER_NAME = "ml-compute-nim"


def _read_state() -> dict | None:
    if not DOCKER_STATE.exists():
        return None
    try:
        return json.loads(DOCKER_STATE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_state(state: dict) -> None:
    DOCKER_STATE.parent.mkdir(parents=True, exist_ok=True)
    DOCKER_STATE.write_text(json.dumps(state, indent=2))


def _clear_state() -> None:
    if DOCKER_STATE.exists():
        DOCKER_STATE.unlink()


def _container_running(name: str = CONTAINER_NAME) -> bool:
    proc = subprocess.run(
        ["docker", "ps", "-q", "-f", f"name=^{name}$"],
        capture_output=True,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def _container_exists(name: str = CONTAINER_NAME) -> bool:
    proc = subprocess.run(
        ["docker", "ps", "-aq", "-f", f"name=^{name}$"],
        capture_output=True,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def _container_exit_info(name: str = CONTAINER_NAME) -> dict | None:
    proc = subprocess.run(
        [
            "docker",
            "inspect",
            "-f",
            "{{.State.Status}}|{{.State.ExitCode}}|{{.State.Error}}",
            name,
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    parts = proc.stdout.decode().strip().split("|", 2)
    if len(parts) < 2:
        return None
    return {
        "status": parts[0],
        "exit_code": int(parts[1]) if parts[1].lstrip("-").isdigit() else None,
        "error": parts[2] if len(parts) > 2 else "",
    }


def resolve_model(name: str) -> tuple[str, dict]:
    """Resolve a Docker-backed model alias into ``(image, config)``."""
    models = get_models()
    if name not in models:
        aliases = [
            alias
            for alias, cfg in models.items()
            if cfg.get("serve_backend") == "docker"
        ]
        raise ValueError(
            f"Unknown Docker model alias: {name!r}. Registered: "
            f"{', '.join(aliases) or '(none)'}"
        )

    cfg = models[name]
    if cfg.get("serve_backend") != "docker":
        backend = cfg.get("serve_backend", "vllm")
        raise ValueError(
            f"{name!r} uses the {backend!r} backend, not Docker. "
            f"Run `python3 -m ml.cli serve {name}` instead."
        )
    image = cfg.get("docker_image")
    if not image:
        raise ValueError(f"Docker model {name!r} has no docker_image configured")
    return str(image), cfg


def _build_run_cmd(
    name: str,
    image: str,
    cfg: dict,
    *,
    port: int,
    foreground: bool,
    gpu_count: int | None,
    cache_dir: str | Path,
    api_key: str | None,
) -> list[str]:
    """Build the complete ``docker run`` command for a registry entry."""
    if gpu_count is not None and gpu_count < 1:
        raise ValueError("gpu_count must be at least 1")
    cmd = ["docker", "run"]
    if not foreground:
        cmd.append("-d")

    network = str(cfg.get("docker_network", "host"))
    ipc = str(cfg.get("docker_ipc", "host"))
    gpus = (
        "all"
        if gpu_count is None
        else f"device={','.join(str(index) for index in range(gpu_count))}"
    )
    cmd += [
        "--name",
        CONTAINER_NAME,
        "--gpus",
        gpus,
        "--network",
        network,
        "--ipc",
        ipc,
        "--label",
        "ml-compute.backend=docker",
        "--label",
        f"ml-compute.model={name}",
        "-v",
        f"{Path(cache_dir).resolve()}:/root/.cache/huggingface",
        "-e",
        "HF_HUB_ENABLE_HF_TRANSFER=1",
    ]
    if network != "host":
        cmd += ["-p", f"{port}:{port}"]
    if os.getenv("HF_TOKEN"):
        # Passing only the variable name copies it from the parent environment
        # without placing its value in the process argument list.
        cmd += ["-e", "HF_TOKEN"]
    for key, value in (cfg.get("docker_env") or {}).items():
        cmd += ["-e", f"{key}={value}"]
    cmd.extend(str(value) for value in (cfg.get("docker_args") or []))
    cmd.append(image)
    cmd.extend(
        build_vllm_serve_args(
            str(cfg["hf_id"]),
            cfg,
            host=VLLM_HOST,
            port=port,
            download_dir="/root/.cache/huggingface/hub",
            api_key=api_key,
        )
    )
    return cmd


def _format_cmd(cmd: list[str], api_key: str | None) -> str:
    """Render a shell-safe command without exposing the API key."""
    redacted = ["***" if api_key and value == api_key else value for value in cmd]
    return " ".join(shlex.quote(value) for value in redacted)


def _validate_local_co_residency(
    local_state: dict | None,
    port: int,
    allow_co_resident: bool,
) -> None:
    if local_state and not allow_co_resident:
        raise RuntimeError(
            "A local vLLM/llama.cpp server is already running. Use "
            "`python3 -m ml.cli stop` first, or pass --allow-co-resident "
            "with a different port and a memory-capped registry profile."
        )
    if local_state and int(local_state["port"]) == port:
        raise RuntimeError(
            f"Port {port} is already used by the local "
            f"{local_state.get('backend', 'vllm')} server"
        )


def start(
    name: str,
    port: int | None = None,
    foreground: bool = False,
    gpu_count: int | None = None,
    allow_co_resident: bool = False,
) -> dict:
    """Start a registry-backed vLLM Docker container."""
    if _container_running():
        raise RuntimeError(
            f"Docker vLLM container {CONTAINER_NAME!r} is already running. "
            "Use `python3 -m ml.cli docker stop` first."
        )
    if _container_running(NIM_CONTAINER_NAME):
        raise RuntimeError(
            "A NIM container is already running. "
            "Use `python3 -m ml.cli nim stop` first."
        )
    if _container_exists():
        subprocess.run(["docker", "rm", CONTAINER_NAME], capture_output=True)

    image, cfg = resolve_model(name)
    port = port or VLLM_PORT
    local_state = running_state()
    _validate_local_co_residency(local_state, port, allow_co_resident)
    api_key = get_api_key()
    HF_CACHE.mkdir(parents=True, exist_ok=True)

    cmd = _build_run_cmd(
        name,
        image,
        cfg,
        port=port,
        foreground=foreground,
        gpu_count=gpu_count,
        cache_dir=HF_CACHE,
        api_key=api_key,
    )

    if foreground:
        print(f"Running: {_format_cmd(cmd, api_key)}")
        os.execvp(cmd[0], cmd)

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        message = (proc.stderr.decode() + "\n" + proc.stdout.decode()).strip()
        raise RuntimeError(f"docker run failed:\n{message}")

    container_id = proc.stdout.decode().strip()
    state = {
        "container_id": container_id,
        "container_name": CONTAINER_NAME,
        "alias": name,
        "image": image,
        "hf_id": cfg["hf_id"],
        "served_model_name": cfg.get("served_model_name", cfg["hf_id"]),
        "port": port,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_state(state)
    return state


def stop(timeout: int = 30) -> bool:
    """Stop and remove the managed Docker vLLM container."""
    if not _container_exists():
        _clear_state()
        return False
    subprocess.run(
        ["docker", "stop", "-t", str(timeout), CONTAINER_NAME],
        capture_output=True,
    )
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
    _clear_state()
    return True


def status() -> dict:
    """Return container lifecycle state and OpenAI API readiness."""
    state = _read_state()
    running = _container_running()
    exists = _container_exists()

    if not exists:
        if state:
            _clear_state()
        return {"running": False}

    if running:
        port = int((state or {}).get("port", VLLM_PORT))
        api_key = get_api_key()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        ready = False
        try:
            response = requests.get(
                f"http://localhost:{port}/v1/models",
                headers=headers,
                timeout=2,
            )
            ready = response.ok
        except requests.RequestException:
            pass
        return {
            "running": True,
            "ready": ready,
            "container_id": (state or {}).get("container_id"),
            "container_name": CONTAINER_NAME,
            "alias": (state or {}).get("alias"),
            "image": (state or {}).get("image"),
            "hf_id": (state or {}).get("hf_id"),
            "served_model_name": (state or {}).get("served_model_name"),
            "port": port,
            "started_at": (state or {}).get("started_at"),
        }

    info = _container_exit_info()
    return {
        "running": False,
        "exited": True,
        "container_name": CONTAINER_NAME,
        "alias": (state or {}).get("alias"),
        "image": (state or {}).get("image"),
        "exit_status": (info or {}).get("status"),
        "exit_code": (info or {}).get("exit_code"),
        "exit_error": (info or {}).get("error"),
        "log_hint": f"python3 -m ml.cli docker logs",
    }


def tail_logs(lines: int = 50, follow: bool = False) -> None:
    """Print logs for the managed running or exited container."""
    if not _container_exists():
        print("No Docker vLLM container present (running or exited).")
        return
    cmd = ["docker", "logs", "--tail", str(lines)]
    if follow:
        cmd.append("-f")
    cmd.append(CONTAINER_NAME)
    subprocess.run(cmd)
