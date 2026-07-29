"""NIM (NVIDIA Inference Microservices) container façade.

Same lifecycle surface as ml.vllm_server (start/stop/status/logs) but
backed by `docker run` against an NGC container instead of a local
vLLM process. The OpenAI-compatible endpoint NIM exposes is wire-
compatible with vLLM's, so clients don't change.

State is tracked via a fixed container name (`ml-compute-nim`) — we
only ever run one NIM at a time, matching the single-vLLM convention.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

import requests
import yaml

from ml.config import DATA, HF_CACHE, LOGS, REGISTRY, VLLM_HOST, VLLM_PORT, get_api_key

CONTAINER_NAME = "ml-compute-nim"
NIM_LOG = LOGS / "nim.log"
NIM_STATE = DATA / "nim_state.json"
NIM_CACHE = DATA / "nim_cache"


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

def get_nim_catalog() -> dict:
    """Load registry/nim_catalog.yaml. Returns {alias: {image, model_name, ...}}."""
    path = REGISTRY / "nim_catalog.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("nim_catalog", {}) or {}


def resolve_image(name: str) -> tuple[str, dict]:
    """Resolve a catalog alias or raw image URI. Returns (image, cfg)."""
    catalog = get_nim_catalog()
    if name in catalog:
        return catalog[name]["image"], catalog[name]
    # Treat as a literal image URI (e.g. user passed nvcr.io/...)
    if "/" in name and ":" in name:
        return name, {}
    raise ValueError(
        f"Unknown NIM alias: {name!r}. Registered: "
        f"{', '.join(catalog.keys()) or '(none)'}; or pass a full nvcr.io/...:tag"
    )


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _ngc_key() -> str:
    key = os.getenv("NGC_API_KEY")
    if not key:
        raise RuntimeError(
            "NGC_API_KEY not set. Get one at https://build.nvidia.com → Login → "
            "API key, then add NGC_API_KEY=... to .env.local"
        )
    return key


def _ensure_docker_login() -> None:
    """Run `docker login nvcr.io` with NGC_API_KEY. Idempotent."""
    key = _ngc_key()
    proc = subprocess.run(
        ["docker", "login", "nvcr.io", "-u", "$oauthtoken", "--password-stdin"],
        input=key.encode(),
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "docker login nvcr.io failed:\n" + proc.stderr.decode()
        )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def _read_state() -> dict | None:
    if not NIM_STATE.exists():
        return None
    try:
        return json.loads(NIM_STATE.read_text())
    except Exception:
        return None


def _write_state(d: dict) -> None:
    NIM_STATE.parent.mkdir(parents=True, exist_ok=True)
    NIM_STATE.write_text(json.dumps(d, indent=2))


def _clear_state() -> None:
    if NIM_STATE.exists():
        NIM_STATE.unlink()


def _container_running(name: str = CONTAINER_NAME) -> bool:
    proc = subprocess.run(
        ["docker", "ps", "-q", "-f", f"name=^{name}$"],
        capture_output=True,
    )
    return bool(proc.stdout.strip())


def _container_exists(name: str = CONTAINER_NAME) -> bool:
    proc = subprocess.run(
        ["docker", "ps", "-aq", "-f", f"name=^{name}$"],
        capture_output=True,
    )
    return bool(proc.stdout.strip())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_models() -> list[dict]:
    """Return catalog entries augmented with a 'pulled' flag."""
    catalog = get_nim_catalog()
    out = []
    for alias, cfg in catalog.items():
        image = cfg["image"]
        proc = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
        )
        out.append({
            "alias": alias,
            "image": image,
            "model_name": cfg.get("model_name"),
            "description": cfg.get("description", ""),
            "vram_gb": cfg.get("vram_gb"),
            "pulled": proc.returncode == 0,
        })
    return out


def start(name: str, port: int | None = None, foreground: bool = False,
          gpu_count: int | None = None) -> dict:
    """Run a NIM container in the background. Returns state dict."""
    if _container_running():
        raise RuntimeError(
            f"A NIM container ({CONTAINER_NAME}) is already running. "
            "Use `ml-compute nim stop` first."
        )
    if _container_exists():
        # Stale stopped container with the same name — remove it.
        subprocess.run(["docker", "rm", CONTAINER_NAME], capture_output=True)

    image, cfg = resolve_image(name)
    port = port or VLLM_PORT
    api_key = get_api_key()

    _ensure_docker_login()

    NIM_CACHE.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    # Docker requires absolute paths for bind-mounts; otherwise it treats the
    # source as a named-volume identifier (with character restrictions). Use
    # .resolve() so HF_HOME from .env.local etc. survives as a relative path.
    nim_cache_abs = NIM_CACHE.resolve()
    hf_cache_abs = HF_CACHE.resolve()

    gpus = "all" if gpu_count is None else f'"device={",".join(str(i) for i in range(gpu_count))}"'

    # NOTE: no --rm. We want the stopped container to survive so the user
    # can inspect logs and exit status. `stop()` calls `docker rm -f` to
    # clean up explicitly.
    cmd = [
        "docker", "run", "-d",
        "--name", CONTAINER_NAME,
        "--gpus", gpus,
        "--shm-size=16g",
        "-p", f"{port}:8000",
        "-v", f"{nim_cache_abs}:/opt/nim/.cache",
        "-v", f"{hf_cache_abs}:/root/.cache/huggingface",
        "-e", f"NGC_API_KEY={_ngc_key()}",
        "-e", "HF_HUB_ENABLE_HF_TRANSFER=1",
    ]
    if api_key:
        # NIM containers honor NIM_API_KEY for endpoint auth.
        cmd += ["-e", f"NIM_API_KEY={api_key}"]
    for k, v in (cfg.get("extra_env") or {}).items():
        cmd += ["-e", f"{k}={v}"]
    cmd.append(image)

    if foreground:
        # Strip the -d flag and exec instead so the user sees output live.
        cmd = [c for c in cmd if c != "-d"]
        print(f"Running: {' '.join(shlex.quote(c) for c in cmd)}")
        os.execvp(cmd[0], cmd)

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "docker run failed:\n" + proc.stderr.decode() + "\n" + proc.stdout.decode()
        )
    container_id = proc.stdout.decode().strip()

    state = {
        "container_id": container_id,
        "container_name": CONTAINER_NAME,
        "alias": name,
        "image": image,
        "model_name": cfg.get("model_name"),
        "port": port,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _write_state(state)
    return {**state, "log_path": str(NIM_LOG)}


def stop(timeout: int = 30) -> bool:
    """Stop the NIM container. Returns True if a container was stopped."""
    if not _container_exists():
        _clear_state()
        return False
    subprocess.run(["docker", "stop", "-t", str(timeout), CONTAINER_NAME],
                   capture_output=True)
    # docker run --rm should remove on stop; clean up anyway.
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
    _clear_state()
    return True


def _container_exit_info(name: str = CONTAINER_NAME) -> dict | None:
    """Return {status, exit_code} for an exited container, or None if absent."""
    proc = subprocess.run(
        ["docker", "inspect", "-f",
         "{{.State.Status}}|{{.State.ExitCode}}|{{.State.Error}}", name],
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


def status() -> dict:
    """Return a status dict describing the current NIM container."""
    state = _read_state()
    running = _container_running()
    exists = _container_exists()

    if not state and not exists:
        return {"running": False}

    info = _container_exit_info() if exists else None

    if running:
        port = state["port"] if state else 8000
        ready = False
        api_key = get_api_key()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            r = requests.get(f"http://localhost:{port}/v1/models",
                             headers=headers, timeout=2)
            ready = r.ok
        except requests.RequestException:
            ready = False
        return {
            "running": True,
            "ready": ready,
            "container_id": (state or {}).get("container_id"),
            "container_name": CONTAINER_NAME,
            "alias": (state or {}).get("alias"),
            "image": (state or {}).get("image"),
            "model_name": (state or {}).get("model_name"),
            "port": port,
            "started_at": (state or {}).get("started_at"),
            "log_path": str(NIM_LOG),
        }

    # Container exists but isn't running — it crashed or exited.
    return {
        "running": False,
        "exited": True,
        "container_name": CONTAINER_NAME,
        "alias": (state or {}).get("alias"),
        "image": (state or {}).get("image"),
        "exit_status": (info or {}).get("status"),
        "exit_code": (info or {}).get("exit_code"),
        "exit_error": (info or {}).get("error"),
        "log_hint": f"docker logs {CONTAINER_NAME}    # see why it died",
    }


def tail_logs(lines: int = 50, follow: bool = False) -> None:
    """Tail the NIM container's stdout/stderr via `docker logs`.

    Works for both running AND exited containers (until they're removed).
    """
    if not _container_exists():
        print("No NIM container present (running or exited).")
        return
    cmd = ["docker", "logs", "--tail", str(lines)]
    if follow:
        cmd.append("-f")
    cmd.append(CONTAINER_NAME)
    subprocess.run(cmd)
