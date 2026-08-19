"""Lifecycle manager for lightweight Transformers vision models.

Unlike the main vLLM backend, this service is intentionally independent: it
has its own state file, log, and port so it can remain resident beside a local
LLM and the dedicated Unlimited-OCR container.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

import requests

from ml.config import DATA, HF_CACHE, LOGS, VLLM_HOST, get_api_key, get_models
from ml.state import is_pid_alive

VISION_STATE = DATA / "vision_server.json"
VISION_LOG = LOGS / "vision.log"
DEFAULT_VISION_PORT = 8002


def _check_dependencies() -> None:
    modules = (
        "fastapi",
        "uvicorn",
        "torch",
        "torchvision",
        "transformers",
        "PIL",
        "cv2",
    )
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "Transformers vision dependencies are missing: "
            f"{', '.join(missing)}. Re-run `bash setup.sh`."
        )


def _read_state() -> dict | None:
    if not VISION_STATE.exists():
        return None
    try:
        return json.loads(VISION_STATE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_state(state: dict) -> None:
    VISION_STATE.parent.mkdir(parents=True, exist_ok=True)
    VISION_STATE.write_text(json.dumps(state, indent=2))


def _clear_state() -> None:
    if VISION_STATE.exists():
        VISION_STATE.unlink()


def _running_state() -> dict | None:
    state = _read_state()
    if not state:
        return None
    if not is_pid_alive(int(state["pid"])):
        _clear_state()
        return None
    return state


def resolve_model(name: str) -> dict:
    models = get_models()
    if name not in models:
        aliases = [
            alias
            for alias, cfg in models.items()
            if cfg.get("serve_backend") == "vision"
        ]
        raise ValueError(
            f"Unknown vision model alias: {name!r}. Registered: "
            f"{', '.join(aliases) or '(none)'}"
        )
    cfg = dict(models[name])
    if cfg.get("serve_backend") != "vision":
        backend = cfg.get("serve_backend", "vllm")
        raise ValueError(
            f"{name!r} uses the {backend!r} backend, not Transformers vision"
        )
    return cfg


def _build_cmd(name: str, cfg: dict, port: int) -> list[str]:
    vision = cfg.get("vision_config") or {}
    cmd = [
        sys.executable,
        "-m",
        "ml.vision_api",
        "--model-id",
        str(cfg["hf_id"]),
        "--served-model-name",
        str(cfg.get("served_model_name", name)),
        "--host",
        VLLM_HOST,
        "--port",
        str(port),
        "--device",
        str(vision.get("device", "auto")),
        "--dtype",
        str(vision.get("dtype", "auto")),
        "--max-concurrency",
        str(vision.get("max_concurrency", 1)),
        "--threshold",
        str(vision.get("threshold", 0.2)),
        "--box-threshold",
        str(vision.get("box_threshold", 0.45)),
        "--max-candidates",
        str(vision.get("max_candidates", 3000)),
        "--unclip-ratio",
        str(vision.get("unclip_ratio", 1.4)),
        "--max-image-pixels",
        str(vision.get("max_image_pixels", 40_000_000)),
        "--max-image-bytes",
        str(vision.get("max_image_bytes", 25_000_000)),
    ]
    return cmd


def _ensure_port_available(port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((VLLM_HOST, port))
    except OSError as exc:
        raise RuntimeError(f"Port {port} is already in use") from exc
    finally:
        sock.close()


def start(name: str, foreground: bool = False, port: int | None = None) -> dict:
    if _running_state():
        raise RuntimeError(
            "A Transformers vision server is already running. "
            "Use `python3 -m ml.cli vision stop` first."
        )

    cfg = resolve_model(name)
    _check_dependencies()
    port = port or int(cfg.get("default_port", DEFAULT_VISION_PORT))
    _ensure_port_available(port)
    cmd = _build_cmd(name, cfg, port)

    LOGS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(HF_CACHE))
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    if foreground:
        print(f"Running: {' '.join(shlex.quote(value) for value in cmd)}")
        os.execvpe(cmd[0], cmd, env)

    log_fh = open(VISION_LOG, "ab", buffering=0)
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )
    state = {
        "pid": proc.pid,
        "alias": name,
        "hf_id": cfg["hf_id"],
        "served_model_name": cfg.get("served_model_name", name),
        "port": port,
        "backend": "vision",
        "log_path": str(VISION_LOG),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_state(state)
    return state


def stop(timeout: int = 30) -> bool:
    state = _read_state()
    if not state:
        return False
    pid = int(state["pid"])
    if not is_pid_alive(pid):
        _clear_state()
        return False

    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_pid_alive(pid):
            _clear_state()
            return True
        time.sleep(0.5)

    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            os.kill(pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
    _clear_state()
    return True


def status() -> dict:
    state = _running_state()
    if not state:
        return {"running": False}

    api_key = get_api_key()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    ready = False
    try:
        response = requests.get(
            f"http://localhost:{state['port']}/v1/models",
            headers=headers,
            timeout=2,
        )
        ready = response.ok
    except requests.RequestException:
        pass
    return {"running": True, "ready": ready, **state}


def tail_logs(lines: int = 50, follow: bool = False) -> None:
    if not VISION_LOG.exists():
        print(f"No vision log yet at {VISION_LOG}")
        return
    cmd = ["tail", "-n", str(lines)]
    if follow:
        cmd.append("-f")
    cmd.append(str(VISION_LOG))
    subprocess.run(cmd)
