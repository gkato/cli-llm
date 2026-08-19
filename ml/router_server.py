"""Lifecycle manager for the single-port model router."""
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

from ml.config import DATA, LOGS, get_router_config
from ml.state import is_pid_alive

ROUTER_STATE = DATA / "router_server.json"
ROUTER_LOG = LOGS / "router.log"


def _check_dependencies() -> None:
    modules = ("fastapi", "uvicorn", "httpx")
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "Router dependencies are missing: "
            f"{', '.join(missing)}. Re-run `bash setup.sh`."
        )


def _read_state() -> dict | None:
    if not ROUTER_STATE.exists():
        return None
    try:
        return json.loads(ROUTER_STATE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_state(state: dict) -> None:
    ROUTER_STATE.parent.mkdir(parents=True, exist_ok=True)
    ROUTER_STATE.write_text(json.dumps(state, indent=2))


def _clear_state() -> None:
    if ROUTER_STATE.exists():
        ROUTER_STATE.unlink()


def _running_state() -> dict | None:
    state = _read_state()
    if not state:
        return None
    if not is_pid_alive(int(state["pid"])):
        _clear_state()
        return None
    return state


def _build_cmd(config: dict, port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "ml.router_api",
        "--host",
        str(config.get("host", "0.0.0.0")),
        "--port",
        str(port),
    ]


def _ensure_port_available(host: str, port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    except OSError as exc:
        raise RuntimeError(f"Router port {port} is already in use") from exc
    finally:
        sock.close()


def start(foreground: bool = False, port: int | None = None) -> dict:
    if _running_state():
        raise RuntimeError(
            "The model router is already running. "
            "Use `python3 -m ml.cli router stop` first."
        )
    config = get_router_config()
    from ml.router_api import validate_config

    validate_config(config)
    _check_dependencies()
    port = port or int(config.get("port", 8000))
    host = str(config.get("host", "0.0.0.0"))
    _ensure_port_available(host, port)
    cmd = _build_cmd(config, port)

    DATA.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if foreground:
        print(f"Running: {' '.join(shlex.quote(value) for value in cmd)}")
        os.execvpe(cmd[0], cmd, env)

    log_fh = open(ROUTER_LOG, "ab", buffering=0)
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
        "host": host,
        "port": port,
        "backend": "router",
        "log_path": str(ROUTER_LOG),
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
    ready = False
    backends = {}
    try:
        response = requests.get(
            f"http://localhost:{state['port']}/health",
            timeout=5,
        )
        if response.ok:
            payload = response.json()
            ready = bool(payload.get("ready"))
            backends = payload.get("backends") or {}
    except (requests.RequestException, ValueError):
        pass
    return {"running": True, "ready": ready, "backends": backends, **state}


def tail_logs(lines: int = 50, follow: bool = False) -> None:
    if not ROUTER_LOG.exists():
        print(f"No router log yet at {ROUTER_LOG}")
        return
    cmd = ["tail", "-n", str(lines)]
    if follow:
        cmd.append("-f")
    cmd.append(str(ROUTER_LOG))
    subprocess.run(cmd)
