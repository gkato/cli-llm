"""Process lifecycle for the authenticated DSpark allow-list proxy."""

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

from ml.config import DATA, LOGS, get_api_key, get_dspark_proxy_config
from ml.state import is_pid_alive

PROXY_STATE = DATA / "dspark_proxy_server.json"
PROXY_LOG = LOGS / "dspark_proxy.log"


def _check_dependencies() -> None:
    missing = [
        name
        for name in ("fastapi", "uvicorn", "httpx")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(
            "DSpark proxy dependencies are missing: "
            f"{', '.join(missing)}. Re-run `bash setup.sh`."
        )


def _read_state() -> dict | None:
    if not PROXY_STATE.exists():
        return None
    try:
        return json.loads(PROXY_STATE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_state(state: dict) -> None:
    PROXY_STATE.parent.mkdir(parents=True, exist_ok=True)
    PROXY_STATE.write_text(json.dumps(state, indent=2))


def _clear_state() -> None:
    if PROXY_STATE.exists():
        PROXY_STATE.unlink()


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
        "ml.dspark_proxy_api",
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
        raise RuntimeError(f"DSpark proxy port {port} is already in use") from exc
    finally:
        sock.close()


def start(foreground: bool = False, port: int | None = None) -> dict:
    running = _running_state()
    if running:
        return running
    if not get_api_key():
        raise RuntimeError("API_KEY is missing from .env.local; proxy will not start")

    config = get_dspark_proxy_config()
    from ml.dspark_proxy_api import validate_config

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

    log_fh = open(PROXY_LOG, "ab", buffering=0)
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
        "backend": "dspark-proxy",
        "log_path": str(PROXY_LOG),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_state(state)

    deadline = time.time() + 10
    while time.time() < deadline:
        if proc.poll() is not None:
            _clear_state()
            raise RuntimeError(f"DSpark proxy exited early; inspect {PROXY_LOG}")
        try:
            requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
            break
        except requests.RequestException:
            time.sleep(0.25)
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
    try:
        response = requests.get(
            f"http://127.0.0.1:{state['port']}/health",
            timeout=5,
        )
        ready = response.ok and bool(response.json().get("ready"))
    except (requests.RequestException, ValueError):
        pass
    return {"running": True, "ready": ready, **state}


def smoke() -> dict:
    """Verify authenticated access and fail-closed route behavior."""
    config = get_dspark_proxy_config()
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("API_KEY is missing from .env.local")
    port = int(config.get("port", 8000))
    base_url = f"http://127.0.0.1:{port}"
    auth = {"Authorization": f"Bearer {api_key}"}
    try:
        allowed = requests.get(f"{base_url}/v1/models", headers=auth, timeout=10)
        unauthenticated = requests.get(f"{base_url}/v1/models", timeout=10)
        denied = requests.post(
            f"{base_url}/invocations", headers=auth, json={}, timeout=10
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"DSpark proxy is unavailable: {exc}") from exc
    result = {
        "authorized_models": allowed.status_code,
        "unauthenticated_models": unauthenticated.status_code,
        "denied_invocations": denied.status_code,
    }
    if not (
        allowed.ok
        and unauthenticated.status_code == 401
        and denied.status_code == 404
    ):
        raise RuntimeError(f"DSpark proxy safety smoke failed: {result}")
    return result


def tail_logs(lines: int = 50, follow: bool = False) -> None:
    if not PROXY_LOG.exists():
        print(f"No DSpark proxy log yet at {PROXY_LOG}")
        return
    cmd = ["tail", "-n", str(lines)]
    if follow:
        cmd.append("-f")
    cmd.append(str(PROXY_LOG))
    subprocess.run(cmd)
