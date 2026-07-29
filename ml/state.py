import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ml.config import DATA

STATE_FILE = DATA / "server.json"


def write_state(
    pid: int,
    model_alias: str,
    hf_id: str,
    port: int,
    backend: str = "vllm",
    log_path: str | None = None,
) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    state = {
        "pid": pid,
        "model_alias": model_alias,
        "hf_id": hf_id,
        "port": port,
        "backend": backend,
        "log_path": log_path,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))


def read_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def clear_state() -> None:
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def running_state() -> dict | None:
    """Return state dict if a server is actually running, else None (and cleans stale)."""
    state = read_state()
    if not state:
        return None
    if not is_pid_alive(state["pid"]):
        clear_state()
        return None
    return state
