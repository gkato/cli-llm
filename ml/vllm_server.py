import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

import requests

from ml.config import DATA, HF_CACHE, LOGS, VLLM_HOST, VLLM_PORT, get_api_key, get_models
from ml.models import resolve_hf_id
from ml.state import clear_state, is_pid_alive, read_state, running_state, write_state
from ml.vllm_args import build_vllm_serve_args

VLLM_LOG = LOGS / "vllm.log"


def _build_cmd(model_alias: str, hf_id: str, port: int) -> list[str]:
    cfg = get_models().get(model_alias, {})
    backend = cfg.get("serve_backend", "vllm")
    if backend != "vllm":
        if backend == "docker":
            raise RuntimeError(
                f"{model_alias!r} uses the Docker vLLM backend. Run "
                f"`python3 -m ml.cli docker serve {model_alias}`."
            )
        recipe = cfg.get("external_recipe", f"scripts/{backend}.sh")
        raise RuntimeError(
            f"{model_alias!r} uses the {backend!r} backend, not local vLLM. "
            f"Run `python3 -m ml.cli {backend} setup`, then "
            f"`python3 -m ml.cli {backend} start` (recipe: {recipe})."
        )
    return [
        "vllm",
        "serve",
        *build_vllm_serve_args(
            hf_id,
            cfg,
            host=VLLM_HOST,
            port=port,
            download_dir=HF_CACHE / "hub",
            api_key=get_api_key(),
        ),
    ]


def start(model_alias: str, foreground: bool = False, port: int | None = None) -> dict:
    """Start vLLM serving the given model. Returns server state dict."""
    if running_state():
        raise RuntimeError("A server is already running. Use `ml-compute stop` first.")

    hf_id = resolve_hf_id(model_alias)
    port = port or VLLM_PORT
    cmd = _build_cmd(model_alias, hf_id, port)

    LOGS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    env.setdefault("HF_HOME", str(HF_CACHE))

    # Blackwell (sm_120+) needs hints for vLLM's FlashInfer JIT compiler:
    # without these, FlashInfer's CUDA arch detection bails out with
    # "FlashInfer requires GPUs with sm75 or higher" and the engine
    # never starts. Disabling the FlashInfer sampler is the safest path
    # until vLLM ships optimized Blackwell sampling kernels.
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            if cap[0] >= 12:
                env.setdefault("TORCH_CUDA_ARCH_LIST", f"{cap[0]}.{cap[1]}+PTX")
                env.setdefault("VLLM_FLASHINFER_FORCE_TARGET", f"sm_{cap[0]}{cap[1]}")
                env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    except Exception:
        pass

    if foreground:
        print(f"Running: {' '.join(shlex.quote(c) for c in cmd)}")
        os.execvpe(cmd[0], cmd, env)

    log_fh = open(VLLM_LOG, "ab", buffering=0)
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )
    write_state(
        proc.pid,
        model_alias if model_alias in get_models() else hf_id,
        hf_id,
        port,
        backend="vllm",
        log_path=str(VLLM_LOG),
    )
    return {
        "pid": proc.pid,
        "model_alias": model_alias,
        "hf_id": hf_id,
        "port": port,
        "log_path": str(VLLM_LOG),
    }


def stop(timeout: int = 30) -> bool:
    """Stop the running vLLM server. Returns True if stopped, False if nothing running."""
    state = read_state()
    if not state:
        return False
    pid = state["pid"]
    if not is_pid_alive(pid):
        clear_state()
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
            clear_state()
            return True
        time.sleep(0.5)

    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            os.kill(pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
    clear_state()
    return True


def status() -> dict:
    """Return a status dict describing the current server."""
    state = running_state()
    if not state:
        return {"running": False}

    port = state["port"]
    ready = False
    api_key = get_api_key()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = requests.get(f"http://localhost:{port}/v1/models", headers=headers, timeout=2)
        ready = r.ok
    except requests.RequestException:
        ready = False

    return {
        "running": True,
        "ready": ready,
        "backend": "vllm",
        "pid": state["pid"],
        "model_alias": state["model_alias"],
        "hf_id": state["hf_id"],
        "port": port,
        "started_at": state["started_at"],
        "log_path": state.get("log_path") or str(VLLM_LOG),
    }


def tail_logs(lines: int = 50, follow: bool = False) -> None:
    """Print the last N lines of the vLLM log; optionally follow."""
    if not VLLM_LOG.exists():
        print(f"No log yet at {VLLM_LOG}")
        return
    if follow:
        subprocess.run(["tail", "-n", str(lines), "-f", str(VLLM_LOG)])
    else:
        subprocess.run(["tail", "-n", str(lines), str(VLLM_LOG)])
