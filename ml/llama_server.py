"""llama.cpp serving backend.

`llama-server` is a local process exposing the same OpenAI-compatible /v1
API as vLLM, but serving GGUF weights. It shares vLLM's single-process
lifecycle: one server at a time, tracked in DATA/server.json, managed by
the top-level `serve` / `stop` / `status` / `logs` commands.

Models can be a registry alias (registry/llama_models.yaml), a HuggingFace
GGUF repo (`org/repo` or `org/repo:QUANT`, downloaded via `-hf`), or a local
`.gguf` file path.
"""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import requests

from ml.config import (
    DATA,
    HF_CACHE,
    LOGS,
    VLLM_HOST,
    VLLM_PORT,
    get_api_key,
    get_llama_models,
)
from ml.state import running_state, write_state

LLAMA_LOG = LOGS / "llama.log"


def _resolve(model_id: str) -> dict:
    """Resolve a model_id to a llama.cpp config dict.

    Order: registry alias → local .gguf path → HuggingFace GGUF repo.
    """
    registry = get_llama_models()
    if model_id in registry:
        cfg = dict(registry[model_id])
        cfg.setdefault("served_name", model_id)
        return cfg

    p = Path(model_id).expanduser()
    if model_id.endswith(".gguf") or p.exists():
        return {"gguf_path": str(p), "served_name": p.stem}

    # `org/repo` or `org/repo:QUANT` → let llama.cpp fetch it via -hf.
    if "/" in model_id:
        served = model_id.split(":", 1)[0]
        return {"hf_repo": model_id, "served_name": served}

    raise ValueError(
        f"Unknown llama model: {model_id!r}. Use a registered alias "
        f"({', '.join(registry) or 'none registered'}), a HuggingFace GGUF "
        f"repo like 'org/repo:Q4_K_M', or a local '/path/to/model.gguf'."
    )


def _build_cmd(cfg: dict, port: int) -> list[str]:
    cmd = [
        "llama-server",
        "--host", VLLM_HOST,
        "--port", str(port),
    ]

    if cfg.get("hf_repo"):
        cmd += ["-hf", str(cfg["hf_repo"])]
    elif cfg.get("gguf_path"):
        cmd += ["-m", str(cfg["gguf_path"])]
    else:
        raise ValueError("llama config must set either 'hf_repo' or 'gguf_path'.")

    # 999 offloads every layer to GPU; 0 keeps it on CPU.
    n_gpu = cfg.get("n_gpu_layers", 999)
    cmd += ["-ngl", str(n_gpu)]

    if (ctx := cfg.get("ctx_size")):
        cmd += ["-c", str(ctx)]
    if (served := cfg.get("served_name")):
        cmd += ["-a", str(served)]
    if (threads := cfg.get("threads")):
        cmd += ["-t", str(threads)]

    for extra in (cfg.get("extra_args") or []):
        cmd.append(str(extra))

    if (api_key := get_api_key()):
        cmd += ["--api-key", api_key]

    return cmd


def start(
    model_id: str,
    foreground: bool = False,
    port: int | None = None,
    n_gpu_layers: int | None = None,
    ctx_size: int | None = None,
) -> dict:
    """Start llama-server for the given model. Returns server state dict."""
    if running_state():
        raise RuntimeError("A server is already running. Use `ml-compute stop` first.")

    cfg = _resolve(model_id)  # validate input before environmental checks

    if not shutil.which("llama-server"):
        raise RuntimeError(
            "llama-server not found on PATH. Build llama.cpp with CUDA "
            "(cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release) "
            "and add build/bin to PATH, or `brew install llama.cpp`."
        )

    if n_gpu_layers is not None:
        cfg["n_gpu_layers"] = n_gpu_layers
    if ctx_size is not None:
        cfg["ctx_size"] = ctx_size

    port = port or VLLM_PORT
    cmd = _build_cmd(cfg, port)
    source = cfg.get("hf_repo") or cfg.get("gguf_path")

    LOGS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    env.setdefault("HF_HOME", str(HF_CACHE))

    if foreground:
        print(f"Running: {' '.join(shlex.quote(c) for c in cmd)}")
        os.execvpe(cmd[0], cmd, env)

    log_fh = open(LLAMA_LOG, "ab", buffering=0)
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
        model_id,
        source,
        port,
        backend="llama",
        log_path=str(LLAMA_LOG),
    )
    return {
        "pid": proc.pid,
        "model_alias": model_id,
        "hf_id": source,
        "served_name": cfg.get("served_name"),
        "port": port,
        "log_path": str(LLAMA_LOG),
    }


def status() -> dict:
    """Return a status dict describing the current llama.cpp server."""
    state = running_state()
    if not state:
        return {"running": False}

    port = state["port"]
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
        "backend": "llama",
        "pid": state["pid"],
        "model_alias": state["model_alias"],
        "hf_id": state["hf_id"],
        "port": port,
        "started_at": state["started_at"],
        "log_path": state.get("log_path") or str(LLAMA_LOG),
    }


def tail_logs(lines: int = 50, follow: bool = False) -> None:
    """Print the last N lines of the llama.cpp log; optionally follow."""
    if not LLAMA_LOG.exists():
        print(f"No log yet at {LLAMA_LOG}")
        return
    if follow:
        subprocess.run(["tail", "-n", str(lines), "-f", str(LLAMA_LOG)])
    else:
        subprocess.run(["tail", "-n", str(lines), str(LLAMA_LOG)])
