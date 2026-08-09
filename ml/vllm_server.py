import json
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

VLLM_LOG = LOGS / "vllm.log"


def _normalize_rope_scaling(value) -> str | None:
    """Accept either a JSON string or a dict in the registry; emit a JSON string for vLLM."""
    if value is None:
        return None
    if isinstance(value, str):
        # Validate it parses, then re-serialize compactly
        return json.dumps(json.loads(value))
    if isinstance(value, dict):
        return json.dumps(value)
    raise ValueError(f"rope_scaling must be a JSON string or dict, got {type(value).__name__}")


def _normalize_json_option(name: str, value) -> str | None:
    """Accept a JSON string or YAML mapping and emit compact JSON for vLLM."""
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        return json.dumps(value, separators=(",", ":"))
    raise ValueError(f"{name} must be a JSON string or dict, got {type(value).__name__}")


def _build_cmd(model_alias: str, hf_id: str, port: int) -> list[str]:
    cfg = get_models().get(model_alias, {})
    cmd = [
        "vllm", "serve", hf_id,
        "--host", VLLM_HOST,
        "--port", str(port),
        "--download-dir", str(HF_CACHE / "hub"),
    ]

    if (max_len := cfg.get("max_model_len")):
        cmd += ["--max-model-len", str(max_len)]
    if (dtype := cfg.get("dtype")) and dtype != "auto":
        cmd += ["--dtype", str(dtype)]
    if (quant := cfg.get("quantization")):
        cmd += ["--quantization", str(quant)]
    if (gpu_mem := cfg.get("gpu_memory_utilization")):
        cmd += ["--gpu-memory-utilization", str(gpu_mem)]
    if (max_seqs := cfg.get("max_num_seqs")):
        cmd += ["--max-num-seqs", str(max_seqs)]
    if (max_batched_tokens := cfg.get("max_num_batched_tokens")):
        cmd += ["--max-num-batched-tokens", str(max_batched_tokens)]

    if cfg.get("enable_prefix_caching"):
        cmd += ["--enable-prefix-caching"]
    if cfg.get("enable_chunked_prefill"):
        cmd += ["--enable-chunked-prefill"]

    if (speculative := _normalize_json_option(
        "speculative_config", cfg.get("speculative_config")
    )):
        cmd += ["--speculative-config", speculative]

    # YaRN / context extension. The registry value can be a YAML mapping
    # or an inline JSON string; vLLM expects a JSON string on the CLI.
    if (rope := _normalize_rope_scaling(cfg.get("rope_scaling"))):
        cmd += ["--rope-scaling", rope]

    # Reasoning parser: routes <think>…</think> into a separate
    # `reasoning_content` field so `content` is the clean final answer.
    # Common values: "qwen3", "deepseek_r1". Set to "qwen3" for Qwen3-*.
    # Different vLLM versions vary on whether --enable-reasoning is required
    # alongside --reasoning-parser; recent versions auto-enable when the
    # parser is set, and older 0.8.x reject --enable-reasoning as unknown.
    # Setting just --reasoning-parser works on both.
    if (parser := cfg.get("reasoning_parser")):
        cmd += ["--reasoning-parser", str(parser)]

    if (parser := cfg.get("tool_call_parser")):
        cmd += ["--tool-call-parser", str(parser)]
    if cfg.get("enable_auto_tool_choice"):
        cmd += ["--enable-auto-tool-choice"]
    if cfg.get("language_model_only"):
        cmd += ["--language-model-only"]

    # LoRA adapters: a list of {name, path, [rank]} entries. vLLM applies the
    # adapter on top of the base at runtime; clients request the LoRA by its
    # `name` as the OpenAI `model` field.
    loras = cfg.get("loras") or []
    if loras:
        cmd += ["--enable-lora"]
        max_rank = max((int(l.get("rank", 16)) for l in loras), default=16)
        cmd += ["--max-lora-rank", str(max_rank)]
        cmd += ["--max-loras", str(len(loras))]
        # vLLM expects `--lora-modules name=path [name=path ...]`
        lora_specs = [f"{l['name']}={l['path']}" for l in loras]
        cmd += ["--lora-modules", *lora_specs]

    # Extra raw flags for any uncommon vLLM option (list of strings).
    for extra in (cfg.get("extra_args") or []):
        cmd.append(str(extra))

    api_key = get_api_key()
    if api_key:
        cmd += ["--api-key", api_key]

    return cmd


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
