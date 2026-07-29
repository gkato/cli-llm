import os
import secrets
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env.local")

_data_dir = os.getenv("DATA_DIR", "data")
_registry_dir = os.getenv("REGISTRY_DIR", "registry")

DATA = Path(_data_dir) if Path(_data_dir).is_absolute() else ROOT / _data_dir
REGISTRY = Path(_registry_dir) if Path(_registry_dir).is_absolute() else ROOT / _registry_dir

LOGS = DATA / "logs"
HF_CACHE = Path(os.getenv("HF_HOME", str(DATA / "hf_cache")))

VLLM_HOST = os.getenv("VLLM_HOST", "0.0.0.0")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))


def _load_yaml(name: str) -> dict:
    path = REGISTRY / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def get_models() -> dict:
    return _load_yaml("models")["models"]


def get_llama_models() -> dict:
    """GGUF model registry for the llama.cpp backend (registry/llama_models.yaml)."""
    return _safe_load("llama_models")


def _safe_load(name: str) -> dict:
    try:
        return _load_yaml(name).get(name, {}) or {}
    except FileNotFoundError:
        return {}


# Back-compat: train.py and datasets.py expect module-level dicts.
# These are populated lazily on import and only used by the optional
# fine-tuning path.
MODELS = get_models()
DATASETS = _safe_load("datasets")


def get_api_key() -> str | None:
    return os.getenv("API_KEY")


def generate_api_key() -> str:
    return f"sk-mlc-{secrets.token_urlsafe(24)}"


def write_env_var(key: str, value: str) -> None:
    """Write or update a single key=value pair in .env.local."""
    env_path = ROOT / ".env.local"
    lines: list[str] = []
    if env_path.exists():
        with open(env_path) as f:
            lines = f.readlines()

    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            found = True
            break
    if not found:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f"{key}={value}\n")

    with open(env_path, "w") as f:
        f.writelines(lines)
    os.environ[key] = value
