import os
import shutil
from pathlib import Path

from ml.config import HF_CACHE, get_models


def resolve_hf_id(name: str) -> str:
    """Accept either a registry alias (e.g. 'qwen2.5-7b') or a raw HF ID."""
    models = get_models()
    if name in models:
        return models[name]["hf_id"]
    if "/" in name:
        return name
    raise ValueError(
        f"Unknown model: {name!r}. Use a registered alias ({', '.join(models)}) "
        f"or a HuggingFace ID like 'org/repo'."
    )


def model_cache_dir(hf_id: str) -> Path:
    """Return the local cache dir for a given HF ID."""
    safe = "models--" + hf_id.replace("/", "--")
    return HF_CACHE / "hub" / safe


def is_downloaded(hf_id: str) -> bool:
    p = model_cache_dir(hf_id)
    if not p.exists():
        return False
    snapshots = p / "snapshots"
    return snapshots.exists() and any(snapshots.iterdir())


def pull(name: str) -> str:
    """Download a model from HuggingFace into the local cache. Returns the HF ID."""
    hf_id = resolve_hf_id(name)
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HOME", str(HF_CACHE))

    from huggingface_hub import snapshot_download

    token = os.getenv("HF_TOKEN") or None
    snapshot_download(
        repo_id=hf_id,
        cache_dir=str(HF_CACHE / "hub"),
        token=token,
        allow_patterns=[
            "*.safetensors", "*.json", "*.txt",
            "tokenizer*", "*.model", "*.tiktoken",
        ],
    )
    return hf_id


def remove(name: str) -> str:
    """Delete a model from local cache. Returns the HF ID removed."""
    hf_id = resolve_hf_id(name)
    p = model_cache_dir(hf_id)
    if not p.exists():
        raise FileNotFoundError(f"{hf_id} is not in the cache at {p}")
    shutil.rmtree(p)
    return hf_id


def _dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path, followlinks=False):
        for name in files:
            fp = Path(root) / name
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def list_local() -> list[dict]:
    """List models present in the cache, with sizes."""
    hub = HF_CACHE / "hub"
    if not hub.exists():
        return []
    out: list[dict] = []
    for entry in sorted(hub.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("models--"):
            continue
        hf_id = entry.name.removeprefix("models--").replace("--", "/", 1).replace("--", "/")
        out.append({"hf_id": hf_id, "path": str(entry), "size_bytes": _dir_size(entry)})
    return out


def format_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"
