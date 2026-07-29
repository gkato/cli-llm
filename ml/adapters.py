import json
from datetime import datetime
from pathlib import Path

from ml.config import DATA

ADAPTERS = DATA / "adapters"


def save_metadata(adapter_name: str, **fields) -> Path:
    path = ADAPTERS / adapter_name / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "adapter_name": adapter_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        **fields,
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    return path


def list_adapters() -> list[dict]:
    if not ADAPTERS.exists():
        return []
    out = []
    for d in sorted(ADAPTERS.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        if meta_path.exists():
            out.append(json.loads(meta_path.read_text()))
        else:
            out.append({"adapter_name": d.name, "created_at": "unknown"})
    return out


def path_for(adapter_name: str) -> Path:
    p = ADAPTERS / adapter_name
    if not p.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_name}")
    return p
