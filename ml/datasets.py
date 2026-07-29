import json
import random
from pathlib import Path

from ml.config import DATASETS, ROOT


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def load(dataset_key: str) -> list[dict]:
    cfg = DATASETS[dataset_key]
    path = _resolve(cfg["path"])
    if not path.exists():
        raise FileNotFoundError(f"Dataset file missing: {path}")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def validate(dataset_key: str) -> dict:
    cfg = DATASETS[dataset_key]
    rows = load(dataset_key)
    required = {"alpaca": ["instruction", "output"]}.get(cfg["format"])
    if required is None:
        raise ValueError(f"Unknown format: {cfg['format']}")
    for i, row in enumerate(rows):
        for k in required:
            if k not in row:
                raise ValueError(f"Row {i} missing field '{k}' in {dataset_key}")
    return {"count": len(rows), "format": cfg["format"]}


def split(dataset_key: str, seed: int = 42) -> tuple[list, list, list]:
    cfg = DATASETS[dataset_key]
    rows = load(dataset_key)
    random.Random(seed).shuffle(rows)
    n = len(rows)
    a, b, _ = cfg["split"]
    train = rows[: int(n * a)]
    val = rows[int(n * a) : int(n * (a + b))]
    test = rows[int(n * (a + b)) :]
    return train, val, test
