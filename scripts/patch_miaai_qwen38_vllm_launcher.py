#!/usr/bin/env python3
"""Materialize ml-compute's safety/pinning overlay for MiaAI's vLLM launcher."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {label} anchor, found {count}")
    return text.replace(old, new, 1)


def continued(*lines: str) -> str:
    """Return shell continuation lines with a trailing newline."""
    return "\n".join(f"{line} \\" for line in lines) + "\n"


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} SOURCE DESTINATION", file=sys.stderr)
        return 2

    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    text = source.read_text(encoding="utf-8")

    text = replace_once(
        text,
        'HF_TOKEN="${HF_TOKEN:-}"\n',
        'HF_TOKEN="${HF_TOKEN:-}"\n'
        'HF_REVISION="${HF_REVISION:?HF_REVISION must pin the model snapshot}"\n'
        'HOST_BIND="${HOST_BIND:-127.0.0.1}"\n'
        'REVISION_ARGS=(--revision "$HF_REVISION")\n',
        "runtime override",
    )

    for command in (
        'HF_HOME="$HF_CACHE_DIR" uvx hf download "$MODEL_ID" --cache-dir "$HUB_PATH"',
        'HF_HOME="$HF_CACHE_DIR" huggingface-cli download "$MODEL_ID" --cache-dir "$HUB_PATH"',
        'HF_HOME="$HF_CACHE_DIR" hf download "$MODEL_ID" --cache-dir "$HUB_PATH"',
    ):
        text = replace_once(
            text,
            command,
            f'{command} "${{REVISION_ARGS[@]}}"',
            f"{command.split()[0]} download",
        )

    # In huggingface_hub >= 1.0 the `huggingface-cli` entry point still resolves
    # (so `command -v` succeeds) but is a dead no-op that exits without
    # downloading. Only fall into that branch when the working `hf` CLI is
    # absent, so modern hosts drop through to the `hf download` branch below.
    text = replace_once(
        text,
        "elif command -v huggingface-cli &>/dev/null; then",
        "elif command -v huggingface-cli &>/dev/null && ! command -v hf &>/dev/null; then",
        "huggingface-cli download guard",
    )

    text = replace_once(
        text,
        'HEAD_HAS=$( [[ -d "$HUB_PATH/models--${ORG}--${NAME}" ]] && echo 1 || echo 0 )',
        'HEAD_HAS=$( [[ -f "$HUB_PATH/models--${ORG}--${NAME}/snapshots/$HF_REVISION/config.json" ]] && echo 1 || echo 0 )',
        "head snapshot detection",
    )
    text = replace_once(
        text,
        'WORKER_HAS=$(ssh_worker "test -d \'$REMOTE_HUB/models--${ORG}--${NAME}\' && echo 1 || echo 0" 2>/dev/null || echo 0)',
        'WORKER_HAS=$(ssh_worker "test -f \'$REMOTE_HUB/models--${ORG}--${NAME}/snapshots/$HF_REVISION/config.json\' && echo 1 || echo 0" 2>/dev/null || echo 0)',
        "worker snapshot detection",
    )

    model_anchor = continued(
        "    $MODEL_ID",
        "    --served-model-name $SERVED_MODEL_NAME",
    )
    pinned_model_anchor = continued(
        "    $MODEL_ID",
        "    --revision $HF_REVISION",
        "    --served-model-name $SERVED_MODEL_NAME",
    )
    count = text.count(model_anchor)
    if count != 2:
        raise RuntimeError(f"expected two vLLM model anchors, found {count}")
    text = text.replace(model_anchor, pinned_model_anchor)

    text = replace_once(
        text,
        continued("    --host 0.0.0.0"),
        continued("    --host $HOST_BIND"),
        "head bind",
    )

    destination.write_text(text, encoding="utf-8")
    os.chmod(destination, source.stat().st_mode | 0o100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
