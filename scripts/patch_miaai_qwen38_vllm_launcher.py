#!/usr/bin/env python3
"""Materialize ml-compute's pinning and private-bind overlay for MiaAI vLLM."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {label} anchor, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    if len(sys.argv) != 5:
        print(
            f"usage: {sys.argv[0]} START_SOURCE START_DEST DOWNLOAD_SOURCE DOWNLOAD_DEST",
            file=sys.stderr,
        )
        return 2

    start_source, start_destination, download_source, download_destination = map(
        Path, sys.argv[1:]
    )
    text = start_source.read_text(encoding="utf-8")
    text = replace_once(
        text,
        'HF_TOKEN="${HF_TOKEN:-}"\n',
        'HF_TOKEN="${HF_TOKEN:-}"\n'
        'HF_REVISION="${HF_REVISION:?HF_REVISION must pin the model snapshot}"\n'
        'HOST_BIND="${HOST_BIND:-127.0.0.1}"\n',
        "runtime overrides",
    )
    text = replace_once(
        text,
        '    "$SCRIPT_DIR/download.sh" "$MODEL_ID"',
        '    "$SCRIPT_DIR/download.ml-compute.sh" "$MODEL_ID" "$HF_REVISION"',
        "pinned downloader call",
    )
    text = replace_once(
        text,
        '    VLLM_ARGS+=("--served-model-name" "$SERVED_MODEL_NAME")',
        '    VLLM_ARGS+=("--revision" "$HF_REVISION")\n'
        '    VLLM_ARGS+=("--served-model-name" "$SERVED_MODEL_NAME")',
        "vLLM revision",
    )
    text = replace_once(
        text,
        'PLE_CONFIG_DIR="$MODEL_DIR"\n'
        'if [[ ! -f "$PLE_CONFIG_DIR/config.json" ]]; then\n'
        '    PLE_CONFIG_DIR=$(ls -d "$HEAD_MODEL_PATH"/snapshots/*/ 2>/dev/null | head -1)\n'
        'fi',
        'PLE_CONFIG_DIR="$MODEL_DIR/snapshots/$HF_REVISION"\n'
        '[[ -f "$PLE_CONFIG_DIR/config.json" ]] || err "Pinned snapshot missing: $PLE_CONFIG_DIR"',
        "pinned checkpoint configuration",
    )
    text = replace_once(
        text,
        "    --host 0.0.0.0 \\\n",
        "    --host $HOST_BIND \\\n",
        "head bind",
    )

    uvm_anchor = "    --device /dev/infiniband:/dev/infiniband \\\n"
    if text.count(uvm_anchor) != 2:
        raise RuntimeError("expected two infiniband device anchors")
    text = text.replace(
        uvm_anchor,
        uvm_anchor + "    --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \\\n",
    )
    start_destination.write_text(text, encoding="utf-8")
    os.chmod(start_destination, start_source.stat().st_mode | 0o100)

    download = download_source.read_text(encoding="utf-8")
    download = replace_once(
        download,
        'MODEL_ID="${MODEL_ID:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"\n',
        'MODEL_ID="${MODEL_ID:-nvidia/Qwen3.8-Flash-Next-NVFP4}"\n'
        'HF_REVISION="${2:?model revision is required}"\n'
        'set -- "$1"\n',
        "download revision",
    )
    for command in (
        'HF_HOME="$HF_CACHE_DIR" uvx hf download "$MODEL_ID" --cache-dir "$HUB_PATH"',
        'HF_HOME="$HF_CACHE_DIR" huggingface-cli download "$MODEL_ID" --cache-dir "$HUB_PATH"',
        'HF_HOME="$HF_CACHE_DIR" hf download "$MODEL_ID" --cache-dir "$HUB_PATH"',
    ):
        download = replace_once(
            download,
            command,
            f'{command} --revision "$HF_REVISION"',
            "revision-pinned download",
        )
    download = replace_once(
        download,
        "elif command -v huggingface-cli &>/dev/null; then",
        "elif command -v huggingface-cli &>/dev/null && ! command -v hf &>/dev/null; then",
        "huggingface-cli guard",
    )
    download_destination.write_text(download, encoding="utf-8")
    os.chmod(download_destination, download_source.stat().st_mode | 0o100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
