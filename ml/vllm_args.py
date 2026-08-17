"""Shared vLLM ``serve`` argument construction.

Both the host-process backend and the Docker backend consume entries from
``registry/models.yaml``. Keeping the translation in one place prevents the
two launch paths from drifting as registry fields are added.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def _normalize_json_option(name: str, value: object) -> str | None:
    """Accept a JSON string or mapping and return compact JSON for vLLM."""
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, Mapping):
        return json.dumps(dict(value), separators=(",", ":"))
    raise ValueError(
        f"{name} must be a JSON string or mapping, got {type(value).__name__}"
    )


def build_vllm_serve_args(
    hf_id: str,
    cfg: Mapping[str, object],
    *,
    host: str,
    port: int,
    download_dir: str | Path | None = None,
    api_key: str | None = None,
) -> list[str]:
    """Translate one model registry entry into arguments after ``vllm serve``."""
    args = [hf_id, "--host", host, "--port", str(port)]

    if download_dir is not None:
        args += ["--download-dir", str(download_dir)]
    if served_name := cfg.get("served_model_name"):
        args += ["--served-model-name", str(served_name)]
    if max_len := cfg.get("max_model_len"):
        args += ["--max-model-len", str(max_len)]
    if (dtype := cfg.get("dtype")) and dtype != "auto":
        args += ["--dtype", str(dtype)]
    if quant := cfg.get("quantization"):
        args += ["--quantization", str(quant)]
    if gpu_mem := cfg.get("gpu_memory_utilization"):
        args += ["--gpu-memory-utilization", str(gpu_mem)]
    if max_seqs := cfg.get("max_num_seqs"):
        args += ["--max-num-seqs", str(max_seqs)]
    if max_batched_tokens := cfg.get("max_num_batched_tokens"):
        args += ["--max-num-batched-tokens", str(max_batched_tokens)]

    if cfg.get("enable_prefix_caching"):
        args.append("--enable-prefix-caching")
    if cfg.get("enable_chunked_prefill"):
        args.append("--enable-chunked-prefill")

    if speculative := _normalize_json_option(
        "speculative_config", cfg.get("speculative_config")
    ):
        args += ["--speculative-config", speculative]
    if rope := _normalize_json_option("rope_scaling", cfg.get("rope_scaling")):
        args += ["--rope-scaling", rope]

    if parser := cfg.get("reasoning_parser"):
        args += ["--reasoning-parser", str(parser)]
    if parser := cfg.get("tool_call_parser"):
        args += ["--tool-call-parser", str(parser)]
    if cfg.get("enable_auto_tool_choice"):
        args.append("--enable-auto-tool-choice")
    if cfg.get("language_model_only"):
        args.append("--language-model-only")

    loras = cfg.get("loras") or []
    if loras:
        if not isinstance(loras, list):
            raise ValueError("loras must be a list of adapter mappings")
        args.append("--enable-lora")
        max_rank = max((int(adapter.get("rank", 16)) for adapter in loras), default=16)
        args += ["--max-lora-rank", str(max_rank), "--max-loras", str(len(loras))]
        args += [
            "--lora-modules",
            *(f"{adapter['name']}={adapter['path']}" for adapter in loras),
        ]

    args.extend(str(value) for value in (cfg.get("extra_args") or []))

    if api_key:
        args += ["--api-key", api_key]

    return args
