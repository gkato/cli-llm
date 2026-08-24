#!/usr/bin/env python3
"""Validate DSpark KV capacity from vLLM's authoritative metrics label."""

from __future__ import annotations

import os
import re
import urllib.request


CACHE_CONFIG_METRIC = re.compile(r"vllm:cache_config_info\{([^}]*)\}")
METRIC_LABEL = re.compile(r'(\w+)="([^"]*)"')


def parse_cache_capacity(metrics: str) -> tuple[int, str]:
    match = CACHE_CONFIG_METRIC.search(metrics)
    if not match:
        raise ValueError("vLLM cache_config_info metric is missing")
    labels = dict(METRIC_LABEL.findall(match.group(1)))

    direct = labels.get("kv_cache_size_tokens")
    if direct is not None:
        try:
            capacity = int(float(direct))
        except ValueError as error:
            raise ValueError(
                f"invalid kv_cache_size_tokens in cache_config_info: {labels}"
            ) from error
        details = "vLLM kv_cache_size_tokens"
        if "num_gpu_blocks" in labels and "block_size" in labels:
            details += (
                f"; {labels['num_gpu_blocks']} groups; "
                f"reported block_size {labels['block_size']}"
            )
        return capacity, details

    # Compatibility fallback for older/non-hybrid vLLM builds that do not
    # publish the direct token-capacity label. This multiplication is not valid
    # for the DSpark hybrid metric, which is why the direct label always wins.
    try:
        block_size = int(float(labels["block_size"]))
        gpu_blocks = int(float(labels["num_gpu_blocks"]))
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid cache_config_info labels: {labels}") from error
    return block_size * gpu_blocks, f"legacy {gpu_blocks} blocks x {block_size}"


def main() -> int:
    url = os.environ["DSPARK_METRICS_URL"]
    required = int(os.environ["DSPARK_REQUIRED_TOKENS"])
    api_key = os.environ.get("DSPARK_METRICS_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            metrics = response.read().decode("utf-8", errors="replace")
        capacity, details = parse_cache_capacity(metrics)
    except (OSError, ValueError) as error:
        print(f"[dspark] ERROR: {error}")
        return 1

    print(f"[dspark] Measured KV capacity: {capacity:,} tokens ({details})")
    if capacity < required:
        print(
            f"[dspark] ERROR: KV capacity {capacity:,} is below MAX_MODEL_LEN "
            f"{required:,}; increase the constrained node's allocation or lower "
            "MAX_MODEL_LEN"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
