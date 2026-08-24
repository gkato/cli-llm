#!/usr/bin/env python3
"""Validate DSpark KV capacity from vLLM's model-aware startup result."""

from __future__ import annotations

import os
import re
import subprocess
import urllib.request


CACHE_CONFIG_METRIC = re.compile(r"vllm:cache_config_info\{([^}]*)\}")
METRIC_LABEL = re.compile(r'(\w+)="([^"]*)"')
STARTUP_CAPACITY = re.compile(r"GPU KV cache size:\s*([\d,]+)\s+tokens")
STARTUP_CONCURRENCY = re.compile(
    r"Maximum concurrency for\s*([\d,]+)\s+tokens per request:\s*([\d.]+)x"
)


def parse_startup_capacity(logs: str, required: int) -> tuple[int | None, float | None]:
    capacities = STARTUP_CAPACITY.findall(logs)
    capacity = int(capacities[-1].replace(",", "")) if capacities else None

    concurrency = None
    for token_text, concurrency_text in STARTUP_CONCURRENCY.findall(logs):
        if int(token_text.replace(",", "")) == required:
            concurrency = float(concurrency_text)
    return capacity, concurrency


def startup_capacity_fits(
    capacity: int | None, concurrency: float | None, required: int
) -> bool:
    if concurrency is not None:
        return concurrency >= 1.0
    return capacity is not None and capacity >= required


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
    container = os.environ["DSPARK_CONTAINER_NAME"]
    api_key = os.environ.get("DSPARK_METRICS_API_KEY", "")

    try:
        completed = subprocess.run(
            ["docker", "logs", container],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"[dspark] ERROR: could not read startup logs for {container}: {error}")
        return 1

    startup_capacity, startup_concurrency = parse_startup_capacity(
        completed.stdout, required
    )
    if startup_capacity is None and startup_concurrency is None:
        print(
            "[dspark] ERROR: vLLM startup KV-capacity result is missing; "
            "refusing to rely on the known-broken DeepSeek V4 cache metric"
        )
        return 1

    details = []
    if startup_capacity is not None:
        details.append(f"{startup_capacity:,} startup tokens")
    if startup_concurrency is not None:
        details.append(f"{startup_concurrency:.2f}x at {required:,}")
    print(f"[dspark] Startup KV capacity: {'; '.join(details)}")

    # vLLM issues #50456 and #51163: EngineCoreReadyResponse can overwrite the
    # correct per-worker hybrid-cache values before cache_config_info is exposed.
    # Report the metric discrepancy for diagnosis, but never gate on it.
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            metrics = response.read().decode("utf-8", errors="replace")
        metric_capacity, metric_details = parse_cache_capacity(metrics)
        print(
            f"[dspark] Diagnostic /metrics capacity: {metric_capacity:,} tokens "
            f"({metric_details}; not used for this gate)"
        )
        if startup_capacity is not None and metric_capacity != startup_capacity:
            print(
                "[dspark] WARNING: ignoring the known DeepSeek V4 hybrid-cache "
                "startup/metrics capacity mismatch"
            )
    except (OSError, ValueError) as error:
        print(f"[dspark] WARNING: could not read diagnostic cache metric: {error}")

    if not startup_capacity_fits(startup_capacity, startup_concurrency, required):
        print(
            f"[dspark] ERROR: startup KV capacity cannot hold one {required:,}-token "
            "request; increase the constrained node's allocation or lower MAX_MODEL_LEN"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
