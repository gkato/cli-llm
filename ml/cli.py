import subprocess
from pathlib import Path

import click

from ml.config import (
    HF_CACHE,
    VLLM_HOST,
    VLLM_PORT,
    generate_api_key,
    get_api_key,
    get_models,
    write_env_var,
)


CLI_EPILOG = """
\b
COMMON WORKFLOWS

\b
  Local vLLM serving (default):
    ml.cli models pull qwen3.5-9b-fp8     # download weights
    ml.cli serve qwen3.5-9b-fp8           # start vLLM
    ml.cli status                         # check readiness
    ml.cli stop                           # shut down

\b
  Local llama.cpp serving (GGUF):
    ml.cli serve llama qwen3.5-4b-gguf    # start llama-server
    ml.cli status                         # check readiness (shared)
    ml.cli stop                           # shut down (shared)

\b
  Dedicated vLLM image serving (Docker):
    ml.cli docker serve unlimited-ocr     # start registry image
    ml.cli docker status                  # check readiness
    ml.cli docker logs -f                 # follow container logs
    ml.cli docker stop                    # stop + remove container

\b
  Lightweight Transformers vision serving:
    ml.cli vision serve pp-ocrv6-medium-det --port 8103
    ml.cli vision status
    ml.cli vision stop

\b
  Single-port model router:
    ml.cli router serve --port 8000
    ml.cli router status
    ml.cli router logs -f

\b
  DSpark safety proxy:
    ml.cli dspark-proxy serve              # authenticated allow-list on :8000
    ml.cli dspark-proxy status
    ml.cli dspark-proxy logs -f

\b
  NVIDIA NIM container serving (preferred on GB10):
    ml.cli nim models                     # list NIM catalog
    ml.cli nim serve qwen3.5-27b-nim      # start NIM container
    ml.cli nim status                     # check readiness
    ml.cli nim stop                       # shut down

\b
  Two-node DSpark serving (DeepSeek V4 Flash 0731):
    ml.cli dspark setup                   # configure and validate both nodes
    ml.cli dspark build                   # pull/verify pinned Anemll image
    ml.cli dspark download                # download and mirror model weights
    ml.cli dspark start                   # worker-first TP=2 launch

\b
  Two-node SGLang serving (Qwen3.8 Flash Next NVFP4):
    ml.cli qwen38-flash-next setup        # pin/configure and check both nodes
    ml.cli qwen38-flash-next download     # patch images + mirror model weights
    ml.cli qwen38-flash-next start        # worker-first TP=2 + safety proxy

\b
  Two-node Ray/vLLM serving (GLM-5.3 Flash NVFP4, experimental GB10):
    ml.cli glm53-flash setup              # configure and check both nodes
    ml.cli glm53-flash pull               # pinned arm64 base + Ray runtime layer
    ml.cli glm53-flash download           # download + mirror 181 GiB weights
    ml.cli glm53-flash start              # Ray TP=2 + safety proxy

\b
  One-node DSpark serving (DeepSeek V4 Flash 0731 EXL3):
    ml.cli dspark-one setup               # pin/configure the MiaAI TP=1 recipe
    ml.cli dspark-one build               # pull the digest-pinned image
    ml.cli dspark-one download            # download + coalesce the EXL3 weights
    ml.cli dspark-one start               # private :8888 + authenticated :8000

\b
  Box diagnostic:
    ml.cli info                           # GPU, CPU arch, library versions

\b
CO-RESIDENT SERVICES

\b
  Backends bind port 8000 by default. Normally stop the active backend
  before starting another. A memory-capped local vLLM can intentionally
  coexist with Docker vLLM by using different ports and passing
  `docker serve --allow-co-resident`. The optional router exposes co-resident
  services on one public port and serializes inference across them.

\b
DOCS

\b
  Registry of models:    registry/models.yaml      (local, Docker, cluster)
  NIM catalog:           registry/nim_catalog.yaml (Docker-served)
  README:                README.md
"""


@click.group(epilog=CLI_EPILOG)
def cli():
    """ml-compute — OpenAI-compatible local and clustered inference.

    \b
    Nine serving backends behind a single CLI:
      • vLLM      pip-installed Python process, safetensors →  `ml.cli serve <alias>`
      • llama.cpp local llama-server process, GGUF          →  `ml.cli serve llama <id>`
      • Docker    dedicated vLLM image from model registry  →  `ml.cli docker serve <alias>`
      • Vision    Transformers object detection             →  `ml.cli vision serve <alias>`
      • NIM       NVIDIA TensorRT-LLM container, Docker      →  `ml.cli nim serve <alias>`
      • DSpark    patched Docker vLLM, two GB10 nodes, TP=2  →  `ml.cli dspark <action>`
      • Qwen Next patched SGLang, two GB10 nodes, TP=2       →  `ml.cli qwen38-flash-next <action>`
      • GLM Flash Ray + dedicated vLLM, two GB10 nodes, TP=2 →  `ml.cli glm53-flash <action>`
      • DSpark One EXL3/SparkInfer, one GB10 node, TP=1      →  `ml.cli dspark-one <action>`

    Language-model backends expose an OpenAI-compatible /v1 API. The vision
    backend exposes /v1/models and /v1/text/detections. Run `ml.cli info` to
    see what's installed on this box and which backend is recommended. The
    optional router presents all resident services on one public port.
    """


# ---------------------------------------------------------------------------
# Models management
# ---------------------------------------------------------------------------

@cli.group()
def models():
    """Download and manage local model weights."""


@models.command("list")
def models_list():
    """Show registered aliases and what's downloaded locally."""
    from ml.models import is_downloaded, list_local, format_size

    registry = get_models()
    local = {m["hf_id"]: m for m in list_local()}

    click.echo("Registered models:")
    alias_width = max(14, *(len(alias) for alias in registry))
    for alias, cfg in registry.items():
        hf_id = cfg["hf_id"]
        mark = "✓" if is_downloaded(hf_id) else " "
        size = format_size(local[hf_id]["size_bytes"]) if hf_id in local else "—"
        backend = cfg.get("serve_backend", "vllm")
        nodes = int(cfg.get("nodes", 1))
        memory = "aggregate" if nodes > 1 else "VRAM"
        click.echo(
            f"  [{mark}] {alias:{alias_width}} {hf_id:42} {size:>10}  "
            f"({cfg['vram_gb']}GB {memory}, {backend})"
        )

    extra = [m for m in local if m not in {c["hf_id"] for c in registry.values()}]
    if extra:
        click.echo("\nDownloaded but not registered:")
        for hf_id in extra:
            click.echo(f"      {hf_id:42} {format_size(local[hf_id]['size_bytes']):>10}")


@models.command("pull")
@click.argument("name")
def models_pull(name: str):
    """Download a model from HuggingFace (alias or org/repo)."""
    from ml.models import pull, resolve_hf_id

    hf_id = resolve_hf_id(name)
    click.echo(f"Pulling {hf_id} → {HF_CACHE}")
    pull(name)
    click.echo(f"✓ {hf_id} ready")


@models.command("remove")
@click.argument("name")
@click.confirmation_option(prompt="Delete the model from disk?")
def models_remove(name: str):
    """Remove a downloaded model from local cache."""
    from ml.models import remove

    hf_id = remove(name)
    click.echo(f"✓ Removed {hf_id}")


@models.command("size")
def models_size():
    """Show disk usage by model."""
    from ml.models import format_size, list_local

    rows = list_local()
    if not rows:
        click.echo("(nothing downloaded)")
        return
    total = 0
    for r in rows:
        click.echo(f"  {r['hf_id']:42} {format_size(r['size_bytes']):>10}")
        total += r["size_bytes"]
    click.echo(f"  {'TOTAL':42} {format_size(total):>10}")


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

SERVE_EPILOG = """
\b
BACKENDS

\b
  vLLM (default) — pip-installed Python process, serves safetensors:
    ml.cli serve qwen3.5-9b-fp8          # bare alias → vLLM
    ml.cli serve vllm qwen3.5-9b-fp8     # same, explicit

\b
  llama.cpp — local `llama-server` process, serves GGUF:
    ml.cli serve llama qwen3.5-4b-gguf                       # registry alias
    ml.cli serve llama bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M  # HF GGUF repo
    ml.cli serve llama /path/to/model.gguf                   # local file

\b
  Both are local processes sharing one lifecycle — `ml.cli stop`,
  `ml.cli status`, and `ml.cli logs` work for whichever is running.
  (NIM is Docker-backed and lives under its own `ml.cli nim` group.)

\b
  Registries:  registry/models.yaml        (vLLM)
               registry/llama_models.yaml   (llama.cpp GGUF)
"""


class ServeGroup(click.Group):
    """`serve` dispatches to a backend subcommand.

    A bare `serve <alias>` with no explicit backend defaults to vLLM, so
    the historical `ml.cli serve <alias>` shape keeps working.
    """

    def resolve_command(self, ctx, args):
        if args and not args[0].startswith("-") and args[0] not in self.commands:
            args = ["vllm", *args]
        return super().resolve_command(ctx, args)


@cli.group("serve", cls=ServeGroup, epilog=SERVE_EPILOG)
def serve():
    """Start a model server (vLLM by default; `serve llama` for GGUF).

    \b
    A bare alias defaults to vLLM:  ml.cli serve qwen3.5-9b-fp8
    Pick a backend explicitly:      ml.cli serve llama <model_id>
    """


@serve.command("vllm")
@click.argument("name")
@click.option("--foreground", is_flag=True, help="Run vLLM in the foreground")
@click.option("--port", type=int, default=None, help=f"Override port (default {VLLM_PORT})")
def serve_vllm(name: str, foreground: bool, port: int | None):
    """Start vLLM serving the given model (safetensors)."""
    from ml.vllm_server import start

    info = start(name, foreground=foreground, port=port)
    click.echo(f"✓ Started vLLM (pid={info['pid']}) serving {info['hf_id']}")
    click.echo(f"  Port: {info['port']}")
    click.echo(f"  Logs: {info['log_path']}")
    click.echo(f"  Tail: ml-compute logs -f")
    click.echo(f"  Wait ~30s for model to load, then check: ml-compute status")


@serve.command("llama")
@click.argument("model_id")
@click.option("--foreground", is_flag=True, help="Run llama-server in the foreground")
@click.option("--port", type=int, default=None, help=f"Override port (default {VLLM_PORT})")
@click.option("--ngl", "n_gpu_layers", type=int, default=None,
              help="GPU layers to offload (999 = all, 0 = CPU only)")
@click.option("--ctx", "ctx_size", type=int, default=None, help="Context window in tokens")
def serve_llama(model_id: str, foreground: bool, port: int | None,
                n_gpu_layers: int | None, ctx_size: int | None):
    """Start llama.cpp serving a GGUF model.

    \b
    MODEL_ID can be:
      • a registry alias (see registry/llama_models.yaml), e.g. qwen3.5-4b-gguf
      • a HuggingFace GGUF repo, e.g. bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M
      • a local .gguf file path

    \b
    Requires the llama.cpp `llama-server` binary on PATH.
    """
    from ml.llama_server import start

    info = start(model_id, foreground=foreground, port=port,
                 n_gpu_layers=n_gpu_layers, ctx_size=ctx_size)
    click.echo(f"✓ Started llama.cpp (pid={info['pid']}) serving {info['hf_id']}")
    click.echo(f"  Model name: {info['served_name']}")
    click.echo(f"  Port: {info['port']}")
    click.echo(f"  Logs: {info['log_path']}")
    click.echo(f"  Tail: ml-compute logs -f")
    click.echo(f"  Wait ~30s for model to load, then check: ml-compute status")


def _serving_module():
    """Return the backend module (vllm/llama) for the server in state."""
    from ml.state import read_state

    backend = (read_state() or {}).get("backend", "vllm")
    if backend == "llama":
        from ml import llama_server as mod
    else:
        from ml import vllm_server as mod
    return mod


@cli.command("stop")
def stop_cmd():
    """Stop the running server (vLLM or llama.cpp)."""
    # The pid-based stop is backend-agnostic; vllm_server.stop handles either.
    from ml.vllm_server import stop

    if stop():
        click.echo("✓ Stopped")
    else:
        click.echo("Nothing running")


@cli.command("restart")
@click.argument("name")
@click.option("--port", type=int, default=None)
def restart(name: str, port: int | None):
    """Stop the current server and start a new one (model swap)."""
    from ml.vllm_server import start, stop

    if stop():
        click.echo("✓ Stopped previous server")
    info = start(name, port=port)
    click.echo(f"✓ Started vLLM (pid={info['pid']}) serving {info['hf_id']}")


@cli.command("status")
def status_cmd():
    """Show server status, GPU info, and the API key."""
    s = _serving_module().status()
    if not s["running"]:
        click.echo("State: not running")
        click.echo(f"Default port when started: {VLLM_PORT}")
    else:
        readiness = "ready ✓" if s["ready"] else "loading…"
        click.echo(f"State: {readiness}")
        click.echo(f"  Backend:    {s.get('backend', 'vllm')}")
        click.echo(f"  Model:      {s['model_alias']}  ({s['hf_id']})")
        click.echo(f"  PID:        {s['pid']}")
        click.echo(f"  URL:        http://{VLLM_HOST}:{s['port']}/v1")
        click.echo(f"  Started:    {s['started_at']}")
        click.echo(f"  Logs:       {s['log_path']}")

    api_key = get_api_key()
    click.echo("")
    if api_key:
        click.echo(f"API key:    {api_key}")
    else:
        click.echo("API key:    (not set — run `ml-compute config gen-key`)")


@cli.command("logs")
@click.option("-f", "--follow", is_flag=True, help="Follow log output (like tail -f)")
@click.option("-n", "--lines", type=int, default=50, show_default=True)
def logs_cmd(follow: bool, lines: int):
    """Tail the running server's log (vLLM or llama.cpp)."""
    _serving_module().tail_logs(lines=lines, follow=follow)


# ---------------------------------------------------------------------------
# Docker vLLM — dedicated per-model images from registry/models.yaml
# ---------------------------------------------------------------------------

DOCKER_EPILOG = """
\b
QUICKSTART

\b
  Start a Docker-backed registry model and watch it load:
    ml.cli docker serve unlimited-ocr
    ml.cli docker logs -f

\b
  Check API readiness, then stop and remove the container:
    ml.cli docker status
    ml.cli docker stop

\b
REGISTRY

\b
  Entries live in registry/models.yaml and must set:
    serve_backend: docker
    docker_image: org/image:tag

  Standard vLLM fields and extra_args are passed to the image's
  `vllm serve` entrypoint. The Hugging Face cache is mounted from HF_HOME.
"""


@cli.group("docker", epilog=DOCKER_EPILOG)
def docker_group():
    """Manage dedicated vLLM Docker images from the model registry."""


@docker_group.command("serve")
@click.argument("name")
@click.option("--port", type=int, default=None,
              help=f"Override port (default {VLLM_PORT})")
@click.option("--foreground", is_flag=True,
              help="Run attached with logs in the terminal")
@click.option("--gpus", type=click.IntRange(min=1), default=None,
              help="Number of GPUs to bind (default: all)")
@click.option(
    "--allow-co-resident",
    is_flag=True,
    help="Keep a memory-capped local server running on another port",
)
def docker_serve(
    name: str,
    port: int | None,
    foreground: bool,
    gpus: int | None,
    allow_co_resident: bool,
):
    """Start a Docker-backed vLLM model by registry alias.

    \b
    NAME must reference an entry in registry/models.yaml with
    `serve_backend: docker` and a `docker_image`.
    """
    from ml.docker_server import start

    try:
        info = start(
            name,
            port=port,
            foreground=foreground,
            gpu_count=gpus,
            allow_co_resident=allow_co_resident,
        )
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"✓ Started Docker vLLM container ({info['container_name']})")
    click.echo(f"  Image:     {info['image']}")
    click.echo(f"  Model:     {info['served_model_name']}")
    click.echo(f"  Port:      {info['port']}")
    click.echo(f"  Container: {info['container_id'][:12]}")
    click.echo("  Logs:      python3 -m ml.cli docker logs -f")
    click.echo("  Status:    python3 -m ml.cli docker status")


@docker_group.command("status")
def docker_status():
    """Show Docker container state and OpenAI API readiness."""
    from ml.docker_server import status

    s = status()
    if not s["running"]:
        if s.get("exited"):
            click.echo(
                f"State: EXITED ✗  (status={s.get('exit_status')}, "
                f"exit_code={s.get('exit_code')})"
            )
            if s.get("exit_error"):
                click.echo(f"  Error: {s['exit_error']}")
            click.echo(f"  Alias: {s.get('alias')}")
            click.echo(f"  Image: {s.get('image')}")
            click.echo(f"  Hint:  {s.get('log_hint')}")
        else:
            click.echo("State: not running")
        return

    readiness = "ready ✓" if s["ready"] else "loading…"
    container_id = s.get("container_id") or "unknown"
    click.echo(f"State: {readiness}")
    click.echo(f"  Container: {s['container_name']} ({container_id[:12]})")
    click.echo(f"  Alias:     {s.get('alias')}")
    click.echo(f"  Image:     {s.get('image')}")
    click.echo(f"  Model:     {s.get('served_model_name')}")
    click.echo(f"  URL:       http://localhost:{s['port']}/v1")
    click.echo(f"  Started:   {s.get('started_at')}")
    click.echo("  Logs:      python3 -m ml.cli docker logs -f")


@docker_group.command("stop")
def docker_stop():
    """Stop and remove the managed Docker vLLM container."""
    from ml.docker_server import stop

    if stop():
        click.echo("✓ Stopped and removed Docker vLLM container")
    else:
        click.echo("Nothing running")


@docker_group.command("logs")
@click.option("-f", "--follow", is_flag=True, help="Follow log output")
@click.option("-n", "--lines", type=int, default=50, show_default=True)
def docker_logs(follow: bool, lines: int):
    """Show logs from the running or exited Docker vLLM container."""
    from ml.docker_server import tail_logs

    tail_logs(lines=lines, follow=follow)


# ---------------------------------------------------------------------------
# Transformers vision — lightweight object/text detection
# ---------------------------------------------------------------------------

VISION_EPILOG = """
\b
QUICKSTART

\b
  Start PP-OCRv6 text detection beside the two vLLM services:
    ml.cli vision serve pp-ocrv6-medium-det --port 8103
    ml.cli vision logs -f

\b
  Send raw PNG/JPEG bytes (this returns regions, not recognized text):
    curl --data-binary @page.png -H 'Content-Type: image/png' \\
      http://localhost:8103/v1/text/detections
"""


@cli.group("vision", epilog=VISION_EPILOG)
def vision_group():
    """Manage lightweight Transformers vision models."""


@vision_group.command("serve")
@click.argument("name")
@click.option("--port", type=int, default=None, help="Override registry port")
@click.option("--foreground", is_flag=True, help="Run attached in the terminal")
def vision_serve(name: str, port: int | None, foreground: bool):
    """Start a registry-backed Transformers vision model."""
    from ml.vision_server import start

    try:
        info = start(name, port=port, foreground=foreground)
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"✓ Started Transformers vision server (pid={info['pid']})")
    click.echo(f"  Model:    {info['served_model_name']}")
    click.echo(f"  Port:     {info['port']}")
    click.echo(f"  Endpoint: http://localhost:{info['port']}/v1/text/detections")
    click.echo("  Logs:     python3 -m ml.cli vision logs -f")
    click.echo("  Status:   python3 -m ml.cli vision status")


@vision_group.command("status")
def vision_status():
    """Show Transformers vision server state and readiness."""
    from ml.vision_server import status

    state = status()
    if not state["running"]:
        click.echo("State: not running")
        return
    readiness = "ready ✓" if state["ready"] else "loading…"
    click.echo(f"State: {readiness}")
    click.echo(f"  Model:    {state['served_model_name']}")
    click.echo(f"  PID:      {state['pid']}")
    click.echo(f"  URL:      http://localhost:{state['port']}")
    click.echo(f"  Started:  {state['started_at']}")
    click.echo(f"  Logs:     {state['log_path']}")


@vision_group.command("stop")
def vision_stop():
    """Stop the Transformers vision server."""
    from ml.vision_server import stop

    if stop():
        click.echo("✓ Stopped Transformers vision server")
    else:
        click.echo("Nothing running")


@vision_group.command("logs")
@click.option("-f", "--follow", is_flag=True, help="Follow log output")
@click.option("-n", "--lines", type=int, default=50, show_default=True)
def vision_logs(follow: bool, lines: int):
    """Show Transformers vision server logs."""
    from ml.vision_server import tail_logs

    tail_logs(lines=lines, follow=follow)


# ---------------------------------------------------------------------------
# Single-port model router
# ---------------------------------------------------------------------------

ROUTER_EPILOG = """
\b
ROUTING

\b
  JSON OpenAI requests are routed by their `model` field. Raw-image PP-OCR
  requests are routed by /v1/text/detections or an explicit X-Model header.
  A shared concurrency limit serializes inference across all backends.
  Backend URLs and accepted model IDs live in registry/router.yaml.
"""


@cli.group("router", epilog=ROUTER_EPILOG)
def router_group():
    """Expose resident model services through one public port."""


@router_group.command("serve")
@click.option("--port", type=int, default=None, help="Override registry port")
@click.option("--foreground", is_flag=True, help="Run attached in the terminal")
def router_serve(port: int | None, foreground: bool):
    """Start the streaming model router."""
    from ml.router_server import start

    try:
        info = start(port=port, foreground=foreground)
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"✓ Started model router (pid={info['pid']})")
    click.echo(f"  URL:    http://localhost:{info['port']}/v1")
    click.echo("  Routes: registry/router.yaml")
    click.echo("  Logs:   python3 -m ml.cli router logs -f")
    click.echo("  Status: python3 -m ml.cli router status")


@router_group.command("status")
def router_status():
    """Show router and backend readiness."""
    from ml.router_server import status

    state = status()
    if not state["running"]:
        click.echo("State: not running")
        return
    readiness = "ready ✓" if state["ready"] else "degraded…"
    click.echo(f"State: {readiness}")
    click.echo(f"  PID:     {state['pid']}")
    click.echo(f"  URL:     http://localhost:{state['port']}")
    for name, backend in state.get("backends", {}).items():
        mark = "ready ✓" if backend.get("ready") else "unavailable ✗"
        click.echo(f"  {name}: {mark}")
    click.echo(f"  Logs:    {state['log_path']}")


@router_group.command("stop")
def router_stop():
    """Stop the model router without stopping its backends."""
    from ml.router_server import stop

    if stop():
        click.echo("✓ Stopped model router")
    else:
        click.echo("Nothing running")


@router_group.command("logs")
@click.option("-f", "--follow", is_flag=True, help="Follow log output")
@click.option("-n", "--lines", type=int, default=50, show_default=True)
def router_logs(follow: bool, lines: int):
    """Show model-router logs."""
    from ml.router_server import tail_logs

    tail_logs(lines=lines, follow=follow)


# ---------------------------------------------------------------------------
# DSpark authenticated allow-list proxy
# ---------------------------------------------------------------------------


@cli.group("dspark-proxy")
def dspark_proxy_group():
    """Operate the authenticated safety proxy in front of DSpark."""


@dspark_proxy_group.command("serve")
@click.option("--port", type=int, default=None, help="Override registry port")
@click.option("--foreground", is_flag=True, help="Run attached in the terminal")
def dspark_proxy_serve(port: int | None, foreground: bool):
    """Start the DSpark allow-list proxy."""
    from ml.dspark_proxy_server import start

    try:
        info = start(port=port, foreground=foreground)
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"✓ DSpark safety proxy running (pid={info['pid']})")
    click.echo(f"  Public API: http://localhost:{info['port']}/v1")
    click.echo("  Raw vLLM:   http://127.0.0.1:8888 (loopback only)")


@dspark_proxy_group.command("status")
def dspark_proxy_status():
    """Show proxy and upstream readiness."""
    from ml.dspark_proxy_server import status

    state = status()
    if not state["running"]:
        click.echo("State: not running")
        return
    readiness = "ready ✓" if state["ready"] else "degraded…"
    click.echo(f"State: {readiness}")
    click.echo(f"  PID:  {state['pid']}")
    click.echo(f"  URL:  http://localhost:{state['port']}")
    click.echo(f"  Logs: {state['log_path']}")


@dspark_proxy_group.command("stop")
def dspark_proxy_stop():
    """Stop the proxy without stopping the two-node model."""
    from ml.dspark_proxy_server import stop

    click.echo("✓ Stopped DSpark safety proxy" if stop() else "Nothing running")


@dspark_proxy_group.command("smoke")
def dspark_proxy_smoke():
    """Verify proxy auth, upstream access, and denied unsafe routes."""
    from ml.dspark_proxy_server import smoke

    try:
        result = smoke()
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("✓ DSpark proxy safety smoke passed")
    click.echo(f"  authorized /v1/models: {result['authorized_models']}")
    click.echo(f"  no-key /v1/models:     {result['unauthenticated_models']}")
    click.echo(f"  keyed /invocations:    {result['denied_invocations']}")


@dspark_proxy_group.command("logs")
@click.option("-f", "--follow", is_flag=True, help="Follow log output")
@click.option("-n", "--lines", type=int, default=50, show_default=True)
def dspark_proxy_logs(follow: bool, lines: int):
    """Show DSpark proxy logs."""
    from ml.dspark_proxy_server import tail_logs

    tail_logs(lines=lines, follow=follow)


# ---------------------------------------------------------------------------
# NIM — NVIDIA Inference Microservices façade
# ---------------------------------------------------------------------------


NIM_EPILOG = """
\b
QUICKSTART

\b
  1. Get a free NGC API key:
       https://build.nvidia.com → Login → API key
       echo 'NGC_API_KEY=nvapi-...' >> .env.local

\b
  2. List the registered NIM containers:
       ml.cli nim models

\b
  3. Run one (downloads the image on first use, ~20-40 GB):
       ml.cli nim serve qwen3.5-27b-nim

\b
  4. Check it's ready and tail logs:
       ml.cli nim status
       ml.cli nim logs -f

\b
  5. Hit the API exactly like vLLM:
       curl http://localhost:8000/v1/chat/completions \\
         -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \\
         -d '{"model":"Qwen/Qwen3.5-27B","messages":[...]}'

\b
  6. When done:
       ml.cli nim stop

\b
WHY NIM (vs vLLM)

\b
  Pros: 20-30% faster decode on Blackwell, NVFP4 support, NVIDIA-tuned
        TensorRT-LLM kernels, production-grade observability.
  Cons: Requires Docker, NGC account, per-model container images,
        LoRA-at-runtime is harder than vLLM.

\b
  On GB10 / DGX Spark: NIM is the recommended path for production.
  On RTX RunPod boxes / x86: vLLM is usually easier.

\b
ONE CONTAINER AT A TIME

\b
  The CLI manages a single container named `ml-compute-nim`. Stop it
  before starting another model: `ml.cli nim stop`.

\b
DOCS

\b
  Catalog file:  registry/nim_catalog.yaml
  Add your own:  paste a new entry under `nim_catalog:` and `ml.cli nim
                 serve <alias>` picks it up.
"""


@cli.group("nim", epilog=NIM_EPILOG)
def nim_group():
    """Manage NVIDIA NIM containers (Docker-backed serving).

    \b
    Same OpenAI-compatible /v1 API as vLLM — your client code does not
    change. The difference is what runs the model: NIM is a TensorRT-LLM
    container shipped by NVIDIA, optimized for Blackwell hardware.

    \b
    Requires Docker and an NGC API key (free at build.nvidia.com).
    See `ml.cli nim --help` for a quickstart.
    """


@nim_group.command("models")
def nim_models():
    """List NIM catalog entries from registry/nim_catalog.yaml.

    \b
    A ✓ next to an entry means the docker image is already pulled locally.
    """
    from ml.nim_server import list_models

    rows = list_models()
    if not rows:
        click.echo("(no entries in registry/nim_catalog.yaml)")
        return
    click.echo("Registered NIM containers:")
    for r in rows:
        mark = "✓" if r["pulled"] else " "
        vram = f"{r['vram_gb']}GB VRAM" if r["vram_gb"] else ""
        click.echo(f"  [{mark}] {r['alias']:24} {r['image']:60} {vram}")
        if r["description"]:
            click.echo(f"        {r['description']}")


@nim_group.command("serve")
@click.argument("name")
@click.option("--port", type=int, default=None,
              help=f"Override port (default {VLLM_PORT})")
@click.option("--foreground", is_flag=True,
              help="Run in the foreground (logs to terminal)")
@click.option("--gpus", type=int, default=None,
              help="Number of GPUs to bind (default: all)")
def nim_serve(name: str, port: int | None, foreground: bool, gpus: int | None):
    """Start a NIM container in the background.

    \b
    NAME can be either:
      • a catalog alias (see `ml.cli nim models`), e.g. qwen3.5-27b-nim
      • a full nvcr.io image URI, e.g. nvcr.io/nim/qwen/qwen3.5-9b:latest

    \b
    Requires:
      • Docker installed and running
      • NGC_API_KEY set in .env.local (free at build.nvidia.com)
      • vLLM stopped first if it's using the same port (`ml.cli stop`)

    \b
    Logs:   ml.cli nim logs -f
    Status: ml.cli nim status
    """
    from ml.nim_server import start

    info = start(name, port=port, foreground=foreground, gpu_count=gpus)
    click.echo(f"✓ Started NIM container ({info['container_name']})")
    click.echo(f"  Image:      {info['image']}")
    click.echo(f"  Model name: {info['model_name'] or '(see container default)'}")
    click.echo(f"  Port:       {info['port']}")
    click.echo(f"  Container:  {info['container_id'][:12]}")
    click.echo(f"  Logs:       ml-compute nim logs -f")
    click.echo(f"  Wait ~30-90s for the first load, then: ml-compute nim status")


@nim_group.command("status")
def nim_status():
    """Show NIM container status and readiness.

    \b
    'ready ✓' means the /v1/models endpoint responded successfully.
    'loading…' means the container is still warming up (first load
    can take 30-90s while it pulls the model and compiles graphs).
    """
    from ml.nim_server import status

    s = status()
    if not s["running"]:
        if s.get("exited"):
            click.echo(f"State: EXITED ✗  (status={s.get('exit_status')}, "
                       f"exit_code={s.get('exit_code')})")
            if s.get("exit_error"):
                click.echo(f"  Error: {s['exit_error']}")
            click.echo(f"  Alias: {s.get('alias')}")
            click.echo(f"  Image: {s.get('image')}")
            click.echo(f"  Hint:  {s.get('log_hint')}")
            click.echo(f"         (or: ml-compute nim logs)")
        else:
            click.echo("State: not running")
        return
    readiness = "ready ✓" if s["ready"] else "loading…"
    click.echo(f"State: {readiness}")
    click.echo(f"  Container:  {s['container_name']} ({s['container_id'][:12]})")
    click.echo(f"  Alias:      {s['alias']}")
    click.echo(f"  Image:      {s['image']}")
    click.echo(f"  Model name: {s['model_name']}")
    click.echo(f"  URL:        http://{VLLM_HOST}:{s['port']}/v1")
    click.echo(f"  Started:    {s['started_at']}")
    click.echo(f"  Logs:       {s['log_path']}")
    api_key = get_api_key()
    if api_key:
        click.echo(f"  API key:    {api_key}")


@nim_group.command("stop")
def nim_stop():
    """Stop and remove the running NIM container.

    \b
    Always safe to call — if nothing is running, prints
    'Nothing running' and exits 0.
    """
    from ml.nim_server import stop

    if stop():
        click.echo("✓ Stopped")
    else:
        click.echo("Nothing running")


@nim_group.command("logs")
@click.option("-f", "--follow", is_flag=True, help="Follow log output (like tail -f)")
@click.option("-n", "--lines", type=int, default=50, show_default=True)
def nim_logs(follow: bool, lines: int):
    """Tail the running NIM container's stdout/stderr via `docker logs`.

    \b
    Useful for watching model load progress, first-token latency,
    and any container-side errors (auth, OOM, missing weights).
    """
    from ml.nim_server import tail_logs

    tail_logs(lines=lines, follow=follow)


# ---------------------------------------------------------------------------
# DSpark — patched two-node vLLM recipe for DeepSeek V4 Flash
# ---------------------------------------------------------------------------

DSPARK_ACTIONS = [
    "network", "bootstrap", "configure", "check", "setup", "build",
    "download", "start", "status", "gpu-check", "memory", "smoke", "logs",
    "stop", "legacy-stop", "update", "all", "path", "help",
]


@cli.command("dspark")
@click.argument(
    "action",
    required=False,
    default="help",
    type=click.Choice(DSPARK_ACTIONS, case_sensitive=False),
)
def dspark_cmd(action: str):
    """Manage DeepSeek V4 Flash across two linked GB10 systems.

    \b
    This is a dedicated cluster backend, not NVIDIA NIM and not the local
    `serve vllm` path. It delegates to scripts/DS4-Flash-DSpark.sh, which
    installs and operates the MiaAI/Anemll Docker/vLLM TP=2 recipe.

    \b
    Typical sequence:
      ml.cli dspark network
      ml.cli dspark setup
      ml.cli dspark build        # immutable Anemll image on both ranks
      ml.cli dspark download
      ml.cli dspark gpu-check
      ml.cli dspark start
      ml.cli dspark smoke

    Run `ml.cli dspark help` for configuration variables and full details.
    """
    script = Path(__file__).resolve().parent.parent / "scripts" / "DS4-Flash-DSpark.sh"
    if not script.is_file():
        raise click.ClickException(f"DSpark recipe not found: {script}")
    result = subprocess.run([str(script), action], check=False)
    raise click.exceptions.Exit(result.returncode)


# ---------------------------------------------------------------------------
# Qwen3.8 Flash Next — patched two-node SGLang recipe for GB10/SM121
# ---------------------------------------------------------------------------

QWEN38_FLASH_NEXT_ACTIONS = [
    "bootstrap", "configure", "check", "setup", "download", "start",
    "status", "memory", "smoke", "logs", "logs-worker", "stop", "update",
    "all", "path", "help",
]


@cli.command("qwen38-flash-next")
@click.argument(
    "action",
    required=False,
    default="help",
    type=click.Choice(QWEN38_FLASH_NEXT_ACTIONS, case_sensitive=False),
)
def qwen38_flash_next_cmd(action: str):
    """Manage Qwen3.8 Flash Next across two linked DGX Sparks.

    \b
    This delegates to MiaAI-Lab's SGLang TP=2 recipe while ml-compute pins
    its revision/profile and keeps raw SGLang on loopback :8888 behind the
    authenticated safety proxy on :8000. The SM121 QSA kernel patch and
    in-checkpoint NEXTN draft are built and configured by the recipe.

    \b
    Typical sequence:
      ml.cli qwen38-flash-next setup
      ml.cli qwen38-flash-next download
      ml.cli qwen38-flash-next start
      ml.cli qwen38-flash-next smoke

    Run `ml.cli qwen38-flash-next help` for memory and network constraints.
    """
    script = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "Qwen38-Flash-Next-Dual-DSpark.sh"
    )
    if not script.is_file():
        raise click.ClickException(f"Qwen Flash Next recipe not found: {script}")
    result = subprocess.run([str(script), action], check=False)
    raise click.exceptions.Exit(result.returncode)


# ---------------------------------------------------------------------------
# GLM-5.3 Flash — dedicated vLLM image over a two-node Ray cluster
# ---------------------------------------------------------------------------

GLM53_FLASH_ACTIONS = [
    "configure", "check", "setup", "pull", "download", "gpu-check", "start",
    "status", "diagnose", "memory", "smoke", "logs", "logs-worker", "stop",
    "all", "path", "help",
]


@cli.command("glm53-flash")
@click.argument(
    "action",
    required=False,
    default="help",
    type=click.Choice(GLM53_FLASH_ACTIONS, case_sensitive=False),
)
def glm53_flash_cmd(action: str):
    """Manage GLM-5.3 Flash NVFP4 across two linked DGX Sparks.

    \b
    This is an experimental GB10/SM121 deployment using the model's dedicated
    arm64 vLLM image and the official two-node Ray pattern. ml-compute pins the
    model/image, mirrors weights to both nodes, and keeps raw vLLM on loopback
    :8888 behind the authenticated safety proxy on :8000.

    \b
    Typical sequence:
      ml.cli glm53-flash setup
      ml.cli glm53-flash pull
      ml.cli glm53-flash download
      ml.cli glm53-flash gpu-check
      ml.cli glm53-flash start
      ml.cli glm53-flash smoke

    Run `ml.cli glm53-flash help` for the conservative memory constraints.
    """
    script = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "GLM53-Flash-Dual-DSpark.sh"
    )
    if not script.is_file():
        raise click.ClickException(f"GLM-5.3 Flash recipe not found: {script}")
    result = subprocess.run([str(script), action], check=False)
    raise click.exceptions.Exit(result.returncode)


# ---------------------------------------------------------------------------
# DSpark One — EXL3/ExLlamaV3 TP1 recipe for one dedicated Spark
# ---------------------------------------------------------------------------

DSPARK_ONE_ACTIONS = [
    "bootstrap", "configure", "check", "setup", "build", "download",
    "gpu-check", "start", "status", "memory", "smoke", "logs", "stop",
    "update", "all", "path", "help",
]


@cli.command("dspark-one")
@click.argument(
    "action",
    required=False,
    default="help",
    type=click.Choice(DSPARK_ONE_ACTIONS, case_sensitive=False),
)
def dspark_one_cmd(action: str):
    """Manage DeepSeek V4 Flash on one dedicated DGX Spark.

    \b
    This is MiaAI-Lab's TP=1, 3.0-bpw EXL3/SparkInfer recipe. It is separate
    from `dspark`, which remains the two-node TP=2 Anemll deployment. The raw
    API binds to loopback :8888; the authenticated safety proxy owns :8000.

    \b
    Typical sequence:
      ml.cli dspark-one setup
      ml.cli dspark-one build
      ml.cli dspark-one download
      ml.cli dspark-one gpu-check
      ml.cli dspark-one start
      ml.cli dspark-one smoke

    Run `ml.cli dspark-one help` for the memory and context constraints.
    """
    script = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "DS4-Flash-One-DSpark.sh"
    )
    if not script.is_file():
        raise click.ClickException(f"One-Spark recipe not found: {script}")
    result = subprocess.run([str(script), action], check=False)
    raise click.exceptions.Exit(result.returncode)


# ---------------------------------------------------------------------------
# Box info
# ---------------------------------------------------------------------------

@cli.command("info")
def info_cmd():
    """Show hardware, library, and recommended-stack info for this box."""
    import platform
    import shutil

    click.echo("=== ml-compute info ===")
    click.echo(f"  Python:       {platform.python_version()}")
    click.echo(f"  CPU arch:     {platform.machine()}")
    click.echo(f"  Platform:     {platform.system()} {platform.release()}")

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            d = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            cap = torch.cuda.get_device_capability(0)
            sm = f"sm_{cap[0]}{cap[1]}"
            click.echo(f"  GPU:          {d}")
            click.echo(f"  VRAM:         {vram:.1f} GB ({sm})")
            click.echo(f"  Torch:        {torch.__version__}  CUDA: {torch.version.cuda}")

            is_gb10 = platform.machine() in ("aarch64", "arm64") and cap[0] >= 12
            if is_gb10:
                click.echo("")
                click.echo("  → GB10 / DGX Spark detected (Grace+Blackwell unified memory)")
                click.echo("    Training: make -f Makefile.gb10 train DATASET=...")
                click.echo("    Serving:  NVIDIA NIM container preferred over vLLM")
            elif cap[0] >= 12:
                click.echo("  → Blackwell detected. Set env vars:")
                click.echo(f"      export TORCH_CUDA_ARCH_LIST=\"{cap[0]}.{cap[1]}+PTX\"")
                click.echo(f"      export VLLM_FLASHINFER_FORCE_TARGET=sm_{cap[0]}{cap[1]}")
        else:
            click.echo("  GPU:          (no CUDA device)")
    except Exception as e:
        click.echo(f"  GPU:          (torch not loaded: {e})")

    # Lib availability — what training/serving paths actually work here
    click.echo("\n=== Libraries ===")
    for name in ["vllm", "transformers", "peft", "trl", "datasets",
                 "accelerate", "unsloth", "bitsandbytes", "flash_attn",
                 "xformers", "triton"]:
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "?")
            click.echo(f"  ✓ {name:<14} {ver}")
        except ImportError:
            click.echo(f"  ✗ {name:<14} not installed")

    # Tooling
    click.echo("\n=== Tooling ===")
    for cmd in ["docker", "nvidia-smi", "git"]:
        path = shutil.which(cmd)
        if path:
            click.echo(f"  ✓ {cmd:<14} {path}")
        else:
            click.echo(f"  ✗ {cmd:<14} not found")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

@cli.group()
def config():
    """Inspect or update local config (.env.local)."""


@config.command("show")
def config_show():
    """Print effective configuration."""
    click.echo(f"VLLM_HOST:    {VLLM_HOST}")
    click.echo(f"VLLM_PORT:    {VLLM_PORT}")
    click.echo(f"HF_CACHE:     {HF_CACHE}")
    api_key = get_api_key()
    click.echo(f"API_KEY:      {api_key or '(unset)'}")


@config.command("gen-key")
def config_gen_key():
    """Generate a new API key and save it to .env.local."""
    key = generate_api_key()
    write_env_var("API_KEY", key)
    click.echo(f"✓ API_KEY written to .env.local")
    click.echo(f"  {key}")


# ---------------------------------------------------------------------------
# Optional: training (kept for future fine-tuning workflow)
# ---------------------------------------------------------------------------

@cli.command("train", hidden=True)
@click.option("--adapter-name", required=True)
@click.option("--model", required=True, help="Registry alias")
@click.option("--dataset", required=True)
@click.option("--epochs", default=3, type=int)
@click.option("--lr", default=2e-4, type=float)
@click.option("--rank", default=16, type=int)
def train_cmd(adapter_name, model, dataset, epochs, lr, rank):
    """[Experimental] Fine-tune a registered model with Unsloth."""
    from ml.train import train as run_train  # imported lazily; needs optional deps
    run_train(model, dataset, adapter_name, epochs, lr, rank)


if __name__ == "__main__":
    cli()
