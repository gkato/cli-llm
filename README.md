# ml-compute

Local LLM inference and fine-tuning on your own hardware. Built for the **NVIDIA DGX Spark
(GB10)** — Grace CPU + Blackwell GPU with 128 GB unified memory — and works on any local
box with an NVIDIA GPU.

One CLI, nine serving backends. Language models use the OpenAI-compatible `/v1`
API; lightweight vision detectors use a small task-specific `/v1` API:

| Backend | Runs | Weights | Command |
|---------|------|---------|---------|
| **vLLM** | pip-installed Python process | safetensors (bf16 / FP8 / NVFP4 / AWQ) | `serve <alias>` |
| **llama.cpp** | local `llama-server` binary | GGUF | `serve llama <id>` |
| **Docker vLLM** | dedicated per-model container | safetensors + custom vLLM build | `docker serve <alias>` |
| **Transformers vision** | local Python process | safetensors object detectors | `vision serve <alias>` |
| **NIM** | Docker container (TensorRT-LLM) | NVIDIA-packaged | `nim serve <alias>` |
| **DSpark cluster** | MiaAI/Anemll vLLM on two GB10 nodes | DeepSeek V4 Flash 0731 | `dspark <action>` |
| **Qwen Flash Next** | MiaAI SGLang TP2 on two GB10 nodes | Qwen3.8 Flash Next NVFP4 | `qwen38-flash-next <action>` |
| **GLM Flash** | Dedicated vLLM + MP TP2 on two GB10 nodes | GLM-5.3 Flash EXL3 + DFlash2 | `glm53-flash <action>` |
| **DSpark One** | MiaAI SparkInfer/EXL3 on one dedicated GB10 | DeepSeek V4 Flash 0731 | `dspark-one <action>` |

An optional streaming router exposes several resident services through one
public port, selects the backend from the request's `model` field, and can
serialize inference across them.

Plus a LoRA fine-tuning path that runs on the same box (aarch64-native, no Unsloth).

## What It Is

- **A local inference server.** Models run on hardware you own, on your LAN. No API bills,
  no data leaving the network.
- **OpenAI protocol.** Point any OpenAI client at it — chat, streaming, tool calls.
- **Bearer-token auth** so you can expose it on the LAN without leaving it wide open.
- **A YAML registry** of tuned per-model launch configs (context length, quantization,
  KV-cache dtype, LoRA adapters) — hard-won settings, not defaults.
- **A fine-tuning rig.** Token-filter → LoRA train → serve the adapter, all on the Spark.

## What It Is Not

- Not a chat UI — point [llm-playground](../llm-playgroung) or any OpenAI client at it.
- Not a multi-tenant scheduler. Multiple explicitly memory-capped services can coexist;
  the optional router provides a small global inference queue for this profile.
- Not a cloud deployment tool. It runs on the box in front of you.

---

## Hardware

**Primary target: DGX Spark / GB10.** Grace (aarch64) + Blackwell (sm_121), 128 GB unified
LPDDR5X shared between CPU and GPU. That unified pool is the whole point: a 31B model in
bf16, or 27B FP8 at 128k context, fits without juggling quantization.

Also runs on: any x86 + NVIDIA GPU workstation (Ampere / Ada / Blackwell). The registry
comments flag which quantization to pick per architecture.

Check what you're on:

```bash
python3 -m ml.cli info
```

That prints GPU / SM version / CPU arch, which of the training and serving libraries
actually imported, and whether Docker + `nvidia-smi` are present.

---

## Install

```bash
git clone https://github.com/gkato/ml-compute.git
cd ml-compute
bash setup.sh
source venv/bin/activate
```

`setup.sh` is architecture-aware:

- **aarch64 (Spark/GB10)** — installs the HF stack (torch, transformers, peft, trl,
  accelerate) plus the CLI deps, then attempts vLLM. Unsloth, flash-attn, and xformers
  are skipped: they're x86-only.
- **x86_64** — installs `requirements.txt` (vLLM + CLI deps).

It also creates `data/{logs,hf_cache}`, copies `.env.example` → `.env.local`, and generates
an `API_KEY`. Add a HuggingFace token if you want gated models (Llama, Gemma):

```bash
echo "HF_TOKEN=hf_xxx" >> .env.local
```

Optional convenience alias — every example below uses the module form:

```bash
alias mlc='python3 -m ml.cli'
```

---

## Quick Start

### vLLM (default backend)

```bash
python3 -m ml.cli models pull qwen3.6-27b-fp8    # download weights
python3 -m ml.cli serve qwen3.6-27b-fp8          # start in background
python3 -m ml.cli status                         # 'loading…' → 'ready ✓'
python3 -m ml.cli logs -f                        # watch it load
```

### llama.cpp (GGUF)

No pull step — llama.cpp fetches and caches the GGUF itself:

```bash
python3 -m ml.cli serve llama qwen2.5-coder-32b                      # registry alias
python3 -m ml.cli serve llama bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M   # any HF GGUF repo
python3 -m ml.cli serve llama /path/to/model.gguf                    # local file
```

Requires the `llama-server` binary on `PATH` (see [GB10 notes](#llamacpp-must-be-built-with-openssl)).

### Docker vLLM (dedicated model images)

Models that require a custom vLLM build set `serve_backend: docker` and
`docker_image` in `registry/models.yaml`. The CLI mounts the shared Hugging Face
cache, applies the normal vLLM registry settings, and manages one fixed container:

```bash
python3 -m ml.cli docker serve unlimited-ocr
python3 -m ml.cli docker status
python3 -m ml.cli docker logs -f
python3 -m ml.cli docker stop
```

This path needs Docker with NVIDIA Container Toolkit GPU support. It does not need
an NGC account; public images are pulled directly from their configured registry.

### Three resident OCR/LLM services on one DGX Spark

This profile keeps Qwen 3.8 27B NVFP4, Unlimited-OCR, and the PP-OCRv6 text
detector loaded at the same time. All public inference goes through a router
with one shared permit, so only one of the three backends runs a request at a
time.

| Service | Internal URL | Memory control |
|---------|--------------|----------------|
| Qwen 3.8 27B NVFP4 | `127.0.0.1:8101` | 30% reservation, 32k context, one sequence, FP8 KV |
| Unlimited-OCR | `127.0.0.1:8102` | 20% reservation, one sequence |
| PP-OCRv6 medium detector | `127.0.0.1:8103` | FP16 weights, one in-flight request |
| Model router | `0.0.0.0:8000` | Streaming proxy; one global inference slot |

The two vLLM reservations total about 64 GB of the Spark's 128 GB unified
pool. PP-OCRv6 is below 0.2 GB for weights. The remaining memory covers CUDA
workspaces, CPU allocations, page cache, and model-loading headroom. Start the
services sequentially and wait for readiness between the two large models:

```bash
# Download host-served weights. Unlimited-OCR downloads inside its container.
python3 -m ml.cli models pull qwen3.8-27b-nvfp4-coserve
python3 -m ml.cli models pull pp-ocrv6-medium-det

# 1. Qwen: loopback-only; wait for `ready ✓` before continuing.
VLLM_HOST=127.0.0.1 python3 -m ml.cli serve \
  qwen3.8-27b-nvfp4-coserve --port 8101
python3 -m ml.cli logs -f
python3 -m ml.cli status

# 2. Unlimited-OCR: explicit opt-in permits coexistence on another port.
VLLM_HOST=127.0.0.1 python3 -m ml.cli docker serve unlimited-ocr \
  --port 8102 --allow-co-resident
python3 -m ml.cli docker logs -f
python3 -m ml.cli docker status

# 3. Lightweight text-region detector, also loopback-only.
VLLM_HOST=127.0.0.1 python3 -m ml.cli vision serve \
  pp-ocrv6-medium-det --port 8103
python3 -m ml.cli vision logs -f
python3 -m ml.cli vision status

# 4. Public router. Its health is ready when all three backends are ready.
python3 -m ml.cli router serve --port 8000
python3 -m ml.cli router status
```

The public API is now always port 8000. OpenAI-compatible requests are routed
by `model`; aliases are translated to the backend's served model ID:

```bash
API_KEY=$(sed -n 's/^API_KEY=//p' .env.local)
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b-nvfp4-coserve","messages":[{"role":"user","content":"Say pong."}],"max_tokens":8}'
```

PP-OCRv6 accepts raw image bytes, so its unique path selects the detector. An
optional `X-Model: pp-ocrv6-medium-det` header can make that selection explicit:

```bash
API_KEY=$(sed -n 's/^API_KEY=//p' .env.local)
curl -fsS --data-binary @page.png \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: image/png" \
  -H "X-Model: pp-ocrv6-medium-det" \
  http://127.0.0.1:8000/v1/text/detections
```

The existing Unlimited-OCR PDF helper can use the same router URL:

```bash
UNLIMITED_BASE_URL=http://127.0.0.1:8000/v1 \
  scripts/pdf_to_gemma_curl.sh document.pdf
```

The router's `max_concurrency: 1` is a global queue: it holds the permit until
the upstream response has finished streaming. Calls made directly to internal
ports 8101–8103 bypass that queue, so keep those ports loopback-only and use
port 8000 for application traffic.

### NIM (Docker / TensorRT-LLM)

```bash
echo 'NGC_API_KEY=nvapi-...' >> .env.local   # free at build.nvidia.com
python3 -m ml.cli nim models                 # list the catalog
python3 -m ml.cli nim serve qwen3.5-27b-nim
python3 -m ml.cli nim status
```

### DSpark cluster (two linked GB10 systems)

NVIDIA publishes a generic DeepSeek V4 Flash NIM, but this project's `nim` backend starts
one local container and cannot orchestrate the two-node 0731/DSpark profile. The dual-Spark
path wraps [MiaAI-Lab's maintained recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark),
using its current hotfix set over the immutable Anemll `0.1.1` image digest and the pinned
official 0731 model revision. The committed profile uses a 512K request ceiling, NVFP4 MLA
KV, TP=2, MTP=5, low reasoning by default, and asymmetric memory utilization: 0.75 on
the NVIDIA/head rank and 0.73 on the Lenovo/Harness rank.

```bash
python3 -m ml.cli dspark network     # QSFP/RoCE state and setup guidance
python3 -m ml.cli dspark setup       # clone, configure, and check both nodes
python3 -m ml.cli dspark build       # pull/verify the pinned image on both ranks
python3 -m ml.cli dspark download    # download/verify and mirror weights
python3 -m ml.cli dspark gpu-check   # initialize CUDA in the image on both nodes
scripts/start-DS4-Flash-DSpark.sh    # raw vLLM :8888 + safety proxy :8000
python3 -m ml.cli dspark smoke
```

For the first deployment, the setup/build/download/start sequence can be run as:

```bash
scripts/start-DS4-Flash-DSpark.sh --first-run
```

Machine addresses, both connected RoCE interfaces, cache paths, context, and memory
budgets are in [`config/dspark-spark4e89-thinkstationpgx.env`](config/dspark-spark4e89-thinkstationpgx.env).
The lifecycle implementation is in [`scripts/DS4-Flash-DSpark.sh`](scripts/DS4-Flash-DSpark.sh).
After launch, `python3 -m ml.cli dspark memory` verifies that the worker has at least
24 GiB `MemAvailable` before the Harness starts. The generated upstream checkout and
its `.env.dspark` live under ignored `data/dspark/` by default.

The first Anemll launch at 0.70 gave the head 2.54 GiB of KV cache while 512K required
3.78 GiB. A subsequent 0.72/0.70 attempt profiled only 1.23 GiB on both ranks. The
profile now uses 0.75/0.73: three extra percentage points per rank supply roughly
3.6 GiB more allocation than that attempt, while the worker's 0.73 target leaves
approximately 24 GiB outside vLLM for the Harness and OS. A version-checked launcher
overlay applies the two values independently. It is
regenerated from the pristine upstream launcher during configure and preflight, and fails
closed if MiaAI changes the patched commands. Startup then reads vLLM's model-aware
`Maximum concurrency for 524,288 tokens` boot result and refuses to expose the proxy
unless at least one full request fits. It deliberately does not gate on
`cache_config_info`: current DeepSeek V4 hybrid-cache builds can publish a corrupted
`block_size=4` and under-report token capacity there. A single full 512K
request is the supported worst case; do not assume two full-window requests fit.
`MAX_NUM_SEQS=4` remains appropriate for shorter concurrent coding-agent turns. After
launch, `dspark memory` must report at least 24 GiB available on the Harness worker
(16 GiB Harness cap plus 8 GiB operating headroom).

The launcher runs `gpu-check` automatically before it starts either TP rank. If a
driver update left Docker's NVIDIA CDI description stale, refresh it on the node
named by the error with `sudo systemctl restart nvidia-cdi-refresh.service`, then
rerun `python3 -m ml.cli dspark gpu-check`. This isolates container/driver failures
from later NCCL and model-loading failures.

Raw vLLM binds only to `127.0.0.1:8888`. This matters because vLLM's native key does not
cover every inference-capable compatibility route. The network-facing service is a
deny-by-default proxy on `0.0.0.0:8000`; it authenticates `API_KEY` from `.env.local` and
allows only reviewed OpenAI/Anthropic client routes. `/invocations`,
`/generative_scoring`, `/tokenize`, `/detokenize`, metrics, and unknown future routes
are not forwarded. Operate it independently with:

```bash
python3 -m ml.cli dspark-proxy status
python3 -m ml.cli dspark-proxy smoke
python3 -m ml.cli dspark-proxy logs -f
```

If Tailscale Funnel previously targeted raw port 8888, move it immediately before
the cutover. The launcher refuses to start while Funnel still points at the raw port:

```bash
sudo tailscale funnel --https=443 off
sudo tailscale funnel --bg=true 8000
sudo tailscale funnel status
```

For minimal downtime, stage the runtime while the old model remains live, benchmark it,
then switch Funnel and cut over:

```bash
python3 -m ml.cli dspark setup
python3 -m ml.cli dspark build
python3 -m ml.cli dspark download
# Run the stage-c baseline here, then move Funnel from 8888 to 8000 as above.
scripts/start-DS4-Flash-DSpark.sh --cutover
```

Once DSpark is running on the default port, query it with:

```bash
API_KEY=$(sed -n 's/^API_KEY=//p' .env.local)
curl -fsS http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer ${API_KEY}" \
  | python3 -m json.tool
```

Before replacing the old server, collect a baseline; repeat with a different label after
the MiaAI cutover. The full suite records C1/C2/C4 throughput, a forced tool call,
32K/128K TTFT, and worker memory before/after:

```bash
VLLM_BASE_URL=http://127.0.0.1:8888 \
  BENCH_LABEL=stage-c-256k scripts/bench_dspark_ab.sh full
BENCH_LABEL=miaai-512k scripts/bench_dspark_ab.sh full
```

### Qwen3.8 Flash Next (two linked GB10 systems)

The Qwen path wraps
[MiaAI-Lab's dual-DGX-Spark recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks)
at pinned revision `f87d586e269df171089a879ee33a5356c0570e70`. It serves the
pinned `RadixArk/Qwen3.8-Flash-Next-NVFP4` checkpoint: a multimodal 176B/6B-active
MoE in ModelOpt NVFP4, using SGLang TP=2 and the model's in-checkpoint NEXTN
draft. The adapter keeps the upstream-generated SM121 QSA Triton patch and JIT
caches under ignored `data/dspark/`, while configuration and service exposure
remain owned by `ml-compute`.

Run these on the head/rank-0 Spark:

```bash
python3 -m ml.cli qwen38-flash-next setup
python3 -m ml.cli qwen38-flash-next download  # patch/build both images + mirror ~135 GB
python3 -m ml.cli qwen38-flash-next start
python3 -m ml.cli qwen38-flash-next smoke
```

Or perform the complete first deployment with:

```bash
scripts/start-Qwen38-Flash-Next-Dual-DSpark.sh --first-run
```

Later boots use the same script without arguments. Machine addresses, RoCE
interfaces, the digest-pinned SGLang base image, model revision, memory budget,
and context profile are in
[`config/dspark-qwen38-flash-next-nvfp4.env`](config/dspark-qwen38-flash-next-nvfp4.env).
The lifecycle adapter is
[`scripts/Qwen38-Flash-Next-Dual-DSpark.sh`](scripts/Qwen38-Flash-Next-Dual-DSpark.sh).

The committed profile uses MiaAI-Lab's full 1,048,576-token YaRN path with
NVFP4 KV cache, 1K-token QSA prefill chunks, PLE auto-offload, Mamba
full-memory ratio 0.3, and NEXTN `3/1/4`. The request ceiling remains 16;
the Mamba cache is pinned to 80 state slots (five per NEXTN request) so the
engine can honor that ceiling instead of silently reducing concurrency.
`MEM_FRACTION_STATIC=0.79` plus a post-start `MemAvailable` guard preserves at
least 20 GiB on the worker. Automatic prompt truncation and short-KV-pool
overrides remain disabled, so the launch fails closed unless one full 1M
request fits. The wrapper also rejects public raw binds, disabled PLE offload,
or a disabled kernel patch. As with the GLM lifecycle, the configured RoCE IPs
are authoritative: interface and RDMA HCA names are detected on both nodes
before MiaAI's `.env` is materialized, avoiding stale PCI-slot name assumptions.
The upstream process also has `API_KEY` removed explicitly: raw SGLang remains
unkeyed on loopback, while client authentication belongs exclusively to the
safety proxy on port 8000.

Do not add `--load-format dummy`: the upstream investigation found that the
temporary FP16 copy of the PLE table can exhaust GB10 unified memory and
hard-freeze both nodes. Cold start also requires at least 112 GiB
`MemAvailable` on each Spark. Stop DeepSeek and any other large CUDA workload
before switching this same two-node cluster to Qwen.

Raw unauthenticated SGLang binds only to `127.0.0.1:8888`. The shared
authenticated allow-list proxy remains the public API on port 8000, including
for image-bearing OpenAI chat requests. Operational commands are independent
from `dspark`:

```bash
python3 -m ml.cli qwen38-flash-next status
python3 -m ml.cli qwen38-flash-next memory
python3 -m ml.cli qwen38-flash-next logs
python3 -m ml.cli qwen38-flash-next logs-worker
python3 -m ml.cli qwen38-flash-next stop
```

### GLM-5.3 Flash EXL3 + DFlash2 (two linked GB10 systems)

The GLM path serves
[`brandonmusic/GLM-5.3-Flash-tr3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw),
a roughly 164 GiB EXL3/TR3 quantization of the multimodal 320B/18B-active MoE.
The lifecycle wraps [MiaAI-Lab's EXL3 dual-DGX-Spark recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
at pinned revision `1df71c1669489ae1f80f05a560732c598db8e615`, and pins the
measured target snapshot `5ab363a8dcf6405955fd5f99671e01a1c9fb124b`.

This replaces the previous NVFP4/Ray profile. It removes Ray and its object
stores, joins one vLLM multiprocessing rank per Spark directly at TP=2, uses
the fused EXL3 MoE path and CUDA graphs, and raises the reviewed request ceiling
from 256K to 1M. The measured default adds
[`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
with seven speculative tokens and a rank-0-only draft. The conservative launch
shape remains four sequences, 1K prefill chunks, 0.87 memory utilization, FP8
MLA KV, prefix caching, and skipped maximum-size multimodal dummy profiling.
Padded DFlash2 pages now share the MLA allocation, allowing the 1M profile
without the previous 0.8847 CUDA-graph workaround. The updated overlay also
fixes hybrid prefix-cache hits, keeps long peer prefills off active decode
steps, persists Triton/TileLang caches, warms common shapes after health, and
keeps client stop strings dormant until GLM exits its reasoning block.

The runtime image is pulled by immutable GHCR digest and shipped to the worker.
The adapter runs MiaAI's GPU self-check on both Sparks, uses the upstream worker-
first lifecycle, and patches the materialized launcher so raw unauthenticated
vLLM binds only to `127.0.0.1:8888`. Only the existing authenticated allow-list
proxy binds `0.0.0.0:8000`. Tools (`glm47`), reasoning (`glm45`), image, and
video support remain enabled. Cold launch still requires 112 GiB `MemAvailable`
on each Spark, and startup performs a completion plus a 20-second EngineCore
watch before reporting ready.

License note: the EXL3 target uses the source-available ShapleyMCG License 1.0,
and DFlash2 is CC BY-NC-ND 4.0. Review both before deployment. For commercial
service without separate DFlash2 permission, set `SPEC_METHOD=mtp`; the
lifecycle then skips downloading and mounting DFlash2 and uses two-token MTP.

Run on the head Spark:

```bash
python3 -m ml.cli glm53-flash setup
python3 -m ml.cli glm53-flash pull
python3 -m ml.cli glm53-flash download
python3 -m ml.cli glm53-flash gpu-check
python3 -m ml.cli glm53-flash start
python3 -m ml.cli glm53-flash smoke
```

Or perform the complete initial staging and launch:

```bash
scripts/start-GLM53-Flash-Dual-DSpark.sh --first-run
```

The upstream revision, target/draft/image pins, RoCE addresses, MP settings,
and memory guards are in
[`config/dspark-glm53-flash-nvfp4.env`](config/dspark-glm53-flash-nvfp4.env).
The legacy profile filename is retained intentionally for existing automation.
At launch, the adapter resolves the actual netdev and RDMA HCA owning each
configured RoCE IP, so the two Sparks do not need identical interface names.
The lifecycle implementation is
[`scripts/GLM53-Flash-Dual-DSpark.sh`](scripts/GLM53-Flash-Dual-DSpark.sh).
The upstream checkout and caches remain under ignored `data/dspark/glm53-flash/`.
Pinned snapshots are reused and rsynced only when needed; allow at least 220 GiB
of free disk on both systems.

Like the other cluster recipes, raw vLLM binds only to `127.0.0.1:8888` and
the authenticated allow-list proxy owns public port 8000. Stop Qwen or
DeepSeek before using the same pair for GLM:

```bash
python3 -m ml.cli glm53-flash status
python3 -m ml.cli glm53-flash diagnose
python3 -m ml.cli glm53-flash memory
python3 -m ml.cli glm53-flash logs
python3 -m ml.cli glm53-flash logs-worker
python3 -m ml.cli glm53-flash stop
```

### DSpark One (one dedicated GB10 system)

The third-Spark path wraps
[MiaAI-Lab's one-DGX-Spark recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark)
at reviewed revision `fdcd538fbf95fb15b2d6850db9613d22b2c889b8`. It uses a
digest-pinned SparkInfer/ExLlamaV3 runtime and the 3.0-bpw EXL3 checkpoint with
TP=1. This is a separate lifecycle from `dspark`; it does not contact or
reconfigure the two-node cluster.

Run these commands on the dedicated third Spark:

```bash
cd ~/workspace/ml-compute
source venv/bin/activate

# First use: pin upstream, preflight, pull, download/coalesce ~107 GB, GPU test.
scripts/start-DS4-Flash-One-DSpark.sh --first-run

# Later boots:
scripts/start-DS4-Flash-One-DSpark.sh

python3 -m ml.cli dspark-one status
python3 -m ml.cli dspark-one memory
python3 -m ml.cli dspark-one smoke
python3 -m ml.cli dspark-one logs
```

The checked-in profile is intentionally dedicated-host: 384K context, one
sequence, `MAX_NUM_BATCHED_TOKENS=8224`, native 432-byte NVFP4 KV records, and
94% GPU/UMA utilization. Cold launch requires at least 114.3 GiB
`MemAvailable`, so stop other models and the Harness first. The raw,
unauthenticated server binds only to `127.0.0.1:8888`; the existing
authenticated allow-list proxy is the public API on port 8000. If using
Tailscale Funnel, target port 8000, never 8888.

### Call it

```bash
API_KEY=$(grep ^API_KEY= .env.local | cut -d= -f2-)

curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.6-27B-FP8","messages":[{"role":"user","content":"Say pong."}],"max_tokens":5}' \
  | python3 -m json.tool
```

The `model` field is whatever `status` reports — the HF ID for vLLM, the `served_name`
for llama.cpp, a LoRA `name` if the registry entry defines adapters.

### Stop / swap

```bash
python3 -m ml.cli stop                        # works for vLLM or llama.cpp
python3 -m ml.cli restart gemma4-12b-it       # stop + start a different vLLM model
python3 -m ml.cli router stop                 # stop public traffic before backends
python3 -m ml.cli docker stop                 # dedicated vLLM image lifecycle
python3 -m ml.cli vision stop                 # Transformers vision lifecycle
python3 -m ml.cli nim stop                    # NIM has its own lifecycle
python3 -m ml.cli dspark stop                 # stops both cluster nodes
python3 -m ml.cli qwen38-flash-next stop      # stops Qwen on both cluster nodes
python3 -m ml.cli glm53-flash stop            # stops GLM MP/vLLM on both nodes
python3 -m ml.cli dspark-one stop             # stops the independent one-Spark service
```

Backends bind port 8000 by default, so ordinary profiles still need to be stopped before
starting another. The DGX co-serving profile above is the explicit exception: it uses
loopback ports 8101–8103, capped registry entries, `--allow-co-resident`, and a router on
public port 8000. vLLM and llama.cpp share `data/server.json`; Docker vLLM, Transformers
vision, the router, NIM, and the DSpark/Qwen cluster recipes have separate state
and lifecycle commands.

---

## Reaching It From Other Machines

The server binds `0.0.0.0` by default, so on a trusted LAN just use the Spark's hostname:

```
Base URL:  http://spark.local:8000/v1
API key:   the API_KEY from .env.local on the Spark
```

Prefer not to expose the port? Tunnel over SSH from the client machine instead:

```bash
ssh -L 8000:localhost:8000 user@spark.local
```

Then the base URL is `http://localhost:8000/v1`.

### With llm-playground

No code changes — register a model with provider `openai`, the base URL above, the
`API_KEY`, and the model ID that `status` reports.

---

## Picking a Backend

| Situation | Use |
|-----------|-----|
| Serving safetensors, need LoRA-at-runtime, long context, prefix caching | **vLLM** |
| GGUF weights, huge quantized context, tight memory, or vLLM won't build | **llama.cpp** |
| Model requires a dedicated/custom vLLM image | **Docker vLLM** |
| Want NVIDIA-tuned TensorRT-LLM kernels and NVFP4 on Blackwell | **NIM** |
| DeepSeek V4 Flash 0731 across two linked GB10 nodes | **DSpark cluster** |
| Qwen3.8 Flash Next NVFP4 at 1M across two linked GB10 nodes | **Qwen Flash Next** |
| GLM-5.3 Flash EXL3 + DFlash2 at 1M across two linked GB10 nodes | **GLM Flash** |
| DeepSeek V4 Flash 0731 on one dedicated GB10 at 384K | **DSpark One** |
| Fine-tuning | none — stop the server, run `Makefile.gb10` |

On GB10, vLLM is the day-to-day workhorse (LoRA adapters and the registry live there).
NIM is faster on decode but heavier to operate and has a known memory bug on this box
(below). llama.cpp is the escape hatch: it always builds on aarch64 and its quantized KV
cache reaches contexts vLLM can't afford.

---

## Model Registries

Registry YAML files cover the serving backends and proxies. Adding a model is normally a
registry edit — no code changes.

| File | Backend | Key fields |
|------|---------|-----------|
| [registry/models.yaml](registry/models.yaml) | vLLM / Docker vLLM / Transformers vision / cluster recipes | `hf_id`, `serve_backend`, `docker_image`, `vision_config`, `external_recipe`, `max_model_len`, `quantization`, `gpu_memory_utilization`, `max_num_seqs`, `max_num_batched_tokens`, `enable_prefix_caching`, `enable_chunked_prefill`, `speculative_config`, `reasoning_parser`, `tool_call_parser`, `enable_auto_tool_choice`, `language_model_only`, `rope_scaling`, `loras`, `extra_args` |
| [registry/llama_models.yaml](registry/llama_models.yaml) | llama.cpp | `hf_repo` or `gguf_path`, `n_gpu_layers`, `ctx_size`, `served_name`, `extra_args` |
| [registry/nim_catalog.yaml](registry/nim_catalog.yaml) | NIM | `image`, `model_name`, `gpu_count`, `extra_env` |
| [registry/router.yaml](registry/router.yaml) | Single-port router | `host`, `port`, `backends`, `models`, `served_model`, `path_routes` |
| [registry/dspark_proxy.yaml](registry/dspark_proxy.yaml) | DSpark safety proxy | `host`, `port`, `upstream_url`, `max_concurrency`, `max_request_bytes` |

`registry/models.yaml` is the big one — every entry carries comments explaining *why* those
numbers, with measured memory breakdowns per model and per context length. Read them before
tuning. A representative slice of what's registered:

| Alias | Notes |
|-------|-------|
| `qwen3.6-27b-fp8` | Official FP8 checkpoint, 128k ctx — good default on Blackwell |
| `qwen3.6-27b-nvfp4` | NVIDIA ModelOpt NVFP4 — **Blackwell only** |
| `qwen3.8-27b-nvfp4-coserve` | 32k/one-request DGX Spark profile for the three-service stack |
| `pp-ocrv6-medium-det` | FP16 Transformers text-region detector, internal port 8103 |
| `qwen3.6-27b` | bf16 base — use this one for fine-tuning |
| `qwen3.5-9b-fp8` / `-fp8-long` | 40k throughput entry / 128k thinking-mode entry |
| `gemma4-12b-it` / `-it-fp8` / `-it-ft` | bf16 / runtime-FP8 / bf16 + LoRA adapter |
| `gemma4-31b-it-nvfp4` | NVIDIA NVFP4, 64k text profile for DGX Spark — **Blackwell only** |
| `gemma4-31b-it-fp8` | 31B via runtime FP8 quant — GB10-sized |
| `qwen2.5-coder-32b` (llama.cpp) | Q8_0 GGUF with `--jinja` — real tool calling for agentic coders |
| `qwen3.8-flash-next-nvfp4-dspark` | Dual-Spark SGLang TP2, SM121 QSA + NVFP4-KV patch, 1M YaRN profile |
| `glm-5.3-flash-nvfp4-dspark` | Legacy key for dual-Spark EXL3/MP TP2 + DFlash2, 1M profile |
| `deepseek-v4-flash-0731-dspark-one` | One-Spark TP=1 EXL3 recipe, 384K single-request profile |

Adding an entry:

```yaml
  my-model:
    hf_id: org/my-model
    vram_gb: 18                    # documentation only; not enforced
    max_model_len: 32768
    dtype: auto
    quantization: fp8
    gpu_memory_utilization: 0.85
    max_num_seqs: 4
    max_num_batched_tokens: 8192
    enable_prefix_caching: true
    enable_chunked_prefill: true
    speculative_config:             # mapping or inline JSON string
      method: mtp
      num_speculative_tokens: 2
    reasoning_parser: qwen3
    tool_call_parser: qwen3_coder
    enable_auto_tool_choice: true
    language_model_only: true
```

Then `python3 -m ml.cli models pull my-model && python3 -m ml.cli serve my-model`.

Serve a LoRA adapter alongside its base by adding `loras:` — clients select the fine-tune
by passing its `name` as the `model` field, or the base HF ID to bypass it:

```yaml
    loras:
      - name: gemma4-12b-gb10-001
        path: data/adapters/gemma4-12b-gb10-001
        rank: 16
```

Confirm a YAML edit actually parsed the way you meant:

```bash
python3 -c "from ml.config import get_models; import json; print(json.dumps(get_models()['my-model'], indent=2))"
```

---

## DGX Spark / GB10 Notes

Things that cost real debugging time on this hardware.

### Unified memory changes what `gpu_memory_utilization` means

There is no separate VRAM pool. `gpu_memory_utilization: 0.9` on a 128 GB Spark asks for
~115 GB — and the Grace CPU, your shell, and the page cache all live in that same 128 GB.
The registry entries deliberately sit at **0.7–0.85** for this reason. Push it higher and
you'll thrash the whole box, not just the GPU.

### Blackwell env vars are set automatically

For `sm_120+`, `ml/vllm_server.py` exports `TORCH_CUDA_ARCH_LIST`,
`VLLM_FLASHINFER_FORCE_TARGET`, and `VLLM_USE_FLASHINFER_SAMPLER=0` before launching vLLM.
Without them FlashInfer's JIT bails out with "requires GPUs with sm75 or higher" and the
engine never comes up.

### NVFP4 is Blackwell-only

`quantization: modelopt_fp4` needs native FP4 tensor cores (sm_100 / sm_120 / sm_121).
Halves the footprint versus FP8. On Ampere/Ada, vLLM either refuses to load or falls back
to a slow emulated path — use the FP8 entry there instead.

### NIM FP8 has a memory-runaway bug on GB10

The FP8 variant of the Qwen3.6-27B NIM surges to 115–124 GB after load and ignores every
memory-limit flag ([NVIDIA forum
thread](https://forums.developer.nvidia.com/t/urgent-dgx-spark-gb10-uma-nim-container-memory-runaway-all-limitation-parameters-invalid-memory-occupies-120gb/371180)).
Prefer NVFP4 or bf16 NIM tags, or serve that model with vLLM.

### llama.cpp must be built with OpenSSL

Without it, `-hf` auto-download fails with "HTTPS is not supported" and you have to fetch
the GGUF by hand and point `gguf_path` at it.

```bash
sudo apt install libssl-dev
cmake -B build -DGGML_CUDA=ON -DLLAMA_OPENSSL=ON
cmake --build build --config Release -j
export PATH="$PWD/build/bin:$PATH"
```

CUDA arch autodetects to `sm_121` (`BLACKWELL_NATIVE_FP4`). Newer model architectures
(e.g. `gemma4_unified`) need a recent llama.cpp checkout.

### aarch64 means no Unsloth

Unsloth, flash-attn, and xformers ship x86-only wheels. The training path here is plain
HF Transformers + PEFT + TRL ([ml/distill_train_hf.py](ml/distill_train_hf.py)), which is
~1.5–2× slower than Unsloth on x86 but fully native on ARM64. Unified memory also makes
QLoRA unnecessary — bf16 LoRA gives cleaner gradients and still fits.

### Not every GGUF can drive an agent

`gemma-4-12b-coder` (Q8_0) has **no tool/function calling** — it's code-gen and chat only,
and will loop or no-op inside opencode/aider. It also *requires* `temp 1.0 / top_p 0.95 /
top_k 64` or it degenerates into repetition (already set in its `extra_args`). For agentic
use pick a mainstream tool-calling model like `qwen2.5-coder-32b` with `--jinja`.

---

## Fine-Tuning on the Spark

bf16 LoRA via HF + PEFT + TRL. Driven by [Makefile.gb10](Makefile.gb10).

```bash
DATASET=datasets/my_data.jsonl        # {"messages":[{user},{assistant}]} per line

# 1. How long are these samples, really? Pick MAX_SEQ_LEN from this.
make -f Makefile.gb10 report DATASET=$DATASET BASE_MODEL=google/gemma-4-12B-it

# 2. Token-cap the dataset → <name>_train.jsonl
make -f Makefile.gb10 filter DATASET=$DATASET MAX_SEQ_LEN=25600

# 3. Worst-case memory check: the 5 BIGGEST records, 1 epoch. If this fits, the run fits.
make -f Makefile.gb10 smoke DATASET=$DATASET MAX_SEQ_LEN=25600

# 4. Train
make -f Makefile.gb10 train DATASET=$DATASET ADAPTER_NAME=my-ft-001 \
     BASE_MODEL=google/gemma-4-12B-it MAX_SEQ_LEN=25600 EPOCHS=3
```

Stop the inference server first — `python3 -m ml.cli stop`. Training and serving will
otherwise fight over the same unified memory.

Knobs: `ADAPTER_NAME`, `BASE_MODEL`, `MAX_SEQ_LEN`, `EPOCHS`, `RANK`, `LR`, `BATCH_SIZE`,
`GRAD_ACCUM`, `SAVE_EVERY`, `QLORA=1` (off by default). Adapters land in
`data/adapters/<ADAPTER_NAME>/`.

### Long runs that survive SSH drops

[scripts/train_gemma4_12b_gb10.sh](scripts/train_gemma4_12b_gb10.sh) token-filters in the
foreground, then `setsid nohup`s the trainer so it's reparented to init and ignores SIGHUP —
close your laptop and the run continues. It writes `data/distill_train.{pid,meta}` for the
status script:

```bash
DATASET=... ADAPTER_NAME=... bash scripts/train_gemma4_12b_gb10.sh

bash scripts/train_status.sh            # elapsed, latest loss/epoch/lr, recent errors
bash scripts/train_status.sh --tail     # follow live
bash scripts/train_status.sh --gpu      # + nvidia-smi
kill $(cat data/distill_train.pid)      # stop
```

[scripts/train_gemma4_gb10.sh](scripts/train_gemma4_gb10.sh) is the 31B sibling. It prints a
memory/wall-time projection and then calls `make -f Makefile.gb10 train` **in the
foreground** — wrap it yourself to detach (its confirmation prompt is skipped when stdin
isn't a tty):

```bash
setsid nohup bash scripts/train_gemma4_gb10.sh > data/logs/gemma4_31b.log 2>&1 < /dev/null &
```

Attention is O(n²) in sequence length, so context dominates wall-time. Rough numbers the
scripts themselves project: 12B, ~180 records × 3 epochs → 1.5–3 h at 16k, 4–6 h at 30k.
31B bf16 → 10–15 h at 16k, 25–40 h at 30k, and 30k peaks near ~103 GB — smoke-test first.

### Serving the adapter

Add a `loras:` block to the model's `registry/models.yaml` entry pointing at
`data/adapters/<name>`, then `serve` it. Match the adapter's training precision — a LoRA
trained on bf16 weights is cleanest served on the **bf16** entry, not a runtime-FP8 one.
For a permanent artifact, merge the adapter into the base and re-quantize instead.

---

## Distillation Pipeline

A larger workflow for building training data from a teacher model, driven by
[Makefile.distill](Makefile.distill):

```bash
make -f Makefile.distill clean-data   # normalize the seed export, swap in a short system prompt
make -f Makefile.distill synth        # synthetic expansion (needs GEMINI_API_KEY)
make -f Makefile.distill train        # QLoRA training
make -f Makefile.distill eval         # score the adapter against the seed set
```

The modules are usable standalone: `distill_clean`, `distill_synth`, `distill_augment`,
`distill_focus`, `distill_combine`, `distill_token_filter`, `distill_eval`.

Note this Makefile's `train` target uses the **4-bit QLoRA** path
([ml/distill_train.py](ml/distill_train.py), bitsandbytes — auto-detects bf16 vs
pre-quantized AWQ bases). On the Spark, use `Makefile.gb10` for the training step and this
pipeline only for the data steps.

## Evaluating

```bash
# Against a running vLLM server with the adapter loaded (fast, repeatable)
python3 -m ml.distill_eval \
  --eval-set data/datasets/seed_clean.jsonl \
  --vllm-url http://localhost:8000/v1 \
  --vllm-model google/gemma-4-12B-it \
  --vllm-adapter gemma4-12b-gb10-001

# Replay a backtest XLSX through the local endpoint and write the results back
python3 -m ml.backtest_runner \
  --input  prompts/backtest-in.xlsx \
  --output data/backtest-out.xlsx \
  --base-url http://localhost:8000/v1 \
  --model my-ft-001 \
  --summary
```

---

## CLI Reference

```bash
python3 -m ml.cli --help          # every command, with a workflow cheat-sheet epilog
```

### Models (vLLM weights)

```bash
python3 -m ml.cli models list              # registered aliases, ✓ = downloaded, sizes
python3 -m ml.cli models pull <alias>      # or a raw org/repo HF ID
python3 -m ml.cli models remove <alias>    # free disk
python3 -m ml.cli models size              # disk usage per model
```

### Serving

```bash
python3 -m ml.cli serve <alias>                  # vLLM (default backend)
python3 -m ml.cli serve <alias> --foreground     # run inline; prints the exact vllm command
python3 -m ml.cli serve <alias> --port 8001
python3 -m ml.cli serve llama <id> [--ngl 999] [--ctx 32768]
python3 -m ml.cli stop
python3 -m ml.cli restart <alias>
python3 -m ml.cli status                         # backend, model, PID, URL, readiness, API key
python3 -m ml.cli logs -f

# Dedicated Docker vLLM, optionally beside a capped local model
python3 -m ml.cli docker serve <alias> --port 8102 --allow-co-resident

# Transformers vision detector
python3 -m ml.cli vision serve pp-ocrv6-medium-det --port 8103
python3 -m ml.cli vision status
python3 -m ml.cli vision logs -f
python3 -m ml.cli vision stop

# Single public endpoint for all resident services
python3 -m ml.cli router serve --port 8000
python3 -m ml.cli router status
python3 -m ml.cli router logs -f
python3 -m ml.cli router stop
```

Benchmark the running server with vLLM's native serving benchmark. For a quick
throughput smoke test (8 requests at each concurrency, 512 input tokens, and
128 output tokens):

```bash
scripts/bench_vllm.sh quick

# Explicit model/tokenizer (useful when benchmarking a remote server).
OPEN_API_KEY=... \
VLLM_MODEL=unsloth/Qwen3.8-27B-NVFP4 \
scripts/bench_vllm.sh quick

# Standard run: 16 requests, 2k input tokens, and 256 output tokens.
scripts/bench_vllm.sh

# Remote server; both the host root and a URL ending in /v1 are accepted.
OPEN_API_KEY=... \
VLLM_BASE_URL=http://thinkstationpgx-fd9c.tail1c73a3.ts.net/v1 \
scripts/bench_vllm.sh quick

# Longer and more statistically stable run.
OPEN_API_KEY=... INPUT_LEN=8192 OUTPUT_LEN=512 NUM_PROMPTS=32 \
scripts/bench_vllm.sh
```

Results are printed as a concurrency comparison and saved as JSON under
`data/benchmarks/vllm/`. When `VLLM_MODEL` is omitted, vLLM uses the first model
reported by `/v1/models`; `.env.local` supplies the local `API_KEY` when present.
Override `VLLM_MODEL`, `VLLM_TOKENIZER`, `CONCURRENCIES`, `NUM_PROMPTS`,
`NUM_WARMUPS`, `INPUT_LEN`, `OUTPUT_LEN`, or `RESULT_DIR` as needed.

### Compare two coding models

The two-endpoint comparison runner measures five quality dimensions and produces
a self-contained HTML report plus auditable JSON. It uses pinned HumanEval+,
CRUXEval, NL2Bash, and BFCL v4 inputs. Code planning and the documentation part
of analysis use transparent custom rubrics because the external benchmarks do
not directly measure those requested behaviors.

```bash
cp config/model-compare.example.yaml config/model-compare.yaml
# Edit endpoint URLs, model IDs, labels, and API-key environment variable names.

# Put these in the shell environment or the ignored .env.local file.
export MODEL_A_API_KEY='...'
export MODEL_B_API_KEY='...'

# Optional: verify downloads, checksums, and deterministic case selection only.
python3 scripts/compare_models.py --config config/model-compare.yaml --prepare-only

# Run both models sequentially against the exact same cases.
python3 scripts/compare_models.py --config config/model-compare.yaml
```

Generated HumanEval+ code runs only inside a networkless, read-only Docker
container with CPU, memory, process, and time limits. On first use, the runner
builds `docker/model-compare-eval/Dockerfile`, which pins NumPy for the EvalPlus
tests. Startup verifies that NumPy imports successfully and stops with an
infrastructure error if the evaluator is incomplete, rather than recording
false model failures. The runner also refuses to run the comparison if Docker
is unavailable; generated Bash commands are scored but never executed. API keys
and custom header values are read at runtime and are not saved. Reports are
written to `data/model-comparisons/` by default, with bar charts for every
quality dimension, decode throughput, effective full-run tokens/second, time to
first token, latency, case errors, source revisions, and dataset SHA-256 hashes.

`benchmark.tool_mode: prompt` is the portable default for orchestration. Set it
to `native` only when both endpoints implement compatible OpenAI tool calling;
the selected mode is prominent in the report.

The common output budget defaults to `max_tokens_per_request: 16384`. Set it to
`null` to omit `max_tokens`; the endpoint may still enforce a server-side cap.
Reasoning controls are not assumed because APIs use different schemas. Put
provider-specific fields under each model's `request_body` (for example,
`reasoning_effort: high` or `chat_template_kwargs: {thinking: false}`). These
fields are sent verbatim and recorded in the report.

### NIM

```bash
python3 -m ml.cli nim models
python3 -m ml.cli nim serve <alias> [--gpus N] [--port P] [--foreground]
python3 -m ml.cli nim status         # reports EXITED + exit code/error when a container dies
python3 -m ml.cli nim stop
python3 -m ml.cli nim logs -f
```

### Box + config

```bash
python3 -m ml.cli info               # GPU, arch, library availability, tooling
python3 -m ml.cli config show
python3 -m ml.cli config gen-key     # new API key → .env.local
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `status` shows nothing right after `serve` | Engine crashed during load | `python3 -m ml.cli logs`, then re-run with `--foreground` |
| Stuck at `loading…` forever | Long first-load (quant pass, graph capture) or a hang | `logs -f`; FP8 runtime quant adds ~30 s on first load, NIM 30–90 s |
| `A server is already running` | Stale `data/server.json` or a live process | `stop`; if it won't die, see below |
| CUDA OOM on the Spark | `gpu_memory_utilization` too high, or context too long | Lower it to 0.7–0.85, drop `max_model_len`, or add `--kv-cache-dtype fp8` |
| Whole box goes unresponsive | Unified memory oversubscribed | Same as above — 0.9+ starves the Grace CPU |
| `FlashInfer requires GPUs with sm75 or higher` | Blackwell env vars missing | Serve via the CLI (it sets them), not raw `vllm serve` |
| `llama-server not found` | Not built / not on PATH | Build with `-DGGML_CUDA=ON -DLLAMA_OPENSSL=ON`, export `build/bin` |
| `HTTPS is not supported` from llama.cpp | Built without OpenSSL | Rebuild, or use `gguf_path` with a hand-downloaded GGUF |
| `OSError: ... gated repo` | Gated model, no token | `echo "HF_TOKEN=hf_..." >> .env.local` |
| 401 from `/v1/*` | Wrong key | `python3 -m ml.cli config show` |
| GGUF model loops or ignores tools | Model has no tool-calling template | Use a tool-calling model + `--jinja` |

Recovering from a wedged server:

```bash
pkill -9 -f "vllm serve" ; pkill -9 -f llama-server
rm -f data/server.json
nvidia-smi                        # confirm memory freed, no stragglers
python3 -m ml.cli serve <alias>
```

---

## Project Layout

```
ml-compute/
├── ml/
│   ├── cli.py                  Click CLI — every command lives here
│   ├── vllm_server.py          vLLM lifecycle + Blackwell env setup
│   ├── docker_server.py        dedicated vLLM Docker-image lifecycle
│   ├── vision_server.py        Transformers vision process lifecycle
│   ├── vision_api.py           text-detection HTTP API + one-request queue
│   ├── router_server.py        single-port router process lifecycle
│   ├── router_api.py           model-aware streaming reverse proxy
│   ├── dspark_proxy_server.py  DSpark safety-proxy lifecycle
│   ├── dspark_proxy_api.py     authenticated deny-by-default proxy
│   ├── vllm_args.py            shared registry → vLLM argument translation
│   ├── llama_server.py         llama.cpp (GGUF) lifecycle
│   ├── nim_server.py           NIM container lifecycle (Docker)
│   ├── models.py               HF download / list / remove
│   ├── state.py                PID + state file (data/server.json)
│   ├── config.py               .env.local + registry loading
│   ├── distill_train_hf.py     bf16 LoRA training (aarch64/GB10 path)
│   ├── distill_train.py        4-bit QLoRA training (bitsandbytes; bf16 or AWQ base)
│   ├── distill_train_unsloth.py  Unsloth QLoRA training (x86 only)
│   ├── distill_*.py            Data pipeline: clean, synth, augment, focus, filter, eval
│   ├── backtest_runner.py      Replay an XLSX backtest through a local endpoint
│   └── gemma4_*.py             Gemma 4 attention / PEFT patches
├── registry/
│   ├── models.yaml             vLLM aliases — heavily commented, the reference doc
│   ├── llama_models.yaml       GGUF aliases
│   ├── nim_catalog.yaml        NIM container catalog
│   ├── router.yaml             public model IDs → internal backend URLs
│   ├── dspark_proxy.yaml       private cluster API → safe public endpoint
│   ├── datasets.yaml           Fine-tuning dataset aliases
│   └── apis.yaml               Hosted-provider model lists (teacher models)
├── config/
│   ├── dspark-spark4e89-thinkstationpgx.env   dual-Spark DeepSeek profile
│   ├── dspark-qwen38-flash-next-nvfp4.env     dual-Spark Qwen 1M NVFP4-KV profile
│   ├── dspark-glm53-flash-nvfp4.env            dual-Spark GLM EXL3 1M profile (legacy name)
│   └── dspark-one-deepseek-v4-flash-0731.env  one-Spark 384K profile
├── Makefile.gb10               LoRA fine-tuning on DGX Spark (bf16, HF+PEFT+TRL)
├── Makefile.distill            Distillation data pipeline (+ x86 QLoRA train)
├── Makefile.qwen{,9b}          QLoRA recipes for a 32 GB RTX PRO 4500 box
├── scripts/
│   ├── DS4-Flash-DSpark.sh         dual-Spark DeepSeek lifecycle
│   ├── Qwen38-Flash-Next-Dual-DSpark.sh dual-Spark Qwen lifecycle
│   ├── start-Qwen38-Flash-Next-Dual-DSpark.sh Qwen first-run/start wrapper
│   ├── GLM53-Flash-Dual-DSpark.sh  dual-Spark GLM MP/vLLM lifecycle
│   ├── start-GLM53-Flash-Dual-DSpark.sh GLM first-run/start wrapper
│   ├── DS4-Flash-One-DSpark.sh     one-Spark lifecycle + safety checks
│   ├── start-DS4-Flash-One-DSpark.sh first-run / normal-start wrapper
│   ├── train_gemma4_12b_gb10.sh   Detached (setsid+nohup) 12B run
│   ├── train_gemma4_gb10.sh       31B run — wraps Makefile.gb10, foreground
│   └── train_status.sh            Progress / loss / errors for a detached run
├── datasets/                   Training data (.jsonl, chat-messages format)
├── prompts/                    System prompts + backtest workbooks
├── data/                       Created by setup.sh
│   ├── hf_cache/               HuggingFace weights
│   ├── adapters/<name>/        Trained LoRA adapters
│   ├── logs/{vllm,llama,vision,router,dspark_proxy}.log  Service logs
│   ├── server.json             Running server state (PID, port, model, backend)
│   ├── docker_state.json       Managed Docker vLLM container state
│   ├── vision_server.json      Transformers vision server state
│   ├── router_server.json      Model-router server state
│   └── dspark_proxy_server.json DSpark safety-proxy state
├── setup.sh                    Arch-aware one-shot setup
├── requirements.txt            x86_64 dependency set
└── .env.example                Template for .env.local
```

---

## Environment Variables

All optional — defaults in [ml/config.py](ml/config.py). Set them in `.env.local`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `API_KEY` | (generated by `setup.sh`) | Bearer token clients send |
| `HF_TOKEN` | — | Gated models (Llama, Gemma). Omit the line entirely if unset — an empty value breaks the auth header |
| `NGC_API_KEY` | — | Required for NIM containers |
| `VLLM_HOST` | `0.0.0.0` | Bind address (shared by all backends) |
| `VLLM_PORT` | `8000` | Default local/Docker language-model port; CLI `--port` overrides it |
| `DATA_DIR` | `./data` | Logs, adapters, server state, HF cache |
| `REGISTRY_DIR` | `./registry` | Where the YAML registries load from |
| `HF_HOME` | `./data/hf_cache` | HuggingFace cache root |

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is worth exporting for training — the
detached scripts set it already.

---

> `DEPLOYMENT.md` documents an older Vast.ai cloud deployment and does not apply to a local
> Spark setup.
