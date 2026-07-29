# ml-compute Deployment Guide (vLLM)

Deploy ml-compute on a Vast.ai instance to serve open-source LLMs over an OpenAI-compatible HTTP API.

---

## ⚡ One-Liner Bootstrap

After SSHing into your Vast.ai instance:

```bash
cd /workspace && git clone https://github.com/gkato/ml-compute.git && cd ml-compute && REMOTE_MODE=1 bash setup.sh
```

Then pull a model and serve it:

```bash
python3 -m ml.cli models pull qwen2.5-7b
python3 -m ml.cli serve qwen2.5-7b
python3 -m ml.cli status
```

The setup script auto-generates an `API_KEY` and prints it. You'll use that from your laptop.

---

## Step-by-Step

### 1. Pick a Vast.ai instance

| Need | GPU |
|------|-----|
| 3B–8B inference | ≥12 GB VRAM (RTX 3060, 4060 Ti, A4000) |
| 7B–13B inference | ≥16 GB VRAM (RTX 4070 Ti, 3090, A4500) |
| 13B–30B inference | ≥24 GB VRAM (RTX 3090, 4090, A5000+) |
| 30B+ or unquantized 14B | 32–48 GB VRAM (L40S, A6000) |

**Other recommendations:**
- Disk ≥ 50 GB (models are 5–30 GB each)
- Use a PyTorch base image — saves install time
- Expose port 8000 in the instance config (Vast.ai maps it to a public port)

### 2. Connect

In the Vast.ai dashboard, click **Connect** to get the exact SSH command:

```bash
ssh -p <PORT> root@<HOST>
```

For local-only API access (recommended for security), add port forwarding:

```bash
ssh -p <PORT> -L 8000:localhost:8000 root@<HOST>
```

### 3. Clone the repo

**Public repo:**
```bash
cd /workspace
git clone https://github.com/gkato/ml-compute.git
cd ml-compute
```

**Private repo (deploy key):**
```bash
ssh-keygen -t ed25519 -C "vastai-deploy"
cat ~/.ssh/id_ed25519.pub   # add to GitHub → Settings → Deploy keys
git clone git@github.com:gkato/ml-compute.git
```

### 4. Run setup

```bash
REMOTE_MODE=1 bash setup.sh
```

This will:
- Create `data/{logs,hf_cache}` directories
- Install vLLM + huggingface-hub + hf-transfer + click + pyyaml
- Create `.env.local` with an auto-generated `API_KEY`
- Verify GPU visibility

### 5. Add HF token (only for gated models)

Llama 3.x and Gemma require accepting a HuggingFace license. Get a token at https://huggingface.co/settings/tokens, then:

```bash
echo "HF_TOKEN=hf_xxx..." >> .env.local
```

### 6. Pull a model

```bash
python3 -m ml.cli models pull qwen2.5-7b
```

For a 7B model expect ~15 GB of weights. The Hugging Face transfer accelerator is enabled, so this is fast on a beefy network link.

```bash
python3 -m ml.cli models list
python3 -m ml.cli models size
```

### 7. Start the server

```bash
python3 -m ml.cli serve qwen2.5-7b
```

This launches vLLM in the background, writes `data/server.json`, and tails `data/logs/vllm.log`. First-time startup takes 30–90 sec while the model loads into VRAM.

```bash
python3 -m ml.cli status        # waits-then-shows readiness
python3 -m ml.cli logs -f       # follow vLLM logs
```

### 8. Test from the server

```bash
API_KEY=$(grep ^API_KEY= .env.local | cut -d= -f2-)

curl http://localhost:8000/v1/models -H "Authorization: Bearer $API_KEY"

curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role":"user","content":"Say hi in one word"}]
  }'
```

### 9. Test from your laptop

With the SSH `-L 8000:localhost:8000` tunnel active:

```bash
# from laptop
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer <your-API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role":"user","content":"Hello"}]
  }'
```

---

## Swapping Models

Only one model fits per GPU. To switch:

```bash
python3 -m ml.cli restart gemma3-12b
```

Or stop, pull, and start manually:

```bash
python3 -m ml.cli stop
python3 -m ml.cli models pull llama-3.1-8b
python3 -m ml.cli serve llama-3.1-8b
```

---

## Wiring up llm-playground

Add a new model config in llm-playground's UI (or via its API):

| Field | Value |
|-------|-------|
| Provider | `openai` |
| Base URL | `http://localhost:8000/v1` (with SSH tunnel) |
| API Key | the contents of `API_KEY` on the server |
| Model ID | the HuggingFace ID, e.g. `Qwen/Qwen2.5-7B-Instruct` |

llm-playground's existing OpenAI provider hits `/v1/chat/completions` with the bearer token — exactly what vLLM expects. **No code changes** are needed in llm-playground.

---

## Persistence Tips

Vast.ai instances are ephemeral. If you destroy the instance, the HuggingFace cache is gone.

**Options:**
- **Persistent volume** (best): mount a Vast.ai volume at `/workspace/ml-compute/data/hf_cache`
- **rsync to laptop** before destroying:
  ```bash
  rsync -av -e "ssh -p <PORT>" \
    root@<HOST>:/workspace/ml-compute/data/hf_cache/ \
    ./hf_cache_backup/
  ```
- **Re-pull on next instance** — for a 7B model, ~5 minutes on a fast link

---

## Keeping the Server Alive

`ml-compute serve` already detaches with `nohup` and a session group, so it survives SSH disconnects. To verify after reconnecting:

```bash
python3 -m ml.cli status
```

If the PID is gone (instance was rebooted), just `serve` again — the model is cached locally.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `vllm: command not found` | Setup didn't complete | `pip install -r requirements.txt` |
| `OSError: ... gated repo` | Missing HF token for Llama/Gemma | Add `HF_TOKEN=hf_...` to `.env.local`, restart |
| `CUDA out of memory` | Model too big for VRAM | Use a smaller alias or `quantization: awq` in registry |
| `Status: not running` immediately | vLLM crashed during load | `python3 -m ml.cli logs` to see error |
| 401 from `/v1/...` | Wrong API key | Check `python3 -m ml.cli config show` |
| Connection refused from laptop | SSH tunnel down | Reconnect with `-L 8000:localhost:8000` |

For deeper debugging:
```bash
python3 -m ml.cli stop
python3 -m ml.cli serve qwen2.5-7b --foreground   # see crashes inline
```

---

## What's Next

- **Cloudflare tunnel** if you want a persistent public URL instead of SSH forwarding
- **Multiple servers** on different ports (different GPUs) — extend `VLLM_PORT` per session
- **LoRA hot-swap** — phase 2, will add `vllm serve --enable-lora` integration
