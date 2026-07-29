"""Memory-efficient chunked attention forward for Gemma 4 on a single A100.

Why this exists
---------------
Gemma 4 has head_dim=1024. Every memory-efficient attention kernel rejects
or struggles with this:
  - Flash Attention 2 / 3: head_dim ≤ 256 / 512 (hard)
  - PyTorch SDPA `mem_efficient`: head_dim ≤ 128 in our config
  - PyTorch `flex_attention`: Triton tiles exceed A100's 167 KB SM budget
  - PyTorch SDPA `math`: works at any head_dim but allocates O(seq²),
    OOMs at seq 24K on A100-80G (~48 GB attention tensor alone)

This module implements attention in pure PyTorch but chunks along the
query dimension, so memory is O(chunk_size × seq) instead of O(seq²).
At chunk_size=1024 and seq=24576, peak attention tensor is ~384 MB
instead of ~48 GB.

Compatible with transformers' AttentionInterface API. Drop-in for
sdpa_attention_forward.

Usage
-----
    from ml.gemma4_chunked_attn import register_chunked_attention
    register_chunked_attention()
    # then pass attn_implementation="chunked" to from_pretrained
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Chunk size in query tokens. Smaller = less peak memory per chunk
# but more Python-loop overhead. 1024 is a good default; tune via env var.
_DEFAULT_CHUNK = int(os.environ.get("CHUNKED_ATTN_CHUNK", "1024"))


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Tile K/V along the head axis to match Q's head count (for GQA)."""
    if n_rep == 1:
        return hidden_states
    b, kv_h, s, d = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(b, kv_h, n_rep, s, d)
        .reshape(b, kv_h * n_rep, s, d)
    )


def chunked_attention_forward(
    module,
    query: torch.Tensor,        # [B, num_q_heads, S, D]
    key: torch.Tensor,          # [B, num_kv_heads, S, D]
    value: torch.Tensor,        # [B, num_kv_heads, S, D]
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """Drop-in replacement for sdpa_attention_forward.

    Computes attention per Q-chunk:
      scores_i = (Q_i @ K^T) * scale
      attn_i   = softmax(scores_i + mask_i)
      out_i    = attn_i @ V

    Peak memory: O(chunk × seq) per chunk, vs O(seq²) for full SDPA-math.

    Returns (output [B, S, H, D], None) matching sdpa_attention_forward.
    """
    B, num_q_heads, S, D = query.shape
    num_kv_heads = key.shape[1]
    n_rep = num_q_heads // num_kv_heads

    if scaling is None:
        scaling = 1.0 / (D ** 0.5)

    # GQA: expand K/V to match Q's head count
    if n_rep > 1:
        key = _repeat_kv(key, n_rep)
        value = _repeat_kv(value, n_rep)

    # Ensure dtype consistency for matmul
    if key.dtype != query.dtype:
        key = key.to(query.dtype)
    if value.dtype != query.dtype:
        value = value.to(query.dtype)

    chunk_size = _DEFAULT_CHUNK
    out = torch.empty_like(query)

    # Precompute K transpose once
    key_t = key.transpose(-2, -1)  # [B, H, D, S]

    for i in range(0, S, chunk_size):
        end = min(i + chunk_size, S)
        q_chunk = query[:, :, i:end]                          # [B, H, c, D]
        scores = torch.matmul(q_chunk, key_t) * scaling       # [B, H, c, S]

        # Apply mask. Gemma 4's preprocessing produces an additive mask
        # of shape [B, 1, S_q, S_k] (or [B, H, S_q, S_k]).
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                mask_chunk = attention_mask[..., i:end, :]
                scores = scores + mask_chunk
            else:
                raise ValueError(
                    f"Unexpected attention_mask shape: {attention_mask.shape}"
                )
        elif is_causal:
            # Build a causal mask for this chunk on the fly
            pos_q = torch.arange(i, end, device=query.device).unsqueeze(1)
            pos_k = torch.arange(S, device=query.device).unsqueeze(0)
            causal_mask = (pos_k > pos_q).to(scores.dtype) * float("-inf")
            # NaN-safe: replace inf*0 cases
            causal_mask = torch.where(
                (pos_k > pos_q), torch.tensor(float("-inf"), device=query.device, dtype=scores.dtype),
                torch.tensor(0.0, device=query.device, dtype=scores.dtype),
            )
            scores = scores + causal_mask

        # Softmax in fp32 for stability, cast back
        attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)

        if dropout > 0.0:
            attn = F.dropout(attn, p=dropout)

        out[:, :, i:end] = torch.matmul(attn, value)

    # Match sdpa_attention_forward's return: [B, S, H, D]
    out = out.transpose(1, 2).contiguous()
    return out, None


def register_chunked_attention() -> None:
    """Register `chunked` as a transformers attention implementation.

    Tries the new AttentionInterface API first, falls back to the legacy
    ALL_ATTENTION_FUNCTIONS dict. Works for transformers >= 4.46.
    """
    registered = False
    try:
        from transformers.modeling_utils import AttentionInterface
        if hasattr(AttentionInterface, "register"):
            AttentionInterface.register("chunked", chunked_attention_forward)
            registered = True
    except Exception:
        pass

    if not registered:
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            ALL_ATTENTION_FUNCTIONS["chunked"] = chunked_attention_forward
            registered = True
        except Exception:
            pass

    if not registered:
        raise RuntimeError(
            "Could not register chunked attention — transformers API changed; "
            "expected either AttentionInterface.register or ALL_ATTENTION_FUNCTIONS"
        )
