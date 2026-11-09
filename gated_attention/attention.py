"""Gated attention (Qiu et al., 2025, https://arxiv.org/abs/2505.06708).

A sigmoid gate, computed from the layer input X, multiplies the output of scaled dot-product
attention, head by head (paper eq. 5, position "G1"):

    Y' = Y * sigmoid(X W_theta)

X is the hidden state after the pre-normalisation (footnote 1 of the paper). The gate is
head-specific and multiplicative. It comes in two granularities:
  - elementwise: one gate value per head dimension, W_theta is (d_model, n_heads * head_dim)
  - headwise:    one gate value per head,           W_theta is (d_model, n_heads)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GatedConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


def precompute_rope(head_dim: int, max_len: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start: int = 0) -> torch.Tensor:
    t = x.shape[-2]
    c = cos[start : start + t].to(x.dtype)
    s = sin[start : start + t].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


class KVCache:
    def __init__(self):
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def update(self, k, v):
        self.k = k if self.k is None else torch.cat([self.k, k], dim=2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=2)
        return self.k, self.v


def causal_mask(t: int, s: int, device) -> Optional[torch.Tensor]:
    """(t, s) mask: of s keys the last t are the queries' own positions; query i sees j <= s - t + i."""
    if t == 1:
        return None
    return torch.arange(s, device=device)[None, :] <= torch.arange(s - t, s, device=device)[:, None]


class GatedAttention(nn.Module):
    def __init__(self, cfg: GatedConfig):
        super().__init__()
        self.nh, self.dh, self.gate_kind = cfg.n_heads, cfg.head_dim, cfg.gate
        inner = self.nh * self.dh
        self.wq = nn.Linear(cfg.d_model, inner, bias=False)
        self.wk = nn.Linear(cfg.d_model, inner, bias=False)
        self.wv = nn.Linear(cfg.d_model, inner, bias=False)
        self.wo = nn.Linear(inner, cfg.d_model, bias=False)
        if cfg.gate == "none":
            self.wg = None
        else:
            self.wg = nn.Linear(cfg.d_model, inner if cfg.gate == "elementwise" else self.nh, bias=False)
        self.record = False            # set True to keep the last gate scores for inspection
        self.last_gate: Optional[torch.Tensor] = None

    def _qkv(self, x, cos, sin, cache):
        b, t, _ = x.shape
        q, k, v = (w(x).view(b, t, self.nh, self.dh).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        start = cache.length if cache is not None else 0
        q, k = apply_rope(q, cos, sin, start), apply_rope(k, cos, sin, start)
        if cache is not None:
            k, v = cache.update(k, v)
        return q, k, v

    def gate_scores(self, x: torch.Tensor) -> torch.Tensor:
        """sigmoid(X W_theta), shaped (b, h, t, d_h) for elementwise or (b, h, t, 1) for headwise."""
        b, t, _ = x.shape
        g = torch.sigmoid(self.wg(x))
        width = self.dh if self.gate_kind == "elementwise" else 1
        return g.view(b, t, self.nh, width).transpose(1, 2)

    def forward(self, x, cos, sin, cache: Optional[KVCache] = None):
        b, t, _ = x.shape
        q, k, v = self._qkv(x, cos, sin, cache)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask(t, k.shape[2], x.device))
        if self.wg is not None:
            g = self.gate_scores(x)
            if self.record:
                self.last_gate = g.detach()
            y = y * g
        return self.wo(y.transpose(1, 2).reshape(b, t, -1))

    @torch.no_grad()
    def attention_probs(self, x, cos, sin) -> torch.Tensor:
        """Full causal softmax attention map (b, h, t, t), for analysis only (no cache)."""
        b, t, _ = x.shape
        q, k, _ = self._qkv(x, cos, sin, None)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        mask = causal_mask(t, t, x.device)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        return scores.softmax(dim=-1)
