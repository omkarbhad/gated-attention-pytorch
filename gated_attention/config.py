"""Model configuration."""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class GatedConfig:
    vocab_size: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    head_dim: Optional[int] = None  # default: d_model // n_heads
    gate: str = "elementwise"       # "none", "headwise" or "elementwise" (position G1 in the paper)
    ffn_dim: Optional[int] = None   # default: 8/3 * d_model, rounded up to a multiple of 64
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.gate not in ("none", "headwise", "elementwise"):
            raise ValueError("gate must be 'none', 'headwise' or 'elementwise'")
        if self.head_dim is None:
            if self.d_model % self.n_heads:
                raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
            self.head_dim = self.d_model // self.n_heads
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.ffn_dim is None:
            self.ffn_dim = (int(8 / 3 * self.d_model) + 63) // 64 * 64

    def to_dict(self) -> dict:
        return asdict(self)
