"""Measurements used in the paper's analysis (sections 4.2 and 4.3).

- mean gate score per layer: how open the gates are on average (the paper reports a mean of 0.116 for
  elementwise gating in its 15B model, i.e. the gates are sparse and mostly closed)
- attention on the first token per layer: the "attention sink" (the paper reports 46.7% on average for
  the ungated baseline and 4.8% with the gate)
"""

from typing import Dict, List

import torch

from .model import GatedLM


@torch.no_grad()
def sink_and_gate_report(model: GatedLM, idx: torch.Tensor) -> Dict[str, List[float]]:
    """Per layer: share of attention that queries 1..T-1 put on token 0, and the mean gate score."""
    was_training = model.training
    model.eval()
    first_token, gate_mean = [], []
    x = model.tok_emb(idx)
    for layer in model.layers:
        h = layer.attn_norm(x)
        probs = layer.attn.attention_probs(h, model.rope_cos, model.rope_sin)   # (b, h, t, t)
        first_token.append(probs[:, :, 1:, 0].mean().item())                     # skip query 0: it can only see itself
        if layer.attn.wg is not None:
            gate_mean.append(layer.attn.gate_scores(h).mean().item())
        x = layer(x, model.rope_cos, model.rope_sin)
    model.train(was_training)
    return {"first_token_attention": first_token, "mean_gate": gate_mean}
