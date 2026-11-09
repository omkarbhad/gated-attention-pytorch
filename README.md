# gated-attention-pytorch

A from-scratch PyTorch implementation of **gated attention** from
[Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708)
(Qiu et al., Qwen team, 2025), with tools to measure the two effects the paper analyses: gate sparsity and
the attention sink.

This is an unofficial implementation written from the paper (section 2.2 and Table 1).

## The idea

After scaled dot-product attention, multiply each head's output by a **sigmoid gate computed from the
layer input**:

```
Y' = Y * sigmoid(X W_theta)          (paper eq. 5, position "G1")
```

`X` is the hidden state after the pre-normalisation. The gate is head-specific and multiplicative. It has
two granularities:

- **elementwise:** one gate value per head dimension, `W_theta` is `(d_model, n_heads * head_dim)`
- **headwise:** one gate value per head, `W_theta` is `(d_model, n_heads)`

In the paper's experiments (15B MoE and 1.7B dense models, up to 3.5T tokens) this one change lowers
perplexity, makes training more stable, and removes the "attention sink": ungated models send 46.7% of
attention to the first token on average, gated ones 4.8%. The paper explains the gain by non-linearity
and by input-dependent sparsity of the gate.

## Install

```bash
git clone https://github.com/omkarbhad/gated-attention-pytorch
cd gated-attention-pytorch
pip install -e ".[dev]"
```

## Usage

```python
import torch
from gated_attention import GatedConfig, GatedLM, sink_and_gate_report

cfg = GatedConfig(vocab_size=256, d_model=256, n_layers=4, n_heads=4, gate="elementwise")  # or "headwise", "none"
model = GatedLM(cfg)

idx = torch.randint(0, 256, (2, 64))
logits, loss = model(idx, targets=idx.roll(-1, 1))
out = model.generate(idx[:, :8], max_new_tokens=32, temperature=0.8)     # uses a KV cache

report = sink_and_gate_report(model, idx)
report["first_token_attention"]   # per layer: share of attention that queries put on token 0
report["mean_gate"]               # per layer: mean gate score (empty for gate="none")
```

## What is implemented

| Paper | Here |
| --- | --- |
| Eq. 5, `Y' = Y * sigmoid(X W_theta)` after SDPA (G1) | `GatedAttention.forward` |
| Head-specific, multiplicative, sigmoid (the default variant) | `gate="elementwise"` / `"headwise"` |
| Footnote 1: the gate reads the pre-normalised hidden state | `Block.forward` |
| Section 4.2: mean gate score (sparsity) | `sink_and_gate_report` -> `mean_gate` |
| Section 4.3: attention to the first token (sink) | `sink_and_gate_report` -> `first_token_attention` |

Not implemented: the other gate positions (G2-G5), head-shared and additive gates, and GQA. The model is
LLaMA-style: pre-RMSNorm, rotary embeddings, SwiGLU, tied embeddings, KV cache.

## Tests

```bash
pytest -q
```

33 tests. The main ones:

- The module equals eq. 5 computed by hand, for both gate granularities.
- `gate="none"` is plain attention with no extra parameters, and the added parameters are exactly `d_model * n_heads` (headwise) or `d_model * n_heads * head_dim` (elementwise) per layer.
- Gate scores lie in (0, 1) and depend on the input. A closed gate silences a head, and an open gate gives plain attention.
- The gate reads the normalised input inside a block.
- The attention map is a causal softmax and reproduces the attention output. On uniform attention the sink measure equals `mean(1/(i+1))`, as it must.
- No token sees the future, and decoding with the cache matches a full forward pass, including a chunk after a cached prefix.

I checked that the tests fail when I break the code on purpose: drop the sigmoid, feed the gate the
un-normalised input, put the gate in the wrong place, or remove the causal mask.

## A small experiment

`examples/train_shakespeare.py` trains a character-level model on tiny Shakespeare. Same seed, data order and schedule.

```bash
python examples/train_shakespeare.py --gate none
python examples/train_shakespeare.py --gate headwise
python examples/train_shakespeare.py --gate elementwise
```

Settings: 4 layers, `d_model` 128, 4 heads, context 128, 2,000 steps. Validation loss at the end, 3 seeds:

| Gate | Seed 0 | Seed 1 | Seed 2 | Mean (nats) | Params | First-token attention | Mean gate score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | 1.5047 | 1.5013 | 1.5102 | 1.5054 | 861,440 | 0.015 | - |
| headwise | 1.5097 | 1.5125 | 1.5084 | 1.5102 | 863,488 | 0.015 | 0.39 |
| elementwise | 1.5042 | 1.4953 | 1.5068 | **1.5021** | 926,976 | 0.014 | 0.34 |

What this shows, and what it does not:

- **Elementwise gating gives a slightly lower loss** than no gate in all three seeds, by 0.003 nats on average.
  That is small, similar in size to the spread between seeds. Headwise gating is slightly worse than no gate in two of three seeds.
- **The elementwise model has 7.6% more parameters.** I did not train a parameter-matched baseline, so the small
  gain could come from the extra parameters, not from the gate. The paper addresses this with parameter-expanding baselines.
- **There is no attention sink to remove here.** The ungated model already puts only 1.5% of attention on the first
  token (uniform attention over a 128-token context would give about 3.5%), so the gate cannot lower it. The paper's sink appears in large models trained on trillions of tokens.
- **The gates are moderately sparse:** mean gate score 0.34 (elementwise) and 0.39 (headwise), below the initial 0.5,
  but far from the paper's 0.116.

So this run does not reproduce the paper's results. It is a small, short run, and it neither confirms nor refutes the paper.

## Layout

```
gated_attention/
  config.py       GatedConfig
  attention.py    GatedAttention, RoPE, KV cache, causal mask
  model.py        GatedLM, generate
  analysis.py     sink_and_gate_report
tests/            33 tests
examples/         train_shakespeare.py
```

## Citation

```bibtex
@article{qiu2025gated,
    title   = {Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free},
    author  = {Zihan Qiu and Zekun Wang and Bo Zheng and Zeyu Huang and Kaiyue Wen and Songlin Yang and Rui Men and Le Yu and Fei Huang and Suozhi Huang and Dayiheng Liu and Jingren Zhou and Junyang Lin},
    journal = {arXiv preprint arXiv:2505.06708},
    year    = {2025}
}
```

## License

MIT
