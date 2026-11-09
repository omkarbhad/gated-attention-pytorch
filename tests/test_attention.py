import math

import pytest
import torch
import torch.nn.functional as F

from gated_attention import GatedAttention, sink_and_gate_report
from gated_attention.attention import apply_rope, precompute_rope
from conftest import cfg, make


def heads_before_gate(attn, x, cos, sin):
    """Plain causal attention output per head, (b, h, t, d_h), computed without using the module's forward."""
    b, t, _ = x.shape
    q, k, v = (w(x).view(b, t, attn.nh, attn.dh).transpose(1, 2) for w in (attn.wq, attn.wk, attn.wv))
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    s = (q @ k.transpose(-1, -2)) / math.sqrt(attn.dh)
    s = s.masked_fill(~(torch.arange(t)[None, :] <= torch.arange(t)[:, None]), -torch.inf)
    return s.softmax(-1) @ v


@pytest.mark.parametrize("kind", ["headwise", "elementwise"])
def test_matches_paper_eq5(kind):
    """Y' = Y * sigmoid(X W_theta), per head, then the output projection."""
    c = cfg(gate=kind)
    attn = GatedAttention(c)
    x = torch.randn(2, 7, c.d_model)
    cos, sin = precompute_rope(c.head_dim, 16, 10000.0)
    y = heads_before_gate(attn, x, cos, sin)
    b, t, h, d = 2, 7, c.n_heads, c.head_dim
    gate = torch.sigmoid(x @ attn.wg.weight.T)                    # (b, t, h*d) or (b, t, h)
    gate = gate.view(b, t, h, d if kind == "elementwise" else 1).transpose(1, 2)
    want = attn.wo((y * gate).transpose(1, 2).reshape(b, t, -1))
    torch.testing.assert_close(attn(x, cos, sin), want, atol=1e-5, rtol=1e-5)


def test_no_gate_is_plain_attention_with_no_extra_parameters():
    c = cfg(gate="none")
    attn = GatedAttention(c)
    assert attn.wg is None
    x = torch.randn(1, 6, c.d_model)
    cos, sin = precompute_rope(c.head_dim, 16, 10000.0)
    want = attn.wo(heads_before_gate(attn, x, cos, sin).transpose(1, 2).reshape(1, 6, -1))
    torch.testing.assert_close(attn(x, cos, sin), want, atol=1e-5, rtol=1e-5)


def test_added_parameters():
    base = make(gate="none").num_params()
    c = cfg()
    assert make(gate="headwise").num_params() - base == c.n_layers * c.d_model * c.n_heads
    assert make(gate="elementwise").num_params() - base == c.n_layers * c.d_model * c.n_heads * c.head_dim


def test_gate_scores_are_in_unit_interval_and_depend_on_the_input():
    c = cfg(gate="elementwise")
    attn = GatedAttention(c)
    nn_init = torch.nn.init.normal_
    nn_init(attn.wg.weight, std=1.0)
    g = attn.gate_scores(torch.randn(2, 9, c.d_model))
    assert g.shape == (2, c.n_heads, 9, c.head_dim)
    assert g.min() >= 0 and g.max() <= 1
    assert g[0, 0, 0].std() > 0 and not torch.allclose(g[:, :, 0], g[:, :, 1])   # differs across dims and tokens


def test_closed_gate_silences_the_head_and_open_gate_is_plain_attention():
    c = cfg(gate="headwise")
    attn = GatedAttention(c)
    x = torch.rand(1, 6, c.d_model) + 0.1       # all positive, so a constant weight gives a large score
    cos, sin = precompute_rope(c.head_dim, 16, 10000.0)
    with torch.no_grad():
        attn.wg.weight.fill_(-50.0)
        closed = attn(x, cos, sin)
        attn.wg.weight.fill_(+50.0)
        opened = attn(x, cos, sin)
    assert closed.abs().max() < 1e-6
    plain = attn.wo(heads_before_gate(attn, x, cos, sin).transpose(1, 2).reshape(1, 6, -1))
    torch.testing.assert_close(opened, plain, atol=1e-4, rtol=1e-4)


def test_gate_reads_the_normalised_input_inside_a_block():
    m = make(gate="elementwise")
    block = m.layers[0]
    x = torch.randn(1, 5, 64) * 7.0     # large scale so that norm vs no norm would differ a lot
    seen = {}
    block.attn.register_forward_pre_hook(lambda mod, args: seen.update(x=args[0]))
    block(x, m.rope_cos, m.rope_sin)
    torch.testing.assert_close(seen["x"], block.attn_norm(x))


def test_attention_probs_are_a_causal_softmax_and_reproduce_sdpa():
    c = cfg(gate="none")
    attn = GatedAttention(c)
    x = torch.randn(2, 8, c.d_model)
    cos, sin = precompute_rope(c.head_dim, 16, 10000.0)
    p = attn.attention_probs(x, cos, sin)
    torch.testing.assert_close(p.sum(-1), torch.ones(2, c.n_heads, 8))
    assert torch.triu(p, diagonal=1).abs().max() == 0
    v = attn.wv(x).view(2, 8, c.n_heads, c.head_dim).transpose(1, 2)
    torch.testing.assert_close(p @ v, heads_before_gate(attn, x, cos, sin), atol=1e-5, rtol=1e-5)


def test_sink_measure_on_uniform_attention():
    """With all query/key weights zero, attention is uniform over the past, so query i puts 1/(i+1) on token 0."""
    m = make(gate="none")
    for layer in m.layers:
        torch.nn.init.zeros_(layer.attn.wq.weight)
        torch.nn.init.zeros_(layer.attn.wk.weight)
    T = 10
    rep = sink_and_gate_report(m, torch.randint(0, 50, (2, T)))
    want = sum(1 / (i + 1) for i in range(1, T)) / (T - 1)
    for got in rep["first_token_attention"]:
        assert math.isclose(got, want, rel_tol=1e-5)
    assert rep["mean_gate"] == []


def test_gate_report_for_closed_and_open_gates():
    m = make(gate="headwise")
    idx = torch.randint(0, 50, (1, 8))
    with torch.no_grad():
        m.tok_emb.weight.abs_().add_(0.1)            # positive embeddings; RMSNorm keeps the sign
        m.layers[0].attn.wg.weight.fill_(-20.0)
        assert sink_and_gate_report(m, idx)["mean_gate"][0] < 1e-3
        m.layers[0].attn.wg.weight.fill_(+20.0)
        assert sink_and_gate_report(m, idx)["mean_gate"][0] > 1 - 1e-3
