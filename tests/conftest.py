import pytest
import torch

from gated_attention import GatedConfig, GatedLM


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def cfg(**kw):
    base = dict(vocab_size=50, d_model=64, n_layers=3, n_heads=4, max_seq_len=64)
    base.update(kw)
    return GatedConfig(**base)


def make(**kw):
    return GatedLM(cfg(**kw)).eval()


@pytest.fixture(params=["none", "headwise", "elementwise"])
def model(request):
    return make(gate=request.param)
