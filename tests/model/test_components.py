import math

import pytest
import torch
from torch.nn import functional as F

from wendyfm.model import ModelConfig, RMSNorm, RoPE, SwiGLU


def test_config_derives_even_head_dimension() -> None:
    config = ModelConfig(256, 32, 4, 2, 128, 64)
    assert config.head_dim == 8


@pytest.mark.parametrize(
    "changes",
    [
        {"vocab_size": 0},
        {"d_model": -1},
        {"n_heads": 0},
        {"n_layers": 0},
        {"max_seq_len": 0},
        {"intermediate_size": 0},
        {"d_model": 30},
        {"n_heads": 32},
        {"norm_eps": 0},
        {"norm_eps": math.nan},
        {"rope_theta": math.inf},
        {"rope_theta": -1},
        {"n_layers": True},
    ],
)
def test_invalid_config(changes: dict) -> None:
    values = dict(
        vocab_size=256, d_model=32, n_heads=4, n_layers=2, max_seq_len=128, intermediate_size=64
    )
    values.update(changes)
    with pytest.raises(ValueError):
        ModelConfig(**values)


def test_rmsnorm_matches_definition_and_propagates_gradients() -> None:
    layer = RMSNorm(4, eps=1e-5)
    x = torch.tensor([[[1.0, -2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]]], requires_grad=True)
    actual = layer(x)
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + layer.eps)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == x.shape
    actual.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()


def test_swiglu_matches_explicit_computation_and_propagates_gradients() -> None:
    torch.manual_seed(7)
    layer = SwiGLU(4, 7)
    x = torch.randn(2, 3, 4, requires_grad=True)
    actual = layer(x)
    expected = F.linear(
        F.silu(F.linear(x, layer.gate.weight)) * F.linear(x, layer.up.weight), layer.down.weight
    )
    torch.testing.assert_close(actual, expected)
    assert actual.shape == x.shape
    actual.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters())


def test_rope_known_angles_and_norm_preservation() -> None:
    rope = RoPE(4, theta=10000)
    x = torch.tensor([[[[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]]], requires_grad=True)
    result = rope(x, torch.tensor([0, 1]))
    expected = torch.tensor(
        [[[[1.0, 0.0, 0.0, 1.0], [math.cos(1), math.sin(1), -math.sin(0.01), math.cos(0.01)]]]]
    )
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(result.norm(dim=-1), x.norm(dim=-1))
    result.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_rope_relative_position_dot_product() -> None:
    rope = RoPE(2)
    q = torch.tensor([[[[1.0, 2.0]]]])
    k = torch.tensor([[[[3.0, 4.0]]]])
    dot_1_3 = (rope(q, torch.tensor([1])) * rope(k, torch.tensor([3]))).sum()
    dot_4_6 = (rope(q, torch.tensor([4])) * rope(k, torch.tensor([6]))).sum()
    torch.testing.assert_close(dot_1_3, dot_4_6)


@pytest.mark.parametrize(
    "constructor,args",
    [
        (RMSNorm, (0,)),
        (RMSNorm, (4, 0)),
        (SwiGLU, (0, 8)),
        (SwiGLU, (4, 0)),
        (RoPE, (3,)),
        (RoPE, (4, -1)),
    ],
)
def test_invalid_layer_settings(constructor, args) -> None:
    with pytest.raises(ValueError):
        constructor(*args)


def test_invalid_rope_input_and_positions() -> None:
    rope = RoPE(4)
    x = torch.ones(1, 2, 3, 4)
    for positions in (
        torch.tensor([-1, 0, 1]),
        torch.tensor([0, 1]),
        torch.tensor([0.0, 1.0, 2.0]),
    ):
        with pytest.raises(ValueError):
            rope(x, positions)
    with pytest.raises(ValueError):
        rope(torch.ones(2, 3, 4))
    with pytest.raises(ValueError):
        RMSNorm(4)(torch.ones(2, 3))
