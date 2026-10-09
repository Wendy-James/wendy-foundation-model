import math

import pytest
import torch
from torch import nn

from wendyfm.model import CausalSelfAttention, ModelConfig


def _config() -> ModelConfig:
    return ModelConfig(31, 8, 2, 1, 12, 16)


def test_attention_matches_per_head_causal_reference() -> None:
    torch.manual_seed(3)
    attention = CausalSelfAttention(_config())
    x = torch.randn(2, 5, 8)
    actual = attention(x)

    # Compute each query using only the keys at or before its position.
    q_flat = attention.q_proj(x)
    k_flat = attention.k_proj(x)
    v_flat = attention.v_proj(x)
    expected_batches = []
    for b in range(x.shape[0]):
        head_values = []
        for h in range(attention.n_heads):
            coordinates = slice(h * attention.head_dim, (h + 1) * attention.head_dim)
            q = attention.rope(q_flat[b, :, coordinates][None, None])[0, 0]
            k = attention.rope(k_flat[b, :, coordinates][None, None])[0, 0]
            head_values.append((q, k, v_flat[b, :, coordinates]))
        expected_tokens = []
        for position in range(x.shape[1]):
            heads = []
            for q, k, v in head_values:
                scores = k[: position + 1] @ q[position]
                weights = torch.softmax(scores / math.sqrt(attention.head_dim), dim=0)
                heads.append(weights @ v[: position + 1])
            expected_tokens.append(attention.out_proj(torch.cat(heads)))
        expected_batches.append(torch.stack(expected_tokens))
    expected = torch.stack(expected_batches)

    assert actual.shape == (2, 5, 8)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("prefix", [1, 3, 5])
def test_future_tokens_cannot_change_attention_outputs_or_logits(prefix: int) -> None:
    torch.manual_seed(11)
    attention = CausalSelfAttention(_config())
    logits_head = nn.Linear(8, 31, bias=False)
    x = torch.randn(2, 6, 8)
    changed = x.clone()
    changed[:, prefix:] = torch.randn_like(changed[:, prefix:]) * 100

    original_output = attention(x)
    changed_output = attention(changed)
    torch.testing.assert_close(
        original_output[:, :prefix], changed_output[:, :prefix], atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        logits_head(original_output)[:, :prefix],
        logits_head(changed_output)[:, :prefix],
        atol=1e-6,
        rtol=1e-6,
    )


def test_prefix_loss_has_no_future_input_gradient_and_all_projections_train() -> None:
    torch.manual_seed(19)
    attention = CausalSelfAttention(_config())
    x = torch.randn(2, 6, 8, requires_grad=True)
    output = attention(x)
    output[:, :3].square().sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad[:, :3].abs().sum() > 0
    torch.testing.assert_close(x.grad[:, 3:], torch.zeros_like(x.grad[:, 3:]))

    attention.zero_grad(set_to_none=True)
    x.grad = None
    attention(x).square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for projection in (attention.q_proj, attention.k_proj, attention.v_proj, attention.out_proj):
        grad = projection.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


@pytest.mark.parametrize("shape", [(2, 8), (2, 3, 7), (2, 0, 8), (2, 13, 8)])
def test_attention_rejects_invalid_input_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        CausalSelfAttention(_config())(torch.randn(shape))
