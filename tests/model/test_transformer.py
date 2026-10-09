import pytest
import torch
from torch.nn import functional as F

from wendyfm.model import DecoderBlock, DecoderOnlyTransformer, ModelConfig
from wendyfm.tokenizer import BPETokenizer


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=17, d_model=16, n_heads=2, n_layers=2, max_seq_len=8, intermediate_size=32
    )


def test_decoder_shapes_and_next_token_loss_matches_explicit_shift() -> None:
    torch.manual_seed(5)
    config = _config()
    model = DecoderOnlyTransformer(config)
    assert len(model.blocks) == config.n_layers
    assert all(isinstance(block, DecoderBlock) for block in model.blocks)

    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    logits = model(ids)
    assert logits.shape == (2, 4, config.vocab_size)
    assert torch.isfinite(logits).all()
    expected = F.cross_entropy(
        logits[:, :-1].reshape(-1, config.vocab_size), ids[:, 1:].reshape(-1)
    )
    torch.testing.assert_close(model.next_token_loss(ids), expected)
    explicit_targets = torch.tensor([[2, 3, 4, 9], [6, 7, 8, 10]])
    explicit_loss = F.cross_entropy(
        logits.reshape(-1, config.vocab_size), explicit_targets.reshape(-1)
    )
    torch.testing.assert_close(model.next_token_loss(ids, explicit_targets), explicit_loss)
    assert model(torch.tensor([[1]])).shape == (1, 1, config.vocab_size)


def test_decoder_block_matches_pre_norm_residual_equations() -> None:
    torch.manual_seed(13)
    block = DecoderBlock(_config())
    x = torch.randn(2, 4, 16)
    after_attention = x + block.attn(block.attn_norm(x))
    expected = after_attention + block.ffn(block.ffn_norm(after_attention))
    torch.testing.assert_close(block(x), expected)


def test_model_initialization_repeats_with_seed_and_norms_start_at_one() -> None:
    torch.manual_seed(42)
    first = DecoderOnlyTransformer(_config())
    torch.manual_seed(42)
    second = DecoderOnlyTransformer(_config())
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(first.final_norm.weight, torch.ones(16))
    for block in first.blocks:
        torch.testing.assert_close(block.attn_norm.weight, torch.ones(16))
        torch.testing.assert_close(block.ffn_norm.weight, torch.ones(16))
    assert first.lm_head.bias is None


def test_full_model_is_causal_and_all_parameters_receive_gradients() -> None:
    torch.manual_seed(7)
    model = DecoderOnlyTransformer(_config())
    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    changed = ids.clone()
    changed[:, 3:] = torch.tensor([[11, 12], [13, 14]])
    torch.testing.assert_close(model(ids)[:, :3], model(changed)[:, :3])

    loss = model.next_token_loss(ids)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


@pytest.mark.parametrize(
    "ids",
    [
        torch.ones(3, dtype=torch.long),
        torch.ones(0, 2, dtype=torch.long),
        torch.ones(1, 0, dtype=torch.long),
        torch.ones(1, 9, dtype=torch.long),
        torch.ones(1, 2, dtype=torch.float32),
        torch.tensor([[-1, 1]]),
        torch.tensor([[1, 17]]),
    ],
)
def test_rejects_invalid_token_inputs(ids: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        DecoderOnlyTransformer(_config())(ids)


def test_next_token_loss_requires_two_tokens() -> None:
    with pytest.raises(ValueError, match="at least two"):
        DecoderOnlyTransformer(_config()).next_token_loss(torch.tensor([[1]]))


def test_explicit_targets_are_not_shifted_and_are_validated() -> None:
    model = DecoderOnlyTransformer(_config())
    ids = torch.tensor([[1, 2, 3]])
    targets = torch.tensor([[4, 5, 6]])
    logits = model(ids)
    expected = F.cross_entropy(logits.reshape(-1, 17), targets.reshape(-1))
    torch.testing.assert_close(model.next_token_loss(ids, targets), expected)
    for invalid in (torch.tensor([[4, 5]]), torch.tensor([[4.0, 5.0, 6.0]]),
                    torch.tensor([[4, 5, 17]])):
        with pytest.raises(ValueError):
            model.next_token_loss(ids, invalid)


def test_model_vocab_capacity_covers_reserved_tokenizer_ids() -> None:
    tokenizer = BPETokenizer.train(["a"], vocab_size=272, special_tokens=("<|eos|>",))
    ids = tokenizer.encode("a<|eos|>", allowed_special=("<|eos|>",))
    assert tokenizer.vocab_size < tokenizer.spec.config.vocab_size
    assert ids[-1] == 271
    config = ModelConfig(
        vocab_size=tokenizer.spec.config.vocab_size,
        d_model=16,
        n_heads=2,
        n_layers=1,
        max_seq_len=8,
        intermediate_size=32,
    )
    assert DecoderOnlyTransformer(config)(torch.tensor([ids])).shape == (1, 2, 272)
    too_small = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        n_heads=2,
        n_layers=1,
        max_seq_len=8,
        intermediate_size=32,
    )
    with pytest.raises(ValueError, match="configured vocabulary"):
        DecoderOnlyTransformer(too_small)(torch.tensor([ids]))


def test_tiny_batch_overfits_on_cpu() -> None:
    torch.manual_seed(123)
    model = DecoderOnlyTransformer(_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0)
    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    initial_loss = model.next_token_loss(ids).item()
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = model.next_token_loss(ids)
        loss.backward()
        optimizer.step()
    final_loss = model.next_token_loss(ids).item()
    print(f"CPU tiny overfit: initial_loss={initial_loss:.6f}, "
          f"final_loss={final_loss:.6f}, steps=100")
    assert final_loss < 0.05
    assert final_loss < initial_loss / 10
