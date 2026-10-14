import pytest

torch = pytest.importorskip("torch")

from wendyfm.model import ModelConfig, TinyDecoderLM


def tiny_model():
    torch.manual_seed(7)
    return TinyDecoderLM(ModelConfig(vocab_size=24, d_model=32, n_heads=4, n_layers=2, max_seq_len=12))


def test_shapes_and_loss():
    model = tiny_model()
    tokens = torch.randint(0, 24, (2, 6))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, 6, 24)
    assert loss is not None and loss.isfinite()
    assert model.embedding.weight is model.lm_head.weight


def test_future_tokens_do_not_change_past_logits():
    model = tiny_model().eval()
    a = torch.tensor([[1, 2, 3, 4, 5, 6]])
    b = torch.tensor([[1, 2, 3, 19, 20, 21]])
    with torch.no_grad():
        y_a, _ = model(a)
        y_b, _ = model(b)
    torch.testing.assert_close(y_a[:, :3], y_b[:, :3], atol=1e-5, rtol=1e-5)


def test_gradients():
    model = tiny_model()
    tokens = torch.randint(0, 24, (2, 6))
    _, loss = model(tokens[:, :-1], tokens[:, 1:])
    loss.backward()
    assert model.blocks[0].attn.qkv.weight.grad is not None
    assert model.blocks[0].ffn.gate.weight.grad is not None
    assert torch.isfinite(model.blocks[0].attn.qkv.weight.grad).all()


def test_invalid_config_and_inputs():
    with pytest.raises(ValueError):
        ModelConfig(d_model=30, n_heads=4)
    with pytest.raises(ValueError):
        ModelConfig(d_model=45, n_heads=9)
    model = tiny_model()
    with pytest.raises(ValueError):
        model(torch.randint(0, 24, (1, 13)))
    with pytest.raises(ValueError):
        model(torch.tensor([[-1, 2, 3]]))
    with pytest.raises(ValueError):
        model(torch.randn(1, 4))


def test_overfit_fixed_sequence():
    torch.set_num_threads(1)
    model = tiny_model().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7]]).repeat(2, 1)
    with torch.no_grad():
        _, start = model(x[:, :-1], x[:, 1:])
    for _ in range(65):
        optimizer.zero_grad()
        _, loss = model(x[:, :-1], x[:, 1:])
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        _, finish = model(x[:, :-1], x[:, 1:])
    assert finish.item() < start.item() * 0.25
