import math

import pytest
import torch
from torch.nn import functional as F

from wendyfm.model import DecoderOnlyTransformer, ModelConfig
from wendyfm.training.loop import TrainConfig, learning_rate_for_step, train


def _model() -> DecoderOnlyTransformer:
    return DecoderOnlyTransformer(
        ModelConfig(
            vocab_size=13, d_model=8, n_heads=2, n_layers=1, max_seq_len=4,
            intermediate_size=16,
        )
    )


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
    targets = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]], dtype=torch.long)
    return inputs, targets


def test_train_updates_parameters_and_reduces_fixed_batch_loss() -> None:
    torch.manual_seed(17)
    model = _model()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    config = TrainConfig(max_steps=24, learning_rate=0.025, weight_decay=0)
    result = train(model, _batch, config)
    assert result.step == 24
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert result.final_loss < result.initial_loss / 4
    assert result.mean_step_seconds > 0
    assert math.isfinite(result.tokens_per_second) and result.tokens_per_second > 0
    assert result.peak_cuda_memory_bytes is None
    print(
        f"CPU tiny batch: initial={result.initial_loss:.6f}, final={result.final_loss:.6f}, "
        f"step_ms={result.mean_step_seconds * 1000:.3f}, "
        f"tokens_per_sec={result.tokens_per_second:.1f}"
    )


def test_targets_are_scored_at_matching_logit_positions() -> None:
    torch.manual_seed(3)
    model = _model()
    inputs, targets = _batch()
    expected = F.cross_entropy(model(inputs).reshape(-1, 13), targets.reshape(-1))
    result = train(model, _batch, TrainConfig(max_steps=1, learning_rate=0.01))
    assert result.initial_loss == pytest.approx(expected.item())


def test_schedule_uses_one_based_global_update_steps() -> None:
    config = TrainConfig(max_steps=6, learning_rate=0.1, min_learning_rate=0.01, warmup_steps=2)
    rates = [learning_rate_for_step(step, config) for step in range(1, 7)]
    assert rates[0] == pytest.approx(0.05)
    assert rates[1] == pytest.approx(0.1)
    assert rates[2] == pytest.approx(0.1)
    assert rates[-1] == pytest.approx(0.01)
    assert rates[2:] == sorted(rates[2:], reverse=True)


def test_metrics_count_targets_and_elapsed_training_time(monkeypatch) -> None:
    times = iter((10.0, 14.0))
    monkeypatch.setattr("wendyfm.training.loop.time.perf_counter", lambda: next(times))
    result = train(_model(), _batch, TrainConfig(max_steps=1))
    assert result.initial_loss == result.final_loss
    assert result.mean_step_seconds == 4.0
    assert result.tokens_per_second == 2.0  # 2 sequences x 4 targets / 4 seconds


def test_rejects_invalid_batch_and_nonfinite_loss() -> None:
    model = _model()
    with pytest.raises(ValueError, match="batch provider"):
        train(model, lambda: (torch.ones(1, 2), torch.ones(1, 2)), TrainConfig(1))
    original = model.next_token_loss
    model.next_token_loss = lambda inputs, targets: original(inputs, targets) * float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite loss"):
        train(model, _batch, TrainConfig(1))


def test_rejects_nonfinite_gradient_before_optimizer_update() -> None:
    model = _model()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    handle = next(model.parameters()).register_hook(lambda grad: grad * float("nan"))
    try:
        with pytest.raises(FloatingPointError, match="nonfinite gradient"):
            train(model, _batch, TrainConfig(1))
    finally:
        handle.remove()
    assert all(torch.equal(old, new) for old, new in zip(before, model.parameters()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_loop_transfers_aligned_cpu_batches_and_measures_memory() -> None:
    model = _model().cuda()
    inputs, targets = _batch()
    expected = F.cross_entropy(
        model(inputs.cuda()).reshape(-1, 13), targets.cuda().reshape(-1)
    )
    result = train(model, _batch, TrainConfig(max_steps=1, learning_rate=0.01))
    assert result.initial_loss == pytest.approx(float(expected))
    assert result.peak_cuda_memory_bytes > 0
    assert result.tokens_per_second > 0
