import random

import pytest
import torch

from wendyfm.model import DecoderOnlyTransformer, ModelConfig
from wendyfm.training.checkpoint import load_checkpoint, save_checkpoint
from wendyfm.training.loop import TrainConfig, train


def _model() -> DecoderOnlyTransformer:
    return DecoderOnlyTransformer(
        ModelConfig(
            vocab_size=13, d_model=8, n_heads=2, n_layers=1, max_seq_len=4,
            intermediate_size=16,
        )
    )


def _random_batch() -> tuple[torch.Tensor, torch.Tensor]:
    start = random.randrange(13)
    inputs = torch.randint(0, 13, (2, 4), dtype=torch.long)
    targets = (inputs + start + 1) % 13
    return inputs, targets


def _assert_same_parameters(left: DecoderOnlyTransformer, right: DecoderOnlyTransformer) -> None:
    for first, second in zip(left.parameters(), right.parameters(), strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_checkpoint_round_trip_restores_step_model_optimizer_and_rng(tmp_path) -> None:
    torch.manual_seed(21)
    random.seed(21)
    model = _model()
    config = TrainConfig(max_steps=4, learning_rate=0.01)
    result = train(model, _random_batch, config, end_step=2)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, result.optimizer, result.step)
    expected_torch = torch.rand(3)
    expected_python = random.random()

    restored = _model()
    optimizer = torch.optim.AdamW(restored.parameters(), lr=config.learning_rate)
    step = load_checkpoint(path, restored, optimizer)
    assert step == 2
    _assert_same_parameters(model, restored)
    assert optimizer.state_dict()["param_groups"] == result.optimizer.state_dict()["param_groups"]
    for state, restored_state in zip(
        result.optimizer.state_dict()["state"].values(),
        optimizer.state_dict()["state"].values(),
        strict=True,
    ):
        for key in state:
            torch.testing.assert_close(state[key], restored_state[key], rtol=0, atol=0)
    torch.testing.assert_close(torch.rand(3), expected_torch, rtol=0, atol=0)
    assert random.random() == expected_python
    assert sorted(tmp_path.iterdir()) == [path]


def test_interrupted_and_resumed_training_matches_uninterrupted(tmp_path) -> None:
    config = TrainConfig(max_steps=6, learning_rate=0.02, min_learning_rate=0.002,
                         warmup_steps=2, weight_decay=0)
    torch.manual_seed(55)
    random.seed(55)
    continuous = _model()
    full = train(continuous, _random_batch, config)

    torch.manual_seed(55)
    random.seed(55)
    interrupted = _model()
    partial = train(interrupted, _random_batch, config, end_step=3)
    path = tmp_path / "resume.pt"
    save_checkpoint(path, interrupted, partial.optimizer, partial.step)
    resumed = _model()
    optimizer = torch.optim.AdamW(resumed.parameters(), lr=config.learning_rate)
    step = load_checkpoint(path, resumed, optimizer)
    final = train(resumed, _random_batch, config, optimizer=optimizer, start_step=step)
    assert final.step == full.step
    assert final.final_loss == pytest.approx(full.final_loss, abs=0, rel=0)
    _assert_same_parameters(continuous, resumed)


@pytest.mark.parametrize("payload", [None, {}, {"format_version": 999}, {"step": -1}])
def test_malformed_checkpoint_is_rejected(tmp_path, payload) -> None:
    path = tmp_path / "bad.pt"
    torch.save(payload, path)
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="checkpoint"):
        load_checkpoint(path, model, optimizer)


def test_corrupt_checkpoint_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.pt"
    path.write_bytes(b"not a torch checkpoint")
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="checkpoint"):
        load_checkpoint(path, model, optimizer)
