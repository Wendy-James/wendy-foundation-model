import random

import pytest
import torch

from wendyfm.model import DecoderOnlyTransformer, ModelConfig
from wendyfm.tokenizer import BPETokenizer
from wendyfm.training import NextTokenBatchSampler, prepare_token_ids
from wendyfm.training.checkpoint import load_checkpoint, save_checkpoint
from wendyfm.training.loop import TrainConfig, learning_rate_for_step, train


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


def test_tokenized_sampler_checkpoint_resume_matches_uninterrupted_cpu(tmp_path) -> None:
    tokenizer = BPETokenizer.train(
        ["alpha beta", "beta gamma", "gamma alpha"],
        vocab_size=272,
        special_tokens=("<|eod|>",),
    )
    documents = ["alpha beta gamma", "gamma beta alpha", "beta alpha gamma"] * 3
    ids = prepare_token_ids(
        documents, tokenizer, model_vocab_size=272, end_of_document="<|eod|>"
    )
    model_config = ModelConfig(
        vocab_size=272, d_model=8, n_heads=2, n_layers=1,
        max_seq_len=4, intermediate_size=16,
    )
    config = TrainConfig(
        max_steps=6, learning_rate=0.02, min_learning_rate=0.002,
        warmup_steps=2, weight_decay=0.01,
    )

    def setup(seed: int) -> tuple[DecoderOnlyTransformer, NextTokenBatchSampler]:
        torch.manual_seed(seed)
        random.seed(seed)
        return (
            DecoderOnlyTransformer(model_config),
            NextTokenBatchSampler(ids.clone(), seq_len=4, model_vocab_size=272, seed=seed),
        )

    def recording_provider(sampler, batches):
        def sample():
            batch = sampler.sample(2)
            batches.append(tuple(tensor.clone() for tensor in batch))
            return batch
        return sample

    continuous, full_sampler = setup(55)
    full_batches = []
    full = train(continuous, recording_provider(full_sampler, full_batches), config)

    interrupted, partial_sampler = setup(55)
    partial_batches = []
    partial = train(
        interrupted, recording_provider(partial_sampler, partial_batches), config, end_step=3
    )
    path = tmp_path / "resume.pt"
    save_checkpoint(
        path, interrupted, partial.optimizer, partial.step,
        sampler=partial_sampler, config=config, batch_size=2,
    )
    payload = torch.load(path, weights_only=True)
    assert payload["format_version"] == 3
    assert partial.optimizer.param_groups[0]["lr"] == learning_rate_for_step(3, config)

    resumed, resumed_sampler = setup(999)
    optimizer = torch.optim.AdamW(resumed.parameters(), lr=config.learning_rate)
    step = load_checkpoint(
        path, resumed, optimizer, sampler=resumed_sampler, config=config, batch_size=2
    )
    assert step == 3
    assert optimizer.param_groups[0]["lr"] == partial.optimizer.param_groups[0]["lr"]
    resumed_batches = []
    final = train(
        resumed, recording_provider(resumed_sampler, resumed_batches), config,
        optimizer=optimizer, start_step=step,
    )

    assert len(full_batches) == 6
    assert len(partial_batches) == len(resumed_batches) == 3
    for actual, expected in zip(partial_batches + resumed_batches, full_batches, strict=True):
        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
        torch.testing.assert_close(actual[0][:, 1:], actual[1][:, :-1], rtol=0, atol=0)
    assert final.step == full.step == 6
    assert final.final_loss == full.final_loss
    assert optimizer.param_groups[0]["lr"] == full.optimizer.param_groups[0]["lr"]
    _assert_same_parameters(continuous, resumed)
    for original, restored in zip(
        full.optimizer.state_dict()["state"].values(),
        optimizer.state_dict()["state"].values(), strict=True,
    ):
        for key in original:
            torch.testing.assert_close(original[key], restored[key], rtol=0, atol=0)


def test_checkpoint_sampler_version_and_corruption_rejected(tmp_path) -> None:
    ids = torch.arange(20, dtype=torch.long)
    sampler = NextTokenBatchSampler(ids, seq_len=4, model_vocab_size=20, seed=1)
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "state.pt"
    save_checkpoint(path, model, optimizer, 0)
    assert torch.load(path, weights_only=True)["format_version"] == 1
    with pytest.raises(ValueError, match="sampler state"):
        load_checkpoint(path, model, optimizer, sampler=sampler)
    config = TrainConfig(max_steps=1)
    with pytest.raises(ValueError, match="sampler, config, and batch_size"):
        save_checkpoint(path, model, optimizer, 0, sampler=sampler)
    optimizer.param_groups[0]["lr"] = config.learning_rate
    save_checkpoint(path, model, optimizer, 0, sampler=sampler, config=config, batch_size=2)
    with pytest.raises(ValueError, match="sampler state"):
        load_checkpoint(path, model, optimizer)
    payload = torch.load(path, weights_only=True)
    for damaged in (
        {**payload, "sampler": {**payload["sampler"], "token_sha256": "bad"}},
        {**payload, "sampler": {**payload["sampler"],
                                "generator_state": torch.tensor([1], dtype=torch.uint8)}},
        {**payload, "sampler": None},
    ):
        torch.save(damaged, path)
        with pytest.raises(ValueError, match="checkpoint"):
            load_checkpoint(path, model, optimizer, sampler=sampler, config=config, batch_size=2)

    legacy = {key: value for key, value in payload.items() if key != "exact_metadata"}
    legacy["format_version"] = 2
    torch.save(legacy, path)
    assert load_checkpoint(path, model, optimizer, sampler=sampler) == 0


def test_exact_checkpoint_rejects_changed_experiment_before_restoring(tmp_path) -> None:
    config = TrainConfig(max_steps=4, learning_rate=0.01)
    model = _model()
    sampler = NextTokenBatchSampler(torch.arange(20), seq_len=4, model_vocab_size=20, seed=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    path = tmp_path / "exact.pt"
    save_checkpoint(path, model, optimizer, 0, sampler=sampler, config=config, batch_size=2)

    restored = _model()
    fresh_optimizer = torch.optim.AdamW(restored.parameters(), lr=config.learning_rate)
    before = [parameter.detach().clone() for parameter in restored.parameters()]
    cases = (
        (sampler, TrainConfig(max_steps=4, learning_rate=0.02), 2, "config"),
        (sampler, config, 3, "batch_size"),
        (NextTokenBatchSampler(torch.arange(20).flip(0), seq_len=4,
                               model_vocab_size=20, seed=1), config, 2, "sampler"),
    )
    for candidate, candidate_config, size, message in cases:
        with pytest.raises(ValueError, match=message):
            load_checkpoint(
                path, restored, fresh_optimizer, sampler=candidate,
                config=candidate_config, batch_size=size,
            )
        assert all(torch.equal(old, new) for old, new in zip(before, restored.parameters()))

    payload = torch.load(path, weights_only=True)
    payload["optimizer"]["param_groups"][0]["lr"] = 0.2
    torch.save(payload, path)
    with pytest.raises(ValueError, match="checkpoint"):
        load_checkpoint(
            path, restored, fresh_optimizer, sampler=sampler, config=config, batch_size=2
        )


def test_exact_checkpoint_rejects_changed_model_config(tmp_path) -> None:
    config = TrainConfig(max_steps=2)
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    sampler = NextTokenBatchSampler(torch.arange(20), seq_len=4, model_vocab_size=20, seed=1)
    path = tmp_path / "exact.pt"
    save_checkpoint(path, model, optimizer, 0, sampler=sampler, config=config, batch_size=2)
    changed = DecoderOnlyTransformer(
        ModelConfig(vocab_size=13, d_model=8, n_heads=2, n_layers=1,
                    max_seq_len=5, intermediate_size=16)
    )
    changed_optimizer = torch.optim.AdamW(changed.parameters(), lr=config.learning_rate)
    with pytest.raises(ValueError, match="config"):
        load_checkpoint(
            path, changed, changed_optimizer, sampler=sampler, config=config, batch_size=2
        )


def test_exact_checkpoint_restores_global_and_sampler_rng(tmp_path) -> None:
    torch.manual_seed(123)
    random.seed(123)
    model = _model()
    config = TrainConfig(max_steps=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    sampler = NextTokenBatchSampler(torch.arange(20), seq_len=4, model_vocab_size=20, seed=7)
    sampler.sample(2)
    path = tmp_path / "exact.pt"
    save_checkpoint(path, model, optimizer, 0, sampler=sampler, config=config, batch_size=2)
    expected_torch = torch.rand(3)
    expected_python = random.random()
    expected_batch = sampler.sample(2)

    restored = _model()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=config.learning_rate)
    restored_sampler = NextTokenBatchSampler(
        torch.arange(20), seq_len=4, model_vocab_size=20, seed=999
    )
    assert load_checkpoint(
        path, restored, restored_optimizer, sampler=restored_sampler,
        config=config, batch_size=2,
    ) == 0
    torch.testing.assert_close(torch.rand(3), expected_torch, rtol=0, atol=0)
    assert random.random() == expected_python
    for actual, expected in zip(restored_sampler.sample(2), expected_batch, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_failed_checkpoint_write_is_atomic(tmp_path, monkeypatch) -> None:
    path = tmp_path / "state.pt"
    path.write_bytes(b"previous complete checkpoint")
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters())

    def failed_save(*args, **kwargs):
        raise OSError("disk write failed")

    monkeypatch.setattr(torch, "save", failed_save)
    with pytest.raises(OSError, match="disk write failed"):
        save_checkpoint(path, model, optimizer, 0)
    assert path.read_bytes() == b"previous complete checkpoint"
    assert list(tmp_path.iterdir()) == [path]


def test_failed_checkpoint_replace_keeps_previous_file(tmp_path, monkeypatch) -> None:
    path = tmp_path / "state.pt"
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters())
    save_checkpoint(path, model, optimizer, 0)
    previous = path.read_bytes()

    def failed_replace(*args):
        raise OSError("replace failed")

    monkeypatch.setattr("wendyfm.training.checkpoint.os.replace", failed_replace)
    with pytest.raises(OSError, match="replace failed"):
        save_checkpoint(path, model, optimizer, 0)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_truncated_checkpoint_and_malformed_fingerprint_are_errors(tmp_path) -> None:
    path = tmp_path / "state.pt"
    model = _model()
    sampler = NextTokenBatchSampler(torch.arange(20), seq_len=4, model_vocab_size=20, seed=1)
    config = TrainConfig(max_steps=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    save_checkpoint(path, model, optimizer, 0, sampler=sampler, config=config, batch_size=2)
    complete = path.read_bytes()
    path.write_bytes(complete[: len(complete) // 2])
    with pytest.raises(ValueError, match="invalid checkpoint file"):
        load_checkpoint(path, model, optimizer, sampler=sampler, config=config, batch_size=2)

    path.write_bytes(complete)
    payload = torch.load(path, weights_only=True)
    payload["exact_metadata"]["batch_size"] = torch.tensor([2, 3])
    torch.save(payload, path)
    with pytest.raises(ValueError, match="fingerprint"):
        load_checkpoint(path, model, optimizer, sampler=sampler, config=config, batch_size=2)
