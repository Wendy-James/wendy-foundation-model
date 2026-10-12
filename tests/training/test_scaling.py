"""Offline contract checks for the three fixed Phase C1 model cells."""

import json
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from wendyfm.model import DecoderOnlyTransformer, ModelConfig
from wendyfm.training.data import NextTokenBatchSampler
from wendyfm.training.profile import aggregate
from wendyfm.training.scaling import MODELS, load_sweep_config, parameter_counts, run_cell

CONFIG = Path(__file__).resolve().parents[2] / "configs/m5_phase_c1_fp32.json"


def test_frozen_models_and_capacity() -> None:
    settings = load_sweep_config(CONFIG)
    assert settings["models"] == list(MODELS)
    assert [(m["d_model"], m["n_heads"], m["n_layers"], m["intermediate_size"])
            for m in MODELS] == [(32, 4, 2, 64), (64, 4, 4, 128), (128, 4, 6, 256)]
    assert all(m["max_seq_len"] == 16 and m["vocab_size"] == 272 for m in MODELS)
    counts = parameter_counts()
    assert counts == sorted(set(counts))
    for model, count in zip(MODELS, counts, strict=True):
        instance = DecoderOnlyTransformer(ModelConfig(**model))
        assert count == sum(p.numel() for p in instance.parameters() if p.requires_grad)


@pytest.mark.parametrize("change", [
    {"batch_size": 8}, {"measured_steps": 31}, {"repeats": 4},
    {"models": [MODELS[0], MODELS[2], MODELS[1]]},
    {"models": [MODELS[0], MODELS[1], {**MODELS[2], "d_model": 256}]},
    {"train": {"max_steps": 35, "learning_rate": 0.003, "min_learning_rate": 0.003,
               "warmup_steps": 0, "weight_decay": 0.0}},
    {"train_text": "../data/fixtures/cpu_pilot_validation.txt"},
])
def test_rejects_expanded_or_changed_protocol(tmp_path, change) -> None:
    settings = {**load_sweep_config(CONFIG), **change}
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(ValueError, match="baseline|allowlist|optimizer|corpus"):
        load_sweep_config(path)


def test_repeat_arithmetic_and_signed_finite_aggregation() -> None:
    repeats = [{"status": "PASS", "target_tokens_per_second": 1920 / seconds}
               for seconds in (2.0, 2.04, 2.08)]
    aggregate_result = aggregate(repeats)
    assert aggregate_result["median_target_tokens_per_second"] == pytest.approx(1920 / 2.04)
    assert aggregate_result["unstable"] is False
    for bad in (0, -1, math.nan, math.inf):
        repeats[0]["target_tokens_per_second"] = bad
        with pytest.raises(ValueError, match="finite and positive"):
            aggregate(repeats)


def test_c1_sampler_targets_are_scored_once() -> None:
    config = ModelConfig(**MODELS[1])
    sampler = NextTokenBatchSampler(torch.arange(80), seq_len=16,
                                    model_vocab_size=272, seed=20261009)
    inputs, targets = sampler.sample(4)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])
    model = DecoderOnlyTransformer(config)
    explicit = F.cross_entropy(model(inputs).reshape(-1, config.vocab_size),
                               targets.reshape(-1))
    torch.testing.assert_close(model.next_token_loss(inputs, targets), explicit)


def test_one_cell_failure_record_and_bounds(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("wendyfm.training.profile._source_sha", lambda _: "a" * 40)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "cell.json"
    result = run_cell(CONFIG, output, expected_sha="a" * 40, cell=2)
    assert result == json.loads(output.read_text())
    assert result["status"] == "FAIL"
    assert result["model"] == MODELS[1]
    assert result["cell"] == 2
    assert result["error_type"] == "RuntimeError"
    assert result["retries"] == 0
    assert result["repeats"] == []
    assert result["config_sha256"] != result["train_text_sha256"]
    with pytest.raises(ValueError, match="cell must"):
        run_cell(CONFIG, tmp_path / "bad.json", expected_sha="a" * 40, cell=4)
    with pytest.raises(ValueError, match="already exists"):
        run_cell(CONFIG, output, expected_sha="a" * 40, cell=2)
