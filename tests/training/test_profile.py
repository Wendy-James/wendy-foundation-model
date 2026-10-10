"""Offline checks for the fixed M5 Phase A profiling contract."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wendyfm.model import DecoderOnlyTransformer, ModelConfig
from wendyfm.training.data import NextTokenBatchSampler
from wendyfm.training.loop import TrainConfig
from wendyfm.training.profile import _preflight, _source_sha, aggregate, load_config, run

CONFIG = Path(__file__).resolve().parents[2] / "configs/m5_phase_a_fp32.json"


def test_frozen_config_rejects_expanded_work(tmp_path) -> None:
    settings = load_config(CONFIG)
    for key, value in (("repeats", 4), ("measured_steps", 31), ("batch_size", 8),
                       ("train", {**settings["train"], "min_learning_rate": 0.0003})):
        altered = {**settings, key: value}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(altered), encoding="utf-8")
        with pytest.raises(ValueError, match="baseline|optimizer"):
            load_config(path)


def test_three_repeat_aggregation_and_instability() -> None:
    repeats = [{"status": "PASS", "target_tokens_per_second": rate}
               for rate in (100.0, 110.0, 125.0)]
    result = aggregate(repeats)
    assert result["median_target_tokens_per_second"] == 110.0
    assert result["throughput_spread"] == pytest.approx(25 / 110)
    assert result["unstable"] is True
    with pytest.raises(ValueError, match="three completed"):
        aggregate(repeats[:2])
    repeats[1]["status"] = "FAIL"
    with pytest.raises(ValueError, match="three completed"):
        aggregate(repeats)


def test_reference_step_checks_aligned_targets_and_update() -> None:
    config = ModelConfig(vocab_size=32, d_model=16, n_heads=2, n_layers=1,
                         intermediate_size=32, max_seq_len=4)
    model = DecoderOnlyTransformer(config)
    sampler = NextTokenBatchSampler(torch.arange(24) % 32, seq_len=4,
                                    model_vocab_size=32, seed=7)
    _preflight(model, sampler, 2, TrainConfig(max_steps=35, learning_rate=0.003,
                                             min_learning_rate=0.003))


def test_missing_cuda_writes_bounded_failure_record(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("wendyfm.training.profile._source_sha", lambda _: "a" * 40)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "result.json"
    result = run(CONFIG, output, expected_sha="a" * 40)
    assert result == json.loads(output.read_text())
    assert result["status"] == "FAIL"
    assert result["failure_phase"] == "setup"
    assert result["error_type"] == "RuntimeError"
    assert result["repeats"] == []
    assert result["retries"] == 0
    with pytest.raises(ValueError, match="already exists"):
        run(CONFIG, output, expected_sha="a" * 40)


def test_provenance_rejects_mismatched_or_dirty_checkout(monkeypatch) -> None:
    outputs = iter((SimpleNamespace(stdout="b" * 40 + "\n"),))
    monkeypatch.setattr("wendyfm.training.profile.subprocess.run", lambda *a, **k: next(outputs))
    with pytest.raises(ValueError, match="differs"):
        _source_sha("a" * 40)
    outputs = iter((SimpleNamespace(stdout="a" * 40 + "\n"),
                    SimpleNamespace(stdout=" M src/wendyfm/training/profile.py\n")))
    with pytest.raises(ValueError, match="uncommitted"):
        _source_sha("a" * 40)
