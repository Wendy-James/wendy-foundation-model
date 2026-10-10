"""CPU profile contract: bounded work, real timing, and honest failures."""

import json
import math

import pytest
from scripts import profile_cpu


def test_profile_writes_measured_cpu_result_with_warmup(tmp_path, monkeypatch) -> None:
    actual_train = profile_cpu.train
    calls = []

    def record_train(*args, **kwargs):
        result = actual_train(*args, **kwargs)
        calls.append((kwargs.get("start_step", 0), kwargs["end_step"], result.optimizer))
        return result

    monkeypatch.setattr(profile_cpu, "train", record_train)
    output = tmp_path / "profile.json"
    result = profile_cpu.run(output, warmup_steps=2, measured_steps=2, batch_size=2, seq_len=4)
    assert [(start, end) for start, end, _ in calls] == [(0, 2), (2, 4)]
    assert calls[0][2] is calls[1][2]
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert result["device"] == "cpu"
    assert result["timed_target_tokens"] == 2 * 2 * 4
    assert result["target_token_definition"] == (
        "one shifted target ID per batch and sequence position"
    )
    assert result["timed_wall_time_seconds"] > 0
    assert result["target_tokens_per_second"] == pytest.approx(
        result["timed_target_tokens"] / result["timed_wall_time_seconds"]
    )
    assert result["timed_wall_time_seconds"] == pytest.approx(
        result["mean_step_seconds"] * result["measured_steps"]
    )
    for key in ("warmup_loss_last_batch", "timed_loss_first_batch", "timed_loss_last_batch"):
        assert math.isfinite(result[key])
    assert not any("cuda" in key or "gpu" in key for key in result)


@pytest.mark.parametrize(
    "settings",
    [
        {"warmup_steps": 0},
        {"measured_steps": 21},
        {"batch_size": 5},
        {"seq_len": 33},
        {"seed": -1},
        {"measured_steps": True},
    ],
)
def test_profile_rejects_unbounded_or_invalid_settings_before_writing(tmp_path, settings) -> None:
    output = tmp_path / "profile.json"
    with pytest.raises(ValueError):
        profile_cpu.run(output, **settings)
    assert not output.exists()


def test_profile_propagates_nonfinite_loss_without_writing(tmp_path, monkeypatch) -> None:
    original = profile_cpu.DecoderOnlyTransformer.next_token_loss

    def nonfinite(self, inputs, targets):
        return original(self, inputs, targets) * float("nan")

    monkeypatch.setattr(profile_cpu.DecoderOnlyTransformer, "next_token_loss", nonfinite)
    output = tmp_path / "profile.json"
    with pytest.raises(FloatingPointError, match="nonfinite loss"):
        profile_cpu.run(output, measured_steps=1)
    assert not output.exists()
