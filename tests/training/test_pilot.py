"""End-to-end regression coverage for the bounded CPU pilot."""

import json
from pathlib import Path

import pytest
import torch

from wendyfm.training.pilot import run

CONFIG = Path(__file__).resolve().parents[2] / "configs/cpu_pilot.json"


def _assert_equal_tree(left, right) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _assert_equal_tree(first, second)
    else:
        assert left == right


def test_pilot_saves_metrics_and_exact_resume(tmp_path) -> None:
    full_dir = tmp_path / "full"
    split_dir = tmp_path / "split"
    full = run(CONFIG, full_dir)
    partial = run(CONFIG, split_dir, end_step=2)
    resumed = run(CONFIG, split_dir, resume=True)

    assert full["device"] == "cpu"
    assert full["optimizer_steps"] == resumed["optimizer_steps"] == 4
    assert partial["optimizer_steps"] == resumed["start_step"] == 2
    assert full["train_documents"] > 0 and full["validation_documents"] > 0
    assert full["validation_windows"] > 0
    assert full["train_text_sha256"] != full["validation_text_sha256"]
    assert full["train_loss_first_batch"] > 0
    assert full["validation_loss"] > 0
    assert full["tokens_per_second"] > 0
    assert full["wall_time_seconds"] > 0
    assert full["training_wall_time_seconds"] > 0
    assert json.loads((full_dir / "manifest.json").read_text()) == full
    assert (full_dir / "tokenizer.json").read_bytes() == (split_dir / "tokenizer.json").read_bytes()
    assert full["validation_loss"] == resumed["validation_loss"]
    assert full["train_loss_last_batch"] == resumed["train_loss_last_batch"]

    uninterrupted = torch.load(full_dir / "checkpoint.pt", weights_only=True)
    continued = torch.load(split_dir / "checkpoint.pt", weights_only=True)
    assert uninterrupted["format_version"] == continued["format_version"] == 3
    for key in ("model", "optimizer", "step", "sampler", "exact_metadata"):
        _assert_equal_tree(uninterrupted[key], continued[key])


def test_pilot_rejects_overlapping_documents(tmp_path) -> None:
    text = tmp_path / "same.txt"
    text.write_text("shared document\n", encoding="utf-8")
    settings = json.loads(CONFIG.read_text())
    settings["train_text"] = str(text)
    settings["validation_text"] = str(tmp_path / "other.txt")
    (tmp_path / "other.txt").write_text("shared document\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(ValueError, match="disjoint"):
        run(config, tmp_path / "output")
