"""Offline M6 acceptance for provenance, loss, metrics, and exact CPU resume."""

import json

import pytest
import torch

from wendyfm.model import DecoderOnlyTransformer, ModelConfig
from wendyfm.training.real_data import prepare_documents
from wendyfm.training.real_pilot import parameter_update_witness, run


def _fixture(tmp_path):
    rows = [{"id": f"doc-{i}", "text": f"A science sentence about stars number {i}. " * 5}
            for i in range(60)]
    data_dir = tmp_path / "raw"
    manifest = tmp_path / "source.json"
    prepare_documents(rows, data_dir, manifest, max_documents=60, max_text_bytes=20_000,
                      vocab_size=272)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "cpu_threads": 1, "microbatch_size": 2, "gradient_accumulation_steps": 2,
        "metric_warmup_steps": 1, "seed": 7,
        "model": {"vocab_size": 272, "d_model": 16, "n_heads": 2,
                  "n_layers": 1, "max_seq_len": 16, "intermediate_size": 32,
                  "attention_backend": "sdpa"},
        "train": {"max_steps": 3, "learning_rate": 0.001},
    }))
    return config, data_dir, manifest


def test_real_pilot_metrics_provenance_and_exact_resume(tmp_path) -> None:
    config, data_dir, manifest = _fixture(tmp_path)
    full = run(config, data_dir, manifest, tmp_path / "full")
    partial = run(config, data_dir, manifest, tmp_path / "split", end_step=1)
    resumed = run(config, data_dir, manifest, tmp_path / "split", resume=True)
    assert full["trainable_parameters"] == sum(
        p.numel() for p in DecoderOnlyTransformer(ModelConfig(
            272, 16, 2, 1, 16, 32, attention_backend="sdpa")).parameters())
    assert full["effective_batch_size"] == 4
    assert full["timed_target_tokens"] == 2 * 4 * 16
    assert full["tokens_per_second"] == pytest.approx(
        full["timed_target_tokens"] / (full["mean_step_seconds"] * full["measured_steps"]))
    assert full["validation_loss"] > 0
    assert full["parameters_updated"] is True
    assert full["parameter_update_witness"]["changed_elements"] > 0
    assert full["train_loss_last_batch"] == resumed["train_loss_last_batch"]
    assert full["validation_loss"] == resumed["validation_loss"]
    assert partial["optimizer_steps"] == resumed["start_step"] == 1
    assert full["source_manifest_sha256"] == resumed["source_manifest_sha256"]
    first = torch.load(tmp_path / "full" / "checkpoint.pt", weights_only=True)
    second = torch.load(tmp_path / "split" / "checkpoint.pt", weights_only=True)
    for name in first["model"]:
        assert torch.equal(first["model"][name], second["model"][name])


def test_real_pilot_rejects_changed_source_and_cpu_fp16(tmp_path) -> None:
    config, data_dir, manifest = _fixture(tmp_path)
    with pytest.raises(ValueError, match="fp16 requires CUDA"):
        run(config, data_dir, manifest, tmp_path / "invalid", precision="fp16")
    train = data_dir / "train.jsonl"
    train.write_text(train.read_text() + "\n")
    with pytest.raises(ValueError, match="artifact hash"):
        run(config, data_dir, manifest, tmp_path / "invalid")


def test_representative_config_parameter_count_and_bounds() -> None:
    from pathlib import Path

    config_path = Path(__file__).resolve().parents[2] / "configs/m6_real_pilot.json"
    settings = json.loads(config_path.read_text())
    config = ModelConfig(**settings["model"])
    assert config.max_seq_len >= 256
    count = sum(p.numel() for p in DecoderOnlyTransformer(config).parameters())
    assert 10_000_000 <= count <= 30_000_000
    assert count == 21_502_464
    assert settings["microbatch_size"] * settings["gradient_accumulation_steps"] == 8


def test_small_parameter_update_witness() -> None:
    before = torch.ones(4)
    unchanged = parameter_update_witness(before, before.clone())
    assert unchanged["changed_elements"] == 0
    assert unchanged["max_abs_delta"] == 0
    assert unchanged["before_sha256"] == unchanged["after_sha256"]
    after = before.clone()
    after[2] -= 0.25
    changed = parameter_update_witness(before, after)
    assert changed["changed_elements"] == 1
    assert changed["max_abs_delta"] == 0.25
    assert changed["before_sha256"] != changed["after_sha256"]
