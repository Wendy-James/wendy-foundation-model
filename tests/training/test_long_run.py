"""CPU-only M7 interval, provenance and exact-resume acceptance tests."""

import json

import pytest
import torch

from wendyfm.model import DecoderOnlyTransformer, ModelConfig
from wendyfm.tokenizer import BPETokenizer
from wendyfm.training.long_data import file_sha256, prepare_corpus
from wendyfm.training.long_run import run_long


def _fixture(tmp_path):
    rows = [{"id": f"doc-{i}",
             "text": f"A distinct astronomy observation number {i}. " * 4}
            for i in range(80)]
    tokenizer = BPETokenizer.train([row["text"] for row in rows], vocab_size=272,
                                   special_tokens=("<|eod|>",))
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    data_dir = tmp_path / "private-data"
    prepare_corpus(rows, data_dir, tokenizer_path, min_train_tokens=500,
                   min_validation_tokens=100, max_documents=80,
                   max_text_bytes=100_000,
                   expected_tokenizer_sha256=file_sha256(tokenizer_path))
    config = tmp_path / "config.json"
    settings = {
        "cpu_threads": 1, "gradient_accumulation_steps": 1,
        "metric_warmup_steps": 1, "microbatch_size": 1,
        "model": {"attention_backend": "sdpa", "d_model": 16,
                  "intermediate_size": 32, "max_seq_len": 16,
                  "n_heads": 2, "n_layers": 1, "vocab_size": 272},
        "precision": "fp32", "seed": 23,
        "train": {"learning_rate": 0.001, "max_steps": 4,
                  "min_learning_rate": 0.0001, "warmup_steps": 1},
        "long_run": {"target_tokens": 64, "validation_interval_steps": 2,
                     "max_cuda_reserved_bytes": 12 * 1024**3,
                     "max_wall_seconds": 60},
    }
    config.write_text(json.dumps(settings))
    kwargs = {"expected_manifest_sha256": file_sha256(data_dir / "manifest.json"),
              "expected_tokenizer_sha256": file_sha256(tokenizer_path)}
    return config, data_dir, settings, kwargs


def _events(output):
    return [json.loads(path.read_text()) for path in sorted((output / "events").glob("*.json"))]


def _assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_state_equal(a, b)
    else:
        assert left == right


def test_frozen_t4_protocol_is_one_million_token_bounded() -> None:
    from wendyfm.training.long_data import ROOT

    settings = json.loads((ROOT / "configs/m7_t4_1m.json").read_text())
    model = DecoderOnlyTransformer(ModelConfig(**settings["model"]))
    assert sum(parameter.numel() for parameter in model.parameters()) == 21_502_464
    tokens_per_step = (settings["microbatch_size"] *
                       settings["gradient_accumulation_steps"] *
                       settings["model"]["max_seq_len"])
    assert tokens_per_step == 2_048
    assert settings["train"]["max_steps"] == 489
    assert settings["long_run"]["validation_interval_steps"] == 489
    assert 1_000_000 <= tokens_per_step * 489 < 1_000_000 + tokens_per_step
    assert settings["precision"] == "fp32"
    assert settings["model"]["attention_backend"] == "sdpa"


def test_intervals_and_exact_resume_match_uninterrupted_cpu_run(tmp_path) -> None:
    config, data_dir, _, kwargs = _fixture(tmp_path)
    full_dir, split_dir = tmp_path / "full", tmp_path / "split"
    full = run_long(config, data_dir, full_dir, **kwargs)
    partial = run_long(config, data_dir, split_dir, stop_after_step=2, **kwargs)
    # Simulate a crash after writing the next checkpoint but before its event.
    (split_dir / "checkpoint_00000004.pt").write_bytes(b"orphan")
    resumed = run_long(config, data_dir, split_dir, resume=True, **kwargs)
    assert (partial["step"], resumed["step"], full["step"]) == (2, 4, 4)
    assert [event["step"] for event in _events(full_dir)] == [0, 2, 4]
    assert [event["step"] for event in _events(split_dir)] == [0, 2, 4]
    for baseline, continued in zip(_events(full_dir), _events(split_dir), strict=True):
        for key in ("step", "cumulative_target_tokens", "validation_loss",
                    "train_loss_curve", "train_loss_first_batch", "train_loss_last_batch",
                    "learning_rate"):
            if key in baseline:
                assert baseline[key] == continued[key]
    assert resumed["cumulative_target_tokens"] == 64
    assert resumed["timed_target_tokens"] == 2 * 16
    assert resumed["timed_target_tokens_per_second"] == pytest.approx(
        resumed["timed_target_tokens"] / resumed["timed_seconds"])
    assert resumed["full_run_target_tokens_per_second"] == pytest.approx(
        64 / resumed["work_wall_seconds"])
    assert resumed["peak_cuda_allocated_bytes"] is None
    assert resumed["peak_cuda_reserved_bytes"] is None
    assert list(full_dir.glob("checkpoint_*.pt")) == [full_dir / "checkpoint_00000004.pt"]
    assert list(split_dir.glob("checkpoint_*.pt")) == [split_dir / "checkpoint_00000004.pt"]
    full_state = torch.load(full_dir / "checkpoint_00000004.pt", weights_only=True)
    split_state = torch.load(split_dir / "checkpoint_00000004.pt", weights_only=True)
    _assert_state_equal(full_state, split_state)


def test_resume_rejects_changed_config_manifest_or_checkpoint(tmp_path) -> None:
    config, data_dir, settings, kwargs = _fixture(tmp_path)
    output = tmp_path / "run"
    run_long(config, data_dir, output, stop_after_step=2, **kwargs)
    settings["train"]["learning_rate"] = 0.002
    config.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="frozen run plan"):
        run_long(config, data_dir, output, resume=True, **kwargs)
    settings["train"]["learning_rate"] = 0.001
    config.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="manifest SHA256"):
        run_long(config, data_dir, output, resume=True,
                 **{**kwargs, "expected_manifest_sha256": "0" * 64})
    checkpoint = output / "checkpoint_00000002.pt"
    checkpoint.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="invalid checkpoint"):
        run_long(config, data_dir, output, resume=True, **kwargs)


def test_real_protocol_rejects_corpus_below_m7_floors(tmp_path, monkeypatch) -> None:
    import wendyfm.training.long_run as long_run

    _, data_dir, _, kwargs = _fixture(tmp_path)
    frozen = tmp_path / "m6-manifest.json"
    frozen.write_text(json.dumps({"tokenizer_sha256": kwargs["expected_tokenizer_sha256"]}))
    monkeypatch.setattr(long_run, "M6_MANIFEST", frozen)
    with pytest.raises(ValueError, match="required real-data token floors"):
        long_run._runtime_source(data_dir,
                                 (data_dir / "manifest.json").read_bytes(), None)


def test_budget_boundaries_and_artifact_tampering_fail_closed(tmp_path) -> None:
    config, data_dir, settings, kwargs = _fixture(tmp_path)
    with pytest.raises(ValueError, match="executable commit SHA"):
        run_long(config, data_dir, tmp_path / "invalid", expected_commit_sha="0" * 40,
                 **kwargs)
    with pytest.raises(ValueError, match="interval boundary"):
        run_long(config, data_dir, tmp_path / "invalid", stop_after_step=1, **kwargs)
    settings["train"]["max_steps"] = 5
    config.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="token budget"):
        run_long(config, data_dir, tmp_path / "invalid", **kwargs)
    settings["train"]["max_steps"] = 4
    config.write_text(json.dumps(settings))
    train_path = data_dir / "train.jsonl"
    train_path.write_text(train_path.read_text() + "\n")
    with pytest.raises(ValueError, match="artifact hash"):
        run_long(config, data_dir, tmp_path / "invalid", **kwargs)


def test_wall_limit_and_unsafe_output_path(tmp_path) -> None:
    config, data_dir, settings, kwargs = _fixture(tmp_path)
    settings["long_run"]["max_wall_seconds"] = 0.000001
    config.write_text(json.dumps(settings))
    with pytest.raises(TimeoutError, match="wall-time limit"):
        run_long(config, data_dir, tmp_path / "limited", **kwargs)
    failure = json.loads((tmp_path / "limited" / "failure.json").read_text())
    assert failure["status"] == "FAIL"
    assert failure["stage"] in ("initial_validation", "checkpoint", "training")
    assert failure["error_type"] == "TimeoutError"
    from wendyfm.training.long_data import ROOT

    with pytest.raises(ValueError, match="ignored outputs"):
        run_long(config, data_dir, ROOT / "docs" / "unsafe-run", **kwargs)
