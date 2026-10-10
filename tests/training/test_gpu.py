"""Offline regressions and optional CUDA acceptance for FP32 pretraining."""

import json
import os
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from wendyfm.model import DecoderOnlyTransformer, ModelConfig
from wendyfm.training.gpu import run

CONFIG = Path(__file__).resolve().parents[2] / "configs/gpu_fp32.json"


def _assert_exact(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _assert_exact(first, second)
    else:
        assert left == right


def _assert_continuation(tmp_path: Path, device: str) -> None:
    full_dir = tmp_path / "full"
    split_dir = tmp_path / "split"
    full = run(CONFIG, full_dir, device_name=device)
    partial = run(CONFIG, split_dir, device_name=device, end_step=4)
    resumed = run(CONFIG, split_dir, device_name=device, resume=True)
    assert full["optimizer_steps"] == resumed["optimizer_steps"] == 8
    assert partial["optimizer_steps"] == resumed["start_step"] == 4
    assert full["timed_target_tokens"] == 8 * 4 * 16
    assert full["train_loss_last_batch"] == resumed["train_loss_last_batch"]
    assert full["validation_loss"] == resumed["validation_loss"]
    assert full["tokens_per_second"] > 0
    assert full["training_wall_time_seconds"] > 0
    assert json.loads((full_dir / "manifest.json").read_text()) == full
    assert (full_dir / "tokenizer.json").read_bytes() == (
        split_dir / "tokenizer.json"
    ).read_bytes()
    uninterrupted = torch.load(full_dir / "checkpoint.pt", weights_only=True)
    continued = torch.load(split_dir / "checkpoint.pt", weights_only=True)
    _assert_exact(uninterrupted, continued)


def test_fp32_entrypoint_exact_resume_on_cpu(tmp_path) -> None:
    _assert_continuation(tmp_path, "cpu")
    with pytest.raises(ValueError, match="output already contains"):
        run(CONFIG, tmp_path / "full", device_name="cpu")


def test_resume_rejects_changed_validation_data(tmp_path) -> None:
    source = json.loads(CONFIG.read_text())
    source["train_text"] = str((CONFIG.parent / source["train_text"]).resolve())
    source["validation_text"] = str((CONFIG.parent / source["validation_text"]).resolve())
    config = tmp_path / "config.json"
    config.write_text(json.dumps(source), encoding="utf-8")
    run(config, tmp_path / "output", device_name="cpu", end_step=4)
    validation = tmp_path / "changed.txt"
    validation.write_text("different held out validation sentence.\n", encoding="utf-8")
    source["validation_text"] = str(validation)
    config.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="resume inputs differ"):
        run(config, tmp_path / "output", device_name="cpu", resume=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp32_cuda_targets_gradients_metrics_and_exact_resume(tmp_path) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(5)
    config = ModelConfig(
        vocab_size=17, d_model=16, n_heads=2, n_layers=1,
        max_seq_len=4, intermediate_size=32,
    )
    model = DecoderOnlyTransformer(config).cuda()
    inputs = torch.tensor([[1, 2, 3, 4]], device="cuda")
    targets = torch.tensor([[2, 3, 4, 5]], device="cuda")
    expected = F.cross_entropy(model(inputs).reshape(-1, 17), targets.reshape(-1))
    actual = model.next_token_loss(inputs, targets)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters())

    _assert_continuation(tmp_path, "cuda")
    manifest = json.loads((tmp_path / "full" / "manifest.json").read_text())
    assert manifest["precision"] == "fp32"
    assert manifest["cuda_device_name"]
    assert manifest["peak_cuda_memory_bytes"] > 0
    checkpoint = torch.load(tmp_path / "full" / "checkpoint.pt", weights_only=True)
    assert checkpoint["format_version"] == 4
    assert checkpoint["cuda_rng_state"].dtype == torch.uint8
