"""Offline packaging and byte-tamper gates for the private M7 job."""

import hashlib
import json
import runpy
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOCAL = runpy.run_path(str(ROOT / "scripts/infra/prepare_kaggle_m7.py"))


def fixture(tmp_path: Path) -> tuple[Path, str, str]:
    data = tmp_path / "corpus"
    data.mkdir()
    artifacts = {}
    for name, key in LOCAL["ARTIFACTS"].items():
        raw = (name + " synthetic fixture\n").encode()
        (data / name).write_bytes(raw)
        artifacts[key] = hashlib.sha256(raw).hexdigest()
    (data / "manifest.json").write_text(json.dumps({
        **artifacts, "train_tokens_with_eod": 20_000_000,
        "validation_tokens_with_eod": 200_000,
    }))
    return data, LOCAL["file_sha256"](data / "manifest.json"), artifacts["tokenizer_sha256"]


def packaged(tmp_path: Path, monkeypatch) -> tuple[Path, Path, str, str]:
    data, manifest_sha, tokenizer_sha = fixture(tmp_path)
    globals_ = LOCAL["prepare_job"].__globals__
    monkeypatch.setitem(globals_, "TOKENIZER_SHA256", tokenizer_sha)
    job = tmp_path / "job"
    LOCAL["prepare_job"](job, data, "owner", "wendyfm-m7-corpus", "a" * 40, manifest_sha)
    return data, job, manifest_sha, tokenizer_sha


def test_private_package_has_no_corpus_bytes(tmp_path, monkeypatch) -> None:
    data, job, manifest_sha, tokenizer_sha = packaged(tmp_path, monkeypatch)
    assert sorted(p.name for p in job.iterdir()) == ["kernel-metadata.json", "remote_m7.py"]
    metadata = json.loads((job / "kernel-metadata.json").read_text())
    assert metadata["is_private"] is True
    assert metadata["dataset_sources"] == ["owner/wendyfm-m7-corpus"]
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    script = (job / "remote_m7.py").read_text()
    assert 'EXPECTED_COMMIT_SHA = "' + "a" * 40 + '"' in script
    assert manifest_sha in script and LOCAL["TOKENIZER_SHA256"] in script
    assert "__WENDYFM_COMMIT_SHA__" not in script
    for name in ("train.jsonl", "validation.jsonl", "tokenizer.json"):
        assert (data / name).read_text() not in script
    compile(script, str(job / "remote_m7.py"), "exec")
    with pytest.raises(ValueError, match="empty"):
        LOCAL["prepare_job"](job, data, "owner", "wendyfm-m7-corpus", "a" * 40,
                             manifest_sha)


@pytest.mark.parametrize("tamper", ["manifest", "train", "validation", "tokenizer",
                                      "extra", "missing"])
def test_package_rejects_changed_or_missing_data(tmp_path, monkeypatch, tamper) -> None:
    data, manifest_sha, tokenizer_sha = fixture(tmp_path)
    monkeypatch.setitem(LOCAL["prepare_job"].__globals__, "TOKENIZER_SHA256", tokenizer_sha)
    if tamper == "manifest":
        (data / "manifest.json").write_text("{}")
    elif tamper == "extra":
        (data / "secret.txt").write_text("no")
    elif tamper == "missing":
        (data / "train.jsonl").unlink()
    else:
        name = "validation.jsonl" if tamper == "validation" else (
            "tokenizer.json" if tamper == "tokenizer" else "train.jsonl")
        (data / name).write_text("tampered")
    with pytest.raises(ValueError, match="SHA256|four regular"):
        LOCAL["prepare_job"](tmp_path / "job", data, "owner", "wendyfm-m7-corpus",
                             "a" * 40, manifest_sha)


def test_remote_rejects_attachment_before_torch(tmp_path, monkeypatch) -> None:
    data, job, manifest_sha, tokenizer_sha = packaged(tmp_path, monkeypatch)
    remote = runpy.run_path(str(job / "remote_m7.py"))
    globals_ = remote["verify_attachment"].__globals__
    monkeypatch.setitem(globals_, "TOKENIZER_SHA256", tokenizer_sha)
    source = remote["verify_attachment"](data, float("inf"))
    assert source["train_artifact_sha256"] == hashlib.sha256(
        (data / "train.jsonl").read_bytes()).hexdigest()
    (data / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="train.jsonl SHA256"):
        remote["verify_attachment"](data, float("inf"))
    (data / "train.jsonl").write_text("train.jsonl synthetic fixture\n")
    (data / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest SHA256"):
        remote["verify_attachment"](data, float("inf"))


def test_pin_and_result_reject_partial_or_wrong_provenance(tmp_path, monkeypatch) -> None:
    data, job, manifest_sha, _ = packaged(tmp_path, monkeypatch)
    remote = runpy.run_path(str(job / "remote_m7.py"))
    globals_ = remote["main"].__globals__
    result_path = tmp_path / "result.json"
    monkeypatch.setitem(globals_, "RESULT_PATH", result_path)

    def fail_clone(*args, **kwargs):
        raise RuntimeError("synthetic clone failure")

    monkeypatch.setitem(globals_, "run_command", fail_clone)
    with pytest.raises(SystemExit, match="1"):
        remote["main"]()
    result = json.loads(result_path.read_text())
    assert result["status"] == "FAIL"
    assert result["failure_stage"] == "clone_source"
    assert result["retries"] == 0
    with pytest.raises(ValueError, match="status"):
        LOCAL["verify_result"](result_path, data, "a" * 40, manifest_sha)
    result["status"] = "PASS"
    result["tested_commit_sha"] = "b" * 40
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="provenance"):
        LOCAL["verify_result"](result_path, data, "a" * 40, manifest_sha)


def test_result_verifier_checks_metrics_and_artifact_hashes(tmp_path, monkeypatch) -> None:
    data, manifest_sha, tokenizer_sha = fixture(tmp_path)
    monkeypatch.setitem(LOCAL["verify_attachment"].__globals__,
                        "TOKENIZER_SHA256", tokenizer_sha)
    source = json.loads((data / "manifest.json").read_text())
    initial = {"event": "initial", "step": 0, "plan_sha256": "b" * 64,
               "cumulative_target_tokens": 0, "validation_loss": 7.0,
               "peak_cuda_reserved_bytes": 100}
    final = {"event": "complete", "step": 489, "plan_sha256": "b" * 64,
             "cumulative_target_tokens": 1_001_472, "validation_loss": 6.0,
             "peak_cuda_reserved_bytes": 200, "work_wall_seconds": 100.0,
             "timed_target_tokens": 997_376, "timed_seconds": 50.0,
             "timed_target_tokens_per_second": 997_376 / 50,
             "train_loss_curve": [{"step": step, "train_loss": 6.0,
                                   "gradient_norm": 1.0} for step in range(1, 490)]}
    result = {"status": "PASS", "test": "m7_1m_private_t4", "retries": 0,
              "expected_commit_sha": "a" * 40, "tested_commit_sha": "a" * 40,
              "expected_manifest_sha256": manifest_sha,
              "plan": {"source_commit_sha": "a" * 40,
                       "source_manifest_sha256": manifest_sha,
                       "config_sha256": LOCAL["CONFIG_SHA256"],
                       "tokenizer_sha256": tokenizer_sha,
                       "train_artifact_sha256": source["train_artifact_sha256"],
                       "validation_artifact_sha256": source["validation_artifact_sha256"],
                       "device": "cuda:0", "cuda_device_count": 1,
                       "precision": "fp32", "attention_backend": "sdpa",
                       "gpu_name": "Tesla T4", "max_steps": 489,
                       "tokens_per_step": 2048, "target_tokens": 1_000_000,
                       "max_wall_seconds": 3600.0,
                       "max_cuda_reserved_bytes": 12 * 1024**3},
              "events": [initial, final],
              "checkpoint": {"step": 489, "sha256": "c" * 64, "bytes": 100}}
    plan_sha = hashlib.sha256(json.dumps(result["plan"], sort_keys=True).encode()).hexdigest()
    initial["plan_sha256"] = final["plan_sha256"] = plan_sha
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    assert LOCAL["verify_result"](path, data, "a" * 40, manifest_sha) == result
    result["events"][1]["train_loss_curve"][0]["step"] = 2
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="training loss curve"):
        LOCAL["verify_result"](path, data, "a" * 40, manifest_sha)
    result["events"][1]["train_loss_curve"][0]["step"] = 1
    (data / "validation.jsonl").write_text("tamper")
    with pytest.raises(ValueError, match="validation.jsonl SHA256"):
        LOCAL["verify_result"](path, data, "a" * 40, manifest_sha)


def test_remote_child_timeout_stops_without_retry(tmp_path, monkeypatch) -> None:
    _, job, _, _ = packaged(tmp_path, monkeypatch)
    remote = runpy.run_path(str(job / "remote_m7.py"))
    with pytest.raises(TimeoutError, match="time limit"):
        remote["run_command"]([sys.executable, "-c", "import time; time.sleep(5)"],
                              deadline=time.monotonic() + 0.05)
