"""Offline source rows stand in for the pinned FineWeb-Edu stream."""

import json
from pathlib import Path

import pytest

from wendyfm.training.real_data import (
    DATASET_CONFIG,
    DATASET_REPO,
    DATASET_REVISION,
    prepare_documents,
    sha256,
    split_for_id,
)


def _rows():
    return [{"id": f"doc-{i}", "text": f"Document {i} contains useful science text. " * 5}
            for i in range(60)]


def test_bounded_manifest_split_and_artifacts(tmp_path) -> None:
    rows = _rows()
    output = tmp_path / "raw"
    manifest_path = tmp_path / "manifest" / "source.json"
    manifest = prepare_documents(rows, output, manifest_path, max_documents=60,
                                 max_text_bytes=20_000, vocab_size=272)
    assert manifest == json.loads(manifest_path.read_text())
    assert manifest["selected_documents"] <= 60
    assert manifest["selected_text_bytes"] <= 20_000
    assert {row["split"] for row in manifest["documents"]} == {"train", "validation"}
    assert len({row["id"] for row in manifest["documents"]}) == len(manifest["documents"])
    for row in manifest["documents"]:
        assert row["split"] == split_for_id(row["id"])
        source = next(item["text"] for item in rows if item["id"] == row["id"])
        assert row["text_sha256"] == sha256(source.encode())
        assert row["token_count_with_eod"] > 1
    assert manifest["tokenizer_sha256"] == sha256((output / "tokenizer.json").read_bytes())
    assert manifest["train_artifact_sha256"] == sha256((output / "train.jsonl").read_bytes())
    assert manifest["validation_artifact_sha256"] == sha256(
        (output / "validation.jsonl").read_bytes())
    second = prepare_documents(rows, tmp_path / "raw2", tmp_path / "manifest2" / "source.json",
                               max_documents=60, max_text_bytes=20_000, vocab_size=272)
    assert second == manifest


def test_invalid_or_insufficient_source_fails_closed(tmp_path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        prepare_documents([_rows()[0]] * 2, tmp_path / "raw", tmp_path / "m" / "x",
                          max_documents=2, max_text_bytes=1000)
    with pytest.raises(ValueError, match="no independent"):
        prepare_documents([_rows()[0]], tmp_path / "raw", tmp_path / "m" / "x",
                          max_documents=1, max_text_bytes=1000)


def test_committed_real_manifest_contract() -> None:
    path = Path(__file__).resolve().parents[2] / "data/manifests/fineweb_edu_m6.json"
    manifest = json.loads(path.read_text())
    assert (manifest["dataset_repo"], manifest["dataset_config"],
            manifest["dataset_revision"]) == (DATASET_REPO, DATASET_CONFIG, DATASET_REVISION)
    assert manifest["selected_documents"] == len(manifest["documents"]) == 64
    assert manifest["selected_text_bytes"] == sum(
        item["text_bytes"] for item in manifest["documents"])
    assert manifest["selected_text_bytes"] <= manifest["max_text_bytes"]
    assert {item["split"] for item in manifest["documents"]} == {"train", "validation"}
    assert len({item["id"] for item in manifest["documents"]}) == 64
    assert all(item["split"] == split_for_id(item["id"]) for item in manifest["documents"])
    assert all(len(item["text_sha256"]) == 64 for item in manifest["documents"])
