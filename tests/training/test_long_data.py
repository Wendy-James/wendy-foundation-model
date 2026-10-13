"""Offline checks for M7 streaming preparation and fail-closed preflight."""

import json

import pytest

from wendyfm.tokenizer import BPETokenizer
from wendyfm.training.long_data import file_sha256, prepare_corpus, verify_corpus
from wendyfm.training.real_pilot import _source_documents, run


def _inputs(tmp_path):
    rows = [{"id": f"doc-{i}",
             "text": f"Document {i} describes a distinct astronomy observation. " * 3}
            for i in range(80)]
    tokenizer = BPETokenizer.train([row["text"] for row in rows], vocab_size=272,
                                   special_tokens=("<|eod|>",))
    tokenizer_path = tmp_path / "frozen-tokenizer.json"
    tokenizer.save(tokenizer_path)
    return rows, tokenizer_path


def _prepare(tmp_path):
    rows, tokenizer_path = _inputs(tmp_path)
    tokenizer_hash = file_sha256(tokenizer_path)
    output = tmp_path / "private-corpus"
    manifest = prepare_corpus(rows, output, tokenizer_path, min_train_tokens=500,
                              min_validation_tokens=100, max_documents=80,
                              max_text_bytes=100_000,
                              expected_tokenizer_sha256=tokenizer_hash)
    return rows, tokenizer_path, output, manifest


def test_streamed_corpus_is_deterministic_and_m6_training_compatible(tmp_path) -> None:
    rows, tokenizer_path, output, manifest = _prepare(tmp_path)
    assert verify_corpus(output, expected_tokenizer_sha256=file_sha256(tokenizer_path)) == manifest
    assert manifest["train_tokens_with_eod"] >= 500
    assert manifest["validation_tokens_with_eod"] >= 100
    assert manifest["selected_documents"] < len(rows)
    assert len({row["id"] for row in manifest["documents"]}) == manifest["selected_documents"]
    assert len({row["text_sha256"] for row in manifest["documents"]}) == len(
        manifest["documents"])
    train, validation = _source_documents(output, manifest)
    assert len(train) + len(validation) == manifest["selected_documents"]
    second = tmp_path / "second-corpus"
    assert prepare_corpus(rows, second, tokenizer_path, min_train_tokens=500,
                          min_validation_tokens=100, max_documents=80,
                          max_text_bytes=100_000,
                          expected_tokenizer_sha256=file_sha256(tokenizer_path)) == manifest
    for filename in ("manifest.json", "train.jsonl", "validation.jsonl", "tokenizer.json"):
        assert (output / filename).read_bytes() == (second / filename).read_bytes()


def test_duplicate_text_is_skipped_without_cross_split_leakage(tmp_path) -> None:
    rows, tokenizer_path = _inputs(tmp_path)
    duplicate = {"id": "different-id", "text": rows[0]["text"]}
    output = tmp_path / "private-corpus"
    manifest = prepare_corpus([rows[0], duplicate, *rows[1:]], output, tokenizer_path,
                              min_train_tokens=500, min_validation_tokens=100,
                              max_documents=80, max_text_bytes=100_000,
                              expected_tokenizer_sha256=file_sha256(tokenizer_path))
    assert manifest["duplicate_text_skipped"] == 1
    assert verify_corpus(output, expected_tokenizer_sha256=file_sha256(tokenizer_path)) == manifest


def test_preparation_fails_atomically_when_budget_cannot_be_met(tmp_path) -> None:
    rows, tokenizer_path = _inputs(tmp_path)
    output = tmp_path / "private-corpus"
    with pytest.raises(ValueError, match="token floors"):
        prepare_corpus(rows[:2], output, tokenizer_path, min_train_tokens=10_000,
                       min_validation_tokens=1_000, max_documents=2,
                       max_text_bytes=100_000,
                       expected_tokenizer_sha256=file_sha256(tokenizer_path))
    assert not output.exists()


def test_default_preflight_requires_the_frozen_m6_tokenizer(tmp_path) -> None:
    rows, tokenizer_path = _inputs(tmp_path)
    with pytest.raises(ValueError, match="frozen M6 tokenizer"):
        prepare_corpus(rows, tmp_path / "private-corpus", tokenizer_path,
                       min_train_tokens=500, min_validation_tokens=100,
                       max_documents=80, max_text_bytes=100_000)


def test_m7_manifest_runs_through_existing_segment_and_resume_api(tmp_path) -> None:
    _, _, output, _ = _prepare(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "cpu_threads": 1, "microbatch_size": 1, "gradient_accumulation_steps": 1,
        "metric_warmup_steps": 0, "seed": 17,
        "model": {"vocab_size": 272, "d_model": 16, "n_heads": 2,
                  "n_layers": 1, "max_seq_len": 16, "intermediate_size": 32,
                  "attention_backend": "sdpa"},
        "train": {"max_steps": 3, "learning_rate": 0.001},
    }))
    first = run(config, output, output / "manifest.json", tmp_path / "run", end_step=1)
    resumed = run(config, output, output / "manifest.json", tmp_path / "run",
                  resume=True, end_step=2)
    assert first["optimizer_steps"] == resumed["start_step"] == 1
    assert resumed["optimizer_steps"] == 2
    assert resumed["source_manifest_sha256"] == first["source_manifest_sha256"]


def test_preflight_rejects_tampering_even_with_updated_artifact_hash(tmp_path) -> None:
    _, tokenizer_path, output, manifest = _prepare(tmp_path)
    path = output / "train.jsonl"
    path.write_text(path.read_text().replace("astronomy", "geology", 1))
    with pytest.raises(ValueError, match="artifact hash"):
        verify_corpus(output, expected_tokenizer_sha256=file_sha256(tokenizer_path))

    manifest["train_artifact_sha256"] = file_sha256(path)
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="document content"):
        verify_corpus(output, expected_tokenizer_sha256=file_sha256(tokenizer_path))


def test_preflight_rejects_changed_token_counts_or_revision(tmp_path) -> None:
    _, tokenizer_path, output, manifest = _prepare(tmp_path)
    manifest["documents"][0]["token_count_with_eod"] += 1
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="token count"):
        verify_corpus(output, expected_tokenizer_sha256=file_sha256(tokenizer_path))
    manifest["documents"][0]["token_count_with_eod"] -= 1
    manifest["dataset_revision"] = "other"
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance"):
        verify_corpus(output, expected_tokenizer_sha256=file_sha256(tokenizer_path))
