"""Stream and verify an M7 corpus using the existing M6 training data contract."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Iterable, Mapping
from itertools import zip_longest
from pathlib import Path

from wendyfm.tokenizer import BPETokenizer
from wendyfm.training.real_data import (
    DATASET_CONFIG,
    DATASET_REPO,
    DATASET_REVISION,
    DATASET_SHARD,
    SPLIT_SEED,
    sha256,
    split_for_id,
)

EOD = "<|eod|>"
SPLITS = ("train", "validation")
ROOT = Path(__file__).resolve().parents[3]
M6_MANIFEST = ROOT / "data/manifests/fineweb_edu_m6.json"
SPLIT_RULE = f"sha256('{SPLIT_SEED}:' + id) % 10 == 0 -> validation"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _tokenizer(path: Path) -> BPETokenizer:
    tokenizer = BPETokenizer.load(path)
    if EOD not in tokenizer.spec.special_tokens:
        raise ValueError("tokenizer lacks the EOD special token")
    return tokenizer


def _expected_tokenizer_sha256(override: str | None) -> str:
    if override is None:
        override = json.loads(M6_MANIFEST.read_text())["tokenizer_sha256"]
    if (not isinstance(override, str) or len(override) != 64 or
            any(character not in "0123456789abcdef" for character in override)):
        raise ValueError("invalid expected tokenizer SHA256")
    return override


def prepare_corpus(
    rows: Iterable[Mapping[str, object]], output_dir: Path, tokenizer_path: Path, *,
    min_train_tokens: int, min_validation_tokens: int, max_documents: int,
    max_text_bytes: int, expected_tokenizer_sha256: str | None = None,
) -> dict:
    """Select the shortest source-order prefix meeting both token floors.

    The source iterator must be the pinned FineWeb-Edu shard. Artifacts are
    written outside Git and become visible only after a complete preparation.
    """
    for name, value in (("min_train_tokens", min_train_tokens),
                        ("min_validation_tokens", min_validation_tokens),
                        ("max_documents", max_documents), ("max_text_bytes", max_text_bytes)):
        _positive_int(value, name)
    output_dir = Path(output_dir)
    resolved = output_dir.resolve()
    if resolved.is_relative_to(ROOT) and not resolved.is_relative_to(ROOT / "data/raw"):
        raise ValueError("repository-local corpora must be under ignored data/raw")
    if output_dir.exists():
        raise ValueError("output directory must not exist")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(tokenizer_path)
    expected_hash = _expected_tokenizer_sha256(expected_tokenizer_sha256)
    if file_sha256(tokenizer_path) != expected_hash:
        raise ValueError("tokenizer differs from the frozen M6 tokenizer")
    tokenizer = _tokenizer(tokenizer_path)
    tokenizer_bytes = tokenizer_path.read_bytes()
    counts = {split: 0 for split in SPLITS}
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    documents: list[dict] = []
    text_bytes = 0
    rows_seen = 0
    duplicate_text_skipped = 0
    with tempfile.TemporaryDirectory(prefix=".m7-data-", dir=output_dir.parent) as temp:
        stage = Path(temp)
        (stage / "tokenizer.json").write_bytes(tokenizer_bytes)
        with (stage / "train.jsonl").open("w", encoding="utf-8") as train_file, \
                (stage / "validation.jsonl").open("w", encoding="utf-8") as val_file:
            files = {"train": train_file, "validation": val_file}
            for row in rows:
                if all(counts[split] >= floor for split, floor in
                       (("train", min_train_tokens), ("validation", min_validation_tokens))):
                    break
                if rows_seen >= max_documents:
                    break
                rows_seen += 1
                document_id, content = row.get("id"), row.get("text")
                if not isinstance(document_id, str) or not document_id:
                    raise ValueError("source row lacks a nonempty document ID")
                if document_id in seen_ids:
                    raise ValueError("duplicate source document ID")
                seen_ids.add(document_id)
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("source row lacks nonempty text")
                encoded = content.encode("utf-8")
                if len(encoded) > max_text_bytes - text_bytes:
                    break
                text_hash = sha256(encoded)
                if text_hash in seen_text:
                    duplicate_text_skipped += 1
                    continue
                seen_text.add(text_hash)
                split = split_for_id(document_id)
                token_count = len(tokenizer.encode(content)) + 1
                record = {"id": document_id, "split": split, "text_sha256": text_hash,
                          "text_bytes": len(encoded), "token_count_with_eod": token_count}
                documents.append(record)
                counts[split] += token_count
                text_bytes += len(encoded)
                files[split].write(json.dumps({"id": document_id, "text": content},
                                              ensure_ascii=False, sort_keys=True) + "\n")
        if counts["train"] < min_train_tokens or counts["validation"] < min_validation_tokens:
            raise ValueError("source ended before both token floors were met")
        manifest = {
            "schema_version": 1, "dataset_repo": DATASET_REPO,
            "dataset_config": DATASET_CONFIG, "dataset_revision": DATASET_REVISION,
            "dataset_split": "train", "dataset_shard": DATASET_SHARD,
            "dataset_license": "odc-by",
            "split_rule": SPLIT_RULE,
            "min_train_tokens": min_train_tokens,
            "min_validation_tokens": min_validation_tokens,
            "max_documents": max_documents, "max_text_bytes": max_text_bytes,
            "rows_seen": rows_seen, "duplicate_text_skipped": duplicate_text_skipped,
            "selected_documents": len(documents), "selected_text_bytes": text_bytes,
            "train_tokens_with_eod": counts["train"],
            "validation_tokens_with_eod": counts["validation"],
            "vocab_size": tokenizer.spec.config.vocab_size,
            "tokenizer_sha256": file_sha256(stage / "tokenizer.json"),
            "train_artifact_sha256": file_sha256(stage / "train.jsonl"),
            "validation_artifact_sha256": file_sha256(stage / "validation.jsonl"),
            "documents": documents,
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        stage.rename(output_dir)
    return manifest


def verify_corpus(output_dir: Path, *, expected_tokenizer_sha256: str | None = None) -> dict:
    """Check bytes, split separation, document hashes and token accounting."""
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("invalid corpus manifest")
    if (manifest.get("schema_version"), manifest.get("dataset_repo"),
            manifest.get("dataset_config"), manifest.get("dataset_revision"),
            manifest.get("dataset_split"), manifest.get("dataset_shard"),
            manifest.get("dataset_license")) != (
                1, DATASET_REPO, DATASET_CONFIG, DATASET_REVISION, "train",
                DATASET_SHARD, "odc-by"):
        raise ValueError("source provenance differs from the pinned M7 contract")
    if manifest.get("split_rule") != SPLIT_RULE:
        raise ValueError("split rule differs from the pinned M7 contract")
    for name in ("min_train_tokens", "min_validation_tokens", "max_documents",
                 "max_text_bytes", "selected_documents", "selected_text_bytes"):
        _positive_int(manifest.get(name), name)
    for name in ("rows_seen", "duplicate_text_skipped"):
        value = manifest.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid {name}")
    if (manifest["selected_documents"] > manifest["rows_seen"] or
            manifest["duplicate_text_skipped"] > manifest["rows_seen"]):
        raise ValueError("invalid source row counts")
    tokenizer_path = output_dir / "tokenizer.json"
    if (file_sha256(tokenizer_path) != manifest.get("tokenizer_sha256") or
            manifest["tokenizer_sha256"] !=
            _expected_tokenizer_sha256(expected_tokenizer_sha256)):
        raise ValueError("tokenizer artifact hash mismatch")
    tokenizer = _tokenizer(tokenizer_path)
    if tokenizer.spec.config.vocab_size != manifest.get("vocab_size"):
        raise ValueError("tokenizer vocabulary mismatch")
    documents = manifest.get("documents")
    if (not isinstance(documents, list) or
            any(not isinstance(record, dict) for record in documents) or
            len(documents) != manifest["selected_documents"]):
        raise ValueError("document count mismatch")
    if len(documents) > manifest["max_documents"]:
        raise ValueError("document cap exceeded")
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    counts = {split: 0 for split in SPLITS}
    total_bytes = 0
    for split in SPLITS:
        path = output_dir / f"{split}.jsonl"
        if file_sha256(path) != manifest.get(f"{split}_artifact_sha256"):
            raise ValueError(f"{split} artifact hash mismatch")
        expected = (record for record in documents if record.get("split") == split)
        with path.open(encoding="utf-8") as source:
            for line, record in zip_longest(source, expected):
                if line is None or record is None:
                    raise ValueError(f"{split} document count mismatch")
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != {"id", "text"}:
                    raise ValueError("invalid source row")
                document_id, content = row["id"], row["text"]
                if not isinstance(document_id, str) or not isinstance(content, str):
                    raise ValueError("invalid source row values")
                encoded = content.encode("utf-8")
                text_hash = sha256(encoded)
                if (document_id != record.get("id") or split_for_id(document_id) != split or
                        text_hash != record.get("text_sha256") or
                        len(encoded) != record.get("text_bytes") or
                        len(tokenizer.encode(content)) + 1 != record.get("token_count_with_eod")):
                    raise ValueError("document content, split, or token count mismatch")
                if document_id in seen_ids or text_hash in seen_text:
                    raise ValueError("duplicate document ID or content")
                seen_ids.add(document_id)
                seen_text.add(text_hash)
                counts[split] += record["token_count_with_eod"]
                total_bytes += len(encoded)
    if (set(record.get("split") for record in documents) != set(SPLITS) or
            total_bytes != manifest["selected_text_bytes"] or
            total_bytes > manifest["max_text_bytes"] or
            any(counts[split] != manifest.get(f"{split}_tokens_with_eod") or
                counts[split] < manifest[f"min_{split}_tokens"] for split in SPLITS)):
        raise ValueError("manifest totals or token floors mismatch")
    return manifest
