"""Bounded, revision-pinned FineWeb-Edu document preparation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from wendyfm.tokenizer import BPETokenizer

DATASET_REPO = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
DATASET_SHARD = "sample/10BT/000_00000.parquet"
SPLIT_SEED = "wendyfm-m6-v1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split_for_id(document_id: str) -> str:
    """Stable 90/10 assignment independent of stream order and future additions."""
    value = int(sha256(f"{SPLIT_SEED}:{document_id}".encode()), 16)
    return "validation" if value % 10 == 0 else "train"


def prepare_documents(
    rows: Iterable[Mapping[str, object]], output_dir: Path, manifest_path: Path, *,
    max_documents: int, max_text_bytes: int, vocab_size: int = 1024,
) -> dict:
    """Write bounded local artifacts and a source manifest from ordered HF rows.

    The caller controls the pinned source iterator. Outputs must be distinct;
    raw text and token artifacts belong in an ignored directory.
    """
    if any(type(value) is not int or value < 1 for value in
           (max_documents, max_text_bytes, vocab_size)):
        raise ValueError("bounds and vocabulary size must be positive integers")
    if vocab_size < 258:
        raise ValueError("vocabulary must fit bytes and the EOD token")
    if output_dir.resolve() == manifest_path.parent.resolve():
        raise ValueError("manifest and raw artifacts must use different directories")
    selected: list[tuple[str, str, str, bytes]] = []
    seen: set[str] = set()
    total_bytes = 0
    for row in rows:
        if len(selected) >= max_documents:
            break
        document_id, content = row.get("id"), row.get("text")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("FineWeb-Edu row lacks a nonempty id")
        if document_id in seen:
            raise ValueError("duplicate FineWeb-Edu document id")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("FineWeb-Edu row lacks nonempty text")
        encoded = content.encode("utf-8")
        if len(encoded) > max_text_bytes - total_bytes:
            break
        seen.add(document_id)
        selected.append((document_id, content, split_for_id(document_id), encoded))
        total_bytes += len(encoded)
    if not selected:
        raise ValueError("no document fits the bounds")
    if {item[2] for item in selected} != {"train", "validation"}:
        raise ValueError("bounds produced no independent train/validation split")

    train_texts = [text for _, text, split, _ in selected if split == "train"]
    tokenizer = BPETokenizer.train(train_texts, vocab_size=vocab_size,
                                   special_tokens=("<|eod|>",))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    documents = []
    for split in ("train", "validation"):
        lines = []
        for document_id, text, assigned, encoded in selected:
            if assigned != split:
                continue
            token_count = len(tokenizer.encode(text)) + 1
            documents.append({"id": document_id, "split": split,
                              "text_sha256": sha256(encoded), "text_bytes": len(encoded),
                              "token_count_with_eod": token_count})
            lines.append(json.dumps({"id": document_id, "text": text},
                                    ensure_ascii=False, sort_keys=True))
        (output_dir / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "dataset_repo": DATASET_REPO, "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION, "dataset_split": "train",
        "dataset_shard": DATASET_SHARD, "dataset_license": "odc-by",
        "split_rule": f"sha256('{SPLIT_SEED}:' + id) % 10 == 0 -> validation",
        "max_documents": max_documents, "max_text_bytes": max_text_bytes,
        "selected_documents": len(selected), "selected_text_bytes": total_bytes,
        "vocab_size": tokenizer.spec.config.vocab_size,
        "tokenizer_sha256": sha256(tokenizer_path.read_bytes()),
        "train_artifact_sha256": sha256((output_dir / "train.jsonl").read_bytes()),
        "validation_artifact_sha256": sha256((output_dir / "validation.jsonl").read_bytes()),
        "documents": documents,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    return manifest


def fineweb_rows():
    """Range-read only the needed rows from one revision-pinned Parquet shard."""
    try:
        import fsspec
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "install optional fsspec and pyarrow for real-data preparation"
        ) from exc
    url = (f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/"
           f"{DATASET_REVISION}/{DATASET_SHARD}")
    with fsspec.open(url, "rb", block_size=1 << 20, timeout=30) as file:
        source = parquet.ParquetFile(file)
        for batch in source.iter_batches(batch_size=64, columns=["id", "text"]):
            columns = batch.to_pydict()
            for document_id, content in zip(columns["id"], columns["text"], strict=True):
                yield {"id": document_id, "text": content}
