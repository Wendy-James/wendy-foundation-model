"""Public tokenizer facade."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from .spec import (
    PRETOKENIZATION_POLICY,
    SERIALIZATION_VERSION,
    MergeRule,
    TokenizerConfig,
    TokenizerSpec,
    pretokenize,
)
from .training import train_naive


class BPETokenizer:
    """A deterministic byte-level BPE tokenizer."""

    def __init__(self, spec: TokenizerSpec) -> None:
        self.spec = spec

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int,
        special_tokens: Sequence[str] = (),
    ) -> BPETokenizer:
        config = TokenizerConfig(vocab_size, tuple(special_tokens))
        return cls(train_naive(list(texts), config))

    def encode(self, text: str, *, allowed_special: Sequence[str] = ()) -> list[int]:
        allowed = set(allowed_special)
        configured = set(self.spec.config.special_tokens)
        unknown = allowed - configured
        if unknown:
            raise ValueError(f"unknown special tokens: {sorted(unknown)!r}")
        ids: list[int] = []
        special_ids = self.spec.special_tokens
        for piece in pretokenize(text, self.spec.config.special_tokens):
            if piece in special_ids:
                if piece not in allowed:
                    raise ValueError(f"special token is not allowed: {piece!r}")
                ids.append(special_ids[piece])
                continue
            ids.extend(self._encode_piece(piece))
        return ids

    def _encode_piece(self, piece: str) -> list[int]:
        sequence = list(piece.encode("utf-8"))
        for merge in self.spec.merges:
            sequence = _replace_pair(sequence, (merge.left_id, merge.right_id), merge.merged_id)
        return sequence

    def decode(self, ids: Sequence[int]) -> str:
        special_by_id = {token_id: text for text, token_id in self.spec.special_token_ids}
        output: list[str] = []
        byte_buffer = bytearray()
        for token_id in ids:
            if token_id in special_by_id:
                output.append(bytes(byte_buffer).decode("utf-8"))
                byte_buffer.clear()
                output.append(special_by_id[token_id])
                continue
            if not 0 <= token_id < len(self.spec.normal_vocabulary):
                raise ValueError(f"unknown token ID: {token_id}")
            byte_buffer.extend(self.spec.normal_vocabulary[token_id])
        output.append(bytes(byte_buffer).decode("utf-8"))
        return "".join(output)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.spec.serialization_dict(), ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("serialization_version") != SERIALIZATION_VERSION:
            raise ValueError("unsupported tokenizer serialization version")
        config_data = payload["config"]
        config = TokenizerConfig(config_data["vocab_size"], tuple(config_data["special_tokens"]))
        if payload["pretokenization"]["policy"] != PRETOKENIZATION_POLICY:
            raise ValueError("unsupported pre-tokenization policy")
        normal_vocabulary = tuple(bytes.fromhex(token) for token in payload["normal_vocabulary"])
        special_token_ids = tuple(
            (item["text"], item["id"]) for item in payload["special_tokens"]
        )
        merges = tuple(
            MergeRule(item["left_id"], item["right_id"], item["merged_id"])
            for item in payload["merges"]
        )
        return cls(TokenizerSpec(config, normal_vocabulary, merges, special_token_ids))

    @property
    def vocab_size(self) -> int:
        return len(self.spec.normal_vocabulary) + len(self.spec.special_token_ids)


def _replace_pair(sequence: list[int], pair: tuple[int, int], merged_id: int) -> list[int]:
    result: list[int] = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
            result.append(merged_id)
            index += 2
        else:
            result.append(sequence[index])
            index += 1
    return result
