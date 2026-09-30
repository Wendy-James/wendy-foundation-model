"""Public tokenizer facade for the M1.1 specification scaffold."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from .spec import TokenizerConfig, TokenizerSpec


class BPETokenizer:
    """Small public API for the forthcoming deterministic byte-level BPE tokenizer."""

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
        del texts, vocab_size, special_tokens
        raise NotImplementedError("BPE training is scheduled for M1.2")

    def encode(self, text: str, *, allowed_special: Sequence[str] = ()) -> list[int]:
        del text, allowed_special
        raise NotImplementedError("encoding is scheduled for M1.2")

    def decode(self, ids: Sequence[int]) -> str:
        del ids
        raise NotImplementedError("decoding is scheduled for M1.2")

    def save(self, path: str | Path) -> None:
        del path
        raise NotImplementedError("serialization is scheduled for M1.2")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        del path
        raise NotImplementedError("serialization is scheduled for M1.2")

    @property
    def vocab_size(self) -> int:
        return self.spec.config.vocab_size


__all__ = ["BPETokenizer", "TokenizerConfig", "TokenizerSpec"]
