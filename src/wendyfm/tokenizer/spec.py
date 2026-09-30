"""Small, explicit data contracts for WendyFM's byte-level BPE tokenizer."""

from __future__ import annotations

import re
from dataclasses import dataclass

BASE_VOCAB_SIZE = 256
SERIALIZATION_VERSION = 1
PRETOKENIZATION_POLICY = "unicode_runs_v1"
_PRETOKENIZE_RE = re.compile(r"\s+|[\w]+|[^\w\s]+", re.UNICODE)


@dataclass(frozen=True)
class TokenizerConfig:
    """Configuration whose vocab_size is the final total vocabulary size."""

    vocab_size: int
    special_tokens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.vocab_size < BASE_VOCAB_SIZE + len(self.special_tokens):
            raise ValueError("vocab_size cannot fit base and special-token vocabularies")
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special_tokens must be unique")
        if any(not token for token in self.special_tokens):
            raise ValueError("special_tokens cannot contain empty strings")

    @property
    def max_learned_tokens(self) -> int:
        return self.vocab_size - BASE_VOCAB_SIZE - len(self.special_tokens)


@dataclass(frozen=True)
class MergeRule:
    """One ordered merge and the ID assigned when it was created."""

    left_id: int
    right_id: int
    merged_id: int


@dataclass(frozen=True)
class TokenizerSpec:
    """Complete tokenizer state with normal and special tokens kept distinct."""

    config: TokenizerConfig
    normal_vocabulary: tuple[bytes, ...]
    merges: tuple[MergeRule, ...]
    special_token_ids: tuple[tuple[str, int], ...]
    pretokenization_policy: str = PRETOKENIZATION_POLICY

    def __post_init__(self) -> None:
        if self.normal_vocabulary[:BASE_VOCAB_SIZE] != tuple(
            bytes([i]) for i in range(BASE_VOCAB_SIZE)
        ):
            raise ValueError("normal IDs 0..255 must be the canonical byte vocabulary")
        if len(self.normal_vocabulary) > self.config.vocab_size - len(self.config.special_tokens):
            raise ValueError("normal vocabulary exceeds the configured vocabulary range")
        if any(not token for token in self.normal_vocabulary):
            raise ValueError("normal vocabulary cannot contain empty-byte placeholders")
        if tuple(token for token, _ in self.special_token_ids) != self.config.special_tokens:
            raise ValueError("special-token ordering must match configuration")
        expected = range(
            self.config.vocab_size - len(self.config.special_tokens), self.config.vocab_size
        )
        if tuple(token_id for _, token_id in self.special_token_ids) != tuple(expected):
            raise ValueError("special tokens must occupy the final vocabulary IDs")
        if len(self.merges) > self.config.max_learned_tokens:
            raise ValueError("merge count exceeds the configured learned-token capacity")
        for offset, merge in enumerate(self.merges):
            expected_id = BASE_VOCAB_SIZE + offset
            if merge.merged_id != expected_id:
                raise ValueError("learned token IDs must follow merge creation order")
            if not (0 <= merge.left_id < merge.merged_id):
                raise ValueError("merge left ID must reference an earlier normal token")
            if not (0 <= merge.right_id < merge.merged_id):
                raise ValueError("merge right ID must reference an earlier normal token")

    @property
    def special_tokens(self) -> dict[str, int]:
        return dict(self.special_token_ids)

    def serialization_dict(self) -> dict[str, object]:
        return {
            "serialization_version": SERIALIZATION_VERSION,
            "config": {
                "vocab_size": self.config.vocab_size,
                "special_tokens": list(self.config.special_tokens),
            },
            "pretokenization": {"policy": self.pretokenization_policy},
            "normal_vocabulary": [token.hex() for token in self.normal_vocabulary],
            "special_tokens": [
                {"text": text, "id": token_id} for text, token_id in self.special_token_ids
            ],
            "merges": [
                {
                    "left_id": merge.left_id,
                    "right_id": merge.right_id,
                    "merged_id": merge.merged_id,
                }
                for merge in self.merges
            ],
        }


def pretokenize(text: str, special_tokens: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Split text into deterministic regions while preserving every input character."""

    if not text:
        return ()
    if not special_tokens:
        return tuple(match.group(0) for match in _PRETOKENIZE_RE.finditer(text))
    ordered_tokens = sorted(enumerate(special_tokens), key=lambda item: (-len(item[1]), item[0]))
    special_pattern = re.compile("|".join(re.escape(token) for _, token in ordered_tokens))
    pieces: list[str] = []
    cursor = 0
    for match in special_pattern.finditer(text):
        pieces.extend(pretokenize(text[cursor : match.start()]))
        pieces.append(match.group(0))
        cursor = match.end()
    pieces.extend(pretokenize(text[cursor:]))
    return tuple(pieces)


def empty_spec(config: TokenizerConfig) -> TokenizerSpec:
    """Create the byte-only initial state; learned tokens do not exist yet."""

    special_start = config.vocab_size - len(config.special_tokens)
    special_ids = tuple(
        (token, special_start + offset) for offset, token in enumerate(config.special_tokens)
    )
    return TokenizerSpec(
        config=config,
        normal_vocabulary=tuple(bytes([i]) for i in range(BASE_VOCAB_SIZE)),
        merges=(),
        special_token_ids=special_ids,
    )
