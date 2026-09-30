"""Straightforward deterministic byte-level BPE training."""

from __future__ import annotations

from collections import Counter

from .spec import BASE_VOCAB_SIZE, MergeRule, TokenizerConfig, TokenizerSpec, pretokenize


def train_naive(
    texts: list[str], config: TokenizerConfig
) -> TokenizerSpec:
    """Train BPE by recounting every adjacent pair after each merge."""

    vocabulary = [bytes([value]) for value in range(BASE_VOCAB_SIZE)]
    pieces: list[list[int]] = []
    for text in texts:
        for piece in pretokenize(text, config.special_tokens):
            if piece in config.special_tokens:
                continue
            pieces.append(list(piece.encode("utf-8")))

    merges: list[MergeRule] = []
    while len(merges) < config.max_learned_tokens:
        counts: Counter[tuple[int, int]] = Counter()
        for sequence in pieces:
            counts.update(zip(sequence, sequence[1:]))
        if not counts:
            break
        pair, frequency = min(counts.items(), key=lambda item: (-item[1], item[0]))
        del frequency
        left_id, right_id = pair
        merged_id = BASE_VOCAB_SIZE + len(merges)
        vocabulary.append(vocabulary[left_id] + vocabulary[right_id])
        merges.append(MergeRule(left_id, right_id, merged_id))
        pieces = [_replace_pair(sequence, pair, merged_id) for sequence in pieces]

    special_start = config.vocab_size - len(config.special_tokens)
    special_ids = tuple(
        (token, special_start + offset) for offset, token in enumerate(config.special_tokens)
    )
    return TokenizerSpec(config, tuple(vocabulary), tuple(merges), special_ids)


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
