"""Straightforward deterministic byte-level BPE training."""

from __future__ import annotations

from collections import Counter, defaultdict

from .spec import BASE_VOCAB_SIZE, MergeRule, TokenizerConfig, TokenizerSpec, pretokenize


def train_naive(
    texts: list[str], config: TokenizerConfig
) -> TokenizerSpec:
    """Train BPE by recounting every adjacent pair after each merge."""

    vocabulary = _base_vocabulary()
    pieces = _prepare_sequences(texts, config)

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

    return _make_spec(config, vocabulary, merges)


def train_optimized(texts: list[str], config: TokenizerConfig) -> TokenizerSpec:
    """Train BPE with incremental pair counts and an inverted sequence index."""

    vocabulary = _base_vocabulary()
    sequences = _prepare_sequences(texts, config)
    sequence_counts = [_pair_counts(sequence) for sequence in sequences]
    global_counts: Counter[tuple[int, int]] = Counter()
    pair_sequences: dict[tuple[int, int], set[int]] = defaultdict(set)
    for sequence_id, counts in enumerate(sequence_counts):
        global_counts.update(counts)
        for pair in counts:
            pair_sequences[pair].add(sequence_id)

    merges: list[MergeRule] = []
    while len(merges) < config.max_learned_tokens and global_counts:
        pair, _ = min(global_counts.items(), key=lambda item: (-item[1], item[0]))
        affected = tuple(pair_sequences.get(pair, ()))
        left_id, right_id = pair
        merged_id = BASE_VOCAB_SIZE + len(merges)
        vocabulary.append(vocabulary[left_id] + vocabulary[right_id])
        merges.append(MergeRule(left_id, right_id, merged_id))

        for sequence_id in affected:
            old_counts = sequence_counts[sequence_id]
            _remove_sequence_counts(global_counts, pair_sequences, sequence_id, old_counts)
            sequences[sequence_id] = _replace_pair(sequences[sequence_id], pair, merged_id)
            new_counts = _pair_counts(sequences[sequence_id])
            sequence_counts[sequence_id] = new_counts
            _add_sequence_counts(global_counts, pair_sequences, sequence_id, new_counts)

    return _make_spec(config, vocabulary, merges)


def _base_vocabulary() -> list[bytes]:
    return [bytes([value]) for value in range(BASE_VOCAB_SIZE)]


def _prepare_sequences(texts: list[str], config: TokenizerConfig) -> list[list[int]]:
    sequences: list[list[int]] = []
    for text in texts:
        for piece in pretokenize(text, config.special_tokens):
            if piece not in config.special_tokens:
                sequences.append(list(piece.encode("utf-8")))
    return sequences


def _make_spec(
    config: TokenizerConfig, vocabulary: list[bytes], merges: list[MergeRule]
) -> TokenizerSpec:
    special_start = config.vocab_size - len(config.special_tokens)
    special_ids = tuple(
        (token, special_start + offset) for offset, token in enumerate(config.special_tokens)
    )
    return TokenizerSpec(config, tuple(vocabulary), tuple(merges), special_ids)


def _pair_counts(sequence: list[int]) -> Counter[tuple[int, int]]:
    return Counter(zip(sequence, sequence[1:]))


def _remove_sequence_counts(
    global_counts: Counter[tuple[int, int]],
    pair_sequences: dict[tuple[int, int], set[int]],
    sequence_id: int,
    counts: Counter[tuple[int, int]],
) -> None:
    for pair, count in counts.items():
        global_counts[pair] -= count
        pair_sequences[pair].discard(sequence_id)
        if global_counts[pair] <= 0:
            del global_counts[pair]
            if not pair_sequences[pair]:
                del pair_sequences[pair]


def _add_sequence_counts(
    global_counts: Counter[tuple[int, int]],
    pair_sequences: dict[tuple[int, int], set[int]],
    sequence_id: int,
    counts: Counter[tuple[int, int]],
) -> None:
    for pair, count in counts.items():
        global_counts[pair] += count
        pair_sequences[pair].add(sequence_id)


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
