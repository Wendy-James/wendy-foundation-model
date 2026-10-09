"""Deterministic document tokenization and contiguous next-token batches."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Set

import torch

from wendyfm.tokenizer import BPETokenizer


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def prepare_token_ids(
    documents: Iterable[str],
    tokenizer: BPETokenizer,
    *,
    model_vocab_size: int,
    end_of_document: str | None = None,
) -> torch.Tensor:
    """Encode documents in input order, optionally appending an EOD ID to each.

    Documents must have a stable iteration order. Raw text is encoded without
    allowed special tokens. Boundaries are inserted
    as IDs, so a literal special-token string in a document cannot silently
    become a boundary. Use ``tokenizer.spec.config.vocab_size`` for the model
    capacity; ``tokenizer.vocab_size`` may omit reserved special-token IDs.
    """
    if not isinstance(tokenizer, BPETokenizer):
        raise TypeError("tokenizer must be a BPETokenizer")
    _positive_int(model_vocab_size, "model_vocab_size")
    tokenizer_capacity = tokenizer.spec.config.vocab_size
    if model_vocab_size < tokenizer_capacity:
        raise ValueError("model_vocab_size must cover the tokenizer's configured vocabulary")
    if isinstance(documents, (str, bytes, Mapping, Set)):
        raise TypeError("documents must be an ordered iterable of strings")

    boundary_id: int | None = None
    if end_of_document is not None:
        if not isinstance(end_of_document, str):
            raise TypeError("end_of_document must be a configured special-token string")
        try:
            boundary_id = tokenizer.spec.special_tokens[end_of_document]
        except KeyError as error:
            raise ValueError("end_of_document must be a configured special token") from error

    token_ids: list[int] = []
    document_count = 0
    for document in documents:
        if not isinstance(document, str):
            raise TypeError("each document must be a string")
        document_count += 1
        token_ids.extend(tokenizer.encode(document))
        if boundary_id is not None:
            token_ids.append(boundary_id)
    if document_count == 0 or not token_ids:
        raise ValueError("documents must produce at least one token")
    if any(token_id < 0 or token_id >= model_vocab_size for token_id in token_ids):
        raise ValueError("token IDs must fit model_vocab_size")
    return torch.tensor(token_ids, dtype=torch.long)


class NextTokenBatchSampler:
    """Draw reproducible contiguous windows from one CPU token stream.

    Each ``sample(batch_size)`` returns ``(input_ids, target_ids)`` of shape
    ``[batch_size, seq_len]``. Targets are shifted once here; pass them as
    explicit targets to ``model.next_token_loss(input_ids, target_ids)``.
    """

    def __init__(
        self,
        token_ids: torch.Tensor,
        *,
        seq_len: int,
        model_vocab_size: int,
        seed: int,
    ) -> None:
        _positive_int(seq_len, "seq_len")
        _positive_int(model_vocab_size, "model_vocab_size")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError("token_ids must be a torch.Tensor")
        if token_ids.ndim != 1 or token_ids.dtype != torch.long or token_ids.device.type != "cpu":
            raise ValueError("token_ids must be a one-dimensional CPU torch.long tensor")
        if token_ids.numel() < seq_len + 1:
            raise ValueError("corpus needs at least seq_len + 1 tokens")
        if bool(torch.any(token_ids < 0)) or bool(torch.any(token_ids >= model_vocab_size)):
            raise ValueError("token IDs must fit model_vocab_size")

        self.token_ids = token_ids
        self.seq_len = seq_len
        self._generator = torch.Generator(device="cpu").manual_seed(seed)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return independent, seeded windows with exactly one target shift."""
        _positive_int(batch_size, "batch_size")
        starts = torch.randint(
            self.token_ids.numel() - self.seq_len,
            (batch_size,),
            generator=self._generator,
        )
        offsets = torch.arange(self.seq_len + 1)
        windows = self.token_ids[starts[:, None] + offsets[None, :]]
        # Separate storage prevents a caller's in-place edit to inputs from
        # silently changing its targets (the slices otherwise overlap).
        return windows[:, :-1].clone(), windows[:, 1:].clone()
