"""Tests for deterministic token streams and one-step-shifted batches."""

import pytest
import torch

from wendyfm.model import ModelConfig
from wendyfm.tokenizer import BPETokenizer
from wendyfm.training import NextTokenBatchSampler, prepare_token_ids


def _tokenizer() -> BPETokenizer:
    return BPETokenizer.train(["a", "b"], vocab_size=272, special_tokens=("<|eod|>",))


def test_document_order_and_explicit_boundaries() -> None:
    tokenizer = _tokenizer()
    documents = (text for text in ("a", "", "b"))
    ids = prepare_token_ids(
        documents,
        tokenizer,
        model_vocab_size=tokenizer.spec.config.vocab_size,
        end_of_document="<|eod|>",
    )
    eod = tokenizer.spec.special_tokens["<|eod|>"]
    expected = tokenizer.encode("a") + [eod, eod] + tokenizer.encode("b") + [eod]
    assert ids.dtype == torch.long
    assert ids.tolist() == expected
    assert eod == 271
    assert prepare_token_ids(
        ["a", "b"], tokenizer, model_vocab_size=272
    ).tolist() == tokenizer.encode("a") + tokenizer.encode("b")


def test_reserved_ids_require_configured_model_capacity() -> None:
    tokenizer = _tokenizer()
    assert tokenizer.vocab_size < tokenizer.spec.config.vocab_size
    config = ModelConfig(
        vocab_size=tokenizer.spec.config.vocab_size,
        d_model=16,
        n_heads=2,
        n_layers=1,
        max_seq_len=8,
        intermediate_size=32,
    )
    ids = prepare_token_ids(
        ["a", "b"], tokenizer, model_vocab_size=config.vocab_size, end_of_document="<|eod|>"
    )
    assert ids.max().item() < config.vocab_size
    assert ids.max().item() >= tokenizer.vocab_size
    with pytest.raises(ValueError, match="configured vocabulary"):
        prepare_token_ids(
            ["a"],
            tokenizer,
            model_vocab_size=tokenizer.vocab_size,
            end_of_document="<|eod|>",
        )


def test_seeded_sampling_shapes_dtype_and_shift() -> None:
    token_ids = torch.arange(20, dtype=torch.long)
    first = NextTokenBatchSampler(token_ids, seq_len=5, model_vocab_size=20, seed=42)
    second = NextTokenBatchSampler(token_ids, seq_len=5, model_vocab_size=20, seed=42)
    global_state = torch.get_rng_state().clone()
    for _ in range(3):
        inputs, targets = first.sample(4)
        repeated_inputs, repeated_targets = second.sample(4)
        assert inputs.shape == targets.shape == (4, 5)
        assert inputs.dtype == targets.dtype == torch.long
        torch.testing.assert_close(inputs, repeated_inputs)
        torch.testing.assert_close(targets, repeated_targets)
        torch.testing.assert_close(targets[:, :-1], inputs[:, 1:])
        torch.testing.assert_close(targets, inputs + 1)
    torch.testing.assert_close(torch.get_rng_state(), global_state)


def test_minimum_length_window_includes_final_target() -> None:
    sampler = NextTokenBatchSampler(
        torch.tensor([3, 4, 5, 6]), seq_len=3, model_vocab_size=7, seed=0
    )
    inputs, targets = sampler.sample(2)
    assert inputs.tolist() == [[3, 4, 5], [3, 4, 5]]
    assert targets.tolist() == [[4, 5, 6], [4, 5, 6]]
    inputs[0, 1] = 0
    assert targets[0].tolist() == [4, 5, 6]


def test_sampler_state_restores_next_draw_and_is_independent_of_returned_state() -> None:
    ids = torch.arange(20, dtype=torch.long)
    source = NextTokenBatchSampler(ids, seq_len=5, model_vocab_size=20, seed=42)
    source.sample(3)
    state = source.state_dict()
    restored = NextTokenBatchSampler(ids.clone(), seq_len=5, model_vocab_size=20, seed=999)
    restored.load_state_dict(state)
    state["generator_state"].zero_()
    for batch_size in (4, 2, 7):
        for actual, expected in zip(restored.sample(batch_size), source.sample(batch_size)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sampler_state_rejects_configuration_stream_and_corruption() -> None:
    ids = torch.arange(20, dtype=torch.long)
    source = NextTokenBatchSampler(ids, seq_len=5, model_vocab_size=20, seed=42)
    state = source.state_dict()
    variants = (
        (NextTokenBatchSampler(ids, seq_len=4, model_vocab_size=20, seed=1), "seq_len"),
        (NextTokenBatchSampler(ids, seq_len=5, model_vocab_size=21, seed=1), "model_vocab_size"),
        (NextTokenBatchSampler(ids[:-1], seq_len=5, model_vocab_size=20, seed=1), "token_count"),
        (NextTokenBatchSampler(ids.flip(0), seq_len=5, model_vocab_size=20, seed=1),
         "token stream"),
    )
    for sampler, message in variants:
        with pytest.raises(ValueError, match=message):
            sampler.load_state_dict(state)
    bad = {**state, "generator_state": torch.tensor([1], dtype=torch.uint8)}
    with pytest.raises(ValueError, match="generator state"):
        source.load_state_dict(bad)
    with pytest.raises(ValueError, match="token stream"):
        source.load_state_dict({**state, "token_sha256": torch.tensor([1, 2])})
    with pytest.raises(ValueError, match="sampler state"):
        source.load_state_dict({**state, "extra": 0})


@pytest.mark.parametrize(
    "documents,boundary,error",
    [
        ([], None, ValueError),
        ([""], None, ValueError),
        ([1], None, TypeError),
        ("abc", None, TypeError),
        ({"a", "b"}, None, TypeError),
        ({"a": 1}, None, TypeError),
        (["a"], "<|missing|>", ValueError),
        (["<|eod|>"], None, ValueError),
    ],
)
def test_invalid_document_inputs(documents, boundary, error) -> None:
    with pytest.raises(error):
        prepare_token_ids(
            documents, _tokenizer(), model_vocab_size=272, end_of_document=boundary
        )


@pytest.mark.parametrize("capacity", [0, True, 271])
def test_invalid_model_vocabulary_capacity(capacity) -> None:
    with pytest.raises(ValueError):
        prepare_token_ids(["a"], _tokenizer(), model_vocab_size=capacity)


@pytest.mark.parametrize(
    "token_ids,seq_len,vocab_size,seed,error",
    [
        (torch.tensor([0, 1]), 2, 3, 0, ValueError),
        (torch.tensor([0, 1, 2]), 0, 3, 0, ValueError),
        (torch.tensor([0, 1, 2]), 2, 0, 0, ValueError),
        (torch.tensor([0, 1, 2]), 2, 3, -1, ValueError),
        (torch.tensor([0, 1, 2]), 2, 3, True, ValueError),
        (torch.tensor([-1, 0, 1]), 2, 3, 0, ValueError),
        (torch.tensor([0, 1, 3]), 2, 3, 0, ValueError),
        (torch.tensor([[0, 1, 2]]), 2, 3, 0, ValueError),
        (torch.tensor([0.0, 1.0, 2.0]), 2, 3, 0, ValueError),
        ([0, 1, 2], 2, 3, 0, TypeError),
    ],
)
def test_sampler_rejects_invalid_corpus_and_settings(
    token_ids, seq_len, vocab_size, seed, error
) -> None:
    with pytest.raises(error):
        NextTokenBatchSampler(
            token_ids, seq_len=seq_len, model_vocab_size=vocab_size, seed=seed
        )


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_sampler_rejects_invalid_batch_size(batch_size) -> None:
    sampler = NextTokenBatchSampler(
        torch.tensor([0, 1, 2]), seq_len=2, model_vocab_size=3, seed=1
    )
    with pytest.raises(ValueError):
        sampler.sample(batch_size)
