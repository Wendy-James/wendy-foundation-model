import json

import pytest

from wendyfm.tokenizer import (
    BASE_VOCAB_SIZE,
    PRETOKENIZATION_POLICY,
    SERIALIZATION_VERSION,
    MergeRule,
    TokenizerConfig,
    TokenizerSpec,
    empty_spec,
    pretokenize,
)


@pytest.mark.parametrize("text", ["", "Hello, world!", "你好，世界", "🙂🚀", "x = 1\n\tprint(x)"])
def test_representative_text_is_preserved_by_pretokenization(text: str) -> None:
    assert "".join(pretokenize(text)) == text


def test_base_ids_are_exactly_all_bytes_and_no_learned_placeholders_exist() -> None:
    spec = empty_spec(TokenizerConfig(vocab_size=260))
    assert spec.normal_vocabulary == tuple(bytes([i]) for i in range(BASE_VOCAB_SIZE))
    assert all(spec.normal_vocabulary)


def test_vocab_size_includes_special_tokens_and_learned_capacity() -> None:
    config = TokenizerConfig(vocab_size=1000, special_tokens=("<|bos|>", "<|eos|>"))
    assert config.max_learned_tokens == 742
    spec = empty_spec(config)
    assert spec.special_tokens == {"<|bos|>": 998, "<|eos|>": 999}
    assert all(token_id not in spec.special_tokens.values() for token_id in range(256))


def test_invalid_minimum_vocab_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        TokenizerConfig(vocab_size=257, special_tokens=("<x>", "<y>"))


def test_special_tokens_are_deterministic_and_atomic() -> None:
    assert pretokenize("a<|bos|>你好", ("<|bos|>",)) == ("a", "<|bos|>", "你好")


def test_pretokenization_has_whitespace_word_and_symbol_boundaries() -> None:
    assert pretokenize("hello 你好\n==") == ("hello", " ", "你好", "\n", "==")


def test_pretokenization_policy_is_explicitly_not_gpt2_compatible() -> None:
    assert PRETOKENIZATION_POLICY == "unicode_runs_v1"


def test_serialization_schema_is_versioned_and_contains_reconstructing_state() -> None:
    config = TokenizerConfig(vocab_size=258, special_tokens=("<|eot|>", "<|pad|>"))
    payload = empty_spec(config).serialization_dict()
    json.dumps(payload, ensure_ascii=False, sort_keys=True)
    assert payload["serialization_version"] == SERIALIZATION_VERSION
    assert payload["config"] == {"vocab_size": 258, "special_tokens": ["<|eot|>", "<|pad|>"]}
    assert payload["pretokenization"] == {"policy": PRETOKENIZATION_POLICY}
    assert len(payload["normal_vocabulary"]) == BASE_VOCAB_SIZE
    assert payload["special_tokens"][-1] == {"text": "<|pad|>", "id": 257}
    assert payload["merges"] == []


def test_invalid_learned_id_order_is_rejected() -> None:
    config = TokenizerConfig(vocab_size=258)
    with pytest.raises(ValueError):
        TokenizerSpec(
            config=config,
            normal_vocabulary=tuple(bytes([i]) for i in range(BASE_VOCAB_SIZE)) + (b"ab",),
            merges=(MergeRule(1, 2, 257),),
            special_token_ids=(),
        )
