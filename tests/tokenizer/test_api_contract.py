import pytest

from wendyfm.tokenizer import BPETokenizer, TokenizerConfig, empty_spec, pretokenize


def test_public_api_exposes_small_stable_surface() -> None:
    tokenizer = BPETokenizer(empty_spec(TokenizerConfig(vocab_size=256)))
    assert tokenizer.vocab_size == 256
    for method in ("train", "encode", "decode", "save", "load"):
        assert callable(getattr(tokenizer, method, None))


@pytest.mark.parametrize(
    "text", ["", "Hello, world!", "你好，世界", "🙂🚀", "x = 1\n\tprint(x)", "a  a"]
)
def test_encode_decode_round_trip(text: str) -> None:
    tokenizer = BPETokenizer.train([text, text], vocab_size=270)
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_tiny_training_has_expected_merge_order_and_bytes() -> None:
    tokenizer = BPETokenizer.train(["abab ab"], vocab_size=257)
    assert tokenizer.spec.merges == (tokenizer.spec.merges[0],)
    merge = tokenizer.spec.merges[0]
    assert (merge.left_id, merge.right_id, merge.merged_id) == (ord("a"), ord("b"), 256)
    assert tokenizer.spec.normal_vocabulary[256] == b"ab"


def test_tie_breaking_uses_lexicographically_smallest_pair() -> None:
    tokenizer = BPETokenizer.train(["ab cd"], vocab_size=257)
    assert (tokenizer.spec.merges[0].left_id, tokenizer.spec.merges[0].right_id) == (
        ord("a"),
        ord("b"),
    )


def test_training_stops_at_capacity_and_without_pairs() -> None:
    assert len(BPETokenizer.train(["abab"], vocab_size=257).spec.merges) == 1
    tokenizer = BPETokenizer.train(["a b"], vocab_size=270)
    assert tokenizer.spec.merges == ()
    assert tokenizer.vocab_size == 256
    assert tokenizer.spec.config.vocab_size == 270


def test_special_tokens_and_overlapping_tokens_are_atomic() -> None:
    tokenizer = BPETokenizer.train(
        ["x<|a|>y", "x<|ab|>y"],
        vocab_size=270,
        special_tokens=("<|a|>", "<|ab|>"),
    )
    text = "x<|ab|>y"
    ids = tokenizer.encode(text, allowed_special=("<|a|>", "<|ab|>"))
    assert ids[1] == tokenizer.spec.special_tokens["<|ab|>"]
    assert tokenizer.decode(ids) == text
    assert pretokenize("x<|ab|>y", ("<|a|>", "<|ab|>")) == ("x", "<|ab|>", "y")


def test_configured_special_token_requires_explicit_permission() -> None:
    tokenizer = BPETokenizer.train(["hello"], vocab_size=270, special_tokens=("<eos>",))
    with pytest.raises(ValueError, match="not allowed"):
        tokenizer.encode("hello<eos>")


def test_every_learned_token_reconstructs_from_its_merge_rule() -> None:
    tokenizer = BPETokenizer.train(["banana bandana"], vocab_size=270)
    vocabulary = tokenizer.spec.normal_vocabulary
    for merge in tokenizer.spec.merges:
        assert vocabulary[merge.merged_id] == vocabulary[merge.left_id] + vocabulary[merge.right_id]


def test_save_load_preserves_behavior(tmp_path) -> None:
    tokenizer = BPETokenizer.train(
        ["hello hello", "你好🙂"], vocab_size=270, special_tokens=("<eos>",)
    )
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = BPETokenizer.load(path)
    for text in ("hello hello", "你好🙂<eos>"):
        assert loaded.decode(loaded.encode(text, allowed_special=("<eos>",))) == text
    assert loaded.spec == tokenizer.spec


def test_deterministic_repeated_training() -> None:
    first = BPETokenizer.train(["banana bandana", "香蕉 banana"], vocab_size=280)
    second = BPETokenizer.train(["banana bandana", "香蕉 banana"], vocab_size=280)
    assert first.spec == second.spec


def test_invalid_token_id_is_rejected() -> None:
    tokenizer = BPETokenizer.train(["hello"], vocab_size=260)
    with pytest.raises(ValueError):
        tokenizer.decode([9999])
