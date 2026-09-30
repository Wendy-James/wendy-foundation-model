import pytest

from wendyfm.tokenizer import BPETokenizer, TokenizerConfig, empty_spec


def test_public_api_exposes_small_stable_surface() -> None:
    tokenizer = BPETokenizer(empty_spec(TokenizerConfig(vocab_size=256)))
    assert tokenizer.vocab_size == 256
    for method in ("train", "encode", "decode", "save", "load"):
        assert callable(getattr(tokenizer, method, None))


@pytest.mark.parametrize("operation", ["encode", "decode", "save", "load"])
def test_unimplemented_runtime_operations_fail_explicitly(operation: str, tmp_path) -> None:
    tokenizer = BPETokenizer(empty_spec(TokenizerConfig(vocab_size=256)))
    with pytest.raises(NotImplementedError):
        if operation == "encode":
            tokenizer.encode("Hello")
        elif operation == "decode":
            tokenizer.decode([72])
        elif operation == "save":
            tokenizer.save(tmp_path / "tokenizer.json")
        else:
            BPETokenizer.load(tmp_path / "tokenizer.json")


def test_training_is_explicitly_not_implemented_yet() -> None:
    with pytest.raises(NotImplementedError):
        BPETokenizer.train(["hello hello"], vocab_size=260)
