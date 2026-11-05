import pytest

from tokenizer import CharacterTokenizer


def test_tokenizer_round_trip() -> None:
    tokenizer = CharacterTokenizer.from_text("hello world")
    text = "hello"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_tokenizer_state_round_trip() -> None:
    tokenizer = CharacterTokenizer.from_text("cab")
    restored = CharacterTokenizer.from_state_dict(tokenizer.state_dict())
    assert restored.characters == ["a", "b", "c"]


def test_tokenizer_rejects_unknown_character() -> None:
    tokenizer = CharacterTokenizer.from_text("abc")
    with pytest.raises(ValueError, match="outside the vocabulary"):
        tokenizer.encode("abcd")

