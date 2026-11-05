from __future__ import annotations

from collections.abc import Iterable


class CharacterTokenizer:
    """A deterministic character-level tokenizer built from a text corpus."""

    def __init__(self, characters: Iterable[str]) -> None:
        self.characters = sorted(set(characters))
        if not self.characters:
            raise ValueError("Tokenizer vocabulary cannot be empty")
        self.stoi = {character: index for index, character in enumerate(self.characters)}
        self.itos = {index: character for character, index in self.stoi.items()}

    @classmethod
    def from_text(cls, text: str) -> CharacterTokenizer:
        return cls(text)

    @property
    def vocab_size(self) -> int:
        return len(self.characters)

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(text).difference(self.stoi))
        if unknown:
            rendered = ", ".join(repr(character) for character in unknown)
            raise ValueError(f"Text contains characters outside the vocabulary: {rendered}")
        return [self.stoi[character] for character in text]

    def decode(self, token_ids: Iterable[int]) -> str:
        ids = list(token_ids)
        unknown = sorted(set(ids).difference(self.itos))
        if unknown:
            raise ValueError(f"Unknown token IDs: {unknown}")
        return "".join(self.itos[token_id] for token_id in ids)

    def state_dict(self) -> dict[str, list[str]]:
        return {"characters": self.characters}

    @classmethod
    def from_state_dict(cls, state: dict[str, list[str]]) -> CharacterTokenizer:
        return cls(state["characters"])

