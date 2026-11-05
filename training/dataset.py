from __future__ import annotations

from pathlib import Path

import torch

from tokenizer import CharacterTokenizer


class LanguageModelData:
    """Tokenized corpus with random next-token training batches."""

    def __init__(
        self,
        text: str,
        tokenizer: CharacterTokenizer,
        context_length: int,
        train_split: float = 0.9,
    ) -> None:
        if len(text) <= context_length:
            raise ValueError(
                f"Corpus must contain more than context_length={context_length} characters"
            )
        tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        split_index = int(len(tokens) * train_split)
        self.train = tokens[:split_index]
        self.validation = tokens[split_index:]
        self.context_length = context_length

        for name, split in (("train", self.train), ("validation", self.validation)):
            if len(split) <= context_length:
                raise ValueError(
                    f"The {name} split has {len(split)} tokens; it needs more than "
                    f"context_length={context_length}. Use more data or a shorter context."
                )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        context_length: int,
        train_split: float = 0.9,
    ) -> tuple[LanguageModelData, CharacterTokenizer]:
        text = Path(path).read_text(encoding="utf-8")
        tokenizer = CharacterTokenizer.from_text(text)
        return cls(text, tokenizer, context_length, train_split), tokenizer

    def get_batch(
        self,
        split: str,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train if split == "train" else self.validation
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")

        starts = torch.randint(len(data) - self.context_length, (batch_size,))
        inputs = torch.stack([data[i : i + self.context_length] for i in starts])
        targets = torch.stack([data[i + 1 : i + self.context_length + 1] for i in starts])
        return inputs.to(device), targets.to(device)

