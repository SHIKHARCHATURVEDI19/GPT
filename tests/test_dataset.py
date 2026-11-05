import torch

from tokenizer import CharacterTokenizer
from training import LanguageModelData


def test_batch_has_shifted_targets() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 10
    tokenizer = CharacterTokenizer.from_text(text)
    dataset = LanguageModelData(text, tokenizer, context_length=8, train_split=0.8)

    inputs, targets = dataset.get_batch("train", batch_size=4)

    assert inputs.shape == (4, 8)
    assert targets.shape == (4, 8)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])

