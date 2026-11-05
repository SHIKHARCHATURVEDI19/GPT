from pathlib import Path

import torch

from config import GPTConfig
from inference.generate import generate_tokens, load_model
from model import GPT
from tokenizer import CharacterTokenizer


def test_generation_appends_requested_number_of_tokens() -> None:
    config = GPTConfig(
        vocab_size=8,
        context_length=4,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )
    model = GPT(config)
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6]])

    generated = generate_tokens(model, prompt, max_new_tokens=3, temperature=0)

    assert generated.shape == (1, 9)
    assert torch.equal(generated[:, :6], prompt)


def test_checkpoint_loader_restores_model_and_tokenizer(tmp_path: Path) -> None:
    tokenizer = CharacterTokenizer.from_text("abcd")
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=4,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )
    model = GPT(config)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.to_dict(),
            "tokenizer": tokenizer.state_dict(),
        },
        checkpoint,
    )

    loaded_model, loaded_tokenizer = load_model(checkpoint, torch.device("cpu"))

    assert loaded_model.config.context_length == 4
    assert loaded_tokenizer.decode([0, 1, 2, 3]) == "abcd"

