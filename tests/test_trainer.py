from pathlib import Path

import torch

from config import GPTConfig
from model import GPT
from tokenizer import CharacterTokenizer
from training import LanguageModelData, Trainer


def test_trainer_saves_and_restores_checkpoint(tmp_path: Path) -> None:
    text = "the quick brown fox jumps over the lazy dog\n" * 20
    tokenizer = CharacterTokenizer.from_text(text)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        batch_size=2,
        max_steps=1,
        eval_interval=1,
        eval_batches=1,
        checkpoint_dir=str(tmp_path),
    )
    data = LanguageModelData(text, tokenizer, config.context_length)
    model = GPT(config)
    trainer = Trainer(model, data, tokenizer, config, torch.device("cpu"))

    trainer.train()

    assert config.checkpoint_path.exists()
    restored = Trainer(GPT(config), data, tokenizer, config, torch.device("cpu"))
    restored.load_checkpoint()
    assert restored.step == 1
    assert restored.history[0]["step"] == 0

