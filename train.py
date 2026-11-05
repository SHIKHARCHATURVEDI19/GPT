from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from config import GPTConfig
from model import GPT
from training import LanguageModelData, Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small GPT model from scratch")
    parser.add_argument("--data", default="data/input.txt")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    config = GPTConfig(data_path=args.data)
    for argument, field in (
        (args.max_steps, "max_steps"),
        (args.batch_size, "batch_size"),
        (args.context_length, "context_length"),
        (args.embedding_dim, "embedding_dim"),
        (args.num_heads, "num_heads"),
        (args.num_layers, "num_layers"),
        (args.learning_rate, "learning_rate"),
        (args.eval_interval, "eval_interval"),
        (args.eval_batches, "eval_batches"),
    ):
        if argument is not None:
            setattr(config, field, argument)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    data, tokenizer = LanguageModelData.from_file(
        config.data_path, config.context_length, config.train_split
    )
    config.vocab_size = tokenizer.vocab_size
    config.validate()
    device = resolve_device(args.device)
    model = GPT(config)
    trainer = Trainer(model, data, tokenizer, config, device)
    if args.resume:
        trainer.load_checkpoint()

    print(f"device={device} parameters={model.parameter_count():,} vocab={config.vocab_size}")
    trainer.train()
    print(f"checkpoint={config.checkpoint_path}")


if __name__ == "__main__":
    main()

