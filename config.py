from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class GPTConfig:
    vocab_size: int = 0
    context_length: int = 128
    embedding_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 2_000
    eval_interval: int = 100
    eval_batches: int = 20
    seed: int = 42
    train_split: float = 0.9
    data_path: str = "data/input.txt"
    checkpoint_dir: str = "checkpoints"

    def validate(self) -> None:
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if not 0 < self.train_split < 1:
            raise ValueError("train_split must be between 0 and 1")
        if self.context_length < 2:
            raise ValueError("context_length must be at least 2")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir) / "latest.pt"

