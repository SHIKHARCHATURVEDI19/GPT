from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from tqdm import trange

from config import GPTConfig
from model import GPT
from tokenizer import CharacterTokenizer
from .dataset import LanguageModelData


class Trainer:
    def __init__(
        self,
        model: GPT,
        data: LanguageModelData,
        tokenizer: CharacterTokenizer,
        config: GPTConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.step = 0
        self.history: list[dict[str, float | int]] = []

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        self.model.eval()
        losses: dict[str, float] = {}
        for split in ("train", "validation"):
            split_losses = []
            for _ in range(self.config.eval_batches):
                inputs, targets = self.data.get_batch(
                    split, self.config.batch_size, self.device
                )
                _, loss = self.model(inputs, targets)
                assert loss is not None
                split_losses.append(loss.item())
            losses[split] = sum(split_losses) / len(split_losses)
        self.model.train()
        return losses

    def train(self) -> list[dict[str, float | int]]:
        self.model.train()
        progress = trange(self.step, self.config.max_steps, desc="training", unit="step")
        for step in progress:
            self.step = step
            if step % self.config.eval_interval == 0 or step == self.config.max_steps - 1:
                losses = self.estimate_loss()
                record: dict[str, float | int] = {"step": step, **losses}
                self.history.append(record)
                progress.set_postfix(
                    train=f"{losses['train']:.3f}", val=f"{losses['validation']:.3f}"
                )

            inputs, targets = self.data.get_batch(
                "train", self.config.batch_size, self.device
            )
            _, loss = self.model(inputs, targets)
            assert loss is not None
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.step = step + 1

        self.save_checkpoint()
        return self.history

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        checkpoint_path = Path(path) if path else self.config.checkpoint_path
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.to_dict(),
                "tokenizer": self.tokenizer.state_dict(),
                "step": self.step,
                "history": self.history,
            },
            checkpoint_path,
        )
        history_path = checkpoint_path.with_name("history.json")
        history_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        return checkpoint_path

    def load_checkpoint(self, path: str | Path | None = None) -> None:
        checkpoint_path = Path(path) if path else self.config.checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step = int(checkpoint["step"])
        self.history = checkpoint.get("history", [])

