from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from config import GPTConfig
from model import GPT
from tokenizer import CharacterTokenizer


@torch.no_grad()
def generate_tokens(
    model: GPT,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Autoregressively append tokens to a batch of prompt token IDs."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if temperature < 0:
        raise ValueError("temperature cannot be negative")

    model.eval()
    for _ in range(max_new_tokens):
        context = token_ids[:, -model.config.context_length :]
        logits, _ = model(context)
        next_token_logits = logits[:, -1, :]

        if temperature == 0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            next_token_logits = next_token_logits / temperature
            if top_k is not None:
                k = min(top_k, next_token_logits.size(-1))
                if k <= 0:
                    raise ValueError("top_k must be positive")
                threshold = torch.topk(next_token_logits, k).values[:, -1, None]
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < threshold, float("-inf")
                )
            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
        token_ids = torch.cat((token_ids, next_token), dim=1)
    return token_ids


def load_model(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[GPT, CharacterTokenizer]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["config"])
    tokenizer = CharacterTokenizer.from_state_dict(checkpoint["tokenizer"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with a trained GPT checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--prompt", default="The ")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
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
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.checkpoint, device)
    prompt_ids = torch.tensor(
        [tokenizer.encode(args.prompt)], dtype=torch.long, device=device
    )
    output_ids = generate_tokens(
        model,
        prompt_ids,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
    )
    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()

