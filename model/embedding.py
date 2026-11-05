import torch
from torch import nn


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, embedding_dim: int) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab_size, embedding_dim)
        self.position = nn.Embedding(context_length, embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, sequence_length = token_ids.shape
        if sequence_length > self.position.num_embeddings:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds model context length "
                f"{self.position.num_embeddings}"
            )
        positions = torch.arange(sequence_length, device=token_ids.device)
        return self.token(token_ids) + self.position(positions)

