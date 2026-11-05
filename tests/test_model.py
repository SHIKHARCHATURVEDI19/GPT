import torch

from config import GPTConfig
from model import GPT
from model.attention import CausalSelfAttention


def tiny_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=20,
        context_length=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )


def test_gpt_returns_logits_and_loss() -> None:
    model = GPT(tiny_config())
    inputs = torch.randint(0, 20, (3, 8))
    targets = torch.randint(0, 20, (3, 8))

    logits, loss = model(inputs, targets)

    assert logits.shape == (3, 8, 20)
    assert loss is not None and loss.ndim == 0
    loss.backward()


def test_attention_cannot_see_future_tokens() -> None:
    torch.manual_seed(7)
    attention = CausalSelfAttention(16, 4, 8, dropout=0.0).eval()
    original = torch.randn(1, 8, 16)
    changed = original.clone()
    changed[:, 4:] = torch.randn(1, 4, 16)

    original_output = attention(original)
    changed_output = attention(changed)

    assert torch.allclose(original_output[:, :4], changed_output[:, :4], atol=1e-6)


def test_embedding_and_output_weights_are_tied() -> None:
    model = GPT(tiny_config())
    assert model.embedding.token.weight.data_ptr() == model.language_head.weight.data_ptr()

