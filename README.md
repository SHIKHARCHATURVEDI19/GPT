# GPT From Scratch

A small, readable GPT-style language model built with PyTorch. The project implements tokenization, causal multi-head self-attention, Transformer blocks, training, checkpointing, and autoregressive text generation without using a prebuilt Transformer library.

## Architecture

```text
text -> character tokenizer -> token + position embeddings
     -> causal Transformer blocks -> layer norm -> vocabulary logits
     -> sampled next token
```

The default model is intentionally compact enough for learning and local experiments. For a quick CPU smoke test, override the training options from the command line.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Replace `data/input.txt` with a larger UTF-8 text corpus for useful results.

## Train

```bash
python train.py
```

Quick smoke run:

```bash
python train.py --max-steps 10 --eval-interval 5 --batch-size 4
```

Training writes `checkpoints/latest.pt`. Resume with:

```bash
python train.py --resume
```

## Generate

```bash
python -m inference.generate --prompt "The " --max-new-tokens 200
```

## Test

```bash
pytest -q
```

## Learning roadmap

1. Character tokenization and sliding-window datasets
2. Causal self-attention and Transformer blocks
3. GPT language model and loss calculation
4. Training, validation, and checkpointing
5. Temperature and top-k text generation

