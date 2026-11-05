# GPT From Scratch

A small, readable GPT-style language model built with PyTorch. The project implements tokenization, causal multi-head self-attention, Transformer blocks, training, checkpointing, and autoregressive text generation without using a prebuilt Transformer library.

[![tests](https://github.com/SHIKHARCHATURVEDI19/GPT/actions/workflows/tests.yml/badge.svg)](https://github.com/SHIKHARCHATURVEDI19/GPT/actions/workflows/tests.yml)

## Architecture

```text
text -> character tokenizer -> token + position embeddings
     -> causal Transformer blocks -> layer norm -> vocabulary logits
     -> sampled next token
```

The default model is intentionally compact enough for learning and local experiments. For a quick CPU smoke test, override the training options from the command line.

## Project layout

```text
data/           training corpus
tokenizer/      character vocabulary and encode/decode logic
model/          embeddings, attention, Transformer blocks, and GPT
training/       random batches, evaluation, optimizer, and checkpoints
inference/      checkpoint loading and autoregressive sampling
tests/          unit and end-to-end smoke tests
config.py       model and training defaults
train.py        training CLI
```

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

Sampling controls:

```bash
python -m inference.generate \
  --prompt "Learning " \
  --temperature 0.8 \
  --top-k 20 \
  --max-new-tokens 300
```

Use `--temperature 0` for deterministic greedy decoding. A one-step smoke-test checkpoint only proves that the pipeline works; coherent output requires a larger corpus and substantially more training.

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

## Design decisions and trade-offs

- **Character tokens:** transparent and dependency-free, but sequences are longer than with subword tokenization.
- **Manual attention:** exposes masking and tensor shapes clearly, but optimized attention kernels are faster.
- **Learned positions:** simple and effective within the configured context window, but cannot extrapolate beyond it.
- **Weight tying:** the token embedding and output projection share parameters, reducing model size.
- **Random batches:** ideal for a compact teaching project; production training usually uses streamed datasets and distributed workers.
- **Full checkpoints:** model, optimizer, tokenizer, configuration, step, and loss history are stored together so a run can be reproduced or resumed.

## Configuration

Defaults live in `config.py`: context length 128, embedding width 256, 8 heads, and 6 Transformer blocks. All important smoke-test settings can be overridden from `train.py --help`. The embedding width must be divisible by the number of heads.

## What to try next

1. Train on a larger public-domain text corpus.
2. Compare character tokens with byte-pair encoding.
3. Add a learning-rate warmup and cosine decay schedule.
4. Benchmark manual attention against PyTorch scaled-dot-product attention.
5. Track perplexity and generated samples across checkpoints.
