# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

This fork is no longer the original generic GPT pretraining harness. It is now a trading research repo with two paths:

- a structured baseline model in `train.py`
- a hybrid QLoRA LLM pipeline built around `Qwen/Qwen3.5-9B`

The repo still keeps the “small autonomous research loop” spirit of the original project, but the objective is now time-split trading validation instead of generic language-model loss.

## How it works

The core files now are:

- **`prepare.py`** — builds the structured market dataset from local raw data.
- **`train.py`** — trains the structured baseline model and exports a checkpoint.
- **`prepare_llm.py`** — builds the instruction dataset for the LLM pipeline.
- **`train_llm.py`** — fine-tunes `Qwen/Qwen3.5-9B` with QLoRA.
- **`evaluate_llm.py`** — evaluates structured baseline, text-only LLM, and hybrid LLM tracks.
- **`program.md`** — the repo-specific research policy.

The primary metric is **`validation_score`**, derived from:

- action precision
- action recall
- average trade return
- Sharpe-like proxy

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Build the structured market dataset
uv run prepare.py

# 4. Train the structured baseline
uv run train.py

# 5. Build the LLM instruction dataset
uv run prepare_llm.py

# 6. Fine-tune the LLM
uv run train_llm.py

# 7. Evaluate the hybrid pipeline
uv run evaluate_llm.py --track llm_hybrid
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py        — structured market dataset builder
train.py          — structured baseline trainer
prepare_llm.py    — LLM instruction dataset builder
train_llm.py      — Qwen3.5-9B QLoRA trainer
evaluate_llm.py   — LLM evaluation
program.md        — agent instructions
pyproject.toml    — dependencies
```

## Design choices

- **Baseline first.** The structured model remains the reference path and teacher.
- **Hybrid, not text-only.** The LLM consumes timestamp-safe text plus numeric/baseline context instead of replacing market structure.
- **Single-GPU target.** The repo targets one RTX 3090 24 GB, so the LLM path uses QLoRA instead of full fine-tuning.
- **Leakage-safe evaluation.** All comparisons remain time-based and backtest-aware.

## Platform support

This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. I'm not 100% sure that I want to take this on personally right now. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## License

MIT
