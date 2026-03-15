# autoresearch trading fork

This repository now runs autonomous research for a **hybrid swing-trading stack**:

- a structured numeric baseline in `train.py`
- a QLoRA LLM pipeline in `prepare_llm.py`, `train_llm.py`, and `evaluate_llm.py`

## Goal

Improve the model’s **validation score** for US large caps and liquid ETFs over a 1-5 day horizon.

The model should output:
- structured trade fields
- a short rationale
- an optional options plan selected by a deterministic wrapper

The primary objective is not language loss. The optimization target is:
- action precision
- action recall
- average trade return
- Sharpe-like proxy

## Scope

- Equity-first prediction
- Options are a wrapper over strong equity signals
- Research-only decision support
- Single GPU, intended for Linux + RTX 3090 24 GB

## Files In Scope

- `prepare.py`
  Builds the market dataset, programmatic labels, split metadata, and options-selection helpers.
- `train.py`
  Defines the structured baseline model, multitask losses, training loop, checkpoint export, and prediction artifact export.
- `prepare_llm.py`
  Builds the LLM instruction dataset from market rows plus timestamp-safe text evidence.
- `train_llm.py`
  Runs QLoRA supervised fine-tuning for `Qwen/Qwen3.5-9B`.
- `evaluate_llm.py`
  Evaluates structured baseline, text-only LLM, and hybrid LLM tracks using the current validation contract.
- `program.md`
  Defines the autonomous experimentation policy.

## Setup

1. Read `README.md`, `prepare.py`, `train.py`, and this file.
2. Verify market data exists under `~/.cache/autoresearch/market/raw/`.
3. If raw data is missing and you only need a smoke test, run:
   `uv run prepare.py --generate-demo-data`
4. Build the dataset:
   `uv run prepare.py`
5. Confirm `~/.cache/autoresearch/market/dataset/market_dataset.parquet` exists.
6. Run a baseline training pass:
   `uv run train.py`

## Market Data Contract

Required:
- `raw/stocks/<SYMBOL>.csv`
- columns: `timestamp,open,high,low,close,volume`

Optional:
- `raw/news/<SYMBOL>.csv`
- columns: `timestamp,headline,summary`

Optional:
- `raw/options/<SYMBOL>.csv`
- columns:
  `timestamp,contract,option_type,strike,expiration,delta,implied_volatility,bid,ask,open_interest,volume`

## Evaluation Contract

Always optimize for `validation_score`, not for raw loss alone.

The score is derived from:
- `val_action_precision`
- `val_action_recall`
- `val_avg_trade_return`
- `val_sharpe_proxy`

Prefer simpler changes when scores are close.

For the LLM pipeline, also enforce:
- JSON parse success >= 99%
- schema-valid outputs >= 99%
- no `<think>` tags or non-JSON preambles in accepted outputs

## Experimentation Rules

- Keep the code runnable on a 24 GB GPU.
- Favor smaller, robust models over larger fragile ones.
- Do not introduce lookahead leakage.
- Keep splits time-based.
- Do not turn the options wrapper into a contract-prediction training target in v1.
- Use deterministic rules for contract selection.

## Good Experiment Areas

- Model width/depth and dropout
- Feature weighting and auxiliary losses
- Confidence thresholding
- Validation-score composition
- Regularization and batch size
- Better programmatic labels or risk bands
- Cleaner rationale templates
- Better leakage-safe prompt construction
- Better text-source ranking across news, filings, and transcripts
- Better hybrid use of baseline numeric context

## Bad Experiment Areas

- Anything that leaks future data into features
- Switching to full live execution logic
- Directly optimizing on test metrics
- Expanding scope beyond liquid US equities/ETFs for v1
- Training on Reddit, X, GitHub, or YouTube text in v1
- Adding chain-of-thought style output to production prompts

## Output Format

Each run should print:
- dataset path
- training seconds
- peak VRAM
- validation loss
- validation precision/recall
- average trade return
- drawdown
- Sharpe proxy
- final `validation_score`

The artifact `~/.cache/autoresearch/market/artifacts/latest_predictions.json` should contain sample validation predictions.
