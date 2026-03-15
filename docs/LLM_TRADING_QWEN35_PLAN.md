# Qwen3.5-9B LLM Trading Plan

## Summary

- Convert the repo from a structured-only trading model into a hybrid trading research stack.
- Keep the current numeric model as the baseline and teacher.
- Use `Qwen/Qwen3.5-9B` in text-only mode for v1.
- Use QLoRA supervised fine-tuning only.
- Keep the current market labels, time splits, and `validation_score` as the evaluation contract.

## Architecture

- `prepare.py` remains the structured dataset builder.
- `train.py` remains the structured baseline trainer and now saves a checkpoint for reuse.
- `prepare_llm.py` builds `market_llm_dataset.parquet` with:
  - `prompt_text`
  - `completion_text`
  - `target_json`
  - `numeric_context_json`
  - `baseline_context_json`
  - `source_manifest_json`
- `train_llm.py` runs QLoRA SFT for `Qwen/Qwen3.5-9B`.
- `evaluate_llm.py` evaluates:
  - `structured_baseline`
  - `llm_text_only`
  - `llm_hybrid`

## Data Policy

- Allowed v1 training sources:
  - news
  - SEC filings
  - earnings transcripts
- Excluded from v1 training:
  - Reddit
  - X
  - GitHub
  - YouTube transcripts

## Runtime Defaults

- Base model: `Qwen/Qwen3.5-9B`
- Fine-tuning: 4-bit QLoRA
- Max sequence length: 2048
- Micro-batch size: 1
- Gradient accumulation: 16
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05

## Acceptance Gates

- No lookahead leakage
- JSON parse success >= 99%
- Schema-valid outputs >= 99%
- Hybrid mean walk-forward `validation_score` >= baseline + 0.01
- Hybrid drawdown not worse than baseline by more than 10% relative
