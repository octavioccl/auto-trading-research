# Adapt `autoresearch` Into a Trading-Idea Research Harness

## Summary

Use this repo as a starting point for **autonomous model iteration**, not as a finished trading stack. In its current form it is a 5-minute next-token pretraining harness over generic text with a fixed `val_bpb` metric, so it cannot directly answer “what stock or option should I trade?” without substantial changes to data prep, labels, evaluation, and model outputs.

For the chosen scope, build **v1 as an equity-first swing-signal model** for US large caps and liquid ETFs, with an **options wrapper** that converts strong stock signals into candidate contracts. The model should output a **hybrid response**: structured fields plus a short rationale. Training labels should be **programmatically generated** from future returns and risk rules.

The target machine is Linux with an RTX 3090 24 GB, so this fork should stay **3090-oriented**. Expect smaller models, lower batch sizes, and simpler attention or MLP-only architectures where needed for stable training.

## Research Findings

- The original repo was deliberately narrow: `prepare.py` downloaded text shards and trained a tokenizer; `train.py` handled single-GPU GPT pretraining; `program.md` optimized a 5-minute `val_bpb` run loop.
- That original design is useful for **rapid autonomous iteration**, but it was missing the trading-specific pieces: market-data ingestion, labels, backtests, risk rules, output schemas, and options selection.
- The necessary raw data is available from common providers such as Alpaca and Polygon, but this fork should operate on a **local normalized dataset contract** so training remains reproducible and decoupled from live APIs.

## Key Implementation Changes

- Reframe the repo from generic LM pretraining to market-example training.
  - `prepare.py` should build a local parquet market dataset instead of downloading text corpora.
  - `train.py` should train a multitask model on structured targets instead of next-token loss.
  - `program.md` should guide autonomous experiments toward validation score, not BPB.
- Define a fixed example schema.
  - Inputs should include trailing price/volume features, technical indicators, compact news summaries, market regime signals, and options snapshot summaries.
  - Targets should include:
    - `symbol`
    - `trade_direction` (`long`, `short`, `no_trade`)
    - `confidence`
    - `holding_window_days`
    - `entry_style`
    - `stop_loss_pct`
    - `take_profit_pct`
    - `rationale`
    - `options_plan` fields for `use_options`, `call_or_put`, `target_dte_bucket`, `target_delta_bucket`
- Use programmatic labels.
  - Generate targets from forward returns, run-up, and drawdown over the next 1-5 days.
  - Emit `no_trade` where the expected move does not clear risk or cost thresholds.
  - Keep options as a wrapper over strong equity signals rather than a direct contract-prediction objective in v1.
- Replace BPB evaluation with multitask validation and backtest-style metrics.
  - Classification for direction and categorical trade fields.
  - Regression or bounded outputs for confidence and risk bands.
  - Rationale generation through template or class selection in v1.
  - Primary metrics:
    - action precision / recall
    - average trade return
    - hit rate
    - drawdown
    - Sharpe-like proxy
    - aggregate `validation_score`
- Add a deterministic options wrapper.
  - Bullish signals map to liquid calls; bearish signals map to liquid puts.
  - Filter on spread, open interest, DTE bucket, and delta bucket.
- Retune for RTX 3090 constraints.
  - Prefer smaller dense models first.
  - Keep feature payloads compact.
  - Use summarized news instead of long raw text.

## Public Interfaces And Types

- `prepare.py`
  - Main entrypoint should expose `build_market_dataset(start_date, end_date, universe, frequency)`.
  - Dataset rows should contain:
    - `asof_ts`
    - `symbol`
    - `feature_payload`
    - `news_payload`
    - `option_payload`
    - `target_payload`
    - `split`
- `train.py`
  - Should expose a multitask training path that predicts structured trade outputs.
  - Final run summary should print validation and backtest metrics instead of `val_bpb`.
- `program.md`
  - Should define the autonomous experiment loop around `validation_score`.

## Test Plan

- Data tests
  - Verify no lookahead leakage.
  - Verify strict time-based train/val/test splits.
  - Verify news and options joins align to the correct symbol and timestamp.
- Label tests
  - Verify `long`, `short`, and `no_trade` follow the formal label rules.
  - Verify option selection respects liquidity and spread thresholds.
- Training tests
  - Smoke test on a reduced synthetic dataset.
  - Confirm training completes on the target hardware class without OOM.
- Evaluation tests
  - Backtest on held-out periods with transaction-cost assumptions.
  - Compare against simple baselines such as always `no_trade`, momentum, and mean reversion.
- Acceptance criteria
  - Stable training on a 24 GB GPU.
  - No leakage.
  - Better validation precision and risk-adjusted return than baseline heuristics.
  - Usable daily ranked trade ideas with structured output and rationale.

## Assumptions

- v1 is research-only decision support, not live automated execution.
- Asset scope is US large caps and liquid ETFs.
- Trading horizon is 1-5 day swing trades.
- Options remain a post-model wrapper in v1.
- Data is normalized locally before training.
- This fork intentionally diverges from the original single-file, BPB-only design.
