# Data Download Service And Job Plan

## Goal

Create a small internal service plus a scheduled job that keep the training-data cache populated in the format already expected by `prepare.py`:

- `~/.cache/autoresearch/market/raw/stocks/<SYMBOL>.csv`
- `~/.cache/autoresearch/market/raw/news/<SYMBOL>.csv`
- `~/.cache/autoresearch/market/raw/options/<SYMBOL>.csv`

The plan should preserve the current training flow:

1. Download or refresh raw market data.
2. Run `uv run prepare.py` to build `market_dataset.parquet`.
3. Run `uv run train.py`.

## Current Constraints From The Repo

- `prepare.py` is file-driven. It does not call external APIs.
- Stock CSVs are required and must include `timestamp, open, high, low, close, volume`.
- News and options CSVs are optional, but when present they must follow the schemas already enforced in `prepare.py`.
- Data is read from the local cache under `MARKET_CACHE_DIR` or `~/.cache/autoresearch/market`.
- The trainer assumes the dataset build is an offline step, so the new service should not change training-time behavior.

## Proposed Architecture

- `market_data_service`
  - A Python module responsible for fetching, normalizing, validating, and writing raw CSV files.
  - Exposes a programmatic API and a CLI entrypoint.
- `market_data_job`
  - A scheduled runner that invokes the service for a configured universe and date window.
  - Can be triggered by cron, systemd timer, GitHub Actions, or a container scheduler later.
- `prepare.py`
  - Remains the dataset builder.
  - Only minor changes should be made, if any, to share path constants or validation helpers.

This keeps external I/O separate from feature engineering and training.

## Service Responsibilities

The service should own these steps:

1. Load configuration.
2. Resolve the symbol universe.
3. Pull data from one provider abstraction for:
   - OHLCV bars
   - news
   - options chains or snapshots
4. Normalize provider payloads into the exact CSV contract expected by `prepare.py`.
5. Write atomically to the raw cache directories.
6. Emit a manifest with coverage, row counts, and fetch timestamps.
7. Return a non-zero exit code when required stock data fails.

The service should not:

- build parquet datasets
- train models
- make trading decisions

## Recommended Code Layout

Add a small package instead of extending `prepare.py` directly:

- `market_data_service/__init__.py`
- `market_data_service/config.py`
- `market_data_service/providers/base.py`
- `market_data_service/providers/<provider>.py`
- `market_data_service/normalize.py`
- `market_data_service/storage.py`
- `market_data_service/validate.py`
- `market_data_service/cli.py`
- `jobs/refresh_market_data.py`

Optional config files:

- `.env`
- `config/market_data.example.yaml`

## Service Interface

Recommended CLI:

```bash
uv run -m market_data_service.cli \
  --provider polygon \
  --symbols AAPL,MSFT,NVDA,SPY,QQQ \
  --start-date 2022-01-01 \
  --end-date 2026-03-12 \
  --include-news \
  --include-options
```

Recommended Python entrypoint:

```python
refresh_market_data(
    provider="polygon",
    symbols=["AAPL", "MSFT", "NVDA", "SPY", "QQQ"],
    start_date="2022-01-01",
    end_date="2026-03-12",
    include_news=True,
    include_options=True,
)
```

## Configuration Plan

Use environment variables first for simplicity:

- `MARKET_DATA_PROVIDER`
- `MARKET_DATA_API_KEY`
- `MARKET_CACHE_DIR`
- `MARKET_DATA_UNIVERSE`
- `MARKET_DATA_START_DATE`
- `MARKET_DATA_END_DATE`
- `MARKET_DATA_INCLUDE_NEWS`
- `MARKET_DATA_INCLUDE_OPTIONS`

Add YAML later only if configuration becomes hard to manage.

## Provider Abstraction

Define a narrow interface so providers can be swapped without changing storage or job code:

- `fetch_stock_bars(symbol, start_date, end_date) -> DataFrame`
- `fetch_news(symbol, start_date, end_date) -> DataFrame`
- `fetch_options(symbol, start_date, end_date) -> DataFrame`

First implementation should target one provider only. Polygon is the most natural fit for equities, news, and options in one abstraction, but Alpaca plus another provider is also workable.

## Normalization Contract

Normalize all provider responses into these exact outputs.

Stocks:

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`

News:

- `timestamp`
- `headline`
- `summary`

Options:

- `timestamp`
- `contract`
- `option_type`
- `strike`
- `expiration`
- `delta`
- `implied_volatility`
- `bid`
- `ask`
- `open_interest`
- `volume`

Normalization should also:

- convert timestamps to UTC ISO-compatible values
- sort by `timestamp`
- drop duplicate rows
- coerce numeric types
- guarantee column presence for optional numeric fields

## Storage Plan

Write files into the existing cache layout so `prepare.py` works unchanged.

Recommended write strategy:

1. Fetch into memory or temp files.
2. Validate schema and non-empty stock data.
3. Write to `*.tmp`.
4. Rename atomically to final CSV path.

Add one manifest file under `raw/`:

- `raw/manifest.json`

Manifest fields:

- provider
- symbols
- date range requested
- per-symbol row counts
- per-feed success or failure
- fetched_at

## Job Design

The job should be a thin orchestrator around the service.

Responsibilities:

1. Read config.
2. Run the refresh service.
3. Optionally invoke `uv run prepare.py` after a successful refresh.
4. Log summary metrics.
5. Exit non-zero on missing required stock updates.

Recommended job modes:

- `backfill`
  - Fetch a large historical window for first-time setup.
- `daily-refresh`
  - Fetch only recent data and merge with existing CSVs.
- `rebuild-dataset`
  - Refresh data, then run `prepare.py`.

## Incremental Refresh Strategy

Avoid full-history downloads on every run.

Per symbol:

- stocks: refetch a small overlap window such as the last 10 trading days
- news: refetch the last 3 to 7 days
- options: refetch the last 1 to 3 days because chains change quickly

Then merge, deduplicate by stable keys, and rewrite the full normalized CSV.

Deduplication keys:

- stocks: `timestamp`
- news: `timestamp + headline`
- options: `timestamp + contract`

## Error Handling

Treat feeds differently based on training requirements.

- Stocks: hard failure if a requested symbol cannot be fetched or validates empty.
- News: soft failure, log and continue.
- Options: soft failure, log and continue.

Retry policy:

- retry transient HTTP failures with backoff
- cap retries to avoid hanging scheduled runs
- log rate-limit responses separately

## Observability

At minimum, log:

- provider
- symbol
- feed type
- request window
- rows fetched
- rows written
- elapsed time
- failure reason

Useful counters:

- symbols requested vs succeeded
- stock/news/options coverage ratio
- stale-symbol count

## Integration With Existing Training Flow

The shortest path is:

1. Run `jobs/refresh_market_data.py`.
2. Run `uv run prepare.py --start-date ... --end-date ... --universe ...`.
3. Run `uv run train.py`.

If desired later, add a wrapper job that performs all three steps, but keep the data refresh service independently runnable.

## Tests

Add tests before wiring this into automation:

- schema validation for stock, news, and options outputs
- timestamp normalization tests
- deduplication tests
- merge tests for incremental refresh
- provider adapter tests with mocked API responses
- smoke test that generated CSVs are accepted by `prepare.py`

Critical regression test:

- run the service against fixture data, then run `build_market_dataset(...)` and confirm a parquet dataset is produced without schema errors

## Implementation Phases

### Phase 1: Service Skeleton

- create the `market_data_service` package
- centralize cache-path helpers
- add config loading
- add validation and atomic CSV writing

### Phase 2: One Provider

- implement a single provider adapter
- support stock bars first
- add optional news and options fetches behind flags

### Phase 3: Job Runner

- add `jobs/refresh_market_data.py`
- support `backfill` and `daily-refresh`
- emit manifest and logs

### Phase 4: Training Integration

- add a documented command sequence or wrapper script
- verify `prepare.py` consumes generated CSVs without modification
- run one end-to-end smoke test through `train.py`

## Minimal Viable Version

For the first cut, keep the scope narrow:

- one provider
- daily OHLCV stock download only
- write `raw/stocks/<SYMBOL>.csv`
- no service daemon, only a callable CLI plus scheduled job
- optional dataset rebuild step after refresh

This is enough to unblock repeatable dataset generation and model training. News and options can be added after the stock pipeline is stable.

## Open Decisions

- Which provider is the source of truth for v1?
- Is the first deployment local cron/systemd, or should this be containerized immediately?
- Should dataset rebuild happen in the same job or in a downstream training workflow?
- Do we want symbol-universe configuration in env vars only, or committed config files?

## Recommended Next Step

Implement the minimal viable version first:

1. provider adapter for stock OHLCV
2. CSV normalization and atomic writes into the current raw cache
3. scheduled refresh job
4. smoke test with `uv run prepare.py`

After that is stable, add news and options as optional feeds.
