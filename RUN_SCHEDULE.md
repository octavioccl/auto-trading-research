# Scheduled Training Runbook

## What Runs

The scheduled cycle uses [jobs/run_training_cycle.py](/home/octavioccl/github/autoresearch/jobs/run_training_cycle.py) to do this in order:

1. refresh raw market data
2. rebuild the parquet dataset with `prepare.py`
3. run `train.py`

It uses the same cache layout already expected by the repo:

- raw data under `MARKET_CACHE_DIR/raw`
- dataset under `MARKET_CACHE_DIR/dataset/market_dataset.parquet`
- predictions under `MARKET_CACHE_DIR/artifacts/latest_predictions.json`

## Manual Run

```bash
MARKET_DATA_API_KEY=... \
MARKET_CACHE_DIR=/home/octavioccl/github/autoresearch/.cache/market \
python -m jobs.run_training_cycle --mode daily-refresh
```

Optional overrides:

- `--include-news`
- `--include-options`
- `--train-seconds 300`
- `--batch-size 64`
- `--hidden-dim 192`
- `--depth 4`

## systemd Setup

Files are provided in [ops/systemd](/home/octavioccl/github/autoresearch/ops/systemd):

- [ops/systemd/autoresearch-training.service](/home/octavioccl/github/autoresearch/ops/systemd/autoresearch-training.service)
- [ops/systemd/autoresearch-training.timer](/home/octavioccl/github/autoresearch/ops/systemd/autoresearch-training.timer)
- [ops/systemd/autoresearch.env.example](/home/octavioccl/github/autoresearch/ops/systemd/autoresearch.env.example)

Install them like this:

```bash
mkdir -p ~/.config/systemd/user
cp /home/octavioccl/github/autoresearch/ops/systemd/autoresearch-training.service ~/.config/systemd/user/
cp /home/octavioccl/github/autoresearch/ops/systemd/autoresearch-training.timer ~/.config/systemd/user/
cp /home/octavioccl/github/autoresearch/ops/systemd/autoresearch.env.example ~/.config/autoresearch.env
systemctl --user daemon-reload
systemctl --user enable --now autoresearch-training.timer
systemctl --user list-timers autoresearch-training.timer
```

## Notes

- The timer runs every 8 hours after the previous successful activation.
- `Persistent=true` means missed runs are caught up after reboot.
- If you want news or options in the scheduled run, set them in `~/.config/autoresearch.env`.
- With the current Polygon key validation, stock data works; news may need more throttling on large backfills, and options depend on plan access.
