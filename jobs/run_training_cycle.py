from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from market_data_service.config import MarketDataConfig, parse_symbols
from market_data_service.service import refresh_market_data

from .refresh_market_data import _run_prepare


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh market data, rebuild the dataset, and run training."
    )
    parser.add_argument("--provider", default="polygon")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--symbols", default="AAPL,MSFT,NVDA,SPY,QQQ")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--include-news", action="store_true")
    parser.add_argument("--include-options", action="store_true")
    parser.add_argument("--mode", choices=["backfill", "daily-refresh"], default="daily-refresh")
    parser.add_argument("--cache-dir", default=str(MarketDataConfig().cache_dir))
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--stock-overlap-days", type=int, default=10)
    parser.add_argument("--news-overlap-days", type=int, default=7)
    parser.add_argument("--options-overlap-days", type=int, default=3)
    parser.add_argument("--prepare-output-path", default=None)
    parser.add_argument("--frequency", default="1d")
    parser.add_argument("--train-seconds", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser


def build_market_data_config(args: argparse.Namespace) -> MarketDataConfig:
    env_config = MarketDataConfig.from_env()
    return MarketDataConfig(
        provider=args.provider,
        api_key=args.api_key or env_config.api_key,
        symbols=parse_symbols(args.symbols),
        start_date=args.start_date,
        end_date=args.end_date or env_config.end_date,
        include_news=args.include_news or env_config.include_news,
        include_options=args.include_options or env_config.include_options,
        cache_dir=Path(args.cache_dir),
        mode=args.mode,
        request_timeout=args.request_timeout,
        stock_overlap_days=args.stock_overlap_days,
        news_overlap_days=args.news_overlap_days,
        options_overlap_days=args.options_overlap_days,
    )


def _dataset_path(config: MarketDataConfig, prepare_output_path: str | None) -> Path:
    if prepare_output_path:
        return Path(prepare_output_path).expanduser().resolve()
    return (config.cache_dir / "dataset" / "market_dataset.parquet").resolve()


def _run_train(config: MarketDataConfig, dataset_path: Path, args: argparse.Namespace) -> None:
    command = [sys.executable, "train.py"]
    env = os.environ.copy()
    env["MARKET_CACHE_DIR"] = str(config.cache_dir)
    env["MARKET_DATASET_PATH"] = str(dataset_path)

    if args.train_seconds is not None:
        env["TRAIN_SECONDS"] = str(args.train_seconds)
    if args.batch_size is not None:
        env["BATCH_SIZE"] = str(args.batch_size)
    if args.hidden_dim is not None:
        env["HIDDEN_DIM"] = str(args.hidden_dim)
    if args.depth is not None:
        env["DEPTH"] = str(args.depth)
    if args.dropout is not None:
        env["DROPOUT"] = str(args.dropout)
    if args.learning_rate is not None:
        env["LEARNING_RATE"] = str(args.learning_rate)

    LOGGER.info("starting training")
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    config = build_market_data_config(args)

    LOGGER.info("refreshing market data")
    refresh_market_data(config)

    LOGGER.info("rebuilding dataset")
    _run_prepare(config, args.prepare_output_path, args.frequency)

    LOGGER.info("running training")
    _run_train(config, _dataset_path(config, args.prepare_output_path), args)


if __name__ == "__main__":
    main()
