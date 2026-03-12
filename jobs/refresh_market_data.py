from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from market_data_service.config import MarketDataConfig, parse_symbols
from market_data_service.service import refresh_market_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh raw market data and optionally rebuild the parquet dataset.")
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
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--prepare-output-path", default=None)
    parser.add_argument("--frequency", default="1d")
    return parser


def _run_prepare(config: MarketDataConfig, output_path: str | None, frequency: str) -> None:
    command = [
        sys.executable,
        "prepare.py",
        "--universe",
        ",".join(config.symbols),
        "--frequency",
        frequency,
    ]
    if config.start_date:
        command.extend(["--start-date", config.start_date])
    if config.end_date:
        command.extend(["--end-date", config.end_date])
    if output_path:
        command.extend(["--output-path", output_path])
    env = os.environ.copy()
    env["MARKET_CACHE_DIR"] = str(config.cache_dir)
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1], env=env)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    env_config = MarketDataConfig.from_env()
    config = MarketDataConfig(
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
    refresh_market_data(config)
    if args.rebuild_dataset:
        _run_prepare(config, args.prepare_output_path, args.frequency)


if __name__ == "__main__":
    main()
