from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import DEFAULT_CACHE_DIR, MarketDataConfig, parse_symbols
from .service import refresh_market_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and refresh raw market data for training.")
    parser.add_argument("--provider", default="polygon")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--symbols", default="AAPL,MSFT,NVDA,SPY,QQQ")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--include-news", action="store_true")
    parser.add_argument("--include-options", action="store_true")
    parser.add_argument("--mode", choices=["backfill", "daily-refresh"], default="backfill")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--stock-overlap-days", type=int, default=10)
    parser.add_argument("--news-overlap-days", type=int, default=7)
    parser.add_argument("--options-overlap-days", type=int, default=3)
    return parser


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
    manifest = refresh_market_data(config)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
