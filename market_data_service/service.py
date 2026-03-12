from __future__ import annotations

import logging
import re
from datetime import timedelta

import pandas as pd

from .config import MarketDataConfig
from .normalize import merge_frames, normalize_news_frame, normalize_options_frame, normalize_stock_frame
from .providers import PolygonProvider
from .storage import build_paths, ensure_directories, latest_timestamp, read_existing_frame, write_frame_atomic, write_manifest
from .validate import validate_news_frame, validate_options_frame, validate_stock_frame


logger = logging.getLogger(__name__)


def _sanitize_error_message(message: str) -> str:
    return re.sub(r"apiKey=[^&\\s]+", "apiKey=[redacted]", message)


def _provider_from_config(config: MarketDataConfig):
    provider_name = config.provider.lower()
    if provider_name == "polygon":
        return PolygonProvider(api_key=config.api_key or "", timeout=config.request_timeout)
    raise ValueError(f"Unsupported market data provider: {config.provider}")


def _resolve_start_date(config: MarketDataConfig, path, overlap_days: int, default_days: int) -> str:
    if config.start_date and config.mode == "backfill":
        return config.start_date

    last_seen = latest_timestamp(path)
    if last_seen is not None:
        return (last_seen - timedelta(days=overlap_days)).date().isoformat()

    if config.start_date:
        return config.start_date

    end_ts = pd.Timestamp(config.end_date, tz="UTC")
    return (end_ts - timedelta(days=default_days)).date().isoformat()


def _write_feed(feed: str, symbol: str, frame: pd.DataFrame, path, mode: str) -> tuple[int, int]:
    existing = read_existing_frame(path, feed) if mode == "daily-refresh" else pd.DataFrame(columns=frame.columns)
    merged = merge_frames(existing, frame, feed) if mode == "daily-refresh" else frame
    write_frame_atomic(merged, path)
    return int(len(frame)), int(len(merged))


def refresh_market_data(config: MarketDataConfig) -> dict[str, object]:
    provider = _provider_from_config(config)
    paths = build_paths(config.cache_dir)
    ensure_directories(paths)

    manifest: dict[str, object] = {
        "provider": config.provider,
        "mode": config.mode,
        "cache_dir": str(config.cache_dir),
        "requested_start_date": config.start_date,
        "requested_end_date": config.end_date,
        "include_news": config.include_news,
        "include_options": config.include_options,
        "symbols": config.symbols,
        "results": {},
    }
    stock_failures: list[str] = []

    for symbol in config.symbols:
        symbol_result: dict[str, object] = {}
        manifest["results"][symbol] = symbol_result

        stocks_path = paths.stocks_dir / f"{symbol}.csv"
        stocks_start = _resolve_start_date(config, stocks_path, config.stock_overlap_days, default_days=365)
        try:
            stocks_frame = normalize_stock_frame(provider.fetch_stock_bars(symbol, stocks_start, config.end_date))
            validate_stock_frame(stocks_frame, symbol)
            fetched_rows, written_rows = _write_feed("stocks", symbol, stocks_frame, stocks_path, config.mode)
            symbol_result["stocks"] = {
                "status": "ok",
                "start_date": stocks_start,
                "end_date": config.end_date,
                "fetched_rows": fetched_rows,
                "written_rows": written_rows,
            }
            logger.info("stocks refreshed", extra={"symbol": symbol, "rows": written_rows})
        except Exception as exc:
            stock_failures.append(symbol)
            symbol_result["stocks"] = {"status": "error", "error": _sanitize_error_message(str(exc))}
            logger.exception("stock refresh failed for %s", symbol)

        if config.include_news:
            news_path = paths.news_dir / f"{symbol}.csv"
            news_start = _resolve_start_date(config, news_path, config.news_overlap_days, default_days=30)
            try:
                news_frame = normalize_news_frame(provider.fetch_news(symbol, news_start, config.end_date))
                validate_news_frame(news_frame, symbol)
                fetched_rows, written_rows = _write_feed("news", symbol, news_frame, news_path, config.mode)
                symbol_result["news"] = {
                    "status": "ok",
                    "start_date": news_start,
                    "end_date": config.end_date,
                    "fetched_rows": fetched_rows,
                    "written_rows": written_rows,
                }
            except Exception as exc:
                symbol_result["news"] = {"status": "error", "error": _sanitize_error_message(str(exc))}
                logger.exception("news refresh failed for %s", symbol)

        if config.include_options:
            options_path = paths.options_dir / f"{symbol}.csv"
            options_start = _resolve_start_date(config, options_path, config.options_overlap_days, default_days=3)
            try:
                options_frame = normalize_options_frame(provider.fetch_options(symbol, options_start, config.end_date))
                validate_options_frame(options_frame, symbol)
                fetched_rows, written_rows = _write_feed("options", symbol, options_frame, options_path, config.mode)
                symbol_result["options"] = {
                    "status": "ok",
                    "start_date": options_start,
                    "end_date": config.end_date,
                    "fetched_rows": fetched_rows,
                    "written_rows": written_rows,
                }
            except Exception as exc:
                symbol_result["options"] = {"status": "error", "error": _sanitize_error_message(str(exc))}
                logger.exception("options refresh failed for %s", symbol)

    manifest["success"] = not stock_failures
    manifest["stock_failures"] = stock_failures
    write_manifest(manifest, paths.manifest_path)
    if stock_failures:
        raise RuntimeError(f"Stock refresh failed for symbols: {', '.join(stock_failures)}")
    return manifest
