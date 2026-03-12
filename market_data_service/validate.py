from __future__ import annotations

import pandas as pd

from .normalize import NEWS_COLUMNS, OPTIONS_COLUMNS, STOCK_COLUMNS


def _require_columns(frame: pd.DataFrame, required_columns: list[str], feed: str, symbol: str) -> None:
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{feed} data for {symbol} is missing required columns: {missing}")


def validate_stock_frame(frame: pd.DataFrame, symbol: str) -> None:
    _require_columns(frame, STOCK_COLUMNS, "stock", symbol)
    if frame.empty:
        raise ValueError(f"stock data for {symbol} is empty")


def validate_news_frame(frame: pd.DataFrame, symbol: str) -> None:
    _require_columns(frame, NEWS_COLUMNS, "news", symbol)


def validate_options_frame(frame: pd.DataFrame, symbol: str) -> None:
    _require_columns(frame, OPTIONS_COLUMNS, "options", symbol)
