from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


DEFAULT_SYMBOLS = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ", "GOOGL", "AMZN", "META", "TSLA", "BRK.B"]
DEFAULT_CACHE_DIR = Path(
    os.environ.get(
        "MARKET_CACHE_DIR",
        str(Path(os.path.expanduser("~")) / ".cache" / "autoresearch" / "market"),
    )
)


def parse_symbols(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_SYMBOLS)
    symbols = [symbol.strip().upper() for symbol in raw.split(",") if symbol.strip()]
    return symbols or list(DEFAULT_SYMBOLS)


def parse_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class MarketDataConfig:
    provider: str = "polygon"
    api_key: str | None = None
    symbols: list[str] = field(default_factory=lambda: list(DEFAULT_SYMBOLS))
    start_date: str | None = None
    end_date: str = field(default_factory=lambda: date.today().isoformat())
    include_news: bool = False
    include_options: bool = False
    cache_dir: Path = DEFAULT_CACHE_DIR
    mode: str = "backfill"
    request_timeout: float = 30.0
    stock_overlap_days: int = 10
    news_overlap_days: int = 7
    options_overlap_days: int = 3

    @classmethod
    def from_env(cls) -> "MarketDataConfig":
        return cls(
            provider=os.environ.get("MARKET_DATA_PROVIDER", "polygon"),
            api_key=os.environ.get("MARKET_DATA_API_KEY"),
            symbols=parse_symbols(os.environ.get("MARKET_DATA_UNIVERSE")),
            start_date=os.environ.get("MARKET_DATA_START_DATE"),
            end_date=os.environ.get("MARKET_DATA_END_DATE", date.today().isoformat()),
            include_news=parse_bool(os.environ.get("MARKET_DATA_INCLUDE_NEWS"), False),
            include_options=parse_bool(os.environ.get("MARKET_DATA_INCLUDE_OPTIONS"), False),
            cache_dir=Path(os.environ.get("MARKET_CACHE_DIR", str(DEFAULT_CACHE_DIR))),
            mode=os.environ.get("MARKET_DATA_MODE", "backfill"),
            request_timeout=float(os.environ.get("MARKET_DATA_REQUEST_TIMEOUT", "30")),
            stock_overlap_days=int(os.environ.get("MARKET_DATA_STOCK_OVERLAP_DAYS", "10")),
            news_overlap_days=int(os.environ.get("MARKET_DATA_NEWS_OVERLAP_DAYS", "7")),
            options_overlap_days=int(os.environ.get("MARKET_DATA_OPTIONS_OVERLAP_DAYS", "3")),
        )
