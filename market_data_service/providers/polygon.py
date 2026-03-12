from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from .base import MarketDataProvider


class PolygonProvider(MarketDataProvider):
    def __init__(self, api_key: str, timeout: float = 30.0):
        if not api_key:
            raise ValueError("Polygon provider requires MARKET_DATA_API_KEY or --api-key.")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.base_url = "https://api.polygon.io"
        self.max_retries = 4
        self.backoff_seconds = 2.0

    def _with_api_key(self, url: str) -> str:
        if "apiKey=" in url:
            return url
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}apiKey={self.api_key}"

    def _get_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(self._with_api_key(url), params=params, timeout=self.timeout)
                if response.status_code == 429 and attempt < self.max_retries - 1:
                    retry_after = float(response.headers.get("Retry-After", self.backoff_seconds * (attempt + 1)))
                    time.sleep(retry_after)
                    continue
                if response.status_code == 403:
                    raise RuntimeError(
                        f"Polygon access forbidden for {url}. "
                        "This API key likely does not include the requested dataset."
                    )
                response.raise_for_status()
                payload = response.json()
                if payload.get("status") == "ERROR":
                    raise RuntimeError(payload.get("error", "Polygon request failed"))
                return payload
            except (requests.RequestException, RuntimeError) as exc:
                last_error = exc
                is_retryable = isinstance(exc, requests.RequestException) and (
                    getattr(exc.response, "status_code", None) in {408, 429, 500, 502, 503, 504}
                    or exc.response is None
                )
                if attempt >= self.max_retries - 1 or not is_retryable:
                    raise
                time.sleep(self.backoff_seconds * (2**attempt))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Polygon request failed without a concrete error.")

    def _paginate(self, url: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        next_url: str | None = url
        next_params = params
        while next_url:
            payload = self._get_json(next_url, params=next_params)
            results.extend(payload.get("results", []))
            next_url = payload.get("next_url")
            next_params = None
        return results

    @staticmethod
    def _from_millis(value: int | float | None) -> str | None:
        if value is None:
            return None
        timestamp = pd.to_datetime(value, unit="ms", utc=True)
        return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _from_nanos(value: int | float | None) -> str | None:
        if value is None:
            return None
        timestamp = pd.to_datetime(value, unit="ns", utc=True)
        return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

    def fetch_stock_bars(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        payload = self._get_json(
            f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}",
            params={"adjusted": "true", "sort": "asc", "limit": 50_000},
        )
        rows = []
        for item in payload.get("results", []):
            rows.append(
                {
                    "timestamp": self._from_millis(item.get("t")),
                    "open": item.get("o"),
                    "high": item.get("h"),
                    "low": item.get("l"),
                    "close": item.get("c"),
                    "volume": item.get("v"),
                }
            )
        return pd.DataFrame(rows)

    def fetch_news(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        results = self._paginate(
            f"{self.base_url}/v2/reference/news",
            params={
                "ticker": symbol,
                "published_utc.gte": start_date,
                "published_utc.lte": end_date,
                "sort": "published_utc",
                "order": "asc",
                "limit": 1000,
            },
        )
        rows = []
        for item in results:
            rows.append(
                {
                    "timestamp": item.get("published_utc"),
                    "headline": item.get("title", ""),
                    "summary": item.get("description", "") or "",
                }
            )
        return pd.DataFrame(rows)

    def fetch_options(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        del start_date, end_date
        results = self._paginate(
            f"{self.base_url}/v3/snapshot/options/{symbol}",
            params={"limit": 250, "sort": "expiration_date", "order": "asc"},
        )
        fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = []
        for item in results:
            details = item.get("details") or {}
            if not details.get("ticker") or not details.get("expiration_date") or not details.get("contract_type"):
                continue
            last_quote = item.get("last_quote") or {}
            last_trade = item.get("last_trade") or {}
            day = item.get("day") or {}
            greeks = item.get("greeks") or {}
            timestamp = (
                self._from_nanos(last_quote.get("last_updated"))
                or self._from_nanos(last_trade.get("sip_timestamp"))
                or self._from_nanos(day.get("last_updated"))
                or fetched_at
            )
            rows.append(
                {
                    "timestamp": timestamp,
                    "contract": details.get("ticker"),
                    "option_type": details.get("contract_type"),
                    "strike": details.get("strike_price"),
                    "expiration": details.get("expiration_date"),
                    "delta": greeks.get("delta", 0.0),
                    "implied_volatility": item.get("implied_volatility", 0.0),
                    "bid": last_quote.get("bid", 0.0),
                    "ask": last_quote.get("ask", 0.0),
                    "open_interest": item.get("open_interest", 0),
                    "volume": day.get("volume", 0),
                }
            )
        return pd.DataFrame(rows)
