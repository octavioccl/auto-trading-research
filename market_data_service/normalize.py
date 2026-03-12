from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


STOCK_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
NEWS_COLUMNS = ["timestamp", "headline", "summary"]
OPTIONS_COLUMNS = [
    "timestamp",
    "contract",
    "option_type",
    "strike",
    "expiration",
    "delta",
    "implied_volatility",
    "bid",
    "ask",
    "open_interest",
    "volume",
]


def empty_frame(columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _normalize_timestamp_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    result = frame.copy()
    result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    result = result.dropna(subset=[column])
    result[column] = result[column].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return result


def normalize_stock_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return empty_frame(STOCK_COLUMNS)

    result = frame.copy()
    for column in STOCK_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    result = result[STOCK_COLUMNS]
    result = _normalize_timestamp_column(result, "timestamp")
    for column in ["open", "high", "low", "close", "volume"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["open", "high", "low", "close", "volume"])
    result = result.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    result["volume"] = result["volume"].astype("int64")
    return result.reset_index(drop=True)


def normalize_news_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return empty_frame(NEWS_COLUMNS)

    result = frame.copy()
    for column in NEWS_COLUMNS:
        if column not in result.columns:
            result[column] = ""
    result = result[NEWS_COLUMNS]
    result = _normalize_timestamp_column(result, "timestamp")
    result["headline"] = result["headline"].fillna("").astype(str).str.strip()
    result["summary"] = result["summary"].fillna("").astype(str).str.strip()
    result = result[result["headline"] != ""]
    result = result.sort_values("timestamp").drop_duplicates(subset=["timestamp", "headline"], keep="last")
    return result.reset_index(drop=True)


def normalize_options_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return empty_frame(OPTIONS_COLUMNS)

    result = frame.copy()
    for column in OPTIONS_COLUMNS:
        if column not in result.columns:
            result[column] = 0.0 if column not in {"timestamp", "contract", "option_type", "expiration"} else ""
    result = result[OPTIONS_COLUMNS]
    result = _normalize_timestamp_column(result, "timestamp")
    result["expiration"] = pd.to_datetime(result["expiration"], utc=True, errors="coerce")
    result = result.dropna(subset=["expiration"])
    result["expiration"] = result["expiration"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    result["contract"] = result["contract"].fillna("").astype(str).str.strip()
    result["option_type"] = result["option_type"].fillna("").astype(str).str.lower().str.strip()
    for column in ["strike", "delta", "implied_volatility", "bid", "ask", "open_interest", "volume"]:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    result = result[result["contract"] != ""]
    result = result[result["option_type"].isin({"call", "put"})]
    result = result.sort_values(["timestamp", "contract"]).drop_duplicates(subset=["timestamp", "contract"], keep="last")
    result["open_interest"] = result["open_interest"].astype("int64")
    result["volume"] = result["volume"].astype("int64")
    return result.reset_index(drop=True)


def merge_frames(existing: pd.DataFrame, incoming: pd.DataFrame, feed: str) -> pd.DataFrame:
    if existing.empty:
        return incoming.reset_index(drop=True)
    if incoming.empty:
        return existing.reset_index(drop=True)

    columns_by_feed = {
        "stocks": STOCK_COLUMNS,
        "news": NEWS_COLUMNS,
        "options": OPTIONS_COLUMNS,
    }
    dedupe_keys = {
        "stocks": ["timestamp"],
        "news": ["timestamp", "headline"],
        "options": ["timestamp", "contract"],
    }
    columns = columns_by_feed[feed]
    merged = pd.concat([existing[columns], incoming[columns]], ignore_index=True)
    merged = merged.drop_duplicates(subset=dedupe_keys[feed], keep="last")
    sort_columns = dedupe_keys[feed]
    return merged.sort_values(sort_columns).reset_index(drop=True)
