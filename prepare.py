"""
Market-data preparation for the trading-idea autoresearch fork.

This module replaces generic text pretraining prep with a deterministic,
time-split market dataset builder for equity-first swing trading research.

Usage:
    uv run prepare.py --generate-demo-data
    uv run prepare.py --start-date 2022-01-01 --end-date 2024-12-31

Expected local raw inputs:
    ~/.cache/autoresearch/market/raw/stocks/<SYMBOL>.csv
    ~/.cache/autoresearch/market/raw/news/<SYMBOL>.csv       (optional)
    ~/.cache/autoresearch/market/raw/options/<SYMBOL>.csv    (optional)

The stock CSV is required for non-demo builds and must contain:
    timestamp, open, high, low, close, volume

Optional news CSV columns:
    timestamp, headline[, summary]

Optional options CSV columns:
    timestamp, contract, option_type, strike, expiration, delta,
    implied_volatility, bid, ask, open_interest, volume
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Fixed constants
# ---------------------------------------------------------------------------

TIME_BUDGET = 300
LOOKBACK_DAYS = 30
HORIZON_DAYS = 5
TARGET_COLUMNS = (
    "direction_id",
    "confidence",
    "holding_days",
    "entry_style_id",
    "stop_loss_pct",
    "take_profit_pct",
    "use_options",
    "option_type_id",
    "dte_bucket_id",
    "delta_bucket_id",
    "rationale_id",
    "forward_return_5d",
    "forward_drawdown_5d",
)

DIRECTION_LABELS = ["short", "no_trade", "long"]
ENTRY_STYLE_LABELS = ["breakout", "pullback", "mean_revert", "other"]
OPTION_TYPE_LABELS = ["none", "put", "call"]
DTE_BUCKET_LABELS = ["none", "7_14", "15_30", "31_45"]
DELTA_BUCKET_LABELS = ["none", "25_35", "35_50", "50_65"]
RATIONALE_LABELS = [
    "trend_breakout_bull",
    "trend_breakout_bear",
    "pullback_reclaim_bull",
    "mean_revert_bear",
    "volatility_unclear_no_trade",
]

# ---------------------------------------------------------------------------
# Local storage
# ---------------------------------------------------------------------------

CACHE_DIR = Path(os.environ.get("MARKET_CACHE_DIR", str(Path(os.path.expanduser("~")) / ".cache" / "autoresearch" / "market")))
RAW_DIR = CACHE_DIR / "raw"
DATASET_DIR = CACHE_DIR / "dataset"
STOCK_RAW_DIR = RAW_DIR / "stocks"
NEWS_RAW_DIR = RAW_DIR / "news"
OPTIONS_RAW_DIR = RAW_DIR / "options"
DEFAULT_DATASET_PATH = DATASET_DIR / "market_dataset.parquet"


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if abs(float(denominator)) > 1e-12 else 0.0


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return ((series - mean) / std).fillna(0.0)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean().fillna(method="bfill").fillna(0.0)


def pick_split(timestamp: pd.Timestamp, train_end: pd.Timestamp, val_end: pd.Timestamp) -> str:
    if timestamp <= train_end:
        return "train"
    if timestamp <= val_end:
        return "val"
    return "test"


def ensure_dirs() -> None:
    for path in [STOCK_RAW_DIR, NEWS_RAW_DIR, OPTIONS_RAW_DIR, DATASET_DIR]:
        path.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetConfig:
    start_date: Optional[str]
    end_date: Optional[str]
    universe: List[str]
    frequency: str = "1d"
    output_path: Path = DEFAULT_DATASET_PATH
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    def to_metadata(self) -> Dict[str, object]:
        data = asdict(self)
        data["output_path"] = str(self.output_path)
        return data


def load_stock_frame(symbol: str) -> pd.DataFrame:
    path = STOCK_RAW_DIR / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing stock CSV for {symbol}: {path}")
    df = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    return df.sort_values("timestamp").drop_duplicates("timestamp")


def load_news_frame(symbol: str) -> pd.DataFrame:
    path = NEWS_RAW_DIR / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "headline", "summary"])
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "headline" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "headline", "summary"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    if "summary" not in df.columns:
        df["summary"] = ""
    return df.sort_values("timestamp")


def load_options_frame(symbol: str) -> pd.DataFrame:
    path = OPTIONS_RAW_DIR / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=[
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
        )
    df = pd.read_csv(path)
    required = {"timestamp", "contract", "option_type", "expiration"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    df["expiration"] = pd.to_datetime(df["expiration"], utc=True).dt.tz_convert(None)
    for column in ["strike", "delta", "implied_volatility", "bid", "ask", "open_interest", "volume"]:
        if column not in df.columns:
            df[column] = 0.0
    return df.sort_values(["timestamp", "contract"])


def summarize_news(news_df: pd.DataFrame, asof_ts: pd.Timestamp) -> Dict[str, object]:
    if news_df.empty:
        return {
            "headline_count_3d": 0,
            "latest_headlines": [],
            "sentiment_proxy": 0.0,
        }

    window_start = asof_ts - pd.Timedelta(days=3)
    subset = news_df[(news_df["timestamp"] <= asof_ts) & (news_df["timestamp"] >= window_start)].tail(3)
    texts = [
        " ".join(str(value) for value in [row["headline"], row.get("summary", "")] if str(value).strip())
        for _, row in subset.iterrows()
    ]
    positive_terms = {"upgrade", "beat", "strong", "growth", "bullish", "record"}
    negative_terms = {"downgrade", "miss", "weak", "lawsuit", "bearish", "cut"}
    score = 0
    for text in texts:
        lowered = text.lower()
        score += sum(term in lowered for term in positive_terms)
        score -= sum(term in lowered for term in negative_terms)
    return {
        "headline_count_3d": int(len(texts)),
        "latest_headlines": texts,
        "sentiment_proxy": float(score),
    }


def summarize_option_snapshot(options_df: pd.DataFrame, asof_ts: pd.Timestamp, spot_price: float) -> Dict[str, object]:
    if options_df.empty:
        return {
            "has_options_data": False,
            "best_call_spread_pct": 1.0,
            "best_put_spread_pct": 1.0,
            "best_call_open_interest": 0.0,
            "best_put_open_interest": 0.0,
            "call_iv": 0.0,
            "put_iv": 0.0,
        }

    same_day = options_df[options_df["timestamp"] <= asof_ts]
    if same_day.empty:
        return {
            "has_options_data": False,
            "best_call_spread_pct": 1.0,
            "best_put_spread_pct": 1.0,
            "best_call_open_interest": 0.0,
            "best_put_open_interest": 0.0,
            "call_iv": 0.0,
            "put_iv": 0.0,
        }

    latest_ts = same_day["timestamp"].max()
    snap = same_day[same_day["timestamp"] == latest_ts].copy()
    snap["mid"] = (snap["bid"] + snap["ask"]) / 2.0
    snap["spread_pct"] = np.where(snap["mid"] > 0, (snap["ask"] - snap["bid"]) / snap["mid"], 1.0)
    snap["moneyness"] = (snap["strike"] - spot_price) / max(spot_price, 1e-6)
    call = snap[snap["option_type"].str.lower().eq("call")].sort_values(["spread_pct", "moneyness"]).head(1)
    put = snap[snap["option_type"].str.lower().eq("put")].sort_values(["spread_pct", "moneyness"]).head(1)

    def row_or_default(frame: pd.DataFrame) -> Dict[str, float]:
        if frame.empty:
            return {"spread_pct": 1.0, "open_interest": 0.0, "iv": 0.0}
        row = frame.iloc[0]
        return {
            "spread_pct": float(row["spread_pct"]),
            "open_interest": float(row["open_interest"]),
            "iv": float(row["implied_volatility"]),
        }

    call_data = row_or_default(call)
    put_data = row_or_default(put)
    return {
        "has_options_data": True,
        "best_call_spread_pct": call_data["spread_pct"],
        "best_put_spread_pct": put_data["spread_pct"],
        "best_call_open_interest": call_data["open_interest"],
        "best_put_open_interest": put_data["open_interest"],
        "call_iv": call_data["iv"],
        "put_iv": put_data["iv"],
    }


def choose_entry_style(row: pd.Series) -> int:
    if row["close_to_20d_high"] > -0.01 and row["ret_5d"] > 0:
        return ENTRY_STYLE_LABELS.index("breakout")
    if row["trend_gap"] > 0 and row["ret_3d"] < 0:
        return ENTRY_STYLE_LABELS.index("pullback")
    if row["rsi14"] > 68 or row["rsi14"] < 32:
        return ENTRY_STYLE_LABELS.index("mean_revert")
    return ENTRY_STYLE_LABELS.index("other")


def choose_rationale(direction_id: int, entry_style_id: int, volatility_pct: float) -> int:
    direction = DIRECTION_LABELS[direction_id]
    entry_style = ENTRY_STYLE_LABELS[entry_style_id]
    if direction == "no_trade" or volatility_pct > 0.06:
        return RATIONALE_LABELS.index("volatility_unclear_no_trade")
    if direction == "long" and entry_style == "breakout":
        return RATIONALE_LABELS.index("trend_breakout_bull")
    if direction == "short" and entry_style == "breakout":
        return RATIONALE_LABELS.index("trend_breakout_bear")
    if direction == "long" and entry_style == "pullback":
        return RATIONALE_LABELS.index("pullback_reclaim_bull")
    return RATIONALE_LABELS.index("mean_revert_bear")


def choose_option_targets(direction_id: int, confidence: float) -> Dict[str, int]:
    if direction_id == DIRECTION_LABELS.index("no_trade") or confidence < 0.58:
        return {
            "use_options": 0,
            "option_type_id": OPTION_TYPE_LABELS.index("none"),
            "dte_bucket_id": DTE_BUCKET_LABELS.index("none"),
            "delta_bucket_id": DELTA_BUCKET_LABELS.index("none"),
        }

    option_type = "call" if DIRECTION_LABELS[direction_id] == "long" else "put"
    delta_bucket = "35_50" if confidence < 0.72 else "50_65"
    return {
        "use_options": 1,
        "option_type_id": OPTION_TYPE_LABELS.index(option_type),
        "dte_bucket_id": DTE_BUCKET_LABELS.index("15_30"),
        "delta_bucket_id": DELTA_BUCKET_LABELS.index(delta_bucket),
    }


def build_feature_table(stock_df: pd.DataFrame) -> pd.DataFrame:
    df = stock_df.copy()
    df["ret_1d"] = df["close"].pct_change().fillna(0.0)
    df["ret_3d"] = df["close"].pct_change(3).fillna(0.0)
    df["ret_5d"] = df["close"].pct_change(5).fillna(0.0)
    df["ret_10d"] = df["close"].pct_change(10).fillna(0.0)
    df["ma_5"] = df["close"].rolling(5).mean().fillna(method="bfill")
    df["ma_20"] = df["close"].rolling(20).mean().fillna(method="bfill")
    df["trend_gap"] = safe_div(1.0, 1.0) * ((df["ma_5"] / df["ma_20"]) - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["volatility_20d"] = df["ret_1d"].rolling(20).std().fillna(0.0)
    df["volume_z20"] = rolling_zscore(np.log1p(df["volume"]), 20)
    df["rsi14"] = compute_rsi(df["close"])
    df["atr14"] = compute_atr(df)
    df["close_to_20d_high"] = safe_div(1.0, 1.0) * ((df["close"] / df["high"].rolling(20).max()) - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["close_to_20d_low"] = safe_div(1.0, 1.0) * ((df["close"] / df["low"].rolling(20).min()) - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["ema_fast"] = ema(df["close"], 8)
    df["ema_slow"] = ema(df["close"], 21)
    df["ema_gap"] = ((df["ema_fast"] / df["ema_slow"]) - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    future_close = df["close"].shift(-HORIZON_DAYS)
    future_min = df["low"].shift(-1).rolling(HORIZON_DAYS).min().shift(-(HORIZON_DAYS - 1))
    future_max = df["high"].shift(-1).rolling(HORIZON_DAYS).max().shift(-(HORIZON_DAYS - 1))
    df["forward_return_5d"] = ((future_close / df["close"]) - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["forward_drawdown_5d"] = ((future_min / df["close"]) - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["forward_runup_5d"] = ((future_max / df["close"]) - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df


def build_market_examples(
    symbol: str,
    stock_df: pd.DataFrame,
    news_df: pd.DataFrame,
    options_df: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
) -> List[Dict[str, object]]:
    features = build_feature_table(stock_df)
    usable = features.iloc[LOOKBACK_DAYS:-HORIZON_DAYS].copy()
    rows: List[Dict[str, object]] = []

    for _, row in usable.iterrows():
        asof_ts = pd.Timestamp(row["timestamp"])
        news_payload = summarize_news(news_df, asof_ts)
        option_payload = summarize_option_snapshot(options_df, asof_ts, float(row["close"]))
        entry_style_id = choose_entry_style(row)

        long_score = (
            0.45 * row["forward_return_5d"]
            + 0.20 * row["trend_gap"]
            + 0.10 * row["ema_gap"]
            + 0.05 * news_payload["sentiment_proxy"]
            - 0.20 * abs(row["forward_drawdown_5d"])
        )
        short_score = (
            -0.45 * row["forward_return_5d"]
            - 0.15 * row["trend_gap"]
            - 0.10 * row["ema_gap"]
            - 0.05 * news_payload["sentiment_proxy"]
            - 0.20 * row["forward_runup_5d"]
        )

        if long_score > 0.03 and row["forward_return_5d"] > 0.025 and row["forward_drawdown_5d"] > -0.04:
            direction_id = DIRECTION_LABELS.index("long")
            confidence = min(0.95, 0.55 + 4.0 * long_score)
        elif short_score > 0.03 and row["forward_return_5d"] < -0.025 and row["forward_runup_5d"] < 0.04:
            direction_id = DIRECTION_LABELS.index("short")
            confidence = min(0.95, 0.55 + 4.0 * short_score)
        else:
            direction_id = DIRECTION_LABELS.index("no_trade")
            confidence = max(0.05, 0.45 - max(long_score, short_score))

        if direction_id == DIRECTION_LABELS.index("long"):
            stop_loss_pct = float(max(0.02, min(0.08, abs(row["forward_drawdown_5d"]) * 1.15 + 0.01)))
            take_profit_pct = float(max(0.03, min(0.15, row["forward_runup_5d"] * 0.85 + 0.015)))
        elif direction_id == DIRECTION_LABELS.index("short"):
            stop_loss_pct = float(max(0.02, min(0.08, row["forward_runup_5d"] * 1.15 + 0.01)))
            take_profit_pct = float(max(0.03, min(0.15, abs(row["forward_return_5d"]) * 0.85 + 0.015)))
        else:
            stop_loss_pct = 0.0
            take_profit_pct = 0.0

        option_targets = choose_option_targets(direction_id, confidence)
        rationale_id = choose_rationale(direction_id, entry_style_id, float(row["volatility_20d"]))

        feature_payload = {
            "ret_1d": float(row["ret_1d"]),
            "ret_3d": float(row["ret_3d"]),
            "ret_5d": float(row["ret_5d"]),
            "ret_10d": float(row["ret_10d"]),
            "trend_gap": float(row["trend_gap"]),
            "ema_gap": float(row["ema_gap"]),
            "rsi14": float(row["rsi14"]) / 100.0,
            "atr_pct": safe_div(row["atr14"], row["close"]),
            "volatility_20d": float(row["volatility_20d"]),
            "volume_z20": float(row["volume_z20"]),
            "close_to_20d_high": float(row["close_to_20d_high"]),
            "close_to_20d_low": float(row["close_to_20d_low"]),
            "headline_count_3d": float(news_payload["headline_count_3d"]),
            "news_sentiment_proxy": float(news_payload["sentiment_proxy"]),
            "best_call_spread_pct": float(option_payload["best_call_spread_pct"]),
            "best_put_spread_pct": float(option_payload["best_put_spread_pct"]),
            "best_call_open_interest": float(option_payload["best_call_open_interest"]),
            "best_put_open_interest": float(option_payload["best_put_open_interest"]),
            "call_iv": float(option_payload["call_iv"]),
            "put_iv": float(option_payload["put_iv"]),
        }

        target_payload = {
            "direction_id": int(direction_id),
            "confidence": float(confidence),
            "holding_days": HORIZON_DAYS,
            "entry_style_id": int(entry_style_id),
            "stop_loss_pct": float(stop_loss_pct),
            "take_profit_pct": float(take_profit_pct),
            "use_options": int(option_targets["use_options"]),
            "option_type_id": int(option_targets["option_type_id"]),
            "dte_bucket_id": int(option_targets["dte_bucket_id"]),
            "delta_bucket_id": int(option_targets["delta_bucket_id"]),
            "rationale_id": int(rationale_id),
            "forward_return_5d": float(row["forward_return_5d"]),
            "forward_drawdown_5d": float(row["forward_drawdown_5d"]),
        }

        rows.append(
            {
                "asof_ts": asof_ts,
                "symbol": symbol,
                "feature_payload": json.dumps(feature_payload, sort_keys=True),
                "news_payload": json.dumps(news_payload, sort_keys=True),
                "option_payload": json.dumps(option_payload, sort_keys=True),
                "target_payload": json.dumps(target_payload, sort_keys=True),
                "split": pick_split(asof_ts, train_end, val_end),
            }
        )

    return rows


def build_market_dataset(
    start_date: Optional[str],
    end_date: Optional[str],
    universe: Iterable[str],
    frequency: str = "1d",
    output_path: Path = DEFAULT_DATASET_PATH,
) -> Path:
    ensure_dirs()
    symbols = sorted(set(universe))
    if not symbols:
        raise ValueError("Universe must contain at least one symbol.")

    stock_frames = {}
    for symbol in symbols:
        df = load_stock_frame(symbol)
        if start_date:
            df = df[df["timestamp"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["timestamp"] <= pd.Timestamp(end_date)]
        if len(df) < LOOKBACK_DAYS + HORIZON_DAYS + 20:
            raise ValueError(f"Not enough history for {symbol}; need at least {LOOKBACK_DAYS + HORIZON_DAYS + 20} rows.")
        stock_frames[symbol] = df.reset_index(drop=True)

    all_times = pd.concat([frame["timestamp"] for frame in stock_frames.values()]).sort_values().unique()
    train_end = pd.Timestamp(all_times[int(len(all_times) * 0.70)])
    val_end = pd.Timestamp(all_times[int(len(all_times) * 0.85)])

    rows: List[Dict[str, object]] = []
    for symbol in symbols:
        rows.extend(
            build_market_examples(
                symbol=symbol,
                stock_df=stock_frames[symbol],
                news_df=load_news_frame(symbol),
                options_df=load_options_frame(symbol),
                train_end=train_end,
                val_end=val_end,
            )
        )

    dataset = pd.DataFrame(rows).sort_values(["asof_ts", "symbol"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)

    metadata = {
        "config": DatasetConfig(start_date, end_date, symbols, frequency, output_path).to_metadata(),
        "row_count": int(len(dataset)),
        "splits": dataset["split"].value_counts().to_dict(),
        "symbols": symbols,
        "direction_labels": DIRECTION_LABELS,
        "entry_style_labels": ENTRY_STYLE_LABELS,
        "option_type_labels": OPTION_TYPE_LABELS,
        "dte_bucket_labels": DTE_BUCKET_LABELS,
        "delta_bucket_labels": DELTA_BUCKET_LABELS,
        "rationale_labels": RATIONALE_LABELS,
    }
    (output_path.parent / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return output_path


def generate_demo_market_data(symbols: Iterable[str], periods: int = 320) -> None:
    ensure_dirs()
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-03", periods=periods)
    positive_words = ["upgrade", "strong demand", "record margin", "bullish setup"]
    negative_words = ["downgrade", "soft guidance", "lawsuit", "bearish tone"]
    universe = list(symbols)

    for index, symbol in enumerate(universe):
        base_price = 70 + index * 35
        trend = 0.0008 + index * 0.0001
        noise = rng.normal(0.0, 0.018, size=periods)
        returns = trend + noise
        close = base_price * np.cumprod(1.0 + returns)
        open_ = close * (1.0 + rng.normal(0.0, 0.005, size=periods))
        high = np.maximum(open_, close) * (1.0 + rng.uniform(0.002, 0.02, size=periods))
        low = np.minimum(open_, close) * (1.0 - rng.uniform(0.002, 0.02, size=periods))
        volume = rng.integers(2_000_000, 18_000_000, size=periods)

        stock_df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
        stock_df.to_csv(STOCK_RAW_DIR / f"{symbol}.csv", index=False)

        news_rows = []
        for ts in dates[::5]:
            positive = random.random() > 0.42
            headline = positive_words[random.randrange(len(positive_words))] if positive else negative_words[random.randrange(len(negative_words))]
            summary = f"{symbol} {headline}"
            news_rows.append({"timestamp": ts, "headline": headline, "summary": summary})
        pd.DataFrame(news_rows).to_csv(NEWS_RAW_DIR / f"{symbol}.csv", index=False)

        option_rows = []
        for ts, spot in zip(dates, close):
            for option_type in ["call", "put"]:
                for strike_offset in [-0.05, 0.0, 0.05]:
                    strike = round(spot * (1.0 + strike_offset), 2)
                    expiration = ts + pd.Timedelta(days=21 + 7 * (strike_offset > 0))
                    mid = max(0.5, abs(spot - strike) * 0.15 + rng.uniform(1.0, 4.0))
                    spread = mid * rng.uniform(0.04, 0.12)
                    bid = max(0.01, mid - spread / 2.0)
                    ask = mid + spread / 2.0
                    delta = 0.55 - abs(strike_offset) * 3.0
                    if option_type == "put":
                        delta = -delta
                    option_rows.append(
                        {
                            "timestamp": ts,
                            "contract": f"{symbol}-{ts:%Y%m%d}-{option_type}-{strike:.2f}",
                            "option_type": option_type,
                            "strike": strike,
                            "expiration": expiration,
                            "delta": delta,
                            "implied_volatility": rng.uniform(0.18, 0.42),
                            "bid": bid,
                            "ask": ask,
                            "open_interest": int(rng.integers(250, 6000)),
                            "volume": int(rng.integers(50, 2500)),
                        }
                    )
        pd.DataFrame(option_rows).to_csv(OPTIONS_RAW_DIR / f"{symbol}.csv", index=False)


class MarketDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, symbol_to_id: Dict[str, int], stats: Optional[Dict[str, List[float]]] = None):
        self.frame = dataframe.reset_index(drop=True)
        feature_dicts = [json.loads(value) for value in self.frame["feature_payload"]]
        self.feature_names = sorted(feature_dicts[0].keys()) if feature_dicts else []
        features = np.array([[payload[name] for name in self.feature_names] for payload in feature_dicts], dtype=np.float32)
        if stats is None:
            mean = features.mean(axis=0)
            std = features.std(axis=0)
            std[std < 1e-6] = 1.0
            stats = {"mean": mean.tolist(), "std": std.tolist()}
        mean = np.array(stats["mean"], dtype=np.float32)
        std = np.array(stats["std"], dtype=np.float32)
        self.features = torch.tensor((features - mean) / std, dtype=torch.float32)
        self.symbol_ids = torch.tensor([symbol_to_id[sym] for sym in self.frame["symbol"]], dtype=torch.long)
        targets = [json.loads(value) for value in self.frame["target_payload"]]
        self.targets = {name: torch.tensor([target[name] for target in targets], dtype=torch.float32) for name in TARGET_COLUMNS}
        self.stats = stats

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict[str, object]:
        item = {
            "features": self.features[index],
            "symbol_id": self.symbol_ids[index],
            "row": self.frame.iloc[index].to_dict(),
        }
        for name, tensor in self.targets.items():
            item[name] = tensor[index]
        return item


def collate_market_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    rows = [item["row"] for item in batch]
    merged = {
        "features": torch.stack([item["features"] for item in batch]),
        "symbol_id": torch.stack([item["symbol_id"] for item in batch]),
        "rows": rows,
    }
    for name in TARGET_COLUMNS:
        merged[name] = torch.stack([item[name] for item in batch])
    return merged


def load_market_dataframe(path: Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run `uv run prepare.py --generate-demo-data` or provide raw CSVs.")
    frame = pd.read_parquet(path)
    frame["asof_ts"] = pd.to_datetime(frame["asof_ts"])
    return frame


def make_dataloaders(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Dict[str, object]:
    frame = load_market_dataframe(dataset_path)
    symbols = sorted(frame["symbol"].unique())
    symbol_to_id = {symbol: idx for idx, symbol in enumerate(symbols)}
    train_frame = frame[frame["split"] == "train"].reset_index(drop=True)
    val_frame = frame[frame["split"] == "val"].reset_index(drop=True)
    test_frame = frame[frame["split"] == "test"].reset_index(drop=True)

    train_dataset = MarketDataset(train_frame, symbol_to_id)
    val_dataset = MarketDataset(val_frame, symbol_to_id, train_dataset.stats)
    test_dataset = MarketDataset(test_frame, symbol_to_id, train_dataset.stats)

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_market_batch),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_market_batch),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_market_batch),
        "feature_names": train_dataset.feature_names,
        "symbol_to_id": symbol_to_id,
        "normalization_stats": train_dataset.stats,
    }


def generate_rationale_text(direction_id: int, entry_style_id: int, confidence: float, stop_loss_pct: float, take_profit_pct: float) -> str:
    direction = DIRECTION_LABELS[direction_id]
    entry_style = ENTRY_STYLE_LABELS[entry_style_id]
    if direction == "no_trade":
        return "No-trade: edge is weak or volatility is too noisy for a clean 1-5 day swing."
    if direction == "long":
        return f"Long setup: {entry_style} structure with confidence {confidence:.2f}, stop near {stop_loss_pct:.1%}, target near {take_profit_pct:.1%}."
    return f"Short setup: {entry_style} structure with confidence {confidence:.2f}, stop near {stop_loss_pct:.1%}, target near {take_profit_pct:.1%}."


def select_option_contract(
    options_df: pd.DataFrame,
    asof_ts: pd.Timestamp,
    direction_id: int,
    dte_bucket_id: int,
    delta_bucket_id: int,
) -> Optional[Dict[str, object]]:
    if options_df.empty or direction_id == DIRECTION_LABELS.index("no_trade"):
        return None
    option_type = "call" if direction_id == DIRECTION_LABELS.index("long") else "put"
    latest = options_df[options_df["timestamp"] <= asof_ts]
    if latest.empty:
        return None
    latest = latest[latest["timestamp"] == latest["timestamp"].max()].copy()
    latest["dte"] = (latest["expiration"] - asof_ts).dt.days.clip(lower=0)
    latest["mid"] = (latest["bid"] + latest["ask"]) / 2.0
    latest["spread_pct"] = np.where(latest["mid"] > 0, (latest["ask"] - latest["bid"]) / latest["mid"], 99.0)
    latest = latest[latest["option_type"].str.lower().eq(option_type)]

    dte_range = {
        DTE_BUCKET_LABELS.index("7_14"): (7, 14),
        DTE_BUCKET_LABELS.index("15_30"): (15, 30),
        DTE_BUCKET_LABELS.index("31_45"): (31, 45),
    }.get(dte_bucket_id, (0, 365))
    delta_range = {
        DELTA_BUCKET_LABELS.index("25_35"): (0.25, 0.35),
        DELTA_BUCKET_LABELS.index("35_50"): (0.35, 0.50),
        DELTA_BUCKET_LABELS.index("50_65"): (0.50, 0.65),
    }.get(delta_bucket_id, (0.0, 1.0))
    latest["abs_delta"] = latest["delta"].abs()
    latest = latest[
        latest["dte"].between(*dte_range)
        & latest["abs_delta"].between(*delta_range)
        & (latest["open_interest"] >= 250)
        & (latest["spread_pct"] <= 0.20)
    ]
    if latest.empty:
        return None
    best = latest.sort_values(["spread_pct", "open_interest", "volume"], ascending=[True, False, False]).iloc[0]
    return {
        "contract": best["contract"],
        "option_type": best["option_type"],
        "expiration": str(pd.Timestamp(best["expiration"]).date()),
        "delta": float(best["delta"]),
        "spread_pct": float(best["spread_pct"]),
        "open_interest": int(best["open_interest"]),
    }


def parse_universe(raw: str) -> List[str]:
    return [symbol.strip().upper() for symbol in raw.split(",") if symbol.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build market dataset for trading-idea research.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--universe", default="AAPL,MSFT,NVDA,SPY,QQQ")
    parser.add_argument("--frequency", default="1d")
    parser.add_argument("--output-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--generate-demo-data", action="store_true")
    args = parser.parse_args()

    universe = parse_universe(args.universe)
    if args.generate_demo_data:
        generate_demo_market_data(universe)

    output_path = build_market_dataset(
        start_date=args.start_date,
        end_date=args.end_date,
        universe=universe,
        frequency=args.frequency,
        output_path=Path(args.output_path),
    )
    frame = pd.read_parquet(output_path)
    print(f"Dataset written: {output_path}")
    print(f"Rows: {len(frame):,}")
    print("Split counts:")
    for split, count in frame["split"].value_counts().to_dict().items():
        print(f"  {split}: {count:,}")


if __name__ == "__main__":
    main()
