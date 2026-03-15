from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from prepare import (
    DEFAULT_DATASET_PATH,
    DELTA_BUCKET_LABELS,
    DIRECTION_LABELS,
    DTE_BUCKET_LABELS,
    ENTRY_STYLE_LABELS,
    OPTION_TYPE_LABELS,
    RATIONALE_LABELS,
    generate_rationale_text,
)

DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen3.5-9B"
DEFAULT_ACTION_THRESHOLD = float(os.environ.get("ACTION_THRESHOLD", "0.55"))
MAX_EVIDENCE_ITEMS = 3
MAX_TEXT_CHARS = 900
BLOCKED_TRAINING_SOURCES = {"reddit", "x", "github", "youtube"}
ALLOWED_TEXT_SOURCES = {"news", "filings", "transcripts"}
REPO_ROOT = Path(__file__).resolve().parents[1]
HOME_CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "autoresearch" / "market"


def get_cache_dir() -> Path:
    env_value = os.environ.get("MARKET_CACHE_DIR")
    if env_value:
        return Path(env_value)
    repo_cache_dir = REPO_ROOT / ".cache" / "market"
    if repo_cache_dir.exists():
        return repo_cache_dir
    return HOME_CACHE_DIR


def default_llm_dataset_path() -> Path:
    return get_cache_dir() / "dataset" / "market_llm_dataset.parquet"


def default_llm_artifact_dir() -> Path:
    return get_cache_dir() / "artifacts" / "llm"


def get_text_source_dir(source: str) -> Path:
    return get_cache_dir() / "raw" / source


def ensure_llm_dirs(cache_dir: Path | None = None) -> None:
    root = cache_dir or get_cache_dir()
    for path in [
        root / "dataset",
        root / "artifacts" / "llm" / "adapters",
        root / "artifacts" / "llm" / "evals",
        root / "artifacts" / "llm" / "manifests",
        root / "raw" / "filings",
        root / "raw" / "transcripts",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_json(value: str | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return json.loads(value)


def _clip_text(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."


def _completion_direction(direction_id: int) -> str:
    return DIRECTION_LABELS[int(direction_id)]


def _completion_entry_style(entry_style_id: int) -> str:
    return ENTRY_STYLE_LABELS[int(entry_style_id)]


def _completion_option_type(option_type_id: int) -> str:
    return OPTION_TYPE_LABELS[int(option_type_id)]


def _completion_dte_bucket(dte_bucket_id: int) -> str:
    return DTE_BUCKET_LABELS[int(dte_bucket_id)]


def _completion_delta_bucket(delta_bucket_id: int) -> str:
    return DELTA_BUCKET_LABELS[int(delta_bucket_id)]


def _completion_rationale_label(rationale_id: int) -> str:
    return RATIONALE_LABELS[int(rationale_id)]


def build_target_json(target_payload: Dict[str, Any]) -> Dict[str, Any]:
    direction_id = int(target_payload["direction_id"])
    entry_style_id = int(target_payload["entry_style_id"])
    payload = {
        "direction": _completion_direction(direction_id),
        "confidence": round(float(target_payload["confidence"]), 4),
        "holding_days": int(round(float(target_payload["holding_days"]))),
        "entry_style": _completion_entry_style(entry_style_id),
        "stop_loss_pct": round(float(target_payload["stop_loss_pct"]), 4),
        "take_profit_pct": round(float(target_payload["take_profit_pct"]), 4),
        "use_options": bool(int(target_payload["use_options"])),
        "option_type": _completion_option_type(int(target_payload["option_type_id"])),
        "dte_bucket": _completion_dte_bucket(int(target_payload["dte_bucket_id"])),
        "delta_bucket": _completion_delta_bucket(int(target_payload["delta_bucket_id"])),
        "rationale": generate_rationale_text(
            direction_id=direction_id,
            entry_style_id=entry_style_id,
            confidence=float(target_payload["confidence"]),
            stop_loss_pct=float(target_payload["stop_loss_pct"]),
            take_profit_pct=float(target_payload["take_profit_pct"]),
        ),
        "rationale_label": _completion_rationale_label(int(target_payload["rationale_id"])),
    }
    if payload["direction"] == "no_trade":
        payload["use_options"] = False
        payload["option_type"] = "none"
        payload["dte_bucket"] = "none"
        payload["delta_bucket"] = "none"
    return payload


def _heuristic_scores(feature_payload: Dict[str, Any]) -> tuple[float, float]:
    long_score = (
        0.40 * float(feature_payload["trend_gap"])
        + 0.25 * float(feature_payload["ema_gap"])
        + 0.18 * float(feature_payload["ret_5d"])
        + 0.08 * float(feature_payload["ret_10d"])
        + 0.05 * float(feature_payload["news_sentiment_proxy"])
        - 0.10 * float(feature_payload["volatility_20d"])
    )
    short_score = (
        -0.40 * float(feature_payload["trend_gap"])
        - 0.25 * float(feature_payload["ema_gap"])
        - 0.18 * float(feature_payload["ret_5d"])
        - 0.08 * float(feature_payload["ret_10d"])
        - 0.05 * float(feature_payload["news_sentiment_proxy"])
        - 0.10 * float(feature_payload["volatility_20d"])
    )
    return long_score, short_score


def build_heuristic_baseline_summary(feature_payload: Dict[str, Any]) -> Dict[str, Any]:
    long_score, short_score = _heuristic_scores(feature_payload)
    if long_score > 0.025 and long_score >= short_score:
        direction = "long"
        confidence = min(0.9, 0.52 + long_score * 3.0)
    elif short_score > 0.025:
        direction = "short"
        confidence = min(0.9, 0.52 + short_score * 3.0)
    else:
        direction = "no_trade"
        confidence = max(0.15, 0.46 - max(long_score, short_score))

    if float(feature_payload["close_to_20d_high"]) > -0.01 and float(feature_payload["ret_5d"]) > 0:
        entry_style = "breakout"
    elif float(feature_payload["trend_gap"]) > 0 and float(feature_payload["ret_3d"]) < 0:
        entry_style = "pullback"
    elif float(feature_payload["rsi14"]) > 0.68 or float(feature_payload["rsi14"]) < 0.32:
        entry_style = "mean_revert"
    else:
        entry_style = "other"

    volatility = float(feature_payload["volatility_20d"])
    stop_loss_pct = 0.0 if direction == "no_trade" else round(max(0.02, min(0.08, volatility * 1.6 + 0.01)), 4)
    take_profit_pct = 0.0 if direction == "no_trade" else round(max(0.03, min(0.15, abs(float(feature_payload["ret_10d"])) * 1.1 + 0.03)), 4)
    use_options = direction != "no_trade" and confidence >= 0.64
    direction_id = DIRECTION_LABELS.index(direction)
    entry_style_id = ENTRY_STYLE_LABELS.index(entry_style)
    return {
        "source": "heuristic_v1",
        "direction": direction,
        "confidence": round(float(confidence), 4),
        "entry_style": entry_style,
        "holding_days": 5,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "use_options": use_options,
        "option_type": "call" if use_options and direction == "long" else "put" if use_options and direction == "short" else "none",
        "dte_bucket": "15_30" if use_options else "none",
        "delta_bucket": "35_50" if use_options else "none",
        "rationale": generate_rationale_text(direction_id, entry_style_id, float(confidence), stop_loss_pct, take_profit_pct),
    }


def extract_numeric_context(feature_payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ret_1d": round(float(feature_payload["ret_1d"]), 4),
        "ret_3d": round(float(feature_payload["ret_3d"]), 4),
        "ret_5d": round(float(feature_payload["ret_5d"]), 4),
        "ret_10d": round(float(feature_payload["ret_10d"]), 4),
        "trend_gap": round(float(feature_payload["trend_gap"]), 4),
        "ema_gap": round(float(feature_payload["ema_gap"]), 4),
        "rsi14": round(float(feature_payload["rsi14"]), 4),
        "atr_pct": round(float(feature_payload["atr_pct"]), 4),
        "volatility_20d": round(float(feature_payload["volatility_20d"]), 4),
        "volume_z20": round(float(feature_payload["volume_z20"]), 4),
        "close_to_20d_high": round(float(feature_payload["close_to_20d_high"]), 4),
        "close_to_20d_low": round(float(feature_payload["close_to_20d_low"]), 4),
        "headline_count_3d": int(round(float(feature_payload["headline_count_3d"]))),
        "news_sentiment_proxy": round(float(feature_payload["news_sentiment_proxy"]), 4),
        "call_iv": round(float(feature_payload["call_iv"]), 4),
        "put_iv": round(float(feature_payload["put_iv"]), 4),
    }


def _format_lines(mapping: Dict[str, Any]) -> str:
    return "\n".join(f"- {key}: {value}" for key, value in mapping.items())


def build_prompt_text(
    *,
    symbol: str,
    asof_ts: pd.Timestamp,
    numeric_context: Dict[str, Any],
    baseline_context: Optional[Dict[str, Any]],
    source_manifest: Dict[str, Any],
    include_baseline: bool = True,
) -> str:
    sections = [
        "Return only a single JSON object.",
        "Do not output analysis, markdown, commentary, or <think> tags.",
        f"Symbol: {symbol}",
        f"As-Of Date: {pd.Timestamp(asof_ts).date()}",
        "Numeric Context:",
        _format_lines(numeric_context),
    ]
    if include_baseline and baseline_context:
        sections.extend(["Structured Baseline Summary:", _format_lines(baseline_context)])

    evidence = source_manifest.get("evidence", {})
    news_items = evidence.get("news", [])
    filing_items = evidence.get("filings", [])
    transcript_items = evidence.get("transcripts", [])
    if news_items:
        sections.append("Recent News:")
        sections.extend(f"- {item}" for item in news_items)
    if filing_items:
        sections.append("Recent Filing Excerpts:")
        sections.extend(f"- {item}" for item in filing_items)
    if transcript_items:
        sections.append("Recent Earnings Transcript Excerpts:")
        sections.extend(f"- {item}" for item in transcript_items)

    sections.extend(
        [
            "Required JSON keys:",
            "- direction",
            "- confidence",
            "- holding_days",
            "- entry_style",
            "- stop_loss_pct",
            "- take_profit_pct",
            "- use_options",
            "- option_type",
            "- dte_bucket",
            "- delta_bucket",
            "- rationale",
        ]
    )
    return "\n".join(sections)


def parse_completion_text(raw_text: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "missing_json_object"
    candidate = cleaned[start : end + 1]
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error:{exc.msg}"


def validate_completion_payload(payload: Dict[str, Any]) -> Optional[str]:
    required = {
        "direction": str,
        "confidence": (int, float),
        "holding_days": int,
        "entry_style": str,
        "stop_loss_pct": (int, float),
        "take_profit_pct": (int, float),
        "use_options": bool,
        "option_type": str,
        "dte_bucket": str,
        "delta_bucket": str,
        "rationale": str,
    }
    for key, expected in required.items():
        if key not in payload:
            return f"missing_{key}"
        if not isinstance(payload[key], expected):
            return f"invalid_type_{key}"
    if payload["direction"] not in set(DIRECTION_LABELS):
        return "invalid_direction"
    if payload["entry_style"] not in set(ENTRY_STYLE_LABELS):
        return "invalid_entry_style"
    if payload["option_type"] not in set(OPTION_TYPE_LABELS):
        return "invalid_option_type"
    if payload["dte_bucket"] not in set(DTE_BUCKET_LABELS):
        return "invalid_dte_bucket"
    if payload["delta_bucket"] not in set(DELTA_BUCKET_LABELS):
        return "invalid_delta_bucket"
    if payload["direction"] == "no_trade" and payload["use_options"]:
        return "no_trade_cannot_use_options"
    return None


def build_source_manifest(
    *,
    news_items: Iterable[str],
    filing_items: Iterable[str],
    transcript_items: Iterable[str],
    blocked_present: Iterable[str],
) -> Dict[str, Any]:
    evidence = {
        "news": [_clip_text(item) for item in news_items if item][:MAX_EVIDENCE_ITEMS],
        "filings": [_clip_text(item) for item in filing_items if item][:MAX_EVIDENCE_ITEMS],
        "transcripts": [_clip_text(item) for item in transcript_items if item][:MAX_EVIDENCE_ITEMS],
    }
    policy = {
        "allowed_training_sources": sorted(ALLOWED_TEXT_SOURCES),
        "blocked_training_sources": sorted(BLOCKED_TRAINING_SOURCES),
        "blocked_present": sorted(set(blocked_present)),
    }
    return {"evidence": evidence, "policy": policy}


def compute_checksum(records: Iterable[Dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def load_optional_text_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["published_at", "title", "text"])
    if path.suffix == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".jsonl":
        frame = pd.read_json(path, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported corpus file format: {path}")
    columns = {column.lower(): column for column in frame.columns}
    published_column = columns.get("published_at") or columns.get("timestamp")
    title_column = columns.get("title") or columns.get("headline")
    text_column = columns.get("text") or columns.get("summary") or columns.get("content")
    if not published_column or not text_column:
        return pd.DataFrame(columns=["published_at", "title", "text"])
    result = pd.DataFrame(
        {
            "published_at": pd.to_datetime(frame[published_column], utc=True, errors="coerce").dt.tz_convert(None),
            "title": frame[title_column].fillna("").astype(str) if title_column else "",
            "text": frame[text_column].fillna("").astype(str),
        }
    )
    return result.dropna(subset=["published_at"]).sort_values("published_at").reset_index(drop=True)


def select_text_evidence(frame: pd.DataFrame, asof_ts: pd.Timestamp, max_items: int = MAX_EVIDENCE_ITEMS, lookback_days: int = 90) -> list[str]:
    if frame.empty:
        return []
    start_ts = pd.Timestamp(asof_ts) - pd.Timedelta(days=lookback_days)
    subset = frame[(frame["published_at"] <= pd.Timestamp(asof_ts)) & (frame["published_at"] >= start_ts)].tail(max_items)
    items = []
    for _, row in subset.iterrows():
        title = str(row.get("title", "")).strip()
        text = str(row.get("text", "")).strip()
        merged = " - ".join(part for part in [title, text] if part)
        if merged:
            items.append(_clip_text(merged))
    return items


def find_blocked_source_files() -> list[str]:
    blocked = []
    for source in BLOCKED_TRAINING_SOURCES:
        source_dir = get_text_source_dir(source)
        if source_dir.exists() and any(source_dir.iterdir()):
            blocked.append(source)
    return blocked


def normalize_prediction_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    normalized["direction"] = str(normalized["direction"]).lower().strip()
    normalized["entry_style"] = str(normalized["entry_style"]).lower().strip()
    normalized["option_type"] = str(normalized["option_type"]).lower().strip()
    normalized["dte_bucket"] = str(normalized["dte_bucket"]).strip()
    normalized["delta_bucket"] = str(normalized["delta_bucket"]).strip()
    normalized["confidence"] = float(normalized["confidence"])
    normalized["holding_days"] = int(normalized["holding_days"])
    normalized["stop_loss_pct"] = float(normalized["stop_loss_pct"])
    normalized["take_profit_pct"] = float(normalized["take_profit_pct"])
    normalized["use_options"] = bool(normalized["use_options"])
    normalized["rationale"] = str(normalized["rationale"]).strip()
    return normalized


def calculate_backtest_metrics(predictions: list[Dict[str, Any]]) -> Dict[str, float]:
    actionable = [
        row
        for row in predictions
        if row["predicted_direction"] != "no_trade" and row["confidence"] >= DEFAULT_ACTION_THRESHOLD
    ]
    if not actionable:
        return {
            "action_rate": 0.0,
            "avg_forward_return": 0.0,
            "hit_rate": 0.0,
            "avg_trade_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_proxy": 0.0,
            "turnover": 0.0,
        }
    trade_returns = []
    equity_curve = [1.0]
    hits = 0
    for row in actionable:
        signed_return = row["forward_return_5d"] if row["predicted_direction"] == "long" else -row["forward_return_5d"]
        trade_returns.append(signed_return)
        hits += int(signed_return > 0)
        equity_curve.append(equity_curve[-1] * (1.0 + signed_return))
    equity_curve = np.array(equity_curve)
    rolling_peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve / rolling_peak) - 1.0
    sharpe_proxy = np.mean(trade_returns) / (np.std(trade_returns) + 1e-6) * math.sqrt(12)
    return {
        "action_rate": len(actionable) / max(1, len(predictions)),
        "avg_forward_return": float(np.mean([row["forward_return_5d"] for row in actionable])),
        "hit_rate": hits / len(actionable),
        "avg_trade_return": float(np.mean(trade_returns)),
        "max_drawdown": float(drawdowns.min()),
        "sharpe_proxy": float(sharpe_proxy),
        "turnover": len(actionable) / max(1, len({row["asof_ts"] for row in predictions})),
    }


def compute_validation_from_predictions(predictions: list[Dict[str, Any]]) -> Dict[str, Any]:
    if not predictions:
        backtest = calculate_backtest_metrics([])
        return {
            "direction_accuracy": 0.0,
            "action_precision": 0.0,
            "action_recall": 0.0,
            "backtest": backtest,
            "validation_score": 0.0,
            "predictions": [],
        }
    action_tp = 0
    action_fp = 0
    action_fn = 0
    direction_correct = 0
    for row in predictions:
        predicted_direction = row["predicted_direction"]
        target_direction = row["target_direction"]
        if predicted_direction == target_direction:
            direction_correct += 1
        actionable_pred = predicted_direction != "no_trade" and row["confidence"] >= DEFAULT_ACTION_THRESHOLD
        actionable_true = target_direction != "no_trade"
        if actionable_pred and actionable_true:
            action_tp += 1
        elif actionable_pred and not actionable_true:
            action_fp += 1
        elif actionable_true and not actionable_pred:
            action_fn += 1
    precision = action_tp / max(1, action_tp + action_fp)
    recall = action_tp / max(1, action_tp + action_fn)
    backtest = calculate_backtest_metrics(predictions)
    validation_score = (
        0.40 * precision
        + 0.20 * recall
        + 0.25 * max(0.0, backtest["avg_trade_return"])
        + 0.15 * max(0.0, backtest["sharpe_proxy"] / 10.0)
    )
    return {
        "direction_accuracy": direction_correct / max(1, len(predictions)),
        "action_precision": precision,
        "action_recall": recall,
        "backtest": backtest,
        "validation_score": validation_score,
        "predictions": predictions,
    }


def default_market_dataset_path() -> Path:
    env_value = os.environ.get("MARKET_DATASET_PATH")
    if env_value:
        return Path(env_value)
    repo_dataset = get_cache_dir() / "dataset" / DEFAULT_DATASET_PATH.name
    if repo_dataset.exists():
        return repo_dataset
    return DEFAULT_DATASET_PATH
