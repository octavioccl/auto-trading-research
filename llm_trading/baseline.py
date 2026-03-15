from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from prepare import DELTA_BUCKET_LABELS, DIRECTION_LABELS, DTE_BUCKET_LABELS, ENTRY_STYLE_LABELS, OPTION_TYPE_LABELS, generate_rationale_text

from .common import build_heuristic_baseline_summary, extract_numeric_context


def latest_structured_checkpoint(cache_dir: Path) -> Path:
    return cache_dir / "artifacts" / "latest_structured_model.pt"


def load_structured_baseline_predictions(frame: pd.DataFrame, checkpoint_path: Path) -> Dict[str, Dict[str, Any]]:
    import torch

    from train import MarketModelConfig, TradingIdeaModel

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    feature_names = checkpoint["feature_names"]
    symbol_to_id = checkpoint["symbol_to_id"]
    stats = checkpoint["normalization_stats"]
    config = MarketModelConfig(
        input_dim=len(feature_names),
        num_symbols=len(symbol_to_id),
        hidden_dim=checkpoint["model_config"]["hidden_dim"],
        depth=checkpoint["model_config"]["depth"],
        dropout=checkpoint["model_config"]["dropout"],
        symbol_dim=checkpoint["model_config"]["symbol_dim"],
    )
    model = TradingIdeaModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = frame.reset_index(drop=True)
    feature_payloads = [json.loads(value) for value in rows["feature_payload"]]
    features = np.array([[payload[name] for name in feature_names] for payload in feature_payloads], dtype=np.float32)
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    std[std < 1e-6] = 1.0
    normalized = torch.tensor((features - mean) / std, dtype=torch.float32)
    symbol_ids = torch.tensor([symbol_to_id[str(symbol)] for symbol in rows["symbol"]], dtype=torch.long)

    outputs_map: Dict[str, Dict[str, Any]] = {}
    with torch.no_grad():
        outputs = model(normalized, symbol_ids)
        direction = outputs["direction_logits"].argmax(dim=-1).cpu().numpy()
        entry_style = outputs["entry_style_logits"].argmax(dim=-1).cpu().numpy()
        option_type = outputs["option_type_logits"].argmax(dim=-1).cpu().numpy()
        dte_bucket = outputs["dte_bucket_logits"].argmax(dim=-1).cpu().numpy()
        delta_bucket = outputs["delta_bucket_logits"].argmax(dim=-1).cpu().numpy()
        confidence = outputs["confidence"].cpu().numpy()
        stop_loss_pct = outputs["stop_loss_pct"].cpu().numpy()
        take_profit_pct = outputs["take_profit_pct"].cpu().numpy()
        holding_days = outputs["holding_days"].cpu().numpy()
        use_options = torch.sigmoid(outputs["use_options_logit"]).cpu().numpy()

    for idx, row in rows.iterrows():
        key = f"{pd.Timestamp(row['asof_ts']).isoformat()}::{row['symbol']}"
        outputs_map[key] = {
            "source": "structured_checkpoint",
            "direction": DIRECTION_LABELS[int(direction[idx])],
            "confidence": round(float(confidence[idx]), 4),
            "entry_style": ENTRY_STYLE_LABELS[int(entry_style[idx])],
            "holding_days": int(round(float(holding_days[idx]))),
            "stop_loss_pct": round(float(stop_loss_pct[idx]), 4),
            "take_profit_pct": round(float(take_profit_pct[idx]), 4),
            "use_options": bool(use_options[idx] >= 0.5),
            "option_type": OPTION_TYPE_LABELS[int(option_type[idx])],
            "dte_bucket": DTE_BUCKET_LABELS[int(dte_bucket[idx])],
            "delta_bucket": DELTA_BUCKET_LABELS[int(delta_bucket[idx])],
            "rationale": generate_rationale_text(
                int(direction[idx]),
                int(entry_style[idx]),
                float(confidence[idx]),
                float(stop_loss_pct[idx]),
                float(take_profit_pct[idx]),
            ),
        }
    return outputs_map


def resolve_baseline_summary(frame_row: pd.Series, cache_dir: Path, checkpoint_predictions: Dict[str, Dict[str, Any]] | None = None) -> Dict[str, Any]:
    key = f"{pd.Timestamp(frame_row['asof_ts']).isoformat()}::{frame_row['symbol']}"
    if checkpoint_predictions and key in checkpoint_predictions:
        return checkpoint_predictions[key]
    feature_payload = extract_numeric_context(json.loads(frame_row["feature_payload"]))
    return build_heuristic_baseline_summary(feature_payload)
