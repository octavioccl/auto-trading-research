"""
Multitask training script for the trading-idea autoresearch fork.

The model is intentionally compact and 3090-friendly:
- dense market features + symbol embedding
- multitask heads for structured trade outputs
- template-based rationale generation
- offline validation metrics and a simple basket backtest proxy

Usage:
    uv run train.py
    TRAIN_SECONDS=60 uv run train.py
"""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
CACHE_DIR,
    DEFAULT_DATASET_PATH,
    DELTA_BUCKET_LABELS,
    DIRECTION_LABELS,
    DTE_BUCKET_LABELS,
    ENTRY_STYLE_LABELS,
    OPTION_TYPE_LABELS,
    RATIONALE_LABELS,
    TIME_BUDGET,
    generate_rationale_text,
    load_options_frame,
    make_dataloaders,
    select_option_contract,
)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_SECONDS = int(os.environ.get("TRAIN_SECONDS", TIME_BUDGET))
DATASET_PATH = Path(os.environ.get("MARKET_DATASET_PATH", str(DEFAULT_DATASET_PATH)))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "192"))
DEPTH = int(os.environ.get("DEPTH", "4"))
DROPOUT = float(os.environ.get("DROPOUT", "0.10"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "3e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-2"))
ACTION_THRESHOLD = float(os.environ.get("ACTION_THRESHOLD", "0.55"))
MAX_BATCHES_PER_EVAL = int(os.environ.get("MAX_BATCHES_PER_EVAL", "0"))


@dataclass
class MarketModelConfig:
    input_dim: int
    num_symbols: int
    hidden_dim: int = HIDDEN_DIM
    depth: int = DEPTH
    dropout: float = DROPOUT
    symbol_dim: int = 32


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 2)
        self.fc2 = nn.Linear(width * 2, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(self.fc2(h))
        return x + h


class TradingIdeaModel(nn.Module):
    def __init__(self, config: MarketModelConfig):
        super().__init__()
        self.symbol_embedding = nn.Embedding(config.num_symbols, config.symbol_dim)
        encoder_width = config.hidden_dim
        self.feature_proj = nn.Sequential(
            nn.Linear(config.input_dim + config.symbol_dim, encoder_width),
            nn.GELU(),
            nn.LayerNorm(encoder_width),
        )
        self.blocks = nn.ModuleList([ResidualMLPBlock(encoder_width, config.dropout) for _ in range(config.depth)])
        self.dropout = nn.Dropout(config.dropout)

        self.direction_head = nn.Linear(encoder_width, len(DIRECTION_LABELS))
        self.entry_style_head = nn.Linear(encoder_width, len(ENTRY_STYLE_LABELS))
        self.option_type_head = nn.Linear(encoder_width, len(OPTION_TYPE_LABELS))
        self.dte_bucket_head = nn.Linear(encoder_width, len(DTE_BUCKET_LABELS))
        self.delta_bucket_head = nn.Linear(encoder_width, len(DELTA_BUCKET_LABELS))
        self.rationale_head = nn.Linear(encoder_width, len(RATIONALE_LABELS))
        self.confidence_head = nn.Linear(encoder_width, 1)
        self.holding_days_head = nn.Linear(encoder_width, 1)
        self.stop_loss_head = nn.Linear(encoder_width, 1)
        self.take_profit_head = nn.Linear(encoder_width, 1)
        self.use_options_head = nn.Linear(encoder_width, 1)

    def encode(self, features: torch.Tensor, symbol_id: torch.Tensor) -> torch.Tensor:
        symbol_features = self.symbol_embedding(symbol_id)
        x = torch.cat([features, symbol_features], dim=-1)
        x = self.feature_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.dropout(x)

    def forward(self, features: torch.Tensor, symbol_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.encode(features, symbol_id)
        return {
            "direction_logits": self.direction_head(x),
            "entry_style_logits": self.entry_style_head(x),
            "option_type_logits": self.option_type_head(x),
            "dte_bucket_logits": self.dte_bucket_head(x),
            "delta_bucket_logits": self.delta_bucket_head(x),
            "rationale_logits": self.rationale_head(x),
            "confidence": torch.sigmoid(self.confidence_head(x)).squeeze(-1),
            "holding_days": 1.0 + 4.0 * torch.sigmoid(self.holding_days_head(x)).squeeze(-1),
            "stop_loss_pct": 0.15 * torch.sigmoid(self.stop_loss_head(x)).squeeze(-1),
            "take_profit_pct": 0.25 * torch.sigmoid(self.take_profit_head(x)).squeeze(-1),
            "use_options_logit": self.use_options_head(x).squeeze(-1),
        }


def compute_losses(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    direction_target = batch["direction_id"].long()
    entry_target = batch["entry_style_id"].long()
    option_target = batch["option_type_id"].long()
    dte_target = batch["dte_bucket_id"].long()
    delta_target = batch["delta_bucket_id"].long()
    rationale_target = batch["rationale_id"].long()

    losses = {
        "direction": F.cross_entropy(outputs["direction_logits"], direction_target),
        "entry_style": F.cross_entropy(outputs["entry_style_logits"], entry_target),
        "option_type": F.cross_entropy(outputs["option_type_logits"], option_target),
        "dte_bucket": F.cross_entropy(outputs["dte_bucket_logits"], dte_target),
        "delta_bucket": F.cross_entropy(outputs["delta_bucket_logits"], delta_target),
        "rationale": F.cross_entropy(outputs["rationale_logits"], rationale_target),
        "confidence": F.smooth_l1_loss(outputs["confidence"], batch["confidence"]),
        "holding_days": F.smooth_l1_loss(outputs["holding_days"], batch["holding_days"]),
        "stop_loss_pct": F.smooth_l1_loss(outputs["stop_loss_pct"], batch["stop_loss_pct"]),
        "take_profit_pct": F.smooth_l1_loss(outputs["take_profit_pct"], batch["take_profit_pct"]),
        "use_options": F.binary_cross_entropy_with_logits(outputs["use_options_logit"], batch["use_options"]),
    }
    total = (
        2.0 * losses["direction"]
        + 1.2 * losses["confidence"]
        + 1.0 * losses["entry_style"]
        + 0.8 * losses["rationale"]
        + 0.7 * losses["use_options"]
        + 0.6 * losses["option_type"]
        + 0.4 * losses["dte_bucket"]
        + 0.4 * losses["delta_bucket"]
        + 0.5 * losses["holding_days"]
        + 0.6 * losses["stop_loss_pct"]
        + 0.6 * losses["take_profit_pct"]
    )
    return total, {name: float(value.detach().cpu()) for name, value in losses.items()}


def maybe_limit_batches(loader, max_batches: int):
    if max_batches <= 0:
        for batch in loader:
            yield batch
        return
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        yield batch


def to_device(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(DEVICE)
        else:
            moved[key] = value
    return moved


def calculate_backtest_metrics(predictions: List[Dict[str, object]]) -> Dict[str, float]:
    actionable = [row for row in predictions if row["predicted_direction"] != "no_trade" and row["confidence"] >= ACTION_THRESHOLD]
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


def evaluate_model(model: TradingIdeaModel, loader, options_cache: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    model.eval()
    losses = []
    direction_correct = 0
    action_tp = 0
    action_fp = 0
    action_fn = 0
    predictions: List[Dict[str, object]] = []

    with torch.no_grad():
        for batch in maybe_limit_batches(loader, MAX_BATCHES_PER_EVAL):
            batch = to_device(batch)
            outputs = model(batch["features"], batch["symbol_id"])
            loss, _ = compute_losses(outputs, batch)
            losses.append(float(loss.detach().cpu()))

            direction_pred = outputs["direction_logits"].argmax(dim=-1)
            direction_target = batch["direction_id"].long()
            direction_correct += int((direction_pred == direction_target).sum().item())

            confidence = outputs["confidence"].detach().cpu().numpy()
            use_options = torch.sigmoid(outputs["use_options_logit"]).detach().cpu().numpy()
            entry_style = outputs["entry_style_logits"].argmax(dim=-1).detach().cpu().numpy()
            option_type = outputs["option_type_logits"].argmax(dim=-1).detach().cpu().numpy()
            dte_bucket = outputs["dte_bucket_logits"].argmax(dim=-1).detach().cpu().numpy()
            delta_bucket = outputs["delta_bucket_logits"].argmax(dim=-1).detach().cpu().numpy()

            target_np = direction_target.detach().cpu().numpy()
            pred_np = direction_pred.detach().cpu().numpy()
            actionable_pred = (pred_np != DIRECTION_LABELS.index("no_trade")) & (confidence >= ACTION_THRESHOLD)
            actionable_true = target_np != DIRECTION_LABELS.index("no_trade")
            action_tp += int(np.logical_and(actionable_pred, actionable_true).sum())
            action_fp += int(np.logical_and(actionable_pred, np.logical_not(actionable_true)).sum())
            action_fn += int(np.logical_and(np.logical_not(actionable_pred), actionable_true).sum())

            rows = batch["rows"]
            for idx, row in enumerate(rows):
                target_payload = json.loads(row["target_payload"])
                symbol = row["symbol"]
                direction_label = DIRECTION_LABELS[int(pred_np[idx])]
                options_plan = None
                if use_options[idx] >= 0.5:
                    options_plan = select_option_contract(
                        options_df=options_cache.get(symbol, pd.DataFrame()),
                        asof_ts=pd.Timestamp(row["asof_ts"]),
                        direction_id=int(pred_np[idx]),
                        dte_bucket_id=int(dte_bucket[idx]),
                        delta_bucket_id=int(delta_bucket[idx]),
                    )
                predictions.append(
                    {
                        "asof_ts": str(pd.Timestamp(row["asof_ts"]).date()),
                        "symbol": symbol,
                        "predicted_direction": direction_label,
                        "target_direction": DIRECTION_LABELS[int(target_np[idx])],
                        "confidence": float(confidence[idx]),
                        "entry_style": ENTRY_STYLE_LABELS[int(entry_style[idx])],
                        "option_type": OPTION_TYPE_LABELS[int(option_type[idx])],
                        "options_plan": options_plan,
                        "rationale": generate_rationale_text(
                            direction_id=int(pred_np[idx]),
                            entry_style_id=int(entry_style[idx]),
                            confidence=float(confidence[idx]),
                            stop_loss_pct=float(outputs["stop_loss_pct"][idx].detach().cpu()),
                            take_profit_pct=float(outputs["take_profit_pct"][idx].detach().cpu()),
                        ),
                        "forward_return_5d": float(target_payload["forward_return_5d"]),
                    }
                )

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
        "loss": float(np.mean(losses)) if losses else 0.0,
        "direction_accuracy": direction_correct / max(1, len(predictions)),
        "action_precision": precision,
        "action_recall": recall,
        "backtest": backtest,
        "validation_score": validation_score,
        "predictions": predictions,
    }


def train() -> None:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    loaders = make_dataloaders(DATASET_PATH, batch_size=BATCH_SIZE)
    feature_names = loaders["feature_names"]
    symbol_to_id = loaders["symbol_to_id"]
    options_cache = {symbol: load_options_frame(symbol) for symbol in symbol_to_id}

    config = MarketModelConfig(input_dim=len(feature_names), num_symbols=len(symbol_to_id))
    model = TradingIdeaModel(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    autocast_enabled = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=autocast_enabled)

    start = time.time()
    step = 0
    running_loss = 0.0
    train_iter = iter(loaders["train"])

    while time.time() - start < TRAIN_SECONDS:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loaders["train"])
            batch = next(train_iter)
        batch = to_device(batch)

        optimizer.zero_grad(set_to_none=True)
        autocast_context = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if autocast_enabled
            else nullcontext()
        )
        with autocast_context:
            outputs = model(batch["features"], batch["symbol_id"])
            loss, loss_parts = compute_losses(outputs, batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.detach().cpu())
        if step % 25 == 0:
            elapsed = time.time() - start
            print(
                f"step {step:05d} | loss {running_loss / max(1, step + 1):.4f} | "
                f"direction {loss_parts['direction']:.4f} | confidence {loss_parts['confidence']:.4f} | "
                f"elapsed {elapsed:.1f}s",
                flush=True,
            )
        step += 1

    val_metrics = evaluate_model(model, loaders["val"], options_cache)
    test_metrics = evaluate_model(model, loaders["test"], options_cache)
    total_seconds = time.time() - start
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0

    out_dir = CACHE_DIR / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "latest_predictions.json"
    predictions_path.write_text(json.dumps(val_metrics["predictions"][:100], indent=2))
    checkpoint_path = out_dir / "latest_structured_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_dim": config.input_dim,
                "num_symbols": config.num_symbols,
                "hidden_dim": config.hidden_dim,
                "depth": config.depth,
                "dropout": config.dropout,
                "symbol_dim": config.symbol_dim,
            },
            "feature_names": feature_names,
            "symbol_to_id": symbol_to_id,
            "normalization_stats": loaders["normalization_stats"],
            "dataset_path": str(DATASET_PATH),
        },
        checkpoint_path,
    )

    print("---")
    print(f"dataset_path:             {DATASET_PATH}")
    print(f"training_seconds:        {TRAIN_SECONDS:.1f}")
    print(f"total_seconds:           {total_seconds:.1f}")
    print(f"peak_vram_mb:            {peak_vram_mb:.1f}")
    print(f"num_steps:               {step}")
    print(f"num_features:            {len(feature_names)}")
    print(f"num_symbols:             {len(symbol_to_id)}")
    print(f"val_loss:                {val_metrics['loss']:.6f}")
    print(f"val_direction_accuracy:  {val_metrics['direction_accuracy']:.4f}")
    print(f"val_action_precision:    {val_metrics['action_precision']:.4f}")
    print(f"val_action_recall:       {val_metrics['action_recall']:.4f}")
    print(f"val_avg_trade_return:    {val_metrics['backtest']['avg_trade_return']:.4f}")
    print(f"val_hit_rate:            {val_metrics['backtest']['hit_rate']:.4f}")
    print(f"val_max_drawdown:        {val_metrics['backtest']['max_drawdown']:.4f}")
    print(f"val_sharpe_proxy:        {val_metrics['backtest']['sharpe_proxy']:.4f}")
    print(f"validation_score:        {val_metrics['validation_score']:.6f}")
    print(f"test_action_precision:   {test_metrics['action_precision']:.4f}")
    print(f"test_avg_trade_return:   {test_metrics['backtest']['avg_trade_return']:.4f}")
    print(f"test_sharpe_proxy:       {test_metrics['backtest']['sharpe_proxy']:.4f}")
    print(f"predictions_path:        {predictions_path}")
    print(f"checkpoint_path:         {checkpoint_path}")


if __name__ == "__main__":
    train()
