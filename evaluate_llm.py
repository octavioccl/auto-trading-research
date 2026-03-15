from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from llm_trading.common import (
    DEFAULT_QWEN_MODEL_ID,
    TRADING_SYSTEM_PROMPT,
    build_prompt_text,
    compute_validation_from_predictions,
    default_llm_artifact_dir,
    default_llm_dataset_path,
    normalize_prediction_payload,
    parse_completion_text,
    preferred_compute_dtype,
    validate_completion_payload,
)


def _require_llm_dependencies():
    try:
        from peft import PeftModel  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Missing LLM inference dependencies. Install transformers, peft, bitsandbytes, and accelerate."
        ) from exc


def _fallback_prediction() -> Dict[str, Any]:
    return {
        "direction": "no_trade",
        "confidence": 0.0,
        "holding_days": 5,
        "entry_style": "other",
        "stop_loss_pct": 0.0,
        "take_profit_pct": 0.0,
        "use_options": False,
        "option_type": "none",
        "dte_bucket": "none",
        "delta_bucket": "none",
        "rationale": "Invalid model output fallback.",
    }


def _build_generation_prompt(row: pd.Series, include_baseline: bool) -> str:
    return build_prompt_text(
        symbol=str(row["symbol"]),
        asof_ts=pd.Timestamp(row["asof_ts"]),
        numeric_context=json.loads(row["numeric_context_json"]),
        baseline_context=json.loads(row["baseline_context_json"]),
        source_manifest=json.loads(row["source_manifest_json"]),
        include_baseline=include_baseline,
    )


def _target_meta(row: pd.Series) -> Dict[str, Any]:
    payload = json.loads(row["target_json"])
    return payload.get("_meta", {})


def _build_prediction_row(row: pd.Series, normalized: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "asof_ts": str(pd.Timestamp(row["asof_ts"]).date()),
        "symbol": row["symbol"],
        "predicted_direction": normalized["direction"],
        "target_direction": meta["target_direction"],
        "confidence": normalized["confidence"],
        "entry_style": normalized["entry_style"],
        "option_type": normalized["option_type"],
        "options_plan": None,
        "rationale": normalized.get("rationale", ""),
        "forward_return_5d": meta["forward_return_5d"],
    }


def _structured_baseline_predictions(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    predictions = []
    for _, row in frame.iterrows():
        baseline = normalize_prediction_payload(json.loads(row["baseline_context_json"]))
        meta = _target_meta(row)
        predictions.append(_build_prediction_row(row, baseline, meta))
    return predictions


def _load_generation_model(base_model: str, adapter_path: Path):
    _require_llm_dependencies()
    import torch

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(adapter_path if adapter_path.exists() else base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=preferred_compute_dtype(),
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return model, tokenizer


def _generate_completion(model, tokenizer, prompt_text: str, max_new_tokens: int) -> str:
    import torch

    messages = [
        {"role": "system", "content": TRADING_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{TRADING_SYSTEM_PROMPT}\n\n{prompt_text}\n"
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion_ids = generated[0][encoded["input_ids"].shape[1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def _llm_predictions(frame: pd.DataFrame, base_model: str, adapter_path: Path, include_baseline: bool, max_new_tokens: int) -> tuple[List[Dict[str, Any]], int, int]:
    model, tokenizer = _load_generation_model(base_model, adapter_path)
    predictions = []
    parsed = 0
    schema_valid = 0
    for _, row in frame.iterrows():
        completion_text = _generate_completion(model, tokenizer, _build_generation_prompt(row, include_baseline), max_new_tokens)
        payload, parse_error = parse_completion_text(completion_text)
        if parse_error is None and payload is not None:
            parsed += 1
        if payload is None:
            normalized = _fallback_prediction()
        else:
            error = validate_completion_payload(payload)
            if error is None:
                schema_valid += 1
                normalized = normalize_prediction_payload(payload)
            else:
                normalized = _fallback_prediction()
        meta = _target_meta(row)
        predictions.append(_build_prediction_row(row, normalized, meta))
    return predictions, parsed, schema_valid


def _walk_forward_folds(frame: pd.DataFrame, folds: int) -> List[pd.DataFrame]:
    normalized_dates = pd.to_datetime(frame["asof_ts"]).dt.normalize()
    unique_times = sorted(normalized_dates.unique())
    if not unique_times:
        return []
    buckets = np.array_split(unique_times, folds)
    fold_frames = []
    for bucket in buckets:
        if len(bucket) == 0:
            continue
        fold_frames.append(frame[normalized_dates.isin(bucket)].reset_index(drop=True))
    return fold_frames


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    frame = pd.read_parquet(args.dataset_path)
    frame["asof_ts"] = pd.to_datetime(frame["asof_ts"])
    if args.split != "all":
        frame = frame[frame["split"] == args.split].reset_index(drop=True)
    if args.max_samples > 0:
        frame = frame.head(args.max_samples)

    folds = _walk_forward_folds(frame, args.walk_forward_folds)
    fold_summaries = []
    parsed_total = 0
    schema_valid_total = 0
    total_predictions = 0
    for fold_index, fold_frame in enumerate(folds, start=1):
        if args.track == "structured_baseline":
            predictions = _structured_baseline_predictions(fold_frame)
            parsed = len(predictions)
            schema_valid = len(predictions)
        else:
            predictions, parsed, schema_valid = _llm_predictions(
                fold_frame,
                base_model=args.base_model,
                adapter_path=Path(args.adapter_path),
                include_baseline=args.track == "llm_hybrid",
                max_new_tokens=args.max_new_tokens,
            )
        summary = compute_validation_from_predictions(predictions)
        summary["fold_index"] = fold_index
        summary["rows"] = int(len(fold_frame))
        fold_summaries.append(summary)
        parsed_total += parsed
        schema_valid_total += schema_valid
        total_predictions += len(predictions)

    aggregate = {
        "track": args.track,
        "split": args.split,
        "rows": total_predictions,
        "walk_forward_folds": len(fold_summaries),
        "json_parse_rate": parsed_total / max(1, total_predictions),
        "schema_valid_rate": schema_valid_total / max(1, total_predictions),
        "mean_validation_score": float(np.mean([fold["validation_score"] for fold in fold_summaries])) if fold_summaries else 0.0,
        "mean_action_precision": float(np.mean([fold["action_precision"] for fold in fold_summaries])) if fold_summaries else 0.0,
        "mean_action_recall": float(np.mean([fold["action_recall"] for fold in fold_summaries])) if fold_summaries else 0.0,
        "mean_max_drawdown": float(np.mean([fold["backtest"]["max_drawdown"] for fold in fold_summaries])) if fold_summaries else 0.0,
        "folds": fold_summaries,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2))
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the LLM trading pipeline against the current validation score.")
    parser.add_argument("--dataset-path", default=str(default_llm_dataset_path()))
    parser.add_argument("--adapter-path", default=str(default_llm_artifact_dir() / "adapters" / "qwen35_9b_lora"))
    parser.add_argument("--base-model", default=DEFAULT_QWEN_MODEL_ID)
    parser.add_argument("--track", choices=["structured_baseline", "llm_text_only", "llm_hybrid"], default="structured_baseline")
    parser.add_argument("--split", choices=["all", "val", "test"], default="all")
    parser.add_argument("--walk-forward-folds", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-path", default=str(default_llm_artifact_dir() / "evals" / "latest_llm_eval.json"))
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
