from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from llm_trading.baseline import latest_structured_checkpoint, load_structured_baseline_predictions, resolve_baseline_summary
from llm_trading.common import (
    build_prompt_text,
    build_source_manifest,
    build_target_json,
    compute_checksum,
    default_llm_dataset_path,
    default_market_dataset_path,
    ensure_llm_dirs,
    extract_numeric_context,
    find_blocked_source_files,
    get_cache_dir,
    get_text_source_dir,
    load_json,
    load_optional_text_frame,
    select_text_evidence,
)
from prepare import load_market_dataframe, parse_universe


def _find_symbol_corpus_file(source: str, symbol: str) -> Path | None:
    source_dir = get_text_source_dir(source)
    for suffix in [".parquet", ".jsonl", ".csv"]:
        candidate = source_dir / f"{symbol}{suffix}"
        if candidate.exists():
            return candidate
    return None


def load_symbol_text_frames(symbols: Iterable[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    cache: Dict[str, Dict[str, pd.DataFrame]] = {}
    for symbol in symbols:
        cache[symbol] = {}
        for source in ["filings", "transcripts"]:
            file_path = _find_symbol_corpus_file(source, symbol)
            cache[symbol][source] = load_optional_text_frame(file_path) if file_path else pd.DataFrame(columns=["published_at", "title", "text"])
    return cache


def _build_row(
    row: pd.Series,
    text_frames: Dict[str, Dict[str, pd.DataFrame]],
    blocked_sources_present: List[str],
    baseline_predictions: Dict[str, Dict[str, object]] | None,
) -> Dict[str, object]:
    asof_ts = pd.Timestamp(row["asof_ts"])
    symbol = str(row["symbol"])
    feature_payload = load_json(row["feature_payload"])
    news_payload = load_json(row["news_payload"])
    target_payload = load_json(row["target_payload"])

    numeric_context = extract_numeric_context(feature_payload)
    baseline_context = resolve_baseline_summary(row, get_cache_dir(), baseline_predictions)
    filing_items = select_text_evidence(text_frames[symbol]["filings"], asof_ts)
    transcript_items = select_text_evidence(text_frames[symbol]["transcripts"], asof_ts)
    news_items = list(news_payload.get("latest_headlines", []))
    source_manifest = build_source_manifest(
        news_items=news_items,
        filing_items=filing_items,
        transcript_items=transcript_items,
        blocked_present=blocked_sources_present,
    )
    completion_payload = build_target_json(target_payload)
    target_json = {
        **completion_payload,
        "_meta": {
            "target_direction": completion_payload["direction"],
            "forward_return_5d": float(target_payload["forward_return_5d"]),
            "forward_drawdown_5d": float(target_payload["forward_drawdown_5d"]),
        },
    }
    prompt_text = build_prompt_text(
        symbol=symbol,
        asof_ts=asof_ts,
        numeric_context=numeric_context,
        baseline_context=baseline_context,
        source_manifest=source_manifest,
        include_baseline=True,
    )
    return {
        "asof_ts": asof_ts,
        "symbol": symbol,
        "split": row["split"],
        "prompt_text": prompt_text,
        "completion_text": json.dumps(completion_payload, sort_keys=True),
        "target_json": json.dumps(target_json, sort_keys=True),
        "numeric_context_json": json.dumps(numeric_context, sort_keys=True),
        "baseline_context_json": json.dumps(baseline_context, sort_keys=True),
        "source_manifest_json": json.dumps(source_manifest, sort_keys=True),
    }


def build_market_llm_dataset(
    dataset_path: Path,
    output_path: Path,
    start_date: str | None = None,
    end_date: str | None = None,
    universe: Iterable[str] | None = None,
) -> Path:
    ensure_llm_dirs(output_path.resolve().parents[1])
    frame = load_market_dataframe(dataset_path)
    if start_date:
        frame = frame[frame["asof_ts"] >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame[frame["asof_ts"] <= pd.Timestamp(end_date)]
    if universe:
        symbols = sorted(set(universe))
        frame = frame[frame["symbol"].isin(symbols)]
    else:
        symbols = sorted(frame["symbol"].unique())
    if frame.empty:
        raise ValueError("No rows available to build the LLM dataset.")

    text_frames = load_symbol_text_frames(symbols)
    blocked_sources_present = find_blocked_source_files()
    checkpoint_path = latest_structured_checkpoint(get_cache_dir())
    baseline_predictions = None
    if checkpoint_path.exists():
        try:
            baseline_predictions = load_structured_baseline_predictions(frame, checkpoint_path)
        except Exception as exc:
            print(f"Warning: failed to load structured baseline checkpoint, falling back to heuristics: {exc}")

    rows = [
        _build_row(row, text_frames, blocked_sources_present, baseline_predictions)
        for _, row in frame.sort_values(["asof_ts", "symbol"]).iterrows()
    ]
    dataset = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)

    records_for_checksum = [
        {
            "asof_ts": str(row["asof_ts"]),
            "symbol": row["symbol"],
            "split": row["split"],
            "prompt_sha": compute_checksum([{"prompt_text": row["prompt_text"]}]),
            "completion_sha": compute_checksum([{"completion_text": row["completion_text"]}]),
        }
        for row in rows
    ]
    metadata = {
        "input_market_dataset_path": str(dataset_path),
        "output_path": str(output_path),
        "row_count": int(len(dataset)),
        "symbols": symbols,
        "splits": dataset["split"].value_counts().to_dict(),
        "blocked_training_sources_present": blocked_sources_present,
        "baseline_source": "structured_checkpoint" if baseline_predictions else "heuristic_v1",
        "allowed_sources": ["news", "filings", "transcripts"],
        "llm_model_id": "Qwen/Qwen3.5-9B",
        "dataset_checksum": compute_checksum(records_for_checksum),
    }
    metadata_path = output_path.parent / "market_llm_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    manifest_path = get_cache_dir() / "artifacts" / "llm" / "manifests" / "market_llm_dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(metadata, indent=2))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build instruction dataset for the Qwen trading pipeline.")
    parser.add_argument("--market-dataset-path", default=str(default_market_dataset_path()))
    parser.add_argument("--output-path", default=str(default_llm_dataset_path()))
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--universe", default=None)
    args = parser.parse_args()

    output_path = build_market_llm_dataset(
        dataset_path=Path(args.market_dataset_path),
        output_path=Path(args.output_path),
        start_date=args.start_date,
        end_date=args.end_date,
        universe=parse_universe(args.universe) if args.universe else None,
    )
    dataset = pd.read_parquet(output_path)
    print(f"LLM dataset written: {output_path}")
    print(f"Rows: {len(dataset):,}")
    print("Split counts:")
    for split, count in dataset["split"].value_counts().to_dict().items():
        print(f"  {split}: {count:,}")


if __name__ == "__main__":
    main()
