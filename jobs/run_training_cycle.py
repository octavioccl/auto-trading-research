from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from market_data_service.config import MarketDataConfig, parse_symbols
from market_data_service.service import refresh_market_data

from .refresh_market_data import _run_prepare


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh market data, rebuild the dataset, and run training."
    )
    parser.add_argument("--pipeline", choices=["structured", "llm_hybrid"], default="structured")
    parser.add_argument("--provider", default="polygon")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--symbols", default="AAPL,MSFT,NVDA,SPY,QQQ")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--include-news", action="store_true")
    parser.add_argument("--include-options", action="store_true")
    parser.add_argument("--mode", choices=["backfill", "daily-refresh"], default="daily-refresh")
    parser.add_argument("--cache-dir", default=str(MarketDataConfig().cache_dir))
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--stock-overlap-days", type=int, default=10)
    parser.add_argument("--news-overlap-days", type=int, default=7)
    parser.add_argument("--options-overlap-days", type=int, default=3)
    parser.add_argument("--prepare-output-path", default=None)
    parser.add_argument("--frequency", default="1d")
    parser.add_argument("--train-seconds", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--llm-dataset-path", default=None)
    parser.add_argument("--llm-output-dir", default=None)
    parser.add_argument("--llm-base-model", default=None)
    parser.add_argument("--llm-max-seq-length", type=int, default=None)
    parser.add_argument("--llm-max-train-samples", type=int, default=None)
    parser.add_argument("--llm-max-eval-samples", type=int, default=None)
    return parser


def build_market_data_config(args: argparse.Namespace) -> MarketDataConfig:
    env_config = MarketDataConfig.from_env()
    return MarketDataConfig(
        provider=args.provider,
        api_key=args.api_key or env_config.api_key,
        symbols=parse_symbols(args.symbols),
        start_date=args.start_date,
        end_date=args.end_date or env_config.end_date,
        include_news=args.include_news or env_config.include_news,
        include_options=args.include_options or env_config.include_options,
        cache_dir=Path(args.cache_dir),
        mode=args.mode,
        request_timeout=args.request_timeout,
        stock_overlap_days=args.stock_overlap_days,
        news_overlap_days=args.news_overlap_days,
        options_overlap_days=args.options_overlap_days,
    )


def _dataset_path(config: MarketDataConfig, prepare_output_path: str | None) -> Path:
    if prepare_output_path:
        return Path(prepare_output_path).expanduser().resolve()
    return (config.cache_dir / "dataset" / "market_dataset.parquet").resolve()


def _run_train(config: MarketDataConfig, dataset_path: Path, args: argparse.Namespace) -> None:
    command = [sys.executable, "train.py"]
    env = os.environ.copy()
    env["MARKET_CACHE_DIR"] = str(config.cache_dir)
    env["MARKET_DATASET_PATH"] = str(dataset_path)

    if args.train_seconds is not None:
        env["TRAIN_SECONDS"] = str(args.train_seconds)
    if args.batch_size is not None:
        env["BATCH_SIZE"] = str(args.batch_size)
    if args.hidden_dim is not None:
        env["HIDDEN_DIM"] = str(args.hidden_dim)
    if args.depth is not None:
        env["DEPTH"] = str(args.depth)
    if args.dropout is not None:
        env["DROPOUT"] = str(args.dropout)
    if args.learning_rate is not None:
        env["LEARNING_RATE"] = str(args.learning_rate)

    LOGGER.info("starting training")
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)


def _run_llm_prepare(config: MarketDataConfig, dataset_path: Path, args: argparse.Namespace) -> Path:
    output_path = Path(args.llm_dataset_path).resolve() if args.llm_dataset_path else (config.cache_dir / "dataset" / "market_llm_dataset.parquet")
    command = [
        sys.executable,
        "prepare_llm.py",
        "--market-dataset-path",
        str(dataset_path),
        "--output-path",
        str(output_path),
    ]
    if args.start_date:
        command.extend(["--start-date", args.start_date])
    if args.end_date:
        command.extend(["--end-date", args.end_date])
    env = os.environ.copy()
    env["MARKET_CACHE_DIR"] = str(config.cache_dir)
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)
    return output_path


def _run_llm_train(config: MarketDataConfig, llm_dataset_path: Path, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "train_llm.py",
        "--dataset-path",
        str(llm_dataset_path),
    ]
    if args.llm_output_dir:
        command.extend(["--output-dir", args.llm_output_dir])
    if args.llm_base_model:
        command.extend(["--base-model", args.llm_base_model])
    if args.llm_max_seq_length is not None:
        command.extend(["--max-seq-length", str(args.llm_max_seq_length)])
    if args.llm_max_train_samples is not None:
        command.extend(["--max-train-samples", str(args.llm_max_train_samples)])
    if args.llm_max_eval_samples is not None:
        command.extend(["--max-eval-samples", str(args.llm_max_eval_samples)])
    env = os.environ.copy()
    env["MARKET_CACHE_DIR"] = str(config.cache_dir)
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)


def _run_llm_eval(config: MarketDataConfig, llm_dataset_path: Path, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "evaluate_llm.py",
        "--dataset-path",
        str(llm_dataset_path),
        "--track",
        "llm_hybrid",
    ]
    if args.llm_output_dir:
        command.extend(["--adapter-path", args.llm_output_dir])
    if args.llm_base_model:
        command.extend(["--base-model", args.llm_base_model])
    env = os.environ.copy()
    env["MARKET_CACHE_DIR"] = str(config.cache_dir)
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    config = build_market_data_config(args)

    LOGGER.info("refreshing market data")
    refresh_market_data(config)

    LOGGER.info("rebuilding dataset")
    _run_prepare(config, args.prepare_output_path, args.frequency)

    market_dataset_path = _dataset_path(config, args.prepare_output_path)
    if args.pipeline == "structured":
        LOGGER.info("running training")
        _run_train(config, market_dataset_path, args)
        return

    LOGGER.info("building llm dataset")
    llm_dataset_path = _run_llm_prepare(config, market_dataset_path, args)

    LOGGER.info("running llm training")
    _run_llm_train(config, llm_dataset_path, args)

    LOGGER.info("running llm evaluation")
    _run_llm_eval(config, llm_dataset_path, args)


if __name__ == "__main__":
    main()
