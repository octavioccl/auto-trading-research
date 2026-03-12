from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd

from .normalize import NEWS_COLUMNS, OPTIONS_COLUMNS, STOCK_COLUMNS, empty_frame


@dataclass(frozen=True)
class RawDataPaths:
    cache_dir: Path
    raw_dir: Path
    stocks_dir: Path
    news_dir: Path
    options_dir: Path
    manifest_path: Path


def build_paths(cache_dir: Path) -> RawDataPaths:
    raw_dir = cache_dir / "raw"
    return RawDataPaths(
        cache_dir=cache_dir,
        raw_dir=raw_dir,
        stocks_dir=raw_dir / "stocks",
        news_dir=raw_dir / "news",
        options_dir=raw_dir / "options",
        manifest_path=raw_dir / "manifest.json",
    )


def ensure_directories(paths: RawDataPaths) -> None:
    for path in [paths.raw_dir, paths.stocks_dir, paths.news_dir, paths.options_dir]:
        path.mkdir(parents=True, exist_ok=True)


def read_existing_frame(path: Path, feed: str) -> pd.DataFrame:
    columns = {
        "stocks": STOCK_COLUMNS,
        "news": NEWS_COLUMNS,
        "options": OPTIONS_COLUMNS,
    }[feed]
    if not path.exists():
        return empty_frame(columns)
    return pd.read_csv(path)


def latest_timestamp(path: Path, timestamp_column: str = "timestamp") -> pd.Timestamp | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, usecols=[timestamp_column])
    if frame.empty:
        return None
    timestamps = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce").dropna()
    if timestamps.empty:
        return None
    return timestamps.max()


def write_frame_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        frame.to_csv(tmp_path, index=False)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
        tmp_path = Path(handle.name)
        json.dump(manifest, handle, indent=2, sort_keys=True)
    try:
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
