from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class MarketDataProvider(ABC):
    @abstractmethod
    def fetch_stock_bars(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def fetch_news(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def fetch_options(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError
