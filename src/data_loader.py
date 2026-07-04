"""
data_loader.py
----------------
Utilities for fetching and cleaning historical price data from YFinance
for the GMF Investments portfolio optimization project.
"""

from __future__ import annotations

import os
from typing import Iterable

import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ["TSLA", "BND", "SPY"]
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2026-06-30"


def fetch_asset_data(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: str | None = "data/raw",
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for each ticker from YFinance.

    Parameters
    ----------
    tickers : iterable of str
        Ticker symbols to fetch (default: TSLA, BND, SPY).
    start, end : str
        Date range in 'YYYY-MM-DD' format.
    cache_dir : str or None
        If provided, raw pulls are cached to CSV here and reloaded on
        subsequent calls instead of re-hitting the API.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker -> DataFrame indexed by Date with columns
        Open, High, Low, Close, Adj Close, Volume.
    """
    data = {}
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    for ticker in tickers:
        cache_path = os.path.join(cache_dir, f"{ticker}.csv") if cache_dir else None

        if cache_path and os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
            if df.empty:
                raise ValueError(
                    f"No data returned for {ticker}. Check ticker symbol, "
                    f"date range, or network/API access."
                )
            # yfinance can return MultiIndex columns for a single ticker
            # in some versions; flatten if necessary.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if cache_path:
                df.to_csv(cache_path)

        data[ticker] = df

    return data


def combine_close_prices(data: dict[str, pd.DataFrame], price_col: str = "Adj Close") -> pd.DataFrame:
    """Combine the chosen price column from each ticker's DataFrame into one wide DataFrame."""
    combined = pd.DataFrame({ticker: df[price_col] for ticker, df in data.items()})
    combined.index.name = "Date"
    return combined


def clean_data(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """
    Clean a price DataFrame: ensure business-day frequency, handle missing
    values, and enforce numeric dtypes.

    Parameters
    ----------
    df : pd.DataFrame
        Wide DataFrame of prices indexed by Date.
    method : str
        Missing-value handling strategy: 'ffill', 'interpolate', or 'drop'.
    """
    df = df.copy()
    df = df.astype(float)

    # Reindex to business-day frequency so gaps (holidays, missing feeds)
    # are made explicit as NaN rather than silently absent.
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="B")
    df = df.reindex(full_range)
    df.index.name = "Date"

    if method == "ffill":
        df = df.ffill().bfill()
    elif method == "interpolate":
        df = df.interpolate(method="time").ffill().bfill()
    elif method == "drop":
        df = df.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")

    return df


def compute_daily_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily percentage returns from a price DataFrame."""
    return price_df.pct_change().dropna()


def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from a price DataFrame."""
    import numpy as np
    return np.log(price_df / price_df.shift(1)).dropna()
