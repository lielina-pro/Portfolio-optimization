"""
risk_metrics.py
----------------
Foundational risk and performance metrics used across Tasks 1, 4, and 5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical (non-parametric) Value at Risk.

    Returns the loss threshold such that `confidence`% of historical
    returns are better than this value. Reported as a positive number
    representing the magnitude of loss at the given confidence level.
    """
    return -np.percentile(returns.dropna(), (1 - confidence) * 100)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Annualized Sharpe Ratio from a series of periodic (e.g. daily) returns.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns (not annualized).
    risk_free_rate : float
        Annualized risk-free rate (e.g. 0.04 for 4%).
    periods_per_year : int
        Trading periods per year (252 for daily equity data).
    """
    excess = returns - (risk_free_rate / periods_per_year)
    std = excess.std()
    if std == 0 or np.isclose(std, 0.0, atol=1e-12):
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(periods_per_year)


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized return from periodic returns, compounded."""
    growth = (1 + returns).prod()
    n_periods = returns.count()
    if n_periods == 0:
        return np.nan
    return growth ** (periods_per_year / n_periods) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized volatility (standard deviation) from periodic returns."""
    return returns.std() * np.sqrt(periods_per_year)


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Maximum drawdown from a series of cumulative returns (e.g. (1+r).cumprod()).
    Returns a negative number representing the largest peak-to-trough decline.
    """
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()


def total_return(cumulative_returns: pd.Series) -> float:
    """Total return over the full period from a cumulative returns series."""
    return cumulative_returns.iloc[-1] - 1
