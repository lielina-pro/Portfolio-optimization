"""
eda.py
------
Exploratory data analysis helpers: rolling stats, outlier detection,
and stationarity testing (Augmented Dickey-Fuller).
"""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.stattools import adfuller


def rolling_stats(series: pd.Series, window: int = 30) -> pd.DataFrame:
    """Rolling mean and standard deviation for a price or return series."""
    return pd.DataFrame({
        "rolling_mean": series.rolling(window).mean(),
        "rolling_std": series.rolling(window).std(),
    })


def detect_outliers(returns: pd.Series, n_std: float = 3.0) -> pd.Series:
    """
    Flag daily returns more than `n_std` standard deviations from the mean.
    Returns the subset of `returns` considered outliers, sorted by magnitude.
    """
    mean, std = returns.mean(), returns.std()
    mask = (returns - mean).abs() > n_std * std
    return returns[mask].sort_values(key=lambda x: x.abs(), ascending=False)


def adf_test(series: pd.Series, name: str = "") -> dict:
    """
    Run the Augmented Dickey-Fuller test for stationarity.

    Returns a dict with the test statistic, p-value, critical values,
    and a plain-language verdict at the 5% significance level.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    stat, p_value, used_lag, n_obs, crit_values, _ = result

    verdict = (
        "Stationary (reject H0 of unit root)"
        if p_value < 0.05
        else "Non-stationary (fail to reject H0 of unit root)"
    )

    return {
        "series": name,
        "adf_statistic": stat,
        "p_value": p_value,
        "used_lag": used_lag,
        "n_obs": n_obs,
        "critical_values": crit_values,
        "verdict": verdict,
    }


def print_adf_result(result: dict) -> None:
    """Pretty-print the output of adf_test()."""
    print(f"ADF Test: {result['series']}")
    print(f"  Statistic : {result['adf_statistic']:.4f}")
    print(f"  p-value   : {result['p_value']:.4f}")
    for key, value in result["critical_values"].items():
        print(f"  Critical Value ({key}): {value:.4f}")
    print(f"  Verdict   : {result['verdict']}")
