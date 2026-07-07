"""
Task 5 - Strategy Backtesting

Simulates a static (Task 4) optimal-weight portfolio against a passive
60/40 SPY/BND benchmark over a held-out backtesting window, with optional
monthly rebalancing, and computes standard performance metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a DataFrame of adjusted-close prices (one column per ticker,
    DatetimeIndex) into simple daily percentage returns.
    """
    if prices.empty:
        raise ValueError("prices DataFrame is empty")
    returns = prices.sort_index().pct_change().dropna(how="all")
    return returns


# --------------------------------------------------------------------------- #
# Portfolio simulation
# --------------------------------------------------------------------------- #
def simulate_portfolio(
    returns: pd.DataFrame,
    weights: dict,
    rebalance_freq: str | None = None,
    initial_value: float = 1.0,
) -> pd.Series:
    """
    Simulate the cumulative value of a portfolio given target weights.

    Parameters
    ----------
    returns : DataFrame of daily simple returns, must contain a column for
        every ticker in `weights`.
    weights : dict mapping ticker -> target weight (weights need not
        include every column of `returns`; any weight of 0 / omitted ticker
        is simply not held).
    rebalance_freq : None for buy-and-hold (weights drift with the market),
        or a pandas period alias (e.g. "M" for monthly) to rebalance back
        to the target weights at the start of each new period.
    initial_value : starting portfolio value (default 1.0, so the returned
        series can be read directly as a cumulative growth factor).

    Returns
    -------
    pd.Series indexed like `returns`, giving portfolio value at the end of
    each trading day.
    """
    assets = list(weights.keys())
    missing = [a for a in assets if a not in returns.columns]
    if missing:
        raise ValueError(f"returns is missing columns for: {missing}")

    w = np.array([weights[a] for a in assets], dtype=float)
    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError(f"weights must sum to 1.0, got {w.sum():.4f}")

    r = returns[assets].fillna(0.0).values
    dates = returns.index

    asset_values = w * initial_value
    values = np.empty(len(dates))

    if rebalance_freq is None:
        for i in range(len(dates)):
            asset_values = asset_values * (1.0 + r[i])
            values[i] = asset_values.sum()
    else:
        periods = dates.to_period(rebalance_freq)
        current_period = periods[0]
        for i in range(len(dates)):
            if periods[i] != current_period:
                total = asset_values.sum()
                asset_values = w * total
                current_period = periods[i]
            asset_values = asset_values * (1.0 + r[i])
            values[i] = asset_values.sum()

    return pd.Series(values, index=dates, name="portfolio_value")


def simulate_benchmark(
    returns: pd.DataFrame,
    weights: dict | None = None,
    rebalance_freq: str | None = "M",
    initial_value: float = 1.0,
) -> pd.Series:
    """
    Simulate the passive benchmark portfolio (default: static 60% SPY /
    40% BND, rebalanced monthly).
    """
    if weights is None:
        weights = {"SPY": 0.6, "BND": 0.4}
    return simulate_portfolio(
        returns, weights, rebalance_freq=rebalance_freq, initial_value=initial_value
    ).rename("benchmark_value")


# --------------------------------------------------------------------------- #
# Performance metrics
# --------------------------------------------------------------------------- #
def max_drawdown(cum_series: pd.Series) -> float:
    """Maximum peak-to-trough drawdown, expressed as a negative fraction."""
    running_max = cum_series.cummax()
    drawdown = cum_series / running_max - 1.0
    return float(drawdown.min())


def performance_metrics(
    cum_series: pd.Series,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
) -> dict:
    """
    Compute total return, annualized return, Sharpe ratio, and max drawdown
    for a cumulative portfolio-value series that starts at 1.0 (or any
    positive initial value).
    """
    if len(cum_series) < 2:
        raise ValueError("need at least 2 data points to compute metrics")

    daily_returns = cum_series.pct_change().dropna()
    n_days = len(cum_series)

    # cum_series is assumed to start from an implicit baseline of 1.0, as
    # produced by simulate_portfolio / simulate_benchmark with the default
    # initial_value=1.0.
    total_return = float(cum_series.iloc[-1] / 1.0 - 1.0)

    annualized_return = float((1.0 + total_return) ** (periods_per_year / n_days) - 1.0)
    annualized_vol = float(daily_returns.std() * np.sqrt(periods_per_year))
    sharpe_ratio = (
        float((annualized_return - risk_free_rate) / annualized_vol)
        if annualized_vol > 0
        else float("nan")
    )
    mdd = max_drawdown(cum_series)

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": mdd,
    }


def metrics_table(strategy_metrics: dict, benchmark_metrics: dict) -> pd.DataFrame:
    """Combine strategy and benchmark metrics into a single comparison table."""
    df = pd.DataFrame(
        {"Strategy": strategy_metrics, "Benchmark (60/40)": benchmark_metrics}
    )
    df.index = [
        "Total Return",
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
    ]
    return df


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_backtest(
    strategy_cum: pd.Series,
    benchmark_cum: pd.Series,
    title: str = "Strategy vs. Benchmark: Cumulative Returns",
):
    """
    Plot cumulative growth of $1 for the strategy and benchmark portfolios
    over the backtesting window. Returns the matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(strategy_cum.index, strategy_cum.values, label="Strategy (Task 4 weights)", linewidth=2)
    ax.plot(benchmark_cum.index, benchmark_cum.values, label="Benchmark (60% SPY / 40% BND)",
             linewidth=2, linestyle="--")
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# End-to-end convenience wrapper
# --------------------------------------------------------------------------- #
def run_backtest(
    prices: pd.DataFrame,
    strategy_weights: dict,
    benchmark_weights: dict | None = None,
    rebalance_freq: str | None = "M",
    risk_free_rate: float = 0.04,
):
    """
    Run the full Task 5 pipeline: returns -> simulate strategy & benchmark
    -> metrics -> comparison table -> plot.

    Returns a dict with keys: 'strategy_cum', 'benchmark_cum',
    'strategy_metrics', 'benchmark_metrics', 'table', 'figure'.
    """
    returns = compute_daily_returns(prices)

    strategy_cum = simulate_portfolio(returns, strategy_weights, rebalance_freq=rebalance_freq)
    benchmark_cum = simulate_benchmark(returns, benchmark_weights, rebalance_freq=rebalance_freq)

    strategy_metrics = performance_metrics(strategy_cum, risk_free_rate=risk_free_rate)
    benchmark_metrics = performance_metrics(benchmark_cum, risk_free_rate=risk_free_rate)

    table = metrics_table(strategy_metrics, benchmark_metrics)
    fig = plot_backtest(strategy_cum, benchmark_cum)

    return {
        "strategy_cum": strategy_cum,
        "benchmark_cum": benchmark_cum,
        "strategy_metrics": strategy_metrics,
        "benchmark_metrics": benchmark_metrics,
        "table": table,
        "figure": fig,
    }
