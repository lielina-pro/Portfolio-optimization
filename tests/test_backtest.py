"""
Unit tests for src/backtest.py (Task 5 - Strategy Backtesting).
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")  # headless backend for CI

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backtest import (
    compute_daily_returns,
    simulate_portfolio,
    simulate_benchmark,
    max_drawdown,
    performance_metrics,
    metrics_table,
    plot_backtest,
    run_backtest,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def synthetic_prices():
    """
    ~1 year of synthetic daily prices for TSLA, BND, SPY with realistic
    relative volatility (TSLA >> SPY > BND), reproducible via a fixed seed.
    """
    rng = np.random.default_rng(42)
    n_days = 252
    dates = pd.bdate_range("2025-01-02", periods=n_days)

    tsla_returns = rng.normal(0.0000, 0.035, n_days)   # high vol, ~flat drift
    spy_returns = rng.normal(0.0004, 0.010, n_days)    # moderate vol/drift
    bnd_returns = rng.normal(0.0001, 0.003, n_days)    # low vol/drift

    tsla_prices = 400 * np.cumprod(1 + tsla_returns)
    spy_prices = 550 * np.cumprod(1 + spy_returns)
    bnd_prices = 72 * np.cumprod(1 + bnd_returns)

    return pd.DataFrame(
        {"TSLA": tsla_prices, "BND": bnd_prices, "SPY": spy_prices}, index=dates
    )


@pytest.fixture
def synthetic_returns(synthetic_prices):
    return compute_daily_returns(synthetic_prices)


# --------------------------------------------------------------------------- #
# compute_daily_returns
# --------------------------------------------------------------------------- #
def test_compute_daily_returns_shape(synthetic_prices):
    returns = compute_daily_returns(synthetic_prices)
    assert len(returns) == len(synthetic_prices) - 1
    assert list(returns.columns) == list(synthetic_prices.columns)


def test_compute_daily_returns_empty_raises():
    with pytest.raises(ValueError):
        compute_daily_returns(pd.DataFrame())


# --------------------------------------------------------------------------- #
# simulate_portfolio
# --------------------------------------------------------------------------- #
def test_simulate_single_asset_matches_manual_compounding(synthetic_returns):
    cum = simulate_portfolio(synthetic_returns, {"SPY": 1.0}, rebalance_freq=None)
    manual = (1 + synthetic_returns["SPY"]).cumprod()
    assert np.allclose(cum.values, manual.values, rtol=1e-10)


def test_simulate_portfolio_weights_must_sum_to_one(synthetic_returns):
    with pytest.raises(ValueError):
        simulate_portfolio(synthetic_returns, {"SPY": 0.5, "BND": 0.3})


def test_simulate_portfolio_missing_column_raises(synthetic_returns):
    with pytest.raises(ValueError):
        simulate_portfolio(synthetic_returns, {"AAPL": 1.0})


def test_rebalanced_and_buy_and_hold_diverge(synthetic_returns):
    weights = {"TSLA": 0.2, "BND": 0.4, "SPY": 0.4}
    buy_hold = simulate_portfolio(synthetic_returns, weights, rebalance_freq=None)
    rebalanced = simulate_portfolio(synthetic_returns, weights, rebalance_freq="M")
    # With differing asset volatilities, drift should make the two paths differ
    assert not np.allclose(buy_hold.values, rebalanced.values)
    # Both should still start compounding from the same initial capital
    assert buy_hold.index.equals(rebalanced.index)


# --------------------------------------------------------------------------- #
# simulate_benchmark
# --------------------------------------------------------------------------- #
def test_simulate_benchmark_default_weights(synthetic_returns):
    bench = simulate_benchmark(synthetic_returns)
    manual = simulate_portfolio(
        synthetic_returns, {"SPY": 0.6, "BND": 0.4}, rebalance_freq="M"
    )
    assert np.allclose(bench.values, manual.values)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_max_drawdown_is_non_positive(synthetic_returns):
    cum = simulate_portfolio(synthetic_returns, {"SPY": 1.0}, rebalance_freq=None)
    mdd = max_drawdown(cum)
    assert mdd <= 0


def test_performance_metrics_keys_and_types(synthetic_returns):
    cum = simulate_benchmark(synthetic_returns)
    m = performance_metrics(cum, risk_free_rate=0.04)
    expected_keys = {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
    }
    assert set(m.keys()) == expected_keys
    for v in m.values():
        assert isinstance(v, float)


def test_performance_metrics_requires_two_points():
    single = pd.Series([1.0], index=pd.bdate_range("2025-01-02", periods=1))
    with pytest.raises(ValueError):
        performance_metrics(single)


def test_metrics_table_shape(synthetic_returns):
    strat_cum = simulate_portfolio(synthetic_returns, {"SPY": 1.0}, rebalance_freq=None)
    bench_cum = simulate_benchmark(synthetic_returns)
    table = metrics_table(performance_metrics(strat_cum), performance_metrics(bench_cum))
    assert table.shape == (5, 2)
    assert list(table.columns) == ["Strategy", "Benchmark (60/40)"]


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def test_plot_backtest_returns_figure(synthetic_returns):
    strat_cum = simulate_portfolio(synthetic_returns, {"SPY": 1.0}, rebalance_freq=None)
    bench_cum = simulate_benchmark(synthetic_returns)
    fig = plot_backtest(strat_cum, bench_cum)
    assert fig is not None
    assert len(fig.axes) == 1


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #
def test_run_backtest_end_to_end(synthetic_prices):
    result = run_backtest(
        synthetic_prices,
        strategy_weights={"SPY": 1.0},
        rebalance_freq="M",
    )
    assert set(result.keys()) == {
        "strategy_cum",
        "benchmark_cum",
        "strategy_metrics",
        "benchmark_metrics",
        "table",
        "figure",
    }
    assert len(result["strategy_cum"]) == len(result["benchmark_cum"])
    assert result["table"].shape == (5, 2)
