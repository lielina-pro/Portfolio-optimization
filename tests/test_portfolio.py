import numpy as np
import pandas as pd
import pytest

from src.portfolio import (
    annualized_return_from_forecast,
    build_expected_returns,
    annualized_covariance_matrix,
    run_efficient_frontier,
    sample_efficient_frontier,
    random_portfolios,
)


def test_annualized_return_from_forecast_flat_price_gives_zero_return():
    # A flat forecast (no price change) should annualize to ~0% return
    result = annualized_return_from_forecast(100.0, 100.0, n_trading_days=252)
    assert np.isclose(result, 0.0, atol=1e-9)


def test_annualized_return_from_forecast_doubling_in_one_year():
    result = annualized_return_from_forecast(100.0, 200.0, n_trading_days=252)
    assert np.isclose(result, 1.0, atol=1e-6)  # 100% annual return


def test_annualized_return_from_forecast_half_year_horizon():
    # 10% total return over half a year should annualize to > 10%
    result = annualized_return_from_forecast(100.0, 110.0, n_trading_days=126)
    assert result > 0.10


@pytest.fixture
def sample_daily_returns():
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    rng = np.random.default_rng(0)
    data = {
        "TSLA": rng.normal(0.002, 0.035, len(dates)),
        "BND": rng.normal(0.0001, 0.003, len(dates)),
        "SPY": rng.normal(0.0005, 0.011, len(dates)),
    }
    return pd.DataFrame(data, index=dates)


def test_build_expected_returns_overrides_tsla_only(sample_daily_returns):
    forecast_return = 0.0  # e.g. a flat/no-drift forecast
    expected = build_expected_returns(forecast_return, sample_daily_returns, tsla_col="TSLA")
    assert np.isclose(expected["TSLA"], 0.0)
    # BND and SPY should reflect historical annualized means, not be zero/overridden
    assert expected["BND"] != 0.0
    assert expected["SPY"] != 0.0


def test_annualized_covariance_matrix_scales_by_periods(sample_daily_returns):
    daily_cov = sample_daily_returns.cov()
    annual_cov = annualized_covariance_matrix(sample_daily_returns, periods_per_year=252)
    assert np.allclose(annual_cov.values, daily_cov.values * 252)


def test_run_efficient_frontier_weights_sum_to_one(sample_daily_returns):
    expected_returns = build_expected_returns(0.05, sample_daily_returns)
    cov = annualized_covariance_matrix(sample_daily_returns)
    results = run_efficient_frontier(expected_returns, cov, risk_free_rate=0.04)

    for key in ("max_sharpe", "min_vol"):
        weights = results[key]["weights"]
        assert np.isclose(weights.sum(), 1.0, atol=1e-3)
        assert (weights >= -1e-6).all()  # long-only


def test_min_vol_has_lower_or_equal_volatility_than_max_sharpe(sample_daily_returns):
    expected_returns = build_expected_returns(0.05, sample_daily_returns)
    cov = annualized_covariance_matrix(sample_daily_returns)
    results = run_efficient_frontier(expected_returns, cov)
    assert results["min_vol"]["volatility"] <= results["max_sharpe"]["volatility"] + 1e-6


def test_sample_efficient_frontier_returns_arrays(sample_daily_returns):
    expected_returns = build_expected_returns(0.05, sample_daily_returns)
    cov = annualized_covariance_matrix(sample_daily_returns)
    returns, vols = sample_efficient_frontier(expected_returns, cov, n_points=10)
    assert len(returns) == len(vols)
    assert len(returns) > 0
    assert (vols > 0).all()


def test_random_portfolios_weights_sum_to_one(sample_daily_returns):
    expected_returns = build_expected_returns(0.05, sample_daily_returns)
    cov = annualized_covariance_matrix(sample_daily_returns)
    returns, vols, weights = random_portfolios(expected_returns, cov, n_portfolios=200)
    assert weights.shape == (200, 3)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert len(returns) == 200
    assert (vols > 0).all()
