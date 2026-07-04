import numpy as np
import pandas as pd

from src.risk_metrics import (
    historical_var,
    sharpe_ratio,
    annualized_return,
    annualized_volatility,
    max_drawdown,
    total_return,
)


def test_historical_var_positive_for_normal_losses():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.02, 1000))
    var_95 = historical_var(returns, confidence=0.95)
    assert var_95 > 0


def test_sharpe_ratio_zero_vol_returns_nan():
    returns = pd.Series([0.01] * 10)  # zero std
    assert np.isnan(sharpe_ratio(returns))


def test_sharpe_ratio_positive_for_positive_mean_returns():
    np.random.seed(1)
    returns = pd.Series(np.random.normal(0.001, 0.01, 500))
    assert sharpe_ratio(returns) != 0


def test_annualized_return_matches_known_growth():
    # 252 days of 0% daily return each => 0% annualized
    returns = pd.Series([0.0] * 252)
    assert np.isclose(annualized_return(returns), 0.0, atol=1e-6)


def test_annualized_volatility_scales_with_sqrt_time():
    daily_std = 0.01
    returns = pd.Series(np.random.normal(0, daily_std, 10000))
    vol = annualized_volatility(returns)
    assert np.isclose(vol, daily_std * np.sqrt(252), rtol=0.1)


def test_max_drawdown_is_non_positive():
    cum_returns = pd.Series([1.0, 1.1, 0.9, 1.05, 0.8, 1.2])
    dd = max_drawdown(cum_returns)
    assert dd <= 0


def test_total_return_matches_final_value():
    cum_returns = pd.Series([1.0, 1.1, 1.2, 1.3])
    assert np.isclose(total_return(cum_returns), 0.3)
