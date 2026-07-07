"""
portfolio.py
-------------
Utilities for Task 4: Modern Portfolio Theory optimization combining a
forecast-derived expected return for TSLA with historical expected returns
for BND and SPY.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return_from_forecast(
    last_actual_price: float,
    forecast_final_price: float,
    n_trading_days: int,
    periods_per_year: int = 252,
) -> float:
    """
    Convert a point forecast (e.g., the final value of a Task 3 forecast)
    into an annualized expected return, for use as TSLA's expected return
    input into the MPT optimization.

    Parameters
    ----------
    last_actual_price : float
        The last known historical price (forecast start point).
    forecast_final_price : float
        The forecasted price at the end of the forecast horizon.
    n_trading_days : int
        Number of trading days in the forecast horizon (e.g., 252 for ~12mo).
    periods_per_year : int
        Trading days per year, used to annualize.

    Returns
    -------
    float : annualized expected return (e.g., 0.05 for 5%)
    """
    total_return = (forecast_final_price / last_actual_price) - 1
    years = n_trading_days / periods_per_year
    if years <= 0:
        raise ValueError("n_trading_days must be positive")
    # Annualize a total return compounded over `years`
    return (1 + total_return) ** (1 / years) - 1


def build_expected_returns(
    tsla_forecast_return: float,
    historical_returns: pd.DataFrame,
    tsla_col: str = "TSLA",
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Build the expected-returns vector for MPT: TSLA uses the forecast-derived
    annualized return, other assets use historical annualized mean return.

    Parameters
    ----------
    tsla_forecast_return : float
        Annualized expected return for TSLA (from Task 3 forecast).
    historical_returns : pd.DataFrame
        Daily returns for all assets (including TSLA -- TSLA's column is
        overridden with the forecast value, others keep historical means).
    tsla_col : str
        Name of the TSLA column to override.
    periods_per_year : int
        Trading days per year, used to annualize historical means.

    Returns
    -------
    pd.Series indexed by asset, annualized expected returns.
    """
    mean_annualized = (1 + historical_returns.mean()) ** periods_per_year - 1
    expected_returns = mean_annualized.copy()
    expected_returns[tsla_col] = tsla_forecast_return
    return expected_returns


def annualized_covariance_matrix(daily_returns: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """Annualize a daily-returns covariance matrix (cov * periods_per_year)."""
    return daily_returns.cov() * periods_per_year


def run_efficient_frontier(expected_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float = 0.04):
    """
    Run max-Sharpe and min-volatility optimizations using PyPortfolioOpt.

    Returns
    -------
    dict with keys 'max_sharpe' and 'min_vol', each a dict of
    {weights: pd.Series, expected_return: float, volatility: float, sharpe: float}
    """
    from pypfopt.efficient_frontier import EfficientFrontier

    results = {}

    ef_sharpe = EfficientFrontier(expected_returns, cov_matrix)
    ef_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
    weights_sharpe = ef_sharpe.clean_weights()
    ret_s, vol_s, sharpe_s = ef_sharpe.portfolio_performance(risk_free_rate=risk_free_rate)
    results["max_sharpe"] = {
        "weights": pd.Series(weights_sharpe),
        "expected_return": ret_s,
        "volatility": vol_s,
        "sharpe": sharpe_s,
    }

    ef_minvol = EfficientFrontier(expected_returns, cov_matrix)
    ef_minvol.min_volatility()
    weights_minvol = ef_minvol.clean_weights()
    ret_v, vol_v, sharpe_v = ef_minvol.portfolio_performance(risk_free_rate=risk_free_rate)
    results["min_vol"] = {
        "weights": pd.Series(weights_minvol),
        "expected_return": ret_v,
        "volatility": vol_v,
        "sharpe": sharpe_v,
    }

    return results


def sample_efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_points: int = 50,
):
    """
    Trace the efficient frontier by solving for minimum volatility at a
    range of target returns spanning the achievable range.

    Returns
    -------
    (returns, volatilities) : tuple of np.ndarray, each of length <= n_points
    (points where optimization failed/infeasible are skipped)
    """
    from pypfopt.efficient_frontier import EfficientFrontier

    min_ret = expected_returns.min()
    max_ret = expected_returns.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier_returns = []
    frontier_vols = []

    for target in target_returns:
        try:
            ef = EfficientFrontier(expected_returns, cov_matrix)
            ef.efficient_return(target_return=target)
            ret, vol, _ = ef.portfolio_performance()
            frontier_returns.append(ret)
            frontier_vols.append(vol)
        except Exception:
            continue

    return np.array(frontier_returns), np.array(frontier_vols)


def random_portfolios(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_portfolios: int = 5000,
    seed: int = 42,
):
    """
    Generate random long-only portfolios (weights summing to 1) for a
    scatter-cloud visualization behind the efficient frontier line.

    Returns
    -------
    (returns, volatilities, weights_array)
    """
    rng = np.random.default_rng(seed)
    n_assets = len(expected_returns)
    weights = rng.dirichlet(np.ones(n_assets), size=n_portfolios)

    returns = weights @ expected_returns.values
    portfolio_vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov_matrix.values, weights))

    return returns, portfolio_vols, weights
