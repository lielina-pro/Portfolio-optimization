import numpy as np
import pandas as pd

from src.eda import rolling_stats, detect_outliers, adf_test


def test_rolling_stats_columns():
    series = pd.Series(np.random.normal(0, 1, 100))
    stats = rolling_stats(series, window=10)
    assert set(stats.columns) == {"rolling_mean", "rolling_std"}
    assert stats["rolling_mean"].isna().sum() == 9  # first window-1 rows are NaN


def test_detect_outliers_flags_extremes():
    np.random.seed(0)
    returns = pd.Series(np.random.normal(0, 0.01, 500))
    returns.iloc[250] = 0.5  # inject an extreme outlier
    outliers = detect_outliers(returns, n_std=3.0)
    assert 250 in outliers.index or returns.index[250] in outliers.index


def test_adf_test_detects_stationary_white_noise():
    np.random.seed(0)
    white_noise = pd.Series(np.random.normal(0, 1, 500))
    result = adf_test(white_noise, name="white_noise")
    assert result["p_value"] < 0.05
    assert "Stationary" in result["verdict"]


def test_adf_test_detects_nonstationary_random_walk():
    np.random.seed(0)
    random_walk = pd.Series(np.cumsum(np.random.normal(0, 1, 500)))
    result = adf_test(random_walk, name="random_walk")
    assert "Non-stationary" in result["verdict"] or result["p_value"] >= 0.05
