import numpy as np
import pandas as pd
import pytest

from src.data_loader import clean_data, compute_daily_returns, compute_log_returns


@pytest.fixture
def sample_price_df():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    data = {
        "TSLA": [100, 102, np.nan, 105, 107, 106, np.nan, 110, 111, 113],
        "SPY": [400, 401, 402, np.nan, 404, 405, 406, 407, np.nan, 409],
    }
    return pd.DataFrame(data, index=dates)


def test_clean_data_fills_missing(sample_price_df):
    cleaned = clean_data(sample_price_df, method="ffill")
    assert cleaned.isna().sum().sum() == 0


def test_clean_data_reindexes_business_days(sample_price_df):
    cleaned = clean_data(sample_price_df, method="ffill")
    # business-day reindex should not include weekends within the original range
    assert all(cleaned.index.dayofweek < 5)


def test_compute_daily_returns_shape(sample_price_df):
    cleaned = clean_data(sample_price_df, method="ffill")
    returns = compute_daily_returns(cleaned)
    assert len(returns) == len(cleaned) - 1
    assert not returns.isna().any().any()


def test_compute_log_returns_close_to_simple_for_small_moves(sample_price_df):
    cleaned = clean_data(sample_price_df, method="ffill")
    simple = compute_daily_returns(cleaned)
    log_r = compute_log_returns(cleaned)
    # For small daily moves, log returns approximate simple returns
    assert np.allclose(simple.values, log_r.values, atol=0.01)
