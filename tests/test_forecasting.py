import numpy as np
import pandas as pd
import pytest

from src.forecasting import (
    chronological_split,
    mae,
    rmse,
    mape,
    evaluate_forecast,
    create_sequences,
)


@pytest.fixture
def sample_series():
    dates = pd.date_range("2020-01-01", "2026-06-30", freq="D")
    values = np.linspace(100, 300, len(dates))
    return pd.Series(values, index=dates)


def test_chronological_split_respects_order(sample_series):
    train, test = chronological_split(sample_series, split_date="2025-01-01")
    assert train.index.max() < pd.Timestamp("2025-01-01")
    assert test.index.min() >= pd.Timestamp("2025-01-01")
    assert len(train) + len(test) == len(sample_series)


def test_chronological_split_raises_on_empty_split(sample_series):
    with pytest.raises(ValueError):
        chronological_split(sample_series, split_date="2099-01-01")


def test_mae_zero_for_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0


def test_rmse_penalizes_large_errors_more_than_mae():
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 1.0, 1.0, 10.0])  # one big outlier error
    assert rmse(y_true, y_pred) > mae(y_true, y_pred)


def test_mape_percentage_scale():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    # errors are 10% and 10%
    assert np.isclose(mape(y_true, y_pred), 10.0)


def test_evaluate_forecast_returns_all_keys():
    y_true = np.array([100.0, 105.0, 110.0])
    y_pred = np.array([101.0, 104.0, 111.0])
    result = evaluate_forecast(y_true, y_pred, model_name="TestModel")
    assert set(result.keys()) == {"model", "MAE", "RMSE", "MAPE (%)"}
    assert result["model"] == "TestModel"


def test_create_sequences_shapes():
    data = np.arange(100, dtype=float)
    window = 10
    X, y = create_sequences(data, window=window)
    assert X.shape == (90, window, 1)
    assert y.shape == (90,)
    # first sequence should be the first `window` values, predicting index `window`
    assert np.allclose(X[0].flatten(), data[:window])
    assert y[0] == data[window]


def test_future_business_dates_starts_after_last_date():
    from src.forecasting import future_business_dates
    dates = future_business_dates("2026-06-29", n_periods=10)
    assert dates.min() > pd.Timestamp("2026-06-29")
    assert len(dates) == 10
    assert all(d.dayofweek < 5 for d in dates)
