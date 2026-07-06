"""
forecasting.py
---------------
Utilities for Task 2: ARIMA/SARIMA and LSTM forecasting models for TSLA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def chronological_split(series: pd.Series, split_date: str = "2025-01-01") -> tuple[pd.Series, pd.Series]:
    """
    Split a time series chronologically at `split_date`.

    Parameters
    ----------
    series : pd.Series
        Time series indexed by date.
    split_date : str
        Everything before this date is train; this date onward is test.

    Returns
    -------
    (train, test) : tuple of pd.Series
    """
    train = series[series.index < split_date]
    test = series[series.index >= split_date]
    if len(train) == 0 or len(test) == 0:
        raise ValueError(
            f"Split at {split_date} produced an empty train or test set. "
            f"Series spans {series.index.min()} to {series.index.max()}."
        )
    return train, test


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> dict:
    """Return a dict of MAE, RMSE, MAPE for a forecast vs. actuals."""
    return {
        "model": model_name,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE (%)": mape(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# ARIMA / SARIMA
# ---------------------------------------------------------------------------

def fit_auto_arima(train: pd.Series, seasonal: bool = False, m: int = 1, **kwargs):
    """
    Fit an auto_arima model (pmdarima) on the training series.

    Parameters
    ----------
    train : pd.Series
        Training price (or return) series.
    seasonal : bool
        Whether to search SARIMA (seasonal) models.
    m : int
        Seasonal period (e.g. 5 for a business-day "week", 252 for annual
        cycles on daily data). Only used if seasonal=True.
    kwargs :
        Passed through to pmdarima.auto_arima (e.g. max_p, max_q, trace).
    """
    import pmdarima as pm

    model = pm.auto_arima(
        train,
        seasonal=seasonal,
        m=m if seasonal else 1,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        **kwargs,
    )
    return model


def future_business_dates(last_date, n_periods: int) -> pd.DatetimeIndex:
    """
    Generate the next `n_periods` business-day dates strictly after `last_date`.

    Parameters
    ----------
    last_date : str or pd.Timestamp
        The last known historical date.
    n_periods : int
        Number of future business days to generate.
    """
    last_date = pd.Timestamp(last_date)
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_periods)
    return future_dates


def forecast_arima(model, n_periods: int, alpha: float = 0.05):
    """
    Generate a point forecast and confidence interval from a fitted
    pmdarima model.

    Returns
    -------
    forecast : np.ndarray
    conf_int : np.ndarray of shape (n_periods, 2)  [lower, upper]
    """
    forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True, alpha=alpha)
    return np.asarray(forecast), np.asarray(conf_int)


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

def create_sequences(data: np.ndarray, window: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1D array of (scaled) values into overlapping (X, y) sequences
    for supervised LSTM training: use `window` past days to predict the
    next day.

    Parameters
    ----------
    data : np.ndarray
        1D array of scaled values.
    window : int
        Number of past time steps used as input for each prediction.

    Returns
    -------
    X : np.ndarray of shape (n_samples, window, 1)
    y : np.ndarray of shape (n_samples,)
    """
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(data[i])
    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y)
    return X, y


def build_lstm_model(window: int, units: tuple[int, ...] = (50, 50), dropout: float = 0.2):
    """
    Build a stacked LSTM model:
      Input(window, 1) -> LSTM -> Dropout -> LSTM -> Dropout -> Dense(1)

    Parameters
    ----------
    window : int
        Length of the input sequence.
    units : tuple of int
        Number of units in each LSTM layer (length = number of LSTM layers).
    dropout : float
        Dropout rate applied after each LSTM layer.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

    model = Sequential()
    model.add(Input(shape=(window, 1)))
    for i, n_units in enumerate(units):
        return_sequences = i < len(units) - 1
        model.add(LSTM(n_units, return_sequences=return_sequences))
        model.add(Dropout(dropout))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def iterative_lstm_forecast(model, last_window: np.ndarray, n_periods: int) -> np.ndarray:
    """
    Generate a multi-step forecast by iteratively predicting one step
    ahead and feeding the prediction back into the input window.

    Parameters
    ----------
    model : trained keras Model
    last_window : np.ndarray of shape (window,)
        The most recent `window` scaled values from the training/test
        boundary, used to seed the forecast.
    n_periods : int
        Number of future steps to forecast.

    Returns
    -------
    np.ndarray of shape (n_periods,) — scaled predictions (inverse-transform
    with the same scaler used for training before comparing to real prices).
    """
    window = len(last_window)
    current_seq = last_window.copy().reshape(1, window, 1)
    preds = []
    for _ in range(n_periods):
        next_val = model.predict(current_seq, verbose=0)[0, 0]
        preds.append(next_val)
        current_seq = np.append(current_seq[:, 1:, :], [[[next_val]]], axis=1)
    return np.array(preds)
