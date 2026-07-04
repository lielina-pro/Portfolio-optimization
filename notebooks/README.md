# Notebooks

| Notebook | Task | Description |
|---|---|---|
| `task1_eda.ipynb` | Task 1 | Data extraction (YFinance), cleaning, EDA, stationarity testing, risk metrics |
| `task2_forecasting_models.ipynb` | Task 2 | ARIMA/SARIMA and LSTM model training & evaluation |
| `task3_future_forecast.ipynb` | Task 3 | Future forecasts with confidence intervals, trend analysis |
| `task4_portfolio_optimization.ipynb` | Task 4 | Efficient Frontier, Max Sharpe / Min Volatility portfolios |
| `task5_backtesting.ipynb` | Task 5 | Strategy backtest vs. 60/40 SPY/BND benchmark |

Notebooks call into reusable functions defined in `src/` so logic can be
unit-tested independently of notebook execution.
