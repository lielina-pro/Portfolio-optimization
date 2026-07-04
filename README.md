# Time Series Forecasting for Portfolio Management Optimization

**10 Academy — KAIM Week 9 Challenge**
Client: GMF Investments (Guide Me in Finance)

## Business Context

GMF Investments is a financial advisory firm building data-driven, personalized
portfolio strategies. This project applies time series forecasting to historical
data for three assets — **TSLA** (high-growth stock), **BND** (bond ETF, stability),
and **SPY** (S&P 500 ETF, diversified market exposure) — to inform portfolio
construction under Modern Portfolio Theory (MPT).

Per the Efficient Market Hypothesis, exact price prediction from historical prices
alone is inherently difficult; the forecasts here are treated as one input among
several into a broader allocation decision, not a standalone trading signal.

## Project Tasks

| Task | Description |
|---|---|
| 1 | Data extraction (YFinance), cleaning, EDA, stationarity testing, risk metrics |
| 2 | ARIMA/SARIMA and LSTM forecasting models, trained/evaluated on TSLA |
| 3 | 6–12 month future forecast with confidence intervals, trend & risk analysis |
| 4 | Portfolio optimization via Modern Portfolio Theory (Efficient Frontier) |
| 5 | Strategy backtest vs. a 60% SPY / 40% BND benchmark |

## Project Structure

```
portfolio-optimization/
├── .vscode/settings.json
├── .github/workflows/unittests.yml    # CI: runs pytest on push/PR
├── data/
│   ├── raw/                           # cached YFinance pulls (gitignored)
│   └── processed/                     # cleaned prices, returns (gitignored)
├── notebooks/
│   ├── task1_eda.ipynb
│   ├── task2_forecasting_models.ipynb # (to be added)
│   ├── task3_future_forecast.ipynb    # (to be added)
│   ├── task4_portfolio_optimization.ipynb  # (to be added)
│   └── task5_backtesting.ipynb        # (to be added)
├── src/
│   ├── data_loader.py    # fetch/clean/return-calculation utilities
│   ├── eda.py             # rolling stats, outlier detection, ADF test
│   └── risk_metrics.py    # VaR, Sharpe ratio, drawdown, annualized metrics
├── tests/                 # pytest unit tests for src/
├── scripts/
├── reports/                # final investment memo (PDF / blog post)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/task1_eda.ipynb
```

## Running Tests

```bash
pytest tests/ -v
```

## Data

- **Source:** [YFinance](https://pypi.org/project/yfinance/)
- **Assets:** TSLA, BND, SPY
- **Period:** 2015-01-01 to 2026-06-30
- **Fields:** Open, High, Low, Close, Adj Close, Volume

Raw pulls are cached to `data/raw/*.csv` on first fetch to avoid repeated API calls
and to keep notebook runs reproducible without live network access.

## Team

Kerod · Mahbubah · Feven

## Key Dates

- Interim Submission — Sun, 05 Jul 2026, 8:00 PM UTC
- Final Submission — Tue, 07 Jul 2026, 8:00 PM UTC
