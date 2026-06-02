# Task-Oriented Benchmarking of Traditional ML Models in Stock Market Applications

## Project Overview
This project benchmarks traditional Machine Learning models (ARIMA, SVM, Gradient Boosting, Random Forest) within the stock market domain. The primary goal is to bridge the gap between statistical precision (MAE, RMSE, Accuracy) and practical financial performance (Sharpe Ratio, Max Drawdown).

With the latest features, the system implements advanced trading tasks: **Market Regime Detection**, **Dynamic Position Sizing**, **Realistic Entry/Exit Rules**, and **Multi-Asset Portfolio Allocation**.

## Group 9
- Gregorius Willson — 2802449846
- Marco Oden Leo — 2802429453
- Yoel Nathanael — 2802445766

## Core Features
- **Automated Data Ingestion**: Historical OHLCV data fetching via `yfinance` stored in Parquet format. Supports fallback synthetic data.
- **Advanced Preprocessing**: Implementation of Technical Indicators (RSI, MACD, Bollinger Bands, Volatility, SMA) and Triple Barrier Labeling.
- **Market Regime Detection (GMM)**: Unsupervised clustering using Gaussian Mixture Models (GMM) to segment market states into Low Volatility, Moderate Volatility, and High Volatility regimes.
- **Dynamic Position Sizing**: Risk management models including Kelly Criterion (based on prediction confidence) and Volatility-Adjusted sizing.
- **Advanced Entry/Exit Simulator**: Realistic trade state machine supporting Stop Loss (SL), Profit Target (PT), Trailing Stop, Time Barrier, and SMA-50 Trend Filter.
- **Multi-Asset Portfolio Sizing**: Asset allocation models using Mean-Variance Optimization (MVO / Max Sharpe) and Risk Parity.
- **Production-Ready API**: FastAPI backend for real-time inference, GMM regime detection, and Kelly position sizing recommendations.
- **Interactive Dashboard**: Streamlit-based 4-tab UI for exploring benchmarks, regimes, simulator settings, and portfolio allocations in real-time.

## Project Structure
```text
.
├── data/               # Raw and processed data storage (Parquet/CSV)
├── models/             # Serialized model weights, scalers, and GMM regime models
├── src/                # Source code
│   ├── api/            # FastAPI application
│   │   └── main.py
│   ├── features/       # Feature engineering and preprocessing logic
│   │   ├── preprocessing.py
│   │   └── technical_indicators.py
│   ├── models/         # Model wrappers, backtester, and sizing logic
│   │   ├── backtester.py       # Entry/Exit simulator state machine
│   │   ├── market_regime.py    # GMM regime detector
│   │   ├── model_wrappers.py   # Wrapper for SVM, RF, XGB, LGBM, ARIMA
│   │   ├── portfolio_sizing.py # MVO and Risk Parity models
│   │   └── position_sizing.py  # Kelly and Volatility sizing
│   ├── ui/             # Streamlit dashboard
│   │   └── dashboard.py
│   ├── data_ingestion.py # Raw data fetching script
│   └── train_benchmark.py # Main training and evaluation script
├── tests/              # Verification and smoke tests
│   └── verification.py
├── requirements.txt    # Project dependencies
└── venv/               # Virtual environment
```

## Setup & Installation

Use either `uv` for a native Python setup or Docker for a fully containerized setup. Both work on Windows, macOS, and Linux.

### Option A: Native setup with uv

Install `uv` from <https://docs.astral.sh/uv/getting-started/installation/>, then run:

```bash
uv sync
```

Run commands through `uv run` so the correct virtual environment is used automatically:

```bash
uv run python tests/verification.py
uv run python src/train_benchmark.py
uv run uvicorn src.api.main:app --reload
uv run streamlit run src/ui/dashboard.py
```

### Option B: Containerized setup with Docker

Install Docker Desktop or Docker Engine, then run the full smoke test:

```bash
docker compose run --rm verify
```

Run the API:

```bash
docker compose up api
```

Run the dashboard:

```bash
docker compose up dashboard
```

Generated market data and trained models are written to local `data/` and `models/` folders through Docker volumes.

### Option C: Classic pip setup

If you prefer `pip`, create a virtual environment and install the requirements:

```bash
python -m venv venv
source venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## New AI Modules

The following traditional ML modules have been added to expand the system's capabilities:

| Module | File | Purpose |
|---|---|---|
| HMM Regime Detection | `src/models/regime_hmm.py` | Hidden Markov Model 3-state market regime (Bull/Sideways/Bear) |
| GARCH Volatility | `src/models/volatility_garch.py` | GARCH(1,1) forward volatility forecast for position sizing |
| HRP Portfolio | `src/models/portfolio_hrp.py` | Hierarchical Risk Parity allocation |
| Anomaly Detection | `src/models/anomaly_detector.py` | Isolation Forest for unusual market conditions |
| Mean Reversion | `src/models/mean_reversion.py` | Z-score + Ornstein-Uhlenbeck half-life analysis |
| SHAP Explainability | `src/models/explainability.py` | Feature importance via SHAP values |
| CVaR Risk Model | `src/models/risk_model.py` | Tail risk (Conditional Value-at-Risk) |
| Hyperparameter Tuning | `src/training/hyperparameter_tuner.py` | Walk-forward TimeSeriesSplit + GridSearchCV |

## Orchestrator & Auto-Refresh

The `src/orchestrator.py` module manages the system lifecycle:

- **Daily refresh** (9:30 AM ET): Fetches latest OHLCV, runs anomaly detection, updates inference cache
- **Weekly retrain** (Monday 7:00 AM ET): Full model retrain with hyperparameter tuning + all new modules
- **Auto-startup**: If no models found, triggers immediate training on first run

## New API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/status` | GET | Scheduler state, last refresh/retrain per ticker |
| `/regime/{ticker}` | GET | HMM + GMM regimes, transition matrix |
| `/volatility/{ticker}` | GET | GARCH forecast vol vs rolling 20d |
| `/portfolio` | POST | Portfolio weights (EW/RP/MVO/HRP) |
| `/anomaly/{ticker}` | GET | Isolation Forest anomaly flag + score |
| `/mean-reversion/{ticker}` | GET | Z-score, OU half-life, MR signal |
| `/risk/{ticker}` | GET | CVaR 95/99, VaR 95/99, position scale |
| `/explain/{ticker}/{model}` | GET | SHAP feature importances |

## Dashboard Tabs

The Streamlit dashboard now has 8 tabs:
1. 📊 Model Benchmarks
2. 📈 Market Regimes (GMM)
3. ⚙️ Trading Simulator
4. 💼 Portfolio Allocation (EW / RP / MVO / **HRP**)
5. 🔮 Regime (HMM) — transition matrix heatmap
6. 📉 Risk & Volatility — GARCH forecast + CVaR + anomaly
7. 🔄 Mean Reversion — Z-score chart + OU half-life
8. 🔍 Explainability — SHAP feature importance

## Workflow & Usage

### 1. Run Verification Test
Run the smoke test to verify all components and test data ingestion:
```bash
python tests/verification.py
```

### 2. Train Models and Benchmarks
Train all model weights, scale parameters, and GMM regime models for a ticker:
```bash
python src/train_benchmark.py
```

### 3. Launch the FastAPI backend
Start the API server for real-time analysis, GMM regime checks, and position sizing:
```bash
uvicorn src.api.main:app --reload
```
You can query the `/analyze` endpoint with historical OHLCV data.

### 4. Open the Streamlit Dashboard
Launch the interactive dashboard to visualize all outcomes:
```bash
streamlit run src/ui/dashboard.py
```

## Key Evaluation Metrics
- **Statistical**: Accuracy (Hit Rate).
- **Financial**: Sharpe Ratio, Maximum Drawdown (MDD), Cumulative Returns (Alpha) under both baseline and advanced risk-management strategies.

## References
- De Prado, M. L. (2018). *Advances in Financial Machine Learning*. Wiley.
- De Prado, M. L. (2020). *Machine Learning for Asset Managers*. Springer.
- Tiangolo, S. *FastAPI Framework Documentation*.
- Streamlit Inc. *Streamlit Documentation*.

