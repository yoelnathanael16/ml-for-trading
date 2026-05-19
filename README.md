# Task-Oriented Benchmarking of Traditional ML Models in Stock Market Applications

## Project Overview
This project benchmarks traditional Machine Learning models (ARIMA, SVM, Gradient Boosting, Random Forest) within the stock market domain. The primary goal is to bridge the gap between statistical precision (MAE, RMSE) and practical financial performance (Sharpe Ratio, Max Drawdown).

The system identifies "best-fit" models for specific trading tasks such as portfolio sizing, entry/exit signals, and market regime detection.

## Group 9
- Gregorius Willson — 2802449846
- Marco Oden Leo — 2802429453
- Yoel Nathanael — 2802445766

## Features
- **Automated Data Ingestion**: Historical OHLCV data fetching via `yfinance` stored in Parquet format.
- **Advanced Preprocessing**: Implementation of Technical Indicators (RSI, MACD, Bollinger Bands) and Triple Barrier Labeling.
- **Comprehensive Benchmarking**: Unified evaluation of 4+ model types using both statistical accuracy and financial metrics.
- **Production-Ready API**: FastAPI backend for real-time inference and regime detection.
- **Interactive Dashboard**: Streamlit-based UI for visualizing backtesting results and performance matrices.

## Project Structure
```text
.
├── data/               # Raw and processed data storage (Parquet/CSV)
├── models/             # Serialized model weights and scalers
├── src/                # Source code
│   ├── api/            # FastAPI application
│   ├── features/       # Feature engineering and preprocessing logic
│   ├── models/         # Model wrappers and financial metrics
│   ├── ui/             # Streamlit dashboard
│   ├── data_ingestion.py # Raw data fetching script
│   └── train_benchmark.py # Main training and evaluation script
├── tests/              # Verification and smoke tests
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
source venv/bin/activate  # Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Workflow & Usage

### 1. Data Ingestion & Verification
Run the smoke test to verify your environment and fetch initial data:
```bash
uv run python tests/verification.py
```

### 2. Training & Benchmarking
Train the models and generate the task-performance matrix for a specific ticker:
```bash
uv run python src/train_benchmark.py
```
*Note: You can modify the ticker and date range inside the `if __name__ == "__main__":` block of the script.*

### 3. Running the API
Start the FastAPI server for real-time analysis:
```bash
uv run uvicorn src.api.main:app --reload
```

### 4. Running the Dashboard
Launch the Streamlit interface to visualize results:
```bash
uv run streamlit run src/ui/dashboard.py
```

## Key Evaluation Metrics
- **Statistical**: MAE, RMSE, Accuracy (Hit Rate).
- **Financial**: Sharpe Ratio, Maximum Drawdown (MDD), Cumulative Returns (Alpha).

## References
- De Prado, M. L. (2018). *Advances in Financial Machine Learning*.
- De Prado, M. L. (2020). *Machine Learning for Asset Managers*.
- Tiangolo, S. *FastAPI Framework Documentation*.
- Streamlit Inc. *Streamlit Documentation*.
