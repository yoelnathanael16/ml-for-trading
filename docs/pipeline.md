# ML-for-Trading Pipeline Diagram

End-to-end data flow from raw OHLCV ingestion through preprocessing, EDA, model training, and output generation.

```mermaid
flowchart TD
    subgraph INGEST["① Data Ingestion — src/data_ingestion.py"]
        A["yfinance.download()\nor synthetic OHLCV fallback"] --> B[("data/raw/\n{ticker}_{start}_{end}.parquet")]
    end

    subgraph PREP["② Preprocessing & Feature Engineering"]
        B --> C["pd.read_parquet()"]
        C --> D["add_technical_indicators()\n─────────────────────\nSMA_20, SMA_50\nRSI (14-period Wilder)\nMACD / Signal / Histogram\nBollinger Bands (20-period, 2σ)\nMomentum (10-day diff)\nLog_Returns, Volatility\n+ dropna() warm-up rows"]
        D --> E["triple_barrier_labeling()\n─────────────────────\nLabel ∈ {+1, 0, −1}\nprofit-take / time / stop-loss"]
        E --> F["prepare_features_and_labels()\n─────────────────────\ndrop NaN labels\ndrop raw OHLCV columns\nRobustScaler.fit_transform(X)"]
        F --> G["walk_forward_split()\n─────────────────────\nchronological 80 / 20 split\n(no shuffling)"]
    end

    subgraph ML["③ Modelling & Evaluation — src/train_benchmark.py"]
        G --> H["Classification models\nSVM · RandomForest\nXGBoost · LightGBM"]
        D --> I["ARIMA\non Log_Returns"]
        D --> J["GMM Market-Regime Detector\nfit on Log_Returns + Volatility\n3 regimes (low / mid / high vol)"]
        H --> K["Backtest + Financial Metrics\naccuracy · Sharpe · max drawdown\nCVaR · total return"]
        I --> K
    end

    subgraph EDA["④ EDA & Visualisation — src/ui/dashboard.py"]
        direction TB
        V1["📈 Regime-colored price overlay"]
        V2["📊 Backtest equity curve"]
        V3["🔍 SHAP feature importance"]
        V4["🗺 HMM transition heatmap"]
        V5["〰 Mean-reversion Z-score"]
    end

    subgraph OUT["⑤ Outputs"]
        K --> L[("data/processed/\n{ticker}_benchmarking_results.csv")]
        F --> M[("models/\nscaler_{ticker}.joblib")]
        H --> N[("models/\n{model}_{ticker}.joblib")]
        J --> O[("models/\nregime_detector_{ticker}.joblib")]
    end

    D -. "indicators + prices" .-> EDA
    K -. "backtest results" .-> EDA
```

---

## Stage → Code Mapping

| Stage | Key function(s) | File |
|---|---|---|
| Fetch OHLCV | `fetch_stock_data()`, `_build_synthetic_ohlcv()` | `src/data_ingestion.py` |
| Technical indicators | `add_technical_indicators()` | `src/features/technical_indicators.py` |
| Triple-barrier labeling | `triple_barrier_labeling()` | `src/features/preprocessing.py` |
| Scale & split features | `prepare_features_and_labels()`, `walk_forward_split()` | `src/features/preprocessing.py` |
| Classify + ARIMA | `run_benchmarking()`, `run_arima_benchmark()` | `src/train_benchmark.py` |
| Market-regime detection | `MarketRegimeDetector.fit()` (GMM) | `src/models/market_regime.py` |
| Backtest | `run_advanced_backtest()` | `src/models/backtester.py` |
| EDA / dashboard | all `st.*` + matplotlib/seaborn plots | `src/ui/dashboard.py` |
| Orchestration (prod) | `weekly_retrain()`, `daily_refresh()` | `src/orchestrator.py` |

> **Entry point:** `src/train_benchmark.py::run_full_pipeline()` assembles steps ①–③ in order.
