# ML-for-Trading — Full Pipeline Reference

End-to-end data flow from raw OHLCV ingestion through preprocessing, training, orchestration, and dashboard output.

---

## End-to-End Flow

```mermaid
flowchart TD
    subgraph INGEST["① Data Ingestion"]
        YF["yfinance.download(ticker, start, end)"] --> RAW
        SYN["_build_synthetic_ohlcv()\noffline deterministic fallback"] -.-> RAW
        RAW[("data/raw/{ticker}_{start}_{end}.parquet")]
    end

    subgraph PREP["② Feature Engineering"]
        RAW --> TI["add_technical_indicators()\nSMA_20/50 · RSI · MACD · BB\nMomentum · Log_Returns · Volatility"]
        TI --> LBL["triple_barrier_labeling()\nLabel ∈ {+1, −1, 0}\nbarrier=5d · profit=2% · stop=1%"]
        LBL --> PREP2["prepare_features_and_labels()\ndrop NaN & OHLCV · RobustScaler"]
        PREP2 --> SPLIT["walk_forward_split()\n80% train / 20% test · chronological"]
    end

    subgraph TUNE["③ Hyperparameter Tuning"]
        SPLIT --> HT["tune_all_models()\nGridSearchCV · TimeSeriesSplit(5)\nSVM · RF · XGB · LGBM"]
        HT --> BESTPARAMS["best_params per model"]
    end

    subgraph TRAIN_CORE["④ Benchmark Training"]
        SPLIT --> ML["SVM · RandomForest\nXGBoost · LightGBM"]
        TI --> ARIMA["ARIMA(1,0,1)\nwalk-forward Log_Returns forecast"]
        TI --> GMM["MarketRegimeDetector\nGMM(n=3) on Log_Returns · Volatility"]
        ML --> BT["run_advanced_backtest()\nKelly/Vol sizing · SL/PT/TS · SMA-50"]
        ARIMA --> BT
        BT --> METRICS["Accuracy · Sharpe · MaxDD · Return\n→ data/processed/{t}_benchmarking_results.csv"]
    end

    subgraph TRAIN_ADV["⑤ Advanced Models (weekly_retrain)"]
        TI --> HMM["HMMRegimeDetector\nGaussianHMM(n=3) → Bull/Sideways/Bear"]
        TI --> GARCH["GARCHVolatilityModel\nGARCH(1,1) annualized vol forecast"]
        TI --> ANOM["MarketAnomalyDetector\nIsolationForest(contamination=0.05)"]
        TI --> MR["MeanReversionDetector\nZ-score · OU half-life via OLS"]
        TI --> RISK["TailRiskModel\nHistorical CVaR/VaR 95%/99%"]
        ML --> SHAP["ModelExplainer\nTreeSHAP (RF/XGB/LGBM) · KernelSHAP (SVM)"]
    end

    subgraph ARTIFACTS["⑥ Artifact Store — models/  (15 per ticker)"]
        direction LR
        FA["scaler_{t}"] --- FB["regime_detector_{t}"] --- FC["hmm_regime_{t}"]
        FD["garch_{t}"] --- FE["anomaly_detector_{t}"] --- FF["mean_reversion_{t}"]
        FG["risk_model_{t}"] --- FH["{SVM|RF|XGB|LGB}_{t}"] --- FI["explainer_{model}_{t} ×4"]
    end

    subgraph SERVE["⑦ Serving"]
        subgraph API_PATH["FastAPI  (src/api/main.py)"]
            direction TB
            ORCH["QuantOrchestrator.infer(ticker)\ncache check → daily_refresh if stale"] --> EP["/regime · /volatility · /risk\n/anomaly · /mean-reversion · /explain"]
        end
        subgraph LOCAL_PATH["Standalone Streamlit (no API)"]
            direction TB
            LH["@st.cache_data helpers\ncompute_hmm/volatility/risk/anomaly/mr/explain_local()"]
        end
    end

    subgraph DASH["⑧ Dashboard — src/ui/dashboard.py (8 tabs)"]
        T1["📊 Model Benchmarks"]
        T2["📈 Market Regimes GMM"]
        T3["⚙️ Trading Simulator"]
        T4["💼 Portfolio Allocation"]
        T5["🔮 HMM Regime"]
        T6["📉 Risk & Volatility"]
        T7["🔄 Mean Reversion"]
        T8["🔍 Explainability"]
    end

    TRAIN_CORE --> ARTIFACTS
    TRAIN_ADV --> ARTIFACTS
    ARTIFACTS --> API_PATH
    TI --> LOCAL_PATH

    EP --> T5 & T6 & T7 & T8
    LOCAL_PATH --> T5 & T6 & T7 & T8
    METRICS --> T1
    GMM --> T2
    BT --> T3
```

---

## Standalone Compute Detail

Tabs 5–8 work without an API server by fitting models in-process on every **cache miss**:

```mermaid
sequenceDiagram
    participant U as User (browser)
    participant ST as Streamlit
    participant C as @st.cache_data
    participant M as Model classes

    U->>ST: open tab (or change ticker)
    ST->>C: compute_hmm_local(ticker, df)
    alt cache hit (same ticker+df)
        C-->>ST: cached dict (instant)
    else cache miss (first run / new ticker)
        C->>M: HMMRegimeDetector().fit(X)
        M->>M: GaussianHMM.fit(Log_Returns, Volatility)
        M-->>C: {regime_hmm, regime_gmm, hmm_transition_matrix}
        C-->>ST: dict (stored for reuse)
    end
    ST-->>U: render metrics + heatmap
```

The same pattern applies to `compute_volatility_local` (GARCH fit ~2s), `compute_risk_local` (instant), `compute_anomaly_local` (IsolationForest ~1s), `compute_mean_reversion_local` (instant), and `compute_explain_local` (tree SHAP ~3s · SVM KernelExplainer ~30s on 50 rows).

---

## Stage → Code Mapping

| Stage | Key function(s) | File |
|---|---|---|
| Fetch OHLCV | `fetch_stock_data()`, `_build_synthetic_ohlcv()` | `src/data_ingestion.py` |
| Technical indicators | `add_technical_indicators()` | `src/features/technical_indicators.py` |
| Triple-barrier labeling | `triple_barrier_labeling()` | `src/features/preprocessing.py` |
| Scale & split | `prepare_features_and_labels()`, `walk_forward_split()` | `src/features/preprocessing.py` |
| Hyperparameter tuning | `tune_model()`, `tune_all_models()` | `src/training/hyperparameter_tuner.py` |
| ML classifiers | `ModelWrapper.train()`, `run_benchmarking()` | `src/train_benchmark.py` |
| ARIMA baseline | `run_arima_benchmark()` | `src/models/model_wrappers.py` |
| GMM regime | `MarketRegimeDetector.fit()` | `src/models/market_regime.py` |
| HMM regime | `HMMRegimeDetector.fit()` | `src/models/regime_hmm.py` |
| GARCH volatility | `GARCHVolatilityModel.fit()`, `.forecast()` | `src/models/volatility_garch.py` |
| Anomaly detection | `MarketAnomalyDetector.fit()` | `src/models/anomaly_detector.py` |
| Mean reversion | `MeanReversionDetector.predict()` | `src/models/mean_reversion.py` |
| Tail risk | `TailRiskModel.fit()`, `.compute()` | `src/models/risk_model.py` |
| SHAP explainability | `ModelExplainer.get_feature_importance()` | `src/models/explainability.py` |
| Backtest | `run_advanced_backtest()` | `src/models/backtester.py` |
| Portfolio | `calculate_{ew|rp|mvo|hrp}_weights()` | `src/models/portfolio_{sizing|hrp}.py` |
| Position sizing | `calculate_position_sizes()` | `src/models/position_sizing.py` |
| Orchestration | `QuantOrchestrator.weekly_retrain()`, `.daily_refresh()` | `src/orchestrator.py` |
| Standalone helpers | `compute_*_local()` | `src/ui/dashboard.py` |
| API serving | all `@app.get` / `@app.post` routes | `src/api/main.py` |

> **Entry point for full training:** `QuantOrchestrator.weekly_retrain(ticker)` — runs all stages in order, saves all 15 artifacts.
> **Entry point for benchmarks only:** `src/train_benchmark.py::run_full_pipeline()` — saves the 6-artifact subset.
