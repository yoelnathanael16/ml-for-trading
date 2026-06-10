# ML-for-Trading — Full Pipeline Reference

End-to-end data flow from raw OHLCV ingestion through preprocessing, training, orchestration, and dashboard output.

---

## End-to-End Flow

```mermaid
flowchart TD
    subgraph INGEST["① Data Ingestion — src/data_ingestion.py"]
        YF["yfinance.download(ticker, start, end)"] --> RAW
        SYN["_build_synthetic_ohlcv()\ndeterministic fallback for offline tests"] -.-> RAW
        RAW[("data/raw/\n{ticker}_{start}_{end}.parquet\n~1 500 trading rows / 6 years")]
    end

    subgraph PREP["② Feature Engineering — src/features/"]
        RAW --> TI["add_technical_indicators()\n────────────────────────────\nSMA_20, SMA_50\nRSI  (14-period Wilder EMA)\nMACD / Signal / Histogram\nBollinger Bands (20-period, 2σ)\nMomentum (10-day diff)\nLog_Returns = log(Cₜ / Cₜ₋₁)\nVolatility = rolling(20).std(Log_Returns)\n→ dropna() removes warm-up rows"]
        TI --> LBL["triple_barrier_labeling()\n────────────────────────────\nLabel ∈ {+1 profit-take, −1 stop-loss, 0 time}\nvertical_barrier=5d · profit=2% · stop=1%"]
        LBL --> PREP2["prepare_features_and_labels()\n────────────────────────────\ndrop NaN labels\ndrop OHLCV columns\nRobustScaler.fit_transform(X_train)"]
        PREP2 --> SPLIT["walk_forward_split()\n────────────────────────────\nchronological 80% train / 20% test\nno data leakage, no shuffling"]
    end

    subgraph TUNE["③ Hyperparameter Tuning — src/training/hyperparameter_tuner.py"]
        SPLIT --> HT["tune_all_models()\n────────────────────────────\nTimeSeriesSplit(n_splits=5)\nGridSearchCV(scoring='accuracy')\nfilters single-class folds automatically\nSVM · RF · XGB · LGBM"]
        HT --> BESTPARAMS["best_params per model"]
    end

    subgraph TRAIN_CORE["④ Benchmark Training — src/train_benchmark.py"]
        SPLIT --> ML["SVM  · RandomForest\nXGBoost  · LightGBM\nfit with tuned hyperparams"]
        TI --> ARIMA["ARIMA(1,0,1)\nwalk-forward Log_Returns forecast"]
        TI --> GMM["MarketRegimeDetector\nGaussianMixture(n_components=3)\nfit on [Log_Returns, Volatility]\norder by mean volatility → Low/Mid/High"]
        ML --> BT["run_advanced_backtest()\n────────────────────────────\nstate machine: Open → SL/PT/Trailing/Time\nsize: Kelly · Constant · Volatility-adj\nSMA-50 trend filter"]
        ARIMA --> BT
        BT --> METRICS["Accuracy · Sharpe · Max Drawdown\nCVaR · Total Return\n→ data/processed/{t}_benchmarking_results.csv"]
    end

    subgraph TRAIN_ADV["⑤ Advanced Models — QuantOrchestrator.weekly_retrain()"]
        TI --> HMM["HMMRegimeDetector\nGaussianHMM(n_components=3, cov='diag')\nfit on [Log_Returns, Volatility]\norder states by mean return → Bull/Sideways/Bear\n→ get_transition_matrix()"]
        TI --> GARCH["GARCHVolatilityModel\narch.arch_model(returns×100, vol='Garch', p=1, q=1)\nforecast(1) → annualized vol\n→ garch_{t}.joblib"]
        TI --> ANOM["MarketAnomalyDetector\nIsolationForest(contamination=0.05)\nfit on full 12-feature matrix\n→ anomaly_detector_{t}.joblib"]
        TI --> MR["MeanReversionDetector\nZ-score = (Cₜ − μ₂₀) / σ₂₀\nOU half-life via OLS: ΔPₜ = a + b·Pₜ₋₁\nhalf-life = −log(2)/b\n→ mean_reversion_{t}.joblib"]
        TI --> RISK["TailRiskModel\nHistorical CVaR: E[r | r ≤ VaR_α]\nVaR = −quantile(returns, 1−α)\nposition_scale = target_cvar / realized_cvar\n→ risk_model_{t}.joblib"]
        ML --> SHAP["ModelExplainer\nRF/XGB/LGBM → TreeExplainer\nSVM → KernelExplainer (kmeans background)\nprecomputed as {feature: mean_abs_shap}\n→ explainer_{model}_{t}.joblib ×4"]
    end

    subgraph ARTIFACTS["⑥ Artifact Store — models/  (15 per ticker × 4 tickers = 60 total)"]
        direction LR
        FA["scaler_{t}"] --- FB["regime_detector_{t}"] --- FC["hmm_regime_{t}"]
        FD["garch_{t}"] --- FE["anomaly_detector_{t}"] --- FF["mean_reversion_{t}"]
        FG["risk_model_{t}"] --- FH["{SVM|RF|XGB|LGB}_{t}"] --- FI["explainer_{model}_{t} ×4"]
    end

    subgraph SERVE["⑦ Serving — two independent paths"]
        subgraph API_PATH["FastAPI  (src/api/main.py)"]
            direction TB
            ORCH["QuantOrchestrator.infer(ticker)\n────────────────────\n1. check cache age (< 24h)\n2. daily_refresh() if stale\n3. return bundle dict"]
            EP["/regime · /volatility · /risk\n/anomaly · /mean-reversion · /explain"]
            ORCH --> EP
        end
        subgraph LOCAL_PATH["Standalone Streamlit (no API)"]
            direction TB
            LH["@st.cache_data helpers\n────────────────────\ncompute_hmm_local()\ncompute_volatility_local()\ncompute_risk_local()\ncompute_anomaly_local()\ncompute_mean_reversion_local()\ncompute_explain_local()"]
        end
    end

    subgraph DASH["⑧ Dashboard — src/ui/dashboard.py (8 tabs)"]
        T1["📊 Model Benchmarks\n benchmarking_results.csv"]
        T2["📈 Market Regimes GMM\n regime_detector joblib"]
        T3["⚙️ Trading Simulator\n in-process backtest"]
        T4["💼 Portfolio Allocation\n yfinance live returns"]
        T5["🔮 HMM Regime\n transition heatmap"]
        T6["📉 Risk & Volatility\n GARCH · CVaR · anomaly"]
        T7["🔄 Mean Reversion\n Z-score chart · half-life"]
        T8["🔍 Explainability\n SHAP bar chart · top-5"]
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
