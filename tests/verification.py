import os
import pandas as pd
import numpy as np
from src.features.technical_indicators import add_technical_indicators
from src.features.preprocessing import triple_barrier_labeling, prepare_features_and_labels
from src.data_ingestion import fetch_stock_data
from src.train_benchmark import run_benchmarking

def test_pipeline_smoke():
    """
    Smoke test to verify the end-to-end flow and all new trading modules.
    """
    ticker = "AAPL"
    start = "2023-01-01"
    end = "2023-12-31" # Use a fixed period for testing
    
    print("--- Starting Verification Smoke Test ---")
    
    # 1. Test Ingestion
    print("\n1. Testing Data Ingestion...")
    raw_file = fetch_stock_data(ticker, start, end, allow_synthetic_fallback=True)
    if raw_file and os.path.exists(raw_file):
        print("PASS: Data Ingestion")
    else:
        print("FAIL: Data Ingestion")
        return

    # 2. Test Feature Engineering
    print("\n2. Testing Feature Engineering...")
    df = pd.read_parquet(raw_file)
    df_feat = add_technical_indicators(df)
    required_cols = ['RSI', 'MACD', 'BB_Upper', 'Volatility', 'SMA_50']
    if all(col in df_feat.columns for col in required_cols):
        print("PASS: Feature Engineering")
    else:
        print("FAIL: Feature Engineering")
        return

    # 3. Test Market Regime Detector
    print("\n3. Testing Market Regime Detector...")
    from src.models.market_regime import MarketRegimeDetector
    regime_feat = df_feat[['Log_Returns', 'Volatility']].dropna()
    regime_det = MarketRegimeDetector(n_regimes=3)
    regime_det.fit(regime_feat)
    regimes = regime_det.predict(regime_feat)
    regime_names = regime_det.predict_regime_name(regime_feat)
    if len(regimes) == len(regime_feat) and len(regime_names) == len(regime_feat):
        print("PASS: Market Regime Detector")
    else:
        print("FAIL: Market Regime Detector")
        return

    # 4. Test Position Sizing
    print("\n4. Testing Position Sizing...")
    from src.models.position_sizing import calculate_position_sizes
    signals = np.array([1, 0, -1, 1])
    probs = np.array([[0.1, 0.1, 0.8], [0.33, 0.33, 0.33], [0.7, 0.2, 0.1], [0.2, 0.1, 0.7]])
    sizes = calculate_position_sizes(signals, probabilities=probs, method="kelly")
    if len(sizes) == 4 and sizes[1] == 0.0 and sizes[0] > 0.0 and sizes[2] < 0.0:
        print("PASS: Position Sizing")
    else:
        print("FAIL: Position Sizing")
        return

    # 5. Test Portfolio Sizing
    print("\n5. Testing Portfolio Sizing...")
    from src.models.portfolio_sizing import calculate_equal_weights, calculate_risk_parity_weights, calculate_mvo_weights
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    ret = np.array([0.15, 0.20])
    w_ew = calculate_equal_weights(2)
    w_rp = calculate_risk_parity_weights(cov)
    w_mvo = calculate_mvo_weights(ret, cov)
    if len(w_ew) == 2 and len(w_rp) == 2 and len(w_mvo) == 2 and np.isclose(np.sum(w_mvo), 1.0):
        print("PASS: Portfolio Sizing")
    else:
        print("FAIL: Portfolio Sizing")
        return

    # 6. Test Advanced Backtesting
    print("\n6. Testing Advanced Backtesting Simulator...")
    from src.models.backtester import run_advanced_backtest
    sim_res = run_advanced_backtest(
        prices=df_feat['Close'],
        signals=np.ones(len(df_feat)),
        volatilities=df_feat['Volatility'].values,
        sizing_method="volatility"
    )
    if "Total Return" in sim_res and "equity_curve" in sim_res:
        print("PASS: Advanced Backtesting Simulator")
    else:
        print("FAIL: Advanced Backtesting Simulator")
        return

    # 7. Test Benchmarking Pipeline
    print("\n7. Testing Benchmarking Pipeline...")
    results = run_benchmarking(ticker, start, end)
    if results is not None and not results.empty:
        print("PASS: Benchmarking Pipeline")
    else:
        print("FAIL: Benchmarking Pipeline")
        return

    # 8. Test HMMRegimeDetector
    print("\n8. Testing HMM Regime Detector...")
    from src.models.regime_hmm import HMMRegimeDetector
    hmm_det = HMMRegimeDetector(n_components=3)
    hmm_det.fit(regime_feat.values)
    hmm_preds = hmm_det.predict(regime_feat.values)
    hmm_names = hmm_det.get_regime_names()
    tm = hmm_det.get_transition_matrix()
    if (len(hmm_preds) == len(regime_feat)
            and hmm_names == ["Bull", "Sideways", "Bear"]
            and tm.shape == (3, 3)
            and np.allclose(tm.sum(axis=1), 1.0)):
        print("PASS: HMM Regime Detector")
    else:
        print("FAIL: HMM Regime Detector")
        return

    # 9. Test GARCHVolatilityModel
    print("\n9. Testing GARCH Volatility Model...")
    from src.models.volatility_garch import GARCHVolatilityModel
    log_returns = df_feat['Log_Returns'].dropna()
    garch_model = GARCHVolatilityModel(p=1, q=1)
    garch_model.fit(log_returns)
    garch_forecast = garch_model.forecast(horizon=1)
    garch_series = garch_model.forecast_series(log_returns)
    if (isinstance(garch_forecast, float)
            and garch_forecast > 0
            and len(garch_series) == len(log_returns)
            and not garch_series.isna().all()):
        print("PASS: GARCH Volatility Model")
    else:
        print("FAIL: GARCH Volatility Model")
        return

    # 10. Test calculate_hrp_weights
    print("\n10. Testing HRP Portfolio Weights...")
    from src.models.portfolio_hrp import calculate_hrp_weights
    rng = np.random.default_rng(42)
    returns_df = pd.DataFrame(rng.normal(0, 0.01, (100, 3)), columns=["A", "B", "C"])
    hrp_weights = calculate_hrp_weights(returns_df)
    if (isinstance(hrp_weights, dict)
            and set(hrp_weights.keys()) == {"A", "B", "C"}
            and np.isclose(sum(hrp_weights.values()), 1.0)):
        print("PASS: HRP Portfolio Weights")
    else:
        print("FAIL: HRP Portfolio Weights")
        return

    # 11. Test MarketAnomalyDetector
    print("\n11. Testing Market Anomaly Detector...")
    from src.models.anomaly_detector import MarketAnomalyDetector
    from src.features.preprocessing import prepare_features_and_labels
    df_feat_copy = df_feat.copy()
    df_feat_copy['Label'] = 0
    X_anom, _, _ = prepare_features_and_labels(df_feat_copy)
    X_anom_vals = X_anom.values

    anomaly_det = MarketAnomalyDetector(contamination=0.05)
    anomaly_det.fit(X_anom_vals)
    anomaly_preds = anomaly_det.predict(X_anom_vals)
    is_anom = anomaly_det.is_anomaly(X_anom_vals[-1:])
    if (set(np.unique(anomaly_preds)).issubset({1, -1})
            and isinstance(is_anom, (bool, np.bool_))):
        print("PASS: Market Anomaly Detector")
    else:
        print("FAIL: Market Anomaly Detector")
        return

    # 12. Test MeanReversionDetector
    print("\n12. Testing Mean Reversion Detector...")
    from src.models.mean_reversion import MeanReversionDetector
    mr_det = MeanReversionDetector(window=20)
    mr_det.fit(df_feat['Close'])
    mr_result = mr_det.predict(df_feat['Close'])
    if (isinstance(mr_result, dict)
            and "zscore" in mr_result
            and "halflife" in mr_result
            and "signal" in mr_result
            and "is_mean_reverting" in mr_result
            and mr_result["signal"] in {-1, 0, 1}):
        print("PASS: Mean Reversion Detector")
    else:
        print("FAIL: Mean Reversion Detector")
        return

    # 13. Test TailRiskModel
    print("\n13. Testing Tail Risk Model...")
    from src.models.risk_model import TailRiskModel
    risk_model = TailRiskModel()
    risk_model.fit(log_returns)
    risk_result = risk_model.compute()
    if (isinstance(risk_result, dict)
            and "cvar_95" in risk_result
            and "cvar_99" in risk_result
            and risk_result["cvar_95"] >= 0
            and risk_result["cvar_99"] >= risk_result["cvar_95"]):
        print("PASS: Tail Risk Model")
    else:
        print("FAIL: Tail Risk Model")
        return

    # 14. Test ModelExplainer
    print("\n14. Testing Model Explainer (SHAP)...")
    from src.models.explainability import ModelExplainer
    import joblib
    rf_path = "models/RandomForest_AAPL.joblib"
    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
        explainer = ModelExplainer("RandomForest", rf_model, list(X_anom.columns))
        importances = explainer.get_feature_importance(X_anom_vals[:50])
        if (isinstance(importances, dict)
                and len(importances) == X_anom.shape[1]
                and all(v >= 0 for v in importances.values())):
            print("PASS: Model Explainer")
        else:
            print("FAIL: Model Explainer")
            return
    else:
        print("SKIP: Model Explainer (model not trained yet)")

    # 15. Test hyperparameter_tuner
    print("\n15. Testing Hyperparameter Tuner (SVM only)...")
    from src.training.hyperparameter_tuner import tune_model
    df_feat_ht = df_feat.copy()
    df_feat_ht['Label'] = 0
    X_ht, y_ht, _ = prepare_features_and_labels(df_feat_ht)
    X_ht_small = X_ht.values[:100]
    # Round-robin labels so every CV fold has ≥ 2 classes
    y_ht_small = np.tile(np.array([-1, 0, 1]), len(X_ht_small) // 3 + 1)[:len(X_ht_small)]
    try:
        best_params = tune_model("SVM", X_ht_small, y_ht_small, n_splits=2)
        if isinstance(best_params, dict) and "C" in best_params:
            print("PASS: Hyperparameter Tuner")
        else:
            print("FAIL: Hyperparameter Tuner")
            return
    except Exception as e:
        print(f"FAIL: Hyperparameter Tuner — {e}")
        return

    # 16. Test QuantOrchestrator start/stop
    print("\n16. Testing QuantOrchestrator start/stop...")
    from src.orchestrator import QuantOrchestrator
    orch = QuantOrchestrator(tickers=["AAPL"], data_dir="data", models_dir="models")
    orch.start()
    status = orch.get_status()
    if (isinstance(status, dict)
            and "AAPL" in status
            and status["AAPL"]["scheduler_running"] is True):
        orch.stop()
        print("PASS: QuantOrchestrator start/stop")
    else:
        orch.stop()
        print("FAIL: QuantOrchestrator start/stop")
        return

    print("\n--- All 16 Smoke Tests Completed Successfully ---")

if __name__ == "__main__":
    test_pipeline_smoke()

