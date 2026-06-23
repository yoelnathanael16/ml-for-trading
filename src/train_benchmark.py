import pandas as pd
import numpy as np
import os
import joblib
from src.features.technical_indicators import add_technical_indicators
from src.features.preprocessing import triple_barrier_labeling, prepare_features_and_labels, walk_forward_split
from src.models.model_wrappers import ModelWrapper, calculate_financial_metrics, run_arima_benchmark
from src.models.market_regime import MarketRegimeDetector
from src.models.backtester import run_advanced_backtest
from sklearn.metrics import accuracy_score

def run_full_pipeline(ticker: str, start_date: str, end_date: str, data_dir: str = "data", models_dir: str = "models", reporter=None) -> dict:
    """
    Run the full ML benchmarking pipeline with configurable directory paths.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data (YYYY-MM-DD)
        end_date: End date for data (YYYY-MM-DD)
        data_dir: Directory path for data files (default: "data")
        models_dir: Directory path for model files (default: "models")

    Returns:
        dict with keys:
            - results_df: DataFrame with benchmarking results
            - ticker: Stock ticker
            - models_dir: Models directory path
            - model_wrappers: Dict of trained ModelWrapper objects
            - scaler: Fitted scaler object
    """
    if reporter:
        reporter.start()

    # 1. Load Data
    raw_path = f"{data_dir}/raw/{ticker}_{start_date}_{end_date}.parquet"
    print(f"Loading data from {raw_path}...")
    if not os.path.exists(raw_path):
        print(f"Data not found at {raw_path}. Please run data ingestion first.")
        return None

    if reporter:
        reporter.advance("Preprocessing data...", "preprocess")

    df = pd.read_parquet(raw_path)

    # 2. Feature Engineering
    df = add_technical_indicators(df)

    # 3. Labeling
    df['Label'] = triple_barrier_labeling(df)

    # 4. Prepare Features & Labels
    X, y, scaler = prepare_features_and_labels(df)

    # 5. Walk-Forward Split
    X_train, X_test, y_train, y_test = walk_forward_split(X, y)

    # 6. Fit Market Regime Detector (on all historical indicators)
    print("Training Market Regime Detector (GMM)...")
    if reporter:
        reporter.advance("Training Market Regime Detector (GMM)...", "gmm")
    regime_features = df[['Log_Returns', 'Volatility']].dropna()
    regime_detector = MarketRegimeDetector(n_regimes=3)
    regime_detector.fit(regime_features)

    # Save regime detector
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(regime_detector, f"{models_dir}/regime_detector_{ticker}.joblib")
    print(f"Regime detector saved to {models_dir}/regime_detector_{ticker}.joblib")

    # 7. Benchmarking
    results = {}
    model_wrappers = {}
    models = ['SVM', 'RandomForest', 'XGBoost', 'LightGBM']

    test_prices = df.loc[X_test.index, 'Close']
    test_vols = df.loc[X_test.index, 'Volatility'].values
    test_sma_50 = df.loc[X_test.index, 'SMA_50'].values
    test_trend = test_prices.values > test_sma_50

    _model_stage_key = {
        "SVM": "svm", "RandomForest": "randomforest",
        "XGBoost": "xgboost", "LightGBM": "lightgbm",
    }

    for model_name in models:
        print(f"Training {model_name}...")
        if reporter:
            reporter.advance(f"Training {model_name}...", _model_stage_key[model_name])
        wrapper = ModelWrapper(model_name)
        wrapper.train(X_train, y_train)
        preds = wrapper.predict(X_test)
        probas = wrapper.predict_proba(X_test)

        # Store wrapper for later use
        model_wrappers[model_name] = wrapper

        # Statistical Metrics
        acc = accuracy_score(y_test, preds)

        # Financial Metrics - Base (Hold always)
        fin_metrics = calculate_financial_metrics(y_test, preds, test_prices)

        # Financial Metrics - Advanced (Kelly Sizing, Stop Loss=1.5%, Profit Target=3.0%, Trailing Stop=2.0%)
        adv_backtest = run_advanced_backtest(
            prices=test_prices,
            signals=preds,
            probabilities=probas,
            volatilities=test_vols,
            trend_filter=test_trend,
            sizing_method="kelly",
            stop_loss=0.015,
            profit_taking=0.03,
            trailing_stop=0.02,
            time_barrier=5
        )

        results[model_name] = {
            "Accuracy": acc,
            "Total Return (Base)": fin_metrics["Total Return"],
            "Sharpe Ratio (Base)": fin_metrics["Sharpe Ratio"],
            "Max Drawdown (Base)": fin_metrics["Max Drawdown"],
            "Total Return (Adv)": adv_backtest["Total Return"],
            "Sharpe Ratio (Adv)": adv_backtest["Sharpe Ratio"],
            "Max Drawdown (Adv)": adv_backtest["Max Drawdown"],
        }

        # Save model
        joblib.dump(wrapper.model, f"{models_dir}/{model_name}_{ticker}.joblib")

    # ARIMA Special Handling
    print("Running ARIMA Benchmark (This may take a while)...")
    if reporter:
        reporter.advance("Running ARIMA Benchmark...", "arima")
    # For ARIMA, we use log returns of the close price
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    train_size = int(len(log_returns) * 0.8)

    def _arima_cb(i, total):
        if reporter:
            reporter.sub("Running ARIMA Benchmark...", "arima", i / total)

    arima_preds_raw = run_arima_benchmark(log_returns.values, train_size, progress_cb=_arima_cb)

    # Convert ARIMA forecast to signals (1 if pos return, -1 if neg)
    arima_signals = np.where(arima_preds_raw > 0, 1, -1)

    # For comparison, we'll calculate financial metrics for ARIMA
    arima_prices = df['Close'].iloc[train_size+1:]
    arima_vols = df['Volatility'].iloc[train_size+1:].values
    arima_sma_50 = df['SMA_50'].iloc[train_size+1:].values
    arima_trend = arima_prices.values > arima_sma_50

    arima_fin = calculate_financial_metrics(None, arima_signals, arima_prices)

    # ARIMA Advanced (Volatility Sizing, SL 1.5%, PT 3.0%, TS 2.0%)
    arima_adv = run_advanced_backtest(
        prices=arima_prices,
        signals=arima_signals,
        probabilities=None,
        volatilities=arima_vols,
        trend_filter=arima_trend,
        sizing_method="volatility",
        stop_loss=0.015,
        profit_taking=0.03,
        trailing_stop=0.02,
        time_barrier=5
    )

    results["ARIMA"] = {
        "Accuracy": "N/A (Forecasting)",
        "Total Return (Base)": arima_fin["Total Return"],
        "Sharpe Ratio (Base)": arima_fin["Sharpe Ratio"],
        "Max Drawdown (Base)": arima_fin["Max Drawdown"],
        "Total Return (Adv)": arima_adv["Total Return"],
        "Sharpe Ratio (Adv)": arima_adv["Sharpe Ratio"],
        "Max Drawdown (Adv)": arima_adv["Max Drawdown"],
    }

    # 8. Summary
    results_df = pd.DataFrame(results).T
    print("\nBenchmarking Results:")
    print(results_df)

    os.makedirs(f"{data_dir}/processed", exist_ok=True)
    results_df.to_csv(f"{data_dir}/processed/{ticker}_benchmarking_results.csv")
    print(f"\nResults saved to {data_dir}/processed/{ticker}_benchmarking_results.csv")

    # Save scaler for API usage
    joblib.dump(scaler, f"{models_dir}/scaler_{ticker}.joblib")
    if reporter:
        reporter.finish()

    # Return dict with results and supporting objects
    return {
        "results_df": results_df,
        "ticker": ticker,
        "models_dir": models_dir,
        "model_wrappers": model_wrappers,
        "scaler": scaler
    }


def run_benchmarking(ticker, start_date, end_date, reporter=None):
    """
    Legacy function for backward compatibility. Calls run_full_pipeline() with default directories.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data (YYYY-MM-DD)
        end_date: End date for data (YYYY-MM-DD)
        reporter: optional ProgressReporter for UI feedback

    Returns:
        DataFrame with benchmarking results
    """
    result = run_full_pipeline(ticker, start_date, end_date, data_dir="data", models_dir="models", reporter=reporter)
    if result is None:
        return None
    return result["results_df"]

if __name__ == "__main__":
    ticker = "AAPL"
    start = "2020-01-01"
    end = "2026-05-11"  # Current session date
    run_full_pipeline(ticker, start, end, data_dir="data", models_dir="models")

