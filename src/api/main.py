from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import yfinance as yf

from src.features.technical_indicators import add_technical_indicators
from src.models.model_wrappers import ModelWrapper
from src.models.position_sizing import calculate_position_sizes
from src.models.portfolio_sizing import (
    calculate_equal_weights,
    calculate_risk_parity_weights,
    calculate_mvo_weights,
)
from src.models.portfolio_hrp import calculate_hrp_weights
from src.orchestrator import QuantOrchestrator

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

models: dict = {}
scalers: dict = {}
regime_detectors: dict = {}

orchestrator: Optional[QuantOrchestrator] = None


# ---------------------------------------------------------------------------
# Resource loading (kept for /analyze backward compat)
# ---------------------------------------------------------------------------

def load_resources(ticker: str = "AAPL") -> None:
    """Loads trained models, scalers, and regime detectors for a given ticker."""
    model_names = ["SVM", "RandomForest", "XGBoost", "LightGBM"]
    for name in model_names:
        model_path = f"models/{name}_{ticker}.joblib"
        if os.path.exists(model_path):
            raw_model = joblib.load(model_path)
            # Reconstruct ModelWrapper around raw model
            wrapper = ModelWrapper(name)
            wrapper.model = raw_model
            if name == "XGBoost":
                wrapper._class_to_label = {0: -1, 1: 0, 2: 1}
                wrapper._label_to_class = {-1: 0, 0: 1, 1: 2}
            models[name] = wrapper

    scaler_path = f"models/scaler_{ticker}.joblib"
    if os.path.exists(scaler_path):
        scalers[ticker] = joblib.load(scaler_path)

    regime_path = f"models/regime_detector_{ticker}.joblib"
    if os.path.exists(regime_path):
        regime_detectors[ticker] = joblib.load(regime_path)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator
    orchestrator = QuantOrchestrator(tickers=["AAPL", "GOOGL", "MSFT", "TSLA"])
    orchestrator.start()
    load_resources()  # keep existing load for /analyze backward compat
    yield
    orchestrator.stop()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="ML Research: Stock Market Benchmark API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    ticker: str
    ohlcv_data: list  # List of dicts with OHLCV for recent days


class PredictionResponse(BaseModel):
    ticker: str
    predictions: dict
    regime: str
    regime_gmm: str
    position_sizes: dict


class PortfolioRequest(BaseModel):
    tickers: list[str]
    method: str  # "hrp", "mvo", "rp", "ew"


# ---------------------------------------------------------------------------
# Helper: orchestrator guard
# ---------------------------------------------------------------------------

def _require_orchestrator() -> QuantOrchestrator:
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orchestrator


def _require_bundle(ticker: str) -> dict:
    orch = _require_orchestrator()
    bundle = orch.infer(ticker)
    if not bundle:
        raise HTTPException(status_code=404, detail=f"No data available for ticker {ticker}")
    return bundle


# ---------------------------------------------------------------------------
# Existing endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock Market Benchmark API"}


@app.post("/analyze", response_model=PredictionResponse)
def analyze(request: PredictionRequest):
    ticker = request.ticker
    if ticker not in scalers:
        load_resources(ticker)
        if ticker not in scalers:
            raise HTTPException(status_code=404, detail=f"Models for ticker {ticker} not found.")

    # Convert incoming data to DataFrame
    df = pd.DataFrame(request.ohlcv_data)

    # Calculate features (Technical Indicators)
    try:
        df_with_features = add_technical_indicators(df)

        # Features are everything except OHLCV
        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
        X = df_with_features.drop(columns=[col for col in ohlcv_cols if col in df_with_features.columns])

        # Scale features
        X_scaled = scalers[ticker].transform(X)

        # Run inference and position sizing across models
        preds = {}
        position_sizes = {}
        for name, wrapper in models.items():
            # Use the most recent data point for current prediction
            p = wrapper.predict(X_scaled[-1:])
            preds[name] = int(p[0])

            # Probability and Position Sizing (Kelly method)
            prob = wrapper.predict_proba(X_scaled[-1:])
            size = calculate_position_sizes(
                signals=[preds[name]],
                probabilities=prob,
                method="kelly",
            )[0]
            position_sizes[name] = float(size)

        # Determine Market Regime (Simplified consensus)
        consensus = sum(preds.values())
        if consensus > 0:
            regime = "Bullish"
        elif consensus < 0:
            regime = "Bearish"
        else:
            regime = "Neutral"

        # Determine GMM-based Market Regime
        regime_gmm = "Unknown"
        if ticker in regime_detectors:
            # Predict regime using last data point of Log_Returns and Volatility
            last_feat = df_with_features[["Log_Returns", "Volatility"]].iloc[-1:]
            regime_gmm = regime_detectors[ticker].predict_regime_name(last_feat)[0]

        return PredictionResponse(
            ticker=ticker,
            predictions=preds,
            regime=regime,
            regime_gmm=regime_gmm,
            position_sizes=position_sizes,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# New endpoints
# ---------------------------------------------------------------------------

@app.get("/status")
def get_status():
    """Return orchestrator status for all tracked tickers."""
    orch = _require_orchestrator()
    return orch.get_status()


@app.get("/regime/{ticker}")
def get_regime(ticker: str):
    """Return GMM and HMM regime information for a ticker."""
    bundle = _require_bundle(ticker)
    return {
        "ticker": ticker,
        "regime_gmm": str(bundle.get("regime_gmm")),
        "regime_hmm": str(bundle.get("regime_hmm")),
        "hmm_transition_matrix": bundle.get("hmm_transition_matrix"),
    }


@app.get("/volatility/{ticker}")
def get_volatility(ticker: str):
    """Return GARCH and rolling volatility for a ticker."""
    bundle = _require_bundle(ticker)
    return {
        "ticker": ticker,
        "garch_vol": bundle.get("garch_vol"),
        "rolling_vol": bundle.get("rolling_vol"),
    }


@app.post("/portfolio")
def get_portfolio(request: PortfolioRequest):
    """Compute portfolio weights using the specified method."""
    tickers = request.tickers
    method = request.method.lower()

    # Fetch returns
    data = yf.download(tickers, period="1y", progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    returns = np.log(data / data.shift(1)).dropna()

    if method == "hrp":
        weights = calculate_hrp_weights(returns)
    elif method == "ew":
        w = calculate_equal_weights(len(tickers))
        weights = {t: float(w[i]) for i, t in enumerate(tickers)}
    elif method == "rp":
        cov = returns.cov().values
        w = calculate_risk_parity_weights(cov)
        weights = {t: float(w[i]) for i, t in enumerate(tickers)}
    elif method == "mvo":
        mu = returns.mean().values * 252
        cov = returns.cov().values * 252
        w = calculate_mvo_weights(mu, cov)
        weights = {t: float(w[i]) for i, t in enumerate(tickers)}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown method: {method}")

    return {"method": method, "weights": weights}


@app.get("/anomaly/{ticker}")
def get_anomaly(ticker: str):
    """Return anomaly detection results for a ticker."""
    bundle = _require_bundle(ticker)
    return {
        "ticker": ticker,
        "anomaly_flag": bundle.get("anomaly_flag"),
        "anomaly_score": bundle.get("anomaly_score"),
    }


@app.get("/mean-reversion/{ticker}")
def get_mean_reversion(ticker: str):
    """Return mean reversion analysis for a ticker."""
    bundle = _require_bundle(ticker)
    return {
        "ticker": ticker,
        "zscore": bundle.get("zscore"),
        "halflife": bundle.get("halflife"),
        "mr_signal": bundle.get("mr_signal"),
        "is_mean_reverting": bundle.get("is_mean_reverting"),
    }


@app.get("/risk/{ticker}")
def get_risk(ticker: str):
    """Return tail risk metrics for a ticker."""
    bundle = _require_bundle(ticker)
    return {
        "ticker": ticker,
        "cvar_95": bundle.get("cvar_95"),
        "cvar_99": bundle.get("cvar_99"),
        "var_95": bundle.get("var_95"),
        "var_99": bundle.get("var_99"),
        "position_scale": bundle.get("position_scale"),
    }


@app.get("/explain/{ticker}/{model}")
def get_explain(ticker: str, model: str):
    """Return SHAP feature importances for a given ticker and model."""
    bundle = _require_bundle(ticker)
    shap_importances = bundle.get("shap_importances", {})
    if model not in shap_importances or shap_importances[model] is None:
        raise HTTPException(
            status_code=404,
            detail=f"No SHAP importances available for model '{model}' on ticker {ticker}",
        )
    return {
        "ticker": ticker,
        "model": model,
        "feature_importances": shap_importances[model],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
