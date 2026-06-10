#!/usr/bin/env python3
"""
Static snapshot exporter for GitHub Pages deployment.

Trains all models (via QuantOrchestrator.weekly_retrain) and exports
per-ticker JSON snapshots + supporting JSON files to docs/data/.

Usage:
    python scripts/export_static.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

# Ensure src/ imports work when run from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion import fetch_stock_data
from src.features.preprocessing import (
    prepare_features_and_labels,
    walk_forward_split,
)
from src.features.technical_indicators import add_technical_indicators
from src.models.backtester import run_advanced_backtest
from src.models.model_wrappers import ModelWrapper
from src.models.portfolio_hrp import calculate_hrp_weights
from src.models.portfolio_sizing import (
    calculate_equal_weights,
    calculate_mvo_weights,
    calculate_risk_parity_weights,
)
from src.orchestrator import QuantOrchestrator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "GOOGL", "MSFT", "TSLA"]
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
PORTFOLIO_TICKERS = ["AAPL", "MSFT", "GOOGL"]  # dashboard default 3-ticker portfolio
OUTPUT_DIR = "docs/data"
DATA_DIR = "data"
MODELS_DIR = "models"

ML_MODELS = ["SVM", "RandomForest", "XGBoost", "LightGBM"]

# ---------------------------------------------------------------------------
# JSON encoder — handles numpy scalars, pandas Timestamps, NaN/inf
# ---------------------------------------------------------------------------


class SafeEncoder(json.JSONEncoder):
    def iterencode(self, obj, _one_shot=False):
        return super().iterencode(self._clean(obj), _one_shot)

    def _clean(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.ndarray):
            return [self._clean(x) for x in obj.tolist()]
        if isinstance(obj, pd.Timestamp):
            return str(obj.date())
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, dict):
            return {k: self._clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._clean(v) for v in obj]
        return obj


def _sf(v) -> float | None:
    """Safe float — returns None for NaN / inf / None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _date_str(idx) -> str:
    return str(idx.date()) if hasattr(idx, "date") else str(idx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Step 1: pre-fetch raw data (synthetic fallback ON → CI never flakes) ──
    print("=" * 60)
    print("STEP 1: Fetching / pre-fetching OHLCV data")
    print("=" * 60)
    for ticker in TICKERS:
        print(f"\n[{ticker}] Fetching {START_DATE} → {END_DATE} …")
        fetch_stock_data(
            ticker, START_DATE, END_DATE,
            os.path.join(DATA_DIR, "raw"),
            allow_synthetic_fallback=True,
        )

    # ── Step 2: train models ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Training models via QuantOrchestrator.weekly_retrain")
    print("=" * 60)
    orch = QuantOrchestrator(
        tickers=TICKERS,
        data_dir=DATA_DIR,
        models_dir=MODELS_DIR,
    )
    # Do NOT call orch.start() — we don't want the APScheduler or the
    # auto-retrain loop that fires for missing models. Call directly per-ticker.
    for ticker in TICKERS:
        print(f"\n[{ticker}] weekly_retrain …")
        try:
            orch.weekly_retrain(ticker)
            print(f"[{ticker}] Done.")
        except Exception as exc:
            print(f"[{ticker}] ERROR: {exc}")
            import traceback
            traceback.print_exc()

    # ── Step 3: per-ticker JSON ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Exporting per-ticker JSON snapshots")
    print("=" * 60)
    for ticker in TICKERS:
        print(f"\n[{ticker}] Building snapshot …")
        try:
            payload = _build_ticker_payload(ticker, orch)
            out_path = os.path.join(OUTPUT_DIR, f"{ticker}.json")
            with open(out_path, "w") as fh:
                json.dump(payload, fh, cls=SafeEncoder)
            print(f"[{ticker}] → {out_path}")
        except Exception as exc:
            print(f"[{ticker}] ERROR: {exc}")
            import traceback
            traceback.print_exc()

    # ── Step 4: portfolio JSON ────────────────────────────────────────────────
    print("\n[portfolio] Building portfolio snapshot …")
    try:
        port_payload = _build_portfolio_payload(PORTFOLIO_TICKERS)
        with open(os.path.join(OUTPUT_DIR, "portfolio.json"), "w") as fh:
            json.dump(port_payload, fh, cls=SafeEncoder)
        print("[portfolio] Done.")
    except Exception as exc:
        print(f"[portfolio] ERROR: {exc}")
        import traceback
        traceback.print_exc()

    # ── Step 5: status JSON ───────────────────────────────────────────────────
    print("\n[status] Building status snapshot …")
    try:
        status = orch.get_status()
        with open(os.path.join(OUTPUT_DIR, "status.json"), "w") as fh:
            json.dump(status, fh, cls=SafeEncoder)
        print("[status] Done.")
    except Exception as exc:
        print(f"[status] ERROR: {exc}")

    # ── Step 6: manifest ──────────────────────────────────────────────────────
    manifest = {
        "tickers": TICKERS,
        "portfolio_tickers": PORTFOLIO_TICKERS,
        "build_time": datetime.now().isoformat(),
    }
    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as fh:
        json.dump(manifest, fh)
    print(f"\n[manifest] Build time: {manifest['build_time']}")
    print("\n✓ Static snapshot ready at docs/data/")


# ---------------------------------------------------------------------------
# Per-ticker payload
# ---------------------------------------------------------------------------


def _load_df(ticker: str) -> pd.DataFrame:
    """Load the most recent parquet for ticker and add technical indicators."""
    raw_dir = os.path.join(DATA_DIR, "raw")
    candidates = sorted(f for f in os.listdir(raw_dir) if f.startswith(ticker) and f.endswith(".parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet found for {ticker} in {raw_dir}")
    df = pd.read_parquet(os.path.join(raw_dir, candidates[-1]))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return add_technical_indicators(df)


def _build_ticker_payload(ticker: str, orch: QuantOrchestrator) -> dict:
    # Access inference bundle populated by weekly_retrain (avoid calling infer()
    # which would trigger daily_refresh → yfinance call that could fail).
    with orch._lock:
        bundle = dict(orch._cache.get(ticker, {}))

    df = _load_df(ticker)

    payload: dict = {
        "ticker": ticker,
        "last_updated": bundle.get("last_updated", datetime.now().isoformat()),
        "tab1_benchmarks": _build_tab1(ticker),
        "tab2_gmm": _build_tab2(ticker, df),
        "tab3_simulator": _build_tab3(ticker, df),
        "tab5_hmm": {
            "regime_gmm": bundle.get("regime_gmm"),
            "regime_hmm": bundle.get("regime_hmm"),
            "hmm_transition_matrix": bundle.get("hmm_transition_matrix"),
        },
        "tab6_risk": {
            "garch_vol": _sf(bundle.get("garch_vol")),
            "rolling_vol": _sf(bundle.get("rolling_vol")),
            "cvar_95": _sf(bundle.get("cvar_95")),
            "cvar_99": _sf(bundle.get("cvar_99")),
            "var_95": _sf(bundle.get("var_95")),
            "var_99": _sf(bundle.get("var_99")),
            "position_scale": _sf(bundle.get("position_scale")),
            "anomaly_flag": bundle.get("anomaly_flag"),
            "anomaly_score": _sf(bundle.get("anomaly_score")),
        },
        "tab7_mr": _build_tab7(bundle, df),
        "tab8_shap": {m: (bundle.get("shap_importances") or {}).get(m) for m in ML_MODELS},
    }
    return payload


# ---------------------------------------------------------------------------
# Individual tab builders
# ---------------------------------------------------------------------------


def _build_tab1(ticker: str) -> dict:
    """Read benchmarking results CSV → rows list."""
    path = os.path.join(DATA_DIR, "processed", f"{ticker}_benchmarking_results.csv")
    if not os.path.exists(path):
        return {"rows": [], "error": "Benchmarking results not available"}
    df = pd.read_csv(path, index_col=0)
    rows = []
    for model_name, row in df.iterrows():
        rows.append({
            "model": str(model_name),
            "accuracy": str(row.get("Accuracy", "N/A")),
            "total_return_base": _sf(row.get("Total Return (Base)")),
            "sharpe_base": _sf(row.get("Sharpe Ratio (Base)")),
            "max_drawdown_base": _sf(row.get("Max Drawdown (Base)")),
            "total_return_adv": _sf(row.get("Total Return (Adv)")),
            "sharpe_adv": _sf(row.get("Sharpe Ratio (Adv)")),
            "max_drawdown_adv": _sf(row.get("Max Drawdown (Adv)")),
        })
    return {"rows": rows}


def _build_tab2(ticker: str, df: pd.DataFrame) -> dict:
    """GMM regime per-date series + per-regime stats."""
    path = os.path.join(MODELS_DIR, f"regime_detector_{ticker}.joblib")
    if not os.path.exists(path):
        return {"price_series": [], "regime_stats": [], "error": "GMM model not available"}

    det = joblib.load(path)
    feats = df[["Log_Returns", "Volatility"]].dropna()
    regimes = det.predict(feats)

    df_r = df.loc[feats.index].copy()
    df_r["Regime"] = regimes

    price_series = [
        {"date": _date_str(idx), "close": _sf(row["Close"]), "regime": int(row["Regime"])}
        for idx, row in df_r.iterrows()
    ]

    regime_stats = []
    for r_id in range(3):
        sub = feats[df_r["Regime"] == r_id]
        regime_stats.append({
            "regime_id": r_id,
            "total_days": len(sub),
            "mean_daily_return_pct": _sf(sub["Log_Returns"].mean() * 100),
            "daily_vol_pct": _sf(sub["Volatility"].mean() * 100),
        })

    return {"price_series": price_series, "regime_stats": regime_stats}


def _build_tab3(ticker: str, df: pd.DataFrame) -> dict:
    """Trading simulator snapshot — defaults matching the Streamlit dashboard."""
    available = [m for m in ML_MODELS if os.path.exists(os.path.join(MODELS_DIR, f"{m}_{ticker}.joblib"))]
    if not available:
        return {"error": "No ML models available", "metrics": {}}

    selected_model = available[0]

    try:
        scaler = joblib.load(os.path.join(MODELS_DIR, f"scaler_{ticker}.joblib"))
        df_feat = df.copy()
        df_feat["Label"] = 0  # dummy
        X, _, _ = prepare_features_and_labels(df_feat)
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

        # Re-create label for walk-forward split (we only need the split indices)
        from src.features.preprocessing import triple_barrier_labeling
        df_feat2 = df.copy()
        df_feat2["Label"] = triple_barrier_labeling(df_feat2)
        df_feat2 = df_feat2.dropna(subset=["Label"])
        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
        drop_cols = [c for c in ohlcv_cols if c in df_feat2.columns] + ["Label"]
        X2 = df_feat2.drop(columns=drop_cols)
        common_idx = X_scaled.index.intersection(X2.index)
        X_scaled_aligned = X_scaled.loc[common_idx]
        y_dummy = df_feat2.loc[common_idx, "Label"]

        X_train, X_test, _, _ = walk_forward_split(X_scaled_aligned, y_dummy)
        test_prices = df.loc[X_test.index, "Close"]
        test_vols = df.loc[X_test.index, "Volatility"].values
        test_sma_50 = df.loc[X_test.index, "SMA_50"].values
        test_trend = test_prices.values > test_sma_50

        wrapper = ModelWrapper(selected_model)
        wrapper.model = joblib.load(os.path.join(MODELS_DIR, f"{selected_model}_{ticker}.joblib"))
        if selected_model == "XGBoost":
            wrapper._class_to_label = {0: -1, 1: 0, 2: 1}
            wrapper._label_to_class = {-1: 0, 0: 1, 1: 2}

        preds = wrapper.predict(X_test)
        probas = wrapper.predict_proba(X_test)

        # Dashboard defaults: kelly, SL 1.5%, PT 3.0%, TS 2.0%, time_barrier 5
        sim = run_advanced_backtest(
            prices=test_prices,
            signals=preds,
            probabilities=probas,
            volatilities=test_vols,
            trend_filter=test_trend,
            sizing_method="kelly",
            stop_loss=0.015,
            profit_taking=0.03,
            trailing_stop=0.02,
            time_barrier=5,
        )

        bh_curve = test_prices / test_prices.iloc[0]

        def _series_to_list(s):
            return [{"date": _date_str(idx), "value": _sf(v)} for idx, v in s.items()]

        trades = []
        for t in sim["trades"][-10:][::-1]:
            trades.append({
                "entry_date": _date_str(t["entry_date"]),
                "exit_date": _date_str(t["exit_date"]),
                "position": t["position"],
                "entry_price": _sf(t["entry_price"]),
                "exit_price": _sf(t["exit_price"]),
                "size_pct": _sf(t["size"] * 100),
                "reason": t["reason"],
                "pnl_pct": _sf(t["pnl"] * 100),
            })

        return {
            "model": selected_model,
            "params": {
                "sizing_method": "kelly",
                "stop_loss_pct": 1.5,
                "profit_target_pct": 3.0,
                "trailing_stop_pct": 2.0,
                "time_barrier": 5,
                "trend_filter": True,
            },
            "metrics": {
                "total_return": _sf(sim["Total Return"]),
                "sharpe_ratio": _sf(sim["Sharpe Ratio"]),
                "max_drawdown": _sf(sim["Max Drawdown"]),
                "num_trades": len(sim["trades"]),
            },
            "equity_curve": _series_to_list(sim["equity_curve"]),
            "bh_curve": _series_to_list(bh_curve),
            "trades": trades,
        }
    except Exception as exc:
        print(f"  [tab3 {ticker}] WARNING: {exc}")
        import traceback
        traceback.print_exc()
        return {"error": str(exc), "metrics": {}}


def _build_tab7(bundle: dict, df: pd.DataFrame) -> dict:
    """Mean reversion bundle metrics + rolling-20 z-score time series."""
    result = {
        "zscore": _sf(bundle.get("zscore")),
        "halflife": _sf(bundle.get("halflife")),
        "mr_signal": bundle.get("mr_signal"),
        "is_mean_reverting": bundle.get("is_mean_reverting"),
    }
    try:
        prices = df["Close"].dropna()
        rmean = prices.rolling(20).mean()
        rstd = prices.rolling(20).std()
        zs = ((prices - rmean) / rstd).dropna()
        result["zscore_series"] = [
            {"date": _date_str(idx), "zscore": _sf(v)} for idx, v in zs.items()
        ]
    except Exception as exc:
        print(f"  [tab7] WARNING z-score series: {exc}")
        result["zscore_series"] = []
    return result


# ---------------------------------------------------------------------------
# Portfolio payload
# ---------------------------------------------------------------------------


def _build_portfolio_payload(tickers: list[str]) -> dict:
    """EW / RP / MVO / HRP weights + expected performance metrics."""
    raw_dir = os.path.join(DATA_DIR, "raw")
    prices_dict = {}
    for t in tickers:
        candidates = sorted(f for f in os.listdir(raw_dir) if f.startswith(t) and f.endswith(".parquet"))
        if candidates:
            df_t = pd.read_parquet(os.path.join(raw_dir, candidates[-1]))
            if isinstance(df_t.columns, pd.MultiIndex):
                df_t.columns = [col[0] for col in df_t.columns]
            prices_dict[t] = df_t["Close"]

    if len(prices_dict) < 2:
        return {"error": "Insufficient price data", "tickers": tickers}

    df_prices = pd.DataFrame(prices_dict).dropna()
    daily_returns = df_prices.pct_change().dropna()
    log_returns = np.log(df_prices / df_prices.shift(1)).dropna()

    expected_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252
    n = len(tickers)

    weights_ew = calculate_equal_weights(n)
    weights_rp = calculate_risk_parity_weights(cov_matrix.values)
    weights_mvo = calculate_mvo_weights(expected_returns.values, cov_matrix.values)
    hrp_dict = calculate_hrp_weights(log_returns)
    weights_hrp = np.array([hrp_dict.get(t, 0.0) for t in tickers])

    def _to_dict(w):
        return {t: _sf(w[i]) for i, t in enumerate(tickers)}

    weights_out = {
        "EW": _to_dict(weights_ew),
        "RP": _to_dict(weights_rp),
        "MVO": _to_dict(weights_mvo),
        "HRP": _to_dict(weights_hrp),
    }

    cov_arr = cov_matrix.values
    perf_metrics = []
    for method_name, w in [
        ("Equal Weight (EW)", weights_ew),
        ("Risk Parity (RP)", weights_rp),
        ("Mean-Variance (MVO)", weights_mvo),
        ("Hierarchical RP (HRP)", weights_hrp),
    ]:
        port_ret = float(np.dot(w, expected_returns.values))
        port_vol = float(np.sqrt(w.T @ cov_arr @ w))
        port_sharpe = port_ret / port_vol if port_vol > 0 else 0.0
        perf_metrics.append({
            "method": method_name,
            "expected_annual_return_pct": _sf(port_ret * 100),
            "expected_annual_vol_pct": _sf(port_vol * 100),
            "sharpe_ratio": _sf(port_sharpe),
        })

    return {
        "tickers": tickers,
        "weights": weights_out,
        "perf_metrics": perf_metrics,
    }


if __name__ == "__main__":
    main()
