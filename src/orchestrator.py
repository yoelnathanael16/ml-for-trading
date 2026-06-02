"""
QuantOrchestrator — central lifecycle manager for the quant trading system.

Manages scheduled data refresh (daily) and model retraining (weekly) via
APScheduler. Exposes an inference cache for real-time API queries.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler

from src.data_ingestion import fetch_stock_data
from src.features.technical_indicators import add_technical_indicators
from src.features.preprocessing import (
    triple_barrier_labeling,
    prepare_features_and_labels,
    walk_forward_split,
)
from src.models.anomaly_detector import MarketAnomalyDetector
from src.models.explainability import ModelExplainer
from src.models.mean_reversion import MeanReversionDetector
from src.models.market_regime import MarketRegimeDetector
from src.models.regime_hmm import HMMRegimeDetector
from src.models.risk_model import TailRiskModel
from src.models.volatility_garch import GARCHVolatilityModel
from src.models.position_sizing import calculate_position_sizes
from src.train_benchmark import run_full_pipeline
from src.training.hyperparameter_tuner import tune_all_models

logger = logging.getLogger(__name__)

_ML_MODELS = ["SVM", "RandomForest", "XGBoost", "LightGBM"]


class QuantOrchestrator:
    """Central lifecycle manager for the quant trading system.

    Handles:
    - Scheduled daily data refresh (9:30 AM ET)
    - Scheduled weekly full model retrain (Monday 7:00 AM ET)
    - Thread-safe inference cache per ticker
    """

    def __init__(
        self,
        tickers: list[str],
        data_dir: str = "data",
        models_dir: str = "models",
    ) -> None:
        self.tickers = tickers
        self.data_dir = data_dir
        self.models_dir = models_dir

        self._cache: dict[str, dict] = {}
        self._last_refresh: dict[str, Optional[datetime]] = {t: None for t in tickers}
        self._last_retrain: dict[str, Optional[datetime]] = {t: None for t in tickers}
        self._lock = threading.Lock()
        self._scheduler = BackgroundScheduler()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background scheduler.

        Schedules:
        - daily_refresh at 9:30 AM ET every day
        - weekly_retrain at 7:00 AM ET every Monday

        On startup, any ticker missing trained models gets an immediate retrain.
        """
        # Pre-startup check: retrain any ticker that has no saved models
        for ticker in self.tickers:
            if not self._models_exist(ticker):
                logger.info(
                    "Models not found for %s — triggering initial weekly_retrain.", ticker
                )
                try:
                    self.weekly_retrain(ticker)
                except Exception:
                    logger.exception("Initial retrain failed for %s", ticker)

        # Schedule daily refresh at 09:30 ET
        self._scheduler.add_job(
            func=self._refresh_all,
            trigger="cron",
            hour=9,
            minute=30,
            timezone="America/New_York",
            id="daily_refresh",
            replace_existing=True,
        )

        # Schedule weekly retrain on Mondays at 07:00 ET
        self._scheduler.add_job(
            func=self._retrain_all,
            trigger="cron",
            day_of_week="mon",
            hour=7,
            timezone="America/New_York",
            id="weekly_retrain",
            replace_existing=True,
        )

        self._scheduler.start()
        logger.info(
            "QuantOrchestrator started with %d tickers: %s", len(self.tickers), self.tickers
        )

    def stop(self) -> None:
        """Shutdown scheduler gracefully."""
        self._scheduler.shutdown(wait=False)
        logger.info("QuantOrchestrator scheduler shut down.")

    # ------------------------------------------------------------------
    # Scheduled helpers
    # ------------------------------------------------------------------

    def _refresh_all(self) -> None:
        for ticker in self.tickers:
            try:
                self.daily_refresh(ticker)
            except Exception:
                logger.exception("daily_refresh failed for %s", ticker)

    def _retrain_all(self) -> None:
        for ticker in self.tickers:
            try:
                self.weekly_retrain(ticker)
            except Exception:
                logger.exception("weekly_retrain failed for %s", ticker)

    # ------------------------------------------------------------------
    # Daily refresh
    # ------------------------------------------------------------------

    def daily_refresh(self, ticker: str) -> None:
        """Refresh inference cache for a ticker using the latest 90 days of data.

        Does NOT retrain models — only runs inference with existing saved models.
        """
        logger.info("daily_refresh starting for %s", ticker)

        # 1. Fetch latest 90 days of OHLCV via yfinance
        end_str = datetime.now().strftime("%Y-%m-%d")
        start_str = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        try:
            df_raw = yf.download(ticker, start=start_str, end=end_str, progress=False)
            if df_raw.empty:
                logger.warning("No data returned from yfinance for %s in daily_refresh", ticker)
                return

            # Flatten MultiIndex columns if present (yfinance >=0.2 style)
            if isinstance(df_raw.columns, pd.MultiIndex):
                df_raw.columns = [col[0] for col in df_raw.columns]

            # Save latest data
            raw_dir = os.path.join(self.data_dir, "raw")
            os.makedirs(raw_dir, exist_ok=True)
            latest_path = os.path.join(raw_dir, f"{ticker}_latest.parquet")
            df_raw.to_parquet(latest_path)
        except Exception:
            logger.exception("Failed to fetch/save latest data for %s", ticker)
            return

        # 2. Add technical indicators
        try:
            df = add_technical_indicators(df_raw)
        except Exception:
            logger.exception("add_technical_indicators failed for %s", ticker)
            return

        # 3. Extract latest feature row (last row after dropping NaN)
        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
        feature_cols = [c for c in df.columns if c not in ohlcv_cols]
        df_features = df[feature_cols].dropna()
        if df_features.empty:
            logger.warning("No valid feature rows for %s after dropna", ticker)
            return

        latest_features = df_features.iloc[[-1]]  # shape (1, n_features)

        # 4. Scale with saved scaler if available
        X_scaled_row = None
        scaler_path = os.path.join(self.models_dir, f"scaler_{ticker}.joblib")
        if os.path.exists(scaler_path):
            try:
                scaler = joblib.load(scaler_path)
                scaled_arr = scaler.transform(latest_features)
                X_scaled_row = pd.DataFrame(
                    scaled_arr, columns=latest_features.columns, index=latest_features.index
                )
            except Exception:
                logger.exception("Failed to load/apply scaler for %s", ticker)
        else:
            logger.warning("No scaler found for %s; using unscaled features", ticker)
            X_scaled_row = latest_features

        # 5. Anomaly check
        anomaly_flag = None
        anomaly_score = None
        anomaly_path = os.path.join(self.models_dir, f"anomaly_detector_{ticker}.joblib")
        if os.path.exists(anomaly_path):
            try:
                anomaly_detector: MarketAnomalyDetector = joblib.load(anomaly_path)
                anomaly_flag = anomaly_detector.is_anomaly(X_scaled_row.values)
                anomaly_score = float(
                    anomaly_detector.score_samples(X_scaled_row.values)[0]
                )
            except Exception:
                logger.exception("Anomaly detection failed for %s", ticker)

        # 6. Run inference on loaded models
        loaded = self._load_cached_models(ticker)

        # 7 & 8. Build and store inference bundle
        bundle = self._build_inference_bundle(
            ticker=ticker,
            df=df,
            X_scaled=X_scaled_row,
            loaded_models=loaded,
            anomaly_flag=anomaly_flag,
            anomaly_score=anomaly_score,
        )

        with self._lock:
            self._cache[ticker] = bundle
            self._last_refresh[ticker] = datetime.now()

        logger.info("daily_refresh complete for %s", ticker)

    # ------------------------------------------------------------------
    # Weekly retrain
    # ------------------------------------------------------------------

    def weekly_retrain(self, ticker: str) -> None:
        """Full retrain + fit all supplementary models for a ticker."""
        logger.info("weekly_retrain starting for %s", ticker)

        start_date = "2020-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")

        # 1. Fetch fresh data using fetch_stock_data
        raw_dir = os.path.join(self.data_dir, "raw")
        data_path: Optional[str] = None
        try:
            data_path = fetch_stock_data(
                ticker, start_date, end_date, raw_dir, allow_synthetic_fallback=False
            )
        except Exception:
            logger.exception("fetch_stock_data failed for %s", ticker)

        # Fall back to an existing parquet if fetch failed
        if data_path is None or not os.path.exists(data_path):
            candidate = os.path.join(raw_dir, f"{ticker}_{start_date}_{end_date}.parquet")
            if os.path.exists(candidate):
                logger.warning(
                    "Using existing parquet fallback for %s: %s", ticker, candidate
                )
                data_path = candidate
            else:
                # Try any existing parquet for this ticker
                if os.path.exists(raw_dir):
                    candidates = [
                        f for f in os.listdir(raw_dir)
                        if f.startswith(ticker) and f.endswith(".parquet")
                    ]
                    if candidates:
                        data_path = os.path.join(raw_dir, sorted(candidates)[-1])
                        logger.warning(
                            "No fresh data; using last available parquet for %s: %s",
                            ticker, data_path,
                        )
                    else:
                        logger.error(
                            "No data available for %s — aborting weekly_retrain", ticker
                        )
                        return
                else:
                    logger.error(
                        "Raw data directory does not exist; aborting weekly_retrain for %s", ticker
                    )
                    return

        # 2. Load data & build features for hyperparameter tuning
        try:
            df = pd.read_parquet(data_path)
            df = add_technical_indicators(df)
            df["Label"] = triple_barrier_labeling(df)
            X, y, _ = prepare_features_and_labels(df)
            X_train, _X_test, y_train, _y_test = walk_forward_split(X, y)
        except Exception:
            logger.exception("Feature prep failed for %s during weekly_retrain", ticker)
            return

        # 3. Hyperparameter tuning
        try:
            _best_params = tune_all_models(X_train.values, y_train.values)
            logger.info("Hyperparameter tuning complete for %s: %s", ticker, _best_params)
        except Exception:
            logger.exception("Hyperparameter tuning failed for %s (non-fatal)", ticker)

        # 4. Run full pipeline (trains & saves ML models + scaler + GMM regime detector)
        pipeline_result = None
        try:
            pipeline_result = run_full_pipeline(
                ticker, start_date, end_date, self.data_dir, self.models_dir
            )
        except Exception:
            logger.exception("run_full_pipeline failed for %s", ticker)

        # 5. Fit supplementary models
        os.makedirs(self.models_dir, exist_ok=True)

        # Reload the full df in case pipeline_result has issues
        try:
            df_full = pd.read_parquet(data_path)
            df_full = add_technical_indicators(df_full)
            df_full["Label"] = triple_barrier_labeling(df_full)
            X_all, _y_all, scaler_all = prepare_features_and_labels(df_full)
            X_all_scaled = X_all  # already scaled by prepare_features_and_labels

            log_returns = np.log(df_full["Close"] / df_full["Close"].shift(1)).dropna()
            regime_features = df_full[["Log_Returns", "Volatility"]].dropna()
        except Exception:
            logger.exception("Data reload for supplementary models failed for %s", ticker)
            return

        # 5a. HMM Regime Detector
        try:
            hmm = HMMRegimeDetector()
            hmm.fit(regime_features.values)
            joblib.dump(hmm, os.path.join(self.models_dir, f"hmm_regime_{ticker}.joblib"))
            logger.info("HMM regime detector saved for %s", ticker)
        except Exception:
            logger.exception("HMMRegimeDetector fit/save failed for %s", ticker)

        # 5b. GARCH Volatility Model
        try:
            garch = GARCHVolatilityModel()
            garch.fit(log_returns)
            joblib.dump(garch, os.path.join(self.models_dir, f"garch_{ticker}.joblib"))
            logger.info("GARCH model saved for %s", ticker)
        except Exception:
            logger.exception("GARCHVolatilityModel fit/save failed for %s", ticker)

        # 5c. Market Anomaly Detector
        try:
            anomaly_det = MarketAnomalyDetector()
            anomaly_det.fit(X_all_scaled.values)
            joblib.dump(
                anomaly_det,
                os.path.join(self.models_dir, f"anomaly_detector_{ticker}.joblib"),
            )
            logger.info("Anomaly detector saved for %s", ticker)
        except Exception:
            logger.exception("MarketAnomalyDetector fit/save failed for %s", ticker)

        # 5d. Mean Reversion Detector
        try:
            mr = MeanReversionDetector()
            mr.fit(df_full["Close"])
            joblib.dump(
                mr, os.path.join(self.models_dir, f"mean_reversion_{ticker}.joblib")
            )
            logger.info("MeanReversionDetector saved for %s", ticker)
        except Exception:
            logger.exception("MeanReversionDetector fit/save failed for %s", ticker)

        # 5e. Tail Risk Model
        try:
            risk = TailRiskModel()
            risk.fit(log_returns)
            joblib.dump(risk, os.path.join(self.models_dir, f"risk_model_{ticker}.joblib"))
            logger.info("TailRiskModel saved for %s", ticker)
        except Exception:
            logger.exception("TailRiskModel fit/save failed for %s", ticker)

        # 5f. ModelExplainer (SHAP) for each ML model
        for model_name in _ML_MODELS:
            try:
                ml_model_path = os.path.join(
                    self.models_dir, f"{model_name}_{ticker}.joblib"
                )
                if not os.path.exists(ml_model_path):
                    logger.warning(
                        "ML model artifact not found, skipping explainer for %s/%s",
                        model_name, ticker,
                    )
                    continue
                raw_model = joblib.load(ml_model_path)
                explainer = ModelExplainer(
                    model_name=model_name,
                    model=raw_model,
                    feature_names=list(X_all_scaled.columns),
                )
                # Pre-compute importances on training data (stores explainer state)
                explainer.get_feature_importance(X_all_scaled.values)
                joblib.dump(
                    explainer,
                    os.path.join(
                        self.models_dir, f"explainer_{model_name}_{ticker}.joblib"
                    ),
                )
                logger.info("ModelExplainer saved for %s/%s", model_name, ticker)
            except Exception:
                logger.exception(
                    "ModelExplainer fit/save failed for %s/%s", model_name, ticker
                )

        # 6. Build and cache full inference bundle from newly trained models
        try:
            loaded = self._load_cached_models(ticker)
            X_latest = X_all_scaled.iloc[[-1]]

            anomaly_flag_retrain = None
            anomaly_score_retrain = None
            if "anomaly_detector" in loaded and loaded["anomaly_detector"] is not None:
                try:
                    anomaly_flag_retrain = loaded["anomaly_detector"].is_anomaly(
                        X_latest.values
                    )
                    anomaly_score_retrain = float(
                        loaded["anomaly_detector"].score_samples(X_latest.values)[0]
                    )
                except Exception:
                    logger.exception("Anomaly inference failed during retrain cache build for %s", ticker)

            bundle = self._build_inference_bundle(
                ticker=ticker,
                df=df_full,
                X_scaled=X_latest,
                loaded_models=loaded,
                anomaly_flag=anomaly_flag_retrain,
                anomaly_score=anomaly_score_retrain,
            )
            with self._lock:
                self._cache[ticker] = bundle
                self._last_retrain[ticker] = datetime.now()
        except Exception:
            logger.exception("Cache bundle build failed after retrain for %s", ticker)

        logger.info("weekly_retrain complete for %s", ticker)

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def infer(self, ticker: str) -> dict:
        """Return cached inference bundle for ticker.

        If cache is empty or stale (>24 h), triggers daily_refresh first.
        """
        with self._lock:
            cached = self._cache.get(ticker)
            last_refresh = self._last_refresh.get(ticker)

        stale = (
            cached is None
            or last_refresh is None
            or (datetime.now() - last_refresh) > timedelta(hours=24)
        )
        if stale:
            try:
                self.daily_refresh(ticker)
            except Exception:
                logger.exception("daily_refresh triggered by infer() failed for %s", ticker)

        with self._lock:
            return dict(self._cache.get(ticker, {}))

    def get_status(self) -> dict:
        """Return status dict for all tickers."""
        status = {}
        with self._lock:
            for ticker in self.tickers:
                cached = self._cache.get(ticker, {})
                last_refresh = self._last_refresh.get(ticker)
                last_retrain = self._last_retrain.get(ticker)
                status[ticker] = {
                    "last_refresh": last_refresh.isoformat() if last_refresh else None,
                    "last_retrain": last_retrain.isoformat() if last_retrain else None,
                    "anomaly_flag": cached.get("anomaly_flag", None),
                    "scheduler_running": self._scheduler.running,
                }
        return status

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _models_exist(self, ticker: str) -> bool:
        """Return True if the scaler artifact exists for ticker."""
        scaler_path = os.path.join(self.models_dir, f"scaler_{ticker}.joblib")
        return os.path.exists(scaler_path)

    def _load_cached_models(self, ticker: str) -> dict:
        """Load all available joblib artifacts for ticker into a dict."""
        loaded: dict = {}

        def _try_load(key: str, filename: str):
            path = os.path.join(self.models_dir, filename)
            if os.path.exists(path):
                try:
                    loaded[key] = joblib.load(path)
                except Exception:
                    logger.exception("Failed to load artifact %s", path)
                    loaded[key] = None
            else:
                loaded[key] = None

        _try_load("scaler", f"scaler_{ticker}.joblib")
        _try_load("regime_detector", f"regime_detector_{ticker}.joblib")
        _try_load("hmm_regime", f"hmm_regime_{ticker}.joblib")
        _try_load("garch", f"garch_{ticker}.joblib")
        _try_load("anomaly_detector", f"anomaly_detector_{ticker}.joblib")
        _try_load("mean_reversion", f"mean_reversion_{ticker}.joblib")
        _try_load("risk_model", f"risk_model_{ticker}.joblib")

        for model_name in _ML_MODELS:
            _try_load(model_name, f"{model_name}_{ticker}.joblib")
            _try_load(f"explainer_{model_name}", f"explainer_{model_name}_{ticker}.joblib")

        return loaded

    def _build_inference_bundle(
        self,
        ticker: str,
        df: pd.DataFrame,
        X_scaled: pd.DataFrame,
        loaded_models: Optional[dict] = None,
        anomaly_flag: Optional[bool] = None,
        anomaly_score: Optional[float] = None,
    ) -> dict:
        """Build the full inference bundle from loaded models and data.

        Any individual model inference failure is caught and logged.
        """
        if loaded_models is None:
            loaded_models = self._load_cached_models(ticker)

        bundle: dict = {}

        # ── ML model predictions ──────────────────────────────────────────
        predictions: dict[str, Optional[int]] = {}
        position_sizes: dict[str, Optional[float]] = {}

        for model_name in _ML_MODELS:
            raw_model = loaded_models.get(model_name)
            if raw_model is None:
                predictions[model_name] = None
                position_sizes[model_name] = None
                continue
            try:
                pred = int(raw_model.predict(X_scaled.values)[0])
                predictions[model_name] = pred

                # Position sizing via Kelly using predict_proba if available
                try:
                    proba = raw_model.predict_proba(X_scaled.values)  # (1, n_classes)
                    sizes = calculate_position_sizes(
                        signals=np.array([pred]),
                        probabilities=proba,
                        method="kelly",
                    )
                    position_sizes[model_name] = float(sizes[0])
                except Exception:
                    logger.warning(
                        "predict_proba not available for %s/%s; using constant sizing",
                        model_name, ticker,
                    )
                    sizes = calculate_position_sizes(
                        signals=np.array([pred]),
                        method="constant",
                    )
                    position_sizes[model_name] = float(sizes[0])
            except Exception:
                logger.exception("Inference failed for %s/%s", model_name, ticker)
                predictions[model_name] = None
                position_sizes[model_name] = None

        bundle["predictions"] = predictions
        bundle["position_sizes"] = position_sizes

        # ── GMM Regime ────────────────────────────────────────────────────
        bundle["regime_gmm"] = None
        regime_det = loaded_models.get("regime_detector")
        if regime_det is not None:
            try:
                regime_features = df[["Log_Returns", "Volatility"]].dropna()
                if not regime_features.empty:
                    latest_regime_row = regime_features.iloc[[-1]].values
                    bundle["regime_gmm"] = regime_det.predict_regime_name(latest_regime_row)[0]
            except Exception:
                logger.exception("GMM regime prediction failed for %s", ticker)

        # ── HMM Regime ────────────────────────────────────────────────────
        bundle["regime_hmm"] = None
        bundle["hmm_transition_matrix"] = None
        hmm_det = loaded_models.get("hmm_regime")
        if hmm_det is not None:
            try:
                regime_features_vals = df[["Log_Returns", "Volatility"]].dropna().values
                if len(regime_features_vals) > 0:
                    preds = hmm_det.predict(regime_features_vals)
                    state_names = hmm_det.get_regime_names()
                    bundle["regime_hmm"] = state_names[int(preds[-1])]
                    bundle["hmm_transition_matrix"] = (
                        hmm_det.get_transition_matrix().tolist()
                    )
            except Exception:
                logger.exception("HMM regime prediction failed for %s", ticker)

        # ── GARCH Volatility ──────────────────────────────────────────────
        bundle["garch_vol"] = None
        garch_model = loaded_models.get("garch")
        if garch_model is not None:
            try:
                bundle["garch_vol"] = garch_model.forecast(horizon=1)
            except Exception:
                logger.exception("GARCH forecast failed for %s", ticker)

        # Rolling 20-day vol from the df
        bundle["rolling_vol"] = None
        try:
            if "Volatility" in df.columns:
                vol_series = df["Volatility"].dropna()
                if not vol_series.empty:
                    bundle["rolling_vol"] = float(vol_series.iloc[-1])
        except Exception:
            logger.exception("Rolling vol extraction failed for %s", ticker)

        # ── Tail Risk ─────────────────────────────────────────────────────
        bundle["cvar_95"] = None
        bundle["cvar_99"] = None
        bundle["var_95"] = None
        bundle["var_99"] = None
        bundle["position_scale"] = None
        risk_model = loaded_models.get("risk_model")
        if risk_model is not None:
            try:
                risk_result = risk_model.compute()
                bundle["cvar_95"] = risk_result.get("cvar_95")
                bundle["cvar_99"] = risk_result.get("cvar_99")
                bundle["var_95"] = risk_result.get("var_95")
                bundle["var_99"] = risk_result.get("var_99")
                bundle["position_scale"] = risk_result.get("position_scale")
            except Exception:
                logger.exception("TailRiskModel compute failed for %s", ticker)

        # ── Mean Reversion ────────────────────────────────────────────────
        bundle["zscore"] = None
        bundle["halflife"] = None
        bundle["mr_signal"] = None
        bundle["is_mean_reverting"] = None
        mr_model = loaded_models.get("mean_reversion")
        if mr_model is not None:
            try:
                # MeanReversionDetector.predict() refits on the provided prices
                if "Close" in df.columns and not df["Close"].dropna().empty:
                    mr_result = mr_model.predict(df["Close"].dropna())
                    bundle["zscore"] = mr_result.get("zscore")
                    bundle["halflife"] = mr_result.get("halflife")
                    bundle["mr_signal"] = mr_result.get("signal")
                    bundle["is_mean_reverting"] = mr_result.get("is_mean_reverting")
            except Exception:
                logger.exception("MeanReversionDetector predict failed for %s", ticker)

        # ── Anomaly ───────────────────────────────────────────────────────
        bundle["anomaly_flag"] = anomaly_flag
        bundle["anomaly_score"] = anomaly_score

        # ── SHAP importances ──────────────────────────────────────────────
        shap_importances: dict[str, Optional[dict]] = {}
        for model_name in _ML_MODELS:
            explainer = loaded_models.get(f"explainer_{model_name}")
            if explainer is not None:
                try:
                    shap_importances[model_name] = explainer.get_feature_importance(
                        X_scaled.values
                    )
                except Exception:
                    logger.exception(
                        "SHAP importance failed for %s/%s", model_name, ticker
                    )
                    shap_importances[model_name] = None
            else:
                shap_importances[model_name] = None
        bundle["shap_importances"] = shap_importances

        # ── Timestamp ─────────────────────────────────────────────────────
        bundle["last_updated"] = datetime.now().isoformat()

        return bundle
