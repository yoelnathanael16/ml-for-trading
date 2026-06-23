import logging
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class ModelWrapper:
    """Small adapter that gives each benchmark model the same train/predict API."""

    # ponytail: single source of truth for XGBoost {-1,0,1}↔{0,1,2} remap
    XGB_LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
    XGB_CLASS_TO_LABEL = {0: -1, 1: 0, 2: 1}

    def __init__(self, model_name, random_state=42):
        self.model_name = model_name
        self.random_state = random_state
        self.model = self._build_model(model_name)
        self._label_to_class = None
        self._class_to_label = None

    def _build_model(self, model_name):
        if model_name == "SVM":
            return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=self.random_state)
        if model_name == "RandomForest":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=5,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
        if model_name == "XGBoost":
            return XGBClassifier(
                n_estimators=120,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softmax",
                eval_metric="mlogloss",
                random_state=self.random_state,
                n_jobs=-1,
            )
        if model_name == "LightGBM":
            return LGBMClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1,
            )
        raise ValueError(f"Unsupported model: {model_name}")

    def train(self, X_train, y_train):
        y_fit = y_train
        if self.model_name == "XGBoost":
            labels = sorted(pd.Series(y_train).dropna().unique())
            self._label_to_class = {label: idx for idx, label in enumerate(labels)}
            self._class_to_label = {idx: label for label, idx in self._label_to_class.items()}
            y_fit = pd.Series(y_train).map(self._label_to_class)

        self.model.fit(X_train, y_fit)
        return self

    def predict(self, X_test):
        preds = self.model.predict(X_test)
        if self._class_to_label:
            preds = pd.Series(preds).map(self._class_to_label).to_numpy()
        return preds

    def predict_proba(self, X_test):
        if not hasattr(self.model, "predict_proba"):
            logger.warning(
                "%s has no predict_proba — returning uniform priors; Kelly sizing will be unweighted",
                self.model_name,
            )
            return np.ones((len(X_test), 3)) / 3.0
            
        probas_raw = self.model.predict_proba(X_test)
        n_samples = len(X_test)
        aligned_probas = np.zeros((n_samples, 3))
        
        # Target classes are [-1, 0, 1]
        target_classes = [-1, 0, 1]
        
        if self.model_name == "XGBoost" and self._class_to_label:
            model_classes = [self._class_to_label[i] for i in range(len(self.model.classes_))]
        else:
            model_classes = self.model.classes_
            
        for i, cls in enumerate(model_classes):
            if cls in target_classes:
                target_idx = target_classes.index(cls)
                aligned_probas[:, target_idx] = probas_raw[:, i]
                
        return aligned_probas


def calculate_financial_metrics(y_true, predictions, prices):
    """Calculate simple long/short strategy metrics from predicted signals."""
    prices = pd.Series(prices).astype(float).dropna()
    predictions = pd.Series(predictions, index=prices.index[: len(predictions)]).astype(float)

    aligned = pd.concat([prices.rename("price"), predictions.rename("signal")], axis=1).dropna()
    if len(aligned) < 2:
        return {"Total Return": 0.0, "Sharpe Ratio": 0.0, "Max Drawdown": 0.0}

    returns = aligned["price"].pct_change().fillna(0.0)
    strategy_returns = aligned["signal"].shift(1).fillna(0.0) * returns

    equity_curve = (1.0 + strategy_returns).cumprod()
    total_return = equity_curve.iloc[-1] - 1.0

    volatility = strategy_returns.std()
    sharpe_ratio = 0.0
    if volatility and not np.isnan(volatility):
        sharpe_ratio = (strategy_returns.mean() / volatility) * np.sqrt(252)

    drawdown = equity_curve / equity_curve.cummax() - 1.0
    max_drawdown = drawdown.min()

    return {
        "Total Return": float(total_return),
        "Sharpe Ratio": float(sharpe_ratio),
        "Max Drawdown": float(max_drawdown),
    }


def run_arima_benchmark(series, train_size, order=(1, 0, 1)):
    """Generate one-step-ahead ARIMA forecasts for the test portion of a series."""
    values = np.asarray(series, dtype=float)
    forecasts = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # ponytail: fit once, then append(refit=False) — avoids O(n²) full refit per step
            fitted = ARIMA(values[:train_size], order=order).fit()
            for actual in values[train_size:]:
                forecasts.append(float(fitted.forecast(steps=1)[0]))
                fitted = fitted.append([actual], refit=False)
    except Exception as exc:
        logger.warning("ARIMA fit failed: %s — zero forecasts", exc)
        forecasts = [0.0] * (len(values) - train_size)
    return np.asarray(forecasts)
