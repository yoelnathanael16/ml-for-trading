import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBClassifier


class ModelWrapper:
    """Small adapter that gives each benchmark model the same train/predict API."""

    def __init__(self, model_name, random_state=42):
        self.model_name = model_name
        self.random_state = random_state
        self.model = self._build_model(model_name)
        self._label_to_class = None
        self._class_to_label = None

    def _build_model(self, model_name):
        if model_name == "SVM":
            return SVC(kernel="rbf", C=1.0, gamma="scale", random_state=self.random_state)
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
    history = list(values[:train_size])
    forecasts = []

    for actual in values[train_size:]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(history, order=order)
                fitted = model.fit()
                forecast = fitted.forecast(steps=1)[0]
        except Exception:
            forecast = history[-1] if history else 0.0

        forecasts.append(float(forecast))
        history.append(float(actual))

    return np.asarray(forecasts)
