import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PARAM_GRIDS = {
    "SVM": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]},
    "RandomForest": {"n_estimators": [100, 200], "max_depth": [4, 6, 8]},
    "XGBoost": {"learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 4, 5], "n_estimators": [100, 120, 150]},
    "LightGBM": {"learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 4, 5], "n_estimators": [100, 120, 150]},
}

MODEL_FACTORIES = {
    "SVM": lambda: SVC(probability=True, random_state=42),
    "RandomForest": lambda: RandomForestClassifier(random_state=42),
    "XGBoost": lambda: XGBClassifier(
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
        label_encoder=False,
    ),
    "LightGBM": lambda: LGBMClassifier(random_state=42, verbosity=-1),
}


def tune_model(model_name: str, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """
    Walk-forward hyperparameter tuning for a single model.

    Uses TimeSeriesSplit + GridSearchCV.
    Scoring: "accuracy"

    Args:
        model_name: one of "SVM", "RandomForest", "XGBoost", "LightGBM"
        X: feature matrix (already scaled)
        y: labels
        n_splits: number of time-series CV splits

    Returns:
        best_params: dict of best hyperparameters
    """
    print(f"Tuning {model_name}...")

    if model_name not in PARAM_GRIDS:
        raise KeyError(f"Unknown model name: {model_name!r}. Must be one of {list(PARAM_GRIDS)}")

    estimator = MODEL_FACTORIES[model_name]()
    param_grid = PARAM_GRIDS[model_name]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # XGBoost requires labels encoded as 0-based integers when using
    # multi-class objectives.  Map {-1, 0, 1} → {0, 1, 2} for the search.
    y_fit = y
    if model_name == "XGBoost":
        labels = np.sort(np.unique(y))
        label_map = {label: idx for idx, label in enumerate(labels)}
        y_fit = np.vectorize(label_map.__getitem__)(y)

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=tscv,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y_fit)
    return search.best_params_


def tune_all_models(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict[str, dict]:
    """
    Tune all models in PARAM_GRIDS.

    Returns:
        {model_name: best_params_dict}
    """
    results: dict[str, dict] = {}
    for model_name in PARAM_GRIDS:
        try:
            best_params = tune_model(model_name, X, y, n_splits=n_splits)
            results[model_name] = best_params
        except KeyError as exc:
            print(f"Skipping {model_name}: {exc}")
    return results
