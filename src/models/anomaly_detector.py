import numpy as np
from sklearn.ensemble import IsolationForest


class MarketAnomalyDetector:
    """Unsupervised anomaly detector for market feature vectors.

    Wraps sklearn's IsolationForest to flag unusual market conditions.
    IsolationForest convention: predict() returns 1 for normal, -1 for anomaly.
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model: IsolationForest | None = None

    def fit(self, X: np.ndarray) -> None:
        """Fit the isolation forest on a 2-D feature array.

        Parameters
        ----------
        X:
            2-D array of shape (n_samples, n_features) — same technical
            indicators used by the ML benchmark models.
        """
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42,
        )
        self.model.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample labels using sklearn's IsolationForest convention.

        Returns
        -------
        np.ndarray
            Array of int values: 1 = normal, -1 = anomaly.
        """
        if self.model is None:
            raise RuntimeError("Detector has not been fitted yet. Call fit() first.")
        return self.model.predict(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores for each sample.

        Lower scores indicate more anomalous observations.

        Returns
        -------
        np.ndarray
            Array of float anomaly scores with shape (n_samples,).
        """
        if self.model is None:
            raise RuntimeError("Detector has not been fitted yet. Call fit() first.")
        return self.model.score_samples(X)

    def is_anomaly(self, X: np.ndarray) -> bool:
        """Check whether a single feature row is flagged as an anomaly.

        Parameters
        ----------
        X:
            A single sample as shape (n_features,) or (1, n_features).
            The array is reshaped to (1, n_features) before prediction.

        Returns
        -------
        bool
            True if the sample is flagged as an anomaly (-1), False otherwise.
        """
        if self.model is None:
            raise RuntimeError("Detector has not been fitted yet. Call fit() first.")
        sample = np.asarray(X).reshape(1, -1)
        return bool(self.model.predict(sample)[0] == -1)
