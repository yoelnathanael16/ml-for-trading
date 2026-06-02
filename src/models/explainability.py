import numpy as np
import shap


class ModelExplainer:
    """SHAP-based explainability wrapper for SVM, RandomForest, XGBoost, and LightGBM."""

    # Tree-based models that support TreeExplainer
    _TREE_MODELS = {"RandomForest", "XGBoost", "LightGBM"}

    def __init__(self, model_name: str, model, feature_names: list[str]):
        """
        Parameters
        ----------
        model_name : str
            One of "SVM", "RandomForest", "XGBoost", "LightGBM".
        model : fitted sklearn/xgboost/lightgbm model
            Must already be trained before being passed here.
        feature_names : list[str]
            Ordered list of feature column names matching the columns of X.
        """
        self.model_name = model_name
        self.model = model
        self.feature_names = feature_names

        # Will be set lazily on first call (KernelExplainer needs background data)
        self._explainer = None
        # Background data stored for KernelExplainer (SVM)
        self._background = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_tree_explainer(self) -> shap.TreeExplainer:
        return shap.TreeExplainer(self.model)

    def _build_kernel_explainer(self, X: np.ndarray) -> shap.KernelExplainer:
        """Build a KernelExplainer using a small kmeans background from X."""
        n_background = min(50, len(X))
        background = shap.kmeans(X[:n_background], min(10, n_background))
        return shap.KernelExplainer(self.model.predict_proba, background)

    def _get_explainer(self, X: np.ndarray):
        """Return (and cache) the appropriate SHAP explainer."""
        if self._explainer is None:
            if self.model_name in self._TREE_MODELS:
                self._explainer = self._build_tree_explainer()
            else:
                # SVM / fallback — needs background data
                self._background = X
                self._explainer = self._build_kernel_explainer(X)
        return self._explainer

    @staticmethod
    def _normalize_shap_values(sv) -> np.ndarray:
        """
        Normalise raw SHAP output to a 2-D array of shape (n_samples, n_features).

        TreeExplainer may return:
          - a list of arrays  [array(n_samples, n_features), ...]  — one per class
          - a 3-D array       (n_samples, n_features, n_classes)
          - a 2-D array       (n_samples, n_features)              — binary case

        KernelExplainer with predict_proba returns the same variants.
        We average the absolute values over classes to get a single matrix.
        """
        if isinstance(sv, list):
            # Older shap: list of (n_samples, n_features) arrays, one per class
            shap_matrix = np.mean(np.abs(np.array(sv)), axis=0)
        elif isinstance(sv, np.ndarray):
            if sv.ndim == 3:
                # New shap: (n_samples, n_features, n_classes)
                shap_matrix = np.abs(sv).mean(axis=-1)
            else:
                # Binary or already 2-D
                shap_matrix = np.abs(sv)
        else:
            raise TypeError(f"Unexpected SHAP values type: {type(sv)}")
        return shap_matrix  # shape: (n_samples, n_features)

    @staticmethod
    def _normalize_shap_values_signed(sv, sample_idx: int = -1) -> np.ndarray:
        """
        Return *signed* SHAP values for a single sample (1-D array, n_features).

        For multiclass output we average the signed values over classes, which
        preserves the sign of the dominant direction.
        """
        if isinstance(sv, list):
            # list of (1, n_features) — one per class
            arr = np.array(sv)           # (n_classes, 1, n_features)
            row = arr[:, 0, :]           # (n_classes, n_features)
            return row.mean(axis=0)      # (n_features,)
        elif isinstance(sv, np.ndarray):
            if sv.ndim == 3:
                # (1, n_features, n_classes)
                return sv[0].mean(axis=-1)   # (n_features,)
            elif sv.ndim == 2:
                return sv[0]                 # (n_features,)
            else:
                return sv                    # already 1-D
        else:
            raise TypeError(f"Unexpected SHAP values type: {type(sv)}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for X.

        Uses TreeExplainer for tree-based models (RF, XGB, LGBM) and
        KernelExplainer for SVM.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Mean absolute SHAP values averaged over classes for multiclass.
        """
        X = np.asarray(X)
        explainer = self._get_explainer(X)

        if self.model_name in self._TREE_MODELS:
            sv = explainer.shap_values(X)
        else:
            sv = explainer.shap_values(X, nsamples=100)

        return self._normalize_shap_values(sv)

    def get_feature_importance(self, X: np.ndarray) -> dict[str, float]:
        """
        Compute mean absolute SHAP importance for each feature.

        Returns
        -------
        dict[str, float]
            Mapping of feature name → mean |SHAP| value, sorted descending.
        """
        shap_matrix = self.compute_shap_values(X)  # (n_samples, n_features)
        mean_abs = shap_matrix.mean(axis=0)         # (n_features,)

        importance = {
            name: float(val)
            for name, val in zip(self.feature_names, mean_abs)
        }
        return dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))

    def get_last_prediction_explanation(self, X_last: np.ndarray) -> dict[str, float]:
        """
        Return signed SHAP values for the last (most recent) row of X_last.

        Parameters
        ----------
        X_last : np.ndarray
            Either shape (n_features,) or (n_samples, n_features). If 1-D,
            it is reshaped to (1, n_features).

        Returns
        -------
        dict[str, float]
            Mapping of feature name → signed SHAP value (can be negative).
        """
        X_last = np.asarray(X_last)
        if X_last.ndim == 1:
            X_last = X_last.reshape(1, -1)

        # Extract the last row only for efficiency
        X_row = X_last[[-1], :]   # shape (1, n_features)

        explainer = self._get_explainer(X_last)

        if self.model_name in self._TREE_MODELS:
            sv = explainer.shap_values(X_row)
        else:
            sv = explainer.shap_values(X_row, nsamples=100)

        signed_values = self._normalize_shap_values_signed(sv)  # (n_features,)

        return {
            name: float(val)
            for name, val in zip(self.feature_names, signed_values)
        }
