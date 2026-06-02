import numpy as np
from hmmlearn.hmm import GaussianHMM


class HMMRegimeDetector:
    """
    Hidden Markov Model (HMM) based market regime detector.
    Models market conditions as latent states with transition dynamics,
    using log returns and volatility as observations.

    States are ordered by mean log return:
        0 = Bull   (highest mean return)
        1 = Sideways (middle mean return)
        2 = Bear   (lowest mean return)
    """

    def __init__(self, n_components=3, covariance_type="diag", n_iter=100):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=42,
        )
        self.state_map = {}  # {original_hmm_idx: reordered_idx}

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the HMM model on observation sequences.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 2)
            Two-column array: [log_returns, volatility].
        """
        X_arr = np.asarray(X, dtype=float)
        # Drop rows with NaN or Inf
        X_arr = X_arr[np.isfinite(X_arr).all(axis=1)]

        self.model.fit(X_arr)

        # Order states by mean log_return (column 0): highest → Bull(0), lowest → Bear(2)
        mean_returns = self.model.means_[:, 0]
        # argsort ascending: index 0 = lowest (Bear), index 2 = highest (Bull)
        sorted_asc = np.argsort(mean_returns)  # [bear_idx, sideways_idx, bull_idx]

        # Desired ordering: 0=Bull, 1=Sideways, 2=Bear
        # sorted_asc[2] should map to 0 (Bull), sorted_asc[1] → 1, sorted_asc[0] → 2
        self.state_map = {
            sorted_asc[2]: 0,  # Bull
            sorted_asc[1]: 1,  # Sideways
            sorted_asc[0]: 2,  # Bear
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regime labels for each observation.

        Returns
        -------
        np.ndarray of int: 0=Bull, 1=Sideways, 2=Bear
        """
        X_arr = np.asarray(X, dtype=float)
        raw = self.model.predict(X_arr)
        return np.array([self.state_map[s] for s in raw])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return posterior state probabilities for each observation.

        Returns
        -------
        np.ndarray, shape (n_samples, 3)
            Columns reordered as [Bull, Sideways, Bear].
        """
        X_arr = np.asarray(X, dtype=float)
        posteriors = self.model.predict_proba(X_arr)  # (n_samples, n_components)

        # Build column reorder: new_col_idx → original_hmm_col
        # state_map[orig] = new  →  invert to get orig for each new index
        inv_map = {new: orig for orig, new in self.state_map.items()}
        col_order = [inv_map[i] for i in range(self.n_components)]
        return posteriors[:, col_order]

    def get_transition_matrix(self) -> np.ndarray:
        """
        Return the (3, 3) transition probability matrix reordered by state_map.

        Entry [i, j] is the probability of transitioning from state i to state j,
        where rows and columns follow the Bull/Sideways/Bear ordering.
        """
        T = self.model.transmat_  # (n_components, n_components) in original HMM order

        # Invert state_map to get: new_idx → orig_idx
        inv_map = {new: orig for orig, new in self.state_map.items()}
        row_col_order = [inv_map[i] for i in range(self.n_components)]

        # Reorder rows then columns
        T_reordered = T[np.ix_(row_col_order, row_col_order)]
        return T_reordered

    def get_regime_names(self) -> list:
        """Return regime names in label order."""
        return ["Bull", "Sideways", "Bear"]
