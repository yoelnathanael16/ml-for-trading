import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


def _recursive_bisect(cov: pd.DataFrame, sort_ix: list) -> dict:
    """
    Recursively bisect clusters and assign weights via inverse-variance allocation.

    Args:
        cov: Covariance matrix (DataFrame, indexed by integer position).
        sort_ix: Ordered list of integer column/row positions in cov.

    Returns:
        dict mapping integer index -> weight (unnormalized within this subtree,
        but will sum to 1.0 when called from the root).
    """
    if len(sort_ix) == 1:
        return {sort_ix[0]: 1.0}

    split = len(sort_ix) // 2
    left = sort_ix[:split]
    right = sort_ix[split:]

    def cluster_var(cluster_idxs: list) -> float:
        """Compute inverse-variance weighted cluster variance."""
        sub_cov = cov.iloc[cluster_idxs, cluster_idxs]
        inv_diag = 1.0 / np.diag(sub_cov.values)
        w = inv_diag / inv_diag.sum()
        return float(w @ sub_cov.values @ w)

    left_var = cluster_var(left)
    right_var = cluster_var(right)

    total_var = left_var + right_var
    left_weight = right_var / total_var   # allocate less to higher-variance cluster
    right_weight = left_var / total_var

    left_weights = _recursive_bisect(cov, left)
    right_weights = _recursive_bisect(cov, right)

    combined = {k: v * left_weight for k, v in left_weights.items()}
    combined.update({k: v * right_weight for k, v in right_weights.items()})
    return combined


def calculate_hrp_weights(returns: pd.DataFrame) -> dict:
    """
    Compute HRP (Hierarchical Risk Parity) portfolio weights.

    Steps:
      1. Correlation matrix from returns.
      2. Distance matrix D = sqrt(0.5 * (1 - corr)).
      3. Hierarchical clustering (single linkage) on D.
      4. Quasi-diagonalization: reorder assets by cluster leaf order.
      5. Recursive bisection to assign weights.

    Args:
        returns: DataFrame with daily log returns; columns are ticker names.

    Returns:
        dict mapping ticker -> weight; weights sum to 1.0.
    """
    tickers = list(returns.columns)
    n = len(tickers)

    if n == 0:
        return {}
    if n == 1:
        return {tickers[0]: 1.0}

    # Step 1 & 2: correlation -> distance matrix
    corr = returns.corr()
    dist = np.sqrt(0.5 * (1.0 - corr.values))
    np.fill_diagonal(dist, 0.0)

    # Step 3: hierarchical clustering (single linkage)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")

    # Step 4: quasi-diagonalization — leaf order from dendrogram
    sort_ix = list(leaves_list(link))  # integer indices into tickers

    # Step 5: recursive bisection on the covariance matrix
    cov = returns.cov()
    raw_weights = _recursive_bisect(cov, sort_ix)

    # Normalise (should already sum to 1, but guard against floating-point drift)
    total = sum(raw_weights.values())
    weights = {tickers[idx]: w / total for idx, w in raw_weights.items()}

    assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights do not sum to 1.0"

    return weights
