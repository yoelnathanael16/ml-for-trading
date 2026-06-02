import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_zscore(prices: pd.Series, window: int = 20) -> pd.Series:
    """Z-score: (price - rolling_mean) / rolling_std"""
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    return (prices - rolling_mean) / rolling_std


def estimate_ou_halflife(prices: pd.Series) -> float:
    """
    Ornstein-Uhlenbeck half-life estimation via OLS.

    Δp_t = a + b * p_{t-1} + ε
    Half-life = -log(2) / b   (b should be negative for mean-reverting series)

    Returns positive half-life in days, or inf if not mean-reverting (b >= 0).
    """
    if len(prices) < 2:
        return np.inf

    delta_p = prices.diff().dropna()
    lag_p = prices.shift(1).dropna()

    # Align lengths (delta_p and lag_p should be the same length after dropna)
    min_len = min(len(delta_p), len(lag_p))
    delta_p = delta_p.iloc[:min_len]
    lag_p = lag_p.iloc[:min_len]

    X = sm.add_constant(lag_p.values)
    result = sm.OLS(delta_p.values, X).fit()
    b = result.params[1]

    halflife = -np.log(2) / b if b < 0 else np.inf
    return halflife


def mean_reversion_signal(
    prices: pd.Series, z_entry: float = 2.0, z_exit: float = 0.5
) -> pd.Series:
    """
    Generate mean reversion signals based on z-score thresholds.

    Signal values:
      1:  long entry (z-score <= -z_entry: price is too low, expect reversion up)
     -1:  short entry (z-score >= z_entry: price is too high, expect reversion down)
      0:  exit / neutral (|z-score| <= z_exit)

    Uses raw threshold logic without forward fill.
    Returns pd.Series of int signals with same index as prices.
    """
    zscore = compute_zscore(prices)
    signal = pd.Series(np.nan, index=prices.index)

    signal[zscore <= -z_entry] = 1
    signal[zscore >= z_entry] = -1
    signal[zscore.abs() <= z_exit] = 0

    return signal.fillna(0).astype(int)


class MeanReversionDetector:
    def __init__(
        self, window: int = 20, z_entry: float = 2.0, z_exit: float = 0.5
    ) -> None:
        self.window = window
        self.z_entry = z_entry
        self.z_exit = z_exit

        self.zscore_: pd.Series | None = None
        self.halflife_: float | None = None
        self.signal_: pd.Series | None = None

    def fit(self, prices: pd.Series) -> None:
        """Compute and store: zscore series, halflife, signal series."""
        self.zscore_ = compute_zscore(prices, window=self.window)
        self.halflife_ = estimate_ou_halflife(prices)
        self.signal_ = mean_reversion_signal(
            prices, z_entry=self.z_entry, z_exit=self.z_exit
        )

    def predict(self, prices: pd.Series) -> dict:
        """
        Returns:
            {
                "zscore": float,          # latest z-score value
                "halflife": float,        # OU half-life in days
                "signal": int,            # latest signal (-1, 0, 1)
                "is_mean_reverting": bool # True if halflife > 0 and halflife < 252
            }
        """
        self.fit(prices)

        latest_zscore = float(self.zscore_.iloc[-1]) if self.zscore_ is not None else np.nan
        latest_signal = int(self.signal_.iloc[-1]) if self.signal_ is not None else 0
        halflife = self.halflife_ if self.halflife_ is not None else np.inf
        is_mean_reverting = halflife > 0 and halflife < 252

        return {
            "zscore": latest_zscore,
            "halflife": halflife,
            "signal": latest_signal,
            "is_mean_reverting": is_mean_reverting,
        }
