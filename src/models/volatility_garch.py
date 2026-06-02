"""GARCH-based volatility model for position sizing."""

import numpy as np
import pandas as pd
from arch import arch_model


class GARCHVolatilityModel:
    """
    GARCH(p, q) volatility model for estimating annualized volatility.

    Returns passed in should be in decimal form (e.g. 0.01 for 1%).
    Internally, returns are scaled by 100 for numerical stability before
    passing to the arch library, then converted back on output.
    """

    def __init__(self, p: int = 1, q: int = 1):
        self.p = p
        self.q = q
        self.result = None

    def fit(self, returns: pd.Series) -> None:
        """
        Fit GARCH(p, q) model on a returns series.

        Parameters
        ----------
        returns : pd.Series
            Daily returns in decimal form (e.g. 0.01 = 1%).
        """
        scaled = returns.dropna() * 100
        model = arch_model(scaled, vol="Garch", p=self.p, q=self.q, dist="normal")
        self.result = model.fit(disp="off")

    def forecast(self, horizon: int = 1) -> float:
        """
        Forecast annualized volatility h steps ahead from the last observed return.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast (default 1).

        Returns
        -------
        float
            Annualized volatility (e.g. 0.18 for 18%).
        """
        if self.result is None:
            raise RuntimeError("Model must be fitted before forecasting. Call fit() first.")

        forecasts = self.result.forecast(horizon=horizon)
        # variance is in (return * 100)^2 units
        variance = forecasts.variance.iloc[-1, horizon - 1]
        # convert back to decimal annualized vol
        annualized_vol = np.sqrt(variance / 10_000) * np.sqrt(252)
        return float(annualized_vol)

    def forecast_series(self, returns: pd.Series) -> pd.Series:
        """
        Return a rolling one-step-ahead annualized volatility series aligned to `returns`.

        Uses the conditional volatility from the already-fitted model rather than
        refitting per observation. NaN values at the start (GARCH burn-in) are
        forward-filled so the output has the same index as `returns`.

        Parameters
        ----------
        returns : pd.Series
            Daily returns in decimal form (same series used for fitting).

        Returns
        -------
        pd.Series
            Annualized conditional volatility with the same index as `returns`.
        """
        if self.result is None:
            raise RuntimeError("Model must be fitted before calling forecast_series(). Call fit() first.")

        # conditional_volatility is in % units (return * 100 scale)
        cond_vol = self.result.conditional_volatility  # pd.Series indexed by the fitted data

        # Convert to annualized decimal volatility
        ann_vol = cond_vol / 100 * np.sqrt(252)

        # Reindex to match the original returns index (handles any NaN burn-in period)
        ann_vol = ann_vol.reindex(returns.index)

        # Forward-fill NaNs at the beginning (GARCH needs a few observations to initialise)
        ann_vol = ann_vol.ffill()

        return ann_vol
