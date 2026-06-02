import numpy as np
import pandas as pd


def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical Conditional Value-at-Risk (CVaR / Expected Shortfall).

    CVaR_alpha = -E[r | r <= VaR_alpha]
    where VaR_alpha is the (1-confidence) quantile of returns.

    Returns positive float (loss as positive number, e.g. 0.025 = 2.5% expected loss).
    Returns 0.0 if no returns fall below the threshold.
    """
    var_threshold = np.quantile(returns, 1 - confidence)
    tail = returns[returns <= var_threshold]
    cvar = -tail.mean() if len(tail) > 0 else 0.0
    return float(cvar)


def calculate_position_scale_cvar(returns: pd.Series, target_cvar: float = 0.02) -> float:
    """
    Position scaling factor based on CVaR.

    scale = target_cvar / realized_cvar_95
    Clips to [0.1, 2.0] to prevent extreme leverage changes.
    Returns 1.0 if realized_cvar is 0 (no loss data).
    """
    realized_cvar = calculate_cvar(returns, confidence=0.95)
    if realized_cvar == 0.0:
        return 1.0
    scale = target_cvar / realized_cvar
    return float(np.clip(scale, 0.1, 2.0))


class TailRiskModel:
    def __init__(self, confidence_levels: list[float] = None):
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]
        self.confidence_levels = confidence_levels
        self._returns: pd.Series | None = None

    def fit(self, returns: pd.Series) -> None:
        """
        Store returns for later computation.
        Validates that at least 30 data points are provided.
        """
        if len(returns) < 30:
            raise ValueError(
                f"At least 30 data points required for fitting; got {len(returns)}."
            )
        self._returns = returns.copy()

    def compute(self) -> dict:
        """
        Returns:
            {
                "cvar_95": float,        # CVaR at 95% confidence (positive, as % loss)
                "cvar_99": float,        # CVaR at 99% confidence
                "var_95": float,         # VaR at 95% confidence (positive)
                "var_99": float,         # VaR at 99% confidence
                "position_scale": float  # position_scale_cvar using cvar_95, target=2%
            }
        """
        if self._returns is None:
            raise RuntimeError(
                "TailRiskModel.fit() must be called before compute()."
            )

        returns = self._returns

        var_95 = float(-np.quantile(returns, 1 - 0.95))
        var_99 = float(-np.quantile(returns, 1 - 0.99))

        cvar_95 = calculate_cvar(returns, confidence=0.95)
        cvar_99 = calculate_cvar(returns, confidence=0.99)

        position_scale = calculate_position_scale_cvar(returns, target_cvar=0.02)

        return {
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "var_95": var_95,
            "var_99": var_99,
            "position_scale": position_scale,
        }
