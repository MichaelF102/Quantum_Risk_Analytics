"""
Value-at-Risk (VaR) computation routines.
"""
from typing import Union, Tuple, Optional
import numpy as np


def compute_var_from_losses(
    losses: np.ndarray,
    confidence_level: float = 0.95
) -> float:
    """
    Compute Value-at-Risk from an array of losses (where positive values represent losses).
    VaR_alpha = Quantile_alpha(Losses) such that P(L <= VaR) = alpha.
    """
    if len(losses) == 0:
        return 0.0
    return float(np.percentile(losses, confidence_level * 100.0))


def compute_var_from_returns(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    portfolio_value: Optional[float] = None
) -> Tuple[float, float]:
    """
    Compute percentage VaR and monetary VaR from portfolio return series.
    
    Args:
        returns: Array of portfolio returns
        confidence_level: Risk confidence level (e.g. 0.95, 0.99)
        portfolio_value: Optional nominal value to compute cash VaR ($)
        
    Returns:
        (var_percent, var_cash)
    """
    losses = -np.asarray(returns, dtype=float)
    var_pct = compute_var_from_losses(losses, confidence_level)
    
    var_cash = var_pct * portfolio_value if portfolio_value is not None else 0.0
    return var_pct, var_cash
