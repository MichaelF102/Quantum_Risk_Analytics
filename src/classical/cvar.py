"""
Conditional Value-at-Risk (CVaR / Expected Shortfall) computation routines.
"""
from typing import Tuple, Optional
import numpy as np
from src.classical.var import compute_var_from_losses


def compute_cvar_from_losses(
    losses: np.ndarray,
    confidence_level: float = 0.95
) -> float:
    """
    Compute Conditional Value-at-Risk (CVaR / Expected Shortfall) from losses.
    CVaR_alpha = E[L | L >= VaR_alpha]
    """
    if len(losses) == 0:
        return 0.0
    var = compute_var_from_losses(losses, confidence_level)
    tail_losses = losses[losses >= var]
    if len(tail_losses) == 0:
        return var
    return float(np.mean(tail_losses))


def compute_cvar_from_returns(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    portfolio_value: Optional[float] = None
) -> Tuple[float, float]:
    """
    Compute percentage CVaR and monetary CVaR from portfolio returns.
    
    Returns:
        (cvar_percent, cvar_cash)
    """
    losses = -np.asarray(returns, dtype=float)
    cvar_pct = compute_cvar_from_losses(losses, confidence_level)
    cvar_cash = cvar_pct * portfolio_value if portfolio_value is not None else 0.0
    return cvar_pct, cvar_cash
