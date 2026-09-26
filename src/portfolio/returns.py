"""
Portfolio return calculations and performance metrics.
"""
from typing import Tuple
import numpy as np
import pandas as pd


def calculate_portfolio_returns(
    returns: pd.DataFrame,
    weights: np.ndarray
) -> pd.Series:
    """
    Calculate portfolio return series: R_p = w^T R = sum(w_i * R_i).
    
    Args:
        returns: DataFrame of asset returns (T x N)
        weights: 1D array of portfolio weights (length N)
        
    Returns:
        Series of daily portfolio returns (length T)
    """
    weights = np.asarray(weights, dtype=float)
    if returns.shape[1] != len(weights):
        raise ValueError(f"Asset count mismatch: returns has {returns.shape[1]} columns, weights length is {len(weights)}.")
        
    port_returns = returns.values @ weights
    return pd.Series(port_returns, index=returns.index, name="Portfolio_Return")


def calculate_cumulative_returns(portfolio_returns: pd.Series) -> pd.Series:
    """Calculate cumulative compound return series starting at 1.0 (or 0%)."""
    return (1.0 + portfolio_returns).cumprod() - 1.0


def calculate_annualized_metrics(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.02,
    trading_days: int = 252
) -> Tuple[float, float, float]:
    """
    Calculate annualized return, annualized volatility, and Sharpe Ratio.
    
    Returns:
        (ann_return, ann_vol, sharpe_ratio)
    """
    n_days = len(portfolio_returns)
    if n_days == 0:
        return 0.0, 0.0, 0.0
        
    # Compound annualized growth rate (geometric mean)
    comp_cum = (1.0 + portfolio_returns).prod()
    if comp_cum <= 0:
        ann_return = float(portfolio_returns.mean() * trading_days)
    else:
        ann_return = float((comp_cum ** (trading_days / n_days)) - 1.0)
        
    ann_vol = float(portfolio_returns.std() * np.sqrt(trading_days))
    
    if ann_vol > 1e-8:
        sharpe = float((ann_return - risk_free_rate) / ann_vol)
    else:
        sharpe = 0.0
        
    return ann_return, ann_vol, sharpe
