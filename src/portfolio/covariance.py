"""
Covariance matrix and correlation estimations for portfolio risk modeling.
"""
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def calculate_covariance_matrix(
    returns: pd.DataFrame,
    method: str = "sample",
    annualize: bool = True,
    trading_days: int = 252
) -> np.ndarray:
    """
    Calculate covariance matrix of asset returns.
    
    Args:
        returns: Asset return DataFrame (T x N)
        method: 'sample' or 'ledoit_wolf' (shrinkage for high-dimensional or noisy data)
        annualize: Whether to multiply by 252
        trading_days: Number of trading periods in year
        
    Returns:
        N x N covariance matrix
    """
    if method == "sample":
        cov = returns.cov().values
    elif method == "ledoit_wolf":
        lw = LedoitWolf().fit(returns.values)
        cov = lw.covariance_
    else:
        raise ValueError(f"Unknown covariance method: {method}. Choose 'sample' or 'ledoit_wolf'.")
        
    if annualize:
        cov = cov * trading_days
        
    return cov


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate Pearson correlation matrix between assets."""
    return returns.corr()


def calculate_asset_volatilities(
    returns: pd.DataFrame,
    annualize: bool = True,
    trading_days: int = 252
) -> pd.Series:
    """Calculate standard deviations for each asset."""
    vols = returns.std()
    if annualize:
        vols = vols * np.sqrt(trading_days)
    return vols
