"""
Preprocessing utilities for financial price data and returns.
"""
from typing import Tuple
import pandas as pd
import numpy as np


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean price dataframe by ensuring datetime index, sorting chronologically,
    and forward-filling missing values (followed by backward fill for any initial NaNs).
    """
    df = df.copy()
    
    # Try identifying Date column
    date_col = None
    for col in df.columns:
        if str(col).lower() in ["date", "timestamp", "time", "datetime"]:
            date_col = col
            break
            
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass  # Keep existing index if conversion fails
            
    df = df.sort_index()
    # Retain only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]
    
    # Forward-fill and backward-fill gaps
    df = df.ffill().bfill()
    return df


def calculate_returns(
    prices: pd.DataFrame,
    method: str = "log"
) -> pd.DataFrame:
    """
    Calculate asset returns from price series.
    
    Args:
        prices: DataFrame of asset prices
        method: 'log' for continuously compounded returns, 'arithmetic' for simple percentage returns.
        
    Returns:
        DataFrame of returns with first row dropped (or filled).
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "arithmetic":
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown return calculation method: {method}. Choose 'log' or 'arithmetic'.")
        
    returns = returns.dropna(how="all").dropna(axis=0)
    return returns


def calculate_rolling_volatility(
    returns: pd.DataFrame,
    window: int = 21,
    annualize: bool = True,
    trading_days: int = 252
) -> pd.DataFrame:
    """Calculate rolling standard deviation of returns."""
    rolling_vol = returns.rolling(window=window).std()
    if annualize:
        rolling_vol = rolling_vol * np.sqrt(trading_days)
    return rolling_vol.dropna()
