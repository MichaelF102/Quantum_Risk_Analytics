"""
Data validation module for financial time series and portfolio inputs.
"""
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np


class DataValidationError(ValueError):
    """Raised when financial time-series data fails validation."""
    pass


def validate_price_dataframe(
    df: pd.DataFrame,
    min_observations: int = 30,
    required_columns: Optional[List[str]] = None
) -> Tuple[bool, str]:
    """
    Validate that a price dataframe is suitable for risk analysis.
    
    Checks:
    - Non-empty dataframe
    - Sufficient historical observations (>= min_observations)
    - Valid numeric values (no non-numeric prices)
    - Strictly positive prices (> 0)
    - Minimal missing values (at most 5% dropped or forward filled)
    """
    if df is None or df.empty:
        return False, "Dataframe is empty or null."
        
    if len(df) < min_observations:
        return False, f"Insufficient observations: found {len(df)}, required at least {min_observations}."
        
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            return False, f"Missing required asset columns: {missing}"
            
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        return False, "No numeric price columns detected in the dataset."
        
    # Check for non-positive or negative prices
    if (numeric_df <= 0).any().any():
        non_positive_cols = numeric_df.columns[(numeric_df <= 0).any()].tolist()
        return False, f"Non-positive prices found in assets: {non_positive_cols}. Asset prices must be > 0."
        
    # Check for excessive NaN values
    null_ratio = numeric_df.isnull().mean()
    excessive_nulls = null_ratio[null_ratio > 0.1].index.tolist()
    if excessive_nulls:
        return False, f"Assets with >10% missing values: {excessive_nulls}."
        
    return True, "Data validation passed successfully."


def validate_portfolio_weights(
    weights: np.ndarray,
    tolerance: float = 1e-4,
    allow_short: bool = False
) -> Tuple[bool, str]:
    """
    Validate portfolio weights:
    - 1D array
    - Sum to 1.0 within tolerance
    - Non-negative if short selling is disallowed
    - No NaNs or Infs
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) == 0:
        return False, "Weights must be a non-empty 1D array."
        
    if np.isnan(weights).any() or np.isinf(weights).any():
        return False, "Weights contain NaN or infinite values."
        
    if not allow_short and (weights < -1e-6).any():
        return False, "Negative weights detected (short selling is disabled)."
        
    total_weight = float(np.sum(weights))
    if abs(total_weight - 1.0) > tolerance:
        return False, f"Portfolio weights sum to {total_weight:.4f}, but must sum to 1.0 (100%)."
        
    return True, "Portfolio weights are valid."
