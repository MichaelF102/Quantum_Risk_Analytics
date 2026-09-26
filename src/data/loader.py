"""
Data loading utilities for bundled demo datasets and user uploaded files.
"""
import io
import os
from typing import Optional, Union
import pandas as pd
from src.data.validation import validate_price_dataframe, DataValidationError
from src.data.preprocessing import clean_price_data
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_demo_data(filepath: str = "data/sample_portfolio.csv") -> pd.DataFrame:
    """Load the default calibrated multi-asset market dataset."""
    if not os.path.exists(filepath):
        # Fallback search if current working dir is app/ or subfolder
        candidate = os.path.join(os.path.dirname(__file__), "..", "..", filepath)
        if os.path.exists(candidate):
            filepath = os.path.abspath(candidate)
        else:
            raise FileNotFoundError(f"Sample data file not found at: {filepath}")
            
    df = pd.read_csv(filepath)
    df = clean_price_data(df)
    valid, msg = validate_price_dataframe(df)
    if not valid:
        raise DataValidationError(f"Demo data validation failed: {msg}")
    return df


def load_user_csv(
    file_or_buffer: Union[str, io.BytesIO, io.StringIO]
) -> pd.DataFrame:
    """
    Load and parse user-provided CSV data.
    Supports:
    1. Wide format: 'Date, AAPL, MSFT, GOOG, ...'
    2. Long format: 'Date, Asset, Close' (or 'Ticker', 'Price')
    """
    df = pd.read_csv(file_or_buffer)
    
    # Check if long format
    cols_lower = [str(c).strip().lower() for c in df.columns]
    
    is_long_format = False
    asset_col = None
    price_col = None
    date_col = None
    
    for idx, c in enumerate(cols_lower):
        if c in ["asset", "ticker", "symbol", "instrument"]:
            asset_col = df.columns[idx]
        elif c in ["close", "adj close", "price", "value"]:
            price_col = df.columns[idx]
        elif c in ["date", "timestamp", "time"]:
            date_col = df.columns[idx]
            
    if asset_col and price_col and date_col:
        is_long_format = True
        logger.info(f"Detected long format CSV with columns: {date_col}, {asset_col}, {price_col}")
        df = df.pivot(index=date_col, columns=asset_col, values=price_col)
        
    df = clean_price_data(df)
    valid, msg = validate_price_dataframe(df)
    if not valid:
        raise DataValidationError(f"Uploaded CSV validation failed: {msg}")
        
    return df
