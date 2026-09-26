"""
Tests for edge cases, error handling, and input validation.
"""
import pytest
import numpy as np
import pandas as pd

from src.data.validation import validate_price_dataframe, validate_portfolio_weights, DataValidationError
from src.portfolio.portfolio import Portfolio
from src.classical.monte_carlo import MonteCarloEngine


def test_weights_not_sum_to_one():
    invalid_weights = np.array([0.3, 0.3, 0.3])  # Sums to 0.90
    valid, msg = validate_portfolio_weights(invalid_weights)
    assert not valid
    assert "sum to 1.0" in msg


def test_negative_weights_disallowed():
    invalid_weights = np.array([1.2, -0.2])
    valid, msg = validate_portfolio_weights(invalid_weights, allow_short=False)
    assert not valid
    assert "Negative weights" in msg


def test_empty_dataframe_validation():
    empty_df = pd.DataFrame()
    valid, msg = validate_price_dataframe(empty_df)
    assert not valid
    assert "empty" in msg.lower()


def test_insufficient_observations_validation():
    short_df = pd.DataFrame({"AAPL": [150.0, 151.0, 152.0]})
    valid, msg = validate_price_dataframe(short_df, min_observations=30)
    assert not valid
    assert "Insufficient observations" in msg


def test_non_positive_prices_validation():
    bad_df = pd.DataFrame({"AAPL": [150.0, 0.0, -10.0] * 15})
    valid, msg = validate_price_dataframe(bad_df, min_observations=30)
    assert not valid
    assert "Non-positive" in msg


def test_portfolio_raises_on_invalid_weights():
    dates = pd.date_range("2022-01-01", periods=50)
    df = pd.DataFrame({"A": np.linspace(100, 120, 50), "B": np.linspace(50, 60, 50)}, index=dates)
    
    with pytest.raises(DataValidationError):
        Portfolio(df, weights=np.array([0.6, 0.6]))  # 1.2 sum
