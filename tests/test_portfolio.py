"""
Tests for portfolio returns, weights, covariance, and portfolio object.
"""
import pytest
import numpy as np
import pandas as pd

from src.data.loader import load_demo_data
from src.portfolio.portfolio import Portfolio
from src.portfolio.returns import calculate_portfolio_returns, calculate_annualized_metrics
from src.portfolio.covariance import calculate_covariance_matrix, calculate_correlation_matrix
from src.data.validation import validate_portfolio_weights, DataValidationError


def test_portfolio_initialization():
    prices = load_demo_data()
    portfolio = Portfolio(prices, initial_value=1_000_000.0)
    
    assert portfolio.num_assets == len(prices.columns)
    assert len(portfolio.weights) == portfolio.num_assets
    assert pytest.approx(float(np.sum(portfolio.weights)), 1e-4) == 1.0
    assert len(portfolio.portfolio_returns) == len(prices) - 1
    assert portfolio.ann_vol > 0.0


def test_portfolio_custom_weights():
    prices = load_demo_data()
    n = len(prices.columns)
    weights = np.zeros(n)
    weights[0] = 0.5
    weights[1] = 0.5
    
    portfolio = Portfolio(prices, weights=weights)
    assert pytest.approx(portfolio.weights[0], 1e-4) == 0.5
    assert pytest.approx(portfolio.weights[1], 1e-4) == 0.5
    assert pytest.approx(float(np.sum(portfolio.weights)), 1e-4) == 1.0


def test_covariance_and_correlation_properties():
    prices = load_demo_data()
    returns = np.log(prices / prices.shift(1)).dropna()
    
    cov = calculate_covariance_matrix(returns, annualize=True)
    corr = calculate_correlation_matrix(returns)
    
    # Covariance must be symmetric positive semi-definite
    assert np.allclose(cov, cov.T, atol=1e-6)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues >= -1e-6)
    
    # Correlation diagonal must be 1.0
    assert np.allclose(np.diag(corr.values), 1.0, atol=1e-5)
