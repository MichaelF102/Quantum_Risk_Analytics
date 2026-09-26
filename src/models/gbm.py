"""
Vectorized Multivariate Geometric Brownian Motion (GBM) simulation engine.
"""
from typing import Tuple, Optional
import numpy as np


def simulate_multivariate_gbm(
    initial_prices: np.ndarray,
    mu: np.ndarray,
    covariance_matrix: np.ndarray,
    time_horizon_years: float,
    n_simulations: int = 20_000,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate correlated multivariate asset terminal prices and returns using vectorization.
    
    Exact analytical solution for terminal price:
      S_i(T) = S_i(0) * exp( (mu_i - 0.5 * sigma_i^2) * T + sqrt(T) * (L @ Z)_i )
      
    Args:
        initial_prices: 1D array of asset prices S_0 (length N)
        mu: 1D array of annualized expected returns (length N)
        covariance_matrix: Annualized N x N covariance matrix
        time_horizon_years: Time horizon T in years (e.g. 10/252)
        n_simulations: Number of Monte Carlo paths
        seed: Random seed for deterministic reproducibility
        
    Returns:
        terminal_prices: (n_simulations x N)
        terminal_returns: (n_simulations x N) arithmetic returns (S_T - S_0) / S_0
    """
    initial_prices = np.asarray(initial_prices, dtype=float)
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(covariance_matrix, dtype=float)
    n_assets = len(initial_prices)
    
    # Compute asset volatilities from diagonal of covariance matrix
    variances = np.diag(cov)
    
    # Cholesky factor of covariance matrix
    # Add small ridge if necessary for numerical stability
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov_stable = cov + 1e-8 * np.eye(n_assets)
        L = np.linalg.cholesky(cov_stable)
        
    rng = np.random.default_rng(seed)
    # Standard normal matrix: (n_simulations x n_assets)
    z = rng.standard_normal(size=(n_simulations, n_assets))
    
    # Correlated random shocks: (n_simulations x n_assets) = z @ L.T
    correlated_shocks = z @ L.T
    
    # Drift term: (mu - 0.5 * sigma^2) * T
    drift = (mu - 0.5 * variances) * time_horizon_years
    diffusion = np.sqrt(time_horizon_years) * correlated_shocks
    
    # Terminal prices
    terminal_prices = initial_prices * np.exp(drift + diffusion)
    terminal_returns = (terminal_prices - initial_prices) / initial_prices
    
    return terminal_prices, terminal_returns
