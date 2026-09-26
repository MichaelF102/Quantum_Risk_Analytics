"""
Tests for Classical Monte Carlo Engine determinism, vectorization, and confidence intervals.
"""
import pytest
import numpy as np
from src.classical.monte_carlo import MonteCarloEngine


def test_monte_carlo_engine_determinism():
    n_assets = 3
    weights = np.array([0.4, 0.3, 0.3])
    mu = np.array([0.1, 0.12, 0.08])
    cov = np.array([
        [0.04, 0.01, 0.02],
        [0.01, 0.05, 0.015],
        [0.02, 0.015, 0.06]
    ])
    prices = np.array([100.0, 150.0, 80.0])
    
    engine1 = MonteCarloEngine(1_000_000.0, weights, mu, cov, prices, seed=123)
    res1 = engine1.run_simulation(n_simulations=5000, time_horizon_days=10)
    
    engine2 = MonteCarloEngine(1_000_000.0, weights, mu, cov, prices, seed=123)
    res2 = engine2.run_simulation(n_simulations=5000, time_horizon_days=10)
    
    # Identical seed must yield identical losses and VaR
    np.testing.assert_allclose(res1["losses_pct"], res2["losses_pct"], rtol=1e-10)
    assert res1["risk_metrics"]["95%"]["var_pct"] == res2["risk_metrics"]["95%"]["var_pct"]
    assert res1["risk_metrics"]["95%"]["cvar_pct"] == res2["risk_metrics"]["95%"]["cvar_pct"]


def test_monte_carlo_confidence_intervals():
    n_assets = 2
    weights = np.array([0.5, 0.5])
    mu = np.array([0.08, 0.08])
    cov = np.array([[0.04, 0.01], [0.01, 0.04]])
    prices = np.array([100.0, 100.0])
    
    engine = MonteCarloEngine(1_000_000.0, weights, mu, cov, prices, seed=42)
    res = engine.run_simulation(n_simulations=5000, time_horizon_days=10)
    
    var_95 = res["risk_metrics"]["95%"]["var_pct"]
    ci_low, ci_high = res["risk_metrics"]["95%"]["var_ci_95"]
    
    assert ci_low < var_95 < ci_high
    assert res["risk_metrics"]["95%"]["var_se"] > 0
