"""
Tests for VaR, CVaR, Parametric models, and Historical Simulation.
"""
import pytest
import numpy as np
from src.classical.var import compute_var_from_losses, compute_var_from_returns
from src.classical.cvar import compute_cvar_from_losses, compute_cvar_from_returns
from src.classical.parametric_var import compute_parametric_normal_risk, compute_parametric_student_t_risk


def test_var_and_cvar_relationship():
    """Mathematical property: CVaR >= VaR for any non-degenerate loss distribution."""
    np.random.seed(42)
    losses = np.random.normal(loc=0.01, scale=0.03, size=10_000)
    
    var_95 = compute_var_from_losses(losses, confidence_level=0.95)
    cvar_95 = compute_cvar_from_losses(losses, confidence_level=0.95)
    
    assert cvar_95 >= var_95
    assert var_95 > 0.0


def test_parametric_normal_risk():
    mu = 0.0005
    vol = 0.015
    res = compute_parametric_normal_risk(mu, vol, confidence_level=0.95, time_horizon_days=1, portfolio_value=1_000_000.0)
    
    assert res["var_pct"] > 0
    assert res["cvar_pct"] > res["var_pct"]
    assert pytest.approx(res["var_cash"], 1.0) == res["var_pct"] * 1_000_000.0


def test_parametric_student_t_risk():
    np.random.seed(42)
    returns = np.random.standard_t(df=5, size=2000) * 0.01
    res = compute_parametric_student_t_risk(returns, confidence_level=0.95, time_horizon_days=1)
    
    assert res["var_pct"] > 0
    assert res["cvar_pct"] >= res["var_pct"]
    assert res["degrees_of_freedom"] > 0
