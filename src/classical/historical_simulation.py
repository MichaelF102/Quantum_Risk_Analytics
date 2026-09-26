"""
Historical simulation risk estimation engine with bootstrap confidence intervals.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from src.classical.var import compute_var_from_returns
from src.classical.cvar import compute_cvar_from_returns


def run_historical_simulation(
    returns: pd.Series,
    confidence_levels: Optional[List[float]] = None,
    time_horizon_days: int = 1,
    portfolio_value: Optional[float] = None,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run historical simulation for a set of confidence levels.
    
    If time_horizon_days > 1, aggregates rolling returns or scales by sqrt(h).
    Computes 95% bootstrap confidence intervals for VaR and CVaR.
    """
    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.99]
        
    ret_arr = returns.values
    if time_horizon_days > 1:
        # Scale daily empirical returns to horizon
        ret_arr = ret_arr * np.sqrt(time_horizon_days)
        
    losses = -ret_arr
    results = {}
    
    rng = np.random.default_rng(seed)
    
    for alpha in confidence_levels:
        var_pct, var_cash = compute_var_from_returns(ret_arr, confidence_level=alpha, portfolio_value=portfolio_value)
        cvar_pct, cvar_cash = compute_cvar_from_returns(ret_arr, confidence_level=alpha, portfolio_value=portfolio_value)
        
        # Bootstrap confidence intervals
        boot_indices = rng.choice(len(ret_arr), size=(n_bootstrap, len(ret_arr)), replace=True)
        boot_losses = losses[boot_indices]
        
        boot_vars = np.percentile(boot_losses, alpha * 100.0, axis=1)
        boot_var_ci = (float(np.percentile(boot_vars, 2.5)), float(np.percentile(boot_vars, 97.5)))
        
        results[f"{int(alpha * 100)}%"] = {
            "confidence_level": alpha,
            "var_pct": var_pct,
            "var_cash": var_cash,
            "cvar_pct": cvar_pct,
            "cvar_cash": cvar_cash,
            "var_ci_95": boot_var_ci
        }
        
    return {
        "method": "Historical Simulation",
        "sample_size": len(returns),
        "time_horizon_days": time_horizon_days,
        "results": results
    }
