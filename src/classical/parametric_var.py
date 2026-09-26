"""
Parametric Value-at-Risk and Conditional Value-at-Risk models (Gaussian & Student-t).
"""
from typing import Dict, Any, Optional
import numpy as np
from scipy import stats


def compute_parametric_normal_risk(
    mean_daily_return: float,
    daily_volatility: float,
    confidence_level: float = 0.95,
    time_horizon_days: int = 1,
    portfolio_value: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute analytical Gaussian VaR and CVaR.
    
    Formula:
      VaR_pct = - (mu * h - z_alpha * sigma * sqrt(h)) = -mu * h + z_alpha * sigma * sqrt(h)
      CVaR_pct = -mu * h + sigma * sqrt(h) * (phi(z_alpha) / (1 - alpha))
    """
    h = float(time_horizon_days)
    mu_h = mean_daily_return * h
    sigma_h = daily_volatility * np.sqrt(h)
    
    z_alpha = stats.norm.ppf(confidence_level)
    phi_z = stats.norm.pdf(z_alpha)
    
    var_pct = float(-mu_h + z_alpha * sigma_h)
    cvar_pct = float(-mu_h + sigma_h * (phi_z / (1.0 - confidence_level)))
    
    var_cash = var_pct * portfolio_value if portfolio_value is not None else 0.0
    cvar_cash = cvar_pct * portfolio_value if portfolio_value is not None else 0.0
    
    return {
        "var_pct": var_pct,
        "var_cash": var_cash,
        "cvar_pct": cvar_pct,
        "cvar_cash": cvar_cash,
        "confidence_level": confidence_level,
        "time_horizon_days": time_horizon_days,
        "distribution": "Gaussian (Normal)"
    }


def compute_parametric_student_t_risk(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    time_horizon_days: int = 1,
    portfolio_value: Optional[float] = None
) -> Dict[str, float]:
    """
    Fit a Student-t distribution to returns and compute parametric tail risk.
    """
    df, loc, scale = stats.t.fit(returns)
    h = float(time_horizon_days)
    
    # Scaling to horizon h
    loc_h = loc * h
    scale_h = scale * np.sqrt(h)
    
    # 1 - alpha quantile of returns is -(VaR)
    t_q = stats.t.ppf(1.0 - confidence_level, df=df)
    var_pct = float(-(loc_h + scale_h * t_q))
    
    # Numerical expectation for Student-t CVaR
    pdf_func = lambda x: stats.t.pdf(x, df=df, loc=loc_h, scale=scale_h)
    # Loss L = -R, so tail where R <= loc_h + scale_h * t_q
    cutoff_ret = loc_h + scale_h * t_q
    
    # Tail expectation E[-R | R <= cutoff]
    from scipy import integrate
    numerator, _ = integrate.quad(lambda r: (-r) * pdf_func(r), -np.inf, cutoff_ret)
    cvar_pct = float(numerator / (1.0 - confidence_level))
    
    var_cash = var_pct * portfolio_value if portfolio_value is not None else 0.0
    cvar_cash = cvar_pct * portfolio_value if portfolio_value is not None else 0.0
    
    return {
        "var_pct": var_pct,
        "var_cash": var_cash,
        "cvar_pct": cvar_pct,
        "cvar_cash": cvar_cash,
        "degrees_of_freedom": float(df),
        "confidence_level": confidence_level,
        "time_horizon_days": time_horizon_days,
        "distribution": "Student-t"
    }
