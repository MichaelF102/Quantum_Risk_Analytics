"""
EGARCH(1,1) (Exponential GARCH) volatility modeling extension with leverage effect.
"""
from typing import Dict, Any
import numpy as np
import pandas as pd
from arch import arch_model


def fit_egarch(
    returns: pd.Series,
    p: int = 1,
    o: int = 1,
    q: int = 1,
    rescale: bool = True
) -> Dict[str, Any]:
    """
    Fit EGARCH(p, o, q) model to portfolio return series.
    Captures asymmetric volatility (leverage effect).
    
    ln(sigma_t^2) = omega + alpha * |epsilon_{t-1}| + gamma * epsilon_{t-1} + beta * ln(sigma_{t-1}^2)
    """
    scale_factor = 100.0 if rescale else 1.0
    scaled_ret = returns * scale_factor
    
    try:
        am = arch_model(scaled_ret, vol="EGARCH", p=p, o=o, q=q, dist="normal", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        
        cond_vol = res.conditional_volatility / scale_factor
        forecast = res.forecast(horizon=1)
        next_vol = np.sqrt(forecast.variance.iloc[-1].values[0]) / scale_factor
        
        return {
            "success": True,
            "model_type": f"EGARCH({p},{o},{q})",
            "omega": float(res.params.get("omega", 0.0)),
            "alpha": float(res.params.get("alpha[1]", 0.0)),
            "gamma": float(res.params.get("gamma[1]", 0.0)),  # Leverage parameter (<0 means negative shocks increase vol)
            "beta": float(res.params.get("beta[1]", 0.0)),
            "aic": float(res.aic),
            "bic": float(res.bic),
            "conditional_volatility": cond_vol,
            "forecast_volatility": float(next_vol),
            "annualized_forecast_vol": float(next_vol * np.sqrt(252))
        }
    except Exception as e:
        sample_vol = float(returns.std())
        return {
            "success": False,
            "error": str(e),
            "model_type": f"EGARCH({p},{o},{q}) (Fallback)",
            "conditional_volatility": pd.Series(sample_vol, index=returns.index),
            "forecast_volatility": sample_vol,
            "annualized_forecast_vol": sample_vol * np.sqrt(252)
        }
