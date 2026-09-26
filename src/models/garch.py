"""
GARCH(1,1) volatility modeling extension.
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from arch import arch_model


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    rescale: bool = True
) -> Dict[str, Any]:
    """
    Fit GARCH(p,q) model to portfolio return series.
    
    sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
    
    Returns:
        Dict containing model parameters, conditional volatility series, and forecast.
    """
    # Scale returns by 100 for numerical stability during MLE optimization
    scale_factor = 100.0 if rescale else 1.0
    scaled_ret = returns * scale_factor
    
    try:
        am = arch_model(scaled_ret, vol="Garch", p=p, q=q, dist="normal", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        
        # Descale conditional volatility back to original return units
        cond_vol = res.conditional_volatility / scale_factor
        
        # Forecast 1-step ahead
        forecast = res.forecast(horizon=1)
        next_vol = np.sqrt(forecast.variance.iloc[-1].values[0]) / scale_factor
        
        return {
            "success": True,
            "model_type": f"GARCH({p},{q})",
            "omega": float(res.params.get("omega", 0.0)) / (scale_factor ** 2),
            "alpha": float(res.params.get("alpha[1]", 0.0)),
            "beta": float(res.params.get("beta[1]", 0.0)),
            "persistence": float(res.params.get("alpha[1]", 0.0) + res.params.get("beta[1]", 0.0)),
            "aic": float(res.aic),
            "bic": float(res.bic),
            "conditional_volatility": cond_vol,
            "forecast_volatility": float(next_vol),
            "annualized_forecast_vol": float(next_vol * np.sqrt(252))
        }
    except Exception as e:
        # Fallback to sample volatility if fitting fails
        sample_vol = float(returns.std())
        return {
            "success": False,
            "error": str(e),
            "model_type": f"GARCH({p},{q}) (Fallback)",
            "conditional_volatility": pd.Series(sample_vol, index=returns.index),
            "forecast_volatility": sample_vol,
            "annualized_forecast_vol": sample_vol * np.sqrt(252)
        }
