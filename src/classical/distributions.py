"""
Distribution fitting and sampling for risk modeling (Normal, Student-t, Empirical).
"""
from typing import Dict, Any, Tuple
import numpy as np
from scipy import stats


def fit_normal(data: np.ndarray) -> Tuple[float, float]:
    """Fit Gaussian distribution returning (mean, std)."""
    mu, sigma = stats.norm.fit(data)
    return float(mu), float(sigma)


def fit_student_t(data: np.ndarray) -> Tuple[float, float, float]:
    """Fit Student-t distribution returning (degrees_of_freedom, loc, scale)."""
    df, loc, scale = stats.t.fit(data)
    return float(df), float(loc), float(scale)


def sample_fitted_distribution(
    dist_type: str,
    params: Dict[str, Any],
    n_samples: int = 10_000,
    seed: int = 42
) -> np.ndarray:
    """Sample from fitted distribution."""
    rng = np.random.default_rng(seed)
    if dist_type == "normal":
        return rng.normal(loc=params["mean"], scale=params["std"], size=n_samples)
    elif dist_type == "student_t":
        # Standard Student-t scaled and shifted
        raw_t = rng.standard_t(df=params["df"], size=n_samples)
        return params["loc"] + params["scale"] * raw_t
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")
