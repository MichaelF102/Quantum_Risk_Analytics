"""
Vectorized Classical Monte Carlo Engine for portfolio risk evaluation.
"""
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from src.models.gbm import simulate_multivariate_gbm
from src.classical.var import compute_var_from_losses
from src.classical.cvar import compute_cvar_from_losses


class MonteCarloEngine:
    """
    Reusable, high-performance Monte Carlo risk engine supporting GBM and Multivariate Normal simulation.
    """
    def __init__(
        self,
        initial_portfolio_value: float = 1_000_000.0,
        weights: Optional[np.ndarray] = None,
        expected_returns: Optional[np.ndarray] = None,
        covariance_matrix: Optional[np.ndarray] = None,
        asset_prices: Optional[np.ndarray] = None,
        seed: int = 42
    ):
        self.initial_value = float(initial_portfolio_value)
        self.seed = int(seed)
        
        self.weights = np.asarray(weights, dtype=float) if weights is not None else None
        self.expected_returns = np.asarray(expected_returns, dtype=float) if expected_returns is not None else None
        self.covariance_matrix = np.asarray(covariance_matrix, dtype=float) if covariance_matrix is not None else None
        self.asset_prices = np.asarray(asset_prices, dtype=float) if asset_prices is not None else None

    def run_simulation(
        self,
        n_simulations: int = 20_000,
        time_horizon_days: int = 10,
        confidence_levels: Optional[List[float]] = None,
        method: str = "gbm",
        custom_volatility: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation and return distributions, tail metrics, runtime, and confidence intervals.
        
        Args:
            n_simulations: Number of Monte Carlo draws
            time_horizon_days: Risk horizon in trading days
            confidence_levels: List of confidence alphas (e.g. [0.90, 0.95, 0.99])
            method: 'gbm' (multivariate GBM) or 'mvn' (multivariate normal returns)
            custom_volatility: Optional override from GARCH/EGARCH conditional vol
            
        Returns:
            Dict containing simulated returns, portfolio values, losses, VaR, CVaR, runtime, and error bounds.
        """
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]
            
        start_time = time.perf_counter()
        t_years = time_horizon_days / 252.0
        n_assets = len(self.weights) if self.weights is not None else 1
        
        rng = np.random.default_rng(self.seed)
        
        cov = self.covariance_matrix.copy() if self.covariance_matrix is not None else np.eye(n_assets) * 0.04
        
        # If custom volatility supplied (e.g. from GARCH), rescale covariance
        if custom_volatility is not None and self.weights is not None:
            current_port_var = float(self.weights @ cov @ self.weights)
            target_port_var = (custom_volatility * np.sqrt(252)) ** 2
            if current_port_var > 1e-8:
                cov = cov * (target_port_var / current_port_var)
                
        # Run simulation using vectorized operations
        if method == "gbm" and self.asset_prices is not None:
            _, asset_returns = simulate_multivariate_gbm(
                initial_prices=self.asset_prices,
                mu=self.expected_returns if self.expected_returns is not None else np.zeros(n_assets),
                covariance_matrix=cov,
                time_horizon_years=t_years,
                n_simulations=n_simulations,
                seed=self.seed
            )
            sim_port_returns = asset_returns @ self.weights
        else:
            # Multivariate Normal Return simulation
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                cov_stable = cov + 1e-8 * np.eye(n_assets)
                L = np.linalg.cholesky(cov_stable)
                
            z = rng.standard_normal(size=(n_simulations, n_assets))
            mu = self.expected_returns if self.expected_returns is not None else np.zeros(n_assets)
            
            # Asset returns over horizon t_years
            scaled_returns = (z @ L.T) * np.sqrt(t_years) + (mu * t_years)
            sim_port_returns = scaled_returns @ self.weights
            
        # Terminal portfolio values and losses
        sim_port_values = self.initial_value * (1.0 + sim_port_returns)
        # Loss in percentage and in dollars (positive number = loss)
        losses_pct = -sim_port_returns
        losses_cash = self.initial_value * losses_pct
        
        # Calculate risk metrics for each confidence level
        risk_metrics = {}
        for alpha in confidence_levels:
            var_pct = compute_var_from_losses(losses_pct, confidence_level=alpha)
            var_cash = var_pct * self.initial_value
            cvar_pct = compute_cvar_from_losses(losses_pct, confidence_level=alpha)
            cvar_cash = cvar_pct * self.initial_value
            
            # Standard error of quantile estimate: SE(q_alpha) approx sqrt(alpha*(1-alpha) / (N * f(q)^2))
            # We can also compute asymptotic 95% CI
            se_alpha = np.sqrt((alpha * (1.0 - alpha)) / n_simulations)
            # Estimate empirical density f(q) near VaR using kernel window
            h_kde = 1.06 * np.std(losses_pct) * (n_simulations ** (-0.2))
            if h_kde > 1e-8:
                f_q = np.mean(np.abs(losses_pct - var_pct) < (h_kde / 2.0)) / h_kde
                f_q = max(f_q, 1e-4)
                var_se = se_alpha / f_q
            else:
                var_se = np.std(losses_pct) / np.sqrt(n_simulations)
                
            var_ci_95 = (float(var_pct - 1.96 * var_se), float(var_pct + 1.96 * var_se))
            
            risk_metrics[f"{int(alpha * 100)}%"] = {
                "confidence_level": alpha,
                "var_pct": float(var_pct),
                "var_cash": float(var_cash),
                "cvar_pct": float(cvar_pct),
                "cvar_cash": float(cvar_cash),
                "var_se": float(var_se),
                "var_ci_95": var_ci_95
            }
            
        runtime = time.perf_counter() - start_time
        
        return {
            "method": f"Classical Monte Carlo ({method.upper()})",
            "n_simulations": n_simulations,
            "time_horizon_days": time_horizon_days,
            "runtime_seconds": float(runtime),
            "simulated_returns": sim_port_returns,
            "simulated_values": sim_port_values,
            "losses_pct": losses_pct,
            "losses_cash": losses_cash,
            "risk_metrics": risk_metrics,
            "mean_loss_pct": float(np.mean(losses_pct)),
            "std_loss_pct": float(np.std(losses_pct)),
            "min_loss_pct": float(np.min(losses_pct)),
            "max_loss_pct": float(np.max(losses_pct))
        }
