"""
Unified Portfolio class model for classical and quantum risk computation.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.data.validation import validate_portfolio_weights, DataValidationError
from src.data.preprocessing import calculate_returns
from src.portfolio.returns import calculate_portfolio_returns, calculate_annualized_metrics, calculate_cumulative_returns
from src.portfolio.covariance import calculate_covariance_matrix, calculate_correlation_matrix, calculate_asset_volatilities


class Portfolio:
    """
    Quantitative portfolio container managing asset holdings, returns, and statistical properties.
    """
    def __init__(
        self,
        prices: pd.DataFrame,
        weights: Optional[np.ndarray] = None,
        initial_value: float = 1_000_000.0,
        risk_free_rate: float = 0.02
    ):
        if prices.empty:
            raise ValueError("Prices dataframe cannot be empty.")
            
        self.prices = prices.copy()
        self.assets: List[str] = list(prices.columns)
        self.num_assets: int = len(self.assets)
        self.initial_value: float = float(initial_value)
        self.risk_free_rate: float = float(risk_free_rate)
        
        # Default equal weights if not specified
        if weights is None:
            self.weights = np.ones(self.num_assets) / self.num_assets
        else:
            valid, msg = validate_portfolio_weights(weights)
            if not valid:
                raise DataValidationError(msg)
            self.weights = np.asarray(weights, dtype=float)
            
        # Compute asset returns
        self.returns = calculate_returns(self.prices, method="log")
        self.portfolio_returns = calculate_portfolio_returns(self.returns, self.weights)
        self.cumulative_returns = calculate_cumulative_returns(self.portfolio_returns)
        
        # Asset volatilities and covariance
        self.asset_vols = calculate_asset_volatilities(self.returns, annualize=True)
        self.covariance_matrix = calculate_covariance_matrix(self.returns, annualize=True)
        self.correlation_matrix = calculate_correlation_matrix(self.returns)
        
        # Portfolio summary stats
        self.ann_return, self.ann_vol, self.sharpe_ratio = calculate_annualized_metrics(
            self.portfolio_returns,
            risk_free_rate=self.risk_free_rate
        )

    def update_weights(self, new_weights: np.ndarray) -> None:
        """Update portfolio weights and recompute performance metrics."""
        valid, msg = validate_portfolio_weights(new_weights)
        if not valid:
            raise DataValidationError(msg)
        self.weights = np.asarray(new_weights, dtype=float)
        self.portfolio_returns = calculate_portfolio_returns(self.returns, self.weights)
        self.cumulative_returns = calculate_cumulative_returns(self.portfolio_returns)
        self.ann_return, self.ann_vol, self.sharpe_ratio = calculate_annualized_metrics(
            self.portfolio_returns,
            risk_free_rate=self.risk_free_rate
        )

    def get_asset_allocation_dict(self) -> Dict[str, float]:
        """Return dict of asset to weight."""
        return {asset: float(w) for asset, w in zip(self.assets, self.weights)}

    def get_summary(self) -> Dict[str, Any]:
        """Return high level summary metrics."""
        return {
            "initial_value": self.initial_value,
            "num_assets": self.num_assets,
            "annualized_return": self.ann_return,
            "annualized_volatility": self.ann_vol,
            "sharpe_ratio": self.sharpe_ratio,
            "num_historical_observations": len(self.portfolio_returns)
        }
