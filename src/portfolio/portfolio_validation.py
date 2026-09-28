"""
Portfolio validation module for custom portfolio construction.
Validates asset selections, weight allocations, investment amounts, and historical data completeness.
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class PortfolioValidationResult:
    """Encapsulates the result of portfolio validation."""
    def __init__(
        self,
        is_valid: bool,
        status: str,
        errors: List[str],
        warnings: List[str],
        weights: Dict[str, float],
        investment_amounts: Dict[str, float],
        total_allocation_pct: float,
        total_investment_val: float
    ):
        self.is_valid = is_valid
        self.status = status
        self.errors = errors
        self.warnings = warnings
        self.weights = weights
        self.investment_amounts = investment_amounts
        self.total_allocation_pct = total_allocation_pct
        self.total_investment_val = total_investment_val

    def __repr__(self) -> str:
        return f"<PortfolioValidationResult valid={self.is_valid} status={self.status} errors={len(self.errors)}>"


def validate_custom_portfolio(
    selected_tickers: List[str],
    total_portfolio_value: float,
    allocation_mode: str,  # "Portfolio Weight" or "Investment Amount"
    weights_input: Optional[Dict[str, float]] = None,
    investment_input: Optional[Dict[str, float]] = None,
    data_availability: Optional[Dict[str, bool]] = None,
    known_universe_tickers: Optional[set] = None
) -> PortfolioValidationResult:
    """
    Perform comprehensive institutional validation on a custom portfolio.
    
    Validation criteria:
    1. At least one stock selected
    2. Portfolio amount > 0
    3. All selected tickers exist in the dataset
    4. No duplicate tickers
    5. No missing required historical data
    6. Weights sum to 100% (within 0.05% tolerance) OR Investment amounts sum to total portfolio amount
    7. Investment amounts do not exceed total portfolio amount and are non-negative
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    # 1. Check selection
    if not selected_tickers or len(selected_tickers) == 0:
        errors.append("At least one stock must be selected.")
        return PortfolioValidationResult(
            is_valid=False,
            status="INVALID",
            errors=errors,
            warnings=warnings,
            weights={},
            investment_amounts={},
            total_allocation_pct=0.0,
            total_investment_val=0.0
        )

    # 2. Check total portfolio value
    try:
        total_val = float(total_portfolio_value)
    except (TypeError, ValueError):
        total_val = 0.0

    if total_val <= 0:
        errors.append("Total portfolio value must be greater than zero.")

    # 3. Check duplicate tickers
    if len(selected_tickers) != len(set(selected_tickers)):
        errors.append("Duplicate tickers detected in asset selection.")

    # 4. Check known universe
    if known_universe_tickers is not None:
        unknown = [t for t in selected_tickers if t not in known_universe_tickers]
        if unknown:
            errors.append(f"The following ticker(s) were not found in the dataset: {', '.join(unknown)}.")

    # 5. Check historical data availability
    if data_availability is not None:
        missing_data = [t for t in selected_tickers if not data_availability.get(t, False)]
        if missing_data:
            errors.append(f"Historical price data is unavailable for: {', '.join(missing_data)}.")

    # 6 & 7. Allocation calculation and verification
    final_weights: Dict[str, float] = {}
    final_investments: Dict[str, float] = {}
    total_alloc_pct = 0.0
    total_inv_val = 0.0

    mode_lower = allocation_mode.lower()

    if "weight" in mode_lower:
        # Portfolio Weight mode
        weights_input = weights_input or {}
        raw_sum = 0.0
        for ticker in selected_tickers:
            w_pct = float(weights_input.get(ticker, 0.0))
            if w_pct < 0:
                errors.append(f"Negative weight ({w_pct:.2f}%) assigned to {ticker}.")
            raw_sum += w_pct
            w_dec = w_pct / 100.0
            final_weights[ticker] = w_dec
            final_investments[ticker] = w_dec * total_val

        total_alloc_pct = raw_sum
        total_inv_val = sum(final_investments.values())

        if abs(raw_sum - 100.0) > 0.05:
            errors.append(f"Portfolio weights must sum to 100.00% (currently {raw_sum:.2f}%).")

    else:
        # Investment Amount mode
        investment_input = investment_input or {}
        inv_sum = 0.0
        for ticker in selected_tickers:
            amt = float(investment_input.get(ticker, 0.0))
            if amt < 0:
                errors.append(f"Investment amount for {ticker} cannot be negative.")
            if total_val > 0 and amt > total_val:
                errors.append(f"Investment amount for {ticker} ({amt:,.2f}) exceeds total portfolio value ({total_val:,.2f}).")
            inv_sum += amt
            final_investments[ticker] = amt
            if total_val > 0:
                final_weights[ticker] = amt / total_val
            else:
                final_weights[ticker] = 0.0

        total_inv_val = inv_sum
        total_alloc_pct = (inv_sum / total_val * 100.0) if total_val > 0 else 0.0

        if total_val > 0 and abs(inv_sum - total_val) > 1.0:
            errors.append(
                f"Sum of investment amounts ({inv_sum:,.2f}) must equal total portfolio value ({total_val:,.2f})."
            )

    is_valid = len(errors) == 0
    status = "VALID" if is_valid else "INVALID"

    return PortfolioValidationResult(
        is_valid=is_valid,
        status=status,
        errors=errors,
        warnings=warnings,
        weights=final_weights,
        investment_amounts=final_investments,
        total_allocation_pct=total_alloc_pct,
        total_investment_val=total_inv_val
    )
