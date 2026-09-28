"""
Unit and integration tests for Quantum Portfolio Builder.
Validates scenarios TEST 1 through TEST 10 as specified in requirements.
"""
import pytest
import numpy as np
import pandas as pd

from src.portfolio.portfolio_data import (
    load_stock_universe,
    search_universe,
    fetch_historical_prices
)
from src.portfolio.portfolio_validation import validate_custom_portfolio
from src.portfolio.portfolio import Portfolio
from src.classical.monte_carlo import MonteCarloEngine
from src.quantum.state_preparation import LossDistributionDiscretizer
from src.quantum.quantum_var import QuantumVaREstimator
from src.quantum.quantum_cvar import QuantumCVaREstimator


@pytest.fixture(scope="module")
def stock_universes():
    in_uni = load_stock_universe("INDIA")
    us_uni = load_stock_universe("US")
    return in_uni, us_uni


def test_scenario_1_india_portfolio(stock_universes):
    """TEST 1: India portfolio, 3 stocks, ₹10,00,000, weights 40/35/25."""
    in_uni, _ = stock_universes
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    total_val = 1_000_000.0
    weights_input = {"RELIANCE.NS": 40.0, "TCS.NS": 35.0, "INFY.NS": 25.0}

    prices, avail, _ = fetch_historical_prices(tickers)
    assert len(prices) > 20

    res = validate_custom_portfolio(
        selected_tickers=tickers,
        total_portfolio_value=total_val,
        allocation_mode="Portfolio Weight",
        weights_input=weights_input,
        data_availability=avail,
        known_universe_tickers=set(in_uni["ticker"])
    )
    assert res.is_valid
    assert res.status == "VALID"
    assert pytest.approx(res.total_allocation_pct, 1e-3) == 100.0
    assert pytest.approx(res.weights["RELIANCE.NS"], 1e-4) == 0.40
    assert pytest.approx(res.weights["TCS.NS"], 1e-4) == 0.35
    assert pytest.approx(res.weights["INFY.NS"], 1e-4) == 0.25


def test_scenario_2_us_portfolio(stock_universes):
    """TEST 2: US portfolio, 3 stocks, $100,000, weights 50/30/20."""
    _, us_uni = stock_universes
    tickers = ["AAPL", "MSFT", "NVDA"]
    total_val = 100_000.0
    weights_input = {"AAPL": 50.0, "MSFT": 30.0, "NVDA": 20.0}

    prices, avail, _ = fetch_historical_prices(tickers)
    assert len(prices) > 20

    res = validate_custom_portfolio(
        selected_tickers=tickers,
        total_portfolio_value=total_val,
        allocation_mode="Portfolio Weight",
        weights_input=weights_input,
        data_availability=avail,
        known_universe_tickers=set(us_uni["ticker"])
    )
    assert res.is_valid
    assert res.status == "VALID"
    assert pytest.approx(res.total_allocation_pct, 1e-3) == 100.0
    assert pytest.approx(res.weights["AAPL"], 1e-4) == 0.50
    assert pytest.approx(res.weights["MSFT"], 1e-4) == 0.30
    assert pytest.approx(res.weights["NVDA"], 1e-4) == 0.20


def test_scenario_3_single_stock_portfolio(stock_universes):
    """TEST 3: Single-stock portfolio, 100%."""
    in_uni, _ = stock_universes
    tickers = ["RELIANCE.NS"]
    total_val = 500_000.0
    weights_input = {"RELIANCE.NS": 100.0}

    res = validate_custom_portfolio(
        selected_tickers=tickers,
        total_portfolio_value=total_val,
        allocation_mode="Portfolio Weight",
        weights_input=weights_input,
        data_availability={"RELIANCE.NS": True},
        known_universe_tickers=set(in_uni["ticker"])
    )
    assert res.is_valid
    assert res.status == "VALID"
    assert pytest.approx(res.weights["RELIANCE.NS"], 1e-4) == 1.0


def test_scenario_4_invalid_weights(stock_universes):
    """TEST 4: Invalid weights 40/40/30 (sum 110%). UI must reject."""
    in_uni, _ = stock_universes
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    total_val = 1_000_000.0
    weights_input = {"RELIANCE.NS": 40.0, "TCS.NS": 40.0, "INFY.NS": 30.0}

    res = validate_custom_portfolio(
        selected_tickers=tickers,
        total_portfolio_value=total_val,
        allocation_mode="Portfolio Weight",
        weights_input=weights_input,
        data_availability={"RELIANCE.NS": True, "TCS.NS": True, "INFY.NS": True},
        known_universe_tickers=set(in_uni["ticker"])
    )
    assert not res.is_valid
    assert res.status == "INVALID"
    assert any("sum to 100" in err for err in res.errors)


def test_scenario_5_investment_amount_mode():
    """TEST 5: Investment amount mode (₹400k, ₹300k, ₹300k -> 40%, 30%, 30%)."""
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    total_val = 1_000_000.0
    investment_input = {
        "RELIANCE.NS": 400_000.0,
        "TCS.NS": 300_000.0,
        "INFY.NS": 300_000.0
    }

    res = validate_custom_portfolio(
        selected_tickers=tickers,
        total_portfolio_value=total_val,
        allocation_mode="Investment Amount",
        investment_input=investment_input,
        data_availability={"RELIANCE.NS": True, "TCS.NS": True, "INFY.NS": True}
    )
    assert res.is_valid
    assert pytest.approx(res.weights["RELIANCE.NS"], 1e-4) == 0.40
    assert pytest.approx(res.weights["TCS.NS"], 1e-4) == 0.30
    assert pytest.approx(res.weights["INFY.NS"], 1e-4) == 0.30


def test_scenario_6_invalid_investment_exceeding_total():
    """TEST 6: Invalid investment amount exceeding total portfolio value."""
    tickers = ["RELIANCE.NS", "TCS.NS"]
    total_val = 1_000_000.0
    investment_input = {
        "RELIANCE.NS": 800_000.0,
        "TCS.NS": 400_000.0  # Sum 1,200,000 > 1,000,000
    }

    res = validate_custom_portfolio(
        selected_tickers=tickers,
        total_portfolio_value=total_val,
        allocation_mode="Investment Amount",
        investment_input=investment_input,
        data_availability={"RELIANCE.NS": True, "TCS.NS": True}
    )
    assert not res.is_valid
    assert res.status == "INVALID"
    assert any("must equal total portfolio value" in err for err in res.errors)


def test_scenario_7_search_existing_ticker(stock_universes):
    """TEST 7: Search for a ticker that exists in the CSV."""
    in_uni, _ = stock_universes
    results = search_universe(in_uni, "RELIANCE")
    assert len(results) > 0
    assert any("RELIANCE" in sym for sym in results["ticker"])


def test_scenario_8_search_nonexistent_ticker(stock_universes):
    """TEST 8: Search for a ticker that does not exist."""
    in_uni, _ = stock_universes
    results = search_universe(in_uni, "NONEXISTENT_XYZ_999")
    assert len(results) == 0


def test_scenario_9_stock_with_missing_historical_data():
    """TEST 9: Stock with missing historical data must be flagged as invalid."""
    tickers = ["RELIANCE.NS", "DELISTED_FAKE_TICKER"]
    res = validate_custom_portfolio(
        selected_tickers=tickers,
        total_portfolio_value=1_000_000.0,
        allocation_mode="Portfolio Weight",
        weights_input={"RELIANCE.NS": 50.0, "DELISTED_FAKE_TICKER": 50.0},
        data_availability={"RELIANCE.NS": True, "DELISTED_FAKE_TICKER": False}
    )
    assert not res.is_valid
    assert res.status == "INVALID"
    assert any("Historical price data is unavailable" in err for err in res.errors)


def test_scenario_10_quantum_risk_analysis_pipeline(stock_universes):
    """
    TEST 10: Run Quantum Risk Analysis on custom portfolio and verify that
    it reaches Quantum Engine components and executes exact QAE/MLQAE estimation without errors.
    """
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    prices, avail, _ = fetch_historical_prices(tickers)
    assert all(avail.values())

    weights = np.array([0.40, 0.35, 0.25])
    port_val = 1_000_000.0

    # 1. Custom Portfolio
    portfolio = Portfolio(prices[tickers], weights=weights, initial_value=port_val)
    assert portfolio.num_assets == 3
    assert portfolio.ann_vol > 0.0

    # 2. Classical Monte Carlo simulation
    mc_engine = MonteCarloEngine(
        initial_portfolio_value=port_val,
        weights=portfolio.weights,
        expected_returns=portfolio.returns.mean().values * 252,
        covariance_matrix=portfolio.covariance_matrix,
        asset_prices=portfolio.prices.iloc[-1].values,
        seed=42
    )
    sim_res = mc_engine.run_simulation(n_simulations=2000, time_horizon_days=10)
    losses = sim_res["losses_pct"]
    assert len(losses) == 2000

    # 3. Discretization
    discretizer = LossDistributionDiscretizer(losses, num_qubits=4)
    assert len(discretizer.probabilities) == 16

    # 4. Quantum VaR & Quantum CVaR
    q_var_estimator = QuantumVaREstimator(discretizer, seed=42)
    q_var_res = q_var_estimator.estimate_var(confidence_level=0.95, method="mlqae", portfolio_value=port_val)
    assert q_var_res["quantum_var_pct"] > 0.0
    assert q_var_res["quantum_var_cash"] > 0.0

    q_cvar_estimator = QuantumCVaREstimator(discretizer, seed=42)
    q_cvar_res = q_cvar_estimator.estimate_cvar(
        confidence_level=0.95,
        quantum_var_pct=q_var_res["quantum_var_pct"],
        method="mlqae",
        portfolio_value=port_val
    )
    assert q_cvar_res["quantum_cvar_pct"] >= q_var_res["quantum_var_pct"]
    assert q_cvar_res["quantum_cvar_cash"] >= q_var_res["quantum_var_cash"]
