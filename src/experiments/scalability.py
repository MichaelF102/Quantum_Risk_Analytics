"""
Scalability experiment module: testing portfolio sizes (5, 10, 25, 50, 100 assets).
Analyzes classical runtime, quantum circuit dimensions (qubits, depth), and memory.
"""
from typing import Dict, Any, List, Optional
import time
import sys
import numpy as np
import pandas as pd

from src.classical.monte_carlo import MonteCarloEngine
from src.quantum.state_preparation import LossDistributionDiscretizer
from src.quantum.amplitude_estimation import QuantumAmplitudeEstimator
from src.quantum.risk_oracle import build_full_risk_operator, build_grover_operator
from src.utils.reproducibility import save_experiment_results


def run_scalability_experiment(
    portfolio_sizes: Optional[List[int]] = None,
    n_simulations: int = 10_000,
    num_state_qubits: int = 4,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Benchmark classical Monte Carlo and quantum circuit resources across portfolio dimensions.
    
    Tests: 5, 10, 25, 50, 100 assets.
    """
    if portfolio_sizes is None:
        portfolio_sizes = [5, 10, 25, 50, 100]
        
    records = []
    
    for n_assets in portfolio_sizes:
        rng = np.random.default_rng(seed + n_assets)
        # Generate synthetic asset parameters
        initial_prices = rng.uniform(50.0, 300.0, size=n_assets)
        weights = np.ones(n_assets) / n_assets
        mu = rng.uniform(0.05, 0.15, size=n_assets)
        
        # Correlated covariance
        corr = np.full((n_assets, n_assets), 0.4)
        np.fill_diagonal(corr, 1.0)
        vols = rng.uniform(0.15, 0.35, size=n_assets)
        cov = np.outer(vols, vols) * corr
        
        # --- Measure Classical Monte Carlo ---
        mc_engine = MonteCarloEngine(
            initial_portfolio_value=1_000_000.0,
            weights=weights,
            expected_returns=mu,
            covariance_matrix=cov,
            asset_prices=initial_prices,
            seed=seed
        )
        
        t0 = time.perf_counter()
        mc_res = mc_engine.run_simulation(n_simulations=n_simulations, time_horizon_days=10)
        t_mc = time.perf_counter() - t0
        
        # Approximate memory footprint of simulated returns matrix in MB
        mc_memory_mb = (n_simulations * n_assets * 8) / (1024 * 1024)
        
        # --- Quantum Circuit Resource Estimation ---
        # Discretize portfolio loss distribution
        discretizer = LossDistributionDiscretizer(mc_res["losses_pct"], num_qubits=num_state_qubits)
        threshold = float(np.percentile(mc_res["losses_pct"], 95.0))
        k_thresh = discretizer.value_to_bin_index(threshold)
        
        t2 = time.perf_counter()
        full_a = build_full_risk_operator(discretizer, k_thresh)
        grover_q = build_grover_operator(full_a, num_state_qubits)
        t_circuit_build = time.perf_counter() - t2
        
        q_qubits = num_state_qubits + 1
        q_depth_a = full_a.depth()
        q_depth_grover = grover_q.depth()
        
        # Run statevector simulation for error measurement
        qae = QuantumAmplitudeEstimator(discretizer, seed=seed)
        qae_exact = qae.run_statevector_exact(threshold)
        
        classical_p = discretizer.get_classical_tail_probability(threshold)
        quantum_p = qae_exact["estimated_probability"]
        abs_err = abs(quantum_p - classical_p)
        
        records.append({
            "Assets": n_assets,
            "Classical_MC_Time_Sec": float(t_mc),
            "Classical_Memory_MB": float(mc_memory_mb),
            "Quantum_State_Qubits": num_state_qubits,
            "Quantum_Total_Qubits": q_qubits,
            "Circuit_Depth_A": q_depth_a,
            "Circuit_Depth_Grover": q_depth_grover,
            "Circuit_Build_Time_Sec": float(t_circuit_build),
            "Quantum_Exact_Prob": float(quantum_p),
            "Classical_Ref_Prob": float(classical_p),
            "Estimation_Abs_Error": float(abs_err)
        })
        
    df = pd.DataFrame(records)
    
    save_experiment_results(
        experiment_name="scalability_assets_benchmark",
        parameters={"portfolio_sizes": portfolio_sizes, "n_simulations": n_simulations},
        results={"records": records},
        dataframe=df
    )
    
    return {
        "dataframe": df,
        "records": records
    }
