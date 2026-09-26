"""
Convergence experiment module: benchmarking Classical Monte Carlo vs Quantum Amplitude Estimation.
"""
from typing import Dict, Any, List, Optional
import time
import numpy as np
import pandas as pd

from src.classical.monte_carlo import MonteCarloEngine
from src.quantum.state_preparation import LossDistributionDiscretizer
from src.quantum.amplitude_estimation import QuantumAmplitudeEstimator
from src.utils.reproducibility import save_experiment_results, set_seed


def run_convergence_experiment(
    sample_sizes: Optional[List[int]] = None,
    true_tail_probability: float = 0.05,
    n_trials: int = 5,
    num_qubits: int = 4,
    base_seed: int = 42
) -> Dict[str, Any]:
    """
    Run empirical convergence experiment comparing Classical Monte Carlo vs Quantum Amplitude Estimation.
    
    Measures:
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - Bias
    - Variance / Standard Deviation
    across varying sample counts / shot counts.
    """
    if sample_sizes is None:
        sample_sizes = [100, 500, 1000, 5000, 10000]
        
    results_rows = []
    
    # Synthetic calibrated binomial/normal loss distribution for standard benchmark
    np.random.seed(base_seed)
    synthetic_losses = np.random.normal(loc=0.0, scale=0.02, size=50_000)
    discretizer = LossDistributionDiscretizer(synthetic_losses, num_qubits=num_qubits)
    
    # Find threshold corresponding to true target tail probability
    threshold = float(np.percentile(synthetic_losses, (1.0 - true_tail_probability) * 100.0))
    p_true = discretizer.get_classical_tail_probability(threshold)
    
    for n_samples in sample_sizes:
        mc_errors = []
        mc_estimates = []
        mc_times = []
        
        qae_errors = []
        qae_estimates = []
        qae_times = []
        
        for trial in range(n_trials):
            trial_seed = base_seed + trial * 100
            set_seed(trial_seed)
            
            # --- Classical Monte Carlo ---
            t0 = time.perf_counter()
            mc_draws = np.random.normal(loc=0.0, scale=0.02, size=n_samples)
            p_mc = float(np.mean(mc_draws >= threshold))
            t1 = time.perf_counter()
            
            mc_estimates.append(p_mc)
            mc_errors.append(abs(p_mc - p_true))
            mc_times.append(t1 - t0)
            
            # --- Quantum QAE ---
            t2 = time.perf_counter()
            qae = QuantumAmplitudeEstimator(discretizer, seed=trial_seed)
            # Use shots proportional to sample size (capped to prevent simulator timeout)
            shots = min(max(n_samples // 4, 256), 4096)
            qae_res = qae.run_mlqae(threshold, shots_per_circuit=shots)
            t3 = time.perf_counter()
            
            p_qae = qae_res["estimated_probability"]
            qae_estimates.append(p_qae)
            qae_errors.append(abs(p_qae - p_true))
            qae_times.append(t3 - t2)
            
        mc_arr = np.array(mc_estimates)
        qae_arr = np.array(qae_estimates)
        
        results_rows.append({
            "Samples": n_samples,
            "MC_Mean_Estimate": float(np.mean(mc_arr)),
            "MC_MAE": float(np.mean(np.abs(mc_arr - p_true))),
            "MC_RMSE": float(np.sqrt(np.mean((mc_arr - p_true) ** 2))),
            "MC_Bias": float(np.mean(mc_arr) - p_true),
            "MC_Std": float(np.std(mc_arr)),
            "MC_Avg_Time_Sec": float(np.mean(mc_times)),
            
            "QAE_Mean_Estimate": float(np.mean(qae_arr)),
            "QAE_MAE": float(np.mean(np.abs(qae_arr - p_true))),
            "QAE_RMSE": float(np.sqrt(np.mean((qae_arr - p_true) ** 2))),
            "QAE_Bias": float(np.mean(qae_arr) - p_true),
            "QAE_Std": float(np.std(qae_arr)),
            "QAE_Avg_Time_Sec": float(np.mean(qae_times)),
            "True_Probability": p_true
        })
        
    df_results = pd.DataFrame(results_rows)
    
    experiment_payload = {
        "true_tail_probability": p_true,
        "n_trials": n_trials,
        "num_qubits": num_qubits,
        "sample_sizes": sample_sizes,
        "summary": results_rows
    }
    
    save_experiment_results(
        experiment_name="convergence_mc_vs_quantum",
        parameters={"sample_sizes": sample_sizes, "n_trials": n_trials, "num_qubits": num_qubits},
        results=experiment_payload,
        dataframe=df_results
    )
    
    return {
        "dataframe": df_results,
        "true_probability": p_true,
        "sample_sizes": sample_sizes
    }
