"""
Sample efficiency experiment: comparing theoretical and empirical scaling laws of MC vs QAE.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.utils.reproducibility import save_experiment_results


def run_sample_efficiency_experiment(
    target_precisions: Optional[List[float]] = None,
    true_prob: float = 0.05
) -> Dict[str, Any]:
    """
    Calculate required evaluations / samples to reach target estimation precisions (epsilon)
    comparing Classical Monte Carlo O(1/epsilon^2) vs Quantum Amplitude Estimation O(1/epsilon).
    """
    if target_precisions is None:
        target_precisions = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001]
        
    records = []
    # Standard 95% confidence z-factor = 1.96
    z = 1.96
    variance = true_prob * (1.0 - true_prob)
    
    for eps in target_precisions:
        # Classical MC required samples: eps = z * sqrt(var / N) => N = (z^2 * var) / eps^2
        mc_samples = int(np.ceil((z ** 2 * variance) / (eps ** 2)))
        
        # Quantum Amplitude Estimation required Grover evaluations: M = pi / (2 * eps)
        qae_evaluations = int(np.ceil(np.pi / (2.0 * eps)))
        
        # Theoretical speedup ratio: N_classical / M_quantum
        speedup_ratio = mc_samples / max(qae_evaluations, 1)
        
        records.append({
            "Target_Precision_Epsilon": eps,
            "Classical_MC_Samples": mc_samples,
            "Quantum_Oracle_Queries": qae_evaluations,
            "Theoretical_Speedup_Factor": speedup_ratio,
            "Log10_MC_Samples": float(np.log10(mc_samples)),
            "Log10_Quantum_Queries": float(np.log10(qae_evaluations))
        })
        
    df = pd.DataFrame(records)
    
    save_experiment_results(
        experiment_name="sample_efficiency_scaling",
        parameters={"target_precisions": target_precisions, "true_prob": true_prob},
        results={"records": records},
        dataframe=df
    )
    
    return {
        "dataframe": df,
        "true_probability": true_prob,
        "records": records
    }
