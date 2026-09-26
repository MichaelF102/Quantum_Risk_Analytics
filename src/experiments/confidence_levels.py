"""
Confidence-level experiment module: comparing Classical VaR vs Quantum VaR across extreme tails.
Analyzes 90%, 95%, 99%, 99.5% confidence levels.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.quantum.state_preparation import LossDistributionDiscretizer
from src.quantum.quantum_var import QuantumVaREstimator
from src.quantum.quantum_cvar import QuantumCVaREstimator
from src.classical.var import compute_var_from_losses
from src.classical.cvar import compute_cvar_from_losses
from src.utils.reproducibility import save_experiment_results


def run_confidence_levels_experiment(
    losses: np.ndarray,
    confidence_levels: Optional[List[float]] = None,
    num_qubits: int = 5,
    method: str = "mlqae",
    portfolio_value: float = 1_000_000.0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluate Classical VaR/CVaR vs Quantum VaR/CVaR across multiple tail confidence levels:
    [0.90, 0.95, 0.99, 0.995].
    """
    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.99, 0.995]
        
    discretizer = LossDistributionDiscretizer(losses, num_qubits=num_qubits)
    var_estimator = QuantumVaREstimator(discretizer, seed=seed)
    cvar_estimator = QuantumCVaREstimator(discretizer, seed=seed)
    
    records = []
    
    for alpha in confidence_levels:
        # Classical benchmarks
        cl_var_pct = compute_var_from_losses(losses, confidence_level=alpha)
        cl_cvar_pct = compute_cvar_from_losses(losses, confidence_level=alpha)
        
        # Quantum VaR
        q_var_res = var_estimator.estimate_var(confidence_level=alpha, method=method, portfolio_value=portfolio_value)
        q_var_pct = q_var_res["quantum_var_pct"]
        
        # Quantum CVaR
        q_cvar_res = cvar_estimator.estimate_cvar(confidence_level=alpha, quantum_var_pct=q_var_pct, method=method, portfolio_value=portfolio_value)
        q_cvar_pct = q_cvar_res["quantum_cvar_pct"]
        
        var_abs_err = abs(q_var_pct - cl_var_pct)
        var_rel_err = var_abs_err / max(abs(cl_var_pct), 1e-6)
        
        cvar_abs_err = abs(q_cvar_pct - cl_cvar_pct)
        cvar_rel_err = cvar_abs_err / max(abs(cl_cvar_pct), 1e-6)
        
        records.append({
            "Confidence_Level": alpha,
            "Tail_Probability_Pct": float((1.0 - alpha) * 100.0),
            "Classical_VaR_Pct": float(cl_var_pct * 100.0),
            "Quantum_VaR_Pct": float(q_var_pct * 100.0),
            "VaR_Abs_Error_Pct": float(var_abs_err * 100.0),
            "VaR_Rel_Error": float(var_rel_err),
            "Classical_CVaR_Pct": float(cl_cvar_pct * 100.0),
            "Quantum_CVaR_Pct": float(q_cvar_pct * 100.0),
            "CVaR_Abs_Error_Pct": float(cvar_abs_err * 100.0),
            "CVaR_Rel_Error": float(cvar_rel_err),
            "Discretization_Bins": 2 ** num_qubits,
            "Quantum_Runtime_Sec": float(q_var_res["runtime_seconds"] + q_cvar_res["runtime_seconds"])
        })
        
    df = pd.DataFrame(records)
    
    save_experiment_results(
        experiment_name="confidence_levels_comparison",
        parameters={"confidence_levels": confidence_levels, "num_qubits": num_qubits, "method": method},
        results={"records": records},
        dataframe=df
    )
    
    return {
        "dataframe": df,
        "records": records
    }
