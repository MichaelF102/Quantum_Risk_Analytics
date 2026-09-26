"""
Quantum Value-at-Risk (Quantum VaR) research implementation.
Performs quantum bisection search over discrete loss thresholds using Quantum Amplitude Estimation.
"""
import time
from typing import Dict, Any, List, Optional
import numpy as np

from src.quantum.state_preparation import LossDistributionDiscretizer
from src.quantum.amplitude_estimation import QuantumAmplitudeEstimator
from src.classical.var import compute_var_from_losses


class QuantumVaREstimator:
    """
    Quantum Value-at-Risk search algorithm.
    Finds the critical loss threshold T* such that P(Loss >= T*) = 1 - alpha.
    """
    def __init__(
        self,
        discretizer: LossDistributionDiscretizer,
        backend_name: str = "Aer Simulator (Shot-based)",
        seed: int = 42
    ):
        self.discretizer = discretizer
        self.qae = QuantumAmplitudeEstimator(discretizer, backend_name=backend_name, seed=seed)
        self.seed = seed

    def estimate_var(
        self,
        confidence_level: float = 0.95,
        method: str = "mlqae",
        tolerance: float = 0.015,
        portfolio_value: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Estimate Quantum VaR for a target confidence level alpha.
        Target tail probability: p_target = 1.0 - alpha (e.g. 0.05 for 95% VaR).
        
        Performs discrete binary search over the 2^n loss bins:
        - Evaluates candidate threshold T_k using QAE
        - Compares estimated P(L >= T_k) against p_target
        - Converges to optimal bin index k*
        """
        start_time = time.perf_counter()
        target_tail_prob = 1.0 - confidence_level
        num_bins = self.discretizer.num_bins
        
        low_idx = 0
        high_idx = num_bins - 1
        best_idx = 0
        min_prob_diff = float("inf")
        
        search_history = []
        total_evaluations = 0
        
        # Binary search over discrete bin indices
        while low_idx <= high_idx:
            mid_idx = (low_idx + high_idx) // 2
            cand_threshold = self.discretizer.bin_index_to_value(mid_idx)
            total_evaluations += 1
            
            # Run QAE to estimate tail probability P(L >= cand_threshold)
            if method == "mlqae":
                qae_res = self.qae.run_mlqae(cand_threshold, shots_per_circuit=1024)
            elif method == "canonical":
                qae_res = self.qae.run_canonical_qae(cand_threshold, num_eval_qubits=4, shots=1024)
            else:
                qae_res = self.qae.run_statevector_exact(cand_threshold)
                
            est_tail_p = qae_res["estimated_probability"]
            ref_tail_p = qae_res["classical_reference_probability"]
            diff = abs(est_tail_p - target_tail_prob)
            
            search_history.append({
                "step": total_evaluations,
                "bin_index": mid_idx,
                "candidate_loss": cand_threshold,
                "estimated_tail_prob": est_tail_p,
                "reference_tail_prob": ref_tail_p,
                "target_tail_prob": target_tail_prob,
                "abs_prob_error": abs(est_tail_p - ref_tail_p)
            })
            
            if diff < min_prob_diff:
                min_prob_diff = diff
                best_idx = mid_idx
                
            # If estimated tail prob is higher than target, threshold is too low -> move right
            if est_tail_p > target_tail_prob:
                low_idx = mid_idx + 1
            else:
                high_idx = mid_idx - 1
                
        # Best quantum VaR estimate
        quantum_var_pct = self.discretizer.bin_index_to_value(best_idx)
        quantum_var_cash = quantum_var_pct * portfolio_value if portfolio_value is not None else 0.0
        
        # Reference Classical VaR from discrete distribution and continuous loss array
        classical_discrete_var = self.discretizer.bin_index_to_value(
            int(np.clip(
                np.searchsorted(np.cumsum(self.discretizer.probabilities), confidence_level),
                0,
                num_bins - 1
            ))
        )
        
        abs_var_error = float(abs(quantum_var_pct - classical_discrete_var))
        rel_var_error = float(abs_var_error / max(abs(classical_discrete_var), 1e-6))
        
        runtime = time.perf_counter() - start_time
        
        return {
            "algorithm": f"Quantum VaR via {method.upper()}",
            "confidence_level": confidence_level,
            "target_tail_probability": target_tail_prob,
            "quantum_var_pct": quantum_var_pct,
            "quantum_var_cash": quantum_var_cash,
            "classical_reference_var_pct": classical_discrete_var,
            "classical_reference_var_cash": classical_discrete_var * portfolio_value if portfolio_value else 0.0,
            "absolute_error": abs_var_error,
            "relative_error": rel_var_error,
            "discretization_bins": num_bins,
            "bin_width": self.discretizer.bin_width,
            "total_qae_searches": total_evaluations,
            "search_history": search_history,
            "runtime_seconds": float(runtime),
            "methodology_note": (
                "Quantum VaR estimates the (1-alpha) tail quantile by performing binary search "
                "over discretized loss levels, querying a quantum risk oracle and Amplitude Estimation "
                "to evaluate tail loss probabilities P(L >= T)."
            )
        }
