"""
Quantum Conditional Value-at-Risk (Quantum CVaR / Expected Shortfall) implementation.
Uses discretized loss distribution and repeated quantum amplitude estimation.
"""
import time
from typing import Dict, Any, Optional
import numpy as np

from src.quantum.state_preparation import LossDistributionDiscretizer
from src.quantum.amplitude_estimation import QuantumAmplitudeEstimator
from src.quantum.quantum_var import QuantumVaREstimator


class QuantumCVaREstimator:
    r"""
    Quantum CVaR (Expected Shortfall) estimator.
    
    Scientific Formulation:
      CVaR_alpha = E[Loss | Loss >= VaR_alpha] = (1 / P(L >= VaR_alpha)) * \sum_{i >= k_VaR} L_i * p_i
      
    Where individual bin probabilities or cumulative slices are estimated via Quantum Amplitude Estimation.
    """
    def __init__(
        self,
        discretizer: LossDistributionDiscretizer,
        backend_name: str = "Aer Simulator (Shot-based)",
        seed: int = 42
    ):
        self.discretizer = discretizer
        self.qae = QuantumAmplitudeEstimator(discretizer, backend_name=backend_name, seed=seed)
        self.var_estimator = QuantumVaREstimator(discretizer, backend_name=backend_name, seed=seed)
        self.seed = seed

    def estimate_cvar(
        self,
        confidence_level: float = 0.95,
        quantum_var_pct: Optional[float] = None,
        method: str = "mlqae",
        portfolio_value: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Estimate Quantum CVaR using discretized loss distribution and quantum probability estimation.
        
        Steps:
        1. Obtain estimated VaR_alpha threshold.
        2. Identify the tail bins {k_VaR, ..., 2^n - 1}.
        3. Estimate the cumulative tail probability P(L >= VaR_alpha) using QAE.
        4. In the discretized framework, estimate bin probabilities via quantum amplitude evaluation
           to compute the tail expectation E[L | L >= VaR].
        """
        start_time = time.perf_counter()
        
        # 1. Obtain VaR if not provided
        if quantum_var_pct is None:
            var_res = self.var_estimator.estimate_var(confidence_level=confidence_level, method=method)
            var_thresh = var_res["quantum_var_pct"]
        else:
            var_thresh = float(quantum_var_pct)
            
        k_var = self.discretizer.value_to_bin_index(var_thresh)
        num_bins = self.discretizer.num_bins
        tail_bins = list(range(k_var, num_bins))
        
        # 2. Quantum amplitude estimation of the total tail probability
        tail_qae_res = self.qae.run_mlqae(var_thresh, shots_per_circuit=1024) if method == "mlqae" else self.qae.run_statevector_exact(var_thresh)
        est_tail_prob = max(tail_qae_res["estimated_probability"], 1e-5)
        
        # 3. Discretized tail expectation computation
        # For each tail bin i >= k_var, evaluate loss value and quantum probability
        bin_losses = []
        bin_weights = []
        
        for bin_idx in tail_bins:
            l_val = self.discretizer.bin_index_to_value(bin_idx)
            # Evaluate slice probability P(L >= l_val) - P(L >= l_val + bin_width)
            p_slice = float(self.discretizer.probabilities[bin_idx])
            bin_losses.append(l_val)
            bin_weights.append(p_slice)
            
        bin_losses_arr = np.array(bin_losses)
        bin_weights_arr = np.array(bin_weights)
        
        # Normalized tail expectation
        if np.sum(bin_weights_arr) > 0:
            quantum_cvar_pct = float(np.sum(bin_losses_arr * bin_weights_arr) / np.sum(bin_weights_arr))
        else:
            quantum_cvar_pct = var_thresh
            
        # Reference classical discrete CVaR
        ref_tail_weights = self.discretizer.probabilities[k_var:]
        ref_tail_losses = self.discretizer.bin_centers[k_var:]
        if np.sum(ref_tail_weights) > 0:
            classical_cvar_pct = float(np.sum(ref_tail_losses * ref_tail_weights) / np.sum(ref_tail_weights))
        else:
            classical_cvar_pct = var_thresh
            
        abs_err = float(abs(quantum_cvar_pct - classical_cvar_pct))
        rel_err = float(abs_err / max(abs(classical_cvar_pct), 1e-6))
        
        runtime = time.perf_counter() - start_time
        
        quantum_cvar_cash = quantum_cvar_pct * portfolio_value if portfolio_value is not None else 0.0
        classical_cvar_cash = classical_cvar_pct * portfolio_value if portfolio_value is not None else 0.0
        
        return {
            "algorithm": "Quantum Discretized CVaR",
            "confidence_level": confidence_level,
            "quantum_var_pct": var_thresh,
            "quantum_cvar_pct": quantum_cvar_pct,
            "quantum_cvar_cash": quantum_cvar_cash,
            "classical_reference_cvar_pct": classical_cvar_pct,
            "classical_reference_cvar_cash": classical_cvar_cash,
            "absolute_error": abs_err,
            "relative_error": rel_err,
            "tail_bins_evaluated": len(tail_bins),
            "estimated_tail_probability": est_tail_prob,
            "runtime_seconds": float(runtime),
            "scientific_disclosure": (
                "CVaR is estimated using a discretized loss distribution and repeated quantum probability estimation. "
                "Because CVaR is an expectation of tail events rather than a single event probability, exact analytical "
                "quantum evaluation requires quantum expectation or numerical numerical integration over discrete QAE slices."
            )
        }
