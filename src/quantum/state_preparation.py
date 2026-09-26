"""
Quantum State Preparation module for loss distributions.
Discretizes continuous financial loss distributions into 2^n quantum basis states.
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class LossDistributionDiscretizer:
    """
    Discretizes empirical or simulated loss distributions into 2^n discrete probability bins
    for quantum encoding.
    """
    def __init__(
        self,
        losses: np.ndarray,
        num_qubits: int = 4,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None
    ):
        if num_qubits < 2 or num_qubits > 8:
            raise ValueError(f"num_qubits must be between 2 and 8 (current: {num_qubits}) to prevent exponential circuit overhead.")
            
        self.num_qubits = num_qubits
        self.num_bins = 2 ** num_qubits
        
        # Loss bounds (using percentiles if not provided to eliminate extreme outliers)
        if lower_bound is None:
            self.lower_bound = float(np.percentile(losses, 0.1))
        else:
            self.lower_bound = float(lower_bound)
            
        if upper_bound is None:
            self.upper_bound = float(np.percentile(losses, 99.9))
        else:
            self.upper_bound = float(upper_bound)
            
        if self.upper_bound <= self.lower_bound:
            self.upper_bound = self.lower_bound + 1e-4
            
        # Discretize into num_bins
        bin_edges = np.linspace(self.lower_bound, self.upper_bound, self.num_bins + 1)
        self.bin_edges = bin_edges
        self.bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        self.bin_width = float(bin_edges[1] - bin_edges[0])
        
        # Compute histogram probabilities
        clipped_losses = np.clip(losses, self.lower_bound, self.upper_bound)
        counts, _ = np.histogram(clipped_losses, bins=bin_edges)
        
        # Avoid zero probabilities for numerical stability in state preparation
        counts = counts.astype(float) + 1e-7
        self.probabilities = counts / np.sum(counts)
        self.amplitudes = np.sqrt(self.probabilities)

    def value_to_bin_index(self, loss_val: float) -> int:
        """Find the discrete bin index corresponding to a continuous loss value."""
        if loss_val <= self.bin_edges[0]:
            return 0
        if loss_val >= self.bin_edges[-1]:
            return self.num_bins - 1
        idx = int(np.searchsorted(self.bin_edges, loss_val, side="right") - 1)
        return int(np.clip(idx, 0, self.num_bins - 1))

    def bin_index_to_value(self, index: int) -> float:
        """Return the loss value corresponding to a discrete bin index."""
        idx = int(np.clip(index, 0, self.num_bins - 1))
        return float(self.bin_centers[idx])

    def get_classical_tail_probability(self, threshold: float) -> float:
        """Compute the exact discrete probability P(L >= threshold) from the discretized distribution."""
        k = self.value_to_bin_index(threshold)
        return float(np.sum(self.probabilities[k:]))


def create_state_preparation_circuit(
    discretizer: LossDistributionDiscretizer,
    objective_qubit: bool = True
) -> QuantumCircuit:
    """
    Construct a Qiskit QuantumCircuit that loads the discretized loss distribution.
    
    If objective_qubit is True, allocates an extra ancilla qubit (qubit index n)
    for the risk/event oracle.
    """
    n_state_qubits = discretizer.num_qubits
    total_qubits = n_state_qubits + 1 if objective_qubit else n_state_qubits
    
    qc = QuantumCircuit(total_qubits, name="StatePrep")
    
    # Encode discrete probabilities into state register [0, ..., n-1]
    state_qubit_indices = list(range(n_state_qubits))
    qc.prepare_state(discretizer.amplitudes, qubits=state_qubit_indices)
    
    return qc
