"""
Tests for Quantum State Preparation, Risk Oracle, QAE, and Quantum VaR/CVaR.
"""
import pytest
import numpy as np
from src.quantum.state_preparation import LossDistributionDiscretizer, create_state_preparation_circuit
from src.quantum.risk_oracle import build_risk_oracle_circuit, build_full_risk_operator, build_grover_operator
from src.quantum.amplitude_estimation import QuantumAmplitudeEstimator
from src.quantum.quantum_var import QuantumVaREstimator
from src.quantum.quantum_cvar import QuantumCVaREstimator


def test_state_preparation_and_discretization():
    losses = np.linspace(-0.05, 0.05, 1000)
    num_qubits = 3
    discretizer = LossDistributionDiscretizer(losses, num_qubits=num_qubits)
    
    assert discretizer.num_bins == 8
    assert pytest.approx(float(np.sum(discretizer.probabilities)), 1e-5) == 1.0
    
    qc = create_state_preparation_circuit(discretizer, objective_qubit=True)
    assert qc.num_qubits == 4


def test_quantum_statevector_exact_probability():
    losses = np.linspace(-0.05, 0.05, 1000)
    discretizer = LossDistributionDiscretizer(losses, num_qubits=3)
    
    threshold = 0.02
    k_thresh = discretizer.value_to_bin_index(threshold)
    classical_p = discretizer.get_classical_tail_probability(threshold)
    
    qae = QuantumAmplitudeEstimator(discretizer)
    res_sv = qae.run_statevector_exact(threshold)
    
    # Statevector exact probability must match classical discrete probability within machine precision
    assert pytest.approx(res_sv["estimated_probability"], 1e-5) == classical_p


def test_quantum_mlqae_convergence():
    np.random.seed(42)
    losses = np.random.normal(0, 0.02, size=2000)
    discretizer = LossDistributionDiscretizer(losses, num_qubits=3)
    
    threshold = float(np.percentile(losses, 90.0))
    qae = QuantumAmplitudeEstimator(discretizer, seed=42)
    res_ml = qae.run_mlqae(threshold, evaluation_schedule=[0, 1, 2], shots_per_circuit=512)
    
    # Error should be within a couple percentage points
    assert res_ml["absolute_error"] < 0.06
    assert res_ml["estimated_probability"] > 0.0
